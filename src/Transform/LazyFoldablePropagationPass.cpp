/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCst.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"
#include "src/Dialect/LazyCst/LazyFoldableAnalysis.hpp"
#include "src/Dialect/LazyCst/LazyFolder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#include <unordered_map>

namespace {

using namespace mlir;
using namespace onnx_mlir;

[[maybe_unused]] bool isLazyFoldable(Operation *op) {
  return succeeded(op->getContext()
                       ->getLoadedDialect<lazycst::LazyCstDialect>()
                       ->lazyFolders.match(op));
}

bool isConstant(Operation *op) {
  // TODO: consider using mlir::matchPattern(op, m_Constant())
  return op && op->hasTrait<OpTrait::ConstantLike>();
}

[[maybe_unused]] bool isConstantResult(Value v) {
  return isConstant(v.getDefiningOp());
}

// Returns nullptr if v is not a constant result.
[[maybe_unused]] Attribute getConstantAttribute(Operation *op) {
  if (!isConstant(op))
    return nullptr;
  SmallVector<OpFoldResult, 1> folded;
  auto ok = op->fold(folded);
  assert(succeeded(ok));
  assert(folded.size() == 1);
  assert(folded.front().is<Attribute>());
  return folded.front().get<Attribute>();
}

template <typename OpInterfaceType>
class MyOpInterfaceRewritePattern
    : public OpInterfaceRewritePattern<OpInterfaceType> {
  using Base = OpInterfaceRewritePattern<OpInterfaceType>;

public:
  // TODO: figure out if patterns need to fill in generatedNames
  template <typename... Args>
  MyOpInterfaceRewritePattern(
      lazycst::LazyFoldableAnalysis &analysis, Args &&... args)
      : Base(std::forward<Args>(args)...), analysis(analysis) {}

protected:
  lazycst::LazyFoldableAnalysis &analysis;
};

Operation *cloneOpAndSetOperands(
    Operation *op, ValueRange operands, PatternRewriter &rewriter) {
  Operation *clone = rewriter.clone(*op);
  clone->setOperands(operands);
  // TODO: set result type
  return clone;
}

// ACCF: Associative, Commutative, Constant-Foldable.
// TODO: Move declaration to TableGen.
class ACCFOpInterface;
struct ACCFOpInterfaceTraits {
  struct Concept {
    virtual ~Concept() = default;
  };
  template <typename ConcreteOp>
  struct Model : public Concept {
    using Interface = ACCFOpInterface;
  };
  template <typename ConcreteOp>
  struct FallbackModel : public Concept {
    using Interface = ACCFOpInterface;
  };
  template <typename ConcreteModel, typename ConcreteOp>
  struct ExternalModel : public FallbackModel<ConcreteModel> {};
};
template <typename ConcreteOp>
struct ACCFOpInterfaceTrait;
class ACCFOpInterface
    : public OpInterface<ACCFOpInterface, ACCFOpInterfaceTraits> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ACCFOpInterface);
  /// Inherit the base class constructor to support LLVM-style casting.
  using OpInterface<ACCFOpInterface, ACCFOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public ACCFOpInterfaceTrait<ConcreteOp> {};
};
template <typename ConcreteOp>
struct ACCFOpInterfaceTrait : public OpInterface<ACCFOpInterface,
                                  ACCFOpInterfaceTraits>::Trait<ConcreteOp> {};

// This pattern rewrites non-constant-foldable expression op(x1,x2) towards the
// form op(noncstfldb,cstfldb) (in any operand order) in one of two ways:
// 1. op(cstfldb1,op(cstfldb2,x2)) -> op(op(cstfldb1,cstfldb2),x2)
// 2. op(x1,op(cstfldb2,x2)) -> op(op(x1,x2),cstfldb2), if x1 is not constant
// foldable
//
// TODO: generalize to variadic ops, like onnx.Min/Max/Sum
struct BinaryACCFOpPattern
    : public MyOpInterfaceRewritePattern<ACCFOpInterface> {
  using MyOpInterfaceRewritePattern<
      ACCFOpInterface>::MyOpInterfaceRewritePattern;

  // Pair of a non-constant-foldable value and a constant-foldable value.
  using ValuePair = std::array<Value, 2>;

  LogicalResult doRewrite(ACCFOpInterface binaryOp, Value operand, ValuePair vp,
      PatternRewriter &rewriter) const {
    auto [noncstfldb, cstfldb] = vp;
    if (analysis.isLazyFoldable(operand)) {
      // rewrite op to op(noncstfldb,op(operand,cstfldb))
      Operation *cstfldbOp =
          cloneOpAndSetOperands(binaryOp, {operand, cstfldb}, rewriter);
      analysis.insertLazyFoldableOp(cstfldbOp);
      cstfldb = cstfldbOp->getResult(0);
    } else {
      // rewrite op to op(op(operand,noncstfldb),cstfldb)
      noncstfldb =
          cloneOpAndSetOperands(binaryOp, {operand, noncstfldb}, rewriter)
              ->getResult(0);
    }
    rewriter.startRootUpdate(binaryOp);
    binaryOp->setOperands({noncstfldb, cstfldb});
    rewriter.finalizeRootUpdate(binaryOp);
    return success();
  }

  std::optional<ValuePair> hasLazyFoldableOperand(Operation *op) const {
    assert(op->getNumOperands() == 2);
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    if (analysis.isLazyFoldable(rhs))
      return ValuePair{lhs, rhs};
    if (analysis.isLazyFoldable(lhs))
      return ValuePair{rhs, lhs};
    return std::nullopt;
  }

  std::optional<ValuePair> isBubbleable(
      OperationName opName, Value operand) const {
    if (Operation *defop = operand.getDefiningOp()) {
      // Don't bubble up any operand if defop is foldable in whole,
      // and don't decompose it by bubbling if it has multiple uses.
      if (!analysis.isLazyFoldableOp(defop) && defop->hasOneUse() &&
          defop->getName() == opName)
        return hasLazyFoldableOperand(defop);
    }
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(
      ACCFOpInterface binaryOp, PatternRewriter &rewriter) const override {
    // TODO: try to insert this check in the base class MyOpRewritePattern
    if (analysis.isLazyFoldableOp(binaryOp))
      return failure();
    assert(binaryOp->getNumOperands() == 2);
    Value lhs = binaryOp->getOperand(0), rhs = binaryOp->getOperand(1);
    OperationName opName = binaryOp->getName();
    if (auto vp = isBubbleable(opName, lhs))
      return doRewrite(binaryOp, rhs, *vp, rewriter);
    if (auto vp = isBubbleable(opName, rhs))
      return doRewrite(binaryOp, lhs, *vp, rewriter);
    return failure();
  }
};

struct LazyFoldablePropagationPass
    : public PassWrapper<LazyFoldablePropagationPass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LazyFoldablePropagationPass);

  StringRef getArgument() const override {
    return "lazyfoldable-propagation-pass";
  }

  StringRef getDescription() const override {
    return "Propagates lazy-foldable operations sub-graphs";
  }

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    func::FuncOp function = getOperation();
    lazycst::LazyFoldableAnalysis analysis(function);

    // TODO: move this to configurePasses() or something like that
    ONNXAddOp::attachInterface<ACCFOpInterface>(*ctx);
    ONNXMulOp::attachInterface<ACCFOpInterface>(*ctx);
    ONNXXorOp::attachInterface<ACCFOpInterface>(*ctx);
    ONNXBitwiseXorOp::attachInterface<ACCFOpInterface>(*ctx);

    RewritePatternSet patterns(ctx);
    patterns.insert<BinaryACCFOpPattern>(analysis, ctx);
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createLazyFoldablePropagationPass() {
  return std::make_unique<LazyFoldablePropagationPass>();
}
