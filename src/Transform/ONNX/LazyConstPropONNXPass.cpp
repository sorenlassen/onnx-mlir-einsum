/*
 * SPDX-License-Identifier: Apache-2.0
 */

// TODO: remove this pass

#include "src/Dialect/LazyCst/ConstantFoldableAnalysis.hpp"
#include "src/Dialect/LazyCst/ConstantFolder.hpp"
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"
#include "src/Dialect/LazyCst/LazyCstDialect.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CopyOpInterface.h" // TODO: remove
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include <unordered_map>

#define DEBUG_TYPE "lazy-constprop-onnx"

namespace {

using namespace mlir;
using namespace onnx_mlir;

// Populated by configureLazyConstPropONNXPass().
struct LazyConstPropONNXPassConfiguration {
  static int expansionBound;
};

int LazyConstPropONNXPassConfiguration::expansionBound = -1; // -1 == no bound

bool isConstantFoldable(Operation *op) {
  return succeeded(op->getContext()
                       ->getLoadedDialect<lazycst::LazyCstDialect>()
                       ->constantFolders.match(op));
}

bool isConstant(Operation *op) {
  // TODO: consider using mlir::matchPattern(op, m_Constant())
  return op && op->hasTrait<OpTrait::ConstantLike>();
}

bool isConstantResult(Value v) { return isConstant(v.getDefiningOp()); }

// Returns nullptr if v is not a constant result.
Attribute getConstantAttribute(Operation *op) {
  if (!isConstant(op))
    return nullptr;
  SmallVector<OpFoldResult, 1> folded;
  auto ok = op->fold(folded);
  assert(succeeded(ok));
  assert(folded.size() == 1);
  assert(folded.front().is<Attribute>());
  return folded.front().get<Attribute>();
}

struct LazyConstPropRegion {
  using Span = std::pair<size_t, size_t>;

  void run(Region *region) {
    Operation *terminator = region->back().getTerminator();
    int numOperands = terminator->getNumOperands();
    for (int i = 0; i < numOperands; ++i) {
      if (auto span = runOnOperand(terminator->getOperand(i)))
        constantifyResult(terminator->getOperand(i), *span);
    }
  }

  // Returns a span if v is the result of an expression that can be
  // constantified, either it's already a constant, in which case the span is
  // empty, or it's an expression tree where every node can be lazy folded.
  // Returns nullopt otherwise, namely if v is a block argument or it
  // is a larger expression that cannot be constantified because
  // it has non-constant subexpressions or nodes that cannot be lazy folded.
  std::optional<Span> runOnOperand(Value v) {
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE " runOnOperand: " << v << "\n");

    Operation *defop = v.getDefiningOp();
    if (!defop)
      return std::nullopt;

    auto begin = opQueue.size();
    if (isConstant(defop) || opMap.contains(defop))
      return Span(begin, begin);

    // Ignore defop's regions as they were walked separately
    // in LazyConstPropONNXPass::runOnOperation().

    bool isFoldable = isConstantFoldable(defop);
    int numOperands = defop->getNumOperands();
    SmallVector<std::optional<Span>> spans(numOperands, std::nullopt);
    for (int i = 0; i < numOperands; ++i) {
      if ((spans[i] = runOnOperand(defop->getOperand(i))).has_value()) {
        if (!isFoldable)
          constantifyResult(defop->getOperand(i), *spans[i]);
      } else {
        if (isFoldable) {
          for (int j = 0; j < i; ++j)
            constantifyResult(defop->getOperand(j), *spans[j]);
        }
        isFoldable = false;
      }
    }
    if (!isFoldable)
      return std::nullopt;

    auto pos = opQueue.size();
    opQueue.push_back(defop);
    auto [_, inserted] = opMap.try_emplace(defop, pos);
    assert(inserted);
    auto end = opQueue.size();
    return Span(begin, end);
  }

  void constantifyResult(Value v, const Span &span) {
    assert(!v.use_empty());

    if (span.first == span.second) {
      assert(isConstantResult(v));
      return;
    }

    // Put opQueue[span.first:span.second] into a lazy func and make all the
    // results with users outside the span into lazy func results,
    // while checking that no users are before the span.
    assert(span.first < span.second);
    Operation *defop = v.getDefiningOp();
    assert(defop == opQueue[span.second - 1]);
    ModuleOp module = defop->getParentOfType<ModuleOp>();
    OpBuilder b(module.getBodyRegion());
    MLIRContext *ctx = b.getContext();
    auto *lazyCstDialect = ctx->getLoadedDialect<lazycst::LazyCstDialect>();
    StringAttr lazyFuncName =
        lazyCstDialect->lazyFunctionManager.nextName(module);
    auto lazyFunc = FlatSymbolRefAttr::get(lazyFuncName);

    Location loc = defop->getLoc();
    auto cstexpr = b.create<lazycst::LazyFuncOp>(loc, lazyFuncName);
    SymbolTable(module).insert(cstexpr);
    b.setInsertionPointToStart(cstexpr.addEntryBlock());
    auto lazyReturn = b.create<lazycst::LazyReturnOp>(loc, ValueRange{});
    b.setInsertionPoint(lazyReturn);

    const auto opIsOutside = [&](Operation *op) {
      auto it = opMap.find(op);
      if (it != opMap.end()) {
        auto pos = it->second;
        assert(opQueue[pos] == op && "opMap/opQueue invariant");
        assert(span.first <= pos);
        if (pos < span.second)
          return false;
      }
      return true;
    };
    const auto operandIsOutside = [&opIsOutside](OpOperand &operand) {
      return opIsOutside(operand.getOwner());
    };
    assert(llvm::all_of(v.getUsers(), opIsOutside));

    // TODO: consider making it a vector set to ensure determinism
    SmallPtrSet<Operation *, 4> unreachableConstants;
    llvm::SmallDenseMap<Attribute, Value> cloneConstants;
    SmallVector<Attribute> lazyArguments;
    SmallVector<Attribute> lazyResults;
    IRMapping mapping;
    for (auto [pos, end] = span; pos < end; ++pos) {
      Operation *op = opQueue[pos];

      unsigned numOperands = op->getNumOperands();
      SmallVector<Value> cstOperands(numOperands, nullptr);
      for (unsigned i = 0; i < numOperands; ++i) {
        Value operand = op->getOperand(i);
        Operation *operandOp = operand.getDefiningOp();
        if (Attribute attr = getConstantAttribute(operandOp)) {
          Value cst = cloneConstants.lookup(attr);
          if (!cst) {
            if (isa<lazycst::LazyElementsAttr, lazycst::FileDataElementsAttr>(
                    attr)) {
              Region &body = cstexpr.getBody();
              assert(body.getNumArguments() == lazyArguments.size());
              cst = body.addArgument(operand.getType(), operand.getLoc());
              lazyArguments.push_back(attr);
            } else {
              cst = b.clone(*operandOp)->getResult(0);
            }
            auto [_, inserted] = cloneConstants.try_emplace(attr, cst);
            assert(inserted);
          }
          cstOperands[i] = cst;
          if (!llvm::any_of(operandOp->getUsers(), opIsOutside))
            unreachableConstants.insert(operandOp);
        }
      }

      Operation *clone = b.clone(*op, mapping);
      for (unsigned i = 0; i < numOperands; ++i) {
        if (Value cst = cstOperands[i])
          clone->setOperand(i, cst);
      }

      {
        OpBuilder::InsertionGuard guard(b);
        unsigned numResults = op->getNumResults();
        for (unsigned j = 0; j < numResults; ++j) {
          Value res = op->getResult(j);
          if (llvm::any_of(res.getUsers(), opIsOutside)) {
            auto type = cast<ShapedType>(res.getType());
            unsigned index = lazyResults.size();
            auto lazyElms =
                lazycst::LazyElementsAttr::get(type, lazyFunc, index);
            lazyResults.push_back(lazyElms);
            lazyReturn.getOperandsMutable().append({clone->getResult(j)});
            b.setInsertionPointAfter(defop);
            Value cst =
                op->getName()
                    .getDialect()
                    ->materializeConstant(b, lazyElms, type, res.getLoc())
                    ->getResult(0);
            res.replaceUsesWithIf(cst, operandIsOutside);
          }
        }
      }
    }
    assert(!lazyResults.empty());

    const auto getAttrType = [](Attribute ta) {
      return cast<TypedAttr>(ta).getType();
    };
    SmallVector<Type> argTypes(llvm::map_range(lazyArguments, getAttrType));
    SmallVector<Type> resTypes(llvm::map_range(lazyResults, getAttrType));
    cstexpr.setFunctionType(b.getFunctionType(argTypes, resTypes));
    cstexpr.setArgConstantsAttr(b.getArrayAttr(ArrayRef(lazyArguments)));
    cstexpr.setResConstantsAttr(b.getArrayAttr(ArrayRef(lazyResults)));

    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE " cstexpr: " << cstexpr << "\n");
    assert(succeeded(verify(cstexpr)));

    for (auto [begin, pos] = span; pos > begin; --pos) {
      Operation *op = opQueue[pos - 1];
      assert(op->use_empty());
      // TODO: determine if operands are removed from use/def lists, or if we
      //       should walk from front and dropAllUses() before erase()
      op->erase();
    }
    for (Operation *op : unreachableConstants) {
      assert(op->use_empty());
      op->erase();
    }
  }

  SmallVector<Operation *> opQueue;
  DenseMap<Operation *, size_t> opMap;
};

using Ops = SmallPtrSet<Operation *, 1>;

class LazyConstPropAnalysis {
public:
  void run(Region *region) { runOnOperands(region->back().getTerminator()); }

private:
  bool runOnOperands(Operation *op) {
    return llvm::all_of(
        op->getOperands(), [this](Value v) { return runOnOperand(v); });
  }

  bool runOnOperand(Value operand) {
    Operation *defop = operand.getDefiningOp();
    if (!defop)
      return false;

    {
      auto [_, inserted] = visitedOps.insert(defop);
      if (!inserted)
        return LazyFoldableOps.contains(defop);
    }

    // Ignore defop's regions as they were walked separately
    // in LazyConstPropONNXPass::runOnOperation().

    if (isConstant(defop)) {
      assert(defop->getNumOperands() == 0);
    } else {
      bool allOperandsAreLazyFoldable = runOnOperands(defop);
      if (!allOperandsAreLazyFoldable || !isConstantFoldable(defop))
        return false;
    }
    auto [_, inserted] = LazyFoldableOps.insert(defop);
    assert(inserted);
    return true;
  }

public:
  Ops LazyFoldableOps;

private:
  Ops visitedOps;
};

template <typename... Ts>
const SmallPtrSet<TypeID, 1> setOfTypeIDs({TypeID::get<Ts>()...});

// TODO: generalize the more dialects
bool canOpGroup(Operation *op) {
  static const auto typeIDs = setOfTypeIDs<ONNXMinOp, ONNXMaxOp, ONNXSumOp>;
  return typeIDs.contains(op->getName().getTypeID());
}

// Groups constant foldable operands to variadic associaive and commutative ops
// like onnx.Min/Max/Sum, e.g. min(cfop1,x,cfop2) -> min(x,min(cfop1,cfop2))
// and adds the group min(cfop1,cfop2) to the constant foldable ops.
void group(Region *region, Ops &LazyFoldableOps) {
  for (Operation &op : region->getOps()) {
    if (LazyFoldableOps.contains(&op))
      continue;
    if (!canOpGroup(&op))
      continue;
    unsigned numOperands = op.getNumOperands();
    if (numOperands < 3)
      continue;
    SmallVector<Value> operandGroup;
    SmallVector<unsigned> operandGroupIdxs;
    for (unsigned i = 0; i < numOperands; ++i) {
      Value operand = op.getOperand(i);
      if (Operation *defop = operand.getDefiningOp()) {
        if (LazyFoldableOps.contains(defop))
          operandGroup.push_back(operand);
        operandGroupIdxs.push_back(i);
      }
    }
    assert(operandGroup.size() < op.getNumOperands() &&
           "operands can't all constant fold when op can't");
    if (operandGroup.size() >= 2) {
      // TODO: figure out if cloning and mutating op clashes with
      //       region->getOps() traversal
      OpBuilder b(&op);
      Operation *clone = b.clone(op);
      clone->setOperands(operandGroup);
      assert(clone->getNumResults() == 1 && "grouping ops have 1 result");
      op.setOperand(operandGroupIdxs.pop_back_val(), clone->getResult(0));
      for (unsigned i : llvm::reverse(operandGroupIdxs))
        op.eraseOperand(i);
    }
  }
}

// TODO: generalize the more dialects
bool canOpBubble(Operation *op) {
  static const auto typeIDs =
      setOfTypeIDs<ONNXMinOp, ONNXMaxOp, ONNXSumOp, ONNXMulOp>;
  return typeIDs.contains(op->getName().getTypeID());
}

// Grows the set of constant foldable ops (cfops) with rewrites that bubble
// cfops towards their uses, e.g. (x+cfop)+y -> (x+y)+cfop, and forms new cfops,
// e.g. (x+cfop1)+cfop2 -> x+cfop{1+2} where cfop{1+2} = cfop1+cfop2.
void bubble(Region *region, Ops &LazyFoldableOps) {
  SmallVector<Operation *> cfops(
      LazyFoldableOps.begin(), LazyFoldableOps.end());
  for (size_t i = 0; i < cfops.size(); ++i) {
    Operation *cfop = cfops[i];
    SmallVector<Operation *> users(cfop->getUsers());
    for (size_t j = 0; j < users.size(); ++j) {
      Operation *user = users[j];
      if (LazyFoldableOps.contains(user))
        continue;

      // TODO: continue if user is not an op that bubbles
      if (!canOpBubble(user))
        continue;

      if (!user->hasOneUse())
        continue;
      auto use = user->use_begin();
      Operation *root = use->getOwner();

      // TODO: generalize to more combinations, e.g. onnx.Add and Sub
      if (user->getName() != root->getName())
        continue;

      // We assume the user and root ops are associative and commutative.
      // TODO: generalize to e.g. onnx.Sub which isn't commutative

      // TODO: if root has another cfop operand, put it together with our cfop

      // TODO: otherwise just bubble cfop to become root operand

      // users.push_back(new root)
    }
  }
}

void lazyConstPropRegion(Region *region) {
  LazyConstPropAnalysis lcpa;
  lcpa.run(region);
  Ops &cfops = lcpa.LazyFoldableOps;
  group(region, cfops);
  bubble(region, cfops);
  LazyConstPropRegion().run(region);
}

Operation *negate(Value v, PatternRewriter &rewriter) {
  llvm_unreachable("TODO: implement this");
}

template <typename OpType>
class MyOpRewritePattern : public OpRewritePattern<OpType> {
  using Base = OpRewritePattern<OpType>;

public:
  // TODO: figure out if patterns need to fill in generatedNames
  template <typename... Args>
  MyOpRewritePattern(
      lazycst::ConstantFoldableAnalysis &analysis, Args &&... args)
      : Base(std::forward<Args>(args)...), analysis(analysis) {}

protected:
  lazycst::ConstantFoldableAnalysis &analysis;
};

template <typename OpInterfaceType>
class MyOpInterfaceRewritePattern
    : public OpInterfaceRewritePattern<OpInterfaceType> {
  using Base = OpInterfaceRewritePattern<OpInterfaceType>;

public:
  // TODO: figure out if patterns need to fill in generatedNames
  template <typename... Args>
  MyOpInterfaceRewritePattern(
      lazycst::ConstantFoldableAnalysis &analysis, Args &&... args)
      : Base(std::forward<Args>(args)...), analysis(analysis) {}

protected:
  lazycst::ConstantFoldableAnalysis &analysis;
};

bool isBubblableOpImpl(
    Operation *op, const lazycst::ConstantFoldableAnalysis &analysis) {
  assert(!analysis.isConstantFoldableOp(op));
  return op->hasOneUse() && op->getNumOperands() >= 2 &&
         analysis.isConstantFoldable(op->getOperand(0));
}

template <typename... OpTypes>
bool isBubblableOp(
    Operation *op, const lazycst::ConstantFoldableAnalysis &analysis) {
  return isBubblableOpImpl(op, analysis) && isa<OpTypes...>(op);
}

template <typename... OpTypes>
bool isBubblable(Value v, const lazycst::ConstantFoldableAnalysis &analysis) {
  Operation *op = v.getDefiningOp();
  return op && isBubblableOp<OpTypes...>(op, analysis);
}

template <typename OpType>
OpType castIfBubblableOp(
    Operation *op, const lazycst::ConstantFoldableAnalysis &analysis) {
  return isBubblableOpImpl(op, analysis) ? dyn_cast<OpType>(op) : nullptr;
}

Value cloneOpAndSetOperands(
    Operation *op, ValueRange operands, PatternRewriter &rewriter) {
  Operation *clone = rewriter.clone(*op);
  clone->setOperands(operands);
  // TODO: set result type
  return clone->getResult(0);
}

// TODO: create ACCF (Associative, Commutative, Constant-Foldable) op interface
//       and attach it to onnx.Add/Sum/Min/Xor etc
using ACCFOpInterface = CopyOpInterface; // TODO: replace this placeholder

// This pattern rewrites non-constant-foldable expression op(x1,x2) towards the
// form op(noncstfldb,cstfldb) (in any operand order) in one of two ways:
// 1. op(cstfldb1,op(cstfldb2,x2)) -> op(op(cstfldb1,cstfldb2),x2)
// 2. op(x1,op(cstfldb2,x2)) -> op(op(x1,x2),cstfldb2), if x1 is not constant
// foldable
struct BinaryACCFOpPattern
    : public MyOpInterfaceRewritePattern<ACCFOpInterface> {
  using MyOpInterfaceRewritePattern<
      ACCFOpInterface>::MyOpInterfaceRewritePattern;

  // Pair of a non-constant-foldable value and a constant-foldable value.
  using ValuePair = std::array<Value, 2>;

  LogicalResult doRewrite(ACCFOpInterface binaryOp, Value operand, ValuePair vp,
      PatternRewriter &rewriter) const {
    auto [noncstfldb, cstfldb] = vp;
    if (analysis.isConstantFoldable(operand)) {
      // rewrite op to op(noncstfldb,op(operand,cstfldb))
      cstfldb = cloneOpAndSetOperands(binaryOp, {operand, cstfldb}, rewriter);
    } else {
      // rewrite op to op(op(operand,noncstfldb),cstfldb)
      noncstfldb =
          cloneOpAndSetOperands(binaryOp, {operand, noncstfldb}, rewriter);
    }
    rewriter.startRootUpdate(binaryOp);
    binaryOp->setOperands({noncstfldb, cstfldb});
    rewriter.finalizeRootUpdate(binaryOp);
    return success();
  }

  std::optional<ValuePair> isBubbleable(
      OperationName opName, Value operand) const {
    if (Operation *defop = operand.getDefiningOp()) {
      if (defop->hasOneUse() && defop->getName() == opName) {
        assert(defop->getNumOperands() == 2);
        Value lhs = defop->getOperand(0), rhs = defop->getOperand(1);
        if (analysis.isConstantFoldable(rhs))
          return ValuePair{lhs, rhs};
        if (analysis.isConstantFoldable(lhs))
          return ValuePair{rhs, lhs};
      }
    }
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(
      ACCFOpInterface binaryOp, PatternRewriter &rewriter) const override {
    // TODO: try to insert this check in the base class MyOpRewritePattern
    if (analysis.isConstantFoldableOp(binaryOp))
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

struct AddPattern : public MyOpRewritePattern<ONNXAddOp> {
  using MyOpRewritePattern<ONNXAddOp>::MyOpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    // TODO: try to insert this check in the base class MyOpRewritePattern
    if (analysis.isConstantFoldableOp(addOp))
      return failure();
    Value lhs = addOp.getA();
    Value rhs = addOp.getB();
    bool swapped = false;
    if (analysis.isConstantFoldableOp(rhs.getDefiningOp())) {
      std::swap(lhs, rhs);
      swapped = true; // if addOp isn't replaced, swap the operands at the end
    } else if (!analysis.isConstantFoldable(lhs)) {
      if (!isBubblable<ONNXAddOp, ONNXSubOp>(lhs, analysis)) {
        if (!isBubblable<ONNXAddOp, ONNXSubOp>(rhs, analysis))
          return failure();
        std::swap(lhs, rhs);
      }
      // lhs is in canonical form: add/sub(cstfldb, x),
      // now bubble up to add(cstfldb, add/sub(rhs, x))
      Operation *lhsOp = lhs.getDefiningOp();
      Value cstfldb = lhsOp->getOperand(0);
      Operation *newRhs = rewriter.clone(*lhsOp); // add or sub
      newRhs->setOperand(0, rhs);
      Operation *newRoot = rewriter.create<ONNXAddOp>(
          addOp.getLoc(), cstfldb, newRhs->getResult(0));
      rewriter.replaceOp(addOp, newRoot->getResult(0));
      return success();
    }
    // addOp is in canonical form: add(cstfldb, rhs)
    if (isBubblable<ONNXAddOp, ONNXSubOp>(rhs, analysis)) {
      // rhs is in canonical form: add/sub(cstfldb2, x),
      // now bubble up to add/sub(add(cstfldb, cstfldb2), x)
      Operation *rhsOp = rhs.getDefiningOp();
      Value rhsLhs = rhsOp->getOperand(0);
      Location loc = rewriter.getFusedLoc({lhs.getLoc(), rhsLhs.getLoc()});
      Value newLhs = rewriter.create<ONNXAddOp>(loc, lhs, rhsLhs);
      Operation *newRoot = rewriter.clone(*rhsOp); // add or sub
      newRoot->setOperand(0, newLhs);
      rewriter.replaceOp(addOp, newRoot->getResult(0));
      return success();
    }
    if (swapped) {
      rewriter.updateRootInPlace(addOp, [&] {
        addOp->setOperands({lhs, rhs});
      });
      return success();
    }
    return failure();
  }
};

struct NegPattern : public MyOpRewritePattern<ONNXNegOp> {
  using MyOpRewritePattern<ONNXNegOp>::MyOpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXNegOp negOp, PatternRewriter &rewriter) const override {
    if (analysis.isConstantFoldableOp(negOp))
      return failure();
    Operation *defop = negOp.getX().getDefiningOp();
    if (!defop)
      return failure();
#if 1
    if (ONNXAddOp addOp = castIfBubblableOp<ONNXAddOp>(defop, analysis)) {
      Operation *negLhsOp = negate(addOp.getA(), rewriter);
      analysis.insertConstantFoldableOp(negLhsOp);
      Value negLhs = negLhsOp->getResult(0);
      Value rhs = addOp.getB();
      ONNXSubOp subOp = rewriter.create<ONNXSubOp>(addOp.getLoc(), negLhs, rhs);
      rewriter.replaceOp(negOp, subOp);
      return success();
    }
    if (ONNXSubOp subOp = castIfBubblableOp<ONNXSubOp>(defop, analysis)) {
      Operation *negLhsOp = negate(subOp.getA(), rewriter);
      analysis.insertConstantFoldableOp(negLhsOp);
      Value negLhs = negLhsOp->getResult(0);
      Value rhs = subOp.getB();
      ONNXAddOp addOp = rewriter.create<ONNXAddOp>(subOp.getLoc(), negLhs, rhs);
      rewriter.replaceOp(negOp, addOp);
      return success();
    }
#else
    if (!defop->hasOneUse())
      return failure();
    if (ONNXAddOp addOp = dyn_cast<ONNXAddOp>(defop)) {
      Value lhs = addOp.getA();
      if (!analysis.isConstantFoldable(lhs))
        return failure();
      Operation *negLhs = negate(lhs, rewriter);
      analysis.insertConstantFoldableOp(negLhs);
      Value newLhs = negLhs->getResult(0);
      Value rhs = addOp.getB();
      ONNXSubOp subOp = rewriter.create<ONNXSubOp>(addOp.getLoc(), newLhs, rhs);
      rewriter.replaceOp(negOp, subOp);
      return success();
    }
    if (ONNXSubOp subOp = dyn_cast<ONNXSubOp>(defop)) {
      Value lhs = subOp.getA();
      if (!analysis.isConstantFoldable(lhs))
        return failure();
      Operation *negLhs = negate(lhs, rewriter);
      analysis.insertConstantFoldableOp(negLhs);
      Value newLhs = negLhs->getResult(0);
      Value rhs = subOp.getB();
      ONNXAddOp addOp = rewriter.create<ONNXAddOp>(subOp.getLoc(), newLhs, rhs);
      rewriter.replaceOp(negOp, addOp);
      return success();
    }
#endif
    return failure();
  }
};

struct LazyConstPropONNXPass
    : public PassWrapper<LazyConstPropONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LazyConstPropONNXPass);

  StringRef getArgument() const override { return "lazy-constprop-onnx"; }

  StringRef getDescription() const override {
    return "Lazy constant propagation for the ONNX dialect.";
  }

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    func::FuncOp function = getOperation();

    lazycst::ConstantFoldableAnalysis analysis(function);

    RewritePatternSet patterns(ctx);
    patterns.insert<NegPattern>(analysis, ctx);

    function->walk(lazyConstPropRegion);
  }
};

} // namespace

void onnx_mlir::configureLazyConstPropONNXPass(int expansionBound) {
  LazyConstPropONNXPassConfiguration::expansionBound = expansionBound;
}

std::unique_ptr<mlir::Pass> onnx_mlir::createLazyConstPropONNXPass() {
  return std::make_unique<LazyConstPropONNXPass>();
}
