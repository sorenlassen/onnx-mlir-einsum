/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/ConstantFolder.hpp"

#include "src/Dialect/LazyCst/ACConstantFoldableOpInterface.hpp"
#include "src/Dialect/LazyCst/ConstantFoldableAnalysis.hpp"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace lazycst {

namespace {

Operation *cloneOpAndSetOperands(
    Operation *op, ValueRange operands, PatternRewriter &rewriter) {
  Operation *clone = rewriter.clone(*op);
  clone->setOperands(operands);
  // TODO: set result type
  return clone;
}

// This pattern rewrites non-constant-foldable expression op(x1,x2) towards the
// form op(noncstfldb,cstfldb) (in any operand order) in one of two ways:
// 1. op(cstfldb1,op(cstfldb2,x2)) -> op(op(cstfldb1,cstfldb2),x2)
// 2. op(x1,op(cstfldb2,x2)) -> op(op(x1,x2),cstfldb2), if x1 is not constant
// foldable
//
// TODO: generalize to variadic ops, like onnx.Min/Max/Sum
struct BinaryACCFOpPattern
    : public OpInterfaceRewritePattern<lazycst::ACConstantFoldableOpInterface> {
  using OpIF = lazycst::ACConstantFoldableOpInterface;
  using Base = OpInterfaceRewritePattern<OpIF>;

  BinaryACCFOpPattern(
      lazycst::ConstantFoldableAnalysis &analysis, MLIRContext *ctx)
      : Base(ctx), analysis(analysis) {}

  // Pair of a non-constant-foldable value and a constant-foldable value.
  using ValuePair = std::array<Value, 2>;

  LogicalResult doRewrite(OpIF binaryOp, Value operand, ValuePair vp,
      PatternRewriter &rewriter) const {
    auto [noncstfldb, cstfldb] = vp;
    if (analysis.isConstantFoldable(operand)) {
      // rewrite op to op(noncstfldb,op(operand,cstfldb))
      Operation *cstfldbOp =
          cloneOpAndSetOperands(binaryOp, {operand, cstfldb}, rewriter);
      analysis.insertConstantFoldableOp(cstfldbOp);
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

  std::optional<ValuePair> hasConstantFoldableOperand(Operation *op) const {
    assert(op->getNumOperands() == 2);
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    if (analysis.isConstantFoldable(rhs))
      return ValuePair{lhs, rhs};
    if (analysis.isConstantFoldable(lhs))
      return ValuePair{rhs, lhs};
    return std::nullopt;
  }

  std::optional<ValuePair> isBubbleable(
      OperationName opName, Value operand) const {
    if (Operation *defop = operand.getDefiningOp()) {
      // Don't bubble up any operand if defop is foldable in whole,
      // and don't decompose it by bubbling if it has multiple uses.
      if (!analysis.isConstantFoldableOp(defop) && defop->hasOneUse() &&
          defop->getName() == opName)
        return hasConstantFoldableOperand(defop);
    }
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(
      OpIF binaryOp, PatternRewriter &rewriter) const override {
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

  lazycst::ConstantFoldableAnalysis &analysis;
};

} // namespace

bool isConstant(Operation *op) {
  // TODO: consider using mlir::matchPattern(op, m_Constant())
  return op->hasTrait<OpTrait::ConstantLike>();
}

bool isConstantResult(Value v) {
  if (Operation *defop = v.getDefiningOp())
    return isConstant(defop);
  return false;
}

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

void ConstantFolders::getPatterns(ConstantFoldableAnalysis &analysis,
    mlir::RewritePatternSet &results) const {
  results.insert<BinaryACCFOpPattern>(analysis, results.getContext());
  for (const PatternsGetter &getter : patternsGetters)
    getter(analysis, results);
}

} // namespace lazycst
