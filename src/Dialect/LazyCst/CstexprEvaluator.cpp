/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/CstexprEvaluator.hpp"

#include "src/Dialect/LazyCst/ConstantFolder.hpp"
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"

using namespace mlir;

namespace lazycst {

CstexprEvaluator::CstexprEvaluator() {}

CstexprEvaluator::~CstexprEvaluator() = default;

void CstexprEvaluator::initialize(mlir::MLIRContext *ctx) {
  evaluator.initialize(ctx);
}

namespace {

class CstexprConstantFolder : public ConstantFolder {
public:
  void fold(Operation *cstexprOp, ArrayRef<mlir::Attribute> operands,
      llvm::SmallVectorImpl<mlir::Attribute> &results) const override final {
    lazycst::CstexprOp cstexpr = cast<lazycst::CstexprOp>(cstexprOp);
    // TODO: check that all lazy elms args constants are evaluated
    MLIRContext *ctx = cstexpr.getContext();
    const ConstantFolders &constantFolders =
        ctx->getLoadedDialect<lazycst::LazyCstDialect>()->constantFolders;
    GraphEvaluator cstexprEvaluator(ctx);
    // cstexprOp is used as a pseudo-op that folds with the args as results.
    cstexprEvaluator.addEvaluatedNode(
        cstexprOp, cstexpr.getInputs().getValue());
    Operation *terminator = cstexpr.getBody().back().getTerminator();
    for (auto it = cstexpr.getBody().op_begin(); &*it != terminator; ++it) {
      Operation *op = &*it;
      if (Attribute attr = getConstantAttribute(op)) {
        cstexprEvaluator.addEvaluatedNode(op, {attr});
      } else {
        const ConstantFolder *constantFolder =
            constantFolders.lookup(op->getName());
        SmallVector<GraphEvaluator::NodeOperand> operands;
        for (Value v : op->getOperands()) {
          // Value is OpResult or BlockArgument
          if (auto r = dyn_cast<OpResult>(v)) {
            operands.emplace_back(r.getOwner(), r.getResultNumber());
          } else {
            auto a = cast<BlockArgument>(v);
            assert(a.getOwner()->getParentOp() == cstexprOp);
            operands.emplace_back(cstexprOp, a.getArgNumber());
          }
        }
        cstexprEvaluator.addNode(op, operands, constantFolder);
      }
    }

    // Typically terminator's operands are one result of one result op but,
    // for completeness, support cstexpr args and multiple results and ops.
    SmallVector<Operation *, 1> resultOps;
    for (Value v : terminator->getOperands()) {
      if (auto r = dyn_cast<OpResult>(v))
        resultOps.push_back(r.getOwner());
    }
    SmallVector<ArrayRef<Attribute>, 1> attrs;
    cstexprEvaluator.evaluate(resultOps, attrs);
    ArrayAttr argConstants = cstexpr.getInputs();
    unsigned resultOpsIndex = 0;
    for (Value v : terminator->getOperands()) {
      Attribute result;
      if (auto r = dyn_cast<OpResult>(v)) {
        assert(r.getOwner() == resultOps[resultOpsIndex]);
        result = attrs[resultOpsIndex++][r.getResultNumber()];
      } else {
        auto a = cast<BlockArgument>(v);
        assert(a.getOwner()->getParentOp() == cstexprOp);
        result = argConstants[a.getArgNumber()];
      }
      results.push_back(result);
    }
  }
};

} // namespace

void CstexprEvaluator::record(
    lazycst::CstexprOp cstexpr, bool onlyLazyCstexprUsers) {
  static CstexprConstantFolder folder;
  assert(table.contains(cstexpr.getSymNameAttr()));
  SmallVector<GraphEvaluator::NodeOperand> operands;
  if (evaluator.hasNode(cstexpr))
    return;
  for (Attribute cstAttr : cstexpr.getInputs()) {
    if (auto lazyElms = dyn_cast<lazycst::LazyElementsAttr>(cstAttr)) {
      lazycst::CstexprOp callee = lookup(lazyElms.getCallee().getAttr());
      record(callee);
      operands.emplace_back(callee, lazyElms.getIndex());
    }
  }
  evaluator.addNode(cstexpr, operands, &folder, onlyLazyCstexprUsers);
}

void CstexprEvaluator::insert(StringAttr symName, mlir::Block *entryBlock) {
  table[symName] = entryBlock;
}

Attribute CstexprEvaluator::evaluate(
    lazycst::CstexprOp cstexpr, unsigned index) {
  SmallVector<ArrayRef<Attribute>, 1> attrs;
  evaluate({cstexpr}, attrs);
  return attrs.front()[index];
}

Attribute CstexprEvaluator::evaluate(StringAttr symName, unsigned index) {
  return evaluate(lookup(symName), index);
}

void CstexprEvaluator::evaluate(llvm::ArrayRef<lazycst::CstexprOp> cstexprs,
    llvm::SmallVectorImpl<llvm::ArrayRef<mlir::Attribute>> &results) {
  SmallVector<Operation *> ops;
  ops.reserve(cstexprs.size());
  for (lazycst::CstexprOp ce : cstexprs) {
    record(ce);
    ops.push_back(ce);
  }
  evaluator.evaluate(ops, results);
}

lazycst::CstexprOp CstexprEvaluator::lookup(StringAttr symName) const {
  Block *entryBlock = table.lookup(symName);
  return cast<lazycst::CstexprOp>(entryBlock->getParentOp());
}

} // namespace lazycst
