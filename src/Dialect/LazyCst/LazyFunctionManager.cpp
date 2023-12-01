/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyFunctionManager.hpp"

#include "src/Dialect/LazyCst/ConstantFolder.hpp"
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"

#include <algorithm>
#include <iterator>

using namespace mlir;

namespace lazycst {

LazyFunctionManager::LazyFunctionManager() : counter(0) {}

LazyFunctionManager::~LazyFunctionManager() = default;

void LazyFunctionManager::initialize(mlir::MLIRContext *ctx) {
  evaluator.initialize(ctx);
}

LazyFuncOp LazyFunctionManager::create(SymbolTable &symbolTable, Location loc) {
  auto module = cast<ModuleOp>(symbolTable.getOp());
  OpBuilder b(module.getBodyRegion());
  StringAttr name = nextName(symbolTable);
  auto cstexpr = b.create<lazycst::LazyFuncOp>(loc, name);
  symbolTable.insert(cstexpr);
  table.try_emplace(name, cstexpr);
  return cstexpr;
}

namespace {

class CstExprConstantFolder : public ConstantFolder {
public:
  void fold(Operation *cstexprOp, ArrayRef<mlir::Attribute> operands,
      llvm::SmallVectorImpl<mlir::Attribute> &results) const override final {
    LazyFuncOp cstexpr = cast<LazyFuncOp>(cstexprOp);
    // TODO: check that all lazy elms args constants are evaluated
    MLIRContext *ctx = cstexpr.getContext();
    const ConstantFolders &constantFolders =
        ctx->getLoadedDialect<lazycst::LazyCstDialect>()->constantFolders;
    GraphEvaluator cstexprEvaluator(ctx);
    // cstexprOp is used as a pseudo-op that folds with the args as results.
    cstexprEvaluator.addEvaluatedNode(
        cstexprOp, cstexpr.getArgConstantsAttr().getValue());
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
    ArrayAttr argConstants = cstexpr.getArgConstantsAttr();
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

void LazyFunctionManager::record(
    SymbolTable &symbolTable, LazyFuncOp cstexpr, bool onlyLazyFunctionUsers) {
  static CstExprConstantFolder folder;
  SmallVector<GraphEvaluator::NodeOperand> operands;
  for (Attribute cstAttr : cstexpr.getArgConstantsAttr()) {
    if (auto lazyElms = dyn_cast<lazycst::LazyElementsAttr>(cstAttr)) {
      Operation *callee = symbolTable.lookup(lazyElms.getCallee().getAttr());
      operands.emplace_back(callee, lazyElms.getIndex());
    }
  }
  evaluator.addNode(cstexpr, operands, &folder, onlyLazyFunctionUsers);
}

LazyFuncOp LazyFunctionManager::lookup(StringAttr symName) const {
  return table.lookup(symName);
}

Attribute LazyFunctionManager::getResult(LazyFuncOp cstexpr, unsigned index) {
  SmallVector<ArrayRef<Attribute>, 1> attrs;
  evaluate({cstexpr}, attrs);
  return attrs.front()[index];
}

Attribute LazyFunctionManager::getResult(StringAttr symName, unsigned index) {
  return getResult(lookup(symName), index);
}

void LazyFunctionManager::evaluate(llvm::ArrayRef<LazyFuncOp> cstexprs,
    llvm::SmallVectorImpl<llvm::ArrayRef<mlir::Attribute>> &results) {
  SmallVector<Operation *> ops(llvm::map_range(
      cstexprs, [](LazyFuncOp cstexpr) { return cstexpr.getOperation(); }));
  evaluator.evaluate(ops, results);
}

StringAttr LazyFunctionManager::nextName(SymbolTable &symbolTable) {
  unsigned subscript = counter++;
  auto name = StringAttr::get(
      symbolTable.getOp()->getContext(), "lazycst." + Twine(subscript));
  assert(!symbolTable.lookup(name) && "next LazyFuncOp name was already taken");
  return name;
}

} // namespace lazycst
