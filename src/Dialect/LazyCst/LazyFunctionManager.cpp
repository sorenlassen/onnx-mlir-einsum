/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyFunctionManager.hpp"

#include "src/Dialect/LazyCst/ConstantFolder.hpp"
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"

using namespace mlir;

namespace lazycst {

LazyFunctionManager::LazyFunctionManager() : counter(0), evaluator(nullptr) {}

LazyFunctionManager::~LazyFunctionManager() = default;

LazyFuncOp LazyFunctionManager::create(SymbolTable &symbolTable, Location loc) {
  auto module = cast<ModuleOp>(symbolTable.getOp());
  OpBuilder b(module.getBodyRegion());
  auto cstexpr = b.create<lazycst::LazyFuncOp>(loc, nextName(symbolTable));
  symbolTable.insert(cstexpr);
  return cstexpr;
}

namespace {

class CstExprConstantFolder : public ConstantFolder {
public:
  void fold(mlir::Operation *cstexprOp,
      llvm::ArrayRef<mlir::Attribute> operands,
      llvm::SmallVectorImpl<mlir::Attribute> &results) const override final {
    LazyFuncOp cstexpr = cast<LazyFuncOp>(cstexprOp);
    const ConstantFolders &constantFolders =
        cstexpr.getContext()
            ->getLoadedDialect<lazycst::LazyCstDialect>()
            ->constantFolders;
    // TODO: use LazyFunctionManager thread pool
    GraphEvaluator cstexprEvaluator(nullptr);
    Operation *terminator = cstexpr.getBody().back().getTerminator();
    assert(terminator->hasOneUse());
    Operation *resultOp = *terminator->user_begin();
    for (Operation &op : cstexpr.getBody().getOps()) {
      if (&op != terminator) {
        const ConstantFolder *constantFolder =
            constantFolders.lookup(op.getName());
        llvm::SmallVector<mlir::Attribute> results;
        SmallVector<GraphEvaluator::NodeOperand> operands; // TODO: populate
        cstexprEvaluator.addNode(&op, operands, constantFolder);
      }
    }
    SmallVector<ArrayRef<Attribute>, 1> attrs;
    cstexprEvaluator.evaluate({resultOp}, attrs);
    ArrayRef<Attribute> resultAttrs = attrs.front();
    results.assign(resultAttrs.begin(), resultAttrs.end());
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
  evaluator.addNode(
      cstexpr, operands, &folder, onlyLazyFunctionUsers);
}

Attribute LazyFunctionManager::getResult(LazyFuncOp cstexpr, unsigned index) {
  SmallVector<ArrayRef<Attribute>, 1> attrs;
  evaluate({cstexpr}, attrs);
  return attrs.front()[index];
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
