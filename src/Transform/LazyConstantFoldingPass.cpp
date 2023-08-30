/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCst.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"
#include "src/Dialect/LazyCst/LazyFoldableAnalysis.hpp"
#include "src/Dialect/LazyCst/LazyFolder.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

using namespace mlir;
using lazycst::LazyFoldableAnalysis;

bool isConstant(Operation *op) {
  // TODO: consider using mlir::matchPattern(op, m_Constant())
  return op->hasTrait<OpTrait::ConstantLike>();
}

// Return a vector of non-constant lazy foldable ops for every lazy constant.
// Outer and inner vectors are in reverse topological order: successors before
// predecessors.
std::vector<std::vector<Operation *>> lazyConstantResultOps(
    func::FuncOp function) {
  std::vector<std::vector<Operation *>> lazyConstantOps;
  LazyFoldableAnalysis analysis(function);
  DenseMap<Operation *, size_t> lazyConstantMap;
  // assert: function ends in terminator which is no lazy foldable
  function->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isConstant(op))
      return;
    if (!analysis.isLazyFoldableOp(op))
      return;
    std::optional<size_t> idx;
    for (Operation *user : op->getUsers()) {
      assert(!isConstant(user));
      if (analysis.isLazyFoldableOp(user)) {
        auto it = lazyConstantMap.find(user);
        if (it == lazyConstantMap.end())
          continue;
        size_t userIdx = it->second;
        if (!idx.has_value() || idx.value() == userIdx) {
          idx = userIdx;
          continue;
        }
      }
      // op has a non lazy foldable user, or it is used by two or more other
      // lazy constants - in either case, we make op a new lazy constant result
      idx = lazyConstantOps.size();
      lazyConstantOps.emplace_back();
      break;
    }
    if (!idx.has_value())
      return;
    lazyConstantOps[idx.value()].push_back(op);
    auto [_, inserted] = lazyConstantMap.try_emplace(op, idx.value());
    assert(inserted);
  });
  return lazyConstantOps;
}

void convertIntoLazyConstant(const std::vector<Operation *> &ops,
    StringAttr lazyFuncName, SymbolTable &symbolTable) {
  auto module = cast<ModuleOp>(symbolTable.getOp());
  OpBuilder b(module.getBodyRegion());

  assert(!ops.empty());
  Operation *resultOp = ops.front();
  Location loc = resultOp->getLoc();
  auto cstexpr = b.create<lazycst::LazyFuncOp>(loc, lazyFuncName);
  symbolTable.insert(cstexpr);

  llvm_unreachable("TODO: implement this");
}

struct LazyConstantFoldingPass
    : public PassWrapper<LazyConstantFoldingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LazyConstantFoldingPass);

  StringRef getArgument() const override {
    return "lazy-constant-folding-pass";
  }

  StringRef getDescription() const override {
    return "Lazily constant folds into lazy constants";
  }

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    auto resultOps = lazyConstantResultOps(function);
    auto module = function->getParentOfType<ModuleOp>();
    SymbolTable symbolTable(module);
    auto *lazyCstDialect =
        getContext().getLoadedDialect<lazycst::LazyCstDialect>();
    for (const auto &ops : llvm::reverse(resultOps)) {
      StringAttr lazyFuncName =
          lazyCstDialect->lazyFunctionManager.nextName(module);
      convertIntoLazyConstant(ops, lazyFuncName, symbolTable);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createLazyConstantFoldingPass() {
  return std::make_unique<LazyConstantFoldingPass>();
}
