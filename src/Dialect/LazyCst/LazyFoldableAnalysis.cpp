/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyFoldableAnalysis.hpp"

#include "src/Dialect/LazyCst/LazyCst.hpp"

#include "mlir/IR/OpDefinition.h"

using namespace mlir;

namespace lazycst {

namespace {

bool isConstant(Operation *op) {
  // TODO: consider using mlir::matchPattern(op, m_Constant())
  return op->hasTrait<OpTrait::ConstantLike>();
}

} // namespace

LazyFoldableAnalysis::LazyFoldableAnalysis(Operation *root, bool label)
    : label(label),
      lazyFolders(
          root->getContext()->getLoadedDialect<LazyCstDialect>()->lazyFolders),
      root(root) {
  root->walk([this](Operation *op) {
    bool areOperandsFoldable = llvm::all_of(
        op->getOperands(), [this](Value v) { return isConstantFoldable(v); });
    if (areOperandsFoldable &&
        (isConstant(op) || mlir::succeeded(lazyFolders.match(op))))
      insertConstantFoldableOp(op);
    IF_CF_DEBUG({
      bool insd = visited.insert(op).second;
      assert(insd);
    })
  });
}

void LazyFoldableAnalysis::insertConstantFoldableOp(mlir::Operation *op) {
  auto [_, inserted] = cfops.insert(op);
  assert(inserted);
  if (label)
    op->setAttr("lazyfoldable", mlir::UnitAttr::get(op->getContext()));
}

bool LazyFoldableAnalysis::isConstantFoldableOp(mlir::Operation *op) const {
  if (!op)
    return false;
  IF_CF_DEBUG(assert(visited.contains(op));)
  return cfops.contains(op);
}

bool LazyFoldableAnalysis::isConstantFoldable(mlir::Value v) const {
  return isConstantFoldableOp(v.getDefiningOp());
}

llvm::SmallVector<unsigned> LazyFoldableAnalysis::constantFoldableIdxs(
    mlir::ValueRange values) {
  llvm::SmallVector<unsigned> idxs;
  for (unsigned i = 0; i < values.size(); ++i) {
    if (isConstantFoldable(values[i]))
      idxs.push_back(i);
  }
  return idxs;
}

} // namespace lazycst
