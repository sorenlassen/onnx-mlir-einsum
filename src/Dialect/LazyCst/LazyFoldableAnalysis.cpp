/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyFoldableAnalysis.hpp"

#include "src/Dialect/LazyCst/LazyCst.hpp"

using namespace mlir;

namespace lazycst {

LazyFoldableAnalysis::LazyFoldableAnalysis(Operation *root)
    : lazyFolders(
          root->getContext()->getLoadedDialect<LazyCstDialect>()->lazyFolders),
      root(root) {
  root->walk([this](Operation *op) {
    if (llvm::all_of(op->getOperands(),
            [this](Value v) { return isConstantFoldable(v); }) &&
        mlir::succeeded(lazyFolders.match(op))) {
      insertConstantFoldableOp(op);
    }
    IF_CF_DEBUG({
      bool insd = visited.insert(op).second;
      assert(insd);
    })
  });
}

void LazyFoldableAnalysis::insertConstantFoldableOp(mlir::Operation *op) {
  auto [_, inserted] = cfops.insert(op);
  assert(inserted);
}

bool LazyFoldableAnalysis::isConstantFoldableOp(mlir::Operation *op) const {
  IF_CF_DEBUG(assert(visited.contains(op));)
  return op && cfops.contains(op);
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
