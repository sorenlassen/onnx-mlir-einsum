/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyFolder.hpp"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#define IF_CF_DEBUG(X) X

namespace lazycst {

// Given that ONNX If/Loop/Scan regions can refer to values from parent regions,
// the CFAnalysis needs to be done as a "Preorder" region traversal.
class LazyFoldableAnalysis {
public:
#if 1
  LazyFoldableAnalysis(mlir::Operation *root);
  void insertConstantFoldableOp(mlir::Operation *op);
  bool isConstantFoldableOp(mlir::Operation *op) const;
  bool isConstantFoldable(mlir::Value v) const;
  llvm::SmallVector<unsigned> constantFoldableIdxs(mlir::ValueRange values);
#else
  LazyFoldableAnalysis(Operation *root)
      : lazyFolders(
            root->getContext()->getLoadedDialect<LazyCstDialect>().lazyFolders),
        root(root) {
    op->walk([this](Operation *op) {
      if (llvm::all_of(op->getOperands(),
              [this](Value v) { return isConstantFoldable(v); }) &&
          lazyFolders.match(op)) {
        insertConstantFoldableOp(op);
      }
      IF_CF_DEBUG({
        bool insd = visited.insert(op).second;
        assert(insd);
      })
    });
  }
  void insertConstantFoldableOp(mlir::Operation *op) {
    auto [_, inserted] = cfops.insert(op);
    assert(inserted);
  }
  bool isConstantFoldableOp(mlir::Operation *op) const {
    IF_CF_DEBUG(assert(visited.contains(op));)
    return op && cfops.contains(op);
  }
  bool isConstantFoldable(mlir::Value v) const {
    return isConstantFoldableOp(v.getDefiningOp());
  }
  llvm::SmallVector<unsigned> constantFoldableIdxs(mlir::ValueRange values) {
    llvm::SmallVector<unsigned> idxs;
    for (unsigned i = 0; i < values.size(); ++i) {
      if (isConstantFoldable(values[i]))
        idxs.push_back(i);
    }
    return idxs;
  }
#endif

private:
  using Ops = llvm::SmallPtrSet<mlir::Operation *, 1>;

  const LazyFolders &lazyFolders;
  [[maybe_unused]] mlir::Operation *root;
  IF_CF_DEBUG(Ops visited;)
  Ops cfops;
};

} // namespace lazycst
