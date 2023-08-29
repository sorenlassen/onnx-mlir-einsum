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
  LazyFoldableAnalysis(mlir::Operation *root, bool label = false);
  void insertLazyFoldableOp(mlir::Operation *op);
  bool isLazyFoldableOp(mlir::Operation *op) const;
  bool isLazyFoldable(mlir::Value v) const;
  llvm::SmallVector<unsigned> LazyFoldableIdxs(mlir::ValueRange values);

private:
  using Ops = llvm::SmallPtrSet<mlir::Operation *, 1>;

  const bool label;
  const LazyFolders &lazyFolders;
  [[maybe_unused]] mlir::Operation *root;
  IF_CF_DEBUG(Ops visited;)
  Ops cfops;
};

} // namespace lazycst
