/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/ConstantFolder.hpp"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace lazycst {

// Given that ONNX If/Loop/Scan regions can refer to values from parent regions,
// the CFAnalysis needs to be done as a "Preorder" region traversal.
class ConstantFoldableAnalysis {
public:
  ConstantFoldableAnalysis(mlir::Operation *root, bool label = false);
  void insertConstantFoldableOp(mlir::Operation *op);
  bool isConstantFoldableOp(mlir::Operation *op) const;
  bool isConstantFoldable(mlir::Value v) const;
  // llvm::SmallVector<unsigned> ConstantFoldableIdxs(mlir::ValueRange values);

private:
  using Ops = llvm::SmallPtrSet<mlir::Operation *, 1>;

  const bool label;
  const ConstantFolders &constantFolders;
  Ops cfops;
};

} // namespace lazycst
