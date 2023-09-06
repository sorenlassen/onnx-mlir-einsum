/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/ConstantFoldableAnalysis.hpp"

#include "src/Dialect/LazyCst/LazyCstDialect.hpp"

#include "mlir/IR/OpDefinition.h"

using namespace mlir;

namespace lazycst {

namespace {

bool isConstant(Operation *op) {
  // TODO: consider using mlir::matchPattern(op, m_Constant())
  return op->hasTrait<OpTrait::ConstantLike>();
}

} // namespace

ConstantFoldableAnalysis::ConstantFoldableAnalysis(Operation *root, bool label)
    : label(label), constantFolders(root->getContext()
                                        ->getLoadedDialect<LazyCstDialect>()
                                        ->constantFolders) {
  root->walk([this](Operation *op) {
    bool areOperandsFoldable = llvm::all_of(
        op->getOperands(), [this](Value v) { return isConstantFoldable(v); });
    if (areOperandsFoldable &&
        (isConstant(op) || mlir::succeeded(constantFolders.match(op))))
      insertConstantFoldableOp(op);
  });
}

void ConstantFoldableAnalysis::insertConstantFoldableOp(mlir::Operation *op) {
  auto [_, inserted] = cfops.insert(op);
  assert(inserted);
  if (label)
    op->setAttr("constantfoldable", mlir::UnitAttr::get(op->getContext()));
}

bool ConstantFoldableAnalysis::isConstantFoldableOp(mlir::Operation *op) const {
  if (!op)
    return false;
  return cfops.contains(op);
}

bool ConstantFoldableAnalysis::isConstantFoldable(mlir::Value v) const {
  return isConstantFoldableOp(v.getDefiningOp());
}

// llvm::SmallVector<unsigned> ConstantFoldableAnalysis::ConstantFoldableIdxs(
//     mlir::ValueRange values) {
//   llvm::SmallVector<unsigned> idxs;
//   for (unsigned i = 0; i < values.size(); ++i) {
//     if (isConstantFoldable(values[i]))
//       idxs.push_back(i);
//   }
//   return idxs;
// }

} // namespace lazycst
