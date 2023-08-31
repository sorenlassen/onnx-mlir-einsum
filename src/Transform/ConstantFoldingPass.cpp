/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/ConstantFolder.hpp"
#include "src/Dialect/LazyCst/LazyCst.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace {

using namespace mlir;

bool isConstant(Operation *op) {
  // TODO: consider using mlir::matchPattern(op, m_Constant())
  return op->hasTrait<OpTrait::ConstantLike>();
}

bool isConstantResult(Value v) {
  if (Operation *defop = v.getDefiningOp())
    return isConstant(defop);
  return false;
}

// Precondition: isConstant(op)
Attribute getConstantAttribute(Operation *op) {
  SmallVector<OpFoldResult, 1> folded;
  auto ok = op->fold(folded);
  assert(succeeded(ok));
  assert(folded.size() == 1);
  assert(folded.front().is<Attribute>());
  return folded.front().get<Attribute>();
}

bool tryConstantFold(
    Operation *op, const lazycst::ConstantFolders &constantFolders) {
  if (isConstant(op))
    return false;
  if (!llvm::all_of(op->getOperands(), isConstantResult))
    return false;
  OperationName name = op->getName();
  auto *constantFolder = constantFolders.lookup(name);
  if (!constantFolder)
    return false;
  if (!succeeded(constantFolder->match(op)))
    return false;
  SmallVector<Attribute> operandsAttrs(llvm::map_range(op->getOperands(),
      [](Value v) { return getConstantAttribute(v.getDefiningOp()); }));
  SmallVector<Attribute> resultsAttrs;
  constantFolder->fold(op, operandsAttrs, resultsAttrs);
  assert(resultsAttrs.size() == op->getNumResults());
  Dialect *dialect = name.getDialect();
  Location loc = op->getLoc();
  OpBuilder b(op);
  for (auto [attr, res] : llvm::zip(resultsAttrs, op->getResults())) {
    Operation *cstOp =
        dialect->materializeConstant(b, attr, res.getType(), loc);
    res.replaceAllUsesWith(cstOp->getResult(0));
  }
  return true;
}

struct ConstantFoldingPass
    : public PassWrapper<ConstantFoldingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantFoldingPass);

  StringRef getArgument() const override { return "constant-folding-pass"; }

  StringRef getDescription() const override { return "Constant folds"; }

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    MLIRContext *ctx = &getContext();
    const auto &constantFolders =
        ctx->getLoadedDialect<lazycst::LazyCstDialect>()->constantFolders;
    function->walk<WalkOrder::PreOrder>([&](Region *region) {
      for (Operation &op : llvm::make_early_inc_range(region->getOps())) {
        if (tryConstantFold(&op, constantFolders)) {
          assert(op.use_empty());
          op.erase();
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createConstantFoldingPass() {
  return std::make_unique<ConstantFoldingPass>();
}
