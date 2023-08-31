/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/ConstantFoldableAnalysis.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace {

using namespace mlir;

struct ConstantFoldableAnalysisPass
    : public PassWrapper<ConstantFoldableAnalysisPass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantFoldableAnalysisPass);

  StringRef getArgument() const override {
    return "constantfoldable-analysis-pass";
  }

  StringRef getDescription() const override {
    return "Runs ConstantFoldableAnalysis";
  }

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    lazycst::ConstantFoldableAnalysis analysis(function, /*label=*/true);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createConstantFoldableAnalysisPass() {
  return std::make_unique<ConstantFoldableAnalysisPass>();
}
