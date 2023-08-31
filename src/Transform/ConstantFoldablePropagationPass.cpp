/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/ConstantFoldableAnalysis.hpp"
#include "src/Dialect/LazyCst/ConstantFolder.hpp"
#include "src/Dialect/LazyCst/LazyCst.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

using namespace mlir;

struct ConstantFoldablePropagationPass
    : public PassWrapper<ConstantFoldablePropagationPass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantFoldablePropagationPass);

  StringRef getArgument() const override {
    return "constantfoldable-propagation-pass";
  }

  StringRef getDescription() const override {
    return "Propagates constant-foldable operations sub-graphs";
  }

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    func::FuncOp function = getOperation();
    lazycst::ConstantFoldableAnalysis analysis(function);

    RewritePatternSet patterns(ctx);
    ctx->getLoadedDialect<lazycst::LazyCstDialect>()
        ->constantFolders.getPatterns(analysis, patterns);
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createConstantFoldablePropagationPass() {
  return std::make_unique<ConstantFoldablePropagationPass>();
}
