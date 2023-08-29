/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCst.hpp"
#include "src/Dialect/LazyCst/LazyFoldableAnalysis.hpp"
#include "src/Dialect/LazyCst/LazyFolder.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

using namespace mlir;

struct LazyFoldablePropagationPass
    : public PassWrapper<LazyFoldablePropagationPass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LazyFoldablePropagationPass);

  StringRef getArgument() const override {
    return "lazyfoldable-propagation-pass";
  }

  StringRef getDescription() const override {
    return "Propagates lazy-foldable operations sub-graphs";
  }

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    func::FuncOp function = getOperation();
    lazycst::LazyFoldableAnalysis analysis(function);

    RewritePatternSet patterns(ctx);
    ctx->getLoadedDialect<lazycst::LazyCstDialect>()->lazyFolders.getPatterns(
        analysis, patterns);
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createLazyFoldablePropagationPass() {
  return std::make_unique<LazyFoldablePropagationPass>();
}
