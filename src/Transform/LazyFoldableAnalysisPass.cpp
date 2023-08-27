/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyFoldableAnalysis.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lazyfoldableanalysis-pass"

namespace {

using namespace mlir;
using namespace onnx_mlir;

struct LazyFoldableAnalysisPass
    : public PassWrapper<LazyFoldableAnalysisPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LazyFoldableAnalysisPass);

  StringRef getArgument() const override { return "lazyfoldableanalysis-pass"; }

  StringRef getDescription() const override {
    return "Runs LazyFoldableAnalysis";
  }

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    func::FuncOp function = getOperation();

    lazycst::LazyFoldableAnalysis analysis(function);

    // RewritePatternSet patterns(ctx);
    // patterns.insert<NegPattern>(analysis, ctx);
    //
    // function->walk(lazyConstPropRegion);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createLazyFoldableAnalysisPass() {
  return std::make_unique<LazyFoldableAnalysisPass>();
}
