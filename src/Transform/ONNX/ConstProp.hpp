/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace mlir {
class RewritePatternSet;
class MLIRContext;
}

namespace onnx_mlir {
void populateWithConstPropPatterns(mlir::RewritePatternSet &patterns, mlir::MLIRContext *context);
}