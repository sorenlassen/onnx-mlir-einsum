/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXConstProp.hpp - ONNX High Level Rewriting ------------===//
//
// This file implements a set of rewriters to constprop an ONNX operation into
// composition of other ONNX operations.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class RewritePatternSet;
class MLIRContext;
}

namespace onnx_mlir {
void populateWithConstPropPatterns(mlir::RewritePatternSet &patterns, mlir::MLIRContext *context);
}