/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXAttributes.cpp -- ONNX attributes implementation --------===//
//
// ONNX attributes implementation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include <tuple>

using namespace mlir;

namespace onnx_mlir {

DisposableElements DisposableExpression::getResult(size_t i) { return nullptr; }

mlir::DisposableElementsAttr* DisposableElementsAttrBase::getAttr() {
  return static_cast<mlir::DisposableElementsAttr *>(this);
}

}
