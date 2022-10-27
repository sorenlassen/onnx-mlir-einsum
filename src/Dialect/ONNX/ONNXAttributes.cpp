/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXAttributes.cpp -- ONNX attributes implementation --------===//
//
// ONNX attributes implementation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXAttributes.hpp"

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)

namespace mlir {

size_t detail::uniqueNumber() {
  static std::atomic<size_t> counter{0};
  return ++counter;
}

} // namespace mlir