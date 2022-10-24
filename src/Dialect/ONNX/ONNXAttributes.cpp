/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXAttributes.cpp -- ONNX attributes implementation --------===//
//
// ONNX attributes implementation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXAttributes.hpp"

template class ::mlir::DisposableElementsAttr<bool>;
template class ::mlir::DisposableElementsAttr<int8_t>;
template class ::mlir::DisposableElementsAttr<uint8_t>;
template class ::mlir::DisposableElementsAttr<int16_t>;
template class ::mlir::DisposableElementsAttr<uint16_t>;
template class ::mlir::DisposableElementsAttr<::onnx_mlir::float_16>;
template class ::mlir::DisposableElementsAttr<float>;
template class ::mlir::DisposableElementsAttr<uint64_t>;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableBoolElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableI8ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableU8ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableI16ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableU16ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableF16ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableF32ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableU64ElementsAttr)

namespace mlir {

size_t detail::uniqueNumber() {
  static std::atomic<size_t> counter{0};
  return ++counter;
}

} // namespace mlir