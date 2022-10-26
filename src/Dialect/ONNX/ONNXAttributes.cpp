/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXAttributes.cpp -- ONNX attributes implementation --------===//
//
// ONNX attributes implementation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXAttributes.hpp"

template class ::mlir::DisposableElementsAttrBase<bool>;
template class ::mlir::DisposableElementsAttrBase<int8_t>;
template class ::mlir::DisposableElementsAttrBase<uint8_t>;
template class ::mlir::DisposableElementsAttrBase<int16_t>;
template class ::mlir::DisposableElementsAttrBase<uint16_t>;
template class ::mlir::DisposableElementsAttrBase<::onnx_mlir::float_16>;
template class ::mlir::DisposableElementsAttrBase<float>;
template class ::mlir::DisposableElementsAttrBase<uint64_t>;

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