/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXAttributes.cpp -- ONNX attributes implementation --------===//
//
// ONNX attributes implementation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXAttributes.hpp"

template class ::mlir::ImpermanentElementsAttr<bool>;
template class ::mlir::ImpermanentElementsAttr<int8_t>;
template class ::mlir::ImpermanentElementsAttr<uint8_t>;
template class ::mlir::ImpermanentElementsAttr<int16_t>;
template class ::mlir::ImpermanentElementsAttr<uint16_t>;
template class ::mlir::ImpermanentElementsAttr<::onnx_mlir::float_16>;
template class ::mlir::ImpermanentElementsAttr<float>;
template class ::mlir::ImpermanentElementsAttr<uint64_t>;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentBoolElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentI8ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentU8ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentI16ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentU16ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentF16ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentF32ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentU64ElementsAttr)

namespace mlir {

size_t detail::uniqueNumber() {
  static std::atomic<size_t> counter{0};
  return ++counter;
}

}