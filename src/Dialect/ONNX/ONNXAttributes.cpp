/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXAttributes.cpp -- ONNX attributes implementation --------===//
//
// ONNX attributes implementation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXAttributes.hpp"

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentBoolElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentI16ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentF32ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentU64ElementsAttr)

namespace mlir {

size_t uniqueNumber() {
  static std::atomic<size_t> counter{0};
  return ++counter;
}

}