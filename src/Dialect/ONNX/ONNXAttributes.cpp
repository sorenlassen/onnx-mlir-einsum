/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXAttributes.cpp -- ONNX attributes implementation --------===//
//
// ONNX attributes implementation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXAttributes.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include <tuple>

namespace mlir {

using namespace onnx_mlir;

/*static*/
DisposableElementsAttr DisposableElementsAttr::get(DisposableElements result) {
  return DisposableElementsAttr::get(result->getType(), nullptr, 0);
}

}

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentBoolElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentI16ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentF32ElementsAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ImpermanentU64ElementsAttr)

using namespace mlir;

namespace onnx_mlir {

DisposableElements DisposableExpression::getResult(size_t i) {
  force();
  return results[i];
}

// TODO: remove mytest
Attribute mytest(MLIRContext *ctx) {
  Builder b(ctx);
  ShapedType type = RankedTensorType::get({}, b.getF16Type());
  Attribute a;
  a = DisposableElementsAttr::get(type, nullptr, 0);
  a = ImpermanentI16ElementsAttr::get(type, nullptr);
  return a;
}

}
