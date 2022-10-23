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
  return DisposableElementsAttr::get(result->type, nullptr, 0);
}

}

using namespace mlir;

namespace onnx_mlir {

DisposableElements DisposableExpression::getResult(size_t i) { return nullptr; }

// TODO: remove mytest
Attribute mytest(MLIRContext *ctx) {
  Builder b(ctx);
  return DisposableElementsAttr::get(RankedTensorType::get({}, b.getF16Type()), nullptr, 0);
}

}
