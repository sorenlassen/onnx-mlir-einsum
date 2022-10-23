/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ONNXAttributes.hpp - ONNX Attributes ----------------===//
//
// This file defines attributes in the ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/ONNXDialect.hpp"

#include <memory>

namespace onnx_mlir {

struct DisposableElementsImpl {
  mlir::ShapedType type;
};
using DisposableElements = std::shared_ptr<DisposableElementsImpl>;

class DisposableExpression {
public:
  DisposableElements getResult(size_t i);
};
using DisposableResultsHandle = DisposableExpression *;

}

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/ONNX/ONNXAttributes.hpp.inc"

namespace mlir {

inline ::onnx_mlir::DisposableElements DisposableElementsAttr::getElements() const {
  return getResultsHandle()->getResult(getResultIndex());
}

}