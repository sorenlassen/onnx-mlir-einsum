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

namespace mlir {
class DisposableElementsAttr;
}

namespace onnx_mlir {

struct DisposableElementsImpl {};
using DisposableElements = std::shared_ptr<DisposableElementsImpl>;
class DisposableExpression {
public:
  DisposableElements getResult(size_t i);
};
using DisposableResultsHandle = DisposableExpression *;
class DisposableElementsAttrBase : public mlir::Attribute {
public:
  using mlir::Attribute::Attribute;

  /// Allow implicit conversion to ElementsAttr.
  operator mlir::ElementsAttr() const {
    return *this ? cast<mlir::ElementsAttr>() : nullptr;
  }
private:
  mlir::DisposableElementsAttr *getAttr();
};

}

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/ONNX/ONNXAttributes.hpp.inc"
