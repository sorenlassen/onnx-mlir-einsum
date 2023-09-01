/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ONNXOps.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect operations.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "src/Dialect/Mlir/ConstantPrinter.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"

#include "mlir/Dialect/Traits.h"
#include "llvm/ADT/STLExtras.h"

//===----------------------------------------------------------------------===//
// Unsupported Operations
//===---------------------------------------------------------------------===//

// Operations for which shape inference has not been implemented.
#define UNSUPPORTED_OPS(OP_TYPE)                                               \
  /* shape inference interface method */                                       \
  mlir::LogicalResult mlir::OP_TYPE::inferShapes(                              \
      std::function<void(mlir::Region &)> doShapeInference) {                  \
    return mlir::success();                                                    \
  }

#include "src/Dialect/ONNX/ONNXUnsupportedOps.hpp"
#undef UNSUPPORTED_OPS

namespace {

using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Get a broadcasted type for RankedTensorType and MemRefType.
// Used in generated code in ONNXOps.cpp.inc included below.
//===----------------------------------------------------------------------===//
Type getBroadcastedRankedType(
    Type type1, Type type2, Type elementType = nullptr) {
  if (type1.isa<RankedTensorType>() && type2.isa<RankedTensorType>())
    return OpTrait::util::getBroadcastedType(type1, type2, elementType);
  if (type1.isa<MemRefType>() && type2.isa<MemRefType>()) {
    // Construct RankedTensorType(s).
    if (!elementType)
      elementType = type1.cast<MemRefType>().getElementType();
    RankedTensorType ty1 =
        RankedTensorType::get(type1.cast<MemRefType>().getShape(), elementType);
    RankedTensorType ty2 =
        RankedTensorType::get(type2.cast<MemRefType>().getShape(), elementType);
    // Compute a broadcasted type.
    Type outputType = OpTrait::util::getBroadcastedType(ty1, ty2);
    // Construct a MemRefType.
    return MemRefType::get(
        outputType.cast<RankedTensorType>().getShape(), elementType);
  } else
    return {};
}

} // namespace

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Dialect/ONNX/ONNXOps.cpp.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ONNXConstantOp custom assembly format print and parse.
// If the op has a sparse_value attr, it just prints its SparseElementsAttr:
//
//   onnx.Constant sparse<0, 1.000000e+00> : tensor<3xf32>
//
// and, if the op has a value attr, it just prints its DenseElementsAttr:
//
//   onnx.Constant dense<[5, 5, 16, 2]> : tensor<4xi64>
//
// provided the ElementsAttr type matches the op result type.
//
// TODO: Ensure constant result types always match their attributes because
//       constant folding needs that.
//
// In case of type mismatch or if the op has another attr than sparse_value or
// value or has no attr, the op prints an attribute dictionary followed by the
// op result type, just like the default assembly format.
//
//   onnx.Constant {value = dense<1024> : tensor<1xi64>} : tensor<*xi64>
//
//   onnx.Constant {value_int = 1 : si64} : tensor<i64>
//
//   onnx.Constant : tensor<64xf32>
//
//===----------------------------------------------------------------------===//

void ONNXConstantOp::print(OpAsmPrinter &printer) {
  // If the result type is dynamic then it won't match the attribute type and
  // we fall back to printing as attribute dictionary at the end.
  Type resultType = getResult().getType();
  if (auto attr = getValue()) {
    // ONNXConstantOp value must be ElementsAttr, but not SparseElementsAttr.
    auto elements = attr->cast<ElementsAttr>();
    assert(!elements.isa<SparseElementsAttr>() &&
           "ONNXConstantOp value cannot be sparse");
    if (elements.getType() == resultType) {
      printer << ' ';
      ConstantPrinter(printer).printAttribute(elements);
      return;
    }
  }
  if (auto attr = getSparseValue()) {
    // ONNXConstantOp sparse_value must be SparseElementsAttr.
    auto sparseElements = attr->cast<SparseElementsAttr>();
    if (sparseElements.getType() == resultType) {
      printer << ' ';
      printer.printAttribute(sparseElements);
      return;
    }
  }
  // Fallback if there's something funny: no value or sparse_value attribute,
  // or types mismatch.
  ConstantPrinter(printer).printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << resultType;
}

ParseResult ONNXConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  Attribute attr;
  Type type;
  // First try to parse attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  // If there is no attribute dictionary, the parse above succeeds parsing
  // nothing. We detect this case by the absence of result attributes.
  if (result.attributes.empty()) {
    // Try to parse a SparseElementsAttr or or other ElementsAttr.
    OptionalParseResult opt = parser.parseOptionalAttribute(attr, type);
    if (opt.has_value()) {
      if (*opt)
        return failure();
      const char *name =
          attr.isa<SparseElementsAttr>() ? "sparse_value" : "value";
      result.addAttribute(name, attr);
      result.addTypes({attr.cast<ElementsAttr>().getType()});
      return success();
    }
    // No sparse_value or value attr, so attribute dictionary really is empty.
  }
  if (parser.parseColonType(type))
    return failure();
  result.addTypes({type});
  return success();
}

//===----------------------------------------------------------------------===//
// ONNXConstantOfShapeOp custom assembly format print and parse.
// Same as the generic format except that any DenseLikeElementsAttr is
// printed with printAsDenseElementsAttr().
//===----------------------------------------------------------------------===//

void ONNXConstantOfShapeOp::print(OpAsmPrinter &printer) {
  printer << "(";
  printer.printOperand(getInput());
  printer << ")";
  ConstantPrinter(printer).printOptionalAttrDict((*this)->getAttrs());
  printer << " : ";
  printer.printFunctionalType(*this);
}

ParseResult ONNXConstantOfShapeOp::parse(
    OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  Type arg, res;
  if (parser.parseLParen() || parser.parseOperand(input) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseLParen() || parser.parseType(arg) ||
      parser.resolveOperand(input, arg, result.operands) ||
      parser.parseRParen() || parser.parseArrow() || parser.parseType(res))
    return failure();
  result.addTypes({res});
  return success();
}

//===----------------------------------------------------------------------===//
// Constant Materialize for ONNX Dialect
//===----------------------------------------------------------------------===//
Operation *ONNXDialect::materializeConstant(
    OpBuilder &builder, Attribute value, Type type, Location loc) {
  // The attribute could be either a UnitAttr or DenseElementsAttr, IntAttr,
  // FloatAttr and etc.
  // OnnxBuilder converts it into (the result of) a ONNXNoneOp or
  // ONNXConstantOp.
  MultiDialectBuilder<OnnxBuilder> create(builder, loc);
  Value result =
      isa<UnitAttr>(value) ? create.onnx.none() : create.onnx.constant(value);
  return result.getDefiningOp();
}
