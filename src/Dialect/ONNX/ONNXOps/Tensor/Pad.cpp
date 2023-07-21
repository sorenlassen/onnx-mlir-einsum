/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Pad.cpp - ONNX Operations -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Pad operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

LogicalResult ONNXPadOpShapeHelper::computeShape() {
  ONNXPadOpAdaptor operandAdaptor(operands);
  Value dataOperand = operandAdaptor.getData();
  Value padsOperand = operandAdaptor.getPads();
  DimsExpr outputDims;

  // Get info about input data operand.
  uint64_t dataRank = createIE->getShapedTypeRank(dataOperand);

  // Initialize context and results (pads & output)
  pads.resize(2 * dataRank); // pads two sides of each axis.
  outputDims.resize(dataRank);

  // `pads` format is : [x1_begin, x2_begin,...,x1_end, x2_end,...],
  // where
  // - xi_begin: the number of pad values added at the beginning of axis `i`
  // - xi_end: the number of pad values added at the end of axis `i`.

  // Calculate output dimension sizes.
  for (uint64_t i = 0; i < dataRank; i++) {
    // Get begin/end pads.
    SymbolIndexExpr padBegin(createIE->getIntFromArrayAsSymbol(padsOperand, i));
    SymbolIndexExpr padEnd(
        createIE->getIntFromArrayAsSymbol(padsOperand, i + dataRank));
    if (padBegin.isUndefined() || padEnd.isUndefined())
      return op->emitError("pad parameter could not be processed");
    // Get input dim.
    DimIndexExpr dimInput(createIE->getShapeAsDim(dataOperand, i));

    // Calculation for output size.
    IndexExpr dimOutputFinal = (padBegin + dimInput) + padEnd;

    // Save results.
    pads[i] = padBegin;
    pads[i + dataRank] = padEnd;
    outputDims[i] = dimOutputFinal;
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXPadOp::verify() {
  ShapedType dataTy = cast<ShapedType>(getData().getType());

  ShapedType padsTy = cast<ShapedType>(getPads().getType());
  if (padsTy.hasRank() && padsTy.getRank() != 1)
    return emitOpError("non-1D pads: ") << padsTy;

  Type constTy = getConstantValue().getType();
  if (!isa<NoneType>(constTy)) {
    // Check that the constant has the same element type as the input
    ShapedType shapedConstTy = cast<ShapedType>(constTy);
    if (dataTy.getElementType() != shapedConstTy.getElementType()) {
      return emitOpError("Pad with constant_value that doesn't match the "
                         "element type of the input.");
    }
  }

  Type axesTy = getAxes().getType();
  if (!isa<NoneType>(axesTy)) {
    if (!isDenseONNXConstant(getAxes()))
      return emitOpError("non-constant axes input is not currently supported");
    if (cast<ShapedType>(axesTy).getRank() != 1)
      return emitOpError("non-1D axes: ") << axesTy;
    int64_t axesCount = cast<ShapedType>(axesTy).getDimSize(0);
    if (padsTy.hasStaticShape() && padsTy.getDimSize(0) != 2 * axesCount)
      return emitOpError("pads and axes count mismatch: ")
             << padsTy << " vs " << axesTy;
    if (dataTy.hasRank()) {
      ONNXConstantOp constOp = cast<ONNXConstantOp>(getAxes().getDefiningOp());
      ElementsAttr elements = constOp.getValueAttr().cast<ElementsAttr>();
      if (elements.getShapedType().getRank() != 1)
        return emitOpError("non-1D axes: ") << elements.getShapedType();
      llvm::SmallDenseSet<int64_t> axes;
      for (APInt a : elements.getValues<APInt>()) {
        int64_t i = a.getSExtValue();
        if (i < 0)
          i += dataTy.getRank();
        if (!(0 <= i && i < dataTy.getRank()))
          return emitOpError("axis out of range: ") << a.getSExtValue();
        auto [_, inserted] = axes.insert(i);
        if (!inserted)
          return emitOpError("repeated axis: ") << a.getSExtValue();
      }
    }
  } else if (dataTy.hasRank()) {
    if (padsTy.hasStaticShape() && padsTy.getDimSize(0) != 2 * dataTy.getRank())
      return emitOpError("pads count doesn't match data rank: ")
             << padsTy << " vs " << dataTy;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXPadOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getData()) || !hasShapeAndRank(getPads()))
    return success();

  Type elementType = getData().getType().cast<ShapedType>().getElementType();

  ONNXPadOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}
