/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCstOps.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#define GET_OP_CLASSES
#include "src/Dialect/LazyCst/LazyCstOps.cpp.inc"

using namespace mlir;

void lazycst::ExprOp::build(OpBuilder &odsBuilder, OperationState &odsState,
    StringAttr sym_name, ArrayAttr inputs) {
  auto emptyOutputs = odsBuilder.getArrayAttr({});
  build(odsBuilder, odsState, sym_name, inputs, emptyOutputs);
}

// Implementation is adapted from func::ReturnOp.
LogicalResult lazycst::YieldOp::verify() {
  auto cstexpr = cast<lazycst::ExprOp>((*this)->getParentOp());
  auto outputs = cstexpr.getOutputs();

  // The operand number and types must match the outputs.
  if (getNumOperands() != outputs.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << cstexpr.getName() << ") outputs " << outputs.size();

  for (unsigned i = 0, e = outputs.size(); i != e; ++i) {
    auto elms = cast<ElementsAttr>(outputs[i]);
    if (getOperand(i).getType() != elms.getType())
      return emitError() << "type of yield operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match expr type (" << elms.getType()
                         << ")"
                         << " in expr @" << cstexpr.getName();
  }

  return success();
}
