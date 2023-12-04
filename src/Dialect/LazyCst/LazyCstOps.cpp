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

// Implementation is adapted from function_interface_impl::printFunctionOp().
void lazycst::ExprOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  p << '(';
  llvm::interleaveComma(getBody().getArguments(), p,
      [&](auto arg) { p.printRegionArgument(arg); });
  p << ") " << getInputs() << " -> " << getOutputs();
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
      /*elidedAttrs=*/{
          getSymNameAttrName(), getInputsAttrName(), getOutputsAttrName()});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
      /*printBlockTerminators=*/true, /*printEmptyBlock=*/false);
}

// Implementation is adapted from function_interface_impl::parseFunctionOp().
ParseResult lazycst::ExprOp::parse(
    OpAsmParser &parser, OperationState &result) {
  StringAttr name;
  SmallVector<OpAsmParser::Argument> args;
  ArrayAttr inputs;
  ArrayAttr outputs;
  auto *body = result.addRegion();

  OperationName exprOpName(
      lazycst::ExprOp::getOperationName(), parser.getContext());
  StringRef symNameAttrName = lazycst::ExprOp::getSymNameAttrName(exprOpName);
  StringRef inputsAttrName = lazycst::ExprOp::getInputsAttrName(exprOpName);
  StringRef outputsAttrName = lazycst::ExprOp::getOutputsAttrName(exprOpName);

  if (parser.parseSymbolName(name, symNameAttrName, result.attributes) ||
      parser.parseArgumentList(
          args, OpAsmParser::Delimiter::Paren, /*allowType=*/true) ||
      parser.parseAttribute<ArrayAttr>(
          inputs, inputsAttrName, result.attributes) ||
      parser.parseArrow() ||
      parser.parseAttribute<ArrayAttr>(
          outputs, outputsAttrName, result.attributes) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseRegion(*body, args))
    return failure();

  // TODO: record in LazyCstExprManager

  return success();
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
