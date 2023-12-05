/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCstOps.hpp"
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"

#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "src/Dialect/LazyCst/LazyCstOps.cpp.inc"

using namespace mlir;

void lazycst::CstexprOp::build(OpBuilder &odsBuilder, OperationState &odsState,
    const SymbolTable &symbol_table, Block *entry_block, ArrayAttr inputs) {
  StringAttr symName =
      llvm::cast<lazycst::LazyCstDialect>(odsState.name.getDialect())
          ->nextExprName(symbol_table);

  auto symRef = FlatSymbolRefAttr::get(symName);
  SmallVector<Attribute> outputs;
  auto resTypes = entry_block->getTerminator()->getOperandTypes();
  for (auto [index, type] : llvm::enumerate(resTypes)) {
    auto lazyElms =
        lazycst::LazyElementsAttr::get(cast<ShapedType>(type), symRef, index);
    outputs.push_back(lazyElms);
  }

  odsState.getOrAddProperties<Properties>().sym_name = symName;
  odsState.getOrAddProperties<Properties>().inputs = inputs;
  odsState.getOrAddProperties<Properties>().outputs =
      odsBuilder.getArrayAttr(outputs);
  Region *body = odsState.addRegion();
  body->push_back(entry_block);

  OperationName opName = odsState.name;
  auto &cstexprEvaluator =
      llvm::cast<LazyCstDialect>(*opName.getDialect()).cstexprEvaluator;
  cstexprEvaluator.insert(symName, entry_block);
}

lazycst::CstexprOp lazycst::CstexprOp::create(SymbolTable &symbolTable,
    Location loc, Block *entryBlock, ArrayRef<Attribute> inputs) {
  auto module = cast<ModuleOp>(symbolTable.getOp());
  OpBuilder b(module.getBodyRegion());
  auto cstexpr = b.create<lazycst::CstexprOp>(
      loc, symbolTable, entryBlock, b.getArrayAttr(inputs));
  symbolTable.insert(cstexpr);
  return cstexpr;
}

// Example:
// clang-format off
/*
lazycst.cstexpr @lazycst.1(%arg0: tensor<f32>) [#lazycst.lazy_elms<@lazycst.0> : tensor<f32>] -> [#lazycst.lazy_elms<@lazycst.1> : tensor<f32>] {
  %0 = "onnx.Neg"(%arg0) : (tensor<f32>) -> tensor<f32>
  lazycst.yield %0 : tensor<f32>
}
*/
// clang-format on
//
// Implementation is adapted from function_interface_impl::printFunctionOp().
void lazycst::CstexprOp::print(OpAsmPrinter &p) {
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
ParseResult lazycst::CstexprOp::parse(
    OpAsmParser &parser, OperationState &result) {
  StringAttr name;
  SmallVector<OpAsmParser::Argument> args;
  ArrayAttr inputs;
  ArrayAttr outputs;
  auto *body = result.addRegion();

  OperationName opName = result.name;
  StringRef symNameAttrName = lazycst::CstexprOp::getSymNameAttrName(opName);
  StringRef inputsAttrName = lazycst::CstexprOp::getInputsAttrName(opName);
  StringRef outputsAttrName = lazycst::CstexprOp::getOutputsAttrName(opName);

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

  auto &cstexprEvaluator =
      llvm::cast<LazyCstDialect>(*opName.getDialect()).cstexprEvaluator;
  cstexprEvaluator.insert(name, &body->front());

  return success();
}

// Implementation is adapted from func::ReturnOp.
LogicalResult lazycst::YieldOp::verify() {
  auto cstexpr = cast<lazycst::CstexprOp>((*this)->getParentOp());
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
