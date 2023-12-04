/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCstOps.hpp"
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#define GET_OP_CLASSES
#include "src/Dialect/LazyCst/LazyCstOps.cpp.inc"

using namespace mlir;

void lazycst::ExprOp::build(OpBuilder &odsBuilder, OperationState &odsState,
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

  OperationName exprOpName = odsState.name;
  auto &lazyCstExprManager =
      llvm::cast<LazyCstDialect>(*exprOpName.getDialect()).lazyCstExprManager;
  lazyCstExprManager.insert(symName, entry_block);
}

lazycst::ExprOp lazycst::ExprOp::create(SymbolTable &symbolTable, Location loc,
    Block *entryBlock, ArrayRef<Attribute> inputs) {
  auto module = cast<ModuleOp>(symbolTable.getOp());
  OpBuilder b(module.getBodyRegion());
  auto cstexpr = b.create<lazycst::ExprOp>(
      loc, symbolTable, entryBlock, b.getArrayAttr(inputs));
  symbolTable.insert(cstexpr);
  return cstexpr;
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

  OperationName exprOpName = result.name;
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

  auto &lazyCstExprManager =
      llvm::cast<LazyCstDialect>(*exprOpName.getDialect()).lazyCstExprManager;
  lazyCstExprManager.insert(name, &body->front());

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
