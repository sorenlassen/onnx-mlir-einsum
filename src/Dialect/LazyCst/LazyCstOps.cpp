/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCstOps.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#define GET_OP_CLASSES
#include "src/Dialect/LazyCst/LazyCstOps.cpp.inc"

using namespace mlir;

void lazycst::ExprOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, StringAttr sym_name) {
  auto noFuncType = odsBuilder.getFunctionType({}, {});
  auto noArray = odsBuilder.getArrayAttr({});
  build(odsBuilder, odsState, sym_name, noFuncType, noArray, noArray, nullptr,
      nullptr);
}

// Implementation is copied from func::FuncOp.
ParseResult lazycst::ExprOp::parse(
    OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
          function_interface_impl::VariadicFlag,
          std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(parser, result,
      /*allowVariadic=*/false, getFunctionTypeAttrName(result.name),
      buildFuncType, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
}

// Implementation is copied from func::FuncOp.
void lazycst::ExprOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false,
      getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName());
}

// Implementation is copied from func::ReturnOp.
LogicalResult lazycst::YieldOp::verify() {
  auto function = cast<lazycst::ExprOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}
