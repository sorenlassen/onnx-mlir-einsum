/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCstOps.hpp"

#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "src/Dialect/LazyCst/LazyCstOps.cpp.inc"

using namespace mlir;

namespace lazycst {

LogicalResult LazyReturnOp::verify() {
  // Implementation is copied from func::ReturnOp.

  auto function = cast<LazyFuncOp>((*this)->getParentOp());

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

} // namespace lazycst
