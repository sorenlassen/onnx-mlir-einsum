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
  llvm_unreachable("TODO: implement this");
}

}
