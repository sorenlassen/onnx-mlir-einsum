/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCstDialect.hpp"
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"

void lazycst::LazyCstDialect::initialize() {
  // Attributes are added in this private method which is
  // implemented in LazyCstAttributes.cpp where it has
  // the necessary access to the underlying storage classes from
  // TableGen generated code in LazyCstAttributes.cpp.inc.
  // (This emulates the approach in the mlir builtin dialect.)
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "src/Dialect/LazyCst/LazyCstOps.cpp.inc"
      >();
}

#include "src/Dialect/LazyCst/LazyCstDialect.cpp.inc"
