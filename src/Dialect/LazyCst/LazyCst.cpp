/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCst.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"

#include "src/Support/Arrays.hpp"
#include "src/Support/TypeUtilities.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include <string.h>

void lazycst::LazyCstDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialect/LazyCst/LazyCstAttributes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "src/Dialect/LazyCst/LazyCstOps.cpp.inc"
      >();
}

#include "src/Dialect/LazyCst/LazyCstDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/LazyCst/LazyCstAttributes.cpp.inc"

using namespace mlir;

namespace lazycst {

ArrayRef<char> FileDataElementsAttr::getRawBytes() const {
  LazyCstDialect *lazyElementsDialect =
      getContext()->getLoadedDialect<LazyCstDialect>();
  StringRef buffer = lazyElementsDialect->fileDataManager.readFile(getPath());
  uint64_t offset = getOffset();
  uint64_t size = onnx_mlir::getSizeInBytes(getType());
  assert(offset + size <= buffer.size());
  return onnx_mlir::asArrayRef(buffer.substr(offset, size));
}

ArrayRef<char> LazyElementsAttr::getRawBytes() const {
  llvm_unreachable("TODO: implement this");
}

DenseElementsAttr toDenseElementsAttrFromRawBytes(
    ShapedType type, ArrayRef<char> bytes) {
  if (type.getElementType().isInteger(1))
    // don't use getFromRawBuffer which requires bit packing
    return DenseElementsAttr::get(type, onnx_mlir::castArrayRef<bool>(bytes));
  return DenseElementsAttr::getFromRawBuffer(type, bytes);
}

} // namespace lazycst