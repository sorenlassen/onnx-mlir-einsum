/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"
#include "src/Dialect/LazyCst/LazyCst.hpp"

#include "src/Support/Arrays.hpp"
#include "src/Support/TypeUtilities.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

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

// See explanation in LazyCstDialect::initialize() in LaztCst.cpp.
void LazyCstDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialect/LazyCst/LazyCstAttributes.cpp.inc"
      >();
}

} // namespace lazycst