/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"
#include "src/Dialect/LazyCst/LazyCstDialect.hpp"

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
  LazyCstDialect *lazyCstDialect =
      getContext()->getLoadedDialect<LazyCstDialect>();
  StringRef buffer = lazyCstDialect->fileDataManager.readFile(getPath());
  uint64_t offset = getOffset();
  uint64_t size = onnx_mlir::getSizeInBytes(getType());
  assert(offset + size <= buffer.size());
  return onnx_mlir::asArrayRef(buffer.substr(offset, size));
}

ElementsAttr LazyElementsAttr::getElementsAttr() const {
  auto &cstexprEvaluator =
      llvm::cast<LazyCstDialect>(getDialect()).cstexprEvaluator;
  Attribute attr = cstexprEvaluator.evaluate(getCallee().getAttr(), getIndex());
  return llvm::cast<ElementsAttr>(attr);
}

DenseElementsAttr toDenseElementsAttrFromRawBytes(
    ShapedType type, ArrayRef<char> bytes) {
  if (type.getElementType().isInteger(1)) {
    // don't use getFromRawBuffer which requires bit packing
    return DenseElementsAttr::get(type, onnx_mlir::castArrayRef<bool>(bytes));
  }
  return DenseElementsAttr::getFromRawBuffer(type, bytes);
}

DenseElementsAttr toDenseElementsAttrFromElementsAttr(
    ElementsAttr elementsAttr) {
  if (auto dense = dyn_cast<DenseElementsAttr>(elementsAttr))
    return dense;
  if (auto denseLike = dyn_cast<DenseLikeElementsAttrInterface>(elementsAttr))
    return denseLike.toDenseElementsAttr();
  // TODO: try to make this implementation more time and space efficient
  //       if it's ever used
  SmallVector<Attribute> elements(elementsAttr.getValues<Attribute>());
  return DenseElementsAttr::get(elementsAttr.getShapedType(), elements);
}

// See explanation in LazyCstDialect::initialize() in LaztCstDialect.cpp.
void LazyCstDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialect/LazyCst/LazyCstAttributes.cpp.inc"
      >();
}

} // namespace lazycst