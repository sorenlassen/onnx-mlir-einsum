/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyElements/LazyElements.hpp"

#include "src/Support/Arrays.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

void lazy_elements::LazyElementsDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialect/LazyElements/LazyElementsAttributes.cpp.inc"
      >();
}

#include "src/Dialect/LazyElements/LazyElementsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/LazyElements/LazyElementsAttributes.cpp.inc"

namespace lazy_elements {

// template <class C>
// mlir::StringAttr BufferElementsAttr<C>::getPath() const {
//   // using T = typename C::ContiguousIterableTypesT;
//   return static_cast<typename C::ImplType *>(impl)->path;
// }

llvm::ArrayRef<char> FileDataElementsAttr::getRawBytesImpl() const {
  LazyElementsDialect *lazyElementsDialect =
      getContext()->getLoadedDialect<LazyElementsDialect>();
  llvm::StringRef buffer =
      lazyElementsDialect->fileDataManager.readFile(getPath());
  return onnx_mlir::asArrayRef(buffer);
}

} // namespace lazy_elements