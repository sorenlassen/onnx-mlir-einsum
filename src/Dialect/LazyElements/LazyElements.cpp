/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyElements/LazyElements.hpp"

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

llvm::ArrayRef<char> FileDataElementsAttr::getRawBytes() const {
  llvm_unreachable("TODO: implement this");
}

WideNum FileDataElementsAttr::atFlatIndex(size_t flatIndex) const {
  llvm_unreachable("TODO: implement this");
}

} // namespace lazy_elements