/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/Dialect/LazyCst/RawBytesIterator.hpp"
#include "src/Interface/DenseLikeElementsAttrInterface.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

#include "llvm/ADT/ArrayRef.h"

namespace lazycst {

using DenseLikeElementsAttrInterface = mlir::DenseLikeElementsAttrInterface;

// Makes deep copy.
mlir::DenseElementsAttr toDenseElementsAttrFromRawBytes(
    mlir::ShapedType, llvm::ArrayRef<char> bytes);

mlir::DenseElementsAttr toDenseElementsAttrFromElementsAttr(
    mlir::ElementsAttr elementsAttr);

} // namespace lazycst

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp.inc"
