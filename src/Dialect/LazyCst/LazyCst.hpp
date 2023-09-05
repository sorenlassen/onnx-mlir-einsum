/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/Dialect/LazyCst/ConstantFolder.hpp"
#include "src/Dialect/LazyCst/FileDataManager.hpp"
#include "src/Dialect/LazyCst/RawBytesIterator.hpp"

#include "src/Dialect/LazyCst/WideNum.hpp"
#include "src/Interface/DenseLikeElementsAttrInterface.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dialect.h"

#include "llvm/ADT/ArrayRef.h"

#include <tuple>
#include <type_traits>

namespace lazycst {

using DenseLikeElementsAttrInterface = mlir::DenseLikeElementsAttrInterface;

// Makes deep copy.
mlir::DenseElementsAttr toDenseElementsAttrFromRawBytes(
    mlir::ShapedType, llvm::ArrayRef<char> bytes);

class LazyFunctionManager {
public:
  LazyFunctionManager() : counter(0) {}
  mlir::StringAttr nextName(mlir::ModuleOp module);

private:
  std::atomic<unsigned> counter;
};

} // namespace lazycst

#include "src/Dialect/LazyCst/LazyCstDialect.hpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp.inc"
