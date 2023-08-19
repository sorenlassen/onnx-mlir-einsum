/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrHelper.hpp -----------------------===//
//
// Helper functions for accessing ElementsAttr contents.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ElementsAttr/WideNum.hpp"
#include "src/Support/Arrays.hpp"

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <algorithm>

namespace onnx_mlir {

// Preconditions: elms.isSplat() and elms.getElementType().isIntOrFloat().
WideNum getElementsSplatWideNum(mlir::ElementsAttr elms);

// Returns a pointer to the underlying data as a flat WideNum array, if
// everything aligns, otherwise makes and returns a copy.
// Precondition: elms.getElementType().isIntOrFloat().
ArrayBuffer<WideNum> getElementsWideNums(mlir::ElementsAttr elms);

// Copies out the elements in a flat WideNum array in row-major order.
// Precondition: elms.getElementType().isIntOrFloat().
void readElementsWideNums(
    mlir::ElementsAttr elms, llvm::MutableArrayRef<WideNum> dst);

} // namespace onnx_mlir
