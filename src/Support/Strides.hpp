/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- Strides.hpp -----------------------------===//
//
// Strides helper functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace onnx_mlir {

int64_t getStridesNumElements(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides);

bool areStridesContiguous(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides);

size_t getStridesPosition(
    llvm::ArrayRef<int64_t> indices, llvm::ArrayRef<int64_t> strides);

llvm::SmallVector<int64_t, 4> getDefaultStrides(llvm::ArrayRef<int64_t> shape);

// NOTE: this function is expensive, try to avoid calling it
void unflattenIndex(llvm::ArrayRef<int64_t> shape, int64_t flatIndex,
    llvm::SmallVectorImpl<int64_t> &indices);

llvm::SmallVector<int64_t, 4> padStrides(llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides);

llvm::SmallVector<int64_t, 4> paddedStridesOfShape(llvm::ArrayRef<int64_t> shape);

} // namespace onnx_mlir