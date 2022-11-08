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

llvm::SmallVector<int64_t, 4> unpadStrides(llvm::ArrayRef<int64_t> strides);

llvm::SmallVector<int64_t, 4> padStrides(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides);

llvm::SmallVector<int64_t, 4> paddedStridesOfShape(
    llvm::ArrayRef<int64_t> shape);

llvm::Optional<llvm::SmallVector<int64_t, 4>> transposeStrides(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
    llvm::ArrayRef<uint64_t> perm);

llvm::Optional<llvm::SmallVector<int64_t, 4>> reshapeStrides(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
    llvm::ArrayRef<int64_t> reshapedShape);

llvm::Optional<llvm::SmallVector<int64_t, 4>> expandStrides(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
    llvm::ArrayRef<int64_t> expandedShape);

// Requires srcStrides and dstStrides are padded.
void restrideArray(unsigned elementBytewidth, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<char> src, llvm::ArrayRef<int64_t> srcStrides,
    llvm::MutableArrayRef<char> dst, llvm::ArrayRef<int64_t> dstStrides);

// Computes dstStrides from shape, and pads them and srcStrides.
void restrideArray(unsigned elementBytewidth, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<char> src, llvm::ArrayRef<int64_t> srcStrides,
    llvm::MutableArrayRef<char> dst);

// Requires srcStrides and dstStrides are padded.
template <typename T>
void restrideArray(llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<T> src,
    llvm::ArrayRef<int64_t> srcStrides, llvm::MutableArrayRef<T> dst,
    llvm::ArrayRef<int64_t> dstStrides) {
  return restrideArray(sizeof(T), shape, castArrayRef<char>(src), srcStrides,
      castMutableArrayRef<char>(dst), dstStrides);
}

// The following functions are more about shapes than strides but they live
// here for now:

// NOTE: this function is expensive, try to avoid calling it
llvm::SmallVector<int64_t, 4> unflattenIndex(
    llvm::ArrayRef<int64_t> shape, int64_t flatIndex);

llvm::SmallVector<int64_t, 4> transposeDims(
    llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<uint64_t> perm);

llvm::SmallVector<int64_t, 4> untransposeDims(
    llvm::ArrayRef<int64_t> dims, llvm::ArrayRef<uint64_t> perm);

} // namespace onnx_mlir