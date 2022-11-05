/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- Strides.cpp -----------------------------===//
//
// Strides helper functions.
//
//===----------------------------------------------------------------------===//

#include "src/Support/Strides.hpp"

#include "src/Support/Arrays.hpp"

#include "mlir/IR/BuiltinTypeInterfaces.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

// clang-format off
template <unsigned Bytewidth>
using BitcastType =
    std::conditional_t<Bytewidth == 1, std::uint8_t,
    std::conditional_t<Bytewidth == 2, std::uint16_t,
    std::conditional_t<Bytewidth == 4, std::uint32_t,
    std::conditional_t<Bytewidth == 8, std::uint64_t,
    void>>>>;
// clang-format on

template <unsigned Bytewidth>
struct BytewidthToken {
  constexpr BytewidthToken() {}
  constexpr operator unsigned() const { return Bytewidth; }
};

template <typename Action, typename... Args>
auto dispatchByBytewidth(unsigned bytewidth, Action &&act, Args &&...args) {
  // clang-format off
  switch (bytewidth) {
  case 1: return act(BytewidthToken<1>{}, std::forward<Args>(args)...);
  case 2: return act(BytewidthToken<2>{}, std::forward<Args>(args)...);
  case 4: return act(BytewidthToken<4>{}, std::forward<Args>(args)...);
  case 8: return act(BytewidthToken<8>{}, std::forward<Args>(args)...);
  default: llvm_unreachable("unsupported bytewidth");
  }
  // clang-format on
}

} // namespace

int64_t getStridesNumElements(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
  if (ShapedType::getNumElements(shape) == 0)
    return 0;
  assert(shape.size() >= strides.size());
  int64_t last = 0;
  for (int a = shape.size() - 1, s = strides.size() - 1; s >= 0; --a, --s)
    last += (shape[a] - 1) * strides[s];
  return last + 1;
}

bool areStridesContiguous(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
  assert(shape.size() >= strides.size());
  if (shape.size() != strides.size())
    return false;
  int64_t x = 1;
  for (int s = strides.size() - 1; s >= 0; --s) {
    if (strides[s] != x)
      return false;
    x *= shape[s];
  }
  return true;
}

size_t getStridesPosition(
    ArrayRef<int64_t> indices, ArrayRef<int64_t> strides) {
  assert(indices.size() >= strides.size());
  size_t pos = 0;
  for (int a = indices.size() - 1, s = strides.size() - 1; s >= 0; --a, --s)
    pos += indices[a] * strides[s];
  return pos;
}

SmallVector<int64_t, 4> getDefaultStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 4> strides;
  int64_t rank = shape.size();
  if (rank == 0)
    return strides;
  int64_t skip = 0;
  while (shape[skip] == 1) {
    ++skip;
    if (skip == rank)
      return strides;
  }
  strides.resize_for_overwrite(rank - skip);
  int64_t mult = 1;
  for (int64_t axis = rank - 1; axis >= skip; --axis) {
    int64_t dimSize = shape[axis];
    strides[axis - skip] = dimSize == 1 ? 0 : mult;
    mult *= dimSize;
  }
  return strides;
}

void unflattenIndex(ArrayRef<int64_t> shape, int64_t flatIndex,
    SmallVectorImpl<int64_t> &indices) {
  int64_t rank = shape.size();
  indices.resize_for_overwrite(rank);
  if (rank == 0)
    return;
  for (int64_t axis = rank - 1; axis >= 1; --axis) {
    int64_t dimSize = shape[axis];
    assert(dimSize > 0 && "cannot unflatten shape with zeros");
    int64_t rem = flatIndex % dimSize;
    flatIndex /= dimSize;
    indices[axis] = rem;
  }
  assert(flatIndex < shape[0]);
  indices[0] = flatIndex;
}

SmallVector<int64_t, 4> padStrides(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
  int64_t skip = shape.size() - strides.size();
  assert(skip >= 0);
  SmallVector<int64_t, 4> padded(skip, 0);
  padded.append(strides.begin(), strides.end());
  return padded;
}

SmallVector<int64_t, 4> paddedStridesOfShape(ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 4> strides = getDefaultStrides(shape);
  return padStrides(shape, strides);
}

namespace {
template <typename T>
void restrideArray(ArrayRef<int64_t> shape, ArrayRef<int64_t> srcStrides,
    ArrayRef<T> src, ArrayRef<int64_t> dstStrides, MutableArrayRef<T> dst) {
  assert(srcStrides.size() == shape.size() && "src strides must be padded");
  assert(dstStrides.size() == shape.size() && "dst strides must be padded");
  int64_t rank = shape.size();
  auto traverse = [=](int64_t axis, size_t srcPos, size_t dstPos,
                      const auto &recurse) -> void {
    if (axis == rank) {
      dst[dstPos] = src[srcPos];
    } else {
      size_t srcStride = srcStrides[axis];
      size_t dstStride = dstStrides[axis];
      size_t dimSize = shape[axis];
      for (size_t i = 0; i < dimSize; ++i) {
        recurse(axis + 1, srcPos, dstPos, recurse);
        srcPos += srcStride;
        dstPos += dstStride;
      }
    }
  };
  traverse(0, 0, 0, traverse);
}
} // namespace

void restrideArray(unsigned bytewidth, ArrayRef<int64_t> shape,
    ArrayRef<int64_t> srcStrides, ArrayRef<char> src,
    ArrayRef<int64_t> dstStrides, MutableArrayRef<char> dst) {
  dispatchByBytewidth(bytewidth, [&](auto staticBytewidth) {
    using T = BitcastType<staticBytewidth>;
    restrideArray<T>(shape, srcStrides, castArrayRef<T>(src), dstStrides,
        castMutableArrayRef<T>(dst));
  });
}

} // namespace onnx_mlir