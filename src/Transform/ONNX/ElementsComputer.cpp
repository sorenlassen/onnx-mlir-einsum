/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ ElementsComputer.cpp ------------------------===//
//
// ElementsAttr computations.
//
//===----------------------------------------------------------------------===//

#include "src/Transform/ONNX/ElementsComputer.hpp"

#include "src/Dialect/ONNX/AttributesHelper.hpp"

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

using DimsVector = SmallVector<int64_t, 4>;
using Shape = ArrayRef<int64_t>;
using Strides = ArrayRef<int64_t>;

// TODO: reuse with DisposableElementsAttr
inline DimsVector getDefaultStrides(Shape shape) {
  DimsVector strides;
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

DimsVector padStrides(Shape shape, Strides strides) {
  int64_t skip = shape.size() - strides.size();
  assert(skip >= 0);
  DimsVector padded(skip, 0);
  padded.append(strides.begin(), strides.end());
  return padded;
}

DimsVector paddedStridesOfShape(Shape shape) {
  DimsVector strides = getDefaultStrides(shape);
  return padStrides(shape, strides);
}

DimsVector transposeDims(ArrayRef<int64_t> dims, ArrayRef<uint64_t> perm) {
  assert(dims.size() == perm.size());
  DimsVector permutedDims;
  permutedDims.reserve(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    permutedDims.push_back(dims[perm[i]]);
  return permutedDims;
}

DimsVector untransposeDims(ArrayRef<int64_t> dims, ArrayRef<uint64_t> perm) {
  assert(dims.size() == perm.size());
  DimsVector unpermutedDims;
  unpermutedDims.resize_for_overwrite(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    unpermutedDims[perm[i]] = dims[i];
  return unpermutedDims;
}

template <typename T>
void restrideArray(Shape shape, Strides srcStrides, ArrayRef<T> src,
    Strides dstStrides, MutableArrayRef<T> dst) {
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

void restrideArray(unsigned bytewidth, Shape shape, Strides srcStrides,
    ArrayRef<char> src, Strides dstStrides, MutableArrayRef<char> dst) {
  dispatchByBytewidth(bytewidth, [&](auto staticBytewidth) {
    using T = BitcastType<staticBytewidth>;
    restrideArray<T>(shape, srcStrides, castArrayRef<T>(src), dstStrides,
        castMutableArrayRef<T>(dst));
  });
}

void transformArray(Type srcElementType, ArrayRef<IntOrFP> src,
    Type dstElementType, MutableArrayRef<char> dst,
    const Transformation &transformation) {
  dispatchByMlirType(dstElementType, [&](auto dtype) {
    using D = CppType<dtype>;
    auto dbegin = castMutableArrayRef<D>(dst).begin();
    std::transform(src.begin(), src.end(), dbegin, [&](IntOrFP n) -> D {
      return transformation(n).template to<D>(toDType<D>);
    });
  });
}

} // namespace

ElementsAttr transposeElements(ElementsAttr elements, ArrayRef<uint64_t> perm) {
  // TODO: if elements is disposable try to reuse contents with new strides
  ShapedType type = elements.getType();
  Type elementType = type.getElementType();
  Shape shape = type.getShape();
  DimsVector transposedShape = transposeDims(shape, perm);
  ShapedType transposedType =
      RankedTensorType::get(transposedShape, elementType);
  RawBuffer src = getElementsRawBytes(elements);
  if (elements.isSplat()) {
    return makeElementsAttrFromRawBytes(
        transposedType, src.get(), /*mustCopy=*/true);
  }
  DimsVector shapeStrides = paddedStridesOfShape(shape);
  // TODO: change src and elementsStrides to underlying data if
  //       elements is a DisposableElementsAttr
  Strides elementsStrides = shapeStrides;
  DimsVector transposedStrides =
      untransposeDims(paddedStridesOfShape(transposedShape), perm);
  return makeElementsAttrWithRawBytesFiller(
      transposedType, [&](MutableArrayRef<char> dst) {
        restrideArray(getIntOrFloatByteWidth(elementType), shape,
            elementsStrides, src.get(), transposedStrides, dst);
      });
}

ElementsAttr transformElements(ElementsAttr elements,
    Type transformedElementType, const Transformation &transformation) {
  ShapedType transformedType = elements.getType().clone(transformedElementType);
  // TODO: if elements is disposable just compose transformation with its
  // transform
  ArrayBuffer<IntOrFP> src = getElementsIntOrFPs(elements);
  return makeElementsAttrWithRawBytesFiller(
      transformedType, [&](MutableArrayRef<char> dst) {
        transformArray(elements.getElementType(), src.get(),
            transformedElementType, dst, transformation);
      });
}

} // namespace onnx_mlir