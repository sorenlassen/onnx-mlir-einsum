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
#include "src/Support/Strides.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

using DimsVector = SmallVector<int64_t, 4>;
using Shape = ArrayRef<int64_t>;
using Strides = ArrayRef<int64_t>;

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