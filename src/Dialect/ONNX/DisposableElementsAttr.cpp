/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- DisposableElementsAttr.cpp --------------------===//
//
// DisposableElementsAttr, garbage collectible alternative to DenseElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposableElementsAttributeStorage.hpp"

#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/DisposablePool.hpp"
#include "src/Dialect/ONNX/ONNXAttributes.hpp"

using namespace onnx_mlir;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)

namespace mlir {

namespace {

template <DType DTYPE>
void identityReader(StringRef s, MutableArrayRef<WideNum> dst) {
  using X = CppType<DTYPE>;
  auto src = asArrayRef<X>(s);
  assert(src.size() == dst.size());
  std::transform(src.begin(), src.end(), dst.begin(),
      [](X x) { return WideNum::from<X>(DTYPE, x); });
}

DisposableElementsAttr::Reader getIdentityReader(DType dtype) {
  return dispatchByDType(
      dtype, [](auto staticDType) { return identityReader<staticDType>; });
}

} // namespace

/*static*/
DisposableElementsAttr DisposableElementsAttr::get(ShapedType type,
    Optional<Strides> strides, const Buffer &buffer, Reader reader) {
  DType dtype = dtypeOfMlirType(type.getElementType());
  SmallVector<int64_t, 4> actualStrides;
  if (strides.has_value())
    actualStrides.assign(strides->begin(), strides->end());
  else
    actualStrides = getDefaultStrides(type.getShape());
  bool isContiguous = type.getNumElements() == 1 || !actualStrides.empty();
  Properties properties = {.dtype = dtype,
      .bufferDType = dtype,
      .isContiguous = isContiguous,
      .isTransformed = reader != nullptr};
  return create(type, actualStrides, properties, buffer, std::move(reader));
}

/*static*/
DisposableElementsAttr DisposableElementsAttr::get(ShapedType type,
    Strides strides, Properties properties, const Buffer &buffer,
    Reader reader) {
  assert((strides.empty() || strides.front() != 0) &&
         "non-padded strides shouldn't have leading zeros");
  unsigned bufBytewidth = bytewidthOfDType(properties.bufferDType);
  assert(buffer->getBufferSize() % bufBytewidth == 0);
  int64_t numBufferElements = buffer->getBufferSize() / bufBytewidth;
  auto shape = type.getShape();
  // We don't require strides.empty() == (numBufferElements == 1)
  // because strides can be empty when numBufferElements == 0, i.e.
  // when type.getNumElements() == 0.
  assert(!strides.empty() || numBufferElements == 1);
  assert(numBufferElements == getStridesNumElements(shape, strides));
  // TODO: figure out if isContiguous should always be exactly the same as
  //       areStridesContiguous(shape, strides))
  assert(!properties.isContiguous || areStridesContiguous(shape, strides));
  assert(reader || !properties.isTransformed);
  assert(properties.isTransformed || wideDTypeOfDType(properties.bufferDType) ==
                                         wideDTypeOfDType(properties.dtype));
  // TODO: add more checks
  return create(
      type, strides, properties, std::move(buffer), std::move(reader));
}

/*static*/
DisposableElementsAttr DisposableElementsAttr::create(ShapedType type,
    Strides strides, Properties properties, const Buffer &buffer,
    Reader reader) {
  DisposableElementsAttr a =
      Base::get(type.getContext(), type, strides, properties);
  Storage &s = *a.getImpl();
  s.buffer = buffer;
  if (reader) {
    s.reader = std::move(reader);
  } else {
    assert(wideDTypeOfDType(properties.bufferDType) ==
               wideDTypeOfDType(properties.dtype) &&
           "buffer wide type mismatch requires transforming reader");
    s.reader = getIdentityReader(properties.bufferDType);
  }
  return a;
}

namespace {
DisposableElementsAttr::Reader composeReadTransform(
    const DisposableElementsAttr::Reader &reader,
    DisposableElementsAttr::Transformer transformer) {
  return [read = reader, transform = std::move(transformer)](
             StringRef s, MutableArrayRef<WideNum> dst) {
    read(s, dst);
    transform(dst);
  };
}

template <DType SRC_TAG, DType DST_TAG>
void wideCaster(MutableArrayRef<WideNum> nums) {
  using SrcT = CppType<SRC_TAG>;
  using DstT = CppType<DST_TAG>;
  for (WideNum &n : nums) {
    // n = static_cast<DstT>(reinterpret_cast<SrcT>(n)) :
    SrcT src = n.to<SrcT>(SRC_TAG);        // unpack from WideNum to SrcT
    DstT dst = static_cast<DstT>(src);     // cast
    n = WideNum::from<DstT>(DST_TAG, dst); // pack to WideNum from DstT
  }
}

DisposableElementsAttr::Transformer wideCaster(DType src, DType dst) {
  constexpr DType DBL = DType::DOUBLE, I64 = DType::INT64, U64 = DType::UINT64;
  // clang-format off
  if (src == DBL && dst == I64) return wideCaster<DBL, I64>;
  if (src == DBL && dst == U64) return wideCaster<DBL, U64>;
  if (src == I64 && dst == DBL) return wideCaster<I64, DBL>;
  if (src == I64 && dst == U64) return wideCaster<I64, U64>;
  if (src == U64 && dst == DBL) return wideCaster<U64, DBL>;
  if (src == U64 && dst == I64) return wideCaster<U64, I64>;
  // clang-format on
  llvm_unreachable("wideCaster must be called with 2 different wide types");
}
} // namespace

DisposableElementsAttr DisposableElementsAttr::transform(DisposablePool &pool,
    Type transformedElementType, Transformer transformer) const {
  ShapedType transformedType = getType().clone(transformedElementType);
  Properties transformedProperties = getProperties();
  transformedProperties.dtype = dtypeOfMlirType(transformedElementType);
  transformedProperties.isTransformed = true;
  return pool.createElementsAttr(transformedType, getStrides(),
      transformedProperties, getBuffer(),
      composeReadTransform(getReader(), std::move(transformer)));
}

DisposableElementsAttr DisposableElementsAttr::castElementType(
    DisposablePool &pool, Type newElementType) const {
  if (newElementType == getElementType())
    return *this;

  ShapedType newType = getType().clone(newElementType);
  Properties newProperties = getProperties();
  newProperties.dtype = dtypeOfMlirType(newElementType);
  DType oldWideType = wideDTypeOfDType(getDType());
  DType newWideType = wideDTypeOfDType(newProperties.dtype);

  if (oldWideType == newWideType) {
    return pool.createElementsAttr(
        newType, getStrides(), newProperties, getBuffer(), getReader());
  }

  newProperties.isTransformed = true;
  Transformer transformer = wideCaster(oldWideType, newWideType);
  return pool.createElementsAttr(newType, getStrides(), newProperties,
      getBuffer(), composeReadTransform(getReader(), std::move(transformer)));
  llvm_unreachable("TODO: implement DisposableElementsAttr::castElementType");
}

DisposableElementsAttr DisposableElementsAttr::transpose(
    DisposablePool &pool, ArrayRef<uint64_t> perm) const {
  // TODO: Check if perm is identity and then just return *this.

  ShapedType type = getType();
  auto shape = type.getShape();
  auto transposedShape = transposeDims(shape, perm);
  ShapedType transposedType = type.clone(transposedShape);
  Properties transposedProperties = getProperties();
  auto strides = getStrides();

  if (auto transposedStrides = transposeStrides(shape, strides, perm)) {
    transposedProperties.isContiguous =
        (transposedStrides == getDefaultStrides(transposedShape));
    return pool.createElementsAttr(transposedType,
        makeArrayRef(*transposedStrides), transposedProperties, getBuffer(),
        getReader());
  }

  // TODO: Consider transposing without transforming (just carry over the
  //       reader) when getNumBufferElements() == getNumElements(), i.e.
  //       strides have no zeros.

  SmallVector<WideNum> readout;
  readout.resize_for_overwrite(getNumBufferElements());
  getReader()(getBufferString(), readout);
  ArrayRef<char> src(castArrayRef<char>(makeArrayRef(readout)));
  std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(
          getNumElements() * sizeof(WideNum));
  auto reverseStrides =
      untransposeDims(paddedStridesOfShape(transposedShape), perm);
  restrideArray(sizeof(WideNum), shape, src, strides, writeBuffer->getBuffer(),
      reverseStrides);
  transposedProperties.isContiguous = true;
  DType dtype = getDType();
  transposedProperties.bufferDType = wideDTypeOfDType(dtype);
  Buffer transposedBuffer = std::move(writeBuffer);
  return pool.createElementsAttr(transposedType,
      getDefaultStrides(transposedShape), transposedProperties,
      transposedBuffer, getIdentityReader(dtype));
}

DisposableElementsAttr DisposableElementsAttr::reshape(
    DisposablePool &pool, ArrayRef<int64_t> reshapedShape) const {
  // TODO: if getStrides() don't conflict with reshapedShape clone *this
  //       with strides that incorporate reshape, otherwise create a new
  //       MemoryBuffer and restrideArray buffer into it and, if needed,
  //       do a post processing phase to reorder elements
  llvm_unreachable("TODO: implement DisposableElementsAttr::reshape");
}

DisposableElementsAttr DisposableElementsAttr::expand(
    DisposablePool &pool, ArrayRef<int64_t> expandedShape) const {
  // TODO: if getStrides() don't conflict with expandedShape clone *this
  //       with strides that incorporate expandedShape, otherwise create a new
  //       MemoryBuffer and restrideArray / reorder buffer into it
  llvm_unreachable("TODO: implement DisposableElementsAttr::expand");
}

auto DisposableElementsAttr::getStrides() const -> Strides {
  return getImpl()->strides;
}

auto DisposableElementsAttr::getProperties() const -> const Properties & {
  return getImpl()->properties;
}

auto DisposableElementsAttr::getBuffer() const -> const Buffer & {
  assert(!isDisposed());
  return getImpl()->buffer;
}

auto DisposableElementsAttr::getReader() const -> const Reader & {
  assert(!isDisposed());
  return getImpl()->reader;
}

bool DisposableElementsAttr::isDisposed() const {
  //  TODO: Decide if a splat value can be represented with a constant
  //        reader with no buffer; in that case isDisposed should
  //        only return true if both buffer and reader are null.
  return !getImpl()->buffer;
}

bool DisposableElementsAttr::isContiguous() const {
  return getProperties().isContiguous;
}

unsigned DisposableElementsAttr::getBufferElementBytewidth() const {
  return bytewidthOfDType(getProperties().bufferDType);
}

int64_t DisposableElementsAttr::getNumBufferElements() const {
  return getBuffer()->getBufferSize() / getBufferElementBytewidth();
}

StringRef DisposableElementsAttr::getBufferString() const {
  return getBuffer()->getBuffer();
}

ArrayRef<char> DisposableElementsAttr::getBufferBytes() const {
  return asArrayRef(getBufferString());
}

bool DisposableElementsAttr::isSplat() const {
  return getStrides().empty() && getBuffer()->getBufferSize() != 0;
}

DType DisposableElementsAttr::getDType() const { return getProperties().dtype; }

ShapedType DisposableElementsAttr::getType() const { return getImpl()->type; }

WideNum DisposableElementsAttr::readBufferPos(size_t pos) const {
  StringRef s = getBufferString();
  unsigned bufBytewidth = getBufferElementBytewidth();
  StringRef bytes = s.substr(pos * bufBytewidth, bufBytewidth);
  WideNum n;
  getReader()(bytes, llvm::makeMutableArrayRef(n));
  return n;
}

WideNum DisposableElementsAttr::readFlatIndex(size_t flatIndex) const {
  return readBufferPos(flatIndexToBufferPos(flatIndex));
}

size_t DisposableElementsAttr::flatIndexToBufferPos(size_t flatIndex) const {
  if (isContiguous())
    return flatIndex;
  if (isSplat())
    return 0;
  SmallVector<int64_t, 4> indices;
  return getStridesPosition(
      unflattenIndex(getShape(), flatIndex), getStrides());
}

void DisposableElementsAttr::readElements(MutableArrayRef<WideNum> dst) const {
  if (isContiguous()) {
    getReader()(getBuffer()->getBuffer(), dst);
    return;
  }
  SmallVector<WideNum, 1> wideBufferData;
  wideBufferData.resize_for_overwrite(getNumBufferElements());
  getReader()(getBufferString(), wideBufferData);
  ArrayRef<WideNum> src(wideBufferData);
  restrideArray(sizeof(WideNum), getShape(), castArrayRef<char>(src),
      getStrides(), castMutableArrayRef<char>(dst));
}

ArrayBuffer<WideNum> DisposableElementsAttr::getWideNums() const {
  const Properties &properties = getProperties();
  if (!properties.isTransformed && properties.isContiguous &&
      getBufferElementBytewidth() == sizeof(WideNum)) {
    return asArrayRef<WideNum>(getBufferString());
  }
  ArrayBuffer<WideNum>::Vector vec;
  vec.resize_for_overwrite(getNumElements());
  readElements(vec);
  return std::move(vec);
}

ArrayBuffer<char> DisposableElementsAttr::getRawBytes() const {
  const Properties &properties = getProperties();
  bool requiresNoElementwiseTransformOrCast =
      !properties.isTransformed && properties.dtype == properties.bufferDType;
  if (requiresNoElementwiseTransformOrCast && properties.isContiguous)
    return getBufferBytes();
  unsigned attrBytewidth = bytewidthOfDType(properties.dtype);
  ArrayBuffer<char>::Vector vec;
  vec.resize_for_overwrite(getNumElements() * attrBytewidth);
  MutableArrayRef<char> bytes(vec);
  if (requiresNoElementwiseTransformOrCast) {
    auto src = getBufferBytes();
    restrideArray(attrBytewidth, getShape(), src, getStrides(), bytes);
  } else if (attrBytewidth == sizeof(WideNum)) {
    readElements(castMutableArrayRef<WideNum>(bytes));
  } else {
    SmallVector<WideNum, 1> wideData;
    wideData.resize_for_overwrite(getNumElements());
    readElements(wideData);
    narrowArray(getElementType(), wideData, bytes);
  }
  return std::move(vec);
}

void DisposableElementsAttr::printWithoutType(raw_ostream &os) const {
  printIntOrFPElementsAttrAsDenseWithoutType(*this, os);
}

} // namespace mlir