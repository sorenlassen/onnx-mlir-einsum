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

DisposableElementsAttr::Reader getSplatReader(DType dtype, StringRef rawBytes) {
  unsigned bytewidth = bytewidthOfDType(dtype);
  ArrayRef<char> memory = asArrayRef(rawBytes.take_front(bytewidth));
  WideNum splatValue = WideNum::load(dtype, memory);
  return [=](StringRef s, MutableArrayRef<WideNum> dst) {
    assert(s.size() == bytewidth);
    assert(dst.size() == 1);
    *dst.begin() = splatValue;
  };
}

} // namespace

/*static*/
DisposableElementsAttr DisposableElementsAttr::get(
    ShapedType type, const Buffer &buffer, Reader reader) {
  ArrayRef<char> rawBuffer = asArrayRef(buffer->getBuffer());
  bool isBufferSplat = false;
  if (!DenseElementsAttr::isValidRawBuffer(type, rawBuffer, isBufferSplat))
    llvm_unreachable("invalid buffer passed to DisposableElementsAttr::get");
  return get(type, isBufferSplat, buffer, std::move(reader));
}

/*static*/
DisposableElementsAttr DisposableElementsAttr::get(
    ShapedType type, bool isBufferSplat, const Buffer &buffer, Reader reader) {
  DType dtype = dtypeOfMlirType(type.getElementType());
  SmallVector<int64_t, 4> strides;
  if (!isBufferSplat)
    strides = getDefaultStrides(type.getShape());
  bool isContiguous = type.getNumElements() == 1 || !isBufferSplat;
  Properties properties = {.dtype = dtype,
      .bufferDType = dtype,
      .isBufferSplat = isBufferSplat,
      .isContiguous = isContiguous,
      .isTransformed = reader != nullptr};
  return create(type, strides, properties, buffer, std::move(reader));
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
  assert(strides.empty() == (numBufferElements == 1));
  assert(properties.isBufferSplat == (numBufferElements == 1));
  // TODO: decide if isBufferSplat==true and numBufferElements==1
  //       are ok when getNumElements(shape)==0
  assert(numBufferElements == getStridesNumElements(shape, strides));
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
    if (properties.isBufferSplat) {
      // TODO: decide whether to remove this and use identity reader also for
      // splat
      s.reader = getSplatReader(properties.bufferDType, buffer->getBuffer());
    }
    s.reader = getIdentityReader(properties.bufferDType);
  }
  return a;
}

DisposableElementsAttr DisposableElementsAttr::transpose(
    DisposablePool &pool, ArrayRef<uint64_t> perm) const {
  // TODO: if getStrides() don't conflict with perm clone *this
  //       with strides that incorporate perm, otherwise create a new
  //       MemoryBuffer and restrideArray buffer into it
  llvm_unreachable("TODO: implement DisposableElementsAttr::transpose");
}

DisposableElementsAttr DisposableElementsAttr::transform(DisposablePool &pool,
    Type transformedElementType, Transformer transformer) const {
  ShapedType transformedType = getType().clone(transformedElementType);
  Properties transformedProperties = getProperties();
  transformedProperties.isTransformed = true;
  transformedProperties.dtype = dtypeOfMlirType(transformedElementType);
  return pool.createElementsAttr(transformedType, getStrides(),
      transformedProperties, getBuffer(),
      [read = getReader(), transform = std::move(transformer)](
          StringRef s, MutableArrayRef<WideNum> dst) {
        read(s, dst);
        transform(dst);
      });
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
  return bytewidthOfDType(getDType());
}

bool DisposableElementsAttr::isSplat() const {
  return getProperties().isBufferSplat;
}

DType DisposableElementsAttr::getDType() const { return getProperties().dtype; }

ShapedType DisposableElementsAttr::getType() const { return getImpl()->type; }

WideNum DisposableElementsAttr::readBufferPos(size_t pos) const {
  StringRef s = getBuffer()->getBuffer();
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
  unflattenIndex(getShape(), flatIndex, indices);
  return getStridesPosition(indices, getStrides());
}

void DisposableElementsAttr::readElements(MutableArrayRef<WideNum> dst) const {
  if (isContiguous()) {
    getReader()(getBuffer()->getBuffer(), dst);
    return;
  }
  SmallVector<WideNum, 1> wideBufferData;
  wideBufferData.resize_for_overwrite(getNumBufferElements());
  getReader()(getBuffer()->getBuffer(), wideBufferData);
  ArrayRef<WideNum> src(wideBufferData);
  restrideArray(sizeof(WideNum), getShape(), castArrayRef<char>(src),
      getStrides(), castMutableArrayRef<char>(dst));
}

ArrayBuffer<WideNum> DisposableElementsAttr::getWideNums() const {
  const Properties &properties = getProperties();
  if (!properties.isTransformed && properties.isContiguous &&
      getBufferElementBytewidth() == sizeof(WideNum)) {
    return asArrayRef<WideNum>(getBuffer()->getBuffer());
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
    return asArrayRef(getBuffer()->getBuffer());
  unsigned attrBytewidth = bytewidthOfDType(properties.dtype);
  ArrayBuffer<char>::Vector vec;
  vec.resize_for_overwrite(getNumElements() * attrBytewidth);
  MutableArrayRef<char> bytes(vec);
  if (requiresNoElementwiseTransformOrCast) {
    auto src = asArrayRef(getBuffer()->getBuffer());
    restrideArray(attrBytewidth, getShape(), src, getStrides(), bytes);
  } else if (attrBytewidth == sizeof(WideNum)) {
    readElements(castMutableArrayRef<WideNum>(bytes));
  } else {
    SmallVector<WideNum, 1> wideData;
    wideData.resize_for_overwrite(getNumElements());
    readElements(wideData);
    narrowArray(getElementType(), wideData, bytes);
  }
  return std::move(bytes);
}

void DisposableElementsAttr::printWithoutType(raw_ostream &os) const {
  printIntOrFPElementsAttrAsDenseWithoutType(*this, os);
}

} // namespace mlir