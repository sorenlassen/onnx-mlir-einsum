/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ONNXAttributes.hpp - ONNX Attributes ----------------===//
//
// This file defines attributes in the ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h" // ModuleOp
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/MemoryBuffer.h"

#include "src/Dialect/ONNX/AttributesHelper.hpp" // ArrayBuffer<char>
#include "src/Support/Arrays.hpp"
#include "src/Support/DType.hpp"
#include "src/Support/Strides.hpp"
#include "src/Support/WideNum.hpp"

#include <memory>
#include <unordered_set>

namespace mlir {

// TODO: move implementation to .cpp

namespace detail {

// Generates a unique number each time it is called. Is used as hash function
// to defeat the storage uniquer.
size_t uniqueNumber();

inline llvm::iota_range<size_t> seq(size_t numElements) {
  return llvm::seq(size_t(0), numElements);
}

template <typename T>
using MappedIndexIterator =
    llvm::mapped_iterator<llvm::iota_range<size_t>::const_iterator,
        std::function<T(size_t)>>;
template <typename T>
MappedIndexIterator<T> beginMappedIndexIterator(
    size_t numElements, const std::function<T(size_t)> &fun) {
  return llvm::map_iterator(seq(numElements).begin(), fun);
}
template <typename T>
MappedIndexIterator<T> endMappedIndexIterator(
    size_t numElements, const std::function<T(size_t)> &fun) {
  return llvm::map_iterator(seq(numElements).end(), fun);
}

} // namespace detail

struct DisposableElementsAttributeProperties {
  // Data type (BOOL, INT8, FLOAT16, etc) of the type's elements.
  onnx_mlir::DType dtype;

  // Data type of the elements in buffer before transform.
  onnx_mlir::DType bufferDType;

  // Is there a single element in the buffer?
  bool isBufferSplat;

  // Do the strides match the type's shape?
  bool isContiguous;

  // Is the reader just casting the underlying bufferDType to WideNum?
  // In this case dtypeBuffer and dtype must have the same widetype.
  bool isTransformed;
};

using DisposableElementsAttributeReader =
    std::function<void(StringRef, MutableArrayRef<onnx_mlir::WideNum>)>;

struct DisposableElementsAttributeStorage : public AttributeStorage {
  using Strides = ArrayRef<int64_t>;
  using Buffer = std::shared_ptr<llvm::MemoryBuffer>;
  using Properties = DisposableElementsAttributeProperties;
  using Reader = DisposableElementsAttributeReader;
  using KeyTy = std::tuple<ShapedType, Strides, Properties>;

  // Constructs only type and strides while the caller sets buffer and reader
  // after construction to minimize copying.
  DisposableElementsAttributeStorage(
      ShapedType type, Strides strides, Properties properties)
      : type(type), strides(strides), properties(properties) {}

  // Equality and hashKey are engineered to defeat the storage uniquer.
  // We don't want uniqueing because we can't compare readers for equality
  // and we could be in a sitation later where we have the same data or the
  // same buffer address but there is an undetectable mismatch because the
  // buffer and reader were disposed by garbage collection.
  bool operator==(const KeyTy &key) const { return false; }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return detail::uniqueNumber();
  }

  static DisposableElementsAttributeStorage *construct(
      AttributeStorageAllocator &allocator, const KeyTy &key) {
    ShapedType type = std::get<0>(key);
    Strides strides = std::get<1>(key);
    Properties properties = std::get<2>(key);
    return new (allocator.allocate<DisposableElementsAttributeStorage>())
        DisposableElementsAttributeStorage(
            type, allocator.copyInto(strides), properties);
  }

  // The tensor shape and element type that this object represents.
  // The template type T (a Cpp type bool, float, int8_t, etc) may not match
  // the element type and the caller must cast T to the element type to read
  // the underlying data.
  ShapedType type;

  // Specifies how to map positions expressed in type's shape to the flat
  // indices in buffer. strides can express that buffer is not in the default
  // row-major order (maybe as a result of a transpose) or requires broadcast
  // to fill in type's shape. A special case is when the buffer holds a single
  // splat value that broadcasts to shape's size with all-zero strides.
  Strides strides;

  Properties properties;

  // shared_ptr to an underlying MemoryBuffer which can be either heap allocated
  // or a mmap'ed file or point to the raw data of a DenseElementsAttr.
  //
  // The buffer elements' data type may not match T, namely when the transform
  // function transforms the buffer data type to another data type.
  // The buffer elements' data type is not knowable, but you can compute the
  // number of elements from strides and type's shape and then deduce the
  // data type bytewidth from the buffer's size in bytes.
  //
  // Garbage collection clears the buffer when the DisposableElementsAttr is
  // disposed.
  //
  // Multiple DisposableElementsAttr can point to the same MemoryBuffer.
  // The MemoryBuffer is destroyed (and heap allocated data freed or mmap'ed
  // file closed) when no one points to it anymore.
  Buffer buffer;

  // Reads the buffer elements to WideNums corresponding to type's
  // element type. Is set to the identity reader function if data is not
  // transformed, namely when properties.isTransformed is false.
  //
  // Garbage collection clears the reader when the DisposableElementsAttr is
  // disposed.
  Reader reader;
}; // struct DisposableElementsAttributeStorage

// DisposableElementsAttr is an alternative to DenseElementsAttr
// with the following features:
//
// 1. The memory can be heap allocated or mmap'ed from a file and will be
// released (heap allocation freed or file closed) between compiler passes
// when it is no longer reachable from the operation graph.
//
// 2. The data can be represented with higher precision than the element
// data type to avoid cumulative precision loss during constant propagation.
//
// 3. Element wise transformations are recorded lazily as a lambda and
// only materialized on read thus avoiding some memory allocations and
// copies.
//
// 4. Similarly, some tensor shape transformations can be recorded as
// 'strides' metadata without rewriting the underlying data. In particular,
// tensors can be broadcast, reshaped, and transposed in this fashion,
// subject to some constraints, like Numpy arrays and PyTorch tensors.
//
// 5. A set of helper functions makes it possible to work with
// DisposableElementsAttr and DenseElementsAttr interchangeably, and
// DisposableElementsAttr prints the same as DenseElementsAttr so
// we can switch between them without changing lit tests.
// TODO: explain caveats...
//
// NOTE: DenseResourceElementsAttr is an alternative for heap allocated memory
//       (but without garbage collection or the other features listed above).
//
// NOTE: DisposableElementsAttr doesn't support complex numbers and strings.
//       It could be extended with 'DisposableStringElemenetsAttr` and
//       `DisposableComplexElementsAttr' in the same way that
//       DenseElementsAttr has different implementations for strings and
//       numbers.
//
class DisposableElementsAttr
    : public Attribute::AttrBase<DisposableElementsAttr, Attribute,
          DisposableElementsAttributeStorage, ElementsAttr::Trait,
          TypedAttr::Trait> {
public:
  friend class DisposablePool;
  using Storage = DisposableElementsAttributeStorage;
  using Strides = typename Storage::Strides;
  using Buffer = typename Storage::Buffer;
  using Properties = DisposableElementsAttributeProperties;
  using Reader = DisposableElementsAttributeReader;
  using DType = onnx_mlir::DType;     // For convenience.
  using WideNum = onnx_mlir::WideNum; // For convenience.

private:
  using Base::Base;

  // Checks the buffer contents to detect if it's splat.
  // To bypass this check, e.g. if the buffer mmaps a file and you don't
  // want to read it here, call get(type, bufferDType, /*isBufferSplat=*/false,
  // buffer, reader). It's ok to not detect splatness.
  //
  // Assumes isTransformed if reader != nullptr.
  static DisposableElementsAttr get(
      ShapedType type, const Buffer &buffer, Reader reader = nullptr) {
    ArrayRef<char> rawBuffer = onnx_mlir::asArrayRef(buffer->getBuffer());
    bool isBufferSplat = false;
    if (!DenseElementsAttr::isValidRawBuffer(type, rawBuffer, isBufferSplat))
      llvm_unreachable("invalid buffer passed to DisposableElementsAttr::get");
    return get(type, isBufferSplat, buffer, std::move(reader));
  }

  // Assumes isTransformed if reader != nullptr.
  static DisposableElementsAttr get(ShapedType type, bool isBufferSplat,
      const Buffer &buffer, Reader reader = nullptr) {
    DType dtype = onnx_mlir::dtypeOf(type.getElementType());
    SmallVector<int64_t, 4> strides;
    if (!isBufferSplat)
      strides = onnx_mlir::getDefaultStrides(type.getShape());
    bool isContiguous = type.getNumElements() == 1 || !isBufferSplat;
    Properties properties = {.dtype = dtype,
        .bufferDType = dtype,
        .isBufferSplat = isBufferSplat,
        .isContiguous = isContiguous,
        .isTransformed = reader != nullptr};
    return create(type, strides, properties, buffer, std::move(reader));
  }

  static DisposableElementsAttr get(ShapedType type, Strides strides,
      Properties properties, const Buffer &buffer, Reader reader = nullptr) {
    assert((strides.empty() || strides.front() != 0) &&
           "non-padded strides shouldn't have leading zeros");
    unsigned bytewidth = onnx_mlir::bytewidthOfDType(properties.bufferDType);
    assert(buffer->getBufferSize() % bytewidth == 0);
    int64_t numBufferElements = buffer->getBufferSize() / bytewidth;
    auto shape = type.getShape();
    assert(strides.empty() == (numBufferElements == 1));
    assert(properties.isBufferSplat == (numBufferElements == 1));
    // TODO: decide if isBufferSplat==true and numBufferElements==1
    //       are ok when getNumElements(shape)==0
    assert(
        numBufferElements == onnx_mlir::getStridesNumElements(shape, strides));
    assert(!properties.isContiguous ||
           onnx_mlir::areStridesContiguous(shape, strides));
    assert(reader || !properties.isTransformed);
    assert(
        properties.isTransformed || wideDTypeOfDType(properties.bufferDType) ==
                                        wideDTypeOfDType(properties.dtype));
    // TODO: add more checks
    return create(
        type, strides, properties, std::move(buffer), std::move(reader));
  }

  static DisposableElementsAttr create(ShapedType type, Strides strides,
      Properties properties, const Buffer &buffer, Reader reader = nullptr) {
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
        s.reader = getSplatReader(properties.bufferDType, buffer->getBuffer());
      }
      s.reader = getIdentityReader(properties.bufferDType);
    }
    return a;
  }

  static Reader getIdentityReader(DType);

  static Reader getSplatReader(DType, StringRef rawBytes);

public:
  DisposableElementsAttr(std::nullptr_t) {}

  // Allow implicit conversion to ElementsAttr.
  operator ElementsAttr() const {
    return *this ? cast<ElementsAttr>() : nullptr;
  }

  void printWithoutType(raw_ostream &os) const;

private:
  //===----------------------------------------------------------------------===//
  // Instance properties.
  //===----------------------------------------------------------------------===//

  bool isDisposed() const {
    //  TODO: Decide if a splat value can be represented with a constant
    //        reader with no buffer; in that case isDisposed should
    //        only return true if both buffer and reader are null.
    return !getImpl()->buffer;
  }

  bool isContiguous() const { return getProperties().isContiguous; }

  int64_t getNumBufferElements() const {
    unsigned bytewidth = bytewidthOfDType(getProperties().bufferDType);
    return getBuffer()->getBufferSize() / bytewidth;
  }

public:
  ShapedType getType() const { return getImpl()->type; }
  Type getElementType() const { return getType().getElementType(); }
  ArrayRef<int64_t> getShape() const { return getType().getShape(); }
  int64_t getRank() const { return getType().getRank(); }
  int64_t getNumElements() const { return getType().getNumElements(); }
  ArrayRef<int64_t> getStrides() const { return getImpl()->strides; }
  const Properties &getProperties() const { return getImpl()->properties; }
  const Buffer &getBuffer() const {
    assert(!isDisposed());
    return getImpl()->buffer;
  }
  const Reader &getReader() const {
    assert(!isDisposed());
    return getImpl()->reader;
  }
  DType getDType() const { return getProperties().dtype; }
  // isSplat() can return false even if all elements are identical, e.g.
  // no splat check is done to verify if the reader function maps all
  // elements to the same value, or to verify if a mmap'ed file is splat.
  bool isSplat() const { return getProperties().isBufferSplat; }

private:
  //===----------------------------------------------------------------------===//
  // Iteration:
  //===----------------------------------------------------------------------===//

  // True for the types T in NonContiguousIterableTypesT.
  template <typename T>
  static constexpr bool isIterableType =
      (onnx_mlir::CppTypeTrait<T>::dtype != DType::UNDEFINED &&
          onnx_mlir::CppTypeTrait<T>::isIntOrFloat) ||
      std::is_same_v<T, llvm::APInt> || std::is_same_v<T, llvm::APFloat>;

  // Supports all the types T in NonContiguousIterableTypesT.
  template <typename X>
  static X getNumber(DType tag, WideNum n) {
    static_assert(isIterableType<X>);
    if constexpr (std::is_same_v<X, llvm::APFloat>)
      return n.toAPFloat(tag);
    else if constexpr (std::is_same_v<X, llvm::APInt>)
      return n.toAPInt(tag);
    else
      return n.to<X>(tag);
  }

public:
  // All the iterable types are listed as NonContiguous here as no type
  // is guaranteed to be represented contiguously in the underlying buffer
  // because of strides and the possibility that bufferDType != dtype.
  //
  // (We could add Attribute, IntegerAttr, FloatAttr, if getNumber adds
  // support for those.)
  using NonContiguousIterableTypesT = std::tuple<bool, int8_t, uint8_t, int16_t,
      uint16_t, int32_t, uint32_t, int64_t, uint64_t, onnx_mlir::float_16,
      onnx_mlir::bfloat_16, float, double, APInt, APFloat>;

  template <typename X>
  using iterator = detail::MappedIndexIterator<X>;

  template <typename X>
  using iterator_range = llvm::iterator_range<iterator<X>>;

  template <typename X>
  FailureOr<iterator<X>> try_value_begin_impl(OverloadToken<X>) const {
    if constexpr (isIterableType<X>) {
      DType dtype = getDType();
      DisposableElementsAttr attr = *this;
      return detail::beginMappedIndexIterator<X>(
          getNumElements(), [dtype, attr](size_t flatIndex) -> X {
            return getNumber<X>(dtype, attr.readFlatIndex(flatIndex));
          });
    } else {
      // TODO: support iteration over X == Attribute/IntegerAttr/FloatAttr
      return failure();
    }
  }

  // equivalent to getValues<X>().end(), which is probably slower?
  template <typename X>
  iterator<X> value_end() const {
    return detail::endMappedIndexIterator<X>(getNumElements(), nullptr);
  }

private:
  //===----------------------------------------------------------------------===//
  // Other access to the elements:
  //===----------------------------------------------------------------------===//

  WideNum readBufferPos(size_t pos) const {
    WideNum n;
    StringRef s = getBuffer()->getBuffer();
    // TODO: consider precomputing bytewidth in properties so
    //       we don't need to compute it all the time
    unsigned bytewidth = bytewidthOfDType(getProperties().bufferDType);
    StringRef bytes = s.substr(pos * bytewidth, bytewidth);
    getReader()(bytes, llvm::makeMutableArrayRef(n));
    return n;
  }

  // Warning: this is inefficient because it calls unflattenIndex on flatIndex.
  WideNum readFlatIndex(size_t flatIndex) const {
    SmallVector<int64_t, 4> indices;
    onnx_mlir::unflattenIndex(getShape(), flatIndex, indices);
    size_t pos = onnx_mlir::getStridesPosition(indices, getStrides());
    return readBufferPos(pos);
  }

  template <typename X>
  DenseElementsAttr toDenseElementsAttrByType() const {
    if (isSplat())
      return DenseElementsAttr::get(getType(), getSplatValue<X>());
    std::vector<X> xs;
    xs.reserve(getNumElements());
    for (X x : getValues<X>())
      xs.emplace_back(x);
    return DenseElementsAttr::get(getType(), xs);
  }

public:
  template <typename X>
  llvm::Optional<X> tryGetSplatValue() const {
    if (!isSplat())
      return llvm::None;
    return getNumber<X>(getDType(), readBufferPos(0));
  }
  template <typename X>
  X getSplatValue() const {
    return *tryGetSplatValue<X>();
  }

  // TODO: remove this or reimplement using getRawBytes
  DenseElementsAttr toDenseElementsAttr() const {
    if (getElementType().isa<IntegerType>())
      return toDenseElementsAttrByType<APInt>();
    else
      return toDenseElementsAttrByType<APFloat>();
  }

  void readElements(MutableArrayRef<WideNum> dst) const {
    if (isContiguous()) {
      getReader()(getBuffer()->getBuffer(), dst);
    }
    SmallVector<WideNum, 1> wideBufferData;
    wideBufferData.resize_for_overwrite(getNumBufferElements());
    getReader()(getBuffer()->getBuffer(), wideBufferData);
    ArrayRef<int64_t> shape = getShape();
    ArrayRef<int64_t> srcStrides = getStrides();
    ArrayRef<WideNum> src(wideBufferData);
    onnx_mlir::restrideArray(sizeof(WideNum), shape,
        onnx_mlir::castArrayRef<char>(src), srcStrides,
        onnx_mlir::castMutableArrayRef<char>(dst));
  }

  onnx_mlir::ArrayBuffer<WideNum> getWideNums() const {
    const Properties &properties = getProperties();
    if (!properties.isTransformed && properties.isContiguous &&
        onnx_mlir::bytewidthOfDType(properties.bufferDType) ==
            sizeof(WideNum)) {
      return onnx_mlir::asArrayRef<WideNum>(getBuffer()->getBuffer());
    }
    onnx_mlir::ArrayBuffer<WideNum>::Vector vec;
    vec.resize_for_overwrite(getNumElements());
    readElements(vec);
    return std::move(vec);
  }

  onnx_mlir::ArrayBuffer<char> getRawBytes() const {
    const Properties &properties = getProperties();
    if (!properties.isTransformed &&
        properties.dtype == properties.bufferDType) {
      if (properties.isContiguous)
        return onnx_mlir::asArrayRef(getBuffer()->getBuffer());
      // TODO: copy to vector with restrideArray()
    }
    unsigned bytewidth = onnx_mlir::bytewidthOfDType(properties.bufferDType);
    onnx_mlir::ArrayBuffer<char>::Vector vec;
    vec.resize_for_overwrite(getNumElements() * bytewidth);
    MutableArrayRef<char> bytes(vec);
    if (bytewidth == sizeof(WideNum)) {
      readElements(onnx_mlir::castMutableArrayRef<WideNum>(bytes));
    } else {
      SmallVector<WideNum, 1> wideData;
      wideData.resize_for_overwrite(getNumElements());
      readElements(wideData);
      onnx_mlir::narrowArray(getElementType(), wideData, bytes);
    }
    return std::move(bytes);
  }

}; // class DisposableElementsAttr

class DisposablePool : public mlir::DialectInterface::Base<DisposablePool> {
public:
  using Strides = typename DisposableElementsAttr::Strides;
  using Buffer = typename DisposableElementsAttr::Buffer;

  static DisposablePool &create(mlir::MLIRContext *context);

  static DisposablePool *get(mlir::MLIRContext *context);

  DisposablePool(mlir::Dialect *dialect, mlir::MLIRContext *context);
  ~DisposablePool();

  template <typename... Args>
  DisposableElementsAttr createElementsAttr(Args &&...args) {
    auto d = DisposableElementsAttr::get(std::forward<Args>(args)...);
    insert(d);
    return d;
  }

  void garbageCollectUnreachable(ModuleOp moduleOp);

  void scrub(ModuleOp moduleOp);

  void close() {
    assert(pool.empty() && "pool must be scrubbed before close ");
    active = false;
  }

  bool isActive() const { return active; }

private:
  using Pool = std::unordered_set<DisposableElementsAttributeStorage *>;

  void insert(DisposableElementsAttr d);
  void eraseUnreachable(const Pool &reachable);

  Pool pool;
  bool active = true;
};

} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)
