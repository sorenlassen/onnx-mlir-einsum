/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- DisposableElementsAttr.hpp --------------------===//
//
// DisposableElementsAttr, garbage collectible alternative to DenseElementsAttr.
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

// TODO: move more implementation to .cpp

namespace detail {

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

struct DisposableElementsAttributeStorage;

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
  using Base::Base;

public:
  using Storage = DisposableElementsAttributeStorage;
  using Strides = ArrayRef<int64_t>;
  using Buffer = std::shared_ptr<llvm::MemoryBuffer>;
  using Properties = DisposableElementsAttributeProperties;
  using Reader = DisposableElementsAttributeReader;
  // DType and WideNum are ubiquitous in the class definition and these using
  // statements are convenient as they let us omit their namespace qualifier.
  using DType = onnx_mlir::DType;
  using WideNum = onnx_mlir::WideNum;

  //===----------------------------------------------------------------------===//
  // Instantiation:
  //
  // The get methods are private and are only accessed from DisposablePool.
  // Call DisposablePool::get(..) to instantiate DisposableElementsAttr.
  //===----------------------------------------------------------------------===//
public:
  friend class DisposablePool;

private:
  // Checks the buffer contents to detect if it's splat.
  // To bypass this check, e.g. if the buffer mmaps a file and you don't
  // want to read it here, call get(type, bufferDType, /*isBufferSplat=*/false,
  // buffer, reader). It's ok to not detect splatness.
  //
  // Assumes isTransformed if reader != nullptr.
  static DisposableElementsAttr get(
      ShapedType type, const Buffer &buffer, Reader reader = nullptr);

  // Assumes isTransformed if reader != nullptr.
  static DisposableElementsAttr get(ShapedType type, bool isBufferSplat,
      const Buffer &buffer, Reader reader = nullptr);

  static DisposableElementsAttr get(ShapedType type, Strides strides,
      Properties properties, const Buffer &buffer, Reader reader = nullptr);

  // Internal method called by get(..) methods.
  static DisposableElementsAttr create(ShapedType type, Strides strides,
      Properties properties, const Buffer &buffer, Reader reader = nullptr);

public:
  DisposableElementsAttr(std::nullptr_t) {}

  // Allow implicit conversion to ElementsAttr.
  operator ElementsAttr() const {
    return *this ? cast<ElementsAttr>() : nullptr;
  }

  //===----------------------------------------------------------------------===//
  // Instance properties:
  //===----------------------------------------------------------------------===//
private:
  bool isDisposed() const;

  bool isContiguous() const { return getProperties().isContiguous; }

  int64_t getNumBufferElements() const {
    unsigned bytewidth = bytewidthOfDType(getProperties().bufferDType);
    return getBuffer()->getBufferSize() / bytewidth;
  }

public:
  ShapedType getType() const;
  Strides getStrides() const;
  const Properties &getProperties() const;
  const Buffer &getBuffer() const;
  const Reader &getReader() const;

  DType getDType() const { return getProperties().dtype; }
  // isSplat() can return false even if all elements are identical, e.g.
  // no splat check is done to verify if the reader function maps all
  // elements to the same value, or to verify if a mmap'ed file is splat.
  bool isSplat() const { return getProperties().isBufferSplat; }

  Type getElementType() const { return getType().getElementType(); }
  ArrayRef<int64_t> getShape() const { return getType().getShape(); }
  int64_t getRank() const { return getType().getRank(); }
  int64_t getNumElements() const { return getType().getNumElements(); }

  //===----------------------------------------------------------------------===//
  // Iteration:
  //===----------------------------------------------------------------------===//
private:
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
  // (We could add Attribute, IntegerAttr, FloatAttr by adding support for
  // them in getNumber<X>.)
  using NonContiguousIterableTypesT = std::tuple<bool, int8_t, uint8_t, int16_t,
      uint16_t, int32_t, uint32_t, int64_t, uint64_t, onnx_mlir::float_16,
      onnx_mlir::bfloat_16, float, double, APInt, APFloat>;

  template <typename X>
  using iterator = detail::MappedIndexIterator<X>;

  template <typename X>
  using iterator_range = llvm::iterator_range<iterator<X>>;

  // This implementation enables the value_begin() and getValues() methods
  // from the ElementsAttr interface, for the NonContiguousIterableTypesT types.
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
      return failure();
    }
  }

  // equivalent to getValues<X>().end(), which is probably slower?
  template <typename X>
  iterator<X> value_end() const {
    return detail::endMappedIndexIterator<X>(getNumElements(), nullptr);
  }

  //===----------------------------------------------------------------------===//
  // Other access to the elements:
  //===----------------------------------------------------------------------===//
private:
  // Warning: this is somewhat inefficient because it invokes getReader().
  // It's more efficient to copy out data in bulk with readElements().
  WideNum readBufferPos(size_t pos) const;

  // Warning: this is inefficient unless isContiguous() or isSplat().
  WideNum readFlatIndex(size_t flatIndex) const;

  // Warning: this is inefficient because it calls unflattenIndex on flatIndex.
  size_t flatIndexToBufferPos(size_t flatIndex) const;

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

  DenseElementsAttr toDenseElementsAttr() const {
    onnx_mlir::ArrayBuffer<char> bytes = getRawBytes();
    if (!getElementType().isInteger(1))
      return DenseElementsAttr::getFromRawBuffer(getType(), bytes.get());
    // DenseElementsAttr::getFromRawBuffer bit packs bools so we
    // cannot use it, so we pass as ArrayRef<bool> instead:
    auto bools = onnx_mlir::castArrayRef<bool>(bytes.get());
    return DenseElementsAttr::get(getType(), bools);
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

  void printWithoutType(raw_ostream &os) const;

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
