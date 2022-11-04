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

#include "src/Dialect/ONNX/AttributesHelper.hpp" // RawBuffer
#include "src/Support/DType.hpp"

#include <memory>
#include <unordered_set>

namespace mlir {

// TODO: move implementation to .cpp

namespace detail {

// Generates a unique number each time it is called. Is used as hash function
// to defeat the storage uniquer.
size_t uniqueNumber();

inline bool isSplat(ArrayRef<int64_t> strides) {
  return llvm::all_of(strides, [](int64_t s) { return s == 0; });
}

inline int64_t getStridesNumElements(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
  if (ShapedType::getNumElements(shape) == 0)
    return 0;
  assert(shape.size() >= strides.size());
  int64_t last = 0;
  for (int a = shape.size() - 1, s = strides.size() - 1; s >= 0; --a, --s)
    last += (shape[a] - 1) * strides[s];
  return last + 1;
}

inline bool areStridesContiguous(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
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

inline size_t getStridesPosition(
    ArrayRef<int64_t> indices, ArrayRef<int64_t> strides) {
  assert(indices.size() >= strides.size());
  size_t pos = 0;
  for (int a = indices.size() - 1, s = strides.size() - 1; s >= 0; --a, --s)
    pos += indices[a] * strides[s];
  return pos;
}

inline SmallVector<int64_t, 4> getDefaultStrides(ArrayRef<int64_t> shape) {
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

// TODO: re-structure DisposableElementsAttr implementation so we don't need
// this expensive function
inline void unflattenIndex(ArrayRef<int64_t> shape, int64_t flatIndex,
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

inline llvm::iota_range<size_t> seq(size_t numElements) {
  return llvm::seq(size_t(0), numElements);
}

template <typename T>
using MappedIndexIterator =
    llvm::mapped_iterator<llvm::iota_range<size_t>::const_iterator,
        std::function<T(size_t)>>;
template <typename T>
auto begin(int64_t numElements, const std::function<T(size_t)> &fun) {
  return llvm::map_iterator(seq(numElements).begin(), fun);
}
template <typename T>
auto end(int64_t numElements, const std::function<T(size_t)> &fun) {
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

  // Is the transform the identity?
  bool isTransformed;
};

using ElementsTransform = std::function<onnx_mlir::IntOrFP(StringRef, size_t)>;

struct DisposableElementsAttributeStorage : public AttributeStorage {
  using Strides = ArrayRef<int64_t>;
  using Buffer = std::shared_ptr<llvm::MemoryBuffer>;
  using Properties = DisposableElementsAttributeProperties;
  using KeyTy = std::tuple<ShapedType, Strides, Properties>;

  // Constructs only type and strides while the caller sets buffer and transform
  // after construction to minimize copying.
  DisposableElementsAttributeStorage(
      ShapedType type, Strides strides, Properties properties)
      : type(type), strides(strides), properties(properties) {}

  // Equality and hashKey are engineered to defeat the storage uniquer.
  // We don't want uniqueing because we can't compare transforms for equality
  // and we could be in a sitation later where we have the same data or the
  // same buffer address but there is an undetectable mismatch because the
  // buffer and transform were disposed by garbage collection.
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

  // Element wise transform of the buffer elements to values of type's
  // element type. Is set to the identity reader function if data is not
  // transformed, namely when properties.isTransformed is false.
  //
  // Garbage collection clears the transform when the DisposableElementsAttr is
  // disposed.
  ElementsTransform transform;
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

private:
  using Super = Attribute::AttrBase<DisposableElementsAttr, Attribute,
      DisposableElementsAttributeStorage, ElementsAttr::Trait,
      TypedAttr::Trait>;
  using Super::Base::Base;

  // Checks the buffer contents to detect if it's splat.
  // To bypass this check, e.g. if the buffer mmaps a file and you don't
  // want to read it here, call get(type, bufferDType, /*isBufferSplat=*/false,
  // buffer, transform). It's ok to not detect splatness.
  static DisposableElementsAttr get(
      ShapedType type, Buffer buffer, ElementsTransform transform = nullptr) {
    ArrayRef<char> rawBuffer = onnx_mlir::asArrayRef(buffer->getBuffer());
    bool isBufferSplat = false;
    if (!DenseElementsAttr::isValidRawBuffer(type, rawBuffer, isBufferSplat))
      llvm_unreachable("invalid buffer passed to DisposableElementsAttr::get");
    return get(type, isBufferSplat, buffer, transform);
  }

  static DisposableElementsAttr get(ShapedType type, bool isBufferSplat,
      Buffer buffer, ElementsTransform transform = nullptr) {
    onnx_mlir::DType dtype = onnx_mlir::dtypeOf(type.getElementType());
    SmallVector<int64_t, 4> strides;
    if (!isBufferSplat)
      strides = detail::getDefaultStrides(type.getShape());
    bool isContiguous = type.getNumElements() == 1 || !isBufferSplat;
    Properties properties = {.dtype = dtype,
        .bufferDType = dtype,
        .isBufferSplat = isBufferSplat,
        .isContiguous = isContiguous,
        .isTransformed = transform != nullptr};
    return create(
        type, strides, properties, std::move(buffer), std::move(transform));
  }

  static DisposableElementsAttr get(ShapedType type, Strides strides,
      Properties properties, Buffer buffer,
      ElementsTransform transform = nullptr) {
    unsigned w = onnx_mlir::getIntOrFloatByteWidth(type.getElementType());
    assert(buffer->getBufferSize() % w == 0);
    int64_t numBufferElements = buffer->getBufferSize() / w;
    auto shape = type.getShape();
    assert(!detail::isSplat(strides) || numBufferElements == 1);
    assert(numBufferElements == 1 ||
           numBufferElements == detail::getStridesNumElements(shape, strides));
    assert(!properties.isContiguous ||
           detail::areStridesContiguous(shape, strides));
    // TODO: add more checks
    return create(
        type, strides, properties, std::move(buffer), std::move(transform));
  }

  static DisposableElementsAttr create(ShapedType type, Strides strides,
      Properties properties, Buffer buffer,
      ElementsTransform transform = nullptr) {
    DisposableElementsAttr a =
        Super::Base::get(type.getContext(), type, strides, properties);
    Storage &s = *a.getImpl();
    s.buffer = std::move(buffer);
    s.transform = transform ? std::move(transform)
                            : getIdentityTransform(properties.dtype);
    return a;
  }

  static ElementsTransform getIdentityTransform(onnx_mlir::DType);

  template <typename X>
  static X getNumber(onnx_mlir::DType tag, onnx_mlir::IntOrFP n) {
    if constexpr (std::is_same_v<X, llvm::APFloat>)
      return n.toAPFloat(tag);
    else if constexpr (std::is_same_v<X, llvm::APInt>)
      return n.toAPInt(tag);
    else
      return n.to<X>(tag);
  }

public:
  DisposableElementsAttr(std::nullptr_t) {}

  // Allow implicit conversion to ElementsAttr.
  operator ElementsAttr() const {
    return *this ? this->template cast<ElementsAttr>() : nullptr;
  }

  ShapedType getType() const { return this->getImpl()->type; }
  Type getElementType() const { return getType().getElementType(); }
  ArrayRef<int64_t> getShape() const { return getType().getShape(); }
  int64_t getRank() const { return getType().getRank(); }
  int64_t getNumElements() const { return getType().getNumElements(); }
  ArrayRef<int64_t> getStrides() const { return this->getImpl()->strides; }
  const Properties &getProperties() const {
    return this->getImpl()->properties;
  }
  const Buffer &getBuffer() const {
    assert(!isDisposed());
    return this->getImpl()->buffer;
  }
  const ElementsTransform &getTransform() const {
    assert(!isDisposed());
    return this->getImpl()->transform;
  }
  onnx_mlir::DType getDType() const { return getProperties().dtype; }
  // isSplat() can return false even if all elements are identical, e.g.
  // no splat check is done to verify if the transform function maps all
  // elements to the same value, or to verify if a mmap'ed file is splat.
  bool isSplat() const { return getProperties().isBufferSplat; }
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

  template <typename X>
  using iterator = detail::MappedIndexIterator<X>;

  template <typename X>
  using iterator_range = llvm::iterator_range<iterator<X>>;

  // TODO: add Attribute
  using NonContiguousIterableTypesT = std::tuple<bool, int8_t, uint8_t, int16_t,
      uint16_t, int32_t, uint32_t, int64_t, uint64_t, onnx_mlir::float_16,
      float, double, APInt, APFloat>;

  template <typename T>
  static constexpr bool isIntOrFPConvertible =
      (onnx_mlir::CppTypeTrait<T>::dtype != onnx_mlir::DType::UNDEFINED &&
          onnx_mlir::CppTypeTrait<T>::isIntOrFloat) ||
      std::is_same_v<T, llvm::APInt> || std::is_same_v<T, llvm::APFloat>;

  template <typename X>
  using OverloadToken = typename Super::template OverloadToken<X>;

  template <typename X>
  FailureOr<iterator<X>> try_value_begin_impl(OverloadToken<X>) const {
    if constexpr (isIntOrFPConvertible<X>) {
      onnx_mlir::DType dtype = getDType();
      DisposableElementsAttr attr = *this;
      return detail::begin<X>(
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
    return detail::end<X>(getNumElements(), nullptr);
  }

  void printWithoutType(raw_ostream &os) const;

  // TODO: remove this or reimplement using getRawBytes
  DenseElementsAttr toDenseElementsAttr() const {
    if (getElementType().isa<IntegerType>())
      return toDenseElementsAttrByType<APInt>();
    else
      return toDenseElementsAttrByType<APFloat>();
  }

  using Reader = std::function<void(size_t, onnx_mlir::IntOrFP)>;

  void read(const Reader &sink) const {
    onnx_mlir::dispatchByDType(getDType(), [&sink, this](auto dtype) {
      traverse(this->getType(), this->getStrides(),
          [&](size_t pos, size_t flatIndex) {
            sink(flatIndex, this->readBufferPos(pos));
          });
    });
  }

  onnx_mlir::RawBuffer getRawBytes() const {
    onnx_mlir::DType dtype = getDType();
    unsigned bytewidth = onnx_mlir::bytewidthOfDType(dtype);
    if (isSplat()) {
      onnx_mlir::RawBuffer::Vector vec;
      vec.resize_for_overwrite(bytewidth);
      onnx_mlir::dispatchByDType(dtype, [&](auto staticDType) {
        writeIntOrFP<staticDType>(vec, 0, readBufferPos(0));
      });
      return onnx_mlir::RawBuffer(std::move(vec));
    }
    const Properties &properties = getProperties();
    if (properties.isTransformed || !properties.isContiguous) {
      ShapedType type = getType();
      onnx_mlir::RawBuffer::Vector vec;
      vec.resize_for_overwrite(type.getNumElements() * bytewidth);
      read(dispatchByDType(dtype, [&vec](auto staticDType) -> Reader {
        using cpptype = onnx_mlir::CppType<staticDType>;
        return [&vec](size_t flatIndex, onnx_mlir::IntOrFP n) {
          writeIntOrFP<onnx_mlir::toDType<cpptype>>(vec, flatIndex, n);
        };
      }));
      return onnx_mlir::RawBuffer(std::move(vec));
    } else {
      return onnx_mlir::asArrayRef(getBuffer()->getBuffer());
    }
  }

private: // TODO: Figure out if any of the following would be useful publicly.
  onnx_mlir::IntOrFP readBufferPos(size_t pos) const {
    return getTransform()(getBuffer()->getBuffer(), pos);
  }

  // Warning: this is inefficient because it calls unflattenIndex on flatIndex.
  onnx_mlir::IntOrFP readFlatIndex(size_t flatIndex) const {
    SmallVector<int64_t, 4> indices;
    detail::unflattenIndex(getShape(), flatIndex, indices);
    size_t pos = detail::getStridesPosition(indices, getStrides());
    return readBufferPos(pos);
  }

  // This is like the inverse of getIdentityTransform. Whereas identity
  // transform returns an IntOrFP which it reads from a given read-only buffer
  // and pos, writeIntOrFP writes an IntOrFP argument to a given mutable array
  // and index.
  template <onnx_mlir::DType DTYPE>
  static void writeIntOrFP(
      llvm::MutableArrayRef<char> dst, size_t index, onnx_mlir::IntOrFP n) {
    using cpptype = onnx_mlir::CppType<DTYPE>;
    onnx_mlir::castMutableArrayRef<cpptype>(dst)[index] = n.to<cpptype>(DTYPE);
  }

  template <typename Act>
  static size_t traverse(ShapedType type, ArrayRef<int64_t> strides,
      const Act &act, int64_t axis = 0, size_t offset = 0,
      size_t flatIndex = 0) {
    ArrayRef<int64_t> shape = type.getShape();
    int64_t rank = shape.size();
    if (axis == rank) {
      act(offset, flatIndex);
      return flatIndex + 1;
    }
    int64_t skip = rank - strides.size();
    size_t stride = axis < skip ? 0 : strides[axis - skip];
    size_t dimSize = shape[axis];
    for (size_t pos = offset, idx = 0; idx < dimSize; ++idx, pos += stride) {
      flatIndex = traverse(type, strides, act, axis + 1, pos, flatIndex);
    }
    return flatIndex;
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

  bool isDisposed() const {
    //  TODO: Decide if a splat value can be represented with a constant
    //        transform with no buffer; in that case isDisposed should
    //        only return true if both buffer and transform are null.
    return !this->getImpl()->buffer;
  }

  bool isContiguous() const { return getProperties().isContiguous; }

  size_t getBufferElementBytewidth() const {
    size_t n = detail::getStridesNumElements(getShape(), getStrides());
    const Buffer &buffer = getBuffer();
    assert(buffer->getBufferSize() <= n);
    assert(n % buffer->getBufferSize() == 0);
    return n / buffer->getBufferSize();
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
