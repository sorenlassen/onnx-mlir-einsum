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
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/MemoryBuffer.h"

#include <memory>

namespace onnx_mlir {
// Represents a FLOAT16 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint16_t, float, etc.
//
// TODO: Move to DType.h
struct float_16 {
  uint16_t u;
};
} // namespace onnx_mlir

namespace mlir {

namespace detail {
// Generates a unique number each time it is called. Is used as hash function
// to defeat the storage uniquer.
size_t uniqueNumber();

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

class PosIterator {
public:
  static PosIterator end(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
    SmallVector<int64_t, 4> zeros(shape.size(), 0);
    return PosIterator(shape, strides, std::move(zeros), numElements(shape));
  }

  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = int64_t;
  using pointer = const value_type *;
  using reference = const value_type &;

  PosIterator(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides)
      : shape(shape), strides(strides), flatIndex(0), pos(0),
        indices(shape.size(), 0) {}
  PosIterator(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
      SmallVector<int64_t, 4> indices)
      : shape(shape), strides(strides),
        flatIndex(flattenedIndex(indices, shape)),
        pos(position(indices, strides)), indices(std::move(indices)) {}
  PosIterator(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
      SmallVector<int64_t, 4> indices, int64_t flatIndex)
      : shape(shape), strides(strides), flatIndex(flatIndex),
        pos(position(indices, strides)), indices(std::move(indices)) {}
  PosIterator() = delete;
  PosIterator(const PosIterator &other) = default;
  PosIterator(PosIterator &&other) = default;

  reference operator*() const { return pos; }
  pointer operator->() { return &pos; }
  PosIterator &operator++() {
    incr();
    return *this;
  }
  PosIterator operator++(int) {
    PosIterator tmp(*this);
    incr();
    return tmp;
  }
  friend bool operator==(const PosIterator &a, const PosIterator &b) {
    return a.flatIndex == b.flatIndex;
  };
  friend bool operator!=(const PosIterator &a, const PosIterator &b) {
    return a.flatIndex != b.flatIndex;
  };

private:
  static int64_t position(
      ArrayRef<int64_t> indices, ArrayRef<int64_t> strides) {
    assert(indices.size() >= strides.size());
    uint64_t pos = 0;
    for (int a = indices.size() - 1, s = strides.size() - 1; s >= 0; --a, --s)
      pos += indices[a] * strides[s];
    return pos;
  }
  static int64_t flattenedIndex(
      ArrayRef<int64_t> indices, ArrayRef<int64_t> shape) {
    assert(indices.size() == shape.size());
    int64_t multiplier = 1;
    uint64_t idx = 0;
    for (int a = indices.size() - 1; a >= 0; --a) {
      idx += indices[a] * multiplier;
      multiplier *= shape[a];
    }
    return idx;
  }
  static int64_t numElements(ArrayRef<int64_t> shape) {
    return ShapedType::getNumElements(shape);
  }

  void incr() {
    assert(flatIndex < numElements(shape));
    ++flatIndex;
    int64_t r = shape.size();
    while (r > 0) {
      --r;
      int64_t s = strides[r];
      pos += s;
      int64_t i = ++indices[r];
      if (i < shape[r]) {
        break;
      } else {
        pos -= i * s;
        indices[r] = 0;
      }
    }
  }

  ArrayRef<int64_t> shape;
  ArrayRef<int64_t> strides;
  int64_t flatIndex; // runs from 0 through numElements-1
  int64_t pos;       // takes values in range [0, bufferNumElements-1]
  SmallVector<int64_t, 4> indices;
}; // class PosIterator

} // namespace detail

template <typename T>
struct DisposableElementsAttributeStorage : public AttributeStorage {
  using Strides = ArrayRef<int64_t>;
  using Buffer = std::shared_ptr<llvm::MemoryBuffer>;
  using Transform = std::function<T(ArrayRef<char>, size_t)>;
  using KeyTy = std::tuple<ShapedType, Strides>;

  // Constructs only type and strides while the caller sets buffer and transform
  // after construction to minimize copying.
  DisposableElementsAttributeStorage(ShapedType type, Strides strides)
      : type(type), strides(strides) {}

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
    return new (allocator.allocate<DisposableElementsAttributeStorage>())
        DisposableElementsAttributeStorage(type, allocator.copyInto(strides));
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

  // Element wise transform of the buffer elements to values of Cpp type T.
  //
  // Garbage collection clears the transform when the DisposableElementsAttr is
  // disposed.
  Transform transform;
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
//
template <typename T>
class DisposableElementsAttr
    : public Attribute::AttrBase<DisposableElementsAttr<T>, Attribute,
          DisposableElementsAttributeStorage<T>, ElementsAttr::Trait,
          TypedAttr::Trait> {
public:
  // TODO: add iterator, built on PosIterator
  using Storage = DisposableElementsAttributeStorage<T>;
  using Strides = typename Storage::Strides;
  using Buffer = typename Storage::Buffer;
  using Transform = typename Storage::Transform;
  using Super = Attribute::AttrBase<DisposableElementsAttr<T>, Attribute,
      DisposableElementsAttributeStorage<T>, ElementsAttr::Trait,
      TypedAttr::Trait>;
  using Super::Base::Base;
  static DisposableElementsAttr get(
      ShapedType type, Strides strides, Buffer buffer, Transform transform) {
    DisposableElementsAttr a =
        Super::Base::get(type.getContext(), type, strides);
    Storage &s = *a.getImpl();
    s.buffer = std::move(buffer);
    s.transform = std::move(transform);
    return a;
  }
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
  const Buffer &getBuffer() const {
    assert(!isDisposed());
    return this->getImpl()->buffer;
  }
  const Transform &getTransform() const {
    assert(!isDisposed());
    return this->getImpl()->transform;
  }
  // isSplat() can return false even if all elements are identical, e.g.
  // no splat check is done to verify if the transform function maps all
  // elements to the same value, or to verify if a mmap'ed file is splat.
  bool isSplat() const {
    return llvm::all_of(getStrides(), [](int64_t s) { return s == 0; });
  }
  FailureOr<detail::ElementsAttrIndexer> getValuesImpl(TypeID elementID) const {
    return failure(); // TODO: implement
  }

private:
  // TODO: figure out if any of the following would be useful public methods
  bool isDisposed() const {
    return !this->getImpl()->buffer || !this->getImpl()->transform;
  }
  bool isContiguous() const {
    return detail::areStridesContiguous(getShape(), getStrides());
  }
  size_t getBufferElementBytewidth() const {
    size_t n = detail::getStridesNumElements(getShape(), getStrides());
    const Buffer &buffer = getBuffer();
    assert(buffer->getBufferSize() <= n);
    assert(n % buffer->getBufferSize() == 0);
    return n / buffer->getBufferSize();
  }
}; // class DisposableElementsAttr

extern template class DisposableElementsAttr<bool>;
extern template class DisposableElementsAttr<int8_t>;
extern template class DisposableElementsAttr<uint8_t>;
extern template class DisposableElementsAttr<int16_t>;
extern template class DisposableElementsAttr<uint16_t>;
extern template class DisposableElementsAttr<::onnx_mlir::float_16>;
extern template class DisposableElementsAttr<float>;
extern template class DisposableElementsAttr<uint64_t>;

using DisposableBoolElementsAttr = DisposableElementsAttr<bool>;
using DisposableI8ElementsAttr = DisposableElementsAttr<int8_t>;
using DisposableU8ElementsAttr = DisposableElementsAttr<uint8_t>;
using DisposableI16ElementsAttr = DisposableElementsAttr<int16_t>;
using DisposableU16ElementsAttr = DisposableElementsAttr<uint16_t>;
using DisposableF16ElementsAttr = DisposableElementsAttr<::onnx_mlir::float_16>;
using DisposableF32ElementsAttr = DisposableElementsAttr<float>;
using DisposableU64ElementsAttr = DisposableElementsAttr<uint64_t>;

} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DisposableBoolElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DisposableI8ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DisposableU8ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DisposableI16ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DisposableU16ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DisposableF16ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DisposableF32ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DisposableU64ElementsAttr)
