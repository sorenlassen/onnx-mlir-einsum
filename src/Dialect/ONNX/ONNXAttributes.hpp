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
};

template <typename T>
class DisposableElementsAttr
    : public Attribute::AttrBase<DisposableElementsAttr<T>, Attribute,
          DisposableElementsAttributeStorage<T>, ElementsAttr::Trait,
          TypedAttr::Trait> {
public:
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
  bool isSplat() const {
    return llvm::all_of(getStrides(), [](int64_t s) { return s == 0; });
  }
  FailureOr<detail::ElementsAttrIndexer> getValuesImpl(TypeID elementID) const {
    return failure(); // TODO: implement
  }
private:
  // TODO: figure out if any of the following would be useful public methods
  bool isDisposed() const { return !this->getImpl()->buffer || !this->getImpl()->transform; }
  uint64_t getFlattenedIndex(ArrayRef<uint64_t> indices) const {
    ArrayRef<int64_t> strides = getStrides();
    assert(indices.size() >= strides.size());
    uint64_t idx = 0;
    for (int a = indices.size() - 1, s = strides.size() - 1; s >= 0; --a, --s)
      idx += indices[a] * strides[s];
    return idx;
  }
  int64_t getBufferNumElements() const {
    ArrayRef<int64_t> shape = getShape();
    if (ShapedType::getNumElements(shape) == 0)
      return 0;
    ArrayRef<int64_t> strides = getStrides();
    assert(shape.size() >= strides.size());
    int64_t last = 0;
    for (int a = shape.size() - 1, s = strides.size() - 1; s >= 0; --a, --s)
      last += (shape[a] - 1) * strides[s];
    return last + 1;
  }
  bool isContiguous() const {
    ArrayRef<int64_t> shape = getShape();
    ArrayRef<int64_t> strides = getStrides();
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
  size_t getBufferElementBytewidth() const {
    size_t n = getBufferNumElements();
    const Buffer& buffer = getBuffer();
    assert(buffer->getBufferSize() <= n);
    assert(n % buffer->getBufferSize() == 0);
    return n / buffer->getBufferSize();
  }
};

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
