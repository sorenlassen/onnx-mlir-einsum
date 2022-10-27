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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/MemoryBuffer.h"

#include "src/Support/DType.hpp"

#include <memory>

namespace mlir {

// TODO: move implementation to .cpp

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

inline size_t getStridesPosition(
    ArrayRef<int64_t> indices, ArrayRef<int64_t> strides) {
  assert(indices.size() >= strides.size());
  size_t pos = 0;
  for (int a = indices.size() - 1, s = strides.size() - 1; s >= 0; --a, --s)
    pos += indices[a] * strides[s];
  return pos;
}

// TODO: re-structure DisposableElementsAttr implementation so we don't need
// this expensive function
inline void unflattenIndex(ArrayRef<int64_t> shape, int64_t flatIndex,
    SmallVectorImpl<int64_t> &indices) {
  int64_t axis = shape.size();
  if (axis == 0)
    return;
  while (axis > 1) {
    --axis;
    int64_t dimSize = shape[axis];
    assert(dimSize > 0 && "cannot unflatten shape with zeros");
    int64_t rem = flatIndex % dimSize;
    flatIndex /= dimSize;
    indices.push_back(rem);
  }
  assert(flatIndex < shape[0]);
  indices.push_back(flatIndex);
}

class IndexIterator : public llvm::iterator_facade_base<IndexIterator,
                          std::random_access_iterator_tag, size_t> {
public:
  IndexIterator(size_t index = 0) : index(index) {}
  difference_type operator-(const IndexIterator &rhs) const {
    return index - rhs.index;
  }
  bool operator==(const IndexIterator &rhs) const { return index == rhs.index; }
  bool operator<(const IndexIterator &rhs) const { return index < rhs.index; }
  IndexIterator &operator+=(difference_type offset) {
    index += offset;
    return *this;
  }
  IndexIterator &operator-=(difference_type offset) {
    index -= offset;
    return *this;
  }
  size_t operator*() const { return index; }

private:
  size_t index;
};

template <typename T>
using MappedIndexIterator =
    llvm::mapped_iterator<IndexIterator, std::function<T(size_t)>>;

template <typename T>
inline auto makeMappedIndexIterator(
    size_t index, const std::function<T(size_t)> &fun) {
  return MappedIndexIterator<T>(IndexIterator(index), fun);
}

} // namespace detail

using ElementsTransform = std::function<onnx_mlir::Number64(StringRef, size_t)>;

struct DisposableElementsAttributeStorage : public AttributeStorage {
  using Strides = ArrayRef<int64_t>;
  using Buffer = std::shared_ptr<llvm::MemoryBuffer>;
  using KeyTy = std::tuple<ShapedType, Strides, onnx_mlir::DType>;

  // Constructs only type and strides while the caller sets buffer and transform
  // after construction to minimize copying.
  DisposableElementsAttributeStorage(
      ShapedType type, Strides strides, onnx_mlir::DType dtype)
      : type(type), strides(strides), dtype(dtype) {}

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
    onnx_mlir::DType dtype = std::get<2>(key);
    return new (allocator.allocate<DisposableElementsAttributeStorage>())
        DisposableElementsAttributeStorage(
            type, allocator.copyInto(strides), dtype);
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

  // Data type (BOOL, INT8, FLOAT16, etc) of the elements in buffer.
  onnx_mlir::DType dtype;

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
  ElementsTransform transform;
}; // struct DisposableElementsAttributeStorage

// DisposableElementsAttr is an alternative to DenseElementsAttr
// with the following features:
//
// 1. The memory can be heap allocated or mmap'ed from a file and will be
// released (heap allocation freed or file closed) between compiler passes
// when it is no longer reachable from the operation graph.
// TODO: Implement garbage collector that does that.
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
// TODO: Make DisposableElementsAttr print the same as DenseElementsAttr.
//
// NOTE: DenseResourceElementsAttr is an alternative for heap allocated memory
//       (but without garbage collection or the other features listed above).
//
class DisposableElementsAttr
    : public Attribute::AttrBase<DisposableElementsAttr, Attribute,
          DisposableElementsAttributeStorage, ElementsAttr::Trait,
          TypedAttr::Trait> {
public:
  using Storage = DisposableElementsAttributeStorage;
  using Strides = typename Storage::Strides;
  using Buffer = typename Storage::Buffer;
  using Super = Attribute::AttrBase<DisposableElementsAttr, Attribute,
      DisposableElementsAttributeStorage, ElementsAttr::Trait,
      TypedAttr::Trait>;
  using Super::Base::Base;
  static DisposableElementsAttr get(ShapedType type, Strides strides,
      onnx_mlir::DType dtype, Buffer buffer, ElementsTransform transform) {
    assert(onnx_mlir::isIntOrFPType(type.getElementType(), 64));
    DisposableElementsAttr a =
        Super::Base::get(type.getContext(), type, strides, dtype);
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
  onnx_mlir::DType getDType() const { return this->getImpl()->dtype; }
  const Buffer &getBuffer() const {
    assert(!isDisposed());
    return this->getImpl()->buffer;
  }
  const ElementsTransform &getTransform() const {
    assert(!isDisposed());
    return this->getImpl()->transform;
  }
  // isSplat() can return false even if all elements are identical, e.g.
  // no splat check is done to verify if the transform function maps all
  // elements to the same value, or to verify if a mmap'ed file is splat.
  bool isSplat() const {
    return llvm::all_of(getStrides(), [](int64_t s) { return s == 0; });
  }
  template <typename X>
  X getSplatValue() const {
    llvm_unreachable("TODO: implement getSplatValue");
  }

  template <typename X>
  using iterator = detail::MappedIndexIterator<X>;

  template <typename X>
  using iterator_range = llvm::iterator_range<iterator<X>>;

  // TODO: add APInt, APFloat, Attribute
  using NonContiguousIterableTypesT = onnx_mlir::IntOrFPDTypes;

  template <typename X>
  using OverloadToken = typename Super::template OverloadToken<X>;

  template <typename X>
  std::enable_if_t<onnx_mlir::isIntOrFP<X>(64), FailureOr<iterator<X>>> try_value_begin_impl(
      OverloadToken<X>) const {
    DisposableElementsAttributeStorage *s = this->getImpl();
    return detail::makeMappedIndexIterator<X>(0, [s](size_t flatIndex) -> X {
      SmallVector<int64_t, 4> indices;
      detail::unflattenIndex(s->type.getShape(), flatIndex, indices);
      size_t pos = detail::getStridesPosition(indices, s->strides);
      onnx_mlir::Number64 n = s->transform(s->buffer->getBuffer(), pos);
      return onnx_mlir::fromNumber64<X>(n);
    });
  }

  // TODO: support iteration over APInt, APFloat, Attribute
  template <typename X>
  std::enable_if_t<!onnx_mlir::isIntOrFP<X>(64), FailureOr<iterator<X>>>
  try_value_begin_impl(OverloadToken<X>) const {
    return failure();
  }

  // equivalent to getValues<X>().end(), which is probably slower?
  template <typename X>
  iterator<X> value_end() const {
    return detail::makeMappedIndexIterator<X>(getNumElements(), nullptr);
  }

  void print(AsmPrinter &printer) const {
    llvm::errs() << "DisposableElementsAttr::print invoked\n";
    print(printer.getStream());
  }
  void print(raw_ostream &os) const { toDenseElementsAttr().print(os); }
  DenseElementsAttr toDenseElementsAttr() const {
    llvm::errs() << "toDenseElementsAttr invoked\n";
    return nullptr; // TODO: implement this
  }

private: // TODO: Figure out if any of the following would be useful publicly.
  bool isDisposed() const {
    //  TODO: Decide if a splat value can be represented with a constant
    //        transform with no buffer; in that case isDisposed should
    //        only return true if both buffer and transform are null.
    return !this->getImpl()->buffer;
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

} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)
