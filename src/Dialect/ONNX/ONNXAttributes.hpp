/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ONNXAttributes.hpp - ONNX Attributes ----------------===//
//
// This file defines attributes in the ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/MemoryBuffer.h"

#include <memory>

namespace onnx_mlir {
// Represents a FLOAT16 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint16_t, float, etc.
//
// TODO: Move to DType.h
struct float_16 { uint16_t u; };
}

namespace mlir {

namespace detail {
// Generates a unique number each time it is called. Is used as hash function
// to defeat the storage uniquer.
size_t uniqueNumber();
}

template <typename T>
struct ImpermanentElementsAttributeStorage : public AttributeStorage {
  using Strides = ArrayRef<int64_t>;
  using Buffer = std::shared_ptr<llvm::MemoryBuffer>;
  using Transform = std::function<T(ArrayRef<char>, size_t)>;
  using KeyTy = std::tuple<ShapedType, Strides>;

  // Constructs only type and strides while the caller sets buffer and transform
  // after construction to minimize copying.
  ImpermanentElementsAttributeStorage(ShapedType type, Strides strides)
    : type(type), strides(strides) {}

  // Equality and hashKey are engineered to defeat the storage uniquer.
  // We don't want uniqueing because we can't compare transforms for equality
  // and we could be in a sitation later where we have the same data or the
  // same buffer address but there is an undetectable mismatch because the
  // buffer and transform were disposed by garbage collection.
  bool operator==(const KeyTy &key) const { return false; }
  static llvm::hash_code hashKey(const KeyTy &key) { return detail::uniqueNumber(); }

  static ImpermanentElementsAttributeStorage *construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    ShapedType type = std::get<0>(key);
    Strides strides = std::get<1>(key);
    return new (allocator.allocate<ImpermanentElementsAttributeStorage>())
      ImpermanentElementsAttributeStorage(type, allocator.copyInto(strides));
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
  // Garbage collection clears the buffer when the ImpermanentElementsAttr is
  // disposed.
  //
  // Multiple ImpermanentElementsAttr can point to the same MemoryBuffer.
  // The MemoryBuffer is destroyed (and heap allocated data freed or mmap'ed file
  // closed) when no one points to it anymore.
  Buffer buffer;

  // Element wise transform of the buffer elements to values of Cpp type T.
  //
  // Garbage collection clears the transform when the ImpermanentElementsAttr is
  // disposed.
  Transform transform;
};

template <typename T>
class ImpermanentElementsAttr : public Attribute::AttrBase<ImpermanentElementsAttr<T>,
  Attribute, ImpermanentElementsAttributeStorage<T>, ElementsAttr::Trait, TypedAttr::Trait> {
public:
  using Storage = ImpermanentElementsAttributeStorage<T>;
  using Strides = typename Storage::Strides;
  using Buffer = typename Storage::Buffer;
  using Transform = typename Storage::Transform;
  using Super = Attribute::AttrBase<ImpermanentElementsAttr<T>, Attribute,
      ImpermanentElementsAttributeStorage<T>, ElementsAttr::Trait, TypedAttr::Trait>;
  using Super::Base::Base;
  static ImpermanentElementsAttr get(ShapedType type, Strides strides, Buffer buffer, Transform transform) {
    ImpermanentElementsAttr a = Super::Base::get(type.getContext(), type, strides);
    Storage &s = *a.getImpl();
    s.buffer = std::move(buffer);
    s.transform = std::move(transform);
    return a;
  }
  ImpermanentElementsAttr(std::nullptr_t) {}
  // Allow implicit conversion to ElementsAttr.
  operator ElementsAttr() const {
    return *this ? this->template cast<ElementsAttr>() : nullptr;
  }
  bool isSplat() const { return true; } // TODO: return true iff strides is all-zeros
  uint64_t getFlattenedIndex(ArrayRef<uint64_t> index) const { return 0; } // TODO: return strides calculation
  Type getType() const { return this->getImpl()->type; }
  FailureOr<detail::ElementsAttrIndexer> getValuesImpl(TypeID elementID) const { return failure(); } // TODO: implement
};

extern template class ImpermanentElementsAttr<bool>;
extern template class ImpermanentElementsAttr<int8_t>;
extern template class ImpermanentElementsAttr<uint8_t>;
extern template class ImpermanentElementsAttr<int16_t>;
extern template class ImpermanentElementsAttr<uint16_t>;
extern template class ImpermanentElementsAttr<::onnx_mlir::float_16>;
extern template class ImpermanentElementsAttr<float>;
extern template class ImpermanentElementsAttr<uint64_t>;

using ImpermanentBoolElementsAttr = ImpermanentElementsAttr<bool>;
using ImpermanentI8ElementsAttr = ImpermanentElementsAttr<int8_t>;
using ImpermanentU8ElementsAttr = ImpermanentElementsAttr<uint8_t>;
using ImpermanentI16ElementsAttr = ImpermanentElementsAttr<int16_t>;
using ImpermanentU16ElementsAttr = ImpermanentElementsAttr<uint16_t>;
using ImpermanentF16ElementsAttr = ImpermanentElementsAttr<::onnx_mlir::float_16>;
using ImpermanentF32ElementsAttr = ImpermanentElementsAttr<float>;
using ImpermanentU64ElementsAttr = ImpermanentElementsAttr<uint64_t>;

}

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ImpermanentBoolElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ImpermanentI8ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ImpermanentU8ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ImpermanentI16ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ImpermanentU16ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ImpermanentF16ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ImpermanentF32ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ImpermanentU64ElementsAttr)
