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

namespace mlir {

size_t uniqueNumber(); // implemented with static atomic counter

template <typename T>
struct ImpermanentElementsAttributeStorage : public AttributeStorage {
  using Strides = ArrayRef<int64_t>;
  using Buffer = std::shared_ptr<llvm::MemoryBuffer>;
  using Transform = std::function<T(ArrayRef<char>, size_t)>;
  using KeyTy = std::tuple<ShapedType, Strides>;

  ImpermanentElementsAttributeStorage(ShapedType type, Strides strides)
    : type(type), strides(strides) {}

  bool operator==(const KeyTy &key) const { return false; }
  static llvm::hash_code hashKey(const KeyTy &key) { return uniqueNumber(); }

  static ImpermanentElementsAttributeStorage *construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    const ShapedType& type = std::get<0>(key);
    const Strides& strides = std::get<1>(key);
    return new (allocator.allocate<ImpermanentElementsAttributeStorage>())
      ImpermanentElementsAttributeStorage(type, allocator.copyInto(strides));
  }

  ShapedType type;
  Strides strides;
  Buffer buffer;
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
    a.set(std::move(buffer), std::move(transform));
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
private:
  void set(Buffer buffer, Transform transform) {
    this->getImpl()->buffer = std::move(buffer);
    this->getImpl()->transform = std::move(transform);
  }
};
using ImpermanentBoolElementsAttr = ImpermanentElementsAttr<bool>;
using ImpermanentI16ElementsAttr = ImpermanentElementsAttr<int16_t>;
using ImpermanentF32ElementsAttr = ImpermanentElementsAttr<float>;
using ImpermanentU64ElementsAttr = ImpermanentElementsAttr<uint64_t>;

}

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ImpermanentBoolElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ImpermanentI16ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ImpermanentF32ElementsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ImpermanentU64ElementsAttr)
