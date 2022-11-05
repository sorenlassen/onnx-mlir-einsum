/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ AttributesHelper.hpp ------------------------===//
//
// Attributes helper functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Support/DType.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"

namespace llvm {
class raw_ostream;
}

namespace onnx_mlir {

// Light-weight version of MemoryBuffer. Can either point to external memory or
// hold internal memory. An ArrayBuffer can only be moved, not copied.
template <typename T>
class ArrayBuffer {
public:
  using Vector = llvm::SmallVector<T, 8 / sizeof(T)>;

  ArrayBuffer() = default; // empty
  ArrayBuffer(Vector &&vec) : vec(std::move(vec)), ref(this->vec) {}
  ArrayBuffer(llvm::ArrayRef<T> ref) : vec(), ref(ref) {}
  ArrayBuffer(const ArrayBuffer &) = delete;
  ArrayBuffer(ArrayBuffer &&other)
      : vec(std::move(other.vec)),
        ref(vec.empty() ? other.ref : llvm::makeArrayRef(vec)) {}

  llvm::ArrayRef<T> get() const { return ref; }

  static ArrayBuffer make(size_t length,
      const std::function<void(llvm::MutableArrayRef<T>)> &filler) {
    Vector vec;
    vec.resize_for_overwrite(length);
    filler(llvm::makeMutableArrayRef(vec.begin(), length));
    return std::move(vec);
  }

private:
  const Vector vec;
  const llvm::ArrayRef<T> ref;
};

mlir::DenseElementsAttr makeDenseElementsAttrFromRawBytes(
    mlir::ShapedType type, llvm::ArrayRef<char> bytes);

mlir::ElementsAttr makeElementsAttrFromRawBytes(
    mlir::ShapedType type, llvm::ArrayRef<char> bytes, bool mustCopy);

template <typename NumericType>
mlir::ElementsAttr makeElementsAttr(
    mlir::ShapedType type, llvm::ArrayRef<NumericType> numbers, bool mustCopy) {
  return makeElementsAttrFromRawBytes(
      type, castArrayRef<char>(numbers), mustCopy);
}

using RawBuffer = ArrayBuffer<char>;

typedef llvm::function_ref<void(llvm::MutableArrayRef<char>)> RawBytesFiller;

mlir::ElementsAttr makeElementsAttrWithRawBytesFiller(
    mlir::ShapedType type, RawBytesFiller filler);

RawBuffer getElementsRawBytes(mlir::ElementsAttr elements);

ArrayBuffer<IntOrFP> getElementsIntOrFPs(mlir::ElementsAttr elements);

void readIntElements(
    mlir::ElementsAttr elements, llvm::MutableArrayRef<int64_t> ints);

void readFPElements(
    mlir::ElementsAttr elements, llvm::MutableArrayRef<double> fps);

mlir::DenseElementsAttr toDenseElementsAttribute(mlir::ElementsAttr elements);

// Prints elements the same way as DenseElementsAttr.
void printIntOrFPElementsAttrAsDense(
    mlir::ElementsAttr attr, llvm::raw_ostream &os);

void printIntOrFPElementsAttrAsDenseWithoutType(
    mlir::ElementsAttr attr, llvm::raw_ostream &os);

} // namespace onnx_mlir