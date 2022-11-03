/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ AttributesHelper.hpp ------------------------===//
//
// Attributes helper functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"

namespace llvm {
class raw_ostream;
}

namespace onnx_mlir {

// Light-weigh version of MemoryBuffer. Can either point to external
// memory or hold internal memory.
class RawBuffer {
public:
  using Vector = llvm::SmallVector<char, 8>;

  RawBuffer(Vector &&vec) : vec(std::move(vec)), ref(this->vec) {}
  RawBuffer(llvm::ArrayRef<char> ref) : vec(), ref(ref) {}
  RawBuffer() = delete;
  RawBuffer(const RawBuffer &) = delete;
  RawBuffer(RawBuffer &&other)
      : vec(std::move(other.vec)),
        ref(vec.empty() ? other.ref : llvm::makeArrayRef(vec)) {}

  llvm::ArrayRef<char> get() const { return ref; }
  size_t size() const { return ref.size(); }
  const char *data() const { return ref.data(); }

private:
  const Vector vec;
  const llvm::ArrayRef<char> ref;
};

mlir::ElementsAttr makeDenseIntOrFPElementsAttrFromRawBuffer(
    mlir::ShapedType type, llvm::ArrayRef<char> bytes, bool mustCopy);

template <typename NumericType>
mlir::ElementsAttr makeDenseIntOrFPElementsAttr(
    mlir::ShapedType type, llvm::ArrayRef<NumericType> numbers, bool mustCopy) {
  llvm::ArrayRef<char> bytes(reinterpret_cast<const char *>(numbers.data()),
      numbers.size() * sizeof(NumericType));
  return makeDenseIntOrFPElementsAttrFromRawBuffer(type, bytes, mustCopy);
}

typedef llvm::function_ref<void(llvm::MutableArrayRef<char>)>
    FillDenseRawBufferFn;

mlir::ElementsAttr makeDenseIntOrFPElementsAttrWithRawBuffer(
    mlir::ShapedType type, FillDenseRawBufferFn fill);

// llvm::ArrayRef<char> getDenseIntOrFPRawData(mlir::ElementsAttr elements);

RawBuffer getDenseIntOrFPRawData(mlir::ElementsAttr elements);

void readDenseInts(
    mlir::ElementsAttr elements, llvm::MutableArrayRef<int64_t> ints);

void readDenseFPs(
    mlir::ElementsAttr elements, llvm::MutableArrayRef<double> fps);

mlir::DenseElementsAttr toDenseElementsAttribute(mlir::ElementsAttr elements);

// Prints elements the same way as DenseElementsAttr.
void printIntOrFPElementsAttrAsDense(
    mlir::ElementsAttr attr, llvm::raw_ostream &os);

void printIntOrFPElementsAttrAsDenseWithoutType(
    mlir::ElementsAttr attr, llvm::raw_ostream &os);

} // namespace onnx_mlir