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

mlir::ElementsAttr makeDenseIntOrFPElementsAttrFromRawBuffer(
    mlir::ShapedType type, llvm::ArrayRef<char> bytes);

template <typename NumericType>
mlir::ElementsAttr makeDenseIntOrFPElementsAttr(
    mlir::ShapedType type, llvm::ArrayRef<NumericType> numbers) {
  llvm::ArrayRef<char> bytes(reinterpret_cast<const char *>(numbers.data()),
      numbers.size() * sizeof(NumericType));
  return makeDenseIntOrFPElementsAttrFromRawBuffer(type, bytes);
}

typedef llvm::function_ref<void(llvm::MutableArrayRef<char>)>
    FillDenseRawBufferFn;

mlir::ElementsAttr makeDenseIntOrFPElementsAttrWithRawBuffer(
    mlir::ShapedType type, FillDenseRawBufferFn fill);

llvm::ArrayRef<char> getDenseIntOrFPRawData(mlir::ElementsAttr elements);

void readDenseInts(
    mlir::ElementsAttr elements, llvm::MutableArrayRef<int64_t> ints);

void readDenseFPs(
    mlir::ElementsAttr elements, llvm::MutableArrayRef<double> fps);

mlir::DenseElementsAttr toDenseElementsAttribute(mlir::ElementsAttr elements);

// Prints elements the same way as DenseElementsAttr.
void printIntOrFPElementsAttrAsDense(mlir::ElementsAttr attr, llvm::raw_ostream &os);

} // namespace onnx_mlir