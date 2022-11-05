/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ AttributesHelper.hpp ------------------------===//
//
// Attributes helper functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Support/Arrays.hpp"
#include "src/Support/DType.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"

namespace llvm {
class raw_ostream;
}

namespace onnx_mlir {

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