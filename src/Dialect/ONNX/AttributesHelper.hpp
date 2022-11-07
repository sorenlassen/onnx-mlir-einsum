/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ AttributesHelper.hpp ------------------------===//
//
// Attributes helper functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Support/Arrays.hpp"
#include "src/Support/DType.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"

namespace llvm {
class raw_ostream;
}

namespace onnx_mlir {

union WideNum;

// Makes deep copy of elements, unless they are already a DenseElementsAttr.
mlir::DenseElementsAttr toDenseElementsAttr(mlir::ElementsAttr elements);

// Makes a DisposableElementsAttr that points to elements' raw data, if
// elements is DenseElementsAttr, except if the element type is bool, then
// it makes a deep copy because DisposableElementsAttr doesn't bit pack bools.
//
// TODO: decide if caller should pass in DisposablePool
mlir::DisposableElementsAttr toDisposableElementsAttr(
    mlir::ElementsAttr elements);

// TODO: remove some of these functions (they overlap and are not all used):

mlir::DenseElementsAttr makeDenseElementsAttrFromRawBytes(
    mlir::ShapedType type, llvm::ArrayRef<char> bytes);

// TODO: decide if caller should pass in DisposablePool
mlir::DisposableElementsAttr tryMakeDisposableElementsAttrFromRawBytes(
    mlir::ShapedType type, llvm::ArrayRef<char> bytes, bool mustCopy);

mlir::DenseResourceElementsAttr tryMakeDenseResourceElementsAttrFromRawBytes(
    mlir::ShapedType type, llvm::ArrayRef<char> bytes, bool mustCopy);

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

ArrayBuffer<char> getElementsRawBytes(mlir::ElementsAttr elements);

ArrayBuffer<WideNum> getElementsWideNums(mlir::ElementsAttr elements);

void readIntElements(
    mlir::ElementsAttr elements, llvm::MutableArrayRef<int64_t> ints);

void readFPElements(
    mlir::ElementsAttr elements, llvm::MutableArrayRef<double> fps);

// Prints elements the same way as DenseElementsAttr.
void printIntOrFPElementsAttrAsDense(
    mlir::ElementsAttr attr, llvm::raw_ostream &os);

void printIntOrFPElementsAttrAsDenseWithoutType(
    mlir::ElementsAttr attr, llvm::raw_ostream &os);

} // namespace onnx_mlir