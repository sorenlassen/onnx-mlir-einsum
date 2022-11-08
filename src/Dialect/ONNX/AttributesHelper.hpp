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

// Enable DenseElementsAttr to operate on float_16, bfloat_16 data types.
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<onnx_mlir::float_16> {
  static constexpr bool value = true;
};
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<onnx_mlir::bfloat_16> {
  static constexpr bool value = true;
};

namespace onnx_mlir {

union WideNum;

// Makes deep copy of elements, unless they are already a DenseElementsAttr.
mlir::DenseElementsAttr toDenseElementsAttr(mlir::ElementsAttr elements);

// TODO: remove some of these functions (they overlap and are not all used):

typedef llvm::function_ref<void(llvm::MutableArrayRef<char>)> RawBytesFiller;

mlir::ElementsAttr makeElementsAttrWithRawBytesFiller(
    mlir::ShapedType type, RawBytesFiller filler);

ArrayBuffer<char> getElementsRawBytes(mlir::ElementsAttr elements);

ArrayBuffer<WideNum> getElementsWideNums(mlir::ElementsAttr elements);

void readIntElements(
    mlir::ElementsAttr elements, llvm::MutableArrayRef<int64_t> ints);

void readFPElements(
    mlir::ElementsAttr elements, llvm::MutableArrayRef<double> fps);

} // namespace onnx_mlir