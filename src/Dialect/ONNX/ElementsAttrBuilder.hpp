/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrBuilder.hpp ----------------------===//
//
// Builds DisposableElementsAttr instances.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposablePool.hpp"
#include "src/Support/WideNum.hpp"

namespace onnx_mlir {

class ElementsAttrBuilder {
public:
  ElementsAttrBuilder(DisposablePool &disposablePool);

  ElementsAttrBuilder(mlir::MLIRContext *context);

  // Create a DisposableElementsAttr and put it in the pool.
  template <typename... Args>
  mlir::DisposableElementsAttr create(Args &&...args) {
    auto d = mlir::DisposableElementsAttr::get(std::forward<Args>(args)...);
    disposablePool.insert(d);
    return d;
  }

  // Makes a DisposableElementsAttr that points to elements' raw data if
  // elements is DenseElementsAttr, except if the element type is bool, then
  // it makes a deep copy because DisposableElementsAttr doesn't bit pack bools.
  mlir::DisposableElementsAttr fromElementsAttr(mlir::ElementsAttr elements);

  mlir::DisposableElementsAttr fromRawBytes(
      mlir::ShapedType type, llvm::ArrayRef<char> bytes, bool mustCopy);

  using Transformer = std::function<void(llvm::MutableArrayRef<WideNum>)>;

  template <typename UnaryFunction = std::function<WideNum(WideNum)>>
  static Transformer functionTransformer(UnaryFunction fun);

  mlir::DisposableElementsAttr transform(mlir::DisposableElementsAttr elms,
      mlir::Type transformedElementType, Transformer transformer);

#if 0
  template <typename BinaryCombiner = std::function<WideNum(WideNum, WideNum)>>
  DisposableElementsAttr combine(
      mlir::DisposableElementsAttr lhs, mlir::DisposableElementsAttr rhs,
      mlir::ShapedType combinedType, BinaryCombiner combiner);
#endif

  mlir::DisposableElementsAttr castElementType(
      mlir::DisposableElementsAttr elms, mlir::Type newElementType);

  mlir::DisposableElementsAttr transpose(
      mlir::DisposableElementsAttr elms, llvm::ArrayRef<uint64_t> perm);

  mlir::DisposableElementsAttr reshape(
      mlir::DisposableElementsAttr elms, llvm::ArrayRef<int64_t> reshapedShape);

  // Broadcasts like the ONNX Expand op.
  mlir::DisposableElementsAttr expand(
      mlir::DisposableElementsAttr elms, llvm::ArrayRef<int64_t> expandedShape);

  mlir::DisposableElementsAttr transformAndExpand(
      mlir::DisposableElementsAttr elms, mlir::ShapedType resultType,
      Transformer transformer) {
    auto transformed =
        transform(elms, resultType.getElementType(), transformer);
    return expand(transformed, resultType.getShape());
  }

private:
  DisposablePool &disposablePool;
};

//===----------------------------------------------------------------------===//
// Deferred Method Definitions
//===----------------------------------------------------------------------===//

/*static*/
template <typename UnaryFunction>
ElementsAttrBuilder::Transformer ElementsAttrBuilder::functionTransformer(
    UnaryFunction fun) {
  return [fun = std::forward<UnaryFunction>(fun)](
             llvm::MutableArrayRef<WideNum> data) -> void {
    for (WideNum &n : data)
      n = fun(n);
  };
}

#if 0
template <typename BinaryCombiner>
DisposableElementsAttr ElementsAttrBuilder::combine(
    mlir::DisposableElementsAttr lhs, mlir::DisposableElementsAttr rhs,
    mlir::ShapedType combinedType, BinaryCombiner combiner) {
  if (lhs.isSplat()) {
    WideNum lhsNum = lhs.getSplatValue<WideNum>();
    return elementsBuilder.transformAndExpand(rhs, combinedType,
functionTransformer([lhsNum](WideNum n) { return combiner(lhsNum, n); }));
  }
  if (rhs.isSplat()) {
    WideNum rhsNum = rhs.getSplatValue<WideNum>();
    return elementsBuilder.transformAndExpand(lhs, combinedType,
functionTransformer([rhsNum](WideNum n) { return combiner(n, rhsNum); }));
  }
  ArrayBuffer<WideNum> lhsNums = lhs.getBufferAsWideNums();
  ArrayBuffer<WideNum> rhsNums = rhs.getBufferAsWideNums();
    std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
        llvm::WritableMemoryBuffer::getNewUninitMemBuffer(size);
    MutableArrayBuffer<WideNum> dstNums = castMutableArrayRef<WideNum>(writeBuffer->getBuffer());
    // TODO fill in dstNums from lhsNums, rhsNums, combiner
    DType dtype = dtypeOfMlirType(combinedType.getElementType());
    DisposableElementsAttr::Properties properties{
      .dtype = dtype,
      .bufferDType = wideDTypeOfDType(dtype),
      .isContiguous = true,
      .isTransformed = false
    };
    DisposableElementsAttr::Buffer buffer = std::move(writeBuffer);
return create(combinedType, None, properties, buffer);
}
#endif

} // namespace onnx_mlir
