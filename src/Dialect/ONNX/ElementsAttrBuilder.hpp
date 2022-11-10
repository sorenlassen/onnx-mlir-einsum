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

  template <typename T>
  using Filler = std::function<void(llvm::MutableArrayRef<T>)>;

  mlir::DisposableElementsAttr fromRawBytes(mlir::ShapedType type,
      DType bufferDType, llvm::ArrayRef<char> bytes, bool mustCopy);

  mlir::DisposableElementsAttr fromRawBytes(mlir::ShapedType type,
      DType bufferDType, const Filler<char> &bytesFiller);

  template <typename T>
  mlir::DisposableElementsAttr fromArray(
      mlir::ShapedType type, llvm::ArrayRef<T> array, bool mustCopy);

  template <typename T>
  mlir::DisposableElementsAttr fromArray(
      mlir::ShapedType type, const Filler<T> &typedFiller);

  using Transformer = Filler<WideNum>;

  template <typename UnaryFunction = std::function<WideNum(WideNum)>>
  static Transformer functionTransformer(UnaryFunction fun);

  mlir::DisposableElementsAttr transform(mlir::DisposableElementsAttr elms,
      mlir::Type transformedElementType, Transformer transformer);

  template <typename BinaryCombiner = std::function<WideNum(WideNum, WideNum)>>
  mlir::DisposableElementsAttr combine(mlir::DisposableElementsAttr lhs,
      mlir::DisposableElementsAttr rhs, mlir::ShapedType combinedType,
      BinaryCombiner combiner);

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
//
// TODO: move so standalone ElementsAttrBuilder.inc source file
//       like ShapeHelper.inc
//===----------------------------------------------------------------------===//

template <typename T>
mlir::DisposableElementsAttr ElementsAttrBuilder::fromArray(
    mlir::ShapedType type, llvm::ArrayRef<T> array, bool mustCopy) {
  return fromRawBytes(type, toDType<T>, castArrayRef<char>(array), mustCopy);
}

template <typename T>
mlir::DisposableElementsAttr ElementsAttrBuilder::fromArray(
    mlir::ShapedType type, const Filler<T> &typedFiller) {
  return fromRawBytes(
      type, toDType<T>, [&typedFiller](llvm::MutableArrayRef<char> bytes) {
        typedFiller(castArrayRef<T>(bytes));
      });
}

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

template <typename BinaryCombiner>
mlir::DisposableElementsAttr ElementsAttrBuilder::combine(
    mlir::DisposableElementsAttr lhs, mlir::DisposableElementsAttr rhs,
    mlir::ShapedType combinedType, BinaryCombiner combiner) {
  if (lhs.isSplat()) {
    WideNum lhsNum = lhs.getSplatValue<WideNum>();
    return transformAndExpand(rhs, combinedType,
        functionTransformer(
            [lhsNum, combiner = std::forward<BinaryCombiner>(combiner)](
                WideNum n) { return combiner(lhsNum, n); }));
  }
  if (rhs.isSplat()) {
    WideNum rhsNum = rhs.getSplatValue<WideNum>();
    return transformAndExpand(lhs, combinedType,
        functionTransformer(
            [rhsNum, combiner = std::forward<BinaryCombiner>(combiner)](
                WideNum n) { return combiner(n, rhsNum); }));
  }

  // TODO: use fromArray(Filler<T>) to reduce code duplication
  size_t size = combinedType.getNumElements() * sizeof(WideNum);
  std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(size);
  auto dstNums = castMutableArrayRef<WideNum>(writeBuffer->getBuffer());
  auto shape = combinedType.getShape();
  auto dstStrides = getDefaultStrides(shape);
  Strided<llvm::MutableArrayRef<WideNum>> dst{dstStrides, dstNums};
  ArrayBuffer<WideNum> lhsNums = lhs.getBufferAsWideNums();
  Strided<llvm::ArrayRef<WideNum>> lhsStrided{lhs.getStrides(), lhsNums.get()};
  ArrayBuffer<WideNum> rhsNums = rhs.getBufferAsWideNums();
  Strided<llvm::ArrayRef<WideNum>> rhsStrided{rhs.getStrides(), rhsNums.get()};
  transformAndRestrideTwoWideArrays(
      shape, lhsStrided, rhsStrided, dst, combiner);

  DType dtype = dtypeOfMlirType(combinedType.getElementType());
  DType bufferDType = wideDTypeOfDType(dtype);
  return create(combinedType, std::move(writeBuffer),
      llvm::makeArrayRef(dstStrides), bufferDType);
}

} // namespace onnx_mlir
