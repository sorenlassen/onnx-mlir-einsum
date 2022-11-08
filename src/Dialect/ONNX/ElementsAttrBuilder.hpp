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

  mlir::DisposableElementsAttr transform(
      mlir::DisposableElementsAttr __type_pack_element_alias,
      mlir::Type transformedElementType, Transformer transformer);

  mlir::DisposableElementsAttr castElementType(
      mlir::DisposableElementsAttr elms, mlir::Type newElementType);

  mlir::DisposableElementsAttr transpose(
      mlir::DisposableElementsAttr elms, llvm::ArrayRef<uint64_t> perm);

  mlir::DisposableElementsAttr reshape(
      mlir::DisposableElementsAttr elms, llvm::ArrayRef<int64_t> reshapedShape);

  // Broadcasts like the ONNX Expand op.
  mlir::DisposableElementsAttr expand(
      mlir::DisposableElementsAttr elms, llvm::ArrayRef<int64_t> expandedShape);

private:
  DisposablePool &disposablePool;
};

} // namespace onnx_mlir
