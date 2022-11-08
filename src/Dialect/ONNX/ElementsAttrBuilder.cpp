/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrBuilder.hpp ----------------------===//
//
// Builds DisposableElementsAttr instances.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttrBuilder.hpp"

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposablePool.hpp"
#include "src/Support/Strides.hpp"
#include "src/Support/TypeUtilities.hpp"

#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;

namespace onnx_mlir {

ElementsAttrBuilder::ElementsAttrBuilder(DisposablePool &disposablePool)
    : disposablePool(disposablePool) {}

ElementsAttrBuilder::ElementsAttrBuilder(mlir::MLIRContext *context)
    : disposablePool(*DisposablePool::get(context)) {}

mlir::DisposableElementsAttr ElementsAttrBuilder::fromElementsAttr(
    mlir::ElementsAttr elements) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable;
  if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
    bool isSplat = dense.isSplat();
    std::unique_ptr<llvm::MemoryBuffer> buffer;
    if (dense.getElementType().isInteger(1)) {
      size_t size = isSplat ? 1 : dense.getNumElements();
      std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
          llvm::WritableMemoryBuffer::getNewUninitMemBuffer(size);
      std::copy_n(
          dense.value_begin<bool>(), size, writeBuffer->getBuffer().begin());
      buffer = std::move(writeBuffer);
    } else {
      StringRef s = asStringRef(dense.getRawData());
      buffer = llvm::MemoryBuffer::getMemBuffer(
          s, /*BufferName=*/"", /*RequiresNullTerminator=*/false);
    }
    ArrayRef<int64_t> empty; // empty strides when splat
    return isSplat ? create(dense.getType(), empty, std::move(buffer))
                   : create(dense.getType(), None, std::move(buffer));
  }
  // TODO: consider supporting more ElementsAttr types
  llvm_unreachable("unexpected ElementsAttr instance");
}

mlir::DisposableElementsAttr ElementsAttrBuilder::fromRawBytes(
    ShapedType type, ArrayRef<char> bytes, bool mustCopy) {
  size_t bytewidth = getIntOrFloatByteWidth(type.getElementType());
  assert(bytes.size() == type.getNumElements() * bytewidth &&
         "data size must match type");
  std::unique_ptr<llvm::MemoryBuffer> buffer;
  bool isSplat;
  if (!DenseElementsAttr::isValidRawBuffer(type, bytes, isSplat))
    llvm_unreachable("invalid dense int or fps raw buffer");
  StringRef s = asStringRef(isSplat ? bytes.take_front(bytewidth) : bytes);
  if (mustCopy) {
    buffer = llvm::MemoryBuffer::getMemBufferCopy(s);
  } else {
    buffer = llvm::MemoryBuffer::getMemBuffer(
        s, /*BufferName=*/"", /*RequiresNullTerminator=*/false);
  }
  ArrayRef<int64_t> empty; // empty strides when splat
  return isSplat ? create(type, empty, std::move(buffer))
                 : create(type, None, std::move(buffer));
}

mlir::DisposableElementsAttr ElementsAttrBuilder::transform(
    mlir::DisposableElementsAttr elms, Type transformedElementType,
    Transformer transformer) {
  return elms.transform(*this, transformedElementType, transformer);
}

mlir::DisposableElementsAttr ElementsAttrBuilder::castElementType(
    mlir::DisposableElementsAttr elms, Type newElementType) {
  return elms.castElementType(*this, newElementType);
}

mlir::DisposableElementsAttr ElementsAttrBuilder::transpose(
    mlir::DisposableElementsAttr elms, ArrayRef<uint64_t> perm) {
  return elms.transpose(*this, perm);
}

mlir::DisposableElementsAttr ElementsAttrBuilder::reshape(
    mlir::DisposableElementsAttr elms, ArrayRef<int64_t> reshapedShape) {
  return elms.reshape(*this, reshapedShape);
}

// Broadcasts like the ONNX Expand op.
DisposableElementsAttr ElementsAttrBuilder::expand(
    mlir::DisposableElementsAttr elms, ArrayRef<int64_t> expandedShape) {
  return elms.expand(*this, expandedShape);
}

} // namespace onnx_mlir