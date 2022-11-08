/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ AttributesHelper.cpp ------------------------===//
//
// Attributes helper functions.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/AttributesHelper.hpp"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_os_ostream.h"

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposablePool.hpp"
#include "src/Dialect/ONNX/ElementsAttrBuilder.hpp"
#include "src/Support/Arrays.hpp"
#include "src/Support/DType.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "src/Support/WideNum.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

bool splatterBuffer(ShapedType type, ArrayRef<char> buffer) {
  bool isSplat;
  if (!DenseElementsAttr::isValidRawBuffer(type, buffer, isSplat))
    llvm_unreachable("invalid dense int or fps raw buffer");
  return isSplat;
}

DenseElementsAttr makeDenseElementsAttrFromRawBytes(
    ShapedType type, ArrayRef<char> bytes) {
  size_t bytewidth = getIntOrFloatByteWidth(type.getElementType());
  assert(bytes.size() == type.getNumElements() * bytewidth &&
         "data size must match type");
  if (type.getElementType().isInteger(1)) {
    // don't use getFromRawBuffer which requires bit packing
    return DenseElementsAttr::get(type, castArrayRef<bool>(bytes));
  }
  return DenseElementsAttr::getFromRawBuffer(type, bytes);
}

} // namespace

DenseElementsAttr toDenseElementsAttr(ElementsAttr elements) {
  if (auto dense = elements.dyn_cast<DenseElementsAttr>())
    return dense;
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>()) {
    ArrayBuffer<char> bytes = disposable.getRawBytes();
    return makeDenseElementsAttrFromRawBytes(disposable.getType(), bytes.get());
  }
  // TODO: consider reading data from elements.getValues() instead of giving up
  llvm_unreachable("unexpected ElementsAttr instance");
}

ElementsAttr makeElementsAttrWithRawBytesFiller(
    ShapedType type, RawBytesFiller filler) {
  size_t size =
      type.getNumElements() * getIntOrFloatByteWidth(type.getElementType());
  if (DisposablePool *disposablePool = DisposablePool::get(type.getContext());
      disposablePool && disposablePool->isActive()) {
    std::unique_ptr<llvm::WritableMemoryBuffer> buffer =
        llvm::WritableMemoryBuffer::getNewUninitMemBuffer(size);
    filler(buffer->getBuffer());
    bool isSplat = splatterBuffer(type, buffer->getBuffer());
    (void)isSplat; // TODO: decide whether to truncate buffer if isSplat
    return ElementsAttrBuilder(*disposablePool)
        .create(type, None, std::move(buffer));
  }
  SmallVector<char> bytes;
  bytes.resize_for_overwrite(size);
  filler(bytes);
  return makeDenseElementsAttrFromRawBytes(type, bytes);
}

ArrayBuffer<char> getElementsRawBytes(ElementsAttr elements) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable.getRawBytes();
  if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
    if (dense.getElementType().isInteger(1)) {
      // bool is bit packed in dense, so we copy it out
      size_t size = dense.isSplat() ? 1 : dense.getNumElements();
      ArrayBuffer<char>::Vector vec;
      vec.resize_for_overwrite(size);
      std::copy_n(dense.value_begin<bool>(), size, vec.begin());
      return std::move(vec);
    }
    return dense.getRawData(); // Single splat value or a full array.
  }
  llvm_unreachable("unexpected ElementsAttr instance");
}

ArrayBuffer<WideNum> getElementsWideNums(ElementsAttr elements) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable.getWideNums();
  ArrayRef<char> rawData;
  if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
    // TODO: copy out contents if bool, because raw data is bit packed
    assert(!dense.getElementType().isInteger(1) && "bool unsupported");
    rawData = dense.getRawData(); // Single splat value or a full array.
  } else {
    llvm_unreachable("unexpected ElementsAttr instance");
  }
  return widenOrReturnArray(elements.getElementType(), rawData);
}

namespace {

void readElements(ElementsAttr elements, MutableArrayRef<WideNum> dst) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>()) {
    disposable.readElements(dst);
    return;
  }
  ArrayBuffer<char> src = getElementsRawBytes(elements);
  dispatchByMlirType(elements.getElementType(), [&](auto dtype) {
    using W = WideDType<dtype>;
    fillOrTransform(
        castArrayRef<typename W::narrowtype>(src.get()), dst, W::widen);
  });
}

} // namespace

void readIntElements(ElementsAttr elements, MutableArrayRef<int64_t> ints) {
  assert(elements.getType().getElementType().isa<IntegerType>());
  readElements(elements, castMutableArrayRef<WideNum>(ints));
}

void readFPElements(ElementsAttr elements, MutableArrayRef<double> fps) {
  assert(elements.getType().getElementType().isa<FloatType>());
  readElements(elements, castMutableArrayRef<WideNum>(fps));
}

} // namespace onnx_mlir
