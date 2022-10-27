/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ AttributesHelper.cpp ------------------------===//
//
// Attributes helper functions.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/ONNXAttributes.hpp"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"

#include "src/Support/DType.hpp"
#include "src/Dialect/Mlir/ResourcePool.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {
// Always align to the largest possible element type.
// TODO: Consider aligning for SIMD ops.
constexpr size_t ALIGN = std::max(alignof(int64_t), alignof(double));

size_t byteWidth(size_t bitWidth) {
  if (bitWidth == 1)
    return 1;
  constexpr size_t BYTE_BITWIDTH = 8;
  assert(
      bitWidth % BYTE_BITWIDTH == 0 && "non-boolean types must fill out bytes");
  return bitWidth / BYTE_BITWIDTH;
}

void splatterBlob(ShapedType type, AsmResourceBlob &blob) {
  bool isSplat;
  bool isValid =
      DenseElementsAttr::isValidRawBuffer(type, blob.getData(), isSplat);
  assert(isValid && "invalid dense int or fps raw buffer");
  (void)isValid;
  // TODO: change to splat if isSplat
}

} // namespace

ElementsAttr makeDenseIntOrFPElementsAttrFromRawBuffer(
    ShapedType type, ArrayRef<char> bytes) {
  assert(static_cast<size_t>(type.getNumElements()) ==
             bytes.size() /
                 byteWidth(type.getElementType().getIntOrFloatBitWidth()) &&
         "data size must match type");
  if (ResourcePool *resourcePool = ResourcePool::get(type.getContext());
      resourcePool && resourcePool->isActive()) {
    AsmResourceBlob blob =
        HeapAsmResourceBlob::allocateAndCopy(bytes, ALIGN, false);
    splatterBlob(type, blob);
    DenseResourceElementsHandle r =
        resourcePool->createResource(std::move(blob));
    return DenseResourceElementsAttr::get(type, r);
  } else {
    return DenseElementsAttr::getFromRawBuffer(type, bytes);
  }
}

ElementsAttr makeDenseIntOrFPElementsAttrWithRawBuffer(
    ShapedType type, FillDenseRawBufferFn fill) {
  size_t size = type.getNumElements() *
                byteWidth(type.getElementType().getIntOrFloatBitWidth());
  if (ResourcePool *resourcePool = ResourcePool::get(type.getContext());
      resourcePool && resourcePool->isActive()) {
    AsmResourceBlob blob = HeapAsmResourceBlob::allocate(size, ALIGN);
    fill(blob.getMutableData());
    splatterBlob(type, blob);
    DenseResourceElementsHandle r =
        resourcePool->createResource(std::move(blob));
    return DenseResourceElementsAttr::get(type, r);
  } else {
    std::vector<char> bytes(size, 0);
    fill(bytes);
    return DenseElementsAttr::getFromRawBuffer(type, bytes);
  }
}

ArrayRef<char> getDenseIntOrFPRawData(ElementsAttr elements) {
  if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
    ArrayRef<char> raw = dense.getRawData();
    // raw is either a single splat value or a whole array.
    ShapedType type = elements.getType();
    size_t w = byteWidth(type.getElementType().getIntOrFloatBitWidth());
    if (dense.isSplat()) {
      assert(raw.size() == w);
    } else {
      assert(raw.size() == type.getNumElements() * w);
    }
    return raw;
  }
  if (auto x = elements.dyn_cast<DenseResourceElementsAttr>())
    return x.getRawHandle().getResource()->getBlob()->getData();
  llvm_unreachable("unexpected ElementsAttr instance");
}

template <typename D>
struct ReadIntsOrFPs {
  template <typename DTy, typename... Args>
  struct Read {
    using S = typename DTy::type;
    static void eval(ArrayRef<char> src, MutableArrayRef<D> dst) {
      fillOrTransform(castArrayRef<S>(src), dst,
          [](S v) { return static_cast<D>(DTy::unpack(v)); });
    }
  };
};

void readDenseInts(ElementsAttr elements, MutableArrayRef<int64_t> ints) {
  ArrayRef<char> src = getDenseIntOrFPRawData(elements);
  dispatchInt<ReadIntsOrFPs<int64_t>::template Read, void>::eval(
      elements.getElementType(), src, ints);
}

void readDenseFPs(ElementsAttr elements, MutableArrayRef<double> fps) {
  ArrayRef<char> src = getDenseIntOrFPRawData(elements);
  dispatchFP<ReadIntsOrFPs<double>::template Read, void>::eval(
      elements.getElementType(), src, fps);
}

DenseElementsAttr toDenseElementsAttribute(ElementsAttr elements) {
  if (auto dense = elements.dyn_cast<DenseElementsAttr>())
    return dense;
  if (auto resource = elements.dyn_cast<DenseResourceElementsAttr>()) {
    ArrayRef<char> bytes =
        resource.getRawHandle().getResource()->getBlob()->getData();
    return DenseElementsAttr::getFromRawBuffer(resource.getType(), bytes);
  }
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable.toDenseElementsAttr();
  llvm_unreachable("unexpected ElementsAttr instance"); // TODO: read data from
                                                        // elements.getValues()
}

} // namespace onnx_mlir