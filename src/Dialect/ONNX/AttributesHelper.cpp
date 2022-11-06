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
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/Support/raw_os_ostream.h"

#include "src/Dialect/Mlir/ResourcePool.hpp"
#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposablePool.hpp"
#include "src/Support/Arrays.hpp"
#include "src/Support/DType.hpp"
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

bool splatterBlob(ShapedType type, AsmResourceBlob &blob, bool dataIsMutable) {
  // TODO: change blob to splat if returns true and dataIsMutable
  return splatterBuffer(type, blob.getData());
}

// Always align to the largest possible element type.
// TODO: Consider aligning for SIMD ops.
constexpr size_t ALIGN = std::max(alignof(int64_t), alignof(double));

} // namespace

mlir::DenseElementsAttr makeDenseElementsAttrFromRawBytes(
    mlir::ShapedType type, llvm::ArrayRef<char> bytes) {
  if (type.getElementType().isInteger(1)) {
    // don't use getFromRawBuffer which requires bit packing
    return DenseElementsAttr::get(type, castArrayRef<bool>(bytes));
  }
  return DenseElementsAttr::getFromRawBuffer(type, bytes);
}

ElementsAttr makeElementsAttrFromRawBytes(
    ShapedType type, ArrayRef<char> bytes, bool mustCopy) {
  size_t bytewidth = getIntOrFloatByteWidth(type.getElementType());
  assert(bytes.size() == type.getNumElements() * bytewidth &&
         "data size must match type");
  if (DisposablePool *disposablePool = DisposablePool::get(type.getContext());
      disposablePool && disposablePool->isActive()) {
    std::unique_ptr<llvm::MemoryBuffer> buffer;
    bool isSplat = splatterBuffer(type, bytes);
    if (mustCopy) {
      StringRef s = asStringRef(isSplat ? bytes.take_front(bytewidth) : bytes);
      buffer = llvm::MemoryBuffer::getMemBufferCopy(s);
    } else {
      StringRef s = asStringRef(bytes);
      buffer = llvm::MemoryBuffer::getMemBuffer(
          s, /*BufferName=*/"", /*RequiresNullTerminator=*/false);
    }
    return disposablePool->createElementsAttr(type, isSplat, std::move(buffer));
  }
  if (ResourcePool *resourcePool = ResourcePool::get(type.getContext());
      resourcePool && resourcePool->isActive()) {
    bool isSplat = splatterBuffer(type, bytes);
    AsmResourceBlob blob =
        mustCopy ? HeapAsmResourceBlob::allocateAndCopy(
                       isSplat ? bytes.take_front(bytewidth) : bytes, ALIGN,
                       /*dataIsMutable=*/true)
                 : AsmResourceBlob(bytes, ALIGN, /*deleter=*/nullptr,
                       /*dataIsMutable=*/false);
    DenseResourceElementsHandle r =
        resourcePool->createResource(std::move(blob));
    return DenseResourceElementsAttr::get(type, r);
  }
  return makeDenseElementsAttrFromRawBytes(type, bytes);
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
    return disposablePool->createElementsAttr(type, std::move(buffer));
  }
  if (ResourcePool *resourcePool = ResourcePool::get(type.getContext());
      resourcePool && resourcePool->isActive()) {
    AsmResourceBlob blob = HeapAsmResourceBlob::allocate(size, ALIGN);
    filler(blob.getMutableData());
    splatterBlob(type, blob, /*dataIsMutable=*/true);
    DenseResourceElementsHandle r =
        resourcePool->createResource(std::move(blob));
    return DenseResourceElementsAttr::get(type, r);
  }
  SmallVector<char> bytes;
  bytes.resize_for_overwrite(size);
  filler(bytes);
  return makeDenseElementsAttrFromRawBytes(type, bytes);
}

ArrayBuffer<char> getElementsRawBytes(ElementsAttr elements) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable.getRawBytes();
  if (auto denseResrc = elements.dyn_cast<DenseResourceElementsAttr>())
    return denseResrc.getRawHandle().getResource()->getBlob()->getData();
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
  if (auto denseResrc = elements.dyn_cast<DenseResourceElementsAttr>()) {
    rawData = denseResrc.getRawHandle().getResource()->getBlob()->getData();
  } else if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
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
    using S = CppType<dtype>;
    fillOrTransform(castArrayRef<S>(src.get()), dst,
        [](S v) { return WideNum::from<S>(toDType<S>, v); });
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

DenseElementsAttr toDenseElementsAttribute(ElementsAttr elements) {
  llvm::errs() << "toDenseElementsAttribute " << elements.getType() << "\n";
  if (auto dense = elements.dyn_cast<DenseElementsAttr>())
    return dense;
  if (auto denseResrc = elements.dyn_cast<DenseResourceElementsAttr>()) {
    ArrayRef<char> bytes =
        denseResrc.getRawHandle().getResource()->getBlob()->getData();
    return DenseElementsAttr::getFromRawBuffer(denseResrc.getType(), bytes);
  }
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable.toDenseElementsAttr();
  // TODO: read data from elements.getValues() instead of giving up
  llvm_unreachable("unexpected ElementsAttr instance");
}

namespace {
void printDenseFloatElement(const APFloat &value, raw_ostream &os, Type type) {
  FloatAttr::get(type, value).print(os, /*elideType=*/true);
}

// Copied from mlir/lib/IR/AsmPrinter.cpp:
void printDenseIntElement(const APInt &value, raw_ostream &os, Type type) {
  if (type.isInteger(1))
    os << (value.getBoolValue() ? "true" : "false");
  else
    value.print(os, !type.isUnsignedInteger());
}

// Copied from mlir/lib/IR/AsmPrinter.cpp:
void printDenseElementsAttrImpl(bool isSplat, ShapedType type, raw_ostream &os,
    function_ref<void(unsigned)> printEltFn) {
  // Special case for 0-d and splat tensors.
  if (isSplat)
    return printEltFn(0);

  // Special case for degenerate tensors.
  auto numElements = type.getNumElements();
  if (numElements == 0)
    return;

  // We use a mixed-radix counter to iterate through the shape. When we bump a
  // non-least-significant digit, we emit a close bracket. When we next emit an
  // element we re-open all closed brackets.

  // The mixed-radix counter, with radices in 'shape'.
  int64_t rank = type.getRank();
  SmallVector<unsigned, 4> counter(rank, 0);
  // The number of brackets that have been opened and not closed.
  unsigned openBrackets = 0;

  auto shape = type.getShape();
  auto bumpCounter = [&] {
    // Bump the least significant digit.
    ++counter[rank - 1];
    // Iterate backwards bubbling back the increment.
    for (unsigned i = rank - 1; i > 0; --i)
      if (counter[i] >= shape[i]) {
        // Index 'i' is rolled over. Bump (i-1) and close a bracket.
        counter[i] = 0;
        ++counter[i - 1];
        --openBrackets;
        os << ']';
      }
  };

  for (unsigned idx = 0, e = numElements; idx != e; ++idx) {
    if (idx != 0)
      os << ", ";
    while (openBrackets++ < rank)
      os << '[';
    openBrackets = rank;
    printEltFn(idx);
    bumpCounter();
  }
  while (openBrackets-- > 0)
    os << ']';
}

template <typename Iterator>
bool checkIfSplat(ElementsAttr attr, Iterator valueIt) {
  if (attr.isSplat())
    return true;
  if (attr.isa<DenseElementsAttr>()) {
    // DenseElementsAttr always reports accurate isSplat() so no need to check
    // contents when isSplat() returned false.
    return false;
  }
  int64_t numElements = attr.getNumElements();
  if (numElements == 0)
    return false;
  auto first = *valueIt;
  for (int64_t i = 1; i < numElements; ++i) {
    if (first != *++valueIt)
      return false;
  }
  return true;
}
} // namespace

// adapted from AsmPrinter::Impl::printDenseIntOrFPElementsAttr:
void printIntOrFPElementsAttrAsDenseWithoutType(
    ElementsAttr attr, raw_ostream &os) {
  auto type = attr.getType();
  auto elementType = type.getElementType();
  os << "dense<";
  if (elementType.isIntOrIndex()) {
    auto valueIt = attr.value_begin<APInt>();
    bool isSplat = checkIfSplat(attr, valueIt);
    printDenseElementsAttrImpl(isSplat, type, os, [&](unsigned index) {
      printDenseIntElement(*(valueIt + index), os, elementType);
    });
  } else {
    assert(elementType.isa<FloatType>() && "unexpected element type");
    auto valueIt = attr.value_begin<APFloat>();
    bool isSplat = checkIfSplat(attr, valueIt);
    printDenseElementsAttrImpl(isSplat, type, os, [&](unsigned index) {
      printDenseFloatElement(*(valueIt + index), os, elementType);
    });
  }
  os << '>';
}

void printIntOrFPElementsAttrAsDense(ElementsAttr attr, raw_ostream &os) {
  printIntOrFPElementsAttrAsDenseWithoutType(attr, os);
  os << " : " << attr.getType();
}

} // namespace onnx_mlir