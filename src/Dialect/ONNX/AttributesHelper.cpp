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
#include "src/Dialect/ONNX/ONNXAttributes.hpp"
#include "src/Support/DType.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {
// Always align to the largest possible element type.
// TODO: Consider aligning for SIMD ops.
constexpr size_t ALIGN = std::max(alignof(int64_t), alignof(double));

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

} // namespace

ElementsAttr makeDenseIntOrFPElementsAttrFromRawBuffer(
    ShapedType type, ArrayRef<char> bytes, bool mustCopy) {
  // llvm::errs() << "makeDenseIntOrFPElementsAttrFromRawBuffer " << type << ","
  // << bytes.size() << "\n";
  unsigned bytewidth = bytewidthOfIntOrFPType(type.getElementType());
  assert(bytes.size() == type.getNumElements() * bytewidth &&
         "data size must match type");
#if 1
  std::unique_ptr<llvm::MemoryBuffer> buffer;
  bool isSplat = splatterBuffer(type, bytes);
  if (mustCopy) {
    StringRef s(bytes.data(), isSplat ? bytewidth : bytes.size());
    buffer = llvm::MemoryBuffer::getMemBufferCopy(s);
  } else {
    StringRef s(bytes.data(), bytes.size());
    buffer = llvm::MemoryBuffer::getMemBuffer(s, /*BufferName=*/"", /*RequiresNullTerminator=*/false);
  }
  return DisposableElementsAttr::get(type, std::move(buffer));
#else
  if (ResourcePool *resourcePool = ResourcePool::get(type.getContext());
      resourcePool && resourcePool->isActive()) {
    AsmResourceBlob blob =
      mustCopy
      ? HeapAsmResourceBlob::allocateAndCopy(bytes, ALIGN, /*dataIsMutable=*/true)
      : AsmResourceBlob(bytes, ALIGN, /*deleter=*/nullptr, /*dataIsMutable=*/false);
    splatterBlob(type, blob, /*dataIsMutable=*/mustCopy);
    DenseResourceElementsHandle r =
        resourcePool->createResource(std::move(blob));
    return DenseResourceElementsAttr::get(type, r);
  } else {
    return DenseElementsAttr::getFromRawBuffer(type, bytes);
  }
#endif
}

ElementsAttr makeDenseIntOrFPElementsAttrWithRawBuffer(
    ShapedType type, FillDenseRawBufferFn fill) {
  size_t size =
      type.getNumElements() * bytewidthOfIntOrFPType(type.getElementType());
  // llvm::errs() << "makeDenseIntOrFPElementsAttrWithRawBuffer " << type << ","
  // << size << "\n";
#if 1
  std::unique_ptr<llvm::WritableMemoryBuffer> buffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(size);
  fill(buffer->getBuffer());
  return DisposableElementsAttr::get(type, std::move(buffer));
#else
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
#endif
}

RawBuffer getDenseIntOrFPRawData(ElementsAttr elements) {
  // llvm::errs() << "getDenseIntOrFPRawData " << elements.getType() << "\n";
  if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
    ArrayRef<char> raw = dense.getRawData();
    // raw is either a single splat value or a whole array.
    ShapedType type = elements.getType();
    size_t w = bytewidthOfIntOrFPType(type.getElementType());
    if (dense.isSplat()) {
      assert(raw.size() == w);
    } else {
      assert(raw.size() == type.getNumElements() * w);
    }
    return raw;
  }
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>()) {
    return disposable.getRawBuffer();
  }
  if (auto resrc = elements.dyn_cast<DenseResourceElementsAttr>())
    return resrc.getRawHandle().getResource()->getBlob()->getData();
  llvm_unreachable("unexpected ElementsAttr instance");
}

template <typename D>
struct ReadIntsOrFPs {
  template <typename DTy, typename... Args>
  struct Read {
    using S = typename DTy::type;
    static void eval(ArrayRef<char> src, MutableArrayRef<D> dst) {
      fillOrTransform(
          castArrayRef<S>(src), dst, [](S v) { return static_cast<D>(v); });
    }
  };
};

void readDenseInts(ElementsAttr elements, MutableArrayRef<int64_t> ints) {
  // llvm::errs() << "readDenseInts " << elements.getType() << "\n";
  RawBuffer src = getDenseIntOrFPRawData(elements);
  dispatchInt<ReadIntsOrFPs<int64_t>::template Read, void>::eval(
      elements.getElementType(), src.get(), ints);
}

void readDenseFPs(ElementsAttr elements, MutableArrayRef<double> fps) {
  // llvm::errs() << "readDenseFPs " << elements.getType() << "\n";
  RawBuffer src = getDenseIntOrFPRawData(elements);
  dispatchFP<ReadIntsOrFPs<double>::template Read, void>::eval(
      elements.getElementType(), src.get(), fps);
}

DenseElementsAttr toDenseElementsAttribute(ElementsAttr elements) {
  llvm::errs() << "toDenseElementsAttribute " << elements.getType() << "\n";
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
} // namespace

void printIntOrFPElementsAttrAsDense(ElementsAttr attr, raw_ostream &os) {
  printIntOrFPElementsAttrAsDenseWithoutType(attr, os);
  os << " : " << attr.getType();
}

} // namespace onnx_mlir