/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXAttributes.cpp -- ONNX attributes implementation --------===//
//
// ONNX attributes implementation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXAttributes.hpp"

#include "llvm/Support/raw_os_ostream.h"

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)

namespace mlir {

size_t detail::uniqueNumber() {
  static std::atomic<size_t> counter{0};
  return ++counter;
}

static void printDenseFloatElement(const APFloat &value, raw_ostream &os,
                                 Type type) {
  FloatAttr::get(type, value).print(os, /*elideType=*/true);
}

// Copied from mlir/lib/IR/AsmPrinter.cpp:
static void printDenseIntElement(const APInt &value, raw_ostream &os,
                                 Type type) {
  if (type.isInteger(1))
    os << (value.getBoolValue() ? "true" : "false");
  else
    value.print(os, !type.isUnsignedInteger());
}

// Copied from mlir/lib/IR/AsmPrinter.cpp:
static void
printDenseElementsAttrImpl(bool isSplat, ShapedType type, raw_ostream &os,
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

// adapted from AsmPrinter::Impl::printDenseIntOrFPElementsAttr:
void detail::printIntOrFPElementsAttrAsDense(ElementsAttr attr, raw_ostream &os) {
  auto type = attr.getType();
  auto elementType = type.getElementType();
  os << "dense<";
  if (elementType.isIntOrIndex()) {
    auto valueIt = attr.value_begin<APInt>();
    printDenseElementsAttrImpl(attr.isSplat(), type, os, [&](unsigned index) {
      printDenseIntElement(*(valueIt + index), os, elementType);
    });
  } else {
    assert(elementType.isa<FloatType>() && "unexpected element type");
    auto valueIt = attr.value_begin<APFloat>();
    printDenseElementsAttrImpl(attr.isSplat(), type, os, [&](unsigned index) {
      printDenseFloatElement(*(valueIt + index), os, elementType);
    });
  }
  os << '>';
}

} // namespace mlir