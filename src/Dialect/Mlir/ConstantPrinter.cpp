/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/Mlir/ConstantPrinter.hpp"
#include "src/Interface/DenseLikeElementsAttrInterface.hpp"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

//===----------------------------------------------------------------------===//
// Helpers adapted from corresponding methods in mlir/lib/AsmParser/Parser.cpp
//===----------------------------------------------------------------------===//

void printAsDenseElementsAttr(
    AsmPrinter &asmPrinter, DenseLikeElementsAttrInterface densifiable) {
  // It would be ideal if we could read the asmPrinter flags from asmPrinter
  // instead of constructing them here, as asmPrinter may have been constructed
  // with an override of elideLargeElementsAttrs which we cannot see here.
  // Oh well, at least OpPrintingFlags().shouldElideElementsAttr(ElementsAttr)
  // lets us respect the --mlir-elide-elementsattrs-if-larger command line flag.
  static OpPrintingFlags printerFlags{};
  if (!densifiable.isSplat() &&
      printerFlags.shouldElideElementsAttr(densifiable)) {
    // In this special case it's easy to avoid conversion to DenseElementsAttr.
    asmPrinter << "dense<__elided__> : " << densifiable.getType();
    return;
  }
  // Take shortcut by first converting to DenseElementsAttr.
  // NOTE: This creates a copy which is never garbage collected. This is not
  // only slow but also defeats the garbage collection benefits of
  // DisposableElementsAttr at al, depending on when the printing takes place.
  // (The print at the end of onnx-mlir-opt in lit tests is ok.)
  asmPrinter.printAttribute(densifiable.toDenseElementsAttr());
  // TODO: Do the work to print without constructing DenseElementsAttr.
}

} // namespace

void ConstantPrinter::printAttribute(Attribute attr) {
  if (hidingDenseLikeElementsAttrs) {
    if (auto densifiable = attr.dyn_cast<DenseLikeElementsAttrInterface>()) {
      printAsDenseElementsAttr(asmPrinter, densifiable);
      return;
    }
  }
  asmPrinter.printAttribute(attr);
}

void ConstantPrinter::printOptionalAttrDict(ArrayRef<NamedAttribute> attrs) {
  // If there are no attributes, then there is nothing to be done.
  if (attrs.empty())
    return;

  // Otherwise, print them all out in braces.
  asmPrinter << " {";
  llvm::interleaveComma(attrs, asmPrinter.getStream(),
      [&](NamedAttribute attr) { printNamedAttribute(attr); });
  asmPrinter << '}';
}

void ConstantPrinter::printNamedAttribute(NamedAttribute namedAttr) {
  // Print the name without quotes if possible.
  asmPrinter.printKeywordOrString(namedAttr.getName().strref());

  // Pretty printing elides the attribute value for unit attributes.
  if (namedAttr.getValue().isa<UnitAttr>())
    return;

  asmPrinter << " = ";
  printAttribute(namedAttr.getValue());
}

bool ConstantPrinter::hidingDenseLikeElementsAttrs = true;

} // namespace onnx_mlir
