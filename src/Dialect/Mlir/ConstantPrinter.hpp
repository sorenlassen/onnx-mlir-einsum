/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/Attributes.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class AsmPrinter;
}

namespace onnx_mlir {

// Wraps OpAsmPrinter's printAttribute and printOptionalAttrDict methods
// and prints any top level DensifiableElementsAttr as a DenseElementsAttr,
// hiding the DensifiableElementsAttr as an internal representation,
// unless hideDensifiableElementsAttrs(false) is called.
class ConstantPrinter {
public:
  ConstantPrinter(mlir::AsmPrinter &asmPrinter) : asmPrinter(asmPrinter) {}

  // Intended to be called at startup if a command line flag disables hiding.
  static void hideDensifiableElementsAttrs(bool hide) {
    hidingDensifiableElementsAttrs = hide;
  }

  void printAttribute(mlir::Attribute attr);

  void printOptionalAttrDict(llvm::ArrayRef<mlir::NamedAttribute> attrs);

private:
  void printNamedAttribute(mlir::NamedAttribute namedAttr);

  mlir::AsmPrinter &asmPrinter;

  static bool hidingDensifiableElementsAttrs;
};

} // namespace onnx_mlir
