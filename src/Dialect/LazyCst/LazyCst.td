// SPDX-License-Identifier: Apache-2.0

#ifndef LAZYCST_TD
#define LAZYCST_TD

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "src/Interface/DenseLikeElementsAttrInterface.td"

//===----------------------------------------------------------------------===//
// The Lazy Constants Dialect.
//===----------------------------------------------------------------------===//

def LazyCst_Dialect : Dialect {
  let name = "lazycst";
  let description = [{
    The Lazy Elements dialect.
  }];
  let cppNamespace = "::lazycst";
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
  private:
    void registerAttributes(); // Register the builtin Attributes.
  public:
    FileDataManager fileDataManager;
    LazyFunctionManager lazyFunctionManager;
    ConstantFolders constantFolders;
  }];
}

include "src/Dialect/LazyCst/LazyCstAttributes.td"
include "src/Dialect/LazyCst/LazyCstOps.td"

#endif // LAZYCST_TD
