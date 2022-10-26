// SPDX-License-Identifier: Apache-2.0

//===---------- DisposableElementsAttrInterface.td -----------*- tablegen -===//
//
// Defines some methods shared by all DisposableElementsAttr<T> template
// instantiations.
//
//===----------------------------------------------------------------------===//

#ifndef DISPOSABLE_ELEMENTS_ATTR_INTERFACE
#define DISPOSABLE_ELEMENTS_ATTR_INTERFACE

include "mlir/IR/AttrTypeBase.td"

def DisposableElementsAttrInterface : AttrInterface<"DisposableElementsAttrInterface"> {
  let description = [{
    Interface to access some methods shared by all DisposableElementsAttr<T>
    template instantiations.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "DenseElementsAttr", "toDenseElementsAttr">
  ];
  
  let cppNamespace = "::mlir";
}

#endif // DISPOSABLE_ELEMENTS_ATTR_INTERFACE
