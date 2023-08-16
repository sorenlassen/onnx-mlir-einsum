// SPDX-License-Identifier: Apache-2.0

#ifdef DENSIFIABLE_ELEMENTS_ATTR_INTERFACE
#else
#define DENSIFIABLE_ELEMENTS_ATTR_INTERFACE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

// TODO: rename to DenseLikeElementsAttrInterface
def DensifiableElementsAttrInterface : AttrInterface<"DensifiableElementsAttrInterface", [ElementsAttrInterface]> {
  let description = [{
    Interface for elements attributes with a toDenseElementsAttr() method.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns a copy of the elements as a DenseElementsAttr.
    }], "DenseElementsAttr", "toDenseElementsAttr", (ins), [{}], /*defaultImplementation=*/[{
      llvm_unreachable("TODO: implement this");
    }]>
  ];
}

#endif // DENSIFIABLE_ELEMENTS_ATTR_INTERFACE
