// SPDX-License-Identifier: Apache-2.0

#ifdef DENSE_LIKE_ELEMENTS_ATTR_INTERFACE
#else
#define DENSE_LIKE_ELEMENTS_ATTR_INTERFACE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def DenseLikeElementsAttrInterface : AttrInterface<"DenseLikeElementsAttrInterface", [ElementsAttrInterface]> {
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

#endif // DENSE_LIKE_ELEMENTS_ATTR_INTERFACE
