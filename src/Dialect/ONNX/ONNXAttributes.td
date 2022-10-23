// SPDX-License-Identifier: Apache-2.0

//===---- ONNXAttributes.td -- ONNX attributes definitions ---*- tablegen -===//
//
// Defines ONNX attributes.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_ATTRIBUTES_TD
#define ONNX_ATTRIBUTES_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"

class ONNX_Attr<string name, list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<ONNX_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def DisposableElementsAttr : ONNX_Attr<
    "DisposableElements",
    [ElementsAttrInterface, TypedAttrInterface]
  > {
  let summary = "An Attribute containing a dense tensor "
                "backed by a garbage collectible memory buffer";
  let description = [{
    Syntax is the same as DenseElementsAttr, for convenience in lit tests.
  }];
  let parameters = (ins
    AttributeSelfTypeParameter<"", "ShapedType">:$type,
    "::onnx_mlir::DisposableResultsHandle":$resultsHandle, "size_t":$resultIndex
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins "ShapedType":$type,
      "::onnx_mlir::DisposableResultsHandle":$resultsHandle, "size_t":$resultIndex
    )>
  ];
  let extraClassDeclaration = [{
    /// A builder that creates a DisposableExpression with a single result.
    static DisposableElementsAttr get(::onnx_mlir::DisposableElements result);

    /// The set of data types that can be iterated by this attribute.
    /// TODO: Consider moving all to NonContiguousIterableTypesT
    ///       because contiguous or not depends on strides and transformation.
    using ContiguousIterableTypesT = std::tuple<
      uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
      bool, float, double
    >;
    using NonContiguousIterableTypesT = std::tuple<Attribute, APInt, APFloat>;

    ::onnx_mlir::DisposableElements getElements() const;
  }];
  let skipDefaultBuilders = 1;
}

#endif // ONNX_ATTRIBUTES_TD
