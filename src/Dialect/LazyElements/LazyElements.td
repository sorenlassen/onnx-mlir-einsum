// SPDX-License-Identifier: Apache-2.0

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// The Lazy Elements Dialect.
//===----------------------------------------------------------------------===//

def LazyElements_Dialect : Dialect {
  let name = "lazyelms";
  let description = [{
    The Lazy Elements dialect.
  }];
  let cppNamespace = "::lazy_elements";
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    FileDataManager fileDataManager;
  }];
}

//===----------------------------------------------------------------------===//
// Lazy Elements Attributes.
//===----------------------------------------------------------------------===//

class LazyElements_Attr<string attrName, string attrMnemonic, list<Trait> traits = [],
                        string baseCppClass = "::mlir::Attribute">
    : AttrDef<LazyElements_Dialect, attrName, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

def LazyElements_FileDataElementsAttr : LazyElements_Attr<"FileDataElements", "file_data", [
    ElementsAttrInterface
  ], "BufferElementsAttr<FileDataElementsAttr>"> {
  let summary = "Attribute for external data: file with raw bytes.";
  let parameters = (ins
    AttributeSelfTypeParameter<"", "mlir::ShapedType">:$type,
    "mlir::StringAttr":$path,
    OptionalParameter<"uint64_t">:$offset
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
        "mlir::ShapedType":$type,
        "mlir::StringAttr":$path,
        CArg<"uint64_t", "0">:$offset), [{
      return $_get(type.getContext(), type, path, offset);
    }]>,
  ];
  let extraClassDeclaration = [{
    /// The set of data types that can be iterated by this attribute.
    using ContiguousIterableTypesT = std::tuple<
      int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
      bool, double, float,
      onnx_mlir::float_16, onnx_mlir::bfloat_16,
      onnx_mlir::float_8e4m3fn, onnx_mlir::float_8e4m3fnuz,
      onnx_mlir::float_8e5m2, onnx_mlir::float_8e5m2fnuz
    >;
    using NonContiguousIterableTypesT = std::tuple<
      mlir::Attribute, mlir::FloatAttr, mlir::IntegerAttr,
      llvm::APFloat, llvm::APInt, WideNum
    >;

  private:
    template <typename X, typename ...Ts>
    static constexpr bool is_in_pointed_to(std::tuple<Ts...> *p) {
      return llvm::is_one_of<X, Ts...>::value;
    }
    template <typename X, typename TupleType>
    static constexpr bool is_in = is_in_pointed_to<X>((TupleType *)nullptr);
    template <typename X>
    static constexpr bool isContiguousType = is_in<X, ContiguousIterableTypesT>;
    template <typename X>
    static constexpr bool isNonContiguousType = is_in<X, NonContiguousIterableTypesT>;
    template <typename X>
    static constexpr bool isIterableType = isContiguousType<X> || isNonContiguousType<X>;

  public:
    template <typename X>
    using iterator = std::conditional_t<
      isContiguousType<X>,
      const X *,
      llvm::mapped_iterator<llvm::iota_range<size_t>::const_iterator,
          std::function<X(size_t)>>
    >;

    template <typename X>
    using iterator_range = llvm::iterator_range<iterator<X>>;

    // This implementation enables the value_begin() and getValues() methods
    // from the ElementsAttr interface, for the NonContiguousIterableTypesT types.
    template <typename X>
    mlir::FailureOr<iterator<X>> try_value_begin_impl(OverloadToken<X>) const;

    template <typename X>
    iterator<X> value_end() const {
      return getValues<X>().end();
    }

    llvm::ArrayRef<char> getRawBytesImpl() const;

    WideNum atFlatIndex(size_t flatIndex) const;

    BType getBType() const { return btypeOfMlirType(getElementType()); }
  }];
  let assemblyFormat = "`<` $path ( `,` struct($offset)^ )? `>`";
}

def LazyElements_LazyElementsAttr : LazyElements_Attr<"LazyElements", "lazy", [
    ElementsAttrInterface
  ]> {
  let summary = "Lazily evaluated elements attribute.";
  let parameters = (ins
    AttributeSelfTypeParameter<"", "mlir::ShapedType">:$type,
    "mlir::StringAttr":$operation,
    ArrayRefParameter<"mlir::ElementsAttr", "">:$operands,
    OptionalParameter<"mlir::DictionaryAttr">:$attributes
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
        "mlir::ShapedType":$type,
        "mlir::StringAttr":$operation,
        "llvm::ArrayRef<mlir::ElementsAttr>":$operands,
        CArg<"mlir::DictionaryAttr", "nullptr">:$attributes), [{
      return $_get(type.getContext(), type, operation, operands, attributes);
    }]>,
  ];
  let assemblyFormat = "`<` $operation `(` $operands `)` ($attributes^)? `>`";
  // TODO: add fold / materialization methods
}
