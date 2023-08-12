// SPDX-License-Identifier: Apache-2.0

#ifndef LAZYCST_TD
#define LAZYCST_TD

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "src/Interface/DensifiableElementsAttrInterface.td"

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
    FileDataManager fileDataManager;
  }];
}

//===----------------------------------------------------------------------===//
// Lazy Elements Attributes.
//===----------------------------------------------------------------------===//

class LazyCst_Attr<string attrName, string attrMnemonic, list<Trait> traits = [],
                        string baseCppClass = "::mlir::Attribute">
    : AttrDef<LazyCst_Dialect, attrName, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

def FileDataElementsAttr : LazyCst_Attr<"FileDataElements", "file_data", [
    DensifiableElementsAttrInterface
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

    // Makes deep copy.
    mlir::DenseElementsAttr toDenseElementsAttr() const;
  }];
  let assemblyFormat = "`<` $path ( `,` struct($offset)^ )? `>`";
}

def LazyElementsAttr : LazyCst_Attr<"LazyElements", "lazy", [
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

//===----------------------------------------------------------------------===//
// Lazy Constant Ops.
//===----------------------------------------------------------------------===//

class LazyCst_Op<string mnemonic, list<Trait> traits = []>
    : Op<LazyCst_Dialect, mnemonic, traits> {}

def LazyReturnOp : LazyCst_Op<"lazy_return",
    [Pure, HasParent<"LazyConstantOp">, ReturnLike, Terminator]> {
  let summary = "Lazy constant return operation";
  let description = [{
    The `lazycst.lazy_return` operation represents the results of a lazy constant.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def LazyConstantOp : LazyCst_Op<"lazy_constant", [
    SingleBlockImplicitTerminator<"LazyReturnOp">,
    IsolatedFromAbove,
    NoRegionArguments,
    // SingleBlock,
    Symbol
  ]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeArrayAttr:$res_types,
    DefaultValuedAttr<
      TypedArrayAttrBase<LazyElementsAttr, "array of lazy elements attributes">,
      "{}">:$lazy_args);
  let regions = (region SizedRegion<1>:$body);
}

#endif // LAZYCST_TD
