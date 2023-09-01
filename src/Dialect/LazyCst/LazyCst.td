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
    LazyFunctionManager lazyFunctionManager;
    ConstantFolders constantFolders;
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

    // BufferElementsAttr implementation.
    llvm::ArrayRef<char> getRawBytesImpl() const;

    // DensifiableElementsAttrInterface implementation.
    mlir::DenseElementsAttr toDenseElementsAttr() const {
      return denseElementsFromRawBytes(getType(), getRawBytes());
    }
  }];
  let assemblyFormat = "`<` $path ( `,` struct($offset)^ )? `>`";
}

def LazyElementsAttr : LazyCst_Attr<"LazyElements", "lazy_elms", [
    ElementsAttrInterface
  ]> {
  let summary = "Lazily evaluated elements attribute.";
  let parameters = (ins
    AttributeSelfTypeParameter<"", "mlir::ShapedType">:$type,
    "mlir::FlatSymbolRefAttr":$callee, OptionalParameter<"unsigned">:$index
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
        "mlir::ShapedType":$type,
        "mlir::FlatSymbolRefAttr":$callee,
        CArg<"unsigned", "0">:$index), [{
      return $_get(type.getContext(), type, callee, index);
    }]>,
  ];
  let assemblyFormat = "`<` $callee ( `,` struct($index)^ )? `>`";
}

//===----------------------------------------------------------------------===//
// Lazy Constant Ops.
//===----------------------------------------------------------------------===//

class LazyCst_Op<string mnemonic, list<Trait> traits = []>
    : Op<LazyCst_Dialect, mnemonic, traits> {}

def ElementsArrayAttr : TypedArrayAttrBase<ElementsAttr, "elements array attribute">;

def LazyFuncOp : LazyCst_Op<"func", [
  CallableOpInterface, FunctionOpInterface, IsolatedFromAbove,
  SingleBlockImplicitTerminator<"LazyReturnOp">
]> {
  let summary = "Lazy constant expression";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       ElementsArrayAttr:$arg_constants,
                       ElementsArrayAttr:$res_constants,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region SizedRegion<1>:$body);
  let builders = [
    // Creates with empty function type, args, results, to be filled in later.
    OpBuilder<(ins "mlir::StringAttr":$sym_name)>
  ];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    llvm::ArrayRef<mlir::Type> getCallableResults() { return getFunctionType().getResults(); }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    llvm::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    llvm::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
  let hasCustomAssemblyFormat = 1;
}

def LazyReturnOp : LazyCst_Op<"return",
    [Pure, ReturnLike, Terminator, HasParent<"LazyFuncOp">]> {
  let summary = "Lazy constant expression return operation";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

#endif // LAZYCST_TD
