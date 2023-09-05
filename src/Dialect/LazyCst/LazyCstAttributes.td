// SPDX-License-Identifier: Apache-2.0

class LazyCst_Attr<string attrName, string attrMnemonic, list<Trait> traits = [],
                        string baseCppClass = "::mlir::Attribute">
    : AttrDef<LazyCst_Dialect, attrName, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

// def RawBytesElementsAttrTrait : NativeAttrTrait<"RawBytesElementsAttrTrait"> {
//   let cppNamespace = "lazycst";
// }

def FileDataElementsAttr : LazyCst_Attr<"FileDataElements", "file_data", [
    DenseLikeElementsAttrInterface
  ]> {
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
    llvm::ArrayRef<char> getRawBytes() const;

    // DenseLikeElementsAttrInterface implementation.
    mlir::DenseElementsAttr toDenseElementsAttr() const {
      return toDenseElementsAttrFromRawBytes(getType(), getRawBytes());
    }

    // ElementsAttrInterface implementation.
    using ContiguousIterableTypesT = RawBytesContiguousIterTypes;
    using NonContiguousIterableTypesT = RawBytesNonContiguousIterTypes;
    template <typename X> using iterator = RawBytesIterator<X>;
    template <typename X> using iterator_range = llvm::iterator_range<iterator<X>>;
    template <typename T> auto try_value_begin_impl(OverloadToken<T>) const {
      return try_value_begin_from_raw_bytes<T>(getElementType(), getRawBytes());
    }
    template <typename X> auto value_end() const {
      return std::next(value_begin<X>(), size());
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
