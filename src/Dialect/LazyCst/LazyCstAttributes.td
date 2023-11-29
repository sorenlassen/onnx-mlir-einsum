// SPDX-License-Identifier: Apache-2.0

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "src/Interface/DenseLikeElementsAttrInterface.td"

class LazyCst_Attr<string attrName, string attrMnemonic, list<Trait> traits = [],
                        string baseCppClass = "::mlir::Attribute">
    : AttrDef<LazyCst_Dialect, attrName, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

def ExternalElementsAttr : LazyCst_Attr<"ExternalElements", "extern_elms", [
    ElementsAttrInterface
  ]> {
  let summary = "Attribute for elements with 'external linkage'.";
  let description = [{
    Parameter type must be a RankedTensorType or VectorType.
    TODO: Explain more.
  }];
  let parameters = (ins
    AttributeSelfTypeParameter<"", "mlir::ShapedType">:$type,
    "mlir::StringAttr":$name
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
        "mlir::ShapedType":$type,
        "mlir::StringAttr":$name), [{
      return $_get(type.getContext(), type, name);
    }]>,
  ];
  // NOTE: We don't override (Non)ContiguousIterableTypesT, so getValuesImpl()
  //       always fails with "invalid `T` for ElementsAttr::getValues" and
  //       therefore any call to getValues<T>() or value_begin<T>() for any T
  //       via the ElementsAttr interface always fails that way.
  //       We don't implement try_value_begin_impl() and therefore any call to
  //       ExternalElementsAttr::getValues<T>() or value_begin<T>() for any T
  //       directly (not via the ElementsAttr interface) fails to compile.
  let assemblyFormat = "`<` $name `>`";
}

def FileDataElementsAttr : LazyCst_Attr<"FileDataElements", "file_data", [
    DenseLikeElementsAttrInterface
  ]> {
  let summary = "Attribute for external data: file with raw bytes.";
  let description = [{
    Parameter type must be a RankedTensorType or VectorType.
    Parameter path is relative to FileDataManager::Config::readDirectoryPaths.
    Parameter offset is the file offset where the data will be read from.
    The data length is inferred from type.
    The data is assumed to be in row-major order, like NumPy .npy files,
    with elements byte aligned, in particular booleans are not bit packed.
  }];
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
  let assemblyFormat = "`<` $path ( `+` $offset^ )? `>`";
}

// Currently doesn't implement DenseLikeElementsAttrInterface so that, by
// default, it serializes efficiently without materializing.
def LazyElementsAttr : LazyCst_Attr<"LazyElements", "lazy_elms", [
    ElementsAttrInterface // TODO: consider DenseLikeElementsAttrInterface instead
  ]> {
  let summary = "Lazily evaluated elements attribute.";
  let description = [{
    Parameter callee must refer to a LazyFuncOp. Parameter index specifies
    one of callee's results and defaults to 0, the first result.
  }];
  let parameters = (ins
    AttributeSelfTypeParameter<"", "mlir::ShapedType">:$type,
    "mlir::FlatSymbolRefAttr":$callee,
    OptionalParameter<"unsigned">:$index
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
        "mlir::ShapedType":$type,
        "mlir::FlatSymbolRefAttr":$callee,
        CArg<"unsigned", "0">:$index), [{
      return $_get(type.getContext(), type, callee, index);
    }]>,
  ];
  let extraClassDeclaration = [{
    // Returns materialized (evaluated) elements. Forces the evaluation if
    // the materialization wasn't already evaluated and cached.
    mlir::ElementsAttr getElementsAttr() const;

    // DenseLikeElementsAttrInterface implementation.
    mlir::DenseElementsAttr toDenseElementsAttr() const {
      return toDenseElementsAttrFromElementsAttr(getElementsAttr());
    }

    // Implement ElementsAttrInterface by forwarding to getElementsAttr().
    // Forward getValuesImpl(TypeId) to short-circuit ElementsAttr's default
    // (Non)ContiguousIterableTypesT logic (we don't know the contiguous types)
    // and then we can skip try_value_begin_impl<T>(OverloadToken<T>).
    auto getValuesImpl(mlir::TypeID elementID) const {
      return getElementsAttr().getValuesImpl(elementID);
    }
    // Forward isSplat(). The default implementation is not always accurate.
    bool isSplat() const { return getElementsAttr().isSplat(); }
  }];
  let assemblyFormat = "`<` $callee ( `,` struct($index)^ )? `>`";
}
