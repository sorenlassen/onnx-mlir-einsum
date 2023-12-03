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
        "const llvm::Twine &":$name), [{
      mlir::MLIRContext *ctx = type.getContext();
      return $_get(ctx, type, mlir::StringAttr::get(ctx, name));
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
        "const llvm::Twine &":$path,
        CArg<"uint64_t", "0">:$offset), [{
      mlir::MLIRContext *ctx = type.getContext();
      return $_get(ctx, type, mlir::StringAttr::get(ctx, path), offset);
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
    template <typename T> using iterator = RawBytesIterator<T>;
    template <typename T> using iterator_range = llvm::iterator_range<iterator<T>>;
    template <typename T> auto try_value_begin_impl(OverloadToken<T>) const {
      return try_value_begin_from_raw_bytes<T>(getElementType(), getRawBytes());
    }
    template <typename T> auto value_end() const {
      return std::next(value_begin<T>(), size());
    }
    template <typename T> iterator_range<T> getValues() const {
      return {value_begin<T>(), value_end<T>()};
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
    Parameter callee must refer to a LazyCst_ExprOp. Parameter index specifies
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
    template <typename T> auto try_value_begin_impl(OverloadToken<T>) const {
      return getElementsAttr().try_value_begin<T>();
    }
    template <typename T> auto value_end() const {
      return getElementsAttr().value_end<T>();
    }
    // Forward getValuesImpl(TypeId) because the (Non)ContiguousIterableTypesT
    // default logic is unworkable since we don't know the contiguous types.
    auto getValuesImpl(mlir::TypeID elementID) const {
      return getElementsAttr().getValuesImpl(elementID);
    }
    // Forward isSplat() because the "conservative" default implementation uses
    // only the size and getElementsAttr() may differ e.g. if it's a
    // DenseElementsAttr.
    bool isSplat() const { return getElementsAttr().isSplat(); }
  }];
  let assemblyFormat = "`<` $callee ( `,` struct($index)^ )? `>`";
}
