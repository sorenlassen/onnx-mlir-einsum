// SPDX-License-Identifier: Apache-2.0

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
