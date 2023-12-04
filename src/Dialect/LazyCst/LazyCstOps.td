// SPDX-License-Identifier: Apache-2.0

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class LazyCst_Op<string mnemonic, list<Trait> traits = []>
    : Op<LazyCst_Dialect, mnemonic, traits> {}

def ElementsArrayAttr : TypedArrayAttrBase<ElementsAttr, "elements array attribute">;

def LazyCst_ExprOp : LazyCst_Op<"expr", [
  Symbol, IsolatedFromAbove, SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary = "Lazy constant expression";
  let description = [{
    A lazy constant expression represents a constant foldable operation graph.
    It has arguments and results like a function op but it is never called.
    $inputs and $outputs record an ElementsAttr for each argument
    and a LazyElementsAttr for each result.

    Every lazy constant expression op must be constructed with create() or
    parse().

    After construction the lazy constant expression op can be recorded with
    LazyCstExprManager::record(). It can be evaluated later with
    LazyCstExprManager::evaluate() provided all its child ops have constant
    folders in ConstantFolders.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       ElementsArrayAttr:$inputs,
                       ElementsArrayAttr:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let skipDefaultBuilders = 1;
  let builders = [
    // Generates a unique name and deduces lazy_elms outputs.
    OpBuilder<(ins "mlir::Block *":$entry_block, "mlir::ArrayAttr":$inputs)>
  ];
  let extraClassDeclaration = [{
    // Create a new lazy constant expression with a unique name and
    // records it in symbolTable, adds it to the symbolTable op region.
    static lazycst::ExprOp create(
        mlir::SymbolTable &symbol_table, mlir::Location loc,
        mlir::Block *entry_block, llvm::ArrayRef<mlir::Attribute> inputs);
  }];
  let hasCustomAssemblyFormat = 1;
}

def LazyCst_YieldOp : LazyCst_Op<"yield",
    [Pure, ReturnLike, Terminator, HasParent<"ExprOp">]> {
  let summary = "Lazy constant expression yield operation";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}
