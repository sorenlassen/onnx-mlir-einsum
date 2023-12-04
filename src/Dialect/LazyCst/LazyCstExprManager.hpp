/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/Dialect/LazyCst/GraphEvaluator.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <atomic>

namespace lazycst {

class ExprOp;

class LazyCstExprManager {
public:
  LazyCstExprManager();

  ~LazyCstExprManager();

  void initialize(mlir::MLIRContext *ctx);

  // Create a new lazy constant expression with a unique name and
  // records it in symbolTable, adds it to the symbolTable op region.
  lazycst::ExprOp create(mlir::SymbolTable &symbolTable, mlir::Location loc,
      mlir::Block *entryBlock, llvm::ArrayRef<mlir::Attribute> inputs);

  // Record cstexpr for future evaluation with evaluate().
  // `symbolTable` should be the module symbol table and is used to look up
  // any lazy_elms arguments (record could build this but it's cheaper for the
  // caller to pass a built SymbolTable in case record is called repeatedly).
  void record(const mlir::SymbolTable &symbolTable, lazycst::ExprOp cstexpr,
      bool onlyLazyCstExprUsers);

  // Evaluate the index'th result of cstexpr.
  mlir::Attribute evaluate(lazycst::ExprOp cstexpr, unsigned index);

  // Evaluate the index'th result of the lazy cstexpr named symName.
  mlir::Attribute evaluate(mlir::StringAttr symName, unsigned index);

  // Evaluate all results of all cstexprs.
  void evaluate(llvm::ArrayRef<lazycst::ExprOp> cstexprs,
      llvm::SmallVectorImpl<llvm::ArrayRef<mlir::Attribute>> &results);

private:
  mlir::StringAttr nextName(mlir::SymbolTable &symbolTable);

  std::atomic<unsigned> counter;

  // "Shadow" symbol table used to look up lazy constant expressions in
  // evaluate(symNam, index), called from LazyElementsAttr::getElementsAttr()
  // without access to the ModuleOp symbol table.
  llvm::DenseMap<mlir::StringAttr, lazycst::ExprOp> table;

  GraphEvaluator evaluator;
};

} // namespace lazycst
