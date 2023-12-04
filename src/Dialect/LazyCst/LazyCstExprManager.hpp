/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/Dialect/LazyCst/GraphEvaluator.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace lazycst {

class ExprOp;

// TODO: rename to LazyCstExprEvaluator
class LazyCstExprManager {
public:
  LazyCstExprManager();

  ~LazyCstExprManager();

  void initialize(mlir::MLIRContext *ctx);

  // Record cstexpr for future evaluation with evaluate().
  // The cstexpr of all argument lazy_elms must have been recorded beforehand.
  void record(lazycst::ExprOp cstexpr, bool onlyLazyCstExprUsers = false);

  // Record cstexpr with the given name and entry block for future evaluation
  // with evaluate().
  void insert(mlir::StringAttr symName, mlir::Block *entryBlock);

  // Evaluate the index'th result of cstexpr.
  mlir::Attribute evaluate(lazycst::ExprOp cstexpr, unsigned index);

  // Evaluate the index'th result of the lazy cstexpr named symName.
  mlir::Attribute evaluate(mlir::StringAttr symName, unsigned index);

  // Evaluate all results of all cstexprs.
  void evaluate(llvm::ArrayRef<lazycst::ExprOp> cstexprs,
      llvm::SmallVectorImpl<llvm::ArrayRef<mlir::Attribute>> &results);

private:
  lazycst::ExprOp lookup(mlir::StringAttr symName) const;

  // "Shadow" symbol table used to look up lazy constant expressions in
  // evaluate(symNam, index), called from LazyElementsAttr::getElementsAttr()
  // without access to the ModuleOp symbol table.
  // Maps to the lazy constant expression's entry block because that is
  // available during parsing.
  llvm::DenseMap<mlir::StringAttr, mlir::Block *> table;

  GraphEvaluator evaluator;
};

} // namespace lazycst
