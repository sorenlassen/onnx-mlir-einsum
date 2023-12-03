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

  lazycst::ExprOp create(mlir::SymbolTable &symbolTable, mlir::Location loc);

  void record(mlir::SymbolTable &symbolTable, lazycst::ExprOp cstexpr,
      bool onlyLazyCstExprUsers);

  // Evaluate the index'th result of cstexpr.
  mlir::Attribute getResult(lazycst::ExprOp cstexpr, unsigned index);

  // Evaluate the index'th result of the lazy cstexpr named symName.
  mlir::Attribute getResult(mlir::StringAttr symName, unsigned index);

  // Evaluate all results of all cstexprs.
  void evaluate(llvm::ArrayRef<lazycst::ExprOp> cstexprs,
      llvm::SmallVectorImpl<llvm::ArrayRef<mlir::Attribute>> &results);

private:
  mlir::StringAttr nextName(mlir::SymbolTable &symbolTable);

  std::atomic<unsigned> counter;
  llvm::DenseMap<mlir::StringAttr, lazycst::ExprOp> table;
  GraphEvaluator evaluator;
};

} // namespace lazycst
