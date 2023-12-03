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

class LazyFunctionManager {
public:
  LazyFunctionManager();

  ~LazyFunctionManager();

  void initialize(mlir::MLIRContext *ctx);

  lazycst::ExprOp create(mlir::SymbolTable &symbolTable, mlir::Location loc);

  void record(mlir::SymbolTable &symbolTable, lazycst::ExprOp cstexpr,
      bool onlyLazyFunctionUsers);

  lazycst::ExprOp lookup(mlir::StringAttr symName) const;

  mlir::Attribute getResult(lazycst::ExprOp cstexpr, unsigned index);

  mlir::Attribute getResult(mlir::StringAttr symName, unsigned index);

  void evaluate(llvm::ArrayRef<lazycst::ExprOp> cstexprs,
      llvm::SmallVectorImpl<llvm::ArrayRef<mlir::Attribute>> &results);

private:
  mlir::StringAttr nextName(mlir::SymbolTable &symbolTable);

  std::atomic<unsigned> counter;
  llvm::DenseMap<mlir::StringAttr, lazycst::ExprOp> table;
  GraphEvaluator evaluator;
};

} // namespace lazycst
