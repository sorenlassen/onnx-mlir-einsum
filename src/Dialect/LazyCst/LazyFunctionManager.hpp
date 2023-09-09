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
#include "llvm/ADT/SmallVector.h"

#include <atomic>

namespace lazycst {

class LazyFuncOp;

class LazyFunctionManager {
public:
  LazyFunctionManager();
  ~LazyFunctionManager();

  LazyFuncOp create(mlir::SymbolTable &symbolTable, mlir::Location loc);

  void record(mlir::SymbolTable &symbolTable, LazyFuncOp cstexpr,
      bool onlyLazyFunctionUsers);

  mlir::Attribute getResult(LazyFuncOp cstexpr, unsigned index);

  void evaluate(llvm::ArrayRef<LazyFuncOp> cstexprs,
      llvm::SmallVectorImpl<llvm::ArrayRef<mlir::Attribute>> &results);

private:
  mlir::StringAttr nextName(mlir::SymbolTable &symbolTable);

  std::atomic<unsigned> counter;
  GraphEvaluator evaluator;
};

} // namespace lazycst
