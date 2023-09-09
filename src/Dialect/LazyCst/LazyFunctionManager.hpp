/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/Dialect/LazyCst/GraphEvaluator.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace lazycst {

class LazyFuncOp;

struct LazyFuncOpHash {
  std::size_t operator()(const LazyFuncOp &op) const noexcept;
};

class LazyFunctionManager {
public:
  LazyFunctionManager();
  ~LazyFunctionManager();

  LazyFuncOp create(mlir::SymbolTable &symbolTable, mlir::Location loc);

  void record(mlir::SymbolTable &symbolTable, LazyFuncOp cstexpr,
      bool onlyLazyFunctionUsers);

  mlir::Attribute getResult(const LazyFuncOp &op, unsigned index);

  void fold(llvm::ArrayRef<LazyFuncOp> ops);

private:
  struct Result;
  struct Function;

  mlir::StringAttr nextName(mlir::SymbolTable &symbolTable);

  void getResults(
      const LazyFuncOp &op, llvm::SmallVectorImpl<mlir::Attribute> &attrs);

  void foldLocked(
      std::unique_lock<std::mutex> &lock, llvm::ArrayRef<LazyFuncOp> ops);

  std::atomic<unsigned> counter;
  std::mutex functionsMutex;
  std::condition_variable functionsCondition;
  std::unordered_map<LazyFuncOp, Function, LazyFuncOpHash> functions;

  GraphEvaluator evaluator;
};

} // namespace lazycst
