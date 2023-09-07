/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace lazycst {

class LazyFuncOp;

class LazyFunctionManager {
public:
  LazyFunctionManager() : counter(0) {}

  mlir::StringAttr nextName(mlir::ModuleOp module);

  llvm::ArrayRef<mlir::Attribute> getResults(const LazyFuncOp &lazyFunction);

private:
  std::atomic<unsigned> counter;

  std::mutex resultsMutex;
  std::condition_variable resultsCondition;
  llvm::StringMap<std::vector<mlir::Attribute>> results;
};

} // namespace lazycst
