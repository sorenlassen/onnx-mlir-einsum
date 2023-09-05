/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include <atomic>

namespace lazycst {

class LazyFunctionManager {
public:
  LazyFunctionManager() : counter(0) {}
  mlir::StringAttr nextName(mlir::ModuleOp module);

private:
  std::atomic<unsigned> counter;
};

} // namespace lazycst
