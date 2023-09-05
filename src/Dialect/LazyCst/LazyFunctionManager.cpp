/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyFunctionManager.hpp"

#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace lazycst {

mlir::StringAttr LazyFunctionManager::nextName(mlir::ModuleOp module) {
  unsigned subscript = counter++;
  auto name = mlir::StringAttr::get(
      module.getContext(), "lazycst." + llvm::Twine(subscript));
  assert(!SymbolTable::lookupSymbolIn(module, name) &&
         "next LazyFuncOp name was already taken");
  return name;
}

} // namespace lazycst
