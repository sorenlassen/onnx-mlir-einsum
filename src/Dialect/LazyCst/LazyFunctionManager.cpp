/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyFunctionManager.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"

#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace lazycst {

StringAttr LazyFunctionManager::nextName(ModuleOp module) {
  unsigned subscript = counter++;
  auto name =
      StringAttr::get(module.getContext(), "lazycst." + Twine(subscript));
  assert(!SymbolTable::lookupSymbolIn(module, name) &&
         "next LazyFuncOp name was already taken");
  return name;
}

ArrayRef<Attribute> LazyFunctionManager::getResults(
    const LazyFuncOp &lazyFunction) {
  // lazyFunction is passed by reference so its type can be forward declared to
  // avoid cyclical header file dependencies, but its methods are non-const so
  // we copy the function here to access its information most conveniently
  LazyFuncOp f = lazyFunction;
  ArrayRef<Type> resultTypes = f.getResultTypes();
  unsigned numResults = resultTypes.size();
  assert(numResults > 0);
  std::vector<Attribute> *attrs = nullptr;
  {
    std::unique_lock<std::mutex> lock(resultsMutex);
    std::vector<Attribute> nullAttrs(numResults, nullptr);
    auto [iter, inserted] = results.try_emplace(f.getSymName(), nullAttrs);
    attrs = &iter->second;
    if (!inserted) {
      assert(attrs->size() == numResults);
      resultsCondition.wait(
          lock, [attrs] { return attrs->front() != nullptr; });
      return *attrs;
    }
  }
  // We must calculate results and populate attrs.
  llvm_unreachable("TODO");
}

} // namespace lazycst
