/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyFunctionManager.hpp"
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"

#include "mlir/IR/SymbolTable.h"

#include <deque>

using namespace mlir;

namespace lazycst {

std::size_t LazyFuncOpHash::operator()(
    const lazycst::LazyFuncOp &op) const noexcept {
  return llvm::hash_value(op.getAsOpaquePointer());
}

namespace {
unsigned numLazyArgs(LazyFuncOp op) {
  return llvm::count_if(op.getArgConstants(),
      [](Attribute attr) { return isa<LazyElementsAttr>(attr); });
}
} // namespace

// If Result is ready then attr is set,
// but if users is empty then it may have been disposed.
// Add nullptr to users if Result has other users than lazy functions.
//
// TODO: figure out if it's safe to dispose when users becomes empty
//       or if there's a risk that disposable is shared elsewhere,
//       e.g. if it was produced by folding an identity op
//       (maybe another layer of indirection is needed?)
struct LazyFunctionManager::Result {
  ElementsAttr attr;
  SmallVector<Function *, 1> users;
};

// If latch is <= 1 then all lazy operands are ready.
// If latch is zero then folding has commenced.
// If results.front().attr != nullptr then it is folded.
struct LazyFunctionManager::Function {
  LazyFuncOp op;
  SmallVector<Result, 1> results;
  size_t latch;
  Function(LazyFuncOp op) : op(op), results(), latch(numLazyArgs(op)) {}
  bool isFoldable() const { return latch <= 1; }
  bool isFolding() const {
    return latch == 0 && results.front().attr == nullptr;
  }
  bool isFolded() const {
    return latch == 0 && results.front().attr != nullptr;
  }
  void getAttrs(SmallVectorImpl<Attribute> &attrs) {
    assert(isFolded());
    attrs.reserve(results.size());
    for (const Result &result : results)
      attrs.push_back(result.attr);
  }
};

LazyFunctionManager::LazyFunctionManager() : counter(0) {}

LazyFunctionManager::~LazyFunctionManager() = default;

LazyFuncOp LazyFunctionManager::create(SymbolTable &symbolTable, Location loc) {
  auto module = cast<ModuleOp>(symbolTable.getOp());
  OpBuilder b(module.getBodyRegion());
  auto cstexpr = b.create<lazycst::LazyFuncOp>(loc, nextName(symbolTable));
  symbolTable.insert(cstexpr);
  return cstexpr;
}

Attribute LazyFunctionManager::getResult(const LazyFuncOp &op, unsigned index) {
  SmallVector<Attribute> attrs;
  getResults(op, attrs);
  return attrs[index];
}

void LazyFunctionManager::fold(llvm::ArrayRef<LazyFuncOp> ops) {
  {
    std::unique_lock<std::mutex> lock(functionsMutex);
    foldLocked(lock, ops);
  }
}

StringAttr LazyFunctionManager::nextName(SymbolTable &symbolTable) {
  unsigned subscript = counter++;
  auto name = StringAttr::get(
      symbolTable.getOp()->getContext(), "lazycst." + Twine(subscript));
  assert(!symbolTable.lookup(name) && "next LazyFuncOp name was already taken");
  return name;
}

void LazyFunctionManager::getResults(
    const LazyFuncOp &op, SmallVectorImpl<Attribute> &attrs) {
#if 1
  {
    std::unique_lock<std::mutex> lock(functionsMutex);
    foldLocked(lock, op);
    auto it = functions.find(op);
    assert(it != functions.end());
    it->second.getAttrs(attrs);
  }
#else
  // op is passed by reference so its type can be forward declared to
  // avoid cyclical header file dependencies, but its methods are non-const so
  // we copy it here to access its information
  LazyFuncOp lf = op;
  ArrayRef<Type> resultTypes = lf.getResultTypes();
  unsigned numResults = resultTypes.size();
  assert(numResults > 0);
  Function *function = nullptr;
  {
    std::unique_lock<std::mutex> lock(functionsMutex);
    auto [iter, inserted] = functions.try_emplace(lf.getSymName(), lf);
    function = &iter->second;
    if (!inserted) {
      functionsCondition.wait(
          lock, [function] { return function->isFolded(); });
      function->getAttrs(attrs);
      return;
    }
  }
  // We must calculate results and populate attrs.
  llvm_unreachable("TODO");
#endif
}

void LazyFunctionManager::foldLocked(
    std::unique_lock<std::mutex> &lock, llvm::ArrayRef<LazyFuncOp> ops) {
  using FuncEntry = std::pair<const LazyFuncOp, Function>;
  std::deque<FuncEntry *> readyToFold, unreadyToFold, folding;
  for (auto op : ops) {
    auto it = functions.find(op);
    assert(it != functions.end());
    FuncEntry *fe = &*it;
    if (fe->second.isFolded())
      continue;
    // TODO: check if fe is in readyToFold, unreadyToFold, blockingOnOthers
    //       and in that case skip to next op
    if (fe->second.isFolding()) {
      folding.push_back(fe);
      continue;
    }
    if (fe->second.isFoldable()) {
      readyToFold.push_back(fe);
      continue;
    }
    unreadyToFold.push_back(fe);
    // TODO: traverse fe->first's lazy args and add them to
    //       readyToFold/unreadyToFold/blockingOnOthers
  }
  while (!(readyToFold.empty() && folding.empty())) {
    // TODO: launch folding of some readyToFold lazy functions
    //       and add each to folding; on completion of folding
    //       call functionsCondition.notify_all()
    //
    //       simplest initial implementation is to just fold
    //       the first entry in readyToFold and call
    //       functionsCondition.notify_all() and run
    //       through the main loop below
    functionsCondition.wait(lock);
    bool anyFolded = false;
    auto it = folding.begin();
    while (it != folding.end()) {
      FuncEntry *fe = *it;
      if (fe->second.isFolded()) {
        it = folding.erase(it);
        anyFolded = true;
      } else {
        ++it;
      }
    }
    if (anyFolded) {
      auto it = unreadyToFold.begin();
      while (it != unreadyToFold.end()) {
        FuncEntry *fe = *it;
        if (fe->second.isFoldable()) {
          it = unreadyToFold.erase(it);
          if (fe->second.isFolding()) {
            folding.push_back(fe);
          } else if (!fe->second.isFolded()) {
            readyToFold.push_back(fe);
          }
        } else {
          ++it;
        }
      }
    }
  }
}

} // namespace lazycst
