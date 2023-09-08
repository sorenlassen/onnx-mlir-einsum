/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/GraphEvaluator.hpp"

namespace lazycst {

GraphEvaluator::GraphEvaluator(llvm::ThreadPool *threadPool)
    : threadPool(threadPool) {}

GraphEvaluator::~GraphEvaluator() = default;

// All predecessors must have been added beforehand.
void GraphEvaluator::addNode(mlir::Operation *node,
    llvm::ArrayRef<mlir::Operation *> predecessors, Eval eval) {
  assert(eval != nullptr);
  auto [it, inserted] = nodes.try_emplace(node);
  assert(inserted);
  OpRecord &rec = it->second;
  rec.predecessors.reserve(predecessors.size());
  for (mlir::Operation *predecessor : predecessors)
    rec.predecessors.push_back(lookup(predecessor));
  rec.eval = std::move(eval);
}

auto GraphEvaluator::lookup(mlir::Operation *node) -> OpEntry * {
  auto it = nodes.find(node);
  assert(it != nodes.end());
  return &*it;
}

void GraphEvaluator::enqueue(OpEntry *entry) {
  OpRecord &rec = entry->second;
  if (threadPool) {
    auto f = [this, entry, eval = std::move(rec.eval)] {
      eval(entry->first);
      entry->second.who = 0;
      condition.notify_all();
    };
    assert(rec.eval == nullptr);
    llvm_unreachable("TODO: enqueue f in thread pool");
  } else {
    rec.eval(entry->first);
    rec.eval = nullptr;
    rec.who = 0;
    condition.notify_all();
  }
}

bool GraphEvaluator::tryEvaluateNode(OpEntry *entry, int me, Set &awaiting) {
  OpRecord &rec = entry->second;
  if (rec.isVisited()) {
    if (rec.isEvaluated())
      return true;
    assert(rec.who != 0);
    if (rec.who != me)
      awaiting.insert(entry);
    else
      assert(awaiting.contains(entry));
    return false;
  }
  rec.who = me;
  bool queue = true;
  for (OpEntry *predecessor : rec.predecessors)
    queue &= tryEvaluateNode(predecessor, me, awaiting);
  if (queue) {
    enqueue(entry);
    if (rec.isEvaluated())
      return true;
  }
  auto [_, inserted] = awaiting.insert(entry);
  assert(inserted);
  return false;
}

void GraphEvaluator::evaluate(llvm::ArrayRef<mlir::Operation *> nodes) {
  std::unique_lock<std::mutex> lock(mux);
  int me = ++whoCounter;
  assert(me != 0);
  Set awaiting;
  for (auto node : nodes)
    tryEvaluateNode(lookup(node), me, awaiting);
  while (!awaiting.empty()) {
    condition.wait(lock);
    llvm::SmallVector<OpEntry *> evaluated;
    for (OpEntry *entry : awaiting) {
      OpRecord &rec = entry->second;
      if (rec.who == me && rec.eval != nullptr &&
          llvm::all_of(rec.predecessors, isEvaluated)) {
        enqueue(entry);
      }
      if (rec.isEvaluated())
        evaluated.push_back(entry);
    }
    for (OpEntry *entry : evaluated)
      awaiting.erase(entry);
  }
}

} // namespace lazycst
