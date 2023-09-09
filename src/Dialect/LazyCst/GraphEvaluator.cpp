/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/GraphEvaluator.hpp"

namespace lazycst {

GraphEvaluator::GraphEvaluator(llvm::ThreadPool *threadPool)
    : threadPool(threadPool) {}

GraphEvaluator::~GraphEvaluator() = default;

// All predecessors must have been added beforehand.
void GraphEvaluator::addNode(mlir::Operation *op,
    llvm::ArrayRef<NodeOperand> operands, Fold fold, bool usedOutsideGraph) {
  assert(fold != nullptr);
  auto [it, inserted] = nodes.try_emplace(op);
  assert(inserted);
  OpRecord &rec = it->second;
  rec.operands.reserve(operands.size());
  for (auto [op, index] : operands)
    rec.operands.emplace_back(lookup(op), index);
  rec.fold = std::move(fold);
}

auto GraphEvaluator::lookup(mlir::Operation *op) -> OpEntry * {
  auto it = nodes.find(op);
  assert(it != nodes.end());
  return &*it;
}

void GraphEvaluator::enqueue(OpEntry *entry) {
  auto doFold = [this, entry, fold = std::move(entry->second.fold)] {
    OpRecord &rec = entry->second;
    llvm::SmallVector<mlir::Attribute> operands;
    for (auto [node, index] : rec.operands)
      operands.push_back(node->second.results[index]);
    fold(entry->first, operands, rec.results);
    entry->second.who = 0;
    condition.notify_all();
  };
  assert(entry->second.fold == nullptr);
  if (threadPool) {
    llvm_unreachable("TODO: enqueue doFold in thread pool");
  } else {
    doFold();
  }
}

bool GraphEvaluator::tryFoldNode(OpEntry *entry, int me, Set &awaiting) {
  OpRecord &rec = entry->second;
  if (rec.isVisited()) {
    if (rec.isFolded())
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
  for (auto [node, _] : rec.operands)
    queue &= tryFoldNode(node, me, awaiting);
  if (queue) {
    enqueue(entry);
    if (rec.isFolded())
      return true;
  }
  auto [_, inserted] = awaiting.insert(entry);
  assert(inserted);
  return false;
}

void GraphEvaluator::evaluate(llvm::ArrayRef<mlir::Operation *> ops,
    llvm::SmallVectorImpl<llvm::ArrayRef<mlir::Attribute>> &results) {
  std::unique_lock<std::mutex> lock(mux);
  int me = ++whoCounter;
  assert(me != 0);
  Set awaiting;
  llvm::SmallVector<OpEntry *> nodes(
      llvm::map_range(ops, [this](mlir::Operation *op) { return lookup(op); }));
  for (auto *node : nodes)
    tryFoldNode(node, me, awaiting);
  while (!awaiting.empty()) {
    condition.wait(lock);
    llvm::SmallVector<OpEntry *> folded;
    for (OpEntry *entry : awaiting) {
      OpRecord &rec = entry->second;
      if (rec.who == me && rec.fold != nullptr &&
          llvm::all_of(rec.operands, isOperandFolded)) {
        enqueue(entry);
      }
      if (rec.isFolded())
        folded.push_back(entry);
    }
    for (OpEntry *entry : folded)
      awaiting.erase(entry);
  }
  for (auto *node : nodes)
    results.push_back(llvm::ArrayRef(node->second.results));
}

} // namespace lazycst
