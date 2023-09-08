/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ThreadPool.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace lazycst {

// TODO: hard-code NodeRef to Operation* for simplicity
template <typename NodeRef>
class GraphEvaluator {
public:
  using Eval = std::function<void(NodeRef)>;

  GraphEvaluator(llvm::ThreadPool *threadPool) : threadPool(threadPool) {}

  // All predecessors must have been added beforehand.
  void addNode(NodeRef node, llvm::ArrayRef<NodeRef> predecessors, Eval eval) {
    assert(eval != nullptr);
    auto [it, inserted] = nodes.try_emplace(node);
    assert(inserted);
    it->predecessors.reserve(predecessors.size());
    for (NodeRef predecessor : predecessors)
      it->predecessors.push_back(lookup(predecessor));
    it->eval = std::move(eval);
  }

  void evaluate(llvm::ArrayRef<NodeRef> nodes);

private:
  struct NodeRecord;
  using NodeEntry = std::pair<const NodeRef, NodeRecord>;
  struct NodeRecord {
    llvm::SmallVector<NodeEntry *, 2> predecessors;
    // Function to evaluate.
    // Set to nullptr after it has been queued for evaluation.
    Eval eval;
    // Who is assigned to evaluate. Cleared after completion.
    std::atomic<int> who = 0;

    bool isVisited() const { return eval == nullptr || who != 0; }
    bool isQueued() const { return eval == nullptr && who != 0; }
    bool isEvaluated() const { return eval == nullptr && who == 0; }
  };
  using Set = llvm::SmallPtrSet<NodeEntry *, 1>;
  static bool isEvaluated(const NodeEntry *entry) {
    return entry->second.isEvaluated();
  }

  NodeEntry *lookup(NodeRef node) const {
    auto it = nodes.find(node);
    assert(it != nodes.end());
    return &*it;
  }

  void enqueue(NodeRecord &rec);

  bool tryEvaluateNode(NodeEntry *entry, size_t me, Set &awaiting);

  llvm::ThreadPool *threadPool;
  std::mutex mux;
  std::condition_variable condition;
  std::unordered_map<NodeRef, NodeRecord> nodes;
  int whoCounter = 0;
};

template <typename NodeRef>
void GraphEvaluator<NodeRef>::enqueue(NodeRecord &rec) {
  if (threadPool) {
    auto f = [this, &rec, eval = std::move(rec.eval)] {
      eval();
      rec.who = 0;
      condition.notify_all();
    };
    assert(rec.eval == nullptr);
    llvm_unreachable("TODO: enqueue f in thread pool");
  } else {
    rec.eval();
    rec.eval = nullptr;
    rec.who = 0;
    condition.notify_all();
  }
}

template <typename NodeRef>
bool GraphEvaluator<NodeRef>::tryEvaluateNode(
    NodeEntry *entry, size_t me, Set &awaiting) {
  NodeRecord &rec = entry->second;
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
  bool queue = false;
  for (NodeEntry *predecessor : rec.predecessors)
    queue |= tryEvaluateNode(predecessor, me, awaiting);
  if (queue) {
    enqueue(rec);
    if (rec.isEvaluated())
      return true;
    auto [_, inserted] = awaiting.insert(entry);
    assert(inserted);
    return false;
  }
}

template <typename NodeRef>
void GraphEvaluator<NodeRef>::evaluate(llvm::ArrayRef<NodeRef> nodes) {
  std::unique_lock<std::mutex> lock(mux);
  int me = ++whoCounter;
  assert(me != 0);
  Set awaiting;
  for (auto node : nodes)
    tryEvaluateNode(lookup(node), me, awaiting);
  while (!awaiting.empty()) {
    condition.wait(lock);
    llvm::SmallVector<NodeEntry *> evaluated;
    for (NodeEntry *entry : awaiting) {
      NodeRecord &rec = entry->second;
      if (rec.who == me && rec.eval != nullptr &&
          llvm::all_of(rec.predecessors, isEvaluated)) {
        enqueue(entry);
      }
      if (rec.isEvaluated())
        evaluated.push_back(entry);
    }
    for (NodeEntry *entry : evaluated)
      awaiting.erase(entry);
  }
}

} // namespace lazycst
