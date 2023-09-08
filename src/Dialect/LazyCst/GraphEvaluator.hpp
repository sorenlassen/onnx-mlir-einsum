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

namespace mlir {
class Operation;
}

namespace lazycst {

class GraphEvaluator {
public:
  using NodeRef = mlir::Operation *;
  using Eval = std::function<void(NodeRef)>;

  GraphEvaluator(llvm::ThreadPool *threadPool);

  ~GraphEvaluator();

  // All predecessors must have been added beforehand.
  void addNode(NodeRef node, llvm::ArrayRef<NodeRef> predecessors, Eval eval);

  void evaluate(llvm::ArrayRef<NodeRef> nodes);

private:
  struct NodeRecord;
  using NodeEntryPtr = std::pair<const NodeRef, NodeRecord> *;
  struct NodeRecord {
    llvm::SmallVector<NodeEntryPtr, 2> predecessors;
    // Function to evaluate.
    // Set to nullptr after it has been queued for evaluation.
    Eval eval;
    // Who is assigned to evaluate. Cleared after completion.
    std::atomic<int> who = 0;

    bool isVisited() const { return eval == nullptr || who != 0; }
    bool isQueued() const { return eval == nullptr && who != 0; }
    bool isEvaluated() const { return eval == nullptr && who == 0; }
  };
  using Set = llvm::SmallPtrSet<NodeEntryPtr, 1>;
  static bool isEvaluated(NodeEntryPtr entry) {
    return entry->second.isEvaluated();
  }

  NodeEntryPtr lookup(NodeRef node);

  void enqueue(NodeEntryPtr entry);

  bool tryEvaluateNode(NodeEntryPtr entry, int me, Set &awaiting);

  llvm::ThreadPool *threadPool;
  std::mutex mux;
  std::condition_variable condition;
  std::unordered_map<NodeRef, NodeRecord> nodes;
  int whoCounter = 0;
};

} // namespace lazycst
