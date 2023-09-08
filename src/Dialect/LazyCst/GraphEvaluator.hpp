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
  using Eval = std::function<void(mlir::Operation *)>;

  GraphEvaluator(llvm::ThreadPool *threadPool);

  ~GraphEvaluator();

  // All predecessors must have been added beforehand.
  void addNode(mlir::Operation *node,
      llvm::ArrayRef<mlir::Operation *> predecessors, Eval eval);

  void evaluate(llvm::ArrayRef<mlir::Operation *> nodes);

private:
  struct OpRecord;
  using OpEntry = std::pair<mlir::Operation *const, OpRecord>;
  struct OpRecord {
    llvm::SmallVector<OpEntry *, 2> predecessors;
    // Function to evaluate.
    // Set to nullptr after it has been queued for evaluation.
    Eval eval;
    // Who is assigned to evaluate. Cleared after completion.
    std::atomic<int> who = 0;

    bool isVisited() const { return eval == nullptr || who != 0; }
    bool isQueued() const { return eval == nullptr && who != 0; }
    bool isEvaluated() const { return eval == nullptr && who == 0; }
  };
  using Set = llvm::SmallPtrSet<OpEntry *, 1>;
  static bool isEvaluated(OpEntry *entry) {
    return entry->second.isEvaluated();
  }

  OpEntry *lookup(mlir::Operation *node);

  void enqueue(OpEntry *entry);

  bool tryEvaluateNode(OpEntry *entry, int me, Set &awaiting);

  llvm::ThreadPool *threadPool;
  std::mutex mux;
  std::condition_variable condition;
  std::unordered_map<mlir::Operation *, OpRecord> nodes;
  int whoCounter = 0;
};

} // namespace lazycst
