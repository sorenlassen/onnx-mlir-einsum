/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/Attributes.h"

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
#include <utility>

namespace mlir {
class Operation;
}

namespace lazycst {

class GraphEvaluator {
public:
  using NodeOperand = std::pair<mlir::Operation *, unsigned>;
  using Fold =
      std::function<void(mlir::Operation *, llvm::ArrayRef<mlir::Attribute>,
          llvm::SmallVectorImpl<mlir::Attribute> &)>;

  GraphEvaluator(llvm::ThreadPool *threadPool);

  ~GraphEvaluator();

  // All operand ops must have been added beforehand.
  void addNode(mlir::Operation *op, llvm::ArrayRef<NodeOperand> operands,
      Fold fold, bool onlyUsedWithinGraph = true);

  void evaluate(llvm::ArrayRef<mlir::Operation *> ops,
      llvm::SmallVectorImpl<llvm::ArrayRef<mlir::Attribute>> &results);

private:
  struct OpRecord;

  using OpEntry = std::pair<mlir::Operation *const, OpRecord>;

  using OpEntryOperand = std::pair<OpEntry *, unsigned>;

  struct OpRecord {
    llvm::SmallVector<mlir::Attribute, 1> results;
    llvm::SmallVector<OpEntryOperand> operands;
    // Add nullptr to users to represent any out-of-graph users.
    // TODO: consider attaching users to each result instead  of while OpRecord
    llvm::SmallPtrSet<OpEntry *, 1> users;
    // Function to fold.
    // Set to nullptr after it has been queued for folding.
    Fold fold;
    // Who is assigned to fold. Cleared after completion.
    // TODO: consider removing atomic and access under mutex, see: t.ly/DdLO3
    std::atomic<int> who = 0;

    bool isVisited() const { return fold == nullptr || who != 0; }
    bool isQueued() const { return fold == nullptr && who != 0; }
    bool isFolded() const { return fold == nullptr && who == 0; }
  };

  using Set = llvm::SmallPtrSet<OpEntry *, 1>;

  static bool isOperandFolded(OpEntryOperand operand) {
    return operand.first->second.isFolded();
  }

  OpEntry *lookup(mlir::Operation *op);

  void enqueue(OpEntry *entry);

  bool tryFoldNode(OpEntry *entry, int me, Set &awaiting);

  llvm::ThreadPool *threadPool;
  std::mutex mux;
  std::condition_variable condition;
  std::unordered_map<mlir::Operation *, OpRecord> nodes;
  int whoCounter = 0;
};

} // namespace lazycst
