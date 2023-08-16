// SPDX-License-Identifier: Apache-2.0

#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace {

using namespace mlir;
using namespace onnx_mlir;

// Populated by configureLazyConstPropONNXPass().
struct LazyConstPropONNXPassConfiguration {
  static int expansionBound;
};

int LazyConstPropONNXPassConfiguration::expansionBound = -1; // -1 == no bound

struct LazyConstPropONNXPass
    : public PassWrapper<LazyConstPropONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LazyConstPropONNXPass);

  StringRef getArgument() const override { return "lazy-constprop-onnx"; }

  StringRef getDescription() const override {
    return "Lazy constant propagation for the ONNX dialect.";
  }

  void runOnOperation() final;

private:
  void runOnRegion(Region *region, SmallVectorImpl<Region *> &regionQueue);
};

void LazyConstPropONNXPass::runOnOperation() {
  func::FuncOp function = getOperation();
  SmallVector<Region *> regionQueue({&function.getFunctionBody()});
  while (!regionQueue.empty())
    runOnRegion(regionQueue.pop_back_val(), regionQueue);
}

bool isaConstantOp(Operation *op) {
  return op && op->hasTrait<OpTrait::ConstantLike>();
}

bool isaConstantValue(Value v) {
  return isaConstantOp(v.getDefiningOp());
}

bool isLazyFoldable(Operation *op) {
  llvm_unreachable("TODO: implement this");
}

void LazyConstPropONNXPass::runOnRegion(Region *region, SmallVectorImpl<Region *> &regionQueue) {
  SmallVector<Operation *> opQueue;
  DenseMap<Operation *, size_t> opMap;
  using Span = std::pair<size_t, size_t>;

  auto constantify = [&](Value v, const Span &span) {
    assert(span.first <= span.second);
    if (span.first == span.second) {
      assert(isaConstantValue(v));
      return;
    }
    assert(v.getDefiningOp() == opQueue[span.second - 1]);
    // TODO: put opQueue[span.first:span.second] into a lazy func
    //       and make all the results with users outside the
    //       span into lazy func results,
    //       while checking that no users are before the span
    llvm_unreachable("TODO: implement this");
  };

  // Returns a span if the value can be constantified, otherwise nulltopt
  auto traverse = [&](const auto &recurse, Value v) -> std::optional<Span> {
    Operation *defop = v.getDefiningOp();

    auto begin = opQueue.size();
    if (isaConstantOp(defop) || opMap.contains(defop))
      return Span(begin, begin);

    for (auto &subregion : defop->getRegions())
      regionQueue.push_back(&subregion);

    bool isFoldable = isLazyFoldable(defop);
    int numOperands = defop->getNumOperands();
    SmallVector<std::optional<Span>> spans(numOperands, std::nullopt);
    for (int i = 0; i < numOperands; ++i) {
      if ((spans[i] = recurse(recurse, defop->getOperand(i))).has_value()) {
        if (!isFoldable)
          constantify(defop->getOperand(i), *spans[i]);
      } else {
        if (isFoldable) {
          for (int j = 0; j < i; ++j)
            constantify(defop->getOperand(j), *spans[j]);
        }
        isFoldable = false;
      }
    }
    if (!isFoldable)
      return std::nullopt;

    auto pos = opQueue.size();
    opQueue.push_back(defop);
    auto [_, inserted] = opMap.try_emplace(defop, pos);
    assert(inserted);
    auto end = opQueue.size();
    return Span(begin, end);
  };

  Operation *terminator = region->back().getTerminator();
  int numOperands = terminator->getNumOperands();
  for (int i = 0; i < numOperands; ++i) {
    if (auto span = traverse(traverse, terminator->getOperand(i)))
      constantify(terminator->getOperand(i), *span);
  }
}

}

void onnx_mlir::configureLazyConstPropONNXPass(int expansionBound) {
  LazyConstPropONNXPassConfiguration::expansionBound = expansionBound;
}

std::unique_ptr<mlir::Pass> onnx_mlir::createLazyConstPropONNXPass() {
  return std::make_unique<LazyConstPropONNXPass>();
}
