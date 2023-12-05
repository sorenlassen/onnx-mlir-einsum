/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/ConstantFoldableAnalysis.hpp"
#include "src/Dialect/LazyCst/ConstantFolder.hpp"
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"
#include "src/Dialect/LazyCst/LazyCstDialect.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <functional>
#include <optional>
#include <vector>

#define DEBUG_TYPE "lazy-constant-folding-pass"

namespace {

using namespace mlir;

// Return a vector of non-constant lazy foldable ops for every lazy constant.
// Outer and inner vectors are in reverse topological order: successors before
// predecessors.
std::vector<std::vector<Operation *>> lazyConstantResultOps(Operation *root,
    lazycst::ConstantFoldableAnalysis &analysis,
    std::vector<Operation *> &constants) {
  std::vector<std::vector<Operation *>> lazyConstantOps;
  // Maps every used, non-constant, lazy foldable op to a lazy constant's
  // index in lazyConstantOps.
  DenseMap<Operation *, size_t> lazyConstantMap;
  // Walk regions post-order and then the ops in each region backwards to
  // visit them in reverse topological order. This deals correctly with ops
  // like onnx.If whose sub-regions can use values from the parent region
  // while each sub-region's results have no uses outside the sub-region.
  root->walk([&](Region *region) {
    SmallVector<std::reference_wrapper<Operation>> ops(region->getOps());
    for (Operation &op : llvm::reverse(ops)) {
      if (lazycst::isConstant(&op)) {
        constants.push_back(&op);
        continue;
      }
      if (!analysis.isConstantFoldableOp(&op))
        continue;
      std::optional<size_t> idx;
      for (Operation *user : op.getUsers()) {
        assert(!lazycst::isConstant(user));
        if (analysis.isConstantFoldableOp(user)) {
          auto it = lazyConstantMap.find(user);
          assert(it != lazyConstantMap.end());
          size_t userIdx = it->second;
          if (!idx.has_value() || idx.value() == userIdx) {
            idx = userIdx;
            continue;
          }
        }
        // op has a non lazy foldable user, or it is used by two or more other
        // lazy constants; in either case, we make op a new lazy constant result
        idx = lazyConstantOps.size();
        lazyConstantOps.emplace_back();
        break;
      }
      if (!idx.has_value())
        continue;
      lazyConstantOps[idx.value()].push_back(&op);
      auto [_, inserted] = lazyConstantMap.try_emplace(&op, idx.value());
      assert(inserted);
    }
  });
  return lazyConstantOps;
}

void convertIntoLazyConstant(lazycst::CstExprEvaluator &cstexprEvaluator,
    SymbolTable &symbolTable, lazycst::ConstantFoldableAnalysis &analysis,
    const std::vector<Operation *> &ops) {
  assert(!ops.empty());
  Operation *resultOp = ops.front();
  bool onlyConstantFoldableUsers = llvm::all_of(resultOp->getUsers(),
      [&analysis](Operation *op) { return analysis.isConstantFoldableOp(op); });
  Location loc = resultOp->getLoc();
  OpBuilder b(resultOp->getContext());
  Block *block = new Block();
  b.setInsertionPointToStart(block);
  auto yield = b.create<lazycst::YieldOp>(loc, ValueRange{});
  b.setInsertionPoint(yield);

  SmallPtrSet<Operation *, 1> opsSet(ops.begin(), ops.end());
  const auto inOps = [&](Operation *op) { return opsSet.contains(op); };
  assert(!llvm::any_of(resultOp->getUsers(), inOps));

  llvm::SmallDenseMap<Attribute, Value> cloneConstants;
  SmallVector<Attribute> argsAttrs;
  IRMapping mapping;
  Operation *clone;
  for (Operation *op : llvm::reverse(ops)) {
    unsigned numOperands = op->getNumOperands();
    SmallVector<Value> cstOperands(numOperands, nullptr);
    for (unsigned i = 0; i < numOperands; ++i) {
      Value operand = op->getOperand(i);
      Operation *operandOp = operand.getDefiningOp();
      if (Attribute attr = lazycst::getConstantAttribute(operandOp)) {
        auto [it, inserted] = cloneConstants.try_emplace(attr, nullptr);
        Value &cst = it->second;
        if (inserted) {
          // TODO: consider including non-lazy cst attrs in args
          if (isa<lazycst::LazyElementsAttr>(attr)) {
            assert(block->getNumArguments() == argsAttrs.size());
            cst = block->addArgument(operand.getType(), operand.getLoc());
            argsAttrs.push_back(attr);
          } else {
            cst = b.clone(*operandOp)->getResult(0);
          }
        }
        cstOperands[i] = cst;
      }
    }

    clone = b.clone(*op, mapping);
    for (unsigned i = 0; i < numOperands; ++i) {
      if (Value cst = cstOperands[i])
        clone->setOperand(i, cst);
    }
  }
  // clone is now the clone of resultOp which was the op in the last iteration.

  for (unsigned j = 0; j < resultOp->getNumResults(); ++j) {
    auto res = resultOp->getResult(j);
    if (!res.use_empty()) {
      auto cloneRes = clone->getResult(j);
      yield.getOperandsMutable().append({cloneRes});
    }
  }
  assert(yield.getNumOperands() > 0);

  lazycst::CstExprOp cstexpr =
      lazycst::CstExprOp::create(symbolTable, loc, block, argsAttrs);
  auto resultsAttrs = cstexpr.getOutputsAttr().getValue();
  for (auto [cloneRes, lazyElms] :
      llvm::zip_equal(yield.getOperands(), resultsAttrs)) {
    // set res to the result of resultOp corresponding to cloneRes:
    unsigned j = cast<OpResult>(cloneRes).getResultNumber();
    auto res = resultOp->getResult(j);
    b.setInsertionPointAfter(resultOp);
    Dialect *dialect = resultOp->getName().getDialect();
    Operation *cstOp =
        dialect->materializeConstant(b, lazyElms, res.getType(), res.getLoc());
    res.replaceAllUsesWith(cstOp->getResult(0));
  }

  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE " cstexpr: " << cstexpr << "\n");
  assert(succeeded(verify(cstexpr)));

  cstexprEvaluator.record(cstexpr, onlyConstantFoldableUsers);

  for (Operation *op : ops) {
    assert(op->use_empty());
    op->erase();
  }
}

// ModuleOp pass: adds lazy constant expressions to the module region and
// symbol table.
struct LazyConstantFoldingPass
    : public PassWrapper<LazyConstantFoldingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LazyConstantFoldingPass);

  StringRef getArgument() const override {
    return "lazy-constant-folding-pass";
  }

  StringRef getDescription() const override {
    return "Lazily constant folds into lazy constants";
  }

  void runOnOperation() final {
    auto &cstexprEvaluator = getContext()
                                 .getLoadedDialect<lazycst::LazyCstDialect>()
                                 ->cstexprEvaluator;
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    lazycst::ConstantFoldableAnalysis analysis(module);
    std::vector<Operation *> constants;
    auto resultOps = lazyConstantResultOps(module, analysis, constants);
    for (const auto &ops : llvm::reverse(resultOps))
      convertIntoLazyConstant(cstexprEvaluator, symbolTable, analysis, ops);
    for (auto cst : constants) {
      if (cst->use_empty())
        cst->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createLazyConstantFoldingPass() {
  return std::make_unique<LazyConstantFoldingPass>();
}
