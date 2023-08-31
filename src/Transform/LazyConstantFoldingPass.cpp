/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/ConstantFoldableAnalysis.hpp"
#include "src/Dialect/LazyCst/ConstantFolder.hpp"
#include "src/Dialect/LazyCst/LazyCst.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
using lazycst::ConstantFoldableAnalysis;

bool isConstant(Operation *op) {
  // TODO: consider using mlir::matchPattern(op, m_Constant())
  return op->hasTrait<OpTrait::ConstantLike>();
}

// Returns nullptr if v is not a constant result.
Attribute getConstantAttribute(Operation *op) {
  if (!isConstant(op))
    return nullptr;
  SmallVector<OpFoldResult, 1> folded;
  auto ok = op->fold(folded);
  assert(succeeded(ok));
  assert(folded.size() == 1);
  assert(folded.front().is<Attribute>());
  return folded.front().get<Attribute>();
}

// Return a vector of non-constant lazy foldable ops for every lazy constant.
// Outer and inner vectors are in reverse topological order: successors before
// predecessors.
std::vector<std::vector<Operation *>> lazyConstantResultOps(
    func::FuncOp function) {
  std::vector<std::vector<Operation *>> lazyConstantOps;
  ConstantFoldableAnalysis analysis(function);
  // Maps every used, non-constant, lazy foldable op to a lazy constant's
  // index in lazyConstantOps.
  DenseMap<Operation *, size_t> lazyConstantMap;
  // Walk regions post-order and then the ops in each region backwards to
  // visit them in reverse topological order. This deals correctly with ops
  // like onnx.If whose sub-regions can use values from the parent region
  // while each sub-region's results have no uses outside the sub-region.
  function->walk([&](Region *region) {
    SmallVector<std::reference_wrapper<Operation>> ops(region->getOps());
    for (Operation &op : llvm::reverse(ops)) {
      if (isConstant(&op))
        continue;
      if (!analysis.isConstantFoldableOp(&op))
        continue;
      std::optional<size_t> idx;
      for (Operation *user : op.getUsers()) {
        assert(!isConstant(user));
        if (analysis.isConstantFoldableOp(user)) {
          auto it = lazyConstantMap.find(user);
          if (it == lazyConstantMap.end())
            continue;
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

void convertIntoLazyConstant(const std::vector<Operation *> &ops,
    StringAttr lazyFuncName, SymbolTable &symbolTable) {
  auto module = cast<ModuleOp>(symbolTable.getOp());
  OpBuilder b(module.getBodyRegion());

  assert(!ops.empty());
  Operation *resultOp = ops.front();
  Location loc = resultOp->getLoc();
  auto cstexpr = b.create<lazycst::LazyFuncOp>(loc, lazyFuncName);
  symbolTable.insert(cstexpr);

  b.setInsertionPointToStart(cstexpr.addEntryBlock());
  auto lazyReturn = b.create<lazycst::LazyReturnOp>(loc, ValueRange{});
  b.setInsertionPoint(lazyReturn);

  SmallPtrSet<Operation *, 1> opsSet(ops.begin(), ops.end());
  const auto inOps = [&](Operation *op) { return opsSet.contains(op); };
  assert(!llvm::any_of(resultOp->getUsers(), inOps));

  // TODO: consider making it a vector set to ensure determinism
  SmallPtrSet<Operation *, 4> unreachableConstants;
  llvm::SmallDenseMap<Attribute, Value> cloneConstants;
  SmallVector<Attribute> lazyArguments;
  IRMapping mapping;
  Operation *clone;
  for (Operation *op : llvm::reverse(ops)) {
    unsigned numOperands = op->getNumOperands();
    SmallVector<Value> cstOperands(numOperands, nullptr);
    for (unsigned i = 0; i < numOperands; ++i) {
      Value operand = op->getOperand(i);
      Operation *operandOp = operand.getDefiningOp();
      if (Attribute attr = getConstantAttribute(operandOp)) {
        auto [it, inserted] = cloneConstants.try_emplace(attr, nullptr);
        Value &cst = it->second;
        if (inserted) {
          if (isa<lazycst::LazyElementsAttr, lazycst::FileDataElementsAttr>(
                  attr)) {
            Region &body = cstexpr.getBody();
            assert(body.getNumArguments() == lazyArguments.size());
            cst = body.addArgument(operand.getType(), operand.getLoc());
            lazyArguments.push_back(attr);
          } else {
            cst = b.clone(*operandOp)->getResult(0);
          }
        }
        cstOperands[i] = cst;
        if (llvm::all_of(operandOp->getUsers(), inOps))
          unreachableConstants.insert(operandOp);
      }
    }

    clone = b.clone(*op, mapping);
    for (unsigned i = 0; i < numOperands; ++i) {
      if (Value cst = cstOperands[i])
        clone->setOperand(i, cst);
    }
  }
  // clone is now the clone of resultOp which was the op in the last iteration.

  SmallVector<Attribute> lazyResults;
  {
    auto lazyFunc = FlatSymbolRefAttr::get(lazyFuncName);
    unsigned numResults = resultOp->getNumResults();
    for (unsigned j = 0; j < numResults; ++j) {
      Value res = resultOp->getResult(j);
      if (res.use_empty())
        continue;
      auto type = cast<ShapedType>(res.getType());
      unsigned index = lazyResults.size();
      auto lazyElms = lazycst::LazyElementsAttr::get(type, lazyFunc, index);
      lazyResults.push_back(lazyElms);
      lazyReturn.getOperandsMutable().append({clone->getResult(j)});
      b.setInsertionPointAfter(resultOp);
      Dialect *dialect = resultOp->getName().getDialect();
      Operation *cstOp =
          dialect->materializeConstant(b, lazyElms, type, res.getLoc());
      res.replaceAllUsesWith(cstOp->getResult(0));
    }
  }
  assert(!lazyResults.empty());

  const auto getAttrType = [](Attribute ta) {
    return cast<TypedAttr>(ta).getType();
  };
  SmallVector<Type> argTypes(llvm::map_range(lazyArguments, getAttrType));
  SmallVector<Type> resTypes(llvm::map_range(lazyResults, getAttrType));
  cstexpr.setFunctionType(b.getFunctionType(argTypes, resTypes));
  cstexpr.setArgConstantsAttr(b.getArrayAttr(ArrayRef(lazyArguments)));
  cstexpr.setResConstantsAttr(b.getArrayAttr(ArrayRef(lazyResults)));

  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE " cstexpr: " << cstexpr << "\n");
  assert(succeeded(verify(cstexpr)));

  for (Operation *op : ops) {
    assert(op->use_empty());
    op->erase();
  }
  for (Operation *op : unreachableConstants) {
    assert(op->use_empty());
    op->erase();
  }
}

struct LazyConstantFoldingPass
    : public PassWrapper<LazyConstantFoldingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LazyConstantFoldingPass);

  StringRef getArgument() const override {
    return "lazy-constant-folding-pass";
  }

  StringRef getDescription() const override {
    return "Lazily constant folds into lazy constants";
  }

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    auto resultOps = lazyConstantResultOps(function);
    auto module = function->getParentOfType<ModuleOp>();
    SymbolTable symbolTable(module);
    auto *lazyCstDialect =
        getContext().getLoadedDialect<lazycst::LazyCstDialect>();
    for (const auto &ops : llvm::reverse(resultOps)) {
      StringAttr lazyFuncName =
          lazyCstDialect->lazyFunctionManager.nextName(module);
      convertIntoLazyConstant(ops, lazyFuncName, symbolTable);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::createLazyConstantFoldingPass() {
  return std::make_unique<LazyConstantFoldingPass>();
}
