// SPDX-License-Identifier: Apache-2.0

#include "src/Dialect/LazyCst/LazyCst.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lazy-constprop-onnx"

#define USE_FOLD_ADAPTOR 1

namespace {

using namespace mlir;
using namespace onnx_mlir;

// Populated by configureLazyConstPropONNXPass().
struct LazyConstPropONNXPassConfiguration {
  static int expansionBound;
};

int LazyConstPropONNXPassConfiguration::expansionBound = -1; // -1 == no bound

// Similar to OpConversionPattern or Operation::fold().
class LazyFolder {
public:
  virtual ~LazyFolder() = default;

  virtual LogicalResult match(Operation *op) const = 0;
  virtual void fold(Operation *op, ArrayRef<Attribute> operands,
      SmallVectorImpl<Attribute> &results) const = 0;
};

class AlwaysMatchLazyFolder : public LazyFolder {
public:
  LogicalResult match(Operation *op) const override { return success(); }
};

template <typename OP>
class OpLazyFolder : public LazyFolder {
public:
  using Op = OP;
  virtual ~OpLazyFolder() = default;

  virtual LogicalResult match(OP op) const { return success(); }
  LogicalResult match(Operation *op) const override {
    return match(cast<OP>(op));
  }

#if USE_FOLD_ADAPTOR
  using FoldAdaptor = typename OP::FoldAdaptor;
  virtual Attribute fold(OP op, FoldAdaptor adaptor) const {
    llvm_unreachable("unimplemented");
  }
  virtual void fold(
      OP op, FoldAdaptor adaptor, SmallVectorImpl<Attribute> &results) const {
    results.emplace_back(fold(op, adaptor));
  }
  virtual void fold(Operation *op, ArrayRef<Attribute> operands,
      SmallVectorImpl<Attribute> &results) const override {
    return fold(cast<OP>(op),
        FoldAdaptor(operands, op->getAttrDictionary(),
            op->getPropertiesStorage(), op->getRegions()),
        results);
  }
#else
  virtual Attribute fold(OP op, ArrayRef<Attribute> operands) const {
    llvm_unreachable("unimplemented");
  }
  virtual void fold(OP op, ArrayRef<Attribute> operands,
      SmallVectorImpl<Attribute> &results) const {
    results.emplace_back(fold(op), operands);
  }
  virtual void fold(Operation *op, ArrayRef<Attribute> operands,
      SmallVectorImpl<Attribute> &results) const override {
    return fold(cast<OP>(op), operands, results);
  }
#endif
};

// Extracts number from a scalar elements attribute.
WideNum getScalarNum(ElementsAttr elements) {
  Type elementType = elements.getElementType();
  if (isa<FloatType>(elementType)) {
    APFloat f = *elements.value_begin<APFloat>();
    return WideNum::fromAPFloat(f);
  } else if (auto itype = dyn_cast<IntegerType>(elementType)) {
    APInt i = *elements.value_begin<APInt>();
    return WideNum::fromAPInt(i, !itype.isUnsigned());
  } else {
    llvm_unreachable("Only integer and float types are supported");
  }
}

class ONNXRangeOpLazyFolder : public OpLazyFolder<ONNXRangeOp> {
public:
#if USE_FOLD_ADAPTOR
  virtual Attribute fold(ONNXRangeOp op, FoldAdaptor adaptor) const override {
    ElementsAttr start = cast<ElementsAttr>(adaptor.getStart());
    ElementsAttr delta = cast<ElementsAttr>(adaptor.getDelta());
    ShapedType type = cast<ShapedType>(op.getType());
    OnnxElementsAttrBuilder eb(op.getContext());
    return eb.range(type, getScalarNum(start), getScalarNum(delta));
  }
#else
  virtual Attribute fold(
      ONNXRangeOp op, ArrayRef<Attribute> operands) const override {
    assert(operands.size() == 3);
    ElementsAttr start = cast<ElementsAttr>(operands[0]);
    ElementsAttr delta = cast<ElementsAttr>(operands[2]);
    ShapedType type = cast<ShapedType>(op.getType());
    OnnxElementsAttrBuilder eb(op.getContext());
    return eb.range(type, getScalarNum(start), getScalarNum(delta));
  }
#endif
};

struct LazyOpFolders {
  template <class T>
  void insert() {
    auto [_, inserted] =
        map.try_emplace(T::Op::getOperationName(), std::make_unique<T>());
    assert(inserted);
  }
  LazyOpFolders() {
    // TODO: move map initialization elsewhere
    insert<OpLazyFolder<ONNXAddOp>>();
    insert<ONNXRangeOpLazyFolder>();
  }
  llvm::StringMap<std::unique_ptr<LazyFolder>> map;
};

LazyOpFolders lazyOpFolders;

bool isLazyFoldable(Operation *op) {
  auto it = lazyOpFolders.map.find(op->getName().getIdentifier());
  return it != lazyOpFolders.map.end() && succeeded(it->second->match(op));
}

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
  // LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE " runOnOperation: " << function <<
  // "\n");
  SmallVector<Region *> regionQueue({&function.getFunctionBody()});
  while (!regionQueue.empty())
    runOnRegion(regionQueue.pop_back_val(), regionQueue);
}

bool isConstant(Operation *op) {
  return op && op->hasTrait<OpTrait::ConstantLike>();
}

bool isConstantResult(Value v) { return isConstant(v.getDefiningOp()); }

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

void LazyConstPropONNXPass::runOnRegion(
    Region *region, SmallVectorImpl<Region *> &regionQueue) {
  // LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE " runOnRegion: " <<
  // *region->getParentOp() << "\n");

  SmallVector<Operation *> opQueue;
  DenseMap<Operation *, size_t> opMap;
  using Span = std::pair<size_t, size_t>;

  auto constantify = [&](Value v, const Span &span) {
    assert(!v.use_empty());

    if (span.first == span.second) {
      assert(isConstantResult(v));
      return;
    }

    // Put opQueue[span.first:span.second] into a lazy func and make all the
    // results with users outside the span into lazy func results,
    // while checking that no users are before the span.
    assert(span.first < span.second);
    Operation *defop = v.getDefiningOp();
    assert(defop == opQueue[span.second - 1]);
    ModuleOp module = defop->getParentOfType<ModuleOp>();
    OpBuilder b(module.getBodyRegion());
    MLIRContext *ctx = b.getContext();
    auto *lazyCstDialect = ctx->getLoadedDialect<lazycst::LazyCstDialect>();
    StringAttr lazyFuncName =
        lazyCstDialect->lazyFunctionManager.nextName(module);
    auto lazyFunc = FlatSymbolRefAttr::get(lazyFuncName);

    Location loc = defop->getLoc();
    auto noFuncType = b.getFunctionType({}, {});
    auto noArray = b.getArrayAttr({});
    auto cstexpr = b.create<lazycst::LazyFuncOp>(
        loc, lazyFuncName, noFuncType, noArray, noArray, nullptr, nullptr);
    SymbolTable(module).insert(cstexpr);
    b.setInsertionPointToStart(cstexpr.addEntryBlock());
    auto lazyReturn = b.create<lazycst::LazyReturnOp>(loc, ValueRange{});
    b.setInsertionPoint(lazyReturn);

    const auto outsideOp = [&](Operation *op) {
      auto it = opMap.find(op);
      if (it != opMap.end()) {
        auto pos = it->second;
        assert(opQueue[pos] == op && "opMap/opQueue invariant");
        assert(span.first <= pos);
        if (pos < span.second)
          return false;
      }
      return true;
    };
    const auto outsideOperand = [&outsideOp](OpOperand &operand) {
      return outsideOp(operand.getOwner());
    };
    assert(llvm::all_of(v.getUsers(), outsideOp));

    // TODO: consider making it a vector set to ensure determinism
    SmallPtrSet<Operation *, 4> unreachableConstants;
    llvm::SmallDenseMap<Attribute, Value> cloneConstants;
    SmallVector<Attribute> lazyArguments;
    SmallVector<Attribute> lazyResults;
    IRMapping mapping;
    for (auto [pos, end] = span; pos < end; ++pos) {
      Operation *op = opQueue[pos];

      unsigned numOperands = op->getNumOperands();
      SmallVector<Value> cstOperands(numOperands, nullptr);
      for (unsigned i = 0; i < numOperands; ++i) {
        Value operand = op->getOperand(i);
        Operation *operandOp = operand.getDefiningOp();
        if (Attribute attr = getConstantAttribute(operandOp)) {
          Value cst = cloneConstants.lookup(attr);
          if (!cst) {
            if (isa<lazycst::LazyElementsAttr, lazycst::FileDataElementsAttr>(
                    attr)) {
              Region &body = cstexpr.getBody();
              assert(body.getNumArguments() == lazyArguments.size());
              cst = body.addArgument(operand.getType(), operand.getLoc());
              lazyArguments.push_back(attr);
            } else {
              cst = b.clone(*operandOp)->getResult(0);
            }
            auto [_, inserted] = cloneConstants.try_emplace(attr, cst);
            assert(inserted);
          }
          cstOperands[i] = cst;
          if (!llvm::any_of(operandOp->getUsers(), outsideOp))
            unreachableConstants.insert(operandOp);
        }
      }

      Operation *clone = b.clone(*op, mapping);
      for (unsigned i = 0; i < numOperands; ++i) {
        if (Value cst = cstOperands[i])
          clone->setOperand(i, cst);
      }

      {
        OpBuilder::InsertionGuard guard(b);
        unsigned numResults = op->getNumResults();
        for (unsigned j = 0; j < numResults; ++j) {
          Value res = op->getResult(j);
          if (llvm::any_of(res.getUsers(), outsideOp)) {
            auto type = cast<ShapedType>(res.getType());
            unsigned index = lazyResults.size();
            auto lazyElms =
                lazycst::LazyElementsAttr::get(type, lazyFunc, index);
            lazyResults.push_back(lazyElms);
            lazyReturn.getOperandsMutable().append({clone->getResult(j)});
            b.setInsertionPointAfter(defop);
            Value cst =
                op->getName()
                    .getDialect()
                    ->materializeConstant(b, lazyElms, type, res.getLoc())
                    ->getResult(0);
            res.replaceUsesWithIf(cst, outsideOperand);
          }
        }
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

    for (auto [begin, pos] = span; pos > begin; --pos) {
      Operation *op = opQueue[pos - 1];
      assert(op->use_empty());
      // TODO: determine if operands are removed from use/def lists, or if we
      //       should walk from front and dropAllUses() before erase()
      op->erase();
    }
    for (Operation *op : unreachableConstants) {
      assert(op->use_empty());
      op->erase();
    }

    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE " ranOnRegion: "
                            << *region->getParentOp() << "\n");
    assert(succeeded(verify(region->getParentOp())));
  };

  // Returns a span if the value can be constantified, otherwise nulltopt
  auto traverse = [&](const auto &recurse, Value v) -> std::optional<Span> {
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE " traverse: " << v << "\n");

    Operation *defop = v.getDefiningOp();

    auto begin = opQueue.size();
    if (isConstant(defop) || opMap.contains(defop))
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

} // namespace

void onnx_mlir::configureLazyConstPropONNXPass(int expansionBound) {
  LazyConstPropONNXPassConfiguration::expansionBound = expansionBound;
}

std::unique_ptr<mlir::Pass> onnx_mlir::createLazyConstPropONNXPass() {
  return std::make_unique<LazyConstPropONNXPass>();
}
