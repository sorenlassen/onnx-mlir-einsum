/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- ResourceGarbageCollector.cpp ---------------------===//
//
// Garbage collects DenseResourceElementsAttr attributes.
//
//===----------------------------------------------------------------------===//

#include "src/Transform/ResourceGarbageCollector.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Dialect/Mlir/ResourcePool.hpp"
#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/ONNXAttributes.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

void ResourceGarbageCollector::runAfterPass(Pass *pass, Operation *op) {
  ModuleOp moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp)
    return;
  ResourcePool::ResourceSet reachableResources;

  // // TODO: remove after testing
  // llvm::errs() << "ResourceGarbageCollector::runAfterPass ";
  // pass->printAsTextualPipeline(llvm::errs());
  // llvm::errs() << '\n';
  // op->dump();

  moduleOp.walk([&](ONNXConstantOp constOp) {
    assert(constOp->getAttrs().size() == 1);
    if (constOp.sparse_value())
      return;
    auto elements = constOp.valueAttr().cast<ElementsAttr>();
    if (elements.isa<DenseElementsAttr>())
      return;
    if (auto denseResrc = elements.dyn_cast<DenseResourceElementsAttr>()) {
      DenseResourceElementsHandle r = denseResrc.getRawHandle();
      auto insertion = reachableResources.insert(r);
      if (!insertion.second)
        llvm::errs() << "ResourceGarbageCollector::runAfterPass encountered "
                        "the same dense resource twice "
                     << denseResrc << "\n";
      return;
    }
    if (auto disposable = elements.dyn_cast<DisposableElementsAttr>()) {
      // llvm::errs() << "ResourceGarbageCollector::runAfterPass "
      //     "TODO: dispose DisposableAttributeResource " << disposable << "\n";
      return;
    }
    llvm_unreachable("unexpectected ElementsAttr");
  });
  resourcePool.garbageCollect(reachableResources);
}

namespace {

struct ScrubResourcesPass
    : public PassWrapper<ScrubResourcesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScrubResourcesPass)

  ScrubResourcesPass(ResourcePool *resourcePool) : resourcePool(resourcePool) {}

  StringRef getArgument() const override { return "scrub-resources"; }

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    moduleOp.walk([&](ONNXConstantOp constOp) {
      assert(constOp->getAttrs().size() == 1);
      if (constOp.sparse_value())
        return;
      auto elements = constOp.valueAttr().cast<ElementsAttr>();
      if (elements.isa<DenseElementsAttr>())
        return;
      assert(
          (elements.isa<DenseResourceElementsAttr, DisposableElementsAttr>()));
      RawBuffer rawBuffer = getDenseIntOrFPRawData(elements);
      constOp.valueAttr(DenseElementsAttr::getFromRawBuffer(
          elements.getType(), rawBuffer.get()));
    });
    // Every DenseResourceElementsAttr has been replaced in moduleOp. Free
    // all their blobs by closing ResourcePool.
    getResourcePool()->close();
    // TODO: similarly free every DisposableElementsAttr
  }

  ResourcePool *getResourcePool() {
    return resourcePool ? resourcePool : ResourcePool::get(&getContext());
  }

  ResourcePool *resourcePool;
};

} // namespace

std::unique_ptr<mlir::Pass> createScrubResourcesPass(
    ResourcePool *resourcePool) {
  return std::make_unique<ScrubResourcesPass>(resourcePool);
}

} // namespace onnx_mlir