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

#include "src/Dialect/Mlir/ResourcePool.hpp"
#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

void ResourceGarbageCollector::runAfterPass(Pass *pass, Operation *op) {
  ModuleOp moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp)
    return;
  ResourcePool::ResourceSet reachableResources;
  moduleOp.walk([&](ONNXConstantOp constOp) {
    if (auto attr = constOp.value())
      if (auto elements = attr->dyn_cast<DenseResourceElementsAttr>()) {
        DenseResourceElementsHandle r = elements.getRawHandle();
        auto insertion = reachableResources.insert(r);
        if (!insertion.second)
          llvm::errs() << "ResourceGarbageCollector::runAfterPass encountered "
                          "the same dense resource twice "
                       << elements << "\n";
      }
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
      if (auto attr = constOp.value())
        if (auto elements = attr->dyn_cast<DenseResourceElementsAttr>()) {
          RawBuffer rawBuffer = getDenseIntOrFPRawBytes(elements);
          constOp.valueAttr(makeDenseElementsAttrFromRawBytes(
              elements.getType(), rawBuffer.get()));
        }
    });
    // Every DenseResourceElementsAttr has been replaced in moduleOp. Free
    // all their blobs by closing ResourcePool.
    getResourcePool()->close();
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