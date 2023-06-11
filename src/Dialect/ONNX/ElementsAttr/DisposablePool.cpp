/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- DisposablePool.cpp ------------------------===//
//
// DisposablePool manages instances of DisposableElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/DisposablePool.hpp"

#include "mlir/IR/Threading.h"
#include "llvm/ADT/DenseMap.h"

#include <atomic>

using namespace mlir;

namespace onnx_mlir {

DisposablePool::DisposablePool(Dialect *dialect, MLIRContext *context)
    : Base(dialect), pool(), mutex() {}
DisposablePool::~DisposablePool() {}

ElementsAttr DisposablePool::createElementsAttr(ShapedType type,
    BType bufferBType, ArrayRef<int64_t> strides,
    const DisposableElementsAttr::Buffer &buffer,
    DisposableElementsAttr::Transformer transformer) {
  static std::atomic<size_t> counter{0};
  size_t id = ++counter;
  auto disposable = DisposableElementsAttr::create(
      type, id, bufferBType, strides, buffer, std::move(transformer));
  if (insert(disposable)) {
    return disposable;
  } else {
    auto dense = disposable.toDenseElementsAttr();
    disposable.dispose();
    return dense;
  }
}

namespace {
template <typename Action>
void walkOpsAttrs(ModuleOp moduleOp, DisposablePool::OpAttrDictionary opsAttrs,
    const Action &act) {
  llvm::SmallDenseMap<StringRef, StringRef> opAttrMap(
      opsAttrs.begin(), opsAttrs.end());
  moduleOp.walk([&opAttrMap, &act](Operation *op) {
    auto opAttr = opAttrMap.find(op->getName().getIdentifier());
    if (opAttr != opAttrMap.end()) {
      StringRef attrName = opAttr->second;
      if (auto attr = op->getAttrOfType<DisposableElementsAttr>(attrName))
        act(op, attrName, attr);
    }
  });
}
} // namespace

void DisposablePool::garbageCollectUnreachable(
    ModuleOp moduleOp, OpAttrDictionary opsAttrs) {
  Pool reachable;
  walkOpsAttrs(moduleOp, opsAttrs,
      [&reachable](Operation *op, StringRef attrName,
          DisposableElementsAttr disposable) {
        reachable.try_emplace(disposable.getId(), disposable);
      });

  {
    const std::lock_guard<std::mutex> lock(mutex);

    for (auto &entry : reachable)
      assert(pool.count(entry.first) == 1 &&
             "reachable disposables must be in the pool");
    eraseUnreachable(reachable);
  }
}

void DisposablePool::scrub(ModuleOp moduleOp, OpAttrDictionary opsAttrs) {
  using Translation = std::pair<DisposableElementsAttr, DenseElementsAttr>;
  std::unordered_map<size_t, Translation> translations;
  walkOpsAttrs(moduleOp, opsAttrs,
      [&translations](Operation *op, StringRef attrName,
          DisposableElementsAttr disposable) {
        translations.try_emplace(
            disposable.getId(), std::make_pair(disposable, nullptr));
      });
  {
    std::mutex translationMutex;
    auto it = translations.begin();
    auto nextBatch = [&translationMutex, &translations, &it]() {
      const std::lock_guard<std::mutex> lock(translationMutex);
      auto batchBegin = it;
      // TODO: Consider incrementing it more than once to create batch longer
      //       than one when the ElementsAttrs are small so we don't have
      //       to call nextBatch and grab translationMutex as often.
      if (it != translations.end())
        ++it;
      auto batchEnd = it;
      return llvm::make_range(batchBegin, batchEnd);
    };
    MLIRContext *ctx = moduleOp.getContext();
    parallelFor(ctx, 0, ctx->getNumThreads(), [&](size_t threadNumber) {
      for (;;) {
        auto batch = nextBatch();
        if (batch.empty())
          break;
        for (auto &[id, translation] : batch) {
          auto &[disposable, dense] = translation;
          dense = disposable.toDenseElementsAttr();
          // TODO: Consider calling disposable.dispose() here to free up memory
          //       on the go to make memory available to create the next
          //       DenseElementsAttr. In that case should we lock mutex?
        }
      }
    });
  }
  walkOpsAttrs(moduleOp, opsAttrs,
      [&translations](Operation *op, StringRef attrName,
          DisposableElementsAttr disposable) {
        DenseElementsAttr dense = translations.at(disposable.getId()).second;
        op->setAttr(attrName, dense);
      });

  {
    const std::lock_guard<std::mutex> lock(mutex);

    for (const auto &t : translations)
      assert(pool.count(t.first) == 1 &&
             "scrubbed disposables must be in the pool");
    eraseUnreachable({});
  }
}

void DisposablePool::close() {
  const std::lock_guard<std::mutex> lock(mutex);

  assert(pool.empty() && "pool must be scrubbed before close");
  active = false;
}

bool DisposablePool::isActive() const {
  const std::lock_guard<std::mutex> lock(mutex);

  return active;
}

bool DisposablePool::insert(DisposableElementsAttr disposable) {
  const std::lock_guard<std::mutex> lock(mutex);

  if (!active)
    return false;

  auto insertion = pool.try_emplace(disposable.getId(), disposable);
  if (!insertion.second)
    llvm_unreachable("cannot insert existing DisposableElementsAttr");
  return true;
}

void DisposablePool::eraseUnreachable(const Pool &reachable) {
  // Assumes caller holds the mutex.
  for (Pool::iterator it = pool.begin(); it != pool.end();) {
    if (reachable.count(it->first) == 0) {
      // The attribute is unreachable, so we reset the buffer payload shared_ptr
      // which decreases the reference count and, if it reached zero,
      // frees or closes the underlying MemoryBuffer's heap allocation or file.
      it->second.dispose();
      it = pool.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace onnx_mlir