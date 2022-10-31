/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXAttributes.cpp -- ONNX attributes implementation --------===//
//
// ONNX attributes implementation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXAttributes.hpp"

#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp" // ONNXConstantOp

#include "mlir/IR/BuiltinDialect.h"

using namespace onnx_mlir;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)

namespace mlir {

size_t detail::uniqueNumber() {
  static std::atomic<size_t> counter{0};
  return ++counter;
}

void DisposableElementsAttr::printWithoutType(raw_ostream &os) const {
  printIntOrFPElementsAttrAsDenseWithoutType(*this, os);
}

/*static*/
DisposablePool &DisposablePool::create(MLIRContext *context) {
  return context->getLoadedDialect<BuiltinDialect>()
      ->addInterface<DisposablePool>(context);
}

/*static*/
DisposablePool *DisposablePool::get(MLIRContext *context) {
  return context->getLoadedDialect<BuiltinDialect>()
      ->getRegisteredInterface<DisposablePool>();
}

DisposablePool::DisposablePool(Dialect *dialect, MLIRContext *context)
    : Base(dialect), pool() {}
DisposablePool::~DisposablePool() {}

void DisposablePool::insert(DisposableElementsAttr d) {
  auto insertion = pool.insert(d.getImpl());
  if (!insertion.second)
    llvm_unreachable("cannot insert existing DisposableElementsAttr");
}

void DisposablePool::garbageCollectUnreachable(ModuleOp moduleOp) {
  Pool reachable;
  moduleOp.walk([&reachable, this](ONNXConstantOp constOp) {
    if (auto attr = constOp.value())
      if (auto elements = attr->dyn_cast<DisposableElementsAttr>()) {
        assert(this->pool.count(elements.getImpl()) == 1 &&
               "reachable disposables must be in the pool");
        reachable.insert(elements.getImpl());
      }
  });
  eraseUnreachable(reachable);
}

void DisposablePool::scrub(ModuleOp moduleOp) {
  moduleOp.walk([&](ONNXConstantOp constOp) {
    if (auto attr = constOp.value())
      if (auto elements = attr->dyn_cast<DisposableElementsAttr>()) {
        assert(this->pool.count(elements.getImpl()) == 1 &&
               "reachable disposables must be in the pool");
        RawBuffer rawBuffer = getDenseIntOrFPRawData(elements);
        constOp.valueAttr(DenseElementsAttr::getFromRawBuffer(
            elements.getType(), rawBuffer.get()));
      }
  });
  eraseUnreachable({});
}

void DisposablePool::eraseUnreachable(const Pool &reachable) {
  for (Pool::iterator it = pool.begin(); it != pool.end();) {
    DisposableElementsAttributeStorage *p = *it;
    if (pool.count(p) == 0) {
      // p is unreachable, so we reset the buffer payload shared_ptr
      // which decreases the reference count and, if it reached zero,
      // frees or closes the underlying MemoryBuffer's heap allocation or file.
      p->buffer.reset(); // Decrement reference count.
      it = pool.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace mlir