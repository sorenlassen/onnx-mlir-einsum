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
  llvm_unreachable("TODO: implement DisposablePool::garbageCollectUnreachable");
}

void DisposablePool::scrub(ModuleOp moduleOp) {
  llvm_unreachable("TODO: implement DisposablePool::scrub");
}

} // namespace mlir