/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- DisposableElementsAttr.cpp --------------------===//
//
// DisposableElementsAttr, garbage collectible alternative to DenseElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"

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

namespace {
template <DType DTYPE>
void identityReader(StringRef s, MutableArrayRef<WideNum> dst) {
  using X = CppType<DTYPE>;
  auto src = asArrayRef<X>(s);
  assert(src.size() == dst.size());
  std::transform(src.begin(), src.end(), dst.begin(),
      [](X x) { return WideNum::from<X>(DTYPE, x); });
}
} // namespace

auto DisposableElementsAttr::getIdentityReader(onnx_mlir::DType dtype)
    -> Reader {
  return dispatchByDType(
      dtype, [](auto staticDType) { return identityReader<staticDType>; });
}

auto DisposableElementsAttr::getSplatReader(
    onnx_mlir::DType dtype, StringRef rawBytes) -> Reader {
  unsigned bytewidth = bytewidthOfDType(dtype);
  ArrayRef<char> memory = asArrayRef(rawBytes.take_front(bytewidth));
  WideNum splatValue = WideNum::load(dtype, memory);
  return [=](StringRef s, MutableArrayRef<WideNum> dst) {
    assert(s.size() == bytewidth);
    assert(dst.size() == 1);
    *dst.begin() = splatValue;
  };
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
        ArrayBuffer<char> rawBuffer = getElementsRawBytes(elements);
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
      p->buffer.reset();
      p->reader = nullptr;
      it = pool.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace mlir