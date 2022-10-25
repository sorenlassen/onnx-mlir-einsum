/*
 * SPDX-License-Identifier: Apache-2.0
 */

//============-- TestONNXAttributes.cpp - ONNXAttributes tests --=============//
//
// Tests Disposable*ElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXAttributes.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

#include "mlir/IR/Builders.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace mlir;
using namespace onnx_mlir;

namespace {

typedef llvm::SmallVector<int64_t> Shape;

std::ostream &operator<<(std::ostream &os, const ArrayRef<int64_t> &v) {
  os << "(";
  for (auto i : v)
    os << i << ",";
  os << ")";
  return os;
}

MLIRContext *createCtx() {
  MLIRContext *ctx = new MLIRContext();
  ctx->loadDialect<ONNXDialect>();
  return ctx;
}

template <typename Src, typename Dst = char>
ArrayRef<Dst> castArrayRef(ArrayRef<Src> a) {
  return llvm::makeArrayRef(reinterpret_cast<const Dst*>(a.data()), a.size() / sizeof(Src));
}

template <typename T>
std::shared_ptr<llvm::MemoryBuffer> buffer(ArrayRef<T> data) {
  StringRef s(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
  return std::shared_ptr<llvm::MemoryBuffer>(llvm::MemoryBuffer::getMemBufferCopy(s));
}

class Test {
  std::unique_ptr<MLIRContext> ctx;
  Location loc;
  OpBuilder builder;
  Type F32;
  Type I32;

public:
  Test() : ctx(createCtx()), loc(UnknownLoc::get(&*ctx)), builder(&*ctx) {
    F32 = builder.getF32Type();
    I32 = builder.getI32Type();
  }

  int test_PosIterator() {
    int64_t shape[] = {1, 2, 3};
    int64_t strides[] = {0, 1, 2};
    detail::PosIterator begin(shape, strides);
    std::cerr << *begin << "\n";
    std::cerr << *++begin << "\n";
    begin++;
    auto end = detail::PosIterator::end(shape, strides);
    assert(begin != end);
    while (begin != end) { std::cerr << *begin << "\n"; ++begin; }
    return 0;
  }

  int test_attributes() {
    ShapedType type = RankedTensorType::get({2}, builder.getF16Type());
    Attribute a;
    a = DisposableU64ElementsAttr::get(type, {}, nullptr, nullptr);
    assert(a);
    assert(a.isa<DisposableU64ElementsAttr>());
    DisposableU64ElementsAttr i = a.cast<DisposableU64ElementsAttr>();
    ShapedType t = i.getType();
    llvm::errs() << "type:" << t << "\n";
    std::cerr << "shape:" << t.getShape() << "\n";
    assert(i.isa<ElementsAttr>());
    assert(i.isSplat());
    assert(failed(i.getValuesImpl(TypeID::get<uint64_t>())));
    assert(!i.cast<ElementsAttr>().try_value_begin<uint64_t>());
    ElementsAttr e = i; // i.cast<ElementsAttr>();
    t = e.getType();
    assert(e.isSplat());
    assert(t);
    llvm::errs() << "type:" << t << "\n";
    assert(failed(e.getValuesImpl(TypeID::get<uint64_t>())));
    assert(!e.try_value_begin<uint64_t>());
    return 0;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  Test test;
  int failures = 0;
  failures += test.test_PosIterator();
  failures += test.test_attributes();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
