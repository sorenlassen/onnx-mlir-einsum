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
  return llvm::makeArrayRef(reinterpret_cast<const Dst *>(a.data()),
      a.size() * sizeof(Src) / sizeof(Dst));
}

template <typename Dst = char>
ArrayRef<Dst> asArrayRef(StringRef s) {
  return llvm::makeArrayRef(
      reinterpret_cast<const Dst *>(s.data()), s.size() / sizeof(Dst));
}

template <typename T>
std::shared_ptr<llvm::MemoryBuffer> buffer(ArrayRef<T> data) {
  StringRef s(
      reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T));
  return std::shared_ptr<llvm::MemoryBuffer>(
      llvm::MemoryBuffer::getMemBufferCopy(s));
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

  Type getUInt(unsigned width) const {
    return IntegerType::get(ctx.get(), width, IntegerType::Unsigned);
  }

  int test_splat() {
    ShapedType type = RankedTensorType::get({1}, builder.getF32Type());
    auto fun = [](StringRef s, size_t p) -> Number64 {
      return {.f64 = asArrayRef<float>(s)[p]};
    };
    Attribute a = DisposableElementsAttr::get(
        type, {0}, DType::UINT64, buffer<float>({4.2}), fun);
    assert(a);
    assert(a.isa<ElementsAttr>());
    ElementsAttr e = a.cast<ElementsAttr>();
    assert(a.isa<DisposableElementsAttr>());
    DisposableElementsAttr i = a.cast<DisposableElementsAttr>();
    i.print(llvm::errs());
    llvm::errs() << "\n";
    e.print(llvm::outs());
    llvm::errs() << "\n";
    a.print(llvm::outs());
    llvm::errs() << "\n";
    assert(e.isSplat());
    llvm::errs() << "splat value " << i.getSplatValue<float>() << "\n";
    assert(fabs(i.getSplatValue<float>() - 4.2) < 1e-6);
    auto b = i.value_begin<float>();
    auto x = *b;
    llvm::errs() << "x " << x << "\n";
    auto f = i.getSplatValue<APFloat>();
    assert(fabs(f.convertToDouble() - 4.2) < 1e-6);
    auto d = i.toDenseElementsAttr();
    d = i.toDenseElementsAttr();
    d.print(llvm::outs());
    llvm::errs() << "\n";
    return 0;
  }

  int test_f16() {
    assert(fabs(F16ToF32(F32ToF16(4.2)) - 4.2) < 1e-3);
    ShapedType type = RankedTensorType::get({1}, builder.getF16Type());
    auto fun = [](StringRef s, size_t p) -> Number64 {
      return {.f64 = F16ToF32(asArrayRef<float_16>(s)[p])};
    };
    Attribute a = DisposableElementsAttr::get(
        type, {0}, DType::FLOAT16, buffer<float_16>({F32ToF16(4.2)}), fun);
    assert(a);
    assert(a.isa<ElementsAttr>());
    ElementsAttr e = a.cast<ElementsAttr>();
    assert(a.isa<DisposableElementsAttr>());
    DisposableElementsAttr i = a.cast<DisposableElementsAttr>();
    i.print(llvm::errs());
    llvm::errs() << "\n";
    e.print(llvm::outs());
    llvm::errs() << "\n";
    a.print(llvm::outs());
    llvm::errs() << "\n";
    assert(e.isSplat());
    llvm::errs() << "splat value " << i.getSplatValue<float>() << "\n";
    assert(fabs(i.getSplatValue<float>() - 4.2) < 1e-3);
    auto b = i.value_begin<float>();
    auto x = *b;
    llvm::errs() << "x " << x << "\n";
    auto d = i.toDenseElementsAttr();
    d = i.toDenseElementsAttr();
    d.print(llvm::outs());
    llvm::errs() << "\n";
    return 0;
  }

  int test_attributes() {
    ShapedType type = RankedTensorType::get({2}, getUInt(64));
    auto fun = [](StringRef s, size_t p) -> Number64 {
      return {.u64 = asArrayRef<uint64_t>(s)[p]};
    };
    Attribute a;
    a = DisposableElementsAttr::get(
        type, {1}, DType::UINT64, buffer<uint64_t>({7, 9}), fun);
    assert(a);
    assert(a.isa<DisposableElementsAttr>());
    DisposableElementsAttr i = a.cast<DisposableElementsAttr>();
    auto d = i.toDenseElementsAttr();
    d = a.cast<DisposableElementsAttr>().toDenseElementsAttr();
    (void)d;
    i.print(llvm::outs());
    d.print(llvm::outs());
    a.print(llvm::outs());
    ShapedType t = i.getType();
    llvm::errs() << "type:" << t << "\n";
    std::cerr << "shape:" << t.getShape() << "\n";
    assert(i.isa<ElementsAttr>());
    assert(!i.isSplat());
    assert(succeeded(i.getValuesImpl(TypeID::get<uint64_t>())));
    // assert(i.try_value_begin<uint64_t>());
    auto begin = i.value_begin<uint64_t>();
    auto end = i.value_end<uint64_t>();
    assert(begin != end);
    assert(begin == i.getValues<uint64_t>().begin());
    assert(end == i.getValues<uint64_t>().end());
    auto x = *begin;
    llvm::errs() << "x " << x << "\n";
    assert(*begin == 7);
    std::cerr << "next:" << *++begin << "\n";
    // assert(succeeded(i.tryGetValues<uint64_t>()));
    for (auto v : i.getValues<uint64_t>())
      std::cerr << "ivalue:" << v << "\n";
    assert(i.cast<ElementsAttr>().try_value_begin<uint64_t>());
    std::cerr << "empty:" << i.empty() << "\n";

    auto apbegin = i.value_begin<APInt>();
    auto api = *apbegin;
    assert(api.getZExtValue() == 7);

    ElementsAttr e = i; // i.cast<ElementsAttr>();
    t = e.getType();
    assert(!e.isSplat());
    assert(t);
    llvm::errs() << "type:" << t << "\n";
    assert(succeeded(e.getValuesImpl(TypeID::get<uint64_t>())));
    assert(e.try_value_begin<uint64_t>());
    std::cerr << "*e.try_value_begin():" << (**e.try_value_begin<uint64_t>())
              << "\n";
    auto it = *e.try_value_begin<uint64_t>();
    std::cerr << "++*e.try_value_begin():" << *++it << "\n";
    for (auto it = e.tryGetValues<uint64_t>()->begin(),
              en = e.tryGetValues<uint64_t>()->end();
         it != en; ++it)
      std::cerr << "evalue:" << *it << "\n";
    auto vs = e.tryGetValues<uint64_t>();
    for (auto v : *vs) // we crash here, why?
      std::cerr << "evalue:" << v << "\n";
    for (auto v : *e.tryGetValues<uint64_t>()) // we crash here, why?
      std::cerr << "evalue:" << v << "\n";

    return 0;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  Test test;
  int failures = 0;
  failures += test.test_splat();
  failures += test.test_f16();
  failures += test.test_attributes();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
