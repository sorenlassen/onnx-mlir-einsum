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

template <typename Dst = char>
ArrayRef<Dst> asArrayRef(const llvm::MemoryBuffer &b) {
  return asArrayRef<Dst>(b.getBuffer());
}

template <typename T>
std::shared_ptr<llvm::MemoryBuffer> buffer(ArrayRef<T> data) {
  StringRef s(
      reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T));
  return std::shared_ptr<llvm::MemoryBuffer>(
      llvm::MemoryBuffer::getMemBufferCopy(s));
}

class Test {
  MLIRContext *ctx;
  Location loc;
  OpBuilder builder;
  DisposablePool &disposablePool;
  Type F32;
  Type I32;

public:
  Test()
      : ctx(createCtx()), loc(UnknownLoc::get(ctx)), builder(ctx),
        disposablePool(DisposablePool::create(ctx)) {
    F32 = builder.getF32Type();
    I32 = builder.getI32Type();
  }
  ~Test() { delete ctx; }

  Type getUInt(unsigned width) const {
    return IntegerType::get(ctx, width, IntegerType::Unsigned);
  }

  int test_DType() {
    llvm::errs() << "test_DType:\n";
    uint64_t u;
    int8_t i = -128;
    u = i;
    llvm::errs() << "-128i8 as u64 " << u << "\n";
    llvm::errs() << "static_cast<u64>(-128i8) " << static_cast<uint64_t>(i) << "\n";
    assert(CppTypeTrait<float>::is_float);
    return 0;
  }

  int test_IntOrFP() {
    llvm::errs() << "test_IntOrFP:\n";
    constexpr IntOrFP nf = IntOrFP::from(DType::DOUBLE, 42.0);
    llvm::errs() << "nf " << nf.cast<double>() << "\n";
    constexpr int64_t i = 42;
    constexpr IntOrFP ni = IntOrFP::from(DType::INT64, i);
    llvm::errs() << "ni " << ni.cast<int64_t>() << "\n";
    constexpr uint64_t u = 42;
    constexpr IntOrFP nu = IntOrFP::from(DType::UINT64, u);
    llvm::errs() << "nu " << nu.cast<uint64_t>() << "\n";
    constexpr bool b = true;
    constexpr IntOrFP nb = IntOrFP::from(DType::UINT64, b);
    constexpr bool b1 = nb.cast<bool>();
    //constexpr bool b2 = nb.to<DType::BOOL>();
    constexpr bool b3 = nb.to<bool>(DType::BOOL);
    bool b4 = nb.to<bool>(getUInt(1));
    llvm::errs() << "b1 " << b1 << "\n";
    //llvm::errs() << "b2 " << b2 << "\n";
    llvm::errs() << "b3 " << b3 << "\n";
    llvm::errs() << "b4 " << b4 << "\n";
    return 0;
  }

  int test_DisposablePool() {
    llvm::errs() << "test_DisposablePool:\n";
    ShapedType type = RankedTensorType::get({1}, getUInt(1));
    auto dispo = disposablePool.createElementsAttr(type, buffer<bool>({true}));
    assert(dispo.isSplat());
    return 0;
  }

  int test_makeDense() {
    llvm::errs() << "test_makeDense:\n";
    ShapedType type = RankedTensorType::get({2}, builder.getF32Type());
    auto b = buffer<float>({42.0f, 42.0f});

    auto eCopy = makeDenseIntOrFPElementsAttrFromRawBuffer(
        type, asArrayRef(*b), /*mustCopy=*/true);
    llvm::errs() << "eCopy " << eCopy << "\n";
    assert(eCopy.isa<DisposableElementsAttr>());
    auto dCopy = eCopy.cast<DisposableElementsAttr>();
    assert(dCopy.getBuffer()->getBuffer().data() != b->getBuffer().data());

    auto e = makeDenseIntOrFPElementsAttrFromRawBuffer(
        type, asArrayRef(*b), /*mustCopy=*/false);
    assert(e.isa<DisposableElementsAttr>());
    auto d = e.cast<DisposableElementsAttr>();
    assert(d.getBuffer()->getBuffer().data() == b->getBuffer().data());

    return 0;
  }

  int test_splat() {
    llvm::errs() << "test_splat:\n";
    ShapedType type = RankedTensorType::get({1}, builder.getF32Type());
    auto fun = [](StringRef s, size_t p) -> IntOrFP {
      return {.dbl = asArrayRef<float>(s)[p]};
    };
    Attribute a = DisposableElementsAttr::get(
        type, {0}, DType::UINT64, buffer<float>({4.2}), fun);
    assert(a);
    assert(a.isa<ElementsAttr>());
    ElementsAttr e = a.cast<ElementsAttr>();
    assert(a.isa<DisposableElementsAttr>());
    DisposableElementsAttr i = a.cast<DisposableElementsAttr>();
    llvm::errs() << "as DisposableElementsAttr " << i << "\n";
    llvm::errs() << "as ElementsAttr " << e << "\n";
    llvm::errs() << "as Attribute " << a << "\n";
    assert(e.isSplat());
    llvm::errs() << "splat value " << i.getSplatValue<float>() << "\n";
    assert(fabs(i.getSplatValue<float>() - 4.2) < 1e-6);
    auto b = i.value_begin<float>();
    auto x = *b;
    llvm::errs() << "x " << x << "\n";
    // auto f = i.getSplatValue<APFloat>();
    // assert(fabs(f.convertToDouble() - 4.2) < 1e-6);
    auto d = i.toDenseElementsAttr();
    d = i.toDenseElementsAttr();
    llvm::errs() << "as DenseElementsAttr " << d << "\n";
    return 0;
  }

  int test_f16() {
    llvm::errs() << "test_f16:\n";
    assert(fabs(float_16::toFloat(float_16::fromFloat(4.2)) - 4.2) < 1e-3);
    ShapedType type = RankedTensorType::get({1}, builder.getF16Type());
    auto fun = [](StringRef s, size_t p) -> IntOrFP {
      return {.dbl = float_16::toFloat(asArrayRef<float_16>(s)[p])};
    };
    Attribute a = DisposableElementsAttr::get(type, {0}, DType::FLOAT16,
        buffer<float_16>({float_16::fromFloat(4.2)}), fun);
    assert(a);
    assert(a.isa<ElementsAttr>());
    ElementsAttr e = a.cast<ElementsAttr>();
    assert(a.isa<DisposableElementsAttr>());
    DisposableElementsAttr i = a.cast<DisposableElementsAttr>();
    assert(e.isSplat());
    llvm::errs() << "splat value " << i.getSplatValue<float>() << "\n";
    assert(fabs(i.getSplatValue<float>() - 4.2) < 1e-3);
    auto b = i.value_begin<float>();
    auto x = *b;
    llvm::errs() << "x " << x << "\n";
    auto d = i.toDenseElementsAttr();
    d = i.toDenseElementsAttr();
    llvm::errs() << "as DenseElementsAttr " << d << "\n";
    return 0;
  }

  int test_bool() {
    llvm::errs() << "test_bool:\n";
    ShapedType type = RankedTensorType::get({1}, getUInt(1));
    auto fun = [](StringRef s, size_t p) -> IntOrFP {
      return {.u64 = asArrayRef<bool>(s)[p]};
    };
    Attribute a = DisposableElementsAttr::get(
        type, {0}, DType::BOOL, buffer<bool>({true}), fun);
    assert(a);
    assert(a.isa<ElementsAttr>());
    ElementsAttr e = a.cast<ElementsAttr>();
    assert(a.isa<DisposableElementsAttr>());
    DisposableElementsAttr i = a.cast<DisposableElementsAttr>();
    assert(e.isSplat());
    llvm::errs() << "splat value " << i.getSplatValue<bool>() << "\n";
    assert(i.getSplatValue<bool>());
    auto b = i.value_begin<bool>();
    auto x = *b;
    llvm::errs() << "x " << x << "\n";
    auto d = i.toDenseElementsAttr();
    d = i.toDenseElementsAttr();
    llvm::errs() << "as DenseElementsAttr " << d << "\n";
    return 0;
  }

  int test_attributes() {
    llvm::errs() << "test_attributes:\n";
    ShapedType type = RankedTensorType::get({2}, getUInt(64));
    auto fun = [](StringRef s, size_t p) -> IntOrFP {
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
    llvm::errs() << "as DisposableElementsAttr " << i << "\n";
    llvm::errs() << "as DenseElementsAttr " << d << "\n";
    llvm::errs() << "as Attribute " << a << "\n";

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

    // auto apbegin = i.value_begin<APInt>();
    // auto api = *apbegin;
    // assert(api.getZExtValue() == 7);

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
  failures += test.test_DType();
  failures += test.test_IntOrFP();
  failures += test.test_DisposablePool();
  failures += test.test_makeDense();
  failures += test.test_splat();
  failures += test.test_f16();
  failures += test.test_bool();
  failures += test.test_attributes();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
