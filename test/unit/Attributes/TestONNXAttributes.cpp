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
#include "llvm/Support/SwapByteOrder.h"
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
inline raw_ostream &operator<<(raw_ostream &os, FP16Type fp16) {
  return os << "FP16(" << fp16.bitcastToU16() << ")";
}
inline raw_ostream &operator<<(raw_ostream &os, APFloat af) {
  return os << "APFloat(" << af.convertToDouble() << ")";
}
inline raw_ostream &operator<<(raw_ostream &os, onnx_mlir::IntOrFP n) {
  return os << "IntOrFP(i=" << n.i64 << ",u=" << n.u64 << ",f=" << n.dbl << ")";
}
inline raw_ostream &operator<<(raw_ostream &os, onnx_mlir::DType dtype) {
  return os << "DType(" << static_cast<int>(dtype) << ")";
}

MLIRContext *createCtx() {
  MLIRContext *ctx = new MLIRContext();
  ctx->loadDialect<ONNXDialect>();
  return ctx;
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
  return std::shared_ptr<llvm::MemoryBuffer>(
      llvm::MemoryBuffer::getMemBufferCopy(asStringRef(data)));
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

  int test_getSwappedBytes() {
    llvm::errs() << "test_float_16:\n";
    llvm::errs() << "swap true " << llvm::sys::getSwappedBytes(true) << "\n";
    bool t = true;
    llvm::errs() << "swap t " << llvm::sys::getSwappedBytes(t) << "\n";
    return 0;
  }

  int test_dispatch_DTypeToken() {
    llvm::errs() << "test_dispatch_DTypeToken:\n";
    for (int i = 1; i <= 16; ++i) {
      if (i == 8 || i == 14 || i == 15)
        continue;
      dispatchByDType(static_cast<DType>(i), [](auto dtype) {
        using cpptype = CppType<dtype>;
        constexpr cpptype x{};
        constexpr DType d1 = dtype;
        constexpr DType d2 = dtypeOf(x);
        assert(d1 == d2);
        constexpr DType dtypeO = dtypeOf(x);
        llvm::errs() << "dtypeOf " << dtypeO << ", x=" << x << "\n";
      });
    }
    return 0;
  }

  int test_float_16() {
    llvm::errs() << "test_float_16:\n";
    float_16 f9984(9984);
    bfloat_16 fminus1(-1);
    float_16 bfminus1(fminus1);
    bfloat_16 bf9984(f9984);
    llvm::errs() << "float16 " << f9984.toFloat() << " as uint "
                 << f9984.bitcastToU16() << "\n";
    llvm::errs() << "float16 " << bfminus1.toFloat() << " as uint "
                 << bfminus1.bitcastToU16() << "\n";
    llvm::errs() << "bfloat16 " << bf9984.toFloat() << " as uint "
                 << bf9984.bitcastToU16() << "\n";
    llvm::errs() << "bfloat16 " << fminus1.toFloat() << " as uint "
                 << fminus1.bitcastToU16() << "\n";
    // assert(static_cast<bfloat_16>(bfminus1) == bf9984); // fails, == not
    // defined
    assert(bfminus1.toFloat() == fminus1.toFloat());
    assert(static_cast<float_16>(bf9984).toFloat() ==
           static_cast<bfloat_16>(f9984).toFloat());
    constexpr float_16 f16z = float_16();
    constexpr bfloat_16 bf16z = bfloat_16();
    constexpr float_16 f16z2 = f16z;
    constexpr bfloat_16 bf16z2 = bf16z;
    constexpr uint16_t f16zu = f16z2.bitcastToU16();
    constexpr uint16_t bf16zu = bf16z2.bitcastToU16();
    constexpr DType df16 = dtypeOf(f16z);
    constexpr DType dbf16 = dtypeOf(bf16z);
    assert(df16 == dtypeOf<float_16>());
    assert(dbf16 == dtypeOf<bfloat_16>());
    assert((std::is_same_v<CppType<df16>, float_16>));
    assert((std::is_same_v<CppType<dbf16>, bfloat_16>));
    assert((std::is_same_v<CppType<dtypeOf<float>()>, float>));
    llvm::errs() << "float16 " << f16z.toFloat() << " as uint " << f16zu
                 << ", dtype=" << df16 << "\n";
    llvm::errs() << "bfloat16 " << bf16z.toFloat() << " as uint " << bf16zu
                 << ", dtype=" << dbf16 << "\n";
    return 0;
  }

  int test_DType() {
    llvm::errs() << "test_DType:\n";
    uint64_t u;
    int8_t i = -128;
    u = i;
    llvm::errs() << "-128i8 as u64 " << u << "\n";
    llvm::errs() << "static_cast<u64>(-128i8) " << static_cast<uint64_t>(i)
                 << "\n";
    assert(CppTypeTrait<float>::isFloat);
    return 0;
  }

  int test_IntOrFP() {
    llvm::errs() << "test_IntOrFP:\n";
    constexpr IntOrFP nf = IntOrFP::from(DType::DOUBLE, 42.0);
    llvm::errs() << "nf " << nf << "\n";
    llvm::errs() << "nf as APFloat " << nf.to<APFloat>(builder.getF64Type())
                 << "\n";
    constexpr int64_t i = -42;
    constexpr IntOrFP ni = IntOrFP::from(DType::INT64, i);
    llvm::errs() << "ni " << ni << "\n";
    llvm::errs() << "ni as APInt " << ni.to<APInt>(builder.getI64Type())
                 << "\n";
    constexpr uint64_t u = 1ULL << 63;
    constexpr IntOrFP nu = IntOrFP::from(DType::UINT64, u);
    llvm::errs() << "nu " << nu << "\n";
    llvm::errs() << "nu as APInt " << nu.to<APInt>(getUInt(64)) << "\n";
    constexpr bool b = true;
    constexpr IntOrFP nb = IntOrFP::from(DType::UINT64, b);
    // constexpr bool b1 = nb.cast<bool>();
    // constexpr bool b2 = nb.to<DType::BOOL>();
    constexpr bool b3 = nb.to<bool>(DType::BOOL);
    bool b4 = nb.to<bool>(getUInt(1));
    // llvm::errs() << "b1 " << b1 << "\n";
    // llvm::errs() << "b2 " << b2 << "\n";
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
    ArrayRef<int64_t> strides{};
    bool isSplat = true;
    bool isContiguous = true;
    Attribute a = disposablePool.createElementsAttr(type, strides, DType::FLOAT,
        isSplat, isContiguous, buffer<float>({4.2}), fun);
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
    assert(fabs(float_16::fromFloat(4.2).toFloat() - 4.2) < 1e-3);
    ShapedType type = RankedTensorType::get({1}, builder.getF16Type());
    auto fun = [](StringRef s, size_t p) -> IntOrFP {
      return {.dbl = asArrayRef<float_16>(s)[p].toFloat()};
    };
    ArrayRef<int64_t> strides{};
    bool isSplat = true;
    bool isContiguous = true;
    Attribute a = disposablePool.createElementsAttr(type, strides,
        DType::FLOAT16, isSplat, isContiguous,
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
    ArrayRef<int64_t> strides{};
    bool isSplat = true;
    bool isContiguous = true;
    Attribute a = disposablePool.createElementsAttr(type, strides, DType::BOOL,
        isSplat, isContiguous, buffer<bool>({true}), fun);
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
    a = disposablePool.createElementsAttr(type, buffer<uint64_t>({7, 9}), fun);
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
  failures += test.test_getSwappedBytes();
  failures += test.test_dispatch_DTypeToken();
  failures += test.test_float_16();
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
