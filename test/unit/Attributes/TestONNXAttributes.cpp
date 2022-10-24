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

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"

#include <algorithm>
#include <iostream>
#include <iterator>
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

class Test {
  std::unique_ptr<MLIRContext> ctx;
  Location loc;
  OpBuilder builder;
  Type F32;
  Type I32;

  Attribute zero(Type t) {
    if (t.isa<FloatType>())
      return FloatAttr::get(t, 0);
    assert(t.isa<IntegerType>() && "must be IntegerType if not FloatType");
    return IntegerAttr::get(t, 0);
  }

  Value zeros(ArrayRef<int64_t> shape, Type t) {
    RankedTensorType tensorType = RankedTensorType::get(shape, t);
    SmallVector<Attribute> values(tensorType.getNumElements(), zero(t));
    return createONNXConstantOpWithDenseAttr(
        builder, loc, DenseElementsAttr::get(tensorType, makeArrayRef(values)));
  }

public:
  Test() : ctx(createCtx()), loc(UnknownLoc::get(&*ctx)), builder(&*ctx) {
    F32 = builder.getF32Type();
    I32 = builder.getI32Type();
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
    ElementsAttr e = i; //i.cast<ElementsAttr>();
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
  failures += test.test_attributes();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
