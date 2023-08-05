/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyElements/LazyElements.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include <iostream>

using namespace mlir;
using namespace lazy_elements;

namespace {

class Test {
  MLIRContext *ctx;
  Location loc;
  OpBuilder b;
  Type F32;
  Type I32;

  Attribute zero(Type t) {
    if (isa<FloatType>(t))
      return FloatAttr::get(t, 0);
    assert(isa<IntegerType>(t) && "must be IntegerType if not FloatType");
    return IntegerAttr::get(t, 0);
  }

public:
  Test(MLIRContext *ctx) : ctx(ctx), loc(UnknownLoc::get(ctx)), b(ctx) {
    F32 = b.getF32Type();
    I32 = b.getI32Type();
  }

  int test_file_data() {
    auto m = ModuleOp::create(loc);

    auto type = RankedTensorType::get({5}, F32);
    auto path = b.getStringAttr("yo.data");
    auto f0 = FileDataAttr::get(ctx, type, path, 0);
    auto f1 = FileDataAttr::get(ctx, type, path, 1);
    m->setAttr("f0", f0);
    m->setAttr("f1", f1);

    auto d = DenseElementsAttr::get<float>(type, 3.14f);
    auto neg = b.getStringAttr("neg");
    auto e0 = LazyElementsAttr::get(ctx, type, neg, {d}, nullptr);
    auto add = b.getStringAttr("add");
    auto e1 = LazyElementsAttr::get(ctx, type, add, {e0, e0}, nullptr);

    m->setAttr("e0", e0);
    m->setAttr("e1", e1);

    llvm::outs() << m << "\n";
    return 0;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  MLIRContext context;
  context.loadDialect<LazyElementsDialect>();
  Test test(&context);
  int failures = 0;
  failures += test.test_file_data();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
