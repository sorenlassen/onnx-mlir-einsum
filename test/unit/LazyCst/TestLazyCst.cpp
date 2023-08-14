/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCst.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include <iostream>

using namespace mlir;
using namespace lazycst;

namespace {

class Test {
  MLIRContext *ctx [[maybe_unused]];
  Location loc;
  OpBuilder b;
  Type F32, I32, UI32;

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
    UI32 = b.getIntegerType(32, /*isSigned=*/false);

    auto &lazyDialect = *ctx->getLoadedDialect<lazycst::LazyCstDialect>();
    lazycst::FileDataManager::Config config;
    config.readDirectoryPaths.push_back(".");
    lazyDialect.fileDataManager.configure(config);
  }

  llvm::raw_ostream &print_ea(const std::string &name, ElementsAttr ea) {
    return llvm::outs() << name << "=" << ea << ":" << ea.getShapedType();
  }

  int test_file_data() {
    auto type = RankedTensorType::get({5}, F32);
    // TODO: pre-populate foo.data
    auto path0 = b.getStringAttr("foo.data");
    auto f0 = FileDataElementsAttr::get(type, path0);
    auto path1 = b.getStringAttr("bar.data");
    auto f1 = FileDataElementsAttr::get(type, path1, 1);
    print_ea("f0", f0) << "\n";
    print_ea("f1", f1) << "\n";

    ArrayRef<char> rawBytes = f0.getRawBytes();
    std::cout << "f0.getRawBytes() size=" << rawBytes.size() << " [0]='"
              << rawBytes[0] << "'\n";

    auto m = ModuleOp::create(loc);
    m->setAttr("f0", f0);
    m->setAttr("f1", f1);

    llvm::outs() << m << "\n";
    return 0;
  }

  int test_lazy_elements() {
#if 0
    auto type = RankedTensorType::get({5}, F32);
    auto d = DenseElementsAttr::get<float>(type, 3.14f);
    auto neg = b.getStringAttr("neg");
    auto e0 = LazyElementsAttr::get(type, neg, {d});
    auto add = b.getStringAttr("add");
    auto e1 = LazyElementsAttr::get(type, add, {e0, e0});
    print_ea("e0", e0) << "n";
    print_ea("e1", e1) << "n";

    auto m = ModuleOp::create(loc);
    m->setAttr("e0", e0);
    m->setAttr("e1", e1);

    llvm::outs() << m << "\n";
#endif
    return 0;
  }

  int test_lazy_func() {
    auto f32type = RankedTensorType::get({5}, F32);
    auto ui32type = RankedTensorType::get({5}, UI32);

    auto m = ModuleOp::create(loc);
    SymbolTable symbolTable(m);

    b.setInsertionPointToEnd(m.getBody());
    FunctionType ftype = b.getFunctionType({}, {ui32type});
    func::FuncOp f = func::FuncOp::create(loc, "f", ftype, {});
    symbolTable.insert(f);
    auto fblock = f.addEntryBlock();
    b.setInsertionPointToStart(fblock);
    auto d = DenseElementsAttr::get<float>(f32type, 3.14f);
    auto fcstOp = b.create<arith::ConstantOp>(loc, d);
    auto fcastOp = b.create<arith::FPToUIOp>(loc, ui32type, fcstOp);
    b.create<func::ReturnOp>(loc, fcastOp.getResult());

    b.setInsertionPointToStart(m.getBody());
    constexpr char sym_name[] = "cstexpr0";
    auto lazyElms =
        LazyElementsAttr::get(ui32type, FlatSymbolRefAttr::get(ctx, sym_name));
    FunctionType function_type = b.getFunctionType({f32type}, {ui32type});
    auto arg_op_names =
        b.getArrayAttr({OperationNameAttr::get(fcstOp->getName())});
    auto res_op_names =
        b.getArrayAttr({OperationNameAttr::get(*RegisteredOperationName::lookup(
            arith::FPToUIOp::getOperationName(), ctx))});
    auto arg_attrs = b.getArrayAttr({fcstOp->getAttrDictionary()});
    auto res_attrs = b.getArrayAttr(
        {b.getDictionaryAttr({b.getNamedAttr("value", lazyElms)})});
    auto cexpr0 = b.create<LazyFuncOp>(loc, sym_name, function_type,
        arg_op_names, res_op_names, arg_attrs, res_attrs);
    SymbolTable(m).insert(cexpr0);
    auto block = cexpr0.addEntryBlock();
    b.setInsertionPointToStart(block);
    auto castOp =
        b.create<arith::FPToUIOp>(loc, ui32type, block->getArgument(0));
    b.create<LazyReturnOp>(loc, ValueRange{castOp});

    b.setInsertionPointToEnd(m.getBody());
    func::FuncOp f2 = func::FuncOp::create(loc, "f2", ftype, {});
    symbolTable.insert(f2);
    auto f2block = f2.addEntryBlock();
    b.setInsertionPointToStart(f2block);
    auto f2cstOp = b.create<arith::ConstantOp>(loc, lazyElms);
    b.create<func::ReturnOp>(loc, f2cstOp.getResult());

    llvm::outs() << m << "\n";
    return 0;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<LazyCstDialect>();
  Test test(&context);
  int failures = 0;
  failures += test.test_file_data();
  failures += test.test_lazy_elements();
  failures += test.test_lazy_func();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
