/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/FileDataManager.hpp"
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"
#include "src/Dialect/LazyCst/LazyCstDialect.hpp"
#include "src/Dialect/LazyCst/LazyCstOps.hpp"
#include "src/Support/Arrays.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <iostream>

using namespace mlir;
using namespace lazycst;

namespace {

template <typename T>
llvm::sys::fs::TempFile makeTempFile(
    const std::string &prefix, ArrayRef<T> data) {
  // auto ok = sys::fs::createUniqueFile()
  auto tmp = llvm::sys::fs::TempFile::create(prefix + "_%%%.data");
  if (auto err = tmp.takeError()) {
    llvm::errs() << toString(std::move(err)) << "\n";
    llvm_unreachable("failed to create file");
  }
  llvm::raw_fd_ostream os(tmp->FD, /*shouldClose=*/false);
  ArrayRef<char> bytes = onnx_mlir::castArrayRef<char>(data);
  os.write(bytes.data(), bytes.size());
  return std::move(*tmp);
}

class Test {
  MLIRContext *ctx [[maybe_unused]];
  Location loc;
  OpBuilder b;
  Type F32, I32, UI32;
  lazycst::LazyCstDialect *lazyDialect;

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
    lazyDialect = ctx->getLoadedDialect<lazycst::LazyCstDialect>();

    lazycst::FileDataManager::Config config;
    std::filesystem::path wd(".");
    config.readDirectoryPaths.push_back(wd);
    config.writeDirectoryPath = wd;
    config.writePathPrefix = "prefix";
    config.writePathSuffix = ".data";
    lazyDialect->fileDataManager.configure(config);
  }

  llvm::raw_ostream &print_ea(const std::string &name, ElementsAttr ea) {
    return llvm::outs() << name << "=" << ea << ":" << ea.getShapedType();
  }

  int test_extern_elms() {
    llvm::outs() << "test_extern_elms()\n";

    auto type = RankedTensorType::get({5}, F32);
    auto first = b.getStringAttr("first");
    auto externElms = ExternalElementsAttr::get(type, first);

    if (false) {
      // unreachable: invalid `T` for ElementsAttr::getValues
      cast<ElementsAttr>(externElms).getValues<Attribute>();
      cast<ElementsAttr>(externElms).value_begin<Attribute>();
    }

    // compile error: try_value_begin_impl() is unimplemented
    // externElms.getValues<Attribute>();
    // externElms.value_begin<Attribute>();

    assert(!externElms.empty());
    assert(externElms.getNumElements() == 5);
    // isSplat() just returns true when getNumElements() == 1, which is
    // dubious but shouldn't really matter in practice
    assert(!externElms.isSplat());

    llvm::outs() << "extern:" << externElms << "\n\n";
    return 0;
  }

  int test_read_file_data() {
    llvm::outs() << "test_read_file_data():\n";
    auto type = RankedTensorType::get({7}, F32);
    // TODO: pre-populate foo.data
    SmallVector<float> foo = {
        -INFINITY, -1.1e-11, -0.0, 0.0, 2.2e22, INFINITY, NAN};
    auto fooFile = makeTempFile("foo", ArrayRef(foo));
    auto path0 = b.getStringAttr(fooFile.TmpName);
    auto f0 = FileDataElementsAttr::get(type, path0);
    auto path1 = b.getStringAttr("bar.data");
    auto f1 = FileDataElementsAttr::get(type, path1, 1);
    print_ea("f0", f0) << "\n";
    print_ea("f1", f1) << "\n";

    ArrayRef<char> rawBytes = f0.getRawBytes();
    std::cout << "f0.getRawBytes() size=" << rawBytes.size() << " [0]='"
              << rawBytes[0] << "'\n";

    std::cout << "f0 values: ";
    auto values = f0.getValues<float>();
    assert(values.begin() == f0.value_begin<float>());
    assert(values.end() == f0.value_end<float>());
    for (float f : values) {
      std::cout << f << " ";
    }
    std::cout << "\n";
    ElementsAttr e0 = f0;
    std::cout << "ElementsAttr(f0) values: ";
    assert(e0.getValues<float>().begin() == e0.value_begin<float>());
    assert(e0.getValues<float>().end() == e0.value_end<float>());
    for (float f : e0.getValues<float>()) {
      std::cout << f << " ";
    }
    std::cout << "\n";

    auto m = ModuleOp::create(loc);
    m->setAttr("f0", f0);
    m->setAttr("f1", f1);

    llvm::outs() << m << "\n\n";
    auto fooErr = fooFile.discard();
    assert(!fooErr && "failed to discard temp file");
    return 0;
  }

  int test_write_file_data() {
    llvm::outs() << "test_write_file_data():\n";

    std::string filepath = lazyDialect->fileDataManager.writeFile(
        20, [](MutableArrayRef<char> dst) {
          auto f32s = onnx_mlir::castMutableArrayRef<float>(dst);
          for (int i = 0; i < 5; ++i)
            f32s[i] = i * 1.1;
        });
    auto type = RankedTensorType::get({5}, F32);
    auto f = FileDataElementsAttr::get(type, b.getStringAttr(filepath));
    print_ea("f", f) << "\n";
    std::cout << "f values: ";
    for (auto it = f.value_begin<float>(); it != f.value_end<float>(); ++it) {
      std::cout << *it << " ";
    }
    std::cout << "\n";

    llvm::outs() << "\n";
    return 0;
  }

  int test_lazy_elms() {
    llvm::outs() << "test_lazy_elms()\n";

    constexpr char sym_name[] = "cstexpr0";
    auto type = RankedTensorType::get({5}, F32);
    auto lazyElms =
        LazyElementsAttr::get(type, FlatSymbolRefAttr::get(ctx, sym_name));
    auto cstOp = b.create<arith::ConstantOp>(loc, lazyElms);

    auto uses = SymbolTable::getSymbolUses(cstOp);
    assert(uses.has_value());
    assert(std::distance(uses->begin(), uses->end()) == 1);
    auto sym = uses->begin()->getSymbolRef();
    assert(isa<FlatSymbolRefAttr>(sym));
    assert(cast<FlatSymbolRefAttr>(sym).getValue() == sym_name);

    llvm::outs() << lazyElms << "\n";
    llvm::outs() << cstOp << "\n\n";
    return 0;
  }

  int test_lazy_func() {
    llvm::outs() << "test_lazy_func()\n";

    auto f32tensortype = RankedTensorType::get({5}, F32);
    auto i32tensortype = RankedTensorType::get({5}, I32);

    auto m = ModuleOp::create(loc);
    SymbolTable symbolTable(m);

    b.setInsertionPointToEnd(m.getBody());
    FunctionType ftype = b.getFunctionType({}, {i32tensortype});
    func::FuncOp f = func::FuncOp::create(loc, "f", ftype, {});
    symbolTable.insert(f);
    b.setInsertionPointToStart(f.addEntryBlock());
    auto d = DenseElementsAttr::get<float>(f32tensortype, 3.14f);
    auto fcstOp = b.create<arith::ConstantOp>(loc, d);
    auto fcastOp = b.create<arith::FPToUIOp>(loc, i32tensortype, fcstOp);
    b.create<func::ReturnOp>(loc, fcastOp.getResult());

    b.setInsertionPointToStart(m.getBody());
    constexpr char sym_name[] = "cstexpr0";
    auto lazyElms = LazyElementsAttr::get(
        i32tensortype, FlatSymbolRefAttr::get(ctx, sym_name));
    FunctionType function_type =
        b.getFunctionType({f32tensortype}, {i32tensortype});
    auto arg_constants = b.getArrayAttr({fcstOp.getValue()});
    auto res_constants = b.getArrayAttr({lazyElms});
    auto arg_attrs = nullptr;
    auto res_attrs = nullptr;
    auto cstexpr0 = b.create<LazyFuncOp>(loc, sym_name, function_type,
        arg_constants, res_constants, arg_attrs, res_attrs);
    SymbolTable(m).insert(cstexpr0);
    b.setInsertionPointToStart(cstexpr0.addEntryBlock());
    auto castOp =
        b.create<arith::FPToUIOp>(loc, i32tensortype, cstexpr0.getArgument(0));
    auto returnOp = b.create<LazyReturnOp>(loc, ValueRange{castOp});
    assert(succeeded(verify(returnOp)));
    assert(succeeded(verify(cstexpr0)));

    b.setInsertionPointToEnd(m.getBody());
    func::FuncOp f2 = func::FuncOp::create(loc, "f2", ftype, {});
    symbolTable.insert(f2);
    b.setInsertionPointToStart(f2.addEntryBlock());
    auto f2cstOp = b.create<arith::ConstantOp>(loc, lazyElms);
    b.create<func::ReturnOp>(loc, f2cstOp.getResult());

    auto uses = SymbolTable::getSymbolUses(cstexpr0, &m.getBodyRegion());
    assert(uses.has_value());
    assert(std::distance(uses->begin(), uses->end()) == 2);
    std::vector<Operation *> expected{cstexpr0, f2cstOp};
    for (const auto &use : *uses)
      assert(llvm::find(expected, use.getUser()) != expected.end());

    llvm::outs() << m << "\n\n";

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
  failures += test.test_extern_elms();
  failures += test.test_read_file_data();
  failures += test.test_write_file_data();
  failures += test.test_lazy_elms();
  failures += test.test_lazy_func();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
