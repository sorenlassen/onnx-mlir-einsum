/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Compiler/ElementsBuilders.hpp"

#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Path.h"

#include <string>

using namespace mlir;

namespace onnx_mlir {

namespace {
[[maybe_unused]] std::string dirName(StringRef inputFilename) {
  llvm::SmallVector<char> path(inputFilename.begin(), inputFilename.end());
  llvm::sys::path::remove_filename(path);
  return std::string(path.data(), path.size());
}

class DisposableElementsBuilder : public ElementsBuilder {
public:
  DisposableElementsBuilder(mlir::MLIRContext *context)
      : attrBuilder(context) {}

  mlir::ElementsAttr writeRawBytes(
      mlir::ShapedType type, const Writer<char> &writer) override {
    llvm_unreachable("TODO: implement this");
  }

  virtual mlir::ElementsAttr fromRawBytes(
      mlir::ShapedType type, llvm::ArrayRef<char> values) override {
    llvm_unreachable("TODO: implement this");
  }

  virtual mlir::ElementsAttr fromFile(mlir::ShapedType type,
      mlir::StringAttr path, uint64_t offset, uint64_t length) override {
    llvm_unreachable("TODO: implement this");
  }

private:
  OnnxElementsAttrBuilder attrBuilder;
};

} // namespace

std::unique_ptr<ElementsBuilder> getDisposableElementsBuilder(
    mlir::MLIRContext *context) {
  return std::make_unique<DisposableElementsBuilder>(context);
}

std::unique_ptr<ElementsBuilder> getLazyElementsBuilder(
    mlir::MLIRContext *context) {
  llvm_unreachable("TODO: implement this");
}

} // namespace onnx_mlir
