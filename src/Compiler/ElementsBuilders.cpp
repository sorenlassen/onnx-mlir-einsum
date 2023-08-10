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
  DisposableElementsBuilder(mlir::MLIRContext *context);
  virtual ~DisposableElementsBuilder();

  mlir::ElementsAttr writeRawBytes(
      mlir::ShapedType type, const Writer &writer) override;

  virtual mlir::ElementsAttr fromRawBytes(
      mlir::ShapedType type, llvm::ArrayRef<char> values) override;

  virtual mlir::ElementsAttr fromFile(mlir::ShapedType type,
      mlir::StringAttr path, uint64_t offset, uint64_t length) override;

private:
  OnnxElementsAttrBuilder attrBuilder;
};

DisposableElementsBuilder::DisposableElementsBuilder(MLIRContext *context)
    : attrBuilder(context) {}

DisposableElementsBuilder::~DisposableElementsBuilder() = default;

ElementsAttr DisposableElementsBuilder::writeRawBytes(
    ShapedType type, const Writer &writer) {
  llvm_unreachable("TODO: implement this");
}

ElementsAttr DisposableElementsBuilder::fromRawBytes(
    ShapedType type, llvm::ArrayRef<char> values) {
  llvm_unreachable("TODO: implement this");
}

ElementsAttr DisposableElementsBuilder::fromFile(
    ShapedType type, StringAttr path, uint64_t offset, uint64_t length) {
  llvm_unreachable("TODO: implement this");
}

} // namespace

std::unique_ptr<ElementsBuilder> getDisposableElementsBuilder(mlir::MLIRContext *context) {
  return std::make_unique<DisposableElementsBuilder>(context);
}

std::unique_ptr<ElementsBuilder> getLazyElementsBuilder(mlir::MLIRContext *context) {
  llvm_unreachable("TODO: implement this");
}

} // namespace onnx_mlir
