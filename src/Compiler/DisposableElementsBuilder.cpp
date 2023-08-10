/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Compiler/DisposableElementsBuilder.hpp"

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
} // namespace

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

} // namespace onnx_mlir
