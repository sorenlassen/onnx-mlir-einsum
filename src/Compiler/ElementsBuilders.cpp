/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Compiler/ElementsBuilders.hpp"

#include "src/Dialect/LazyCst/LazyCstAttributes.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Support/Arrays.hpp"
#include "src/Support/TypeUtilities.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Path.h"

#include <string>

using namespace mlir;

namespace onnx_mlir {

namespace {

class DisposableElementsBuilder : public ElementsBuilder {
public:
  DisposableElementsBuilder(MLIRContext *context) : attrBuilder(context) {}

  ElementsAttr writeRawBytes(
      ShapedType type, const Writer<char> &writer) override {
    llvm_unreachable("TODO: implement this");
  }

  virtual ElementsAttr fromRawBytes(
      ShapedType type, ArrayRef<char> values) override {
    llvm_unreachable("TODO: implement this");
  }

  virtual ElementsAttr fromFile(ShapedType type, StringAttr path,
      uint64_t offset, uint64_t length) override {
    llvm_unreachable("TODO: implement this");
  }

private:
  OnnxElementsAttrBuilder attrBuilder;
};

class LazyElementsBuilder : public ElementsBuilder {
public:
  LazyElementsBuilder(MLIRContext *context) {}

  ElementsAttr writeRawBytes(
      ShapedType type, const Writer<char> &writer) override {
    // TODO: replace with a zero-copy alternative
    SmallVector<char> values;
    values.resize_for_overwrite(getSizeInBytes(type));
    writer(values);
    return fromRawBytes(type, values);
  }

  ElementsAttr fromRawBytes(ShapedType type, ArrayRef<char> values) override {
    // TODO: use something like DenseResourceElementsAttr that supports GC
    if (type.getElementType().isInteger(1))
      return DenseElementsAttr::get(type, castArrayRef<bool>(values));
    else
      return DenseElementsAttr::getFromRawBuffer(type, values);
  }

  ElementsAttr fromFile(ShapedType type, StringAttr path, uint64_t offset,
      uint64_t length) override {
    return lazycst::FileDataElementsAttr::get(type, path, offset);
  }
};

} // namespace

std::unique_ptr<ElementsBuilder> getDisposableElementsBuilder(
    MLIRContext *context) {
  return std::make_unique<DisposableElementsBuilder>(context);
}

std::unique_ptr<ElementsBuilder> getLazyElementsBuilder(MLIRContext *context) {
  return std::make_unique<LazyElementsBuilder>(context);
}

} // namespace onnx_mlir
