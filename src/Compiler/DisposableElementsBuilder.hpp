/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/Builder/ElementsBuilder.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"

namespace onnx_mlir {

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

} // namespace onnx_mlir
