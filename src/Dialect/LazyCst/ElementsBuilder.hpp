/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/Dialect/LazyCst/WideNum.hpp"

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/ArrayRef.h"

#include <functional>

namespace mlir {
class MLIRContext;
}

namespace lazycst {

// TODO: resolve name clash with ONNX/ElementsAttr/ElementsAttrBuilder and maybe
//       rename this class to ElementsAttrBuilder or
//       ElementsAttrBuilderInterface
class ElementsBuilder {
public:
  using Writer = std::function<void(llvm::MutableArrayRef<char>)>;

  virtual mlir::ElementsAttr writeRawBytes(
      mlir::ShapedType type, const Writer &writer) = 0;

  virtual mlir::ElementsAttr fromRawBytes(
      mlir::ShapedType type, llvm::ArrayRef<char> values) = 0;

  virtual mlir::ElementsAttr fromSplatValue(
      mlir::ShapedType type, WideNum splatValue) = 0;

  virtual mlir::ElementsAttr fromFile(mlir::ShapedType type,
      mlir::StringAttr path, uint64_t offset, uint64_t length) = 0;

  mlir::ElementsAttr fromFile(mlir::ShapedType type, mlir::StringRef path,
      uint64_t offset, uint64_t length) {
    auto pathAttr = mlir::StringAttr::get(type.getContext(), path);
    return fromFile(type, pathAttr, offset, length);
  }
};

} // namespace lazycst
