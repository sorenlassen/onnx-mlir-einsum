/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/Support/Arrays.hpp"

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <functional>

namespace onnx_mlir {

class ElementsBuilder {
public:
  virtual ~ElementsBuilder() = default;

  template <typename X>
  using Writer = std::function<void(llvm::MutableArrayRef<X>)>;

  virtual mlir::ElementsAttr writeRawBytes(
      mlir::ShapedType type, const Writer<char> &writer) = 0;

  template <typename X, typename W = Writer<X>>
  mlir::ElementsAttr writeArray(mlir::ShapedType type, W &&writer) {
    return writeRawBytes(type, [arrayWriter = std::forward<W>(writer)](
                                   llvm::MutableArrayRef<char> rawBytes) {
      arrayWriter(castMutableArrayRef<X>(rawBytes));
    });
  }

  virtual mlir::ElementsAttr fromRawBytes(
      mlir::ShapedType type, llvm::ArrayRef<char> values) = 0;

  //   virtual mlir::ElementsAttr fromSplatValue(
  //       mlir::ShapedType type, APFloat/APInt/WideNum splatValue) = 0;

  virtual mlir::ElementsAttr fromFile(mlir::ShapedType type,
      mlir::StringAttr path, uint64_t offset, uint64_t length) = 0;

  mlir::ElementsAttr fromFile(mlir::ShapedType type, mlir::StringRef path,
      uint64_t offset, uint64_t length) {
    auto pathAttr = mlir::StringAttr::get(type.getContext(), path);
    return fromFile(type, pathAttr, offset, length);
  }
};

} // namespace onnx_mlir
