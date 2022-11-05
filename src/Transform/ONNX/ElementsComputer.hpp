/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ ElementsComputer.hpp ------------------------===//
//
// ElementsAttr computations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Support/DType.hpp"

#include "mlir/IR/BuiltinAttributes.h"

namespace onnx_mlir {

mlir::ElementsAttr transposeElements(
    mlir::ElementsAttr elements, llvm::ArrayRef<uint64_t> perm);

using Transformation = std::function<IntOrFP(IntOrFP)>;

mlir::ElementsAttr transformElements(mlir::ElementsAttr elements,
    mlir::Type transformedElementType, const Transformation &transformation);

} // namespace onnx_mlir