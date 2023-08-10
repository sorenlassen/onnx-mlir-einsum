/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- FrontendDialectHelper.hpp ----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for handling input ONNX models.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinAttributeInterfaces.h"

#include "onnx/onnx_pb.h"

#include <string>

namespace lazycst {
class ElementsBuilder;
}

namespace onnx_mlir {

mlir::ElementsAttr onnxTensorProtoToElmAttr(mlir::MLIRContext *ctx,
    lazycst::ElementsBuilder &elementsBuilder,
    const onnx::TensorProto &initializer);

} // namespace onnx_mlir
