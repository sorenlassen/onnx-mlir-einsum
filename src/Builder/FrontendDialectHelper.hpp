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

namespace onnx_mlir {

class ElementsBuilder;

mlir::ElementsAttr onnxTensorProtoToElmAttr(mlir::MLIRContext *ctx,
    ElementsBuilder &elementsBuilder, const onnx::TensorProto &initializer);

} // namespace onnx_mlir
