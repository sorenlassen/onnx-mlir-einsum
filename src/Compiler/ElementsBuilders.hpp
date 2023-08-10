/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/Builder/ElementsBuilder.hpp"

#include <memory>

namespace onnx_mlir {

std::unique_ptr<ElementsBuilder> getDisposableElementsBuilder(mlir::MLIRContext *context);

std::unique_ptr<ElementsBuilder> getLazyElementsBuilder(mlir::MLIRContext *context);

} // namespace onnx_mlir
