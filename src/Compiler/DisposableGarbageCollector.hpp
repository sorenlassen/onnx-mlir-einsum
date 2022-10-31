/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ DisposableGarbageCollector.hpp --------------------===//
//
// Garbage collects DisposableElementsAttr attributes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/PassInstrumentation.h"

namespace mlir {
class DisposablePool;
}

namespace onnx_mlir {

struct DisposableGarbageCollector : public mlir::PassInstrumentation {
  DisposableGarbageCollector(mlir::DisposablePool &disposablePool)
      : disposablePool(disposablePool) {}
  ~DisposableGarbageCollector() override = default;

  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

private:
  mlir::DisposablePool &disposablePool;
};

} // namespace onnx_mlir
