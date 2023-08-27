/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ CompilerDialects.cpp ------------------------===//

#include "CompilerDialects.hpp"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/LazyCst/LazyCst.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"

#include "mlir/InitAllDialects.h"

using namespace mlir;

namespace onnx_mlir {

DialectRegistry registerDialects(ArrayRef<accel::Accelerator::Kind> accels) {
  DialectRegistry registry;

  // Note that we cannot consult command line options because they have not yet
  // been parsed when registerDialects() is called.

  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<affine::AffineDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<vector::VectorDialect>();
  registry.insert<shape::ShapeDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<lazycst::LazyCstDialect>();
  registry.insert<ONNXDialect>();
  registry.insert<KrnlDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<mlir::omp::OpenMPDialect>();

  // Initialize accelerator(s) if required.
  accel::initAccelerators(accels);

  // Register dialects for accelerators.
  for (auto *accel : accel::Accelerator::getAccelerators())
    accel->registerDialects(registry);

  if (useOldBufferization)
    memref::registerAllocationOpInterfaceExternalModels(registry);

  return registry;
}

void configureDialects(mlir::DialectRegistry &registry) {
  // We do this in onnx-mlir instead to exclude OpenMP from tests etc.
  // registerOpenMPDialectTranslation(registry);

  registry.addExtension(
      +[](MLIRContext *ctx, lazycst::LazyCstDialect *lazycstDialect) {
        lazycst::FileDataManager::Config config;
        config.readDirectoryPaths.assign(
            externalDataDir.begin(), externalDataDir.end());
        // TODO: configure writePathPrefix/Suffix
        lazycstDialect->fileDataManager.configure(config);
      });
}

} // namespace onnx_mlir
