/*
 * SPDX-License-Identifier: Apache-2.0
 */

namespace lazycst {
class LazyCstDialect;
}
namespace mlir {
class MLIRContext;
class ONNXDialect;
} // namespace mlir

namespace onnx_mlir {

void populateONNXLazyFolders(mlir::MLIRContext *ctx,
    lazycst::LazyCstDialect *lazycstDialect, mlir::ONNXDialect *onnxDalect);

}
