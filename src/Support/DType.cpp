/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- DType.cpp ------------------------------===//
//
// Basic data types.
//
//===----------------------------------------------------------------------===//

#include "src/Support/DType.hpp"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

using llvm::APFloat;
using llvm::APInt;

namespace onnx_mlir {

/*static*/
llvm::APFloat float_16::toAPFloat(float_16 f16) {
  return APFloat(APFloat::IEEEhalf(), APInt(16, f16.u16));
}
/*static*/
float_16 float_16::fromAPFloat(llvm::APFloat a) {
  bool ignored;
  a.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &ignored);
  APInt i = a.bitcastToAPInt();
  uint16_t u16 = i.getZExtValue();
  return {u16};
}
/*static*/
float float_16::toFloat(float_16 f16) {
  return toAPFloat(f16).convertToFloat();
}
/*static*/
float_16 float_16::fromFloat(float f) {
  return fromAPFloat(APFloat(f));
}

bool isIntOrFPType(mlir::Type t, unsigned maxWidth) {
  if (auto i = t.dyn_cast<mlir::IntegerType>())
    return i.getWidth() <= maxWidth;
  if (auto f = t.dyn_cast<mlir::FloatType>())
    return f.getWidth() <= maxWidth;
  return false;
}

llvm::APFloat toAPFloat(mlir::FloatType ftag, IntOrFP n) {
  if (ftag.getWidth() == 16)
    return float_16::toAPFloat(float_16::fromFloat(fromIntOrFP<float>(ftag, n)));
  if (ftag.getWidth() == 32)
    return llvm::APFloat(fromIntOrFP<float>(ftag, n));
  if (ftag.getWidth() == 64)
    return llvm::APFloat(fromIntOrFP<double>(ftag, n));
  llvm_unreachable("unsupported floating point width");
}

llvm::APInt toAPInt(mlir::IntegerType itag, IntOrFP n) {
  if (itag.isSigned())
    return llvm::APInt(itag.getWidth(), n.i64, /*isSigned=*/true);
  else
    return llvm::APInt(itag.getWidth(), n.u64);
}

} // namespace onnx_mlir