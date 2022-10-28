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

uint64_t detail::bitcastAPFloat(llvm::APFloat f, const llvm::fltSemantics &semantics) {
  bool ignored;
  f.convert(semantics, APFloat::rmNearestTiesToEven, &ignored);
  APInt i = f.bitcastToAPInt();
  return i.getZExtValue();
}

bool isIntOrFPType(mlir::Type t, unsigned maxWidth) {
  if (auto i = t.dyn_cast<mlir::IntegerType>())
    return i.getWidth() <= maxWidth;
  if (auto f = t.dyn_cast<mlir::FloatType>())
    return f.getWidth() <= maxWidth;
  return false;
}

llvm::APFloat toAPFloat(mlir::FloatType ftag, IntOrFP n) {
  if (ftag.isa<mlir::Float64Type>())
    return llvm::APFloat(n.dbl);
  if (ftag.isa<mlir::Float32Type>())
    return llvm::APFloat(static_cast<float>(n.dbl));
  if (ftag.isa<mlir::Float16Type>())
    return float_16::toAPFloat(float_16::fromFloat(n.dbl));
  if (ftag.isa<mlir::BFloat16Type>())
    return bfloat_16::toAPFloat(bfloat_16::fromFloat(n.dbl));
  llvm_unreachable("unsupported floating point width");
}

llvm::APInt toAPInt(mlir::IntegerType itag, IntOrFP n) {
  if (itag.isSigned())
    return llvm::APInt(itag.getWidth(), n.i64, /*isSigned=*/true);
  else
    return llvm::APInt(itag.getWidth(), n.u64);
}

} // namespace onnx_mlir