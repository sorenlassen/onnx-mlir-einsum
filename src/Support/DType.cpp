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
  float f = static_cast<float>(n.dbl);
  if (ftag.isa<mlir::Float32Type>())
    return llvm::APFloat(f);
  if (ftag.isa<mlir::Float16Type>())
    return float_16::toAPFloat(float_16(f));
  if (ftag.isa<mlir::BFloat16Type>())
    return bfloat_16::toAPFloat(bfloat_16(f));
  llvm_unreachable("unsupported floating point width");
}

llvm::APInt toAPInt(mlir::IntegerType itag, IntOrFP n) {
  if (itag.isSigned())
    return llvm::APInt(itag.getWidth(), n.i64, /*isSigned=*/true);
  else
    return llvm::APInt(itag.getWidth(), n.u64);
}

DType fromIntOrFPMlirTypeToDType(mlir::Type type) {
  // clang-format off
  if (type.isa<mlir::Float64Type>())  return DType::DOUBLE;
  if (type.isa<mlir::Float32Type>())  return DType::FLOAT;
  if (type.isa<mlir::Float16Type>())  return DType::FLOAT16;
  if (type.isa<mlir::BFloat16Type>()) return DType::BFLOAT16;
  auto itype = type.cast<mlir::IntegerType>();
  switch (itype.getWidth()) {
    case  1: return DType::BOOL;
    case  8: return itype.isUnsigned() ? DType::UINT8  : DType::INT8;
    case 16: return itype.isUnsigned() ? DType::UINT16 : DType::INT16;
    case 32: return itype.isUnsigned() ? DType::UINT32 : DType::INT32;
    case 64: return itype.isUnsigned() ? DType::UINT64 : DType::INT64;
  }
  llvm_unreachable("unsupported int or float type");
  // clang-format on
}

unsigned widthOfIntOrFPType(mlir::Type t) {
  if (auto i = t.dyn_cast<mlir::IntegerType>())
    return i.getWidth();
  auto f = t.cast<mlir::FloatType>();
  return f.getWidth();
}

} // namespace onnx_mlir