/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- DType.cpp ------------------------------===//
//
// Basic data types.
//
//===----------------------------------------------------------------------===//

#include "src/Support/DType.hpp"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

using llvm::APFloat;
using llvm::APInt;

namespace onnx_mlir {

uint64_t detail::bitcastAPFloat(
    llvm::APFloat f, const llvm::fltSemantics &semantics) {
  bool ignored;
  f.convert(semantics, APFloat::rmNearestTiesToEven, &ignored);
  APInt i = f.bitcastToAPInt();
  return i.getZExtValue();
}

DType dtypeOfMlirType(mlir::Type type) {
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

mlir::Type mlirTypeOfDType(DType dtype, mlir::MLIRContext *ctx) {
  constexpr bool isUnsigned = false;
  mlir::Builder b(ctx);
  // clang-format off
  switch (dtype) {
    case DType::BOOL     : return b.getI1Type();
    case DType::INT8     : return b.getIntegerType(8);
    case DType::UINT8    : return b.getIntegerType(8, isUnsigned);
    case DType::INT16    : return b.getIntegerType(16);
    case DType::UINT16   : return b.getIntegerType(16, isUnsigned);
    case DType::INT32    : return b.getIntegerType(32);
    case DType::UINT32   : return b.getIntegerType(32, isUnsigned);
    case DType::INT64    : return b.getIntegerType(64);
    case DType::UINT64   : return b.getIntegerType(64, isUnsigned);
    case DType::DOUBLE   : return b.getF64Type();
    case DType::FLOAT    : return b.getF32Type();
    case DType::FLOAT16  : return b.getF16Type();
    case DType::BFLOAT16 : return b.getBF16Type();
    default: llvm_unreachable("unsupported data type");
  }
  // clang-format on
}

llvm::APFloat IntOrFP::toAPFloat(mlir::FloatType ftag) const {
  if (ftag.isa<mlir::Float64Type>())
    return llvm::APFloat(dbl);
  float f = static_cast<float>(dbl);
  if (ftag.isa<mlir::Float32Type>())
    return llvm::APFloat(f);
  if (ftag.isa<mlir::Float16Type>())
    return float_16(f).toAPFloat();
  if (ftag.isa<mlir::BFloat16Type>())
    return bfloat_16(f).toAPFloat();
  llvm_unreachable("unsupported floating point width");
}

llvm::APInt IntOrFP::toAPInt(mlir::IntegerType itag) const {
  if (itag.isSigned())
    // Actually, isSigned flag is ignored becuase width <= 64.
    return llvm::APInt(itag.getWidth(), i64, /*isSigned=*/true);
  else
    return llvm::APInt(itag.getWidth(), u64);
}

} // namespace onnx_mlir