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

DType dtypeOf(mlir::Type type) {
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

mlir::Type mlirTypeOf(DType dtype, mlir::MLIRContext *ctx) {
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

bool isFloatDType(DType d) {
  return dispatchByDType(
      d, [](auto dtype) { return DTypeTrait<dtype>::isFloat; });
}

bool isIntOrFloatDType(DType d) {
  return dispatchByDType(
      d, [](auto dtype) { return DTypeTrait<dtype>::isIntOrFloat; });
}

bool isSignedIntDType(DType d) {
  return dispatchByDType(
      d, [](auto dtype) { return DTypeTrait<dtype>::isSignedInt; });
}

bool isUnsignedIntDType(DType d) {
  return dispatchByDType(
      d, [](auto dtype) { return DTypeTrait<dtype>::isUnsignedInt; });
}

unsigned bitwidthOfDType(DType d) {
  return dispatchByDType(
      d, [](auto dtype) { return DTypeTrait<dtype>::bitwidth; });
}

unsigned bytewidthOfDType(DType d) {
  return dispatchByDType(
      d, [](auto dtype) { return DTypeTrait<dtype>::bytewidth; });
}

llvm::APFloat IntOrFP::toAPFloat(DType tag) const {
  switch (tag) {
  case DType::DOUBLE:
    return llvm::APFloat(dbl);
  case DType::FLOAT:
    return llvm::APFloat(static_cast<float>(dbl));
  case DType::FLOAT16:
    return float_16(dbl).toAPFloat();
  case DType::BFLOAT16:
    return bfloat_16(dbl).toAPFloat();
  default:
    llvm_unreachable("DType must be a float");
  }
}

llvm::APInt IntOrFP::toAPInt(DType tag) const {
  unsigned bitwidth = bitwidthOfDType(tag);
  if (isSignedIntDType(tag))
    // Actually, isSigned flag is ignored because bitwidth <= 64.
    return llvm::APInt(bitwidth, i64, /*isSigned=*/true);
  if (isUnsignedIntDType(tag))
    return llvm::APInt(bitwidth, u64);
  llvm_unreachable("DType must be an integer");
}

/*static*/
IntOrFP IntOrFP::fromAPFloat(DType tag, llvm::APFloat x) {
  assert(isFloatDType(tag) && "DType must be an integer");
  return {.dbl = x.convertToDouble()};
}

/*static*/
IntOrFP IntOrFP::fromAPInt(DType tag, llvm::APInt x) {
  if (isSignedIntDType(tag))
    return {.i64 = x.getSExtValue()};
  if (isUnsignedIntDType(tag))
    return {.u64 = x.getZExtValue()};
  llvm_unreachable("DType must be an integer");
}

} // namespace onnx_mlir