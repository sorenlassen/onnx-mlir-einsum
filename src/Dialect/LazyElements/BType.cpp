/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- BType.cpp ------------------------------===//
//
// Basic data types.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/LazyElements/BType.hpp"

#include "mlir/IR/Builders.h"

using namespace mlir;

namespace lazy_elements {

BType btypeOfMlirType(Type type) {
  // clang-format off
  if (type.isa<Float64Type>())        return BType::DOUBLE;
  if (type.isa<Float32Type>())        return BType::FLOAT;
  if (type.isa<Float16Type>())        return BType::FLOAT16;
  if (type.isa<BFloat16Type>())       return BType::BFLOAT16;
  if (type.isa<Float8E4M3FNType>())   return BType::FLOAT8E4M3FN;
  if (type.isa<Float8E4M3FNUZType>()) return BType::FLOAT8E4M3FNUZ;
  if (type.isa<Float8E5M2Type>())     return BType::FLOAT8E5M2;
  if (type.isa<Float8E5M2FNUZType>()) return BType::FLOAT8E5M2FNUZ;
  auto itype = type.cast<IntegerType>();
  switch (itype.getWidth()) {
    case  1: return BType::BOOL;
    case  8: return itype.isUnsigned() ? BType::UINT8  : BType::INT8;
    case 16: return itype.isUnsigned() ? BType::UINT16 : BType::INT16;
    case 32: return itype.isUnsigned() ? BType::UINT32 : BType::INT32;
    case 64: return itype.isUnsigned() ? BType::UINT64 : BType::INT64;
  }
  llvm_unreachable("unsupported int or float type");
  // clang-format on
}

Type mlirTypeOfBType(BType btype, MLIRContext *ctx) {
  Builder b(ctx);
  // clang-format off
  switch (btype) {
    case BType::BOOL           : return b.getI1Type();
    case BType::INT8           : return b.getIntegerType(8);
    case BType::UINT8          : return b.getIntegerType(8, false);
    case BType::INT16          : return b.getIntegerType(16);
    case BType::UINT16         : return b.getIntegerType(16, false);
    case BType::INT32          : return b.getIntegerType(32);
    case BType::UINT32         : return b.getIntegerType(32, false);
    case BType::INT64          : return b.getIntegerType(64);
    case BType::UINT64         : return b.getIntegerType(64, false);
    case BType::DOUBLE         : return b.getF64Type();
    case BType::FLOAT          : return b.getF32Type();
    case BType::FLOAT16        : return b.getF16Type();
    case BType::BFLOAT16       : return b.getBF16Type();
    case BType::FLOAT8E4M3FN   : return b.getFloat8E4M3FNType();
    case BType::FLOAT8E4M3FNUZ : return b.getFloat8E4M3FNUZType();
    case BType::FLOAT8E5M2     : return b.getFloat8E5M2Type();
    case BType::FLOAT8E5M2FNUZ : return b.getFloat8E5M2FNUZType();
    default: llvm_unreachable("unsupported data type");
  }
  // clang-format on
}

bool isFloatBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::isFloat; });
}

bool isIntBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::isInt; });
}

bool isIntOrFloatBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::isIntOrFloat; });
}

bool isSignedIntBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::isSignedInt; });
}

bool isUnsignedIntBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::isUnsignedInt; });
}

unsigned bitwidthOfBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::bitwidth; });
}

unsigned bytewidthOfBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::bytewidth; });
}

BType wideBTypeOfBType(BType d) {
  return dispatchByBType(d,
      [](auto btype) { return toBType<typename BTypeTrait<btype>::widetype>; });
}

} // namespace lazy_elements