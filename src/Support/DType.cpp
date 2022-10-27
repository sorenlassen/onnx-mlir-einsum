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

llvm::APFloat U16ToAPFloat(uint16_t u) {
  return APFloat(APFloat::IEEEhalf(), APInt(16, u));
}

uint16_t APFloatToU16(llvm::APFloat a) {
  bool ignored;
  a.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &ignored);
  APInt i = a.bitcastToAPInt();
  return i.getZExtValue();
}

// TODO: Explore if it's feasible and worthwhile to use _cvtss_sh, _cvtsh_ss
//       https://clang.llvm.org/doxygen/f16cintrin_8h.html

float U16ToF32(uint16_t u) {
  return U16ToAPFloat(u).convertToFloat();
}

uint16_t F32ToU16(float f) {
  return APFloatToU16(APFloat(f));
}

} // namespace onnx_mlir