/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- FloatingPoint16.cpp --------------------------===//
//
// 16 bit floating point types.
//
//===----------------------------------------------------------------------===//

#include "src/Support/FloatingPoint16.hpp"

#include "llvm/ADT/APFloat.h"

using llvm::APFloat;
using llvm::APInt;

namespace onnx_mlir {

uint64_t detail::bitcastAPFloat(
    APFloat f, const llvm::fltSemantics &semantics) {
  bool ignored;
  f.convert(semantics, APFloat::rmNearestTiesToEven, &ignored);
  APInt i = f.bitcastToAPInt();
  return i.getZExtValue();
}

} // namespace onnx_mlir