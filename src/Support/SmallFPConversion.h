/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ SmallFPConversion.h -------------------------===//
//
// Conversion to and from 16 bits floating point types.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_SMALLFPCONVERSION_H
#define ONNX_MLIR_SMALLFPCONVERSION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

float om_f16_to_f32(uint16_t u16);

uint16_t om_f32_to_f16(float f32);

float om_bf16_to_f32(uint16_t u16);

uint16_t om_f32_to_bf16(float f32);

uint16_t om_f8e5m2_to_f16(uint8_t u8);

uint8_t om_f16_to_f8e5m2(uint16_t u16);

uint8_t om_f16_to_f8e5m2_saturate(uint16_t u16);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_SMALLFPCONVERSION_H
