/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ SmallFPConversion.c -------------------------===//
//
// Conversion to and from 16 bits floating point types.
//
//===----------------------------------------------------------------------===//

#include "SmallFPConversion.h"

#include <assert.h>
#include <math.h>
#include <string.h>

#ifdef __MVS__
#define static_assert _Static_assert
#endif

// Defines variable TO of type TO_TYPE and copies bytes from variable FROM.
// Using memcpy because the simpler definition
//
//   #define BIT_CAST(TO_TYPE, TO, FROM) TO_TYPE TO = *(const TO_TYPE *)&FROM
//
// might violate the rules about strict aliasing in C++.
#define BIT_CAST(TO_TYPE, TO, FROM)                                            \
  TO_TYPE TO;                                                                  \
  static_assert(sizeof(TO) == sizeof(FROM), "only bit cast same sizes");       \
  memcpy(&TO, &FROM, sizeof(FROM))

#if defined(__x86_64__) && defined(__F16C__)
// On x86-64 build config -DCMAKE_CXX_FLAGS=-march=native defines __F16C__.

// https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-9/details-about-intrinsics-for-half-floats.html
#include <immintrin.h>

float om_f16_to_f32(uint16_t u16) { return _cvtsh_ss(u16); }

uint16_t om_f32_to_f16(float f32) {
  return _cvtss_sh(f32, /*ROUND TO NEAREST EVEN*/ 0);
}

#elif defined(__ARM_FP16_FORMAT_IEEE)
// On MacBook Pro no build config is needed to define __ARM_FP16_FORMAT_IEEE.

// https://arm-software.github.io/acle/main/acle.html#half-precision-floating-point

float om_f16_to_f32(uint16_t u16) {
  BIT_CAST(__fp16, f16, u16);
  return (float)f16;
}

uint16_t om_f32_to_f16(float f32) {
  __fp16 f16 = (__fp16)f32;
  BIT_CAST(uint16_t, u16, f16);
  return u16;
}

#else

#include <stdbool.h>

// Implementation adapted from https://stackoverflow.com/a/3542975

#define f32_sig_bits 23
#define f32_exp_bits 8
#define f32_bits (f32_sig_bits + f32_exp_bits + 1)
#define f32_exp_max ((1u << f32_exp_bits) - 1)
#define f32_exp_bias (f32_exp_max >> 1)
#define f32_sign (1u << (f32_bits - 1))
#define f32_inf (f32_exp_max << f32_sig_bits)

#define f16_sig_bits 10
#define f16_exp_bits 5
#define f16_bits (f16_sig_bits + f16_exp_bits + 1)
#define f16_exp_max ((1u << f16_exp_bits) - 1)
#define f16_exp_bias (f16_exp_max >> 1)
#define f16_sign (1u << (f16_bits - 1))
#define f16_inf (f16_exp_max << f16_sig_bits)
#define f16_qnan (f16_inf | (f16_inf >> 1))

#define sig_diff (f32_sig_bits - f16_sig_bits)
#define bit_diff (f32_bits - f16_bits)

float om_f16_to_f32(uint16_t u16) {
  static const uint32_t bias_mul = (2 * f32_exp_bias - f16_exp_bias)
                                   << f32_sig_bits;
  uint32_t bits = u16;
  uint32_t sign = bits & f16_sign; // save sign
  bits ^= sign;                    // clear sign
  bool is_norm = bits < f16_inf;
  bits = (sign << bit_diff) | (bits << sig_diff);
  BIT_CAST(float, bits_f32, bits);
  BIT_CAST(float, bias_mul_f32, bias_mul);
  float val_f32 = bits_f32 * bias_mul_f32;
  BIT_CAST(uint32_t, val, val_f32);
  val |= is_norm ? 0 : f32_inf;
  BIT_CAST(float, f32, val);
  return f32;
}

uint16_t om_f32_to_f16(float f32) {
  static const uint32_t bias_mul = f16_exp_bias << f32_sig_bits;
  BIT_CAST(float, bias_mul_f32, bias_mul);
  BIT_CAST(uint32_t, bits, f32);
  uint32_t sign = bits & f32_sign; // save sign
  bits ^= sign;                    // clear sign
  bool is_nan = f32_inf < bits;    // compare before rounding!!

  // round:
  {
    static const uint32_t min_norm = (f32_exp_bias - f16_exp_bias + 1)
                                     << f32_sig_bits;
    static const uint32_t sub_rnd = (f16_exp_bias - sig_diff) << f32_sig_bits;
    BIT_CAST(float, sub_rnd_f32, sub_rnd);
    static const uint32_t sub_mul = (f32_exp_bias + sig_diff) << f32_sig_bits;
    BIT_CAST(float, sub_mul_f32, sub_mul);
    bool is_sub = bits < min_norm;
    BIT_CAST(float, norm_f32, bits);
    float subn_f32 = norm_f32;
    subn_f32 *= sub_rnd_f32;  // round subnormals
    subn_f32 *= sub_mul_f32;  // correct subnormal exp
    norm_f32 *= bias_mul_f32; // fix exp bias
    BIT_CAST(uint32_t, norm, norm_f32);
    bits = norm;
    bits += (bits >> sig_diff) & 1;     // add tie breaking bias
    bits += (1u << (sig_diff - 1)) - 1; // round up to half
    BIT_CAST(uint32_t, subn, subn_f32);
    if (is_sub)
      bits = subn;
  }

  bits >>= sig_diff; // truncate
  if (f16_inf < bits)
    bits = f16_inf; // fix overflow
  if (is_nan)
    bits = f16_qnan;
  bits |= sign >> bit_diff; // restore sign
  return bits;
}

#endif

// Implementation adapted from the answers to
// https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c

float om_bf16_to_f32(uint16_t u16) {
  uint32_t u32 = ((uint32_t)u16) << 16;
  BIT_CAST(float, f32, u32);
  return f32;
}

uint16_t om_f32_to_bf16(float f32) {
  BIT_CAST(uint32_t, u32, f32);
  u32 += 32767 + ((u32 & 0x1FFFF) == 0x18000) * 0x10000;
  uint16_t u16 = u32 >> 16;
  if ((u16 & 0x7FFF) == 0x7F80 && isnan(f32))
    return u16 + 0x40; // NAN
  if ((u16 & 0x7FFF) == 0 && isnan(f32))
    return u16 - 1; // NAN
  return u16;
}

uint16_t om_f8e5m2_to_f16(uint8_t u8) {
  uint16_t u16 = u8;
  return u16 << 8;
}

uint8_t om_f16_to_f8e5m2(uint16_t u16) {
  if ((u16 & 0x7C00) == 0x7C00) { // NaN or INF
    if ((u16 & 0x7CFF) != 0x7C00) {
      // need to fix u16 if it's 0x7C and u16 is not INF,
      // so let's just use same NaN value in all cases:
      // 0x7F if positive, 0xFF if negative
      u16 |= 0x7F00;
    }
  } else {
    // emulate llvm::RoundingMode::NearestTiesToEven,
    // note that this rounds large values 0x7B80-7BFF to INF,
    // and the same for large negative values to -INF
    u16 += 0x80 - ((u16 & 0x1ff) == 0x80);
  }
  return u16 >> 8;
}

uint8_t om_f16_to_f8e5m2_saturate(uint16_t u16) {
  // 0x7FFF masks out the sign bit,
  // 0x7B80-7BFF are large numbers that would round to INF,
  // 0x7C00 is INF, 0x7C01-0x7FFF are NaNs
  if ((u16 & 0x7FFF) >= 0x7B80) { // u16 is large or INF or NaN
    // Subtract 1 to change INF (0x7C00) to have high bits 0x7B (plus sign bit),
    // same as large values 0x7B80-7BFF (now 0x7B7F-7BFE),
    // while NaNs (go from 0x7C01-0x7FFF t0 0x7C00-0x7FFF) still
    // have high byte in the range 0x7C-7F (plus sign bit).
    u16 -= 1;
    // Changes all NaNs to have high byte 0x7F (plus sign bit).
    u16 |= 0x300;
  } else {
    // emulate llvm::RoundingMode::NearestTiesToEven
    u16 += 0x80 - ((u16 & 0x1ff) == 0x80);
  }
  return u16 >> 8;
}
