// C inline "implementation", see: https://stackoverflow.com/a/48172918

#include "SmallFPConversion.h"

extern inline float om_f16_to_f32(uint16_t u16);
extern inline uint16_t om_f32_to_f16(float f32);
extern inline float om_bf16_to_f32(uint16_t u16);
extern inline uint16_t om_f32_to_bf16(float f32);
