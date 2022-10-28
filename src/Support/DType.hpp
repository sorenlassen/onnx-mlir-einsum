/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- DType.hpp ------------------------------===//
//
// Basic data types.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/ErrorHandling.h"

namespace onnx_mlir {

namespace detail {
uint64_t bitcastAPFloat(llvm::APFloat, const llvm::fltSemantics &semantics);

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
llvm::APFloat toAPFloatFromInt(T i) {
  // TODO: figure out if it's better to use llvm::APFloat::convertFromAPInt
  return llvm::APFloat(static_cast<double>(i));
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
auto toIntFromAPFloat(llvm::APFloat f) -> T {
  // TODO: figure out if there is a better way to do this
  return static_cast<T>(f.convertToDouble());
}

template <typename ConcreteT>
struct float16Base {
public:
  float16Base() = default;
  explicit float16Base(ConcreteT f16) : u16(f16.u16) {}
  explicit float16Base(float f) : u16(fromAPFloat(llvm::APFloat(f)).u16) {}
  explicit float16Base(double d) : u16(fromAPFloat(llvm::APFloat(d)).u16) {}
  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  explicit float16Base(T i) : u16(fromAPFloat(toAPFloatFromInt(i)).u16) {}

  explicit operator float() const {
    return toAPFloat(*static_cast<const ConcreteT *>(this)).convertToFloat();
  }
  explicit operator double() const {
    return toAPFloat(*static_cast<const ConcreteT *>(this)).convertToDouble();
  }
  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  explicit operator T() const {
    return toIntFromAPFloat<T>(
        toAPFloat(*static_cast<const ConcreteT *>(this)));
  }

  static llvm::APFloat toAPFloat(ConcreteT f16) {
    return llvm::APFloat(ConcreteT::semantics(), llvm::APInt(16, f16.u16));
  }
  static ConcreteT fromAPFloat(llvm::APFloat a) {
    uint16_t u16 = bitcastAPFloat(a, ConcreteT::semantics());
    return bitcastFromU16(u16);
  }
  // Same as static_cast<float>(f16).
  static float toFloat(ConcreteT f16) {
    return toAPFloat(f16).convertToFloat();
  }
  // Same as static_cast<ConcreteT>(f).
  static ConcreteT fromFloat(float f) { return fromAPFloat(llvm::APFloat(f)); }
  // Same as reinterpret_cast<uint16_t>(f16).
  static uint16_t bitcastToU16(ConcreteT f16) { return f16.u16; }
  // Same as reinterpret_cast<ConcreteT>(u).
  static ConcreteT bitcastFromU16(uint16_t u) {
    ConcreteT f16;
    f16.u16 = u;
    return f16;
  }

private:
  uint16_t u16;
};
} // namespace detail

struct float_16;
struct bfloat_16;

// Represents a FLOAT16 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint16_t, float, etc.
struct float_16 : public detail::float16Base<float_16> {
  using Base = detail::float16Base<float_16>;
  using Base::Base;
  // explicit float_16(bfloat_16 bf16) : Base(static_cast<float>(bf16)) {}
  explicit float_16(bfloat_16 bf16);
  explicit operator bfloat_16() const;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::IEEEhalf();
  }
};

// Represents a BFLOAT16 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint16_t, float, etc.
struct bfloat_16 : public detail::float16Base<bfloat_16> {
  // using detail::float16Base<bfloat_16>::float16Base;
  using Base = detail::float16Base<bfloat_16>;
  using Base::Base;
  // explicit bfloat_16(float_16 f16) :
  // detail::float16Base<bfloat_16>::float16Base(static_cast<float>(f16)) {}
  explicit bfloat_16(float_16 f16);
  explicit operator float_16() const;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::BFloat();
  }
};

inline float_16::float_16(bfloat_16 bf16)
    : float_16::Base(bfloat_16::toFloat(bf16)) {}
inline float_16::operator bfloat_16() const { return bfloat_16(*this); }
inline bfloat_16::bfloat_16(float_16 f16)
    : bfloat_16::Base(float_16::toFloat(f16)) {}
inline bfloat_16::operator float_16() const { return float_16(*this); }

bool isIntOrFPType(mlir::Type t, unsigned maxWidth);

// Union of 64-bit integers and double precision floating point numbers.
// It is tagless and should always be used in a conjunction
// with an mlir Type which is either an IntegerType or FloatType.
// The type tags which field of the union is populated:
// dbl if FloatType, i64 or u64 if IntegerType and isSigned() or not.
//
// Use toIntOrFP<T> and fromIntOrFP<T> to create and access with the
// tagging mlir Type.
//
union IntOrFP {
  double dbl;   // Floating point numbers with precision and range up to double.
  int64_t i64;  // Signed ints up to bitwidth 64.
  uint64_t u64; // Unsigned ints up to bitwidth 64, including bool.
};

llvm::APFloat toAPFloat(mlir::FloatType ftag, IntOrFP n);
llvm::APInt toAPInt(mlir::IntegerType itag, IntOrFP n);

template <typename T>
inline T fromIntOrFP(mlir::Type tag, IntOrFP n) {
  assert(isIntOrFPType(tag, 64)); // TODO remove after testing, too expensive
  if (auto itag = tag.dyn_cast<mlir::IntegerType>()) {
    if (itag.isSigned())
      return static_cast<T>(n.i64);
    else
      return static_cast<T>(n.u64);
  }
  return static_cast<T>(n.dbl);
}

template <>
inline auto fromIntOrFP<llvm::APFloat>(mlir::Type tag, IntOrFP n)
    -> llvm::APFloat {
  return toAPFloat(tag.cast<mlir::FloatType>(), n);
}

template <>
inline auto fromIntOrFP<llvm::APInt>(mlir::Type tag, IntOrFP n) -> llvm::APInt {
  return toAPInt(tag.cast<mlir::IntegerType>(), n);
}

// template <typename X>
// inline IntOrFP toInt(mlir::IntegerType itag, X x) {
//   assert(itag.getWidth() <= maxWidth);
//   if (itag.isSigned())
//     return { .i64 = static_cast<int64_t>(x) };
//   else
//     return { .u64 = static_cast<uint64_t>(x) };
// }

// template <typename X>
// inline IntOrFP toFP(mlir::Type tag, X x) {
//   assert(tag.cast<mlir::FloatType>().getWidth() <= maxWidth); // TODO remove
//   return { .dbl = static_cast<double>(x) };
// }

template <typename X>
inline IntOrFP toIntOrFP(mlir::Type tag, X x) {
  assert(isIntOrFPType(tag, 64)); // TODO remove after testing, too expensive
  if (auto itag = tag.dyn_cast<mlir::IntegerType>()) {
    if (itag.isSigned())
      return {.i64 = static_cast<int64_t>(x)};
    else
      return {.u64 = static_cast<uint64_t>(x)};
  }
  return {.dbl = static_cast<double>(x)};
}

template <>
inline IntOrFP toIntOrFP<llvm::APFloat>(mlir::Type tag, llvm::APFloat x) {
  assert(tag.isa<mlir::FloatType>());
  return {.dbl = x.convertToDouble()};
}

template <>
inline IntOrFP toIntOrFP<llvm::APInt>(mlir::Type tag, llvm::APInt x) {
  auto itag = tag.cast<mlir::IntegerType>();
  if (itag.isSigned())
    return {.i64 = x.getSExtValue()};
  else
    return {.u64 = x.getZExtValue()};
}

// Numerical representation of basic data types.
//
// DType faithfully copies onnx::TensorProto::DataType from
// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
// and DType and onnx::TensorProto::DataType can be used interchangeably.
// In some places it is convenient to use DType to avoid compile time
// dependencies on third_party/onnx.
enum class DType : int {
  // clang-format off
  UNDEFINED = 0,
  // Basic types.
  FLOAT = 1,   // float
  UINT8 = 2,   // uint8_t
  INT8 = 3,    // int8_t
  UINT16 = 4,  // uint16_t
  INT16 = 5,   // int16_t
  INT32 = 6,   // int32_t
  INT64 = 7,   // int64_t
  STRING = 8,  // string
  BOOL = 9,    // bool

  // IEEE754 half-precision floating-point format (16 bits wide).
  // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  FLOAT16 = 10,

  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,     // complex with float32 real and imaginary components
  COMPLEX128 = 15,    // complex with float64 real and imaginary components

  // Non-IEEE floating-point format based on IEEE754 single-precision
  // floating-point number truncated to 16 bits.
  // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
  BFLOAT16 = 16
  // clang-format on
};

using IntOrFPDTypes = std::tuple<bool, int8_t, uint8_t, int16_t, uint16_t,
    int32_t, uint32_t, int64_t, uint64_t, double, float, float_16, bfloat_16>;

DType fromIntOrFPMlirTypeToDType(mlir::Type type);

template <DType TY>
struct DTypeTrait {
  static constexpr DType dtype = TY;
};

namespace detail {
template <DType DTYPE, typename ty, typename unpacked_ty = ty>
struct DTypeTraitBase {
  static constexpr DType dtype = DTYPE;
  static constexpr bool is_int = std::is_integral_v<ty>;
  static constexpr bool is_float = std::is_floating_point_v<ty>;
  static constexpr unsigned width =
      std::is_same_v<ty, bool> ? 1 : (8 * sizeof(ty));
  using type = ty;
  using unpacked_type = unpacked_ty;
};
} // namespace detail

template <>
struct DTypeTrait<DType::FLOAT16>
    : public detail::DTypeTraitBase<DType::FLOAT16, float_16, float> {
  static constexpr bool is_float = true;
};

template <>
struct DTypeTrait<DType::BFLOAT16>
    : public detail::DTypeTraitBase<DType::BFLOAT16, bfloat_16, float> {
  static constexpr bool is_float = true;
};

#define DEFINE_DTypeTrait(TY, CPPTY)                                           \
  template <>                                                                  \
  struct DTypeTrait<DType::TY>                                                 \
      : public detail::DTypeTraitBase<DType::TY, CPPTY> {};

DEFINE_DTypeTrait(BOOL, bool);
DEFINE_DTypeTrait(INT8, int8_t);
DEFINE_DTypeTrait(UINT8, uint8_t);
DEFINE_DTypeTrait(INT16, int16_t);
DEFINE_DTypeTrait(UINT16, uint16_t);
DEFINE_DTypeTrait(INT32, int32_t);
DEFINE_DTypeTrait(UINT32, uint32_t);
DEFINE_DTypeTrait(INT64, int64_t);
DEFINE_DTypeTrait(UINT64, uint64_t);
DEFINE_DTypeTrait(DOUBLE, double);
DEFINE_DTypeTrait(FLOAT, float);

template <typename>
struct DTypeTraitByType {};
template <>
struct DTypeTraitByType<bool> : public DTypeTrait<DType::BOOL> {};
template <>
struct DTypeTraitByType<int8_t> : public DTypeTrait<DType::INT8> {};
template <>
struct DTypeTraitByType<uint8_t> : public DTypeTrait<DType::UINT8> {};
template <>
struct DTypeTraitByType<int16_t> : public DTypeTrait<DType::INT16> {};
template <>
struct DTypeTraitByType<uint16_t> : public DTypeTrait<DType::UINT16> {};
template <>
struct DTypeTraitByType<int32_t> : public DTypeTrait<DType::INT32> {};
template <>
struct DTypeTraitByType<uint32_t> : public DTypeTrait<DType::UINT32> {};
template <>
struct DTypeTraitByType<int64_t> : public DTypeTrait<DType::INT64> {};
template <>
struct DTypeTraitByType<uint64_t> : public DTypeTrait<DType::UINT64> {};
template <>
struct DTypeTraitByType<double> : public DTypeTrait<DType::DOUBLE> {};
template <>
struct DTypeTraitByType<float> : public DTypeTrait<DType::FLOAT> {};
template <>
struct DTypeTraitByType<float_16> : public DTypeTrait<DType::FLOAT16> {};
template <>
struct DTypeTraitByType<bfloat_16> : public DTypeTrait<DType::BFLOAT16> {};

template <template <typename, typename...> class Action, typename Out>
struct dispatchInt {
#define ACT(TY) (Action<DTypeTrait<DType::TY>, Ts...>::eval(xs...))
  // clang-format off
  template <typename... Ts>
  static Out eval(DType dtype, Ts... xs) {
    switch (dtype) {
      case DType::BOOL  : return ACT(BOOL);
      case DType::INT8  : return ACT(INT8);
      case DType::UINT8 : return ACT(UINT8);
      case DType::INT16 : return ACT(INT16);
      case DType::UINT16: return ACT(UINT16);
      case DType::INT32 : return ACT(INT32);
      case DType::UINT32: return ACT(UINT32);
      case DType::INT64 : return ACT(INT64);
      case DType::UINT64: return ACT(UINT64);
      default: llvm_unreachable("not a supported integer type");
    }
  }
  template <typename... Ts>
  static Out eval(mlir::Type type, Ts... xs) {
    auto itype = type.cast<mlir::IntegerType>();
    switch (itype.getWidth()) {
      case  1: return ACT(BOOL);
      case  8: return itype.isUnsigned() ? ACT(UINT8)  : ACT(INT8);
      case 16: return itype.isUnsigned() ? ACT(UINT16) : ACT(INT16);
      case 32: return itype.isUnsigned() ? ACT(UINT32) : ACT(INT32);
      case 64: return itype.isUnsigned() ? ACT(UINT64) : ACT(INT64);
      default: llvm_unreachable("unsupported integer width");
    }
  }
  // clang-format on
#undef ACT
};

template <template <typename, typename...> class Action, typename Alt,
    typename Out>
struct dispatchFPOr {
#define ACT(TY) (Action<DTypeTrait<DType::TY>, Ts...>::eval(xs...))
  // clang-format off
  template <typename... Ts>
  static Out eval(DType dtype, Ts... xs) {
    switch (dtype) {
      case DType::DOUBLE  : return ACT(DOUBLE);
      case DType::FLOAT   : return ACT(FLOAT);
      case DType::FLOAT16 : return ACT(FLOAT16);
      case DType::BFLOAT16: return ACT(BFLOAT16);
      default: return Alt::eval(dtype, xs...);
    }
  }
  template <typename... Ts>
  static Out eval(mlir::Type type, Ts... xs) {
    if (type.isa<mlir::Float64Type>())  return ACT(DOUBLE);
    if (type.isa<mlir::Float32Type>())  return ACT(FLOAT);
    if (type.isa<mlir::Float16Type>())  return ACT(FLOAT16);
    if (type.isa<mlir::BFloat16Type>()) return ACT(BFLOAT16);
    return Alt::eval(type, xs...);
  }
  // clang-format on
#undef ACT
};

template <typename Out>
struct dispatchFail {
  template <typename T, typename... Ts>
  static Out eval(T dtype, Ts... xs) {
    llvm_unreachable("unsupported type");
  }
};

template <template <typename, typename...> class Action, typename Out>
using dispatchFP = dispatchFPOr<Action, dispatchFail<Out>, Out>;

template <template <typename, typename...> class Action, typename Out>
using dispatchFPOrInt = dispatchFPOr<Action, dispatchInt<Action, Out>, Out>;

// Helper functions frequently used together with dispatch classes.

template <typename New, typename Old = char>
llvm::ArrayRef<New> castArrayRef(llvm::ArrayRef<Old> a) {
  return llvm::makeArrayRef(reinterpret_cast<const New *>(a.data()),
      (a.size() * sizeof(Old)) / sizeof(New));
}

template <typename New, typename Old = char>
llvm::MutableArrayRef<New> castMutableArrayRef(llvm::MutableArrayRef<Old> a) {
  return llvm::makeMutableArrayRef(reinterpret_cast<New *>(a.data()),
      (a.size() * sizeof(Old)) / sizeof(New));
}

template <typename Src, typename Dst, typename Fn>
void fillOrTransform(
    llvm::ArrayRef<Src> src, llvm::MutableArrayRef<Dst> dst, Fn fn) {
  if (src.size() == 1)
    std::fill(dst.begin(), dst.end(), fn(src.front()));
  else
    std::transform(src.begin(), src.end(), dst.begin(), fn);
}

template <typename U>
using onlyFP = std::enable_if_t<std::is_floating_point_v<U> ||
                                std::is_same_v<U, float_16> ||
                                std::is_same_v<U, bfloat_16>>;

template <typename U>
using notBool = std::enable_if_t<!std::is_same_v<U, bool>>;

unsigned widthOfIntOrFPType(mlir::Type t);

template <typename T>
constexpr bool isIntOrFP(unsigned maxWidth) {
  using Trait = DTypeTraitByType<T>;
  return (Trait::is_int || Trait::is_float) && Trait::width <= maxWidth;
}

template <>
constexpr bool isIntOrFP<llvm::APFloat>(unsigned maxWidth) {
  return true;
}

template <>
constexpr bool isIntOrFP<llvm::APInt>(unsigned maxWidth) {
  return true;
}

} // namespace onnx_mlir