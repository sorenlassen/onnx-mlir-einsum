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

template <typename F16Type> // F16Type is the derived class, float_16 or bfloat_16.
class F16Base {
public:
  F16Base() = default;

  explicit F16Base(F16Type f16) : u16(f16.u16) {}
  // Support static_cast<F16Type>(X) for any x that is convertible to float.
  template <typename T,
      typename = std::enable_if_t<!std::is_same_v<T, F16Type>>>
  explicit F16Base(T x)
      : u16(fromAPFloat(llvm::APFloat(static_cast<float>(x))).u16) {}

  // Support static_cast<T>(*this) for any T that float converts to.
  template <typename T,
      typename = std::enable_if_t<!std::is_same_v<T, F16Type>>>
  explicit operator T() const {
    return static_cast<float>(toFloat());
  }

  llvm::APFloat toAPFloat() const {
    return llvm::APFloat(F16Type::semantics(), llvm::APInt(16, u16));
  }

  // Same as static_cast<float>(*this).
  float toFloat() const { return toAPFloat().convertToFloat(); }

  // Same as reinterpret_cast<uint16_t>(*this).
  uint16_t bitcastToU16() const { return u16; }

  static F16Type fromAPFloat(llvm::APFloat a) {
    uint16_t u16 = bitcastAPFloat(a, F16Type::semantics());
    return bitcastFromU16(u16);
  }

  // Same as static_cast<F16Type>(f).
  static F16Type fromFloat(float f) { return fromAPFloat(llvm::APFloat(f)); }

  // Same as reinterpret_cast<F16Type>(u).
  static F16Type bitcastFromU16(uint16_t u) {
    F16Type f16;
    f16.u16 = u;
    return f16;
  }

private:
  uint16_t u16;
};
} // namespace detail

template <class T>
inline constexpr bool isF16Type = std::is_base_of_v<detail::F16Base<T>, T>;

// Represents a FLOAT16 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint16_t, float, etc.
struct float_16 : public detail::F16Base<float_16> {
  using Base = detail::F16Base<float_16>;
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::IEEEhalf();
  }
};

// Represents a BFLOAT16 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint16_t, float, etc.
struct bfloat_16 : public detail::F16Base<bfloat_16> {
  using Base = detail::F16Base<bfloat_16>;
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::BFloat();
  }
};

template <typename T>
using toArithmetic = std::conditional_t<std::is_arithmetic_v<T>, T, float>;

// Numerical representation of basic data types.
//
// DType faithfully copies onnx::TensorProto::DataType from
// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
// and DType and onnx::TensorProto::DataType can be used interchangeably.
// In some places it is convenient to use DType to avoid compile time
// dependencies on third_party/onnx.
enum class DType : int8_t {
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

namespace detail {
template <DType DTYPE, typename CPPTY>
struct DTypeTraitBase {
  static constexpr DType dtype = DTYPE;
  static constexpr bool is_float =
      std::is_floating_point_v<CPPTY> || isF16Type<CPPTY>;
  static constexpr bool is_signed_int =
      std::is_integral_v<CPPTY> && std::is_signed_v<CPPTY>;
  static constexpr bool is_unsigned_int =
      std::is_integral_v<CPPTY> && !std::is_signed_v<CPPTY>;
  static constexpr unsigned width =
      std::is_same_v<CPPTY, bool> ? 1 : (8 * sizeof(CPPTY));
  static constexpr unsigned bytewidth = (width + 1) / 8;
  using cpptype = CPPTY;
  using widetype = std::conditional_t<is_float, double,
      std::conditional_t<is_signed_int, int64_t, uint64_t>>;
};
} // namespace detail

template <DType DTYPE>
struct DTypeTrait {};

template <typename CPPTY>
struct CppTypeTrait : public detail::DTypeTraitBase<DType::UNDEFINED, CPPTY> {};

#define DEFINE_DTypeCppTypeTraits(DTYPE, CPPTY)                                \
  template <>                                                                  \
  struct DTypeTrait<DTYPE> : public detail::DTypeTraitBase<DTYPE, CPPTY> {};   \
  template <>                                                                  \
  struct CppTypeTrait<CPPTY> : public DTypeTrait<DTYPE> {};

DEFINE_DTypeCppTypeTraits(DType::BOOL, bool);
DEFINE_DTypeCppTypeTraits(DType::INT8, int8_t);
DEFINE_DTypeCppTypeTraits(DType::UINT8, uint8_t);
DEFINE_DTypeCppTypeTraits(DType::INT16, int16_t);
DEFINE_DTypeCppTypeTraits(DType::UINT16, uint16_t);
DEFINE_DTypeCppTypeTraits(DType::INT32, int32_t);
DEFINE_DTypeCppTypeTraits(DType::UINT32, uint32_t);
DEFINE_DTypeCppTypeTraits(DType::INT64, int64_t);
DEFINE_DTypeCppTypeTraits(DType::UINT64, uint64_t);
DEFINE_DTypeCppTypeTraits(DType::DOUBLE, double);
DEFINE_DTypeCppTypeTraits(DType::FLOAT, float);
DEFINE_DTypeCppTypeTraits(DType::FLOAT16, float_16);
DEFINE_DTypeCppTypeTraits(DType::BFLOAT16, bfloat_16);

#undef DEFINE_DTypeCppTypeTraits

#if 0
template <>
struct DTypeTrait<DType::FLOAT16>
    : public detail::DTypeTraitBase<DType::FLOAT16, float_16> {
  static constexpr bool is_float = true;
  using widetype = double;
};
template <>
struct CppTypeTrait<float_16> : public DTypeTrait<DType::FLOAT16> {};

template <>
struct DTypeTrait<DType::BFLOAT16>
    : public detail::DTypeTraitBase<DType::BFLOAT16, bfloat_16> {
  static constexpr bool is_float = true;
  using widetype = double;
};
template <>
struct CppTypeTrait<bfloat_16> : public DTypeTrait<DType::BFLOAT16> {};
#endif

template <DType DTYPE>
using CppTypeOf = typename DTypeTrait<DTYPE>::cpptype;

template <typename T>
constexpr DType dtypeOf() {
  return CppTypeTrait<T>::dtype;
}

DType dtypeOfMlirType(mlir::Type type);

mlir::Type mlirTypeOfDType(DType dtype, mlir::MLIRContext *ctx);

template <typename T>
mlir::Type mlirTypeOfCppType(mlir::MLIRContext *ctx) {
  return mlirTypeOfDType(dtypeOf<T>(), ctx);
}

// TODO: find a better place for this
inline unsigned getIntOrFloatByteWidth(mlir::Type t) {
  return (t.getIntOrFloatBitWidth() + 7) / 8;
}

template <typename Action>
auto dispatch(DType dtype, Action &&act) {
  // clang-format off
  switch (dtype) {
  case DType::BOOL     : return act(static_cast<bool>(0));
  case DType::INT8     : return act(static_cast<int8_t>(0));
  case DType::UINT8    : return act(static_cast<uint8_t>(0));
  case DType::INT16    : return act(static_cast<int16_t>(0));
  case DType::UINT16   : return act(static_cast<uint16_t>(0));
  case DType::INT32    : return act(static_cast<int32_t>(0));
  case DType::UINT32   : return act(static_cast<uint32_t>(0));
  case DType::INT64    : return act(static_cast<int64_t>(0));
  case DType::UINT64   : return act(static_cast<uint64_t>(0));
  case DType::DOUBLE   : return act(static_cast<double>(0));
  case DType::FLOAT    : return act(static_cast<float>(0));
  case DType::FLOAT16  : return act(static_cast<float_16>(0));
  case DType::BFLOAT16 : return act(static_cast<bfloat_16>(0));
  default: llvm_unreachable("not a supported integer type");
  }
  // clang-format on
}

template <typename Action>
auto dispatch(mlir::Type type, Action &&act) {
  return dispatch(dtypeOfMlirType(type), std::forward<Action>(act));
}

template <template <typename, typename...> class Action>
struct dispatchInt {
#define ACT(DTYPE) (Action<DTypeTrait<DType::DTYPE>, Ts...>::eval(xs...))
  // clang-format off
  template <typename... Ts>
  static auto eval(DType dtype, Ts... xs) {
    switch (dtype) {
      case DType::BOOL   : return ACT(BOOL);
      case DType::INT8   : return ACT(INT8);
      case DType::UINT8  : return ACT(UINT8);
      case DType::INT16  : return ACT(INT16);
      case DType::UINT16 : return ACT(UINT16);
      case DType::INT32  : return ACT(INT32);
      case DType::UINT32 : return ACT(UINT32);
      case DType::INT64  : return ACT(INT64);
      case DType::UINT64 : return ACT(UINT64);
      default: llvm_unreachable("not a supported integer type");
    }
  }
  template <typename... Ts>
  static auto eval(mlir::Type type, Ts... xs) {
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

template <template <typename, typename...> class Action, typename Alt>
struct dispatchFPOr {
#define ACT(DTYPE) (Action<DTypeTrait<DType::DTYPE>, Ts...>::eval(xs...))
  // clang-format off
  template <typename... Ts>
  static auto eval(DType dtype, Ts... xs) {
    switch (dtype) {
      case DType::DOUBLE   : return ACT(DOUBLE);
      case DType::FLOAT    : return ACT(FLOAT);
      case DType::FLOAT16  : return ACT(FLOAT16);
      case DType::BFLOAT16 : return ACT(BFLOAT16);
      default: return Alt::eval(dtype, xs...);
    }
  }
  template <typename... Ts>
  static auto eval(mlir::Type type, Ts... xs) {
    if (type.isa<mlir::Float64Type>())  return ACT(DOUBLE);
    if (type.isa<mlir::Float32Type>())  return ACT(FLOAT);
    if (type.isa<mlir::Float16Type>())  return ACT(FLOAT16);
    if (type.isa<mlir::BFloat16Type>()) return ACT(BFLOAT16);
    return Alt::eval(type, xs...);
  }
  // clang-format on
#undef ACT
};

struct dispatchFail {
  template <typename T, typename... Ts>
  static auto eval(T dtype, Ts... xs) {
    llvm_unreachable("unsupported type");
  }
};

template <template <typename, typename...> class Action>
using dispatchFP = dispatchFPOr<Action, dispatchFail>;

template <template <typename, typename...> class Action>
using dispatchFPOrInt = dispatchFPOr<Action, dispatchInt<Action>>;

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
using EnableFloat = std::enable_if_t<CppTypeTrait<U>::is_float>;

template <typename U>
using EnableNotBool = std::enable_if_t<!std::is_same_v<U, bool>>;

// Union of 64-bit integers and double precision floating point numbers.
// It is tagless and should always be used in a conjunction
// with an mlir Type which is either an IntegerType or FloatType.
// The type tags which field of the union is populated:
// dbl if FloatType, i64 or u64 if IntegerType and isSigned() or not.
union IntOrFP { // TODO rename to WideIntOrFP
  double dbl;   // Floating point numbers with precision and range up to double.
  int64_t i64;  // Signed ints up to bitwidth 64.
  uint64_t u64; // Unsigned ints up to bitwidth 64, including bool.

  llvm::APInt toAPInt(mlir::IntegerType itag) const;
  llvm::APFloat toAPFloat(mlir::FloatType ftag) const;

  template <typename T>
  constexpr T to(DType dtag) const {
    switch (dtag) {
    case DType::BOOL:
    case DType::UINT8:
    case DType::UINT16:
    case DType::UINT32:
    case DType::UINT64:
      return static_cast<T>(u64);
    case DType::INT8:
    case DType::INT16:
    case DType::INT32:
    case DType::INT64:
      return static_cast<T>(i64);
    case DType::DOUBLE:
    case DType::FLOAT:
    case DType::FLOAT16:
    case DType::BFLOAT16:
      return static_cast<T>(dbl);
    default:
      llvm_unreachable("to unsupported dtype");
    }
  }

  template <typename T>
  std::enable_if_t<
      !std::is_same_v<T, llvm::APInt> && !std::is_same_v<T, llvm::APFloat>, T>
  to(mlir::Type tag) const {
    assert(tag.getIntOrFloatBitWidth() <= 64); // TODO remove, too expensive
    return to<T>(dtypeOfMlirType(tag));
  }
  template <typename T>
  std::enable_if_t<std::is_same_v<T, llvm::APInt>, T> to(mlir::Type tag) const {
    return toAPInt(tag.cast<mlir::IntegerType>());
  }
  template <typename T>
  std::enable_if_t<std::is_same_v<T, llvm::APFloat>, T> to(
      mlir::Type tag) const {
    return toAPFloat(tag.cast<mlir::FloatType>());
  }

  template <typename T>
  static constexpr IntOrFP from(DType dtag, T x) {
    switch (dtag) {
    case DType::BOOL:
    case DType::UINT8:
    case DType::UINT16:
    case DType::UINT32:
    case DType::UINT64:
      return {.u64 = static_cast<uint64_t>(x)};
    case DType::INT8:
    case DType::INT16:
    case DType::INT32:
    case DType::INT64:
      return {.i64 = static_cast<int64_t>(x)};
    case DType::DOUBLE:
    case DType::FLOAT:
    case DType::FLOAT16:
    case DType::BFLOAT16:
      return {.dbl = static_cast<double>(x)};
    default:
      llvm_unreachable("from unsupported dtype");
    }
  }

  template <typename T>
  static std::enable_if_t<!std::is_same_v<T, llvm::APInt> &&
                              !std::is_same_v<T, llvm::APFloat>,
      IntOrFP>
  from(mlir::Type tag, T x) {
    assert(tag.getIntOrFloatBitWidth() <= 64); // TODO remove, too expensive
    return from<T>(dtypeOfMlirType(tag), x);
  }
  template <typename X>
  static std::enable_if_t<std::is_same_v<X, llvm::APInt>, IntOrFP> from(
      mlir::Type tag, X x) {
    if (tag.cast<mlir::IntegerType>().isSigned())
      return {.i64 = x.getSExtValue()};
    else
      return {.u64 = x.getZExtValue()};
  }
  template <typename X>
  static std::enable_if_t<std::is_same_v<X, llvm::APFloat>, IntOrFP> from(
      mlir::Type tag, X x) {
    assert(tag.cast<mlir::FloatType>().getWidth() <= 64); // TODO remove
    return {.f64 = x.convertToDouble()};
  }
};

template <typename T>
constexpr bool isIntOrFPConvertible() {
  // TODO: change to check it's a simple scalar once CppTypeTrait
  // becomes defined for string and complex types
  return CppTypeTrait<T>::dtype != DType::UNDEFINED;
}

template <>
constexpr bool isIntOrFPConvertible<llvm::APFloat>() {
  return true;
}

template <>
constexpr bool isIntOrFPConvertible<llvm::APInt>() {
  return true;
}

} // namespace onnx_mlir