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

// Base class for float_16, bfloat_16.
class FP16Type {
public:
  using bitcasttype = uint16_t;

  // Substitute for reinterpret_cast<uint16_t>(*this), which C++ doesn't allow.
  constexpr bitcasttype bitcastToU16() const { return u16; }

protected:
  constexpr FP16Type() : u16(){};
  constexpr explicit FP16Type(uint16_t u16) : u16(u16) {}

  bitcasttype u16;
};

template <class T>
inline constexpr bool isFP16Type = std::is_base_of_v<FP16Type, T>;

namespace detail {
uint64_t bitcastAPFloat(llvm::APFloat, const llvm::fltSemantics &semantics);

template <typename FP16> // FP16 is the derived class, float_16 or bfloat_16.
class FP16Base : public FP16Type {
public:
  constexpr FP16Base() : FP16Type() {}

  constexpr explicit FP16Base(FP16 f16) : FP16Type(f16.u16) {}
  // Support static_cast<FP16>(X) for any x that is convertible to float.
  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, FP16>>>
  explicit FP16Base(T x)
      : FP16Type(fromAPFloat(llvm::APFloat(static_cast<float>(x))).u16) {}

  // Support static_cast<T>(*this) for any T that float converts to.
  template <typename T, typename = std::enable_if_t<!std::is_same_v<T, FP16>>>
  explicit operator T() const {
    return static_cast<float>(toFloat());
  }

  llvm::APFloat toAPFloat() const {
    return llvm::APFloat(FP16::semantics(), llvm::APInt(16, u16));
  }

  // Same as static_cast<float>(*this).
  float toFloat() const { return toAPFloat().convertToFloat(); }

  static FP16 fromAPFloat(llvm::APFloat a) {
    bitcasttype u16 = bitcastAPFloat(a, FP16::semantics());
    return bitcastFromU16(u16);
  }

  // Substitute for reinterpret_cast<FP16>(f), which C++ doesn't allow.
  static FP16 fromFloat(float f) { return fromAPFloat(llvm::APFloat(f)); }

  // Same as reinterpret_cast<FP16>(u).
  static constexpr FP16 bitcastFromU16(bitcasttype u) {
    FP16 f16;
    f16.u16 = u;
    return f16;
  }
};
} // namespace detail

// Represents a FLOAT16 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint16_t, float, etc.
struct float_16 : public detail::FP16Base<float_16> {
  using Base = detail::FP16Base<float_16>;
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::IEEEhalf();
  }
};

// Represents a BFLOAT16 value with the correct bitwidth and in a form that
// is unambiguous when used as a template parameter alongside the other basic
// Cpp data types uint16_t, float, etc.
struct bfloat_16 : public detail::FP16Base<bfloat_16> {
  using Base = detail::FP16Base<bfloat_16>;
  using Base::Base;
  static const llvm::fltSemantics &semantics() {
    return llvm::APFloat::BFloat();
  }
};

template <typename T>
using toArithmetic = std::conditional_t<std::is_arithmetic_v<T>, T, float>;

// Numerical representation of basic data types.
//
// DType faithfully copies onnx::TensorProto_DataType from
// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
// and DType and onnx::TensorProto_DataType can be used interchangeably.
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

// DType and enum onnx::TensorProto_DataType convert to each other with
// static_cast because DType faithfully copies onnx::TensorProto_DataType.
// The conversion functions onnxDataTypeOfDType and dtypeOfOnnxDataType pass
// onnx::TensorProto_DataType values as int in line with the C++ protobuf API
// in #include "onnx/onnx_pb.h".

// Returns a value from enum onnx::TensorProto_DataType.
constexpr int onnxDataTypeOfDType(DType dtype) {
  return static_cast<int>(dtype);
}
// Precondition: onnxDataType must be from enum onnx::TensorProto_DataType.
constexpr DType dtypeOfOnnxDataType(int onnxDataType) {
  return static_cast<DType>(onnxDataType);
}

namespace detail {
template <DType DTYPE, typename CPPTY>
struct DTypeTraitBase {
  static constexpr DType dtype = DTYPE;
  static constexpr bool isFloat =
      std::is_floating_point_v<CPPTY> || isFP16Type<CPPTY>;
  static constexpr bool isIntOrFloat = std::is_integral_v<CPPTY> || isFloat;
  static constexpr bool isSignedInt =
      std::is_integral_v<CPPTY> && std::is_signed_v<CPPTY>;
  static constexpr bool isUnsignedInt =
      std::is_integral_v<CPPTY> && !std::is_signed_v<CPPTY>;
  static constexpr unsigned width = // TODO: rename to bitwidth
      std::is_same_v<CPPTY, bool> ? 1 : (8 * sizeof(CPPTY));
  static constexpr unsigned bytewidth = (width + 1) / 8;
  using cpptype = CPPTY;
  using widetype = std::conditional_t<isFloat, double,
      std::conditional_t<isSignedInt, int64_t, uint64_t>>;
};
} // namespace detail

template <DType DTYPE>
struct DTypeTrait : public detail::DTypeTraitBase<DTYPE, void> {};

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

template <DType DTYPE>
using CppType = typename DTypeTrait<DTYPE>::cpptype;

template <typename CPPTY>
constexpr DType toDType = CppTypeTrait<CPPTY>::dtype;

template <typename CPPTY>
constexpr DType dtypeOf(CPPTY = CPPTY()) {
  return toDType<CPPTY>;
}

DType dtypeOf(mlir::Type type);

mlir::Type mlirTypeOf(DType dtype, mlir::MLIRContext *ctx);

template <typename CPPTY>
mlir::Type toMlirType(mlir::MLIRContext *ctx) {
  return dtypeOf(toDType<CPPTY>, ctx);
}

template <typename T>
mlir::Type mlirTypeOfCppType(mlir::MLIRContext *ctx) {
  return mlirTypeOfDType(dtypeOf<T>(), ctx);
}

// The following functions isFloatDType(dtype), widthOfDType(dtype), etc are
// helpful alternatives to DTypeTrait<dtype>::isFloat/width/etc
// when dtype isn't known at compile.

// == mlirTypeOf(dtype, ctx).isa<FloatType>()
bool isFloatDType(DType);

// == mlirTypeOf(dtype, ctx).isIntOrFloat()
bool isIntOrFloatDType(DType);

// == mlirTypeOf(dtype, ctx).isSignlessInteger()
bool isSignedIntDType(DType);

// == mlirTypeOf(dtype, ctx).isUnsignedInteger()
bool isUnsignedIntDType(DType);

// == mlirTypeOf(dtype, ctx).getIntOrFloatBitWidth()
unsigned widthOfDType(DType); // TODO: rename to bitwidth

// == getIntOrFloatByteWidth(mlirTypeOf(dtype, ctx))
unsigned bytewidthOfDType(DType);

inline unsigned getIntOrFloatByteWidth(mlir::Type t) {
  return (t.getIntOrFloatBitWidth() + 7) / 8;
}

template <DType DTYPE>
struct DTypeToken {
  constexpr DTypeToken() {}
  constexpr operator DType() const { return DTYPE; }
};

template <typename Action, typename... Args>
auto dispatchByDType(DType dtype, Action &&act, Args &&...args) {
#define ACT(DTYPE) act(DTypeToken<DTYPE>{}, std::forward<Args>(args)...)
  // clang-format off
  switch (dtype) {
  case DType::BOOL     : return ACT(DType::BOOL);
  case DType::INT8     : return ACT(DType::INT8);
  case DType::UINT8    : return ACT(DType::UINT8);
  case DType::INT16    : return ACT(DType::INT16);
  case DType::UINT16   : return ACT(DType::UINT16);
  case DType::INT32    : return ACT(DType::INT32);
  case DType::UINT32   : return ACT(DType::UINT32);
  case DType::INT64    : return ACT(DType::INT64);
  case DType::UINT64   : return ACT(DType::UINT64);
  case DType::DOUBLE   : return ACT(DType::DOUBLE);
  case DType::FLOAT    : return ACT(DType::FLOAT);
  case DType::FLOAT16  : return ACT(DType::FLOAT16);
  case DType::BFLOAT16 : return ACT(DType::BFLOAT16);
  default: llvm_unreachable("not a supported datatype");
  }
  // clang-format on
#undef ACT
}

template <typename Action, typename... Args>
auto dispatchByMlirType(mlir::Type type, Action &&act, Args &&...args) {
  return dispatchByDType(
      dtypeOf(type), std::forward<Action>(act), std::forward<Args>(args)...);
}

// Helper functions frequently used together with dispatch.

template <typename New, typename Old = char>
llvm::ArrayRef<New> castArrayRef(llvm::ArrayRef<Old> a) {
  return llvm::makeArrayRef(reinterpret_cast<const New *>(a.data()),
      (a.size() * sizeof(Old)) / sizeof(New));
}

template <typename New = char>
llvm::ArrayRef<New> asArrayRef(llvm::StringRef s) {
  return llvm::makeArrayRef(
      reinterpret_cast<const New *>(s.data()), s.size() / sizeof(New));
}

template <typename Old = char>
llvm::StringRef asStringRef(llvm::ArrayRef<Old> a) {
  llvm::ArrayRef<char> c = castArrayRef<char>(a);
  return llvm::StringRef(c.begin(), c.size());
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
using EnableFloat = std::enable_if_t<CppTypeTrait<U>::isFloat>;

template <typename U>
using EnableNotBool = std::enable_if_t<!std::is_same_v<U, bool>>;

// Union of 64-bit integers and double precision floating point numbers.
// It is tagless and should always be used in a conjunction
// with a DType or a corresponding int or float mlir Type.
// The dtype tags which field of the union is populated:
// dbl if isFloat(dtype), i64 or u64 if isSigned/UnsignedInt(dtype).
//
// IntOrFP satisfies the property
//
//   static_cast<Y>(x) == IntOrFP::from<X>(dtype, x).To<Y>(dtype)
//
// for all cpp types X and Y value x of type X, provided the wide type of
// dtype (double, int64_t, uint64_t) has enough precision and range to
// represent x up to the precision and range of Y.
//
// TODO: rename IntOrFP to WideNum
//
union IntOrFP {
  double dbl;   // Floating point numbers with precision and range up to double.
  int64_t i64;  // Signed ints up to bitwidth 64.
  uint64_t u64; // Unsigned ints up to bitwidth 64, including bool.

  llvm::APFloat toAPFloat(DType tag) const;
  llvm::APInt toAPInt(DType tag) const;
  static IntOrFP fromAPFloat(DType tag, llvm::APFloat x);
  static IntOrFP fromAPInt(DType tag, llvm::APInt x);

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
};

} // namespace onnx_mlir