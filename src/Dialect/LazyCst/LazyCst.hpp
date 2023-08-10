/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/Dialect/LazyCst/FileDataManager.hpp"

#include "src/Dialect/LazyCst/WideNum.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

#include <tuple>
#include <type_traits>

namespace lazycst {
template <class C>
struct BufferElementsAttr : public mlir::Attribute {
  using Attribute::Attribute;

  // mlir::StringAttr getPath() const;

  llvm::ArrayRef<char> getRawBytes() const {
    return static_cast<const C *>(this)->getRawBytesImpl();
  }

  llvm::ArrayRef<char> getRawBytesImpl() const {
    llvm_unreachable("derived class must implement getRawBytesImpl");
  }
};

std::string escapeIdentifier(llvm::StringRef unescapedIdentifier);

std::string unescapeIdentifier(llvm::StringRef escapedIdentifier);

} // namespace lazycst

#include "src/Dialect/LazyCst/LazyCstDialect.hpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp.inc"

namespace lazycst {

namespace detail {

template <typename CppType>
inline WideNum lookupWideNum(const CppType *data, size_t idx) {
  return WideNum::from<CppType>(data[idx]);
}

template <typename CppType>
inline llvm::APFloat lookupAPFloat(const CppType *data, size_t idx) {
  if constexpr (llvm::is_one_of<CppType, double, float>::value) {
    return llvm::APFloat(data[idx]);
  } else if constexpr (onnx_mlir::isSmallFPType<CppType>) {
    return data[idx].toAPFloat();
  } else {
    llvm_unreachable("unexpected type");
  }
}

template <typename CppType>
inline llvm::APInt lookupAPInt(const CppType *data, size_t idx) {
  constexpr unsigned bitwidth =
      std::is_same_v<CppType, bool> ? 1 : (CHAR_BIT * sizeof(CppType));
  if constexpr (std::is_signed_v<CppType>) {
    // Actually, isSigned flag is ignored because bitwidth <= 64.
    return llvm::APInt(bitwidth, data[idx], /*isSigned=*/true);
  } else {
    return llvm::APInt(bitwidth, data[idx]);
  }
}

template <typename X, typename CppType>
std::function<X(size_t)> getFPLookupFnHelper(
    mlir::Type elementType, llvm::ArrayRef<char> rawBytes) {
  const CppType *data = reinterpret_cast<const CppType *>(rawBytes.data());
  if constexpr (std::is_same_v<X, WideNum>) {
    return [data](size_t flatIndex) { return lookupWideNum(data, flatIndex); };
  } else if constexpr (std::is_same_v<X, llvm::APFloat>) {
    return [data](size_t flatIndex) { return lookupAPFloat(data, flatIndex); };
  } else if constexpr (std::is_base_of_v<X, mlir::FloatAttr>) {
    return [elementType, data](size_t flatIndex) {
      return mlir::FloatAttr::get(elementType, lookupAPFloat(data, flatIndex));
    };
  } else {
    llvm_unreachable("unexpected type");
  }
}

template <typename X>
std::function<X(size_t)> getFPLookupFunction(
    mlir::Type elementType, llvm::ArrayRef<char> rawBytes) {
  if (llvm::isa<mlir::Float64Type>(elementType))
    return getFPLookupFnHelper<X, double>(elementType, rawBytes);
  if (llvm::isa<mlir::Float32Type>(elementType))
    return getFPLookupFnHelper<X, float>(elementType, rawBytes);
  if (llvm::isa<mlir::Float16Type>(elementType))
    return getFPLookupFnHelper<X, onnx_mlir::float_16>(elementType, rawBytes);
  if (llvm::isa<mlir::BFloat16Type>(elementType))
    return getFPLookupFnHelper<X, onnx_mlir::bfloat_16>(elementType, rawBytes);
  if (llvm::isa<mlir::Float8E5M2Type>(elementType))
    return getFPLookupFnHelper<X, onnx_mlir::float_8e5m2>(
        elementType, rawBytes);
  if (llvm::isa<mlir::Float8E4M3FNType>(elementType))
    return getFPLookupFnHelper<X, onnx_mlir::float_8e4m3fn>(
        elementType, rawBytes);
  if (llvm::isa<mlir::Float8E5M2FNUZType>(elementType))
    return getFPLookupFnHelper<X, onnx_mlir::float_8e5m2fnuz>(
        elementType, rawBytes);
  if (llvm::isa<mlir::Float8E4M3FNUZType>(elementType))
    return getFPLookupFnHelper<X, onnx_mlir::float_8e4m3fnuz>(
        elementType, rawBytes);
  llvm_unreachable("unsupported floating point type");
};

template <typename X, typename CppType>
std::function<X(size_t)> getIntLookupFnHelper(
    mlir::IntegerType elementType, llvm::ArrayRef<char> rawBytes) {
  const CppType *data = reinterpret_cast<const CppType *>(rawBytes.data());
  if constexpr (std::is_same_v<X, WideNum>) {
    return [data](size_t flatIndex) { return lookupWideNum(data, flatIndex); };
  } else if constexpr (std::is_same_v<X, llvm::APInt>) {
    return [data](size_t flatIndex) { return lookupAPInt(data, flatIndex); };
  } else if constexpr (std::is_base_of_v<X, mlir::IntegerAttr>) {
    return [elementType, data](size_t flatIndex) {
      return mlir::IntegerAttr::get(elementType, lookupAPInt(data, flatIndex));
    };
  } else {
    llvm_unreachable("unexpected type");
  }
}

template <typename X>
std::function<X(size_t)> getIntLookupFunction(
    mlir::IntegerType elementType, llvm::ArrayRef<char> rawBytes) {
  switch (elementType.getWidth()) {
  case 1:
    return getIntLookupFnHelper<X, bool>(elementType, rawBytes);
  case 8:
    return elementType.isUnsigned()
               ? getIntLookupFnHelper<X, uint8_t>(elementType, rawBytes)
               : getIntLookupFnHelper<X, int8_t>(elementType, rawBytes);
  case 16:
    return elementType.isUnsigned()
               ? getIntLookupFnHelper<X, uint16_t>(elementType, rawBytes)
               : getIntLookupFnHelper<X, int16_t>(elementType, rawBytes);
  case 32:
    return elementType.isUnsigned()
               ? getIntLookupFnHelper<X, uint32_t>(elementType, rawBytes)
               : getIntLookupFnHelper<X, int32_t>(elementType, rawBytes);
  case 64:
    return elementType.isUnsigned()
               ? getIntLookupFnHelper<X, uint64_t>(elementType, rawBytes)
               : getIntLookupFnHelper<X, int64_t>(elementType, rawBytes);
  default:
    llvm_unreachable("unsupported integer type");
  }
}

template <typename X>
std::function<X(size_t)> getLookupFunction(
    mlir::Type elementType, llvm::ArrayRef<char> rawBytes) {
  if (auto fpType = llvm::dyn_cast<mlir::FloatType>(elementType)) {
    if constexpr (!llvm::is_one_of<X, llvm::APInt, mlir::IntegerAttr>::value)
      return getFPLookupFunction<X>(fpType, rawBytes);
  } else if (auto intType = llvm::dyn_cast<mlir::IntegerType>(elementType)) {
    if constexpr (!llvm::is_one_of<X, llvm::APInt, mlir::IntegerAttr>::value)
      return getIntLookupFunction<X>(intType, rawBytes);
  }
  return nullptr;
}

} // namespace detail

template <typename X>
inline auto FileDataElementsAttr::try_value_begin_impl(OverloadToken<X>) const
    -> mlir::FailureOr<iterator<X>> {
  static_assert(isIterableType<X>, "unsupported cpp type");
  if constexpr (isContiguousType<X>) {
    return reinterpret_cast<const X *>(getRawBytes().data());
  } else if constexpr (isNonContiguousType<X>) {
    if (auto lookupFn =
            detail::getLookupFunction<X>(getElementType(), getRawBytes())) {
      auto range = llvm::seq<size_t>(0, getNumElements());
      return iterator<X>(range.begin(), lookupFn);
    } else {
      return mlir::failure();
    }
  } else {
    llvm_unreachable("unsupported cpp type");
  }
}

} // namespace lazycst