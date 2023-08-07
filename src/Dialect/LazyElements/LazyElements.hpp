/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/Dialect/LazyElements/FileDataManager.hpp"

#include "src/Dialect/LazyElements/BType.hpp"
#include "src/Dialect/LazyElements/WideNum.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

#include <tuple>
#include <type_traits>

namespace lazy_elements {
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
} // namespace lazy_elements

#include "src/Dialect/LazyElements/LazyElementsDialect.hpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/LazyElements/LazyElementsAttributes.hpp.inc"

namespace lazy_elements {

namespace detail {

template <typename T>
T getNumber(mlir::Type elementType, BType tag, WideNum n) {
  (void)elementType; // Suppresses compiler warning.
  (void)tag;         // Suppresses compiler warning.
  if constexpr (std::is_same_v<T, mlir::Attribute>)
    if (isFloatBType(tag))
      return mlir::FloatAttr::get(elementType, n.toAPFloat(tag));
    else
      return mlir::IntegerAttr::get(elementType, n.toAPInt(tag));
  else if constexpr (std::is_same_v<T, mlir::IntegerAttr>)
    return mlir::IntegerAttr::get(
        elementType, n.toAPInt(tag)); // fails if float
  else if constexpr (std::is_same_v<T, mlir::FloatAttr>)
    return mlir::FloatAttr::get(
        elementType, n.toAPFloat(tag)); // fails if !float
  else if constexpr (std::is_same_v<T, llvm::APInt>)
    return n.toAPInt(tag); // fails if isFloatBType(tag)
  else if constexpr (std::is_same_v<T, llvm::APFloat>)
    return n.toAPFloat(tag); // fails unless isFloatBType(tag)
  else if constexpr (std::is_same_v<T, WideNum>)
    return n;
  else
    return n.to<T>(tag);
}

template <typename CppType>
inline WideNum lookupWideNum(const CppType *data, size_t idx) {
  constexpr BType TAG = toBType<CppType>;
  return WideNum::widen<TAG>(data[idx]);
}

template <typename CppType>
inline llvm::APFloat lookupAPFloat(const CppType *data, size_t idx) {
  constexpr BType TAG = toBType<CppType>;
  return lookupWideNum(data, idx).toAPFloat(TAG);
}

template <typename CppType>
inline llvm::APInt lookupAPInt(const CppType *data, size_t idx) {
  constexpr BType TAG = toBType<CppType>;
  return lookupWideNum(data, idx).toAPInt(TAG);
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
    mlir::Type elementType, llvm::ArrayRef<char> rawBytes) {
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
    mlir::Type elementType, llvm::ArrayRef<char> rawBytes) {
  if (elementType.isInteger(1))
    return getIntLookupFnHelper<X, bool>(elementType, rawBytes);
  llvm_unreachable("TODO: support more integer types");
}

} // namespace detail

template <typename X>
inline auto FileDataElementsAttr::try_value_begin_impl(OverloadToken<X>) const
    -> mlir::FailureOr<iterator<X>> {
  if constexpr (isContiguousType<X>) {
    return reinterpret_cast<const X *>(getRawBytes().data());
  } else if constexpr (isNonContiguousType<X>) {
    std::function<X(size_t)> lookupFn;
    mlir::Type elType = getElementType();
    if (auto fpType = llvm::dyn_cast<mlir::FloatType>(elType)) {
      if constexpr (llvm::is_one_of<X, llvm::APInt, mlir::IntegerAttr>::value) {
        return mlir::failure();
      } else {
        lookupFn = detail::getFPLookupFunction<X>(fpType, getRawBytes());
      }
    } else if (auto intType = llvm::dyn_cast<mlir::IntegerType>(elType)) {
      if constexpr (llvm::is_one_of<X, llvm::APInt, mlir::IntegerAttr>::value) {
        return mlir::failure();
      } else {
        lookupFn = detail::getIntLookupFunction<X>(intType, getRawBytes());
      }
    }
    auto range = llvm::seq<size_t>(0, getNumElements());
    return iterator<X>(range.begin(), lookupFn);
  } else {
    llvm_unreachable("unsupported cpp type");
  }
}

template <typename X>
inline X FileDataElementsAttr::getSplatValue() const {
  static_assert(isIterableType<X>, "unsupported cpp type");
  assert(isSplat());
  return detail::getNumber<X>(getElementType(), getBType(), atFlatIndex(0));
}

} // namespace lazy_elements