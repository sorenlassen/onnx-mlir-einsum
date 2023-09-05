/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "src/Dialect/LazyCst/ConstantFolder.hpp"
#include "src/Dialect/LazyCst/FileDataManager.hpp"

#include "src/Dialect/LazyCst/WideNum.hpp"
#include "src/Interface/DenseLikeElementsAttrInterface.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "llvm/ADT/Sequence.h"

#include <tuple>
#include <type_traits>

namespace lazycst {

using DenseLikeElementsAttrInterface = mlir::DenseLikeElementsAttrInterface;

// Makes deep copy.
mlir::DenseElementsAttr toDenseElementsAttrFromRawBytes(
    mlir::ShapedType, llvm::ArrayRef<char> bytes);

class LazyFunctionManager {
public:
  LazyFunctionManager() : counter(0) {}
  mlir::StringAttr nextName(mlir::ModuleOp module);

private:
  std::atomic<unsigned> counter;
};

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
    if constexpr (!llvm::is_one_of<X, llvm::APFloat, mlir::FloatAttr>::value)
      return getIntLookupFunction<X>(intType, rawBytes);
  }
  return nullptr;
}

template <typename X, typename... Ts>
constexpr bool isOneOfTuplePtrTypes(std::tuple<Ts...> *p) {
  return llvm::is_one_of<X, Ts...>::value;
}
template <typename X, typename Tuple>
constexpr bool isOneOfTupleTypes = isOneOfTuplePtrTypes<X>((Tuple *)nullptr);

template <typename X>
bool isValidElementType(mlir::Type elementType) {
  if constexpr (std::is_same_v<X, bool>) {
    return elementType.isInteger(1);
  } else {
    return true; // TODO: check int/float, width, sign, etc
  }
}

} // namespace detail

using RawBytesContiguousIterTypes = std::tuple<int8_t, uint8_t, int16_t,
    uint16_t, int32_t, uint32_t, int64_t, uint64_t, bool, double, float,
    onnx_mlir::float_16, onnx_mlir::bfloat_16, onnx_mlir::float_8e4m3fn,
    onnx_mlir::float_8e4m3fnuz, onnx_mlir::float_8e5m2,
    onnx_mlir::float_8e5m2fnuz>;

using RawBytesNonContiguousIterTypes = std::tuple<mlir::Attribute,
    mlir::FloatAttr, mlir::IntegerAttr, llvm::APFloat, llvm::APInt, WideNum>;

template <typename X>
using RawBytesIterator = std::conditional_t<
    detail::isOneOfTupleTypes<X, RawBytesContiguousIterTypes>, const X *,
    llvm::mapped_iterator<llvm::iota_range<size_t>::const_iterator,
        std::function<X(size_t)>>>;

template <typename X>
inline auto try_value_begin_from_raw_bytes(mlir::Type elementType,
    llvm::ArrayRef<char> rawBytes) -> mlir::FailureOr<RawBytesIterator<X>> {
  if constexpr (detail::isOneOfTupleTypes<X, RawBytesContiguousIterTypes>) {
    if (detail::isValidElementType<X>(elementType)) {
      return reinterpret_cast<const X *>(rawBytes.data());
    } else {
      return mlir::failure();
    }
  } else if constexpr (detail::isOneOfTupleTypes<X,
                           RawBytesNonContiguousIterTypes>) {
    if (auto lookupFn = detail::getLookupFunction<X>(elementType, rawBytes)) {
      llvm::iota_range<size_t>::const_iterator begin(0);
      return RawBytesIterator<X>(begin, std::move(lookupFn));
    } else {
      return mlir::failure();
    }
  } else {
    llvm_unreachable("unsupported cpp type");
  }
}

} // namespace lazycst

#include "src/Dialect/LazyCst/LazyCstDialect.hpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/LazyCst/LazyCstAttributes.hpp.inc"
