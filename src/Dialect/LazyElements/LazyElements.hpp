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

#include "src/Dialect/LazyElements/LazyElementsDialect.hpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/LazyElements/LazyElementsAttributes.hpp.inc"

namespace lazy_elements {

namespace detail {
// True for the types T in FileDataElementsAttr::ContiguousIterableTypesT
// and FileDataElementsAttr::NonContiguousIterableTypesT.
template <typename T>
constexpr bool isIterableType =
    llvm::is_one_of<T, mlir::Attribute, mlir::IntegerAttr, mlir::FloatAttr,
        llvm::APInt, llvm::APFloat, WideNum>::value ||
    CppTypeTrait<T>::isIntOrFloat;

// Supports all the types T in NonContiguousIterableTypesT.
template <typename T>
T getNumber(mlir::Type elementType, BType tag, WideNum n) {
  static_assert(isIterableType<T>);
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
} // namespace detail

template <typename X>
inline auto FileDataElementsAttr::try_value_begin_impl(OverloadToken<X>) const
    -> mlir::FailureOr<iterator<X>> {
  if constexpr (detail::isIterableType<X>) {
    if constexpr (is_in<X, ContiguousIterableTypesT>) {
      return reinterpret_cast<const X *>(getRawBytes().data());
    } else {
      BType btype = getBType();
      if constexpr (llvm::is_one_of<X, llvm::APFloat, mlir::FloatAttr>::value) {
        if (!isFloatBType(btype))
          return mlir::failure();
      } else if constexpr (llvm::is_one_of<X, llvm::APInt,
                               mlir::IntegerAttr>::value) {
        if (isFloatBType(btype))
          return mlir::failure();
      }
      // Translate "this" to a FileDataElementsAttr to work around that "this"
      // becomes something strange as we wind our way to try_value_begin_impl()
      // via interfaces from the original call to this->value_end()/getValues().
      FileDataElementsAttr attr = *this;
      auto range = llvm::seq<size_t>(0, getNumElements());
      return iterator<X>(range.begin(), [btype, attr](size_t flatIndex) -> X {
        WideNum n = attr.atFlatIndex(flatIndex);
        return detail::getNumber<X>(attr.getElementType(), btype, n);
      });
    }
  } else {
    return mlir::failure();
  }
}

template <typename X>
inline X FileDataElementsAttr::getSplatValue() const {
  assert(isSplat());
  return detail::getNumber<X>(getElementType(), getBType(), atFlatIndex(0));
}

} // namespace lazy_elements