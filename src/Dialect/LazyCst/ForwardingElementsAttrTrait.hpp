/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/TypeID.h"

namespace lazycst {

// An attribute can implement this trait together with ElementsAttrInterface
// to forward all ElementsAttr methods to another ElementsAttr.
//
// Contract:
// 1. Attr must have a getElementsAttr() instance method which returns
//    an ElementsAttr which ForwardingElementsAttrTrait forwards to.
// 2. Attr must implement ElementsAttrInterface and the method:
//    template <typename T> auto try_value_begin_impl(OverloadToken<T>) const {
//      return getElementsAttr().try_value_begin<T>();
//    }
template <typename Attr>
class ForwardingElementsAttrTrait : public mlir::AttributeTrait::TraitBase<Attr,
                                        ForwardingElementsAttrTrait> {
private:
  using Base =
      mlir::AttributeTrait::TraitBase<Attr, ForwardingElementsAttrTrait>;
  mlir::ElementsAttr getElementsAttr() const {
    return Base::getInstance().getElementsAttr();
  }

public:
  auto getValuesImpl(mlir::TypeID elementID) const {
    return getElementsAttr().getValuesImpl(elementID);
  }
  template <typename X>
  auto value_end() const {
    return getElementsAttr().template value_end<X>();
  }
};

} // namespace lazycst
