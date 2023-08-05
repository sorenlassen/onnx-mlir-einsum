/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

#include "src/Dialect/LazyElements/LazyElementsDialect.hpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/LazyElements/LazyElementsAttributes.hpp.inc"
