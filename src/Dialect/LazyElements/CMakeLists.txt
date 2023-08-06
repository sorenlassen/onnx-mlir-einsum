# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_dialect(LazyElements lazyelms)
add_onnx_mlir_dialect_doc(lazyelms LazyElements.td)

add_onnx_mlir_library(OMLazyElements
  LazyElements.cpp
  FileDataManager.cpp

  DEPENDS
  OMLazyElementsIncGen

  LINK_LIBS PUBLIC
  OMMlirUtilities
  MLIRIR
  MLIRDialect # Dialect/Traits
  )
