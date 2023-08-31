# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_dialect(LazyCst lazycst)
add_onnx_mlir_dialect_doc(lazycst LazyCst.td)

add_onnx_mlir_library(OMLazyCst
  LazyCst.cpp
  LazyCstOps.cpp
  BType.cpp
  ConstantFoldableAnalysis.cpp
  ConstantFolder.cpp
  FileDataManager.cpp
  WideNum.cpp

  DEPENDS
  OMLazyCstIncGen

  LINK_LIBS PUBLIC
  OMMlirUtilities
  MLIRIR
  MLIRDialect # Dialect/Traits
  )
