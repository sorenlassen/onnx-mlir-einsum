# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_interface(ACConstantFoldableOpInterface)
add_onnx_mlir_dialect(LazyCst lazycst)
add_onnx_mlir_dialect_doc(lazycst LazyCst.td)

add_onnx_mlir_library(OMLazyCst
  ACConstantFoldableOpInterface.cpp
  LazyCstDialect.cpp
  LazyCstAttributes.cpp
  LazyCstOps.cpp
  ConstantFoldableAnalysis.cpp
  ConstantFolder.cpp
  GraphEvaluator.cpp
  FileDataManager.cpp
  LazyFunctionManager.cpp

  DEPENDS
  OMACConstantFoldableOpInterfaceIncGen
  OMLazyCstIncGen

  LINK_LIBS PUBLIC
  OMMlirUtilities
  OMDenseLikeElementsAttrInterface
  MLIRIR
  MLIRDialect # Dialect/Traits
  )
