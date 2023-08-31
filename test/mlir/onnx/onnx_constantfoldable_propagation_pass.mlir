// RUN: onnx-mlir-opt --constantfoldable-propagation-pass --constantfoldable-analysis-pass --hideDensifiableElementsAttrs=false %s -split-input-file | FileCheck %s

func.func @test_add_arg(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %0 = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
  %1 = onnx.Constant dense<1.0> : tensor<f32>
  %2 = onnx.Constant dense<2.0> : tensor<f32>
  %3 = "onnx.Add"(%0, %1) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  %4 = "onnx.Add"(%3, %2) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  %5 = "onnx.Add"(%4, %arg0) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  onnx.Return %5 : tensor<5xf32>
}
// CHECK-LABEL:  func.func @test_add_arg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5xf32>) -> tensor<5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_1_]]) {constantfoldable} : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_3_]], [[VAR_2_]]) {constantfoldable} : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Add"([[VAR_4_]], [[PARAM_0_]]) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK:           onnx.Return [[VAR_5_]] : tensor<5xf32>
// CHECK:         }

// -----

func.func @test_add_arg_2(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %0 = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
  %1 = onnx.Constant dense<1.0> : tensor<f32>
  %2 = onnx.Constant dense<2.0> : tensor<f32>
  %3 = "onnx.Add"(%0, %1) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  %4 = "onnx.Add"(%3, %arg0) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  %5 = "onnx.Add"(%4, %2) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  onnx.Return %5 : tensor<5xf32>
}
// CHECK-LABEL:  func.func @test_add_arg_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5xf32>) -> tensor<5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_1_]]) {constantfoldable} : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_2_]], [[VAR_3_]]) {constantfoldable} : (tensor<f32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_4_]]) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK:           onnx.Return [[VAR_5_]] : tensor<5xf32>
// CHECK:         }

// -----

func.func @test_sub_const(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %0 = onnx.Constant dense<1.0> : tensor<f32>
  %1 = "onnx.Sub"(%arg0, %0) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  onnx.Return %1 : tensor<5xf32>
}
// CHECK-LABEL:  func.func @test_sub_const
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5xf32>) -> tensor<5xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Neg"([[VAR_0_]]) {constantfoldable} : (tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_1_]]) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<5xf32>
// CHECK:         }

// -----

func.func @test_sub_constantfoldable(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %0 = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
  %1 = onnx.Constant dense<1.0> : tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  %3 = "onnx.Sub"(%arg0, %2) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  %4 = "onnx.Add"(%3, %1) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  onnx.Return %4 : tensor<5xf32>
}
// CHECK-LABEL:  func.func @test_sub_constantfoldable
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5xf32>) -> tensor<5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_1_]]) {constantfoldable} : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Neg"([[VAR_2_]]) {constantfoldable} : (tensor<5xf32>) -> tensor<5xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_1_]], [[VAR_3_]]) {constantfoldable} : (tensor<f32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_4_]]) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK:           onnx.Return [[VAR_5_]] : tensor<5xf32>
// CHECK:         }

// -----

func.func @test_if(%arg0: tensor<i1>, %arg1: tensor<f32>) -> (tensor<f32>) {
  %0 = onnx.Constant dense<1.0> : tensor<f32>
  %1 = "onnx.Neg"(%0) : (tensor<f32>) -> tensor<f32>
  %2 = "onnx.If"(%arg0) ({
    %3 = "onnx.Neg"(%1) : (tensor<f32>) -> tensor<f32>
    onnx.Yield %3 : tensor<f32>
  }, {
    %3 = "onnx.Sub"(%arg1, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    onnx.Yield %3 : tensor<f32>
  }) : (tensor<i1>) -> (tensor<f32>)
  onnx.Return %2 : tensor<f32>
}
// CHECK-LABEL:  func.func @test_if
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<i1>, [[PARAM_1_:%.+]]: tensor<f32>) -> tensor<f32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Neg"([[VAR_0_]]) {constantfoldable} : (tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.If"([[PARAM_0_]]) ({
// CHECK:             [[VAR_3_:%.+]] = "onnx.Neg"([[VAR_1_]]) {constantfoldable} : (tensor<f32>) -> tensor<f32>
// CHECK:             onnx.Yield [[VAR_3_]] : tensor<f32>
// CHECK:           }, {
// CHECK:             [[VAR_3_1_:%.+]] = "onnx.Neg"([[VAR_1_]]) {constantfoldable} : (tensor<f32>) -> tensor<f32>
// CHECK:             [[VAR_4_:%.+]] = "onnx.Add"([[PARAM_1_]], [[VAR_3_1_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:             onnx.Yield [[VAR_4_]] : tensor<f32>
// CHECK:           }) : (tensor<i1>) -> tensor<f32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<f32>
// CHECK:         }
