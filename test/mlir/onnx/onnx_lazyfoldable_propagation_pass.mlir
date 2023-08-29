// RUN: onnx-mlir-opt --lazyfoldable-propagation-pass --lazyfoldable-analysis-pass --hideDensifiableElementsAttrs=false %s -split-input-file | FileCheck %s

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
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_1_]]) {lazyfoldable} : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_3_]], [[VAR_2_]]) {lazyfoldable} : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
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
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_1_]]) {lazyfoldable} : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_2_]], [[VAR_3_]]) {lazyfoldable} : (tensor<f32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_4_]]) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK:           onnx.Return [[VAR_5_]] : tensor<5xf32>
// CHECK:         }
