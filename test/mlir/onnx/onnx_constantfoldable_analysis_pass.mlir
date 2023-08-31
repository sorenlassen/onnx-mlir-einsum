// RUN: onnx-mlir-opt --constantfoldable-analysis-pass --hideDensifiableElementsAttrs=false %s -split-input-file | FileCheck %s

func.func @test_add_scalars() -> tensor<f32> {
  %0 = onnx.Constant dense<1.0> : tensor<f32>
  %1 = onnx.Constant dense<2.0> : tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  onnx.Return %2 : tensor<f32>
}
// CHECK-LABEL:  func.func @test_add_scalars
// CHECK-SAME:   () -> tensor<f32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_1_]]) {constantfoldable} : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<f32>
// CHECK:         }

// -----

func.func @test_add_file() -> tensor<5xf32> {
  %0 = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
  %1 = onnx.Constant dense<2.0> : tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  onnx.Return %2 : tensor<5xf32>
}
// CHECK-LABEL:  func.func @test_add_file
// CHECK-SAME:   () -> tensor<5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_1_]]) {constantfoldable} : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<5xf32>
// CHECK:         }

// -----

func.func @test_add_add() -> tensor<5xf32> {
  %0 = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
  %1 = onnx.Constant dense<1.0> : tensor<f32>
  %2 = onnx.Constant dense<2.0> : tensor<f32>
  %3 = "onnx.Add"(%0, %1) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  %4 = "onnx.Add"(%3, %2) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  onnx.Return %4 : tensor<5xf32>
}
// CHECK-LABEL:  func.func @test_add_add
// CHECK-SAME:   () -> tensor<5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_1_]]) {constantfoldable} : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_3_]], [[VAR_2_]]) {constantfoldable} : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<5xf32>
// CHECK:         }

// -----

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
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_3_]], [[PARAM_0_]]) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Add"([[VAR_4_]], [[VAR_2_]]) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK:           onnx.Return [[VAR_5_]] : tensor<5xf32>
// CHECK:         }

// -----

func.func @test_add_sum_arg(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %0 = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
  %1 = onnx.Constant dense<1.0> : tensor<f32>
  %2 = onnx.Constant dense<2.0> : tensor<f32>
  %3 = onnx.Constant dense<3.0> : tensor<f32>
  %4 = "onnx.Add"(%0, %1) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  %5 = "onnx.Add"(%2, %3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %6 = "onnx.Sum"(%4, %5, %arg0) : (tensor<5xf32>, tensor<f32>, tensor<5xf32>) -> tensor<5xf32>
  onnx.Return %6 : tensor<5xf32>
}
// CHECK-LABEL:  func.func @test_add_sum_arg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5xf32>) -> tensor<5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_1_]]) {constantfoldable} : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Add"([[VAR_2_]], [[VAR_3_]]) {constantfoldable} : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sum"([[VAR_4_]], [[VAR_5_]], [[PARAM_0_]]) : (tensor<5xf32>, tensor<f32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK:           onnx.Return [[VAR_6_]] : tensor<5xf32>
// CHECK:         }
