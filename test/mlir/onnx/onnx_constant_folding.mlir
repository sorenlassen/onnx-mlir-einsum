// RUN: onnx-mlir-opt --constant-folding-pass %s -split-input-file | FileCheck %s

func.func @test_range_fp() -> tensor<3xf32> {
  %start = onnx.Constant dense<0.2> : tensor<f32>
  %limit = onnx.Constant dense<0.5> : tensor<f32>
  %delta = onnx.Constant dense<0.1> : tensor<f32>
  %1 = "onnx.Range"(%start, %limit, %delta) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<3xf32>
  onnx.Return %1 : tensor<3xf32>
}
// CHECK-LABEL:  func.func @test_range_fp
// CHECK-SAME:   () -> tensor<3xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[2.000000e-01, 3.000000e-01, 4.000000e-01]> : tensor<3xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<3xf32>
// CHECK:         }

// -----

func.func @test_add_scalars() -> tensor<f32> {
  %0 = onnx.Constant dense<1.0> : tensor<f32>
  %1 = onnx.Constant dense<2.0> : tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  onnx.Return %2 : tensor<f32>
}
// CHECK-LABEL:  func.func @test_add_scalars
// CHECK-SAME:   () -> tensor<f32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<f32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<f32>
// CHECK:         }

// -----

func.func @test_add_add() -> tensor<5xf32> {
  %0 = onnx.Constant dense<0.1> : tensor<5xf32>
  %1 = onnx.Constant dense<1.0> : tensor<f32>
  %2 = onnx.Constant dense<2.0> : tensor<f32>
  %3 = "onnx.Add"(%0, %1) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  %4 = "onnx.Add"(%3, %2) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  onnx.Return %4 : tensor<5xf32>
}
// CHECK-LABEL:  func.func @test_add_add
// CHECK-SAME:   () -> tensor<5xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<3.100000e+00> : tensor<5xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<5xf32>
// CHECK:         }

// -----

func.func @test_add_arg(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %0 = onnx.Constant dense<0.1> : tensor<5xf32>
  %1 = onnx.Constant dense<1.0> : tensor<f32>
  %2 = onnx.Constant dense<2.0> : tensor<f32>
  %3 = "onnx.Add"(%0, %1) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  %4 = "onnx.Add"(%3, %2) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  %5 = "onnx.Add"(%4, %arg0) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  onnx.Return %5 : tensor<5xf32>
}
// CHECK-LABEL:  func.func @test_add_arg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5xf32>) -> tensor<5xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<3.100000e+00> : tensor<5xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[PARAM_0_]]) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<5xf32>
// CHECK:         }

// -----

func.func @test_add_sum_arg(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %0 = onnx.Constant dense<0.1> : tensor<5xf32>
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
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.100000e+00> : tensor<5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<5.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sum"([[VAR_0_]], [[VAR_1_]], [[PARAM_0_]]) : (tensor<5xf32>, tensor<f32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<5xf32>
// CHECK:         }

// -----

func.func @test_if(%arg0: tensor<i1>, %arg1: tensor<f32>) -> (tensor<f32>) {
  %0 = onnx.Constant dense<1.0> : tensor<f32>
  %1 = "onnx.Neg"(%0) : (tensor<f32>) -> tensor<f32>
  %2 = "onnx.If"(%arg0) ({
    %3 = "onnx.Neg"(%1) : (tensor<f32>) -> tensor<f32>
    onnx.Yield %3 : tensor<f32>
  }, {
    %3 = "onnx.Sub"(%arg1, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    onnx.Yield %3 : tensor<f32>
  }) : (tensor<i1>) -> (tensor<f32>)
  onnx.Return %2 : tensor<f32>
}
// CHECK-LABEL:  func.func @test_if
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<i1>, [[PARAM_1_:%.+]]: tensor<f32>) -> tensor<f32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.If"([[PARAM_0_]]) ({
// CHECK-DAG:         [[VAR_2_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:             onnx.Yield [[VAR_2_]] : tensor<f32>
// CHECK:           }, {
// CHECK:             [[VAR_2_1_:%.+]] = "onnx.Sub"([[PARAM_1_]], [[VAR_0_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:             onnx.Yield [[VAR_2_1_]] : tensor<f32>
// CHECK:           }) : (tensor<i1>) -> tensor<f32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<f32>
// CHECK:         }

// -----

func.func @test_if_2(%arg0: tensor<i1>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0 = onnx.Constant dense<1.0> : tensor<f32>
  %1 = "onnx.Neg"(%0) : (tensor<f32>) -> tensor<f32>
  %2 = "onnx.If"(%arg0) ({
    %3 = "onnx.Neg"(%1) : (tensor<f32>) -> tensor<f32>
    onnx.Yield %3 : tensor<f32>
  }, {
    %3 = "onnx.Sub"(%arg1, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    onnx.Yield %3 : tensor<f32>
  }) : (tensor<i1>) -> (tensor<f32>)
  onnx.Return %1, %2 : tensor<f32>, tensor<f32>
}
// CHECK-LABEL:  func.func @test_if_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<i1>, [[PARAM_1_:%.+]]: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.If"([[PARAM_0_]]) ({
// CHECK-DAG:         [[VAR_3_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:             onnx.Yield [[VAR_3_]] : tensor<f32>
// CHECK:           }, {
// CHECK:             [[VAR_3_1_:%.+]] = "onnx.Sub"([[PARAM_1_]], [[VAR_0_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:             onnx.Yield [[VAR_3_1_]] : tensor<f32>
// CHECK:           }) : (tensor<i1>) -> tensor<f32>
// CHECK:           onnx.Return [[VAR_1_]], [[VAR_2_]] : tensor<f32>, tensor<f32>
// CHECK:         }
