// RUN: onnx-mlir-opt --lazy-constprop-onnx %s -split-input-file | FileCheck %s

func.func @test_add_scalars() -> tensor<f32> {
  %0 = onnx.Constant dense<1.0> : tensor<f32>
  %1 = onnx.Constant dense<2.0> : tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  onnx.Return %2 : tensor<f32>
// CHECK:         lazycst.func @lazycst.0() -> tensor<f32> attributes {arg_constants = [], res_constants = [#lazycst.lazy_elms<@lazycst.0> : tensor<f32>]} {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_1_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           lazycst.return [[VAR_2_]] : tensor<f32>
// CHECK:         }
// CHECK-LABEL:  func.func @test_add_scalars
// CHECK-SAME:   () -> tensor<f32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant #lazycst.lazy_elms<@lazycst.0> : tensor<f32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<f32>
// CHECK:         }
}

// -----

// TODO: figure out a robust file location, for now you need to place a
//       20 bytes file in /tmp/foo.data for this test to not crash
func.func @test_add_file() -> tensor<5xf32> {
  %0 = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
  %1 = onnx.Constant dense<2.0> : tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  onnx.Return %2 : tensor<5xf32>
}

// -----

func.func @test_add_add() -> tensor<5xf32> {
  %0 = onnx.Constant #lazycst.file_data<"/tmp/foo.data"> : tensor<5xf32>
  %1 = onnx.Constant dense<1.0> : tensor<f32>
  %2 = onnx.Constant dense<2.0> : tensor<f32>
  %3 = "onnx.Add"(%0, %1) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  %4 = "onnx.Add"(%3, %2) : (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
  onnx.Return %4 : tensor<5xf32>
}

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
