// RUN: onnx-mlir-opt --lazy-constprop-onnx %s -split-input-file | FileCheck %s

func.func @test_add_scalars() -> tensor<f32> {
  %0 = onnx.Constant dense<1.0> : tensor<f32>
  %1 = onnx.Constant dense<2.0> : tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<f32> , tensor<f32>) -> tensor<f32>
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
