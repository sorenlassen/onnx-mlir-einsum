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
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[2.000000e-01, 3.000000e-01, 4.000000e-01]> : tensor<3xf32>
// CHECK:           onnx.Return [[VAR_3_]] : tensor<3xf32>
// CHECK:         }
