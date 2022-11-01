// RUN: onnx-mlir-opt --shape-inference --constprop-onnx %s -split-input-file | FileCheck %s

// CHECK-LABEL: @test_sqrt() -> tensor<1x2xf32>
func.func @test_sqrt() -> tensor<1x2xf32> {
  %0 = "onnx.Constant"() {value = dense<[[4.0, 16.0]]> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[2.000000e+00, 4.000000e+00]{{\]}}> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Sqrt"{{.*}}
}

// -----

// CHECK-LABEL: @test_relu() -> tensor<1x2xf32>
func.func @test_relu() -> tensor<1x2xf32> {
  %0 = "onnx.Constant"() {value = dense<[[-4.0, 16.0]]> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  "func.return"(%1) : (tensor<1x2xf32>) -> ()
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 1.600000e+01]{{\]}}> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Relu"{{.*}}
}
