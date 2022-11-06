// RUN: onnx-mlir-opt --shape-inference --constprop-onnx %s -split-input-file | FileCheck %s

// CHECK-LABEL: test_default_transpose_const_1
  func.func @test_default_transpose_const_1() -> tensor<*xi32> {
  %0 = "onnx.Constant"() {value = dense<[[[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]], [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]]> : tensor<2x3x4xi32>} : () -> tensor<2x3x4xi32>
  %1 = "onnx.Transpose"(%0) : (tensor<2x3x4xi32>) -> tensor<*xi32>
  "func.return"(%1) : (tensor<*xi32>) -> ()
  // CHECK: [[RES:%.+]] = "onnx.Constant"() {value = dense<[{{.}}[111, 211], [121, 221], [131, 231]{{.}}, [{{.}}112, 212], [122, 222], [132, 232]{{.}}, [{{.}}113, 213], [123, 223], [133, 233]{{.}}, [{{.}}114, 214], [124, 224], [134, 234]{{.}}]> : tensor<4x3x2xi32>} : () -> tensor<4x3x2xi32>
  // CHECK: return [[RES]] : tensor<4x3x2xi32>
}

// -----  

// CHECK-LABEL: test_default_transpose_const_2
func.func @test_default_transpose_const_2() -> tensor<*xi32> {
  %0 = "onnx.Constant"() {value = dense<[[[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]], [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]]> : tensor<2x3x4xi32>} : () -> tensor<2x3x4xi32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 2, 1]} : (tensor<2x3x4xi32>) -> tensor<*xi32>
  "func.return"(%1) : (tensor<*xi32>) -> ()
  // CHECK: [[RES:%.+]] = "onnx.Constant"() {value = dense<[{{.}}[111, 121, 131], [112, 122, 132], [113, 123, 133], [114, 124, 134]{{.}}, [{{.}}211, 221, 231], [212, 222, 232], [213, 223, 233], [214, 224, 234]{{.}}]> : tensor<2x4x3xi32>} : () -> tensor<2x4x3xi32>
  // CHECK: return [[RES]] : tensor<2x4x3xi32>
}

// -----  

// CHECK-LABEL: test_default_transpose_const_3
func.func @test_default_transpose_const_3() -> tensor<*xi32> {
  %0 = "onnx.Constant"() {value = dense<[[[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]], [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]]> : tensor<2x3x4xi32>} : () -> tensor<2x3x4xi32>
  %1 = "onnx.Transpose"(%0) {perm = [1, 0, 2]} : (tensor<2x3x4xi32>) -> tensor<*xi32>
  "func.return"(%1) : (tensor<*xi32>) -> ()
  // CHECK: [[RES:%.+]] =  "onnx.Constant"() {value = dense<[{{.}}[111, 112, 113, 114], [211, 212, 213, 214]{{.}}, [{{.}}121, 122, 123, 124], [221, 222, 223, 224]{{.}}, [{{.}}131, 132, 133, 134], [231, 232, 233, 234]{{.}}]> : tensor<3x2x4xi32>} : () -> tensor<3x2x4xi32>
  // CHECK: return [[RES]] : tensor<3x2x4xi32>
}
