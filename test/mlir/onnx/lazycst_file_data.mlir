// Create arange5xf32.npy with 128 bytes header and 20 bytes payload:
// RUN: python3 -c "import numpy; numpy.save('arange5xf32.npy', numpy.arange(5, dtype=numpy.float32)); import os; assert os.path.getsize('arange5xf32.npy') == 128+20"

// RUN: onnx-mlir-opt --external-data-dir=. %s -split-input-file | FileCheck --check-prefix=DENSE %s
// RUN: onnx-mlir-opt --hideDenseLikeElementsAttrs=false %s -split-input-file | FileCheck --check-prefix=FILEDATA %s

func.func @test_add_arg(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %0 = onnx.Constant #lazycst.file_data<"arange5xf32.npy" + 128> : tensor<5xf32>
  onnx.Return %0 : tensor<5xf32>
}
// DENSE: %0 = onnx.Constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<5xf32>
// FILEDATA: %0 = onnx.Constant #lazycst.file_data<"arange5xf32.npy" + 128> : tensor<5xf32>
