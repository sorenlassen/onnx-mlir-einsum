#include <errno.h>
#include <iostream>
#include <cstring>

#include <OnnxMlirCompiler.h>
#include <OnnxMlirRuntime.h>

#include "src/Runtime/ExecutionSession.hpp"

// Adapted from https://stackoverflow.com/a/60047308
static uint32_t as_uint(const float x) {
  uint32_t ret;
  std::memcpy(&ret, &x, sizeof(x));
  return ret;
}
float as_float(const uint32_t x) {
  float ret;
  std::memcpy(&ret, &x, sizeof(x));
  return ret;
}
static float f16_to_f32(uint16_t x) {
  uint32_t e = (x & 0x7C00) >> 10; // exponent
  uint32_t m = (x & 0x03FF) << 13; // mantissa
  // evil log2 bit hack to count leading zeros in denormalized format:
  uint32_t v = as_uint(static_cast<float>(m)) >> 23;
  uint32_t r = // sign : normalized : denormalized
      (x & 0x8000u) << 16 | (e != 0) * ((e + 112) << 23 | m) |
      ((e == 0) & (m != 0)) *
          ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000));
  return as_float(r);
}
static uint16_t f32_to_f16(float x) {
  // round-to-nearest-even: add last bit after truncated mantissa
  uint32_t b = as_uint(x) + 0x00001000;
  uint32_t e = (b & 0x7F800000) >> 23; // exponent
  uint32_t m = b & 0x007FFFFF;         // mantissa
  // in line below: 0x007FF000 = 0x00800000 - 0x00001000
  //                           = decimal indicator flag - initial rounding
  return // sign : normalized : denormalized : saturate
      (b & 0x80000000u) >> 16 |
      (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
      ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
      (e > 143) * 0x7FFF;
}

// Read the arguments from the command line and return a std::string
static std::string readArgs(int argc, char *argv[]) {
  std::string commandLineStr;
  for (int i = 1; i < argc; i++) {
    commandLineStr.append(std::string(argv[i]) + " ");
  }
  return commandLineStr;
}

int main(int argc, char *argv[]) {
  // Read compiler options from command line and compile the doc example into a
  // model library.
  const char *errorMessage = NULL;
  const char *compiledFilename;
  std::string flags = readArgs(argc, argv);
  flags += "-o add-cpp-interface";
  std::cout << "Compile with options \"" << flags << "\"\n";
  int rc = onnx_mlir::omCompileFromFile(
      "add_f16.onnx", flags.c_str(), &compiledFilename, &errorMessage);
  if (rc != onnx_mlir::CompilerSuccess) {
    std::cerr << "Failed to compile add.onnx with error code " << rc;
    if (errorMessage)
      std::cerr << " and message \"" << errorMessage << "\"";
    std::cerr << "." << std::endl;
    return rc;
  }
  std::string libFilename(compiledFilename);
  std::cout << "Compiled succeeded with results in file: " << libFilename
            << std::endl;

  // Prepare the execution session.
  onnx_mlir::ExecutionSession *session;
  try {
    session = new onnx_mlir::ExecutionSession("./" + libFilename);
  } catch (const std::runtime_error &error) {
    std::cerr << "error while creating execution session: " << error.what()
              << " and errno " << errno << std::endl;
    return errno;
  }

  // Get input signature and print it.
  std::string inputSignature;
  try {
    inputSignature = session->inputSignature();
  } catch (const std::runtime_error &error) {
    std::cerr << "error while loading input signature: " << error.what()
              << " and errno " << errno << std::endl;
    return errno;
  }
  std::cout << "Compiled add.onnx model has input signature: \""
            << inputSignature << "\"." << std::endl;

  // Build the inputs, starts with shared shape & rank.
  int64_t shape[] = {3, 2};
  int64_t rank = 2;
  // Construct x1 omt filled with 1.
  const uint16_t one = f32_to_f16(1.);
  const uint16_t two = f32_to_f16(2.);
  uint16_t x1Data[] = {one, one, one, one, one, one};
  OMTensor *x1 = omTensorCreate(x1Data, shape, rank, ONNX_TYPE_FLOAT16);
  // Construct x2 omt filled with 2.
  uint16_t x2Data[] = {two, two, two, two, two, two};
  OMTensor *x2 = omTensorCreate(x2Data, shape, rank, ONNX_TYPE_FLOAT16);
  // Construct a list of omts as input.
  OMTensor *list[2] = {x1, x2};
  OMTensorList *input = omTensorListCreate(list, 2);

  // Call the compiled onnx model function.
  std::cout << "Start running model " << std::endl;
  OMTensorList *outputList;
  try {
    outputList = session->run(input);
  } catch (const std::runtime_error &error) {
    std::cerr << "error while running model: " << error.what() << " and errno "
              << errno << std::endl;
    return errno;
  }
  std::cout << "Finished running model " << std::endl;

  // Get the first omt as output.
  OMTensor *y = omTensorListGetOmtByIndex(outputList, 0);
  omTensorPrint("Result tensor: ", y);
  std::cout << std::endl;
  uint16_t *outputPtr = (uint16_t *)omTensorGetDataPtr(y);
  // Print its content, should be all 3.
  for (int i = 0; i < 6; i++) {
    if (f16_to_f32(outputPtr[i]) != 3.0) {
      std::cerr << "Iteration " << i << ": expected 3.0, got " << f16_to_f32(outputPtr[i])
                << "." << std::endl;
      return 100;
    }
  }
  std::cout << "Model verified successfully" << std::endl;
  delete session;
  return 0;
}
