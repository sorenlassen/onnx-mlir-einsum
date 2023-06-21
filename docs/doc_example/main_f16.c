#include <OnnxMlirRuntime.h>
#include <stdio.h>
#include <string.h>

OMTensorList *run_main_graph(OMTensorList *);

// Adapted from https://stackoverflow.com/a/60047308
static uint32_t as_uint(const float x) {
  uint32_t ret;
  memcpy(&ret, &x, sizeof(x));
  return ret;
}
static float as_float(const uint32_t x) {
  float ret;
  memcpy(&ret, &x, sizeof(x));
  return ret;
}
static float f16_to_f32(uint16_t x) {
  uint32_t e = (x & 0x7C00) >> 10; // exponent
  uint32_t m = (x & 0x03FF) << 13; // mantissa
  // evil log2 bit hack to count leading zeros in denormalized format:
  uint32_t v = as_uint((float)m) >> 23;
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

static OMTensorList *create_input_list() {
  // Shared shape & rank.
  int64_t shape[] = {3, 2};
  int64_t num_elements = shape[0] * shape[1];
  int64_t rank = 2;

  // Construct float arrays filled with 1s or 2s.
  uint16_t *x1Data = (uint16_t *)malloc(sizeof(uint16_t) * num_elements);
  for (int i = 0; i < num_elements; i++)
    x1Data[i] = f32_to_f16(1.0);
  uint16_t *x2Data = (uint16_t *)malloc(sizeof(uint16_t) * num_elements);
  for (int i = 0; i < num_elements; i++)
    x2Data[i] = f32_to_f16(2.0);

  // Use omTensorCreateWithOwnership "true" so float arrays are automatically
  // freed when the Tensors are destroyed.
  OMTensor *x1 = omTensorCreateWithOwnership(x1Data, shape, rank, ONNX_TYPE_FLOAT, true);
  OMTensor *x2 = omTensorCreateWithOwnership(x2Data, shape, rank, ONNX_TYPE_FLOAT, true);

  // Construct a TensorList using the Tensors
  OMTensor *list[2] = {x1, x2};
  return omTensorListCreate(list, 2);
}

int main() {
  // Generate input TensorList
  OMTensorList *input_list = create_input_list();

  // Call the compiled onnx model function.
  OMTensorList *output_list = run_main_graph(input_list);
  if (!output_list) {
    // May inspect errno to get info about the error.
    return 1;
  }

  // Get the first tensor from output list.
  OMTensor *y = omTensorListGetOmtByIndex(output_list, 0);
  omTensorPrint("Result tensor: ", y);
  uint16_t *outputPtr = (uint16_t *) omTensorGetDataPtr(y);

  // Print its content, should be all 3.
  for (int i = 0; i < 6; i++) {
    float f = f16_to_f32(outputPtr[i]);
    if (f != 3.0) {
      fprintf(stderr, "Iteration %d: expected 3.0, got %f.\n", i, f);
      exit(100);
    }
  }
  printf("\n");

  // Destory the list and the tensors inside of it.
  // Use omTensorListDestroyShallow if only want to destroy the list themselves.
  omTensorListDestroy(input_list);
  omTensorListDestroy(output_list);
  return 0;
}
