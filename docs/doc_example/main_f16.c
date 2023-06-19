#include <OnnxMlirRuntime.h>
#include <stdio.h>

float f16_to_f32(uint16_t x);
uint16_t f32_to_f16(float x);

OMTensorList *run_main_graph(OMTensorList *);

OMTensorList *create_input_list() {
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
  for (int i = 0; i < 6; i++)
    printf("%f ", f16_to_f32(outputPtr[i]));
  printf("\n");

  // Destory the list and the tensors inside of it.
  // Use omTensorListDestroyShallow if only want to destroy the list themselves.
  omTensorListDestroy(input_list);
  omTensorListDestroy(output_list);
  return 0;
}
