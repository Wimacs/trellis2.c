#ifndef TRELLIS2_C_TOOLS_TRELLIS_TOOL_MODEL_H
#define TRELLIS2_C_TOOLS_TRELLIS_TOOL_MODEL_H

#include "trellis.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trellis_tool_model_load_result {
    size_t tensors;
    uint64_t bytes;
    double seconds;
} trellis_tool_model_load_result;

int trellis_tool_load_tensor_store_f32(
    const trellis_cuda_context * cuda,
    const char * label,
    const char * path,
    bool transpose_linear_weights,
    size_t tensor_slack,
    trellis_tensor_store * store,
    trellis_tool_model_load_result * result);

#ifdef __cplusplus
}
#endif

#endif
