#ifndef TRELLIS2_C_SRC_PIPELINE_TRELLIS_DIT_FLOW_EXECUTOR_H
#define TRELLIS2_C_SRC_PIPELINE_TRELLIS_DIT_FLOW_EXECUTOR_H

#include "trellis.h"

typedef struct trellis_dit_flow_executor {
    const trellis_cuda_context * cuda;
    const trellis_dit_flow_weights * weights;
    struct ggml_context * ctx;
    struct ggml_cgraph * graph;
    ggml_gallocr_t alloc;
    struct ggml_tensor * x;
    struct ggml_tensor * t;
    struct ggml_tensor * c;
    struct ggml_tensor * cos_phase;
    struct ggml_tensor * sin_phase;
    struct ggml_tensor * y;
    float * x_host;
    float * context_host;
    float * y_host;
    int64_t tokens;
    int cond_tokens;
    size_t input_count;
    size_t output_count;
    size_t context_count;
} trellis_dit_flow_executor;

trellis_status trellis_dit_flow_executor_init_single(
    trellis_dit_flow_executor * executor,
    const trellis_cuda_context * cuda,
    const trellis_dit_flow_weights * weights,
    int64_t tokens,
    int cond_tokens,
    const float * context,
    const float * cos_phase,
    const float * sin_phase);

trellis_status trellis_dit_flow_executor_run_single(
    trellis_dit_flow_executor * executor,
    const float * latent,
    float timestep,
    float * pred);

void trellis_dit_flow_executor_free(trellis_dit_flow_executor * executor);

#endif
