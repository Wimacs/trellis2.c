#ifndef TRELLIS2_C_DIT_FLOW_EXECUTOR_H
#define TRELLIS2_C_DIT_FLOW_EXECUTOR_H

#include "trellis.h"
#include "projected_dit_flow.h"

typedef struct trellis_dit_flow_executor {
    const trellis_backend_context * backend;
    const trellis_dit_flow_weights * weights;
    const trellis_dit_flow_model * model;
    struct ggml_context * ctx;
    struct ggml_cgraph * graph;
    ggml_gallocr_t alloc;
    struct ggml_tensor * x;
    struct ggml_tensor * t;
    struct ggml_tensor * c;
    struct ggml_tensor * p;
    struct ggml_tensor * cos_phase;
    struct ggml_tensor * sin_phase;
    struct ggml_tensor * y;
    float * x_host;
    float * context_host;
    float * projected_host;
    float * y_host;
    int64_t tokens;
    int cond_tokens;
    int batch;
    size_t single_input_count;
    size_t single_output_count;
    size_t single_context_count;
    size_t single_projected_count;
    size_t input_count;
    size_t output_count;
    size_t context_count;
    size_t projected_count;
    trellis_ggml_attention_policy attention_policy;
} trellis_dit_flow_executor;

trellis_status trellis_dit_flow_executor_init_single(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_weights * weights,
    int64_t tokens,
    int cond_tokens,
    const float * context,
    const float * cos_phase,
    const float * sin_phase);

trellis_status trellis_dit_flow_executor_init_single_with_policy(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_weights * weights,
    int64_t tokens,
    int cond_tokens,
    const float * context,
    const float * cos_phase,
    const float * sin_phase,
    const trellis_ggml_attention_policy * attention_policy);

trellis_status trellis_dit_flow_executor_run_single(
    trellis_dit_flow_executor * executor,
    const float * latent,
    float timestep,
    float * pred);

/* Replaces the conditioning inputs of an initialized batch=1 executor while
 * retaining its graph allocation. projected_context is required only for a
 * projected DiT model. */
trellis_status trellis_dit_flow_executor_set_single_conditioning(
    trellis_dit_flow_executor * executor,
    const float * context,
    const float * projected_context);

trellis_status trellis_dit_flow_executor_init_single_projected(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_model * model,
    int64_t tokens,
    int global_tokens,
    const float * global_context,
    const float * projected_context,
    const float * cos_phase,
    const float * sin_phase);

trellis_status trellis_dit_flow_executor_init_single_projected_with_policy(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_model * model,
    int64_t tokens,
    int global_tokens,
    const float * global_context,
    const float * projected_context,
    const float * cos_phase,
    const float * sin_phase,
    const trellis_ggml_attention_policy * attention_policy);

trellis_status trellis_dit_flow_executor_init_cfg_batch(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_weights * weights,
    int64_t tokens,
    int cond_tokens,
    const float * context,
    const float * neg_context,
    const float * cos_phase,
    const float * sin_phase);

trellis_status trellis_dit_flow_executor_init_cfg_batch_with_policy(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_weights * weights,
    int64_t tokens,
    int cond_tokens,
    const float * context,
    const float * neg_context,
    const float * cos_phase,
    const float * sin_phase,
    const trellis_ggml_attention_policy * attention_policy);

trellis_status trellis_dit_flow_executor_run_cfg_batch(
    trellis_dit_flow_executor * executor,
    const float * latent,
    float timestep,
    float * pred_pos,
    float * pred_neg);

trellis_status trellis_dit_flow_executor_init_cfg_batch_projected(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_model * model,
    int64_t tokens,
    int global_tokens,
    const float * global_context,
    const float * neg_global_context,
    const float * projected_context,
    const float * neg_projected_context,
    const float * cos_phase,
    const float * sin_phase);

trellis_status trellis_dit_flow_executor_init_cfg_batch_projected_with_policy(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_model * model,
    int64_t tokens,
    int global_tokens,
    const float * global_context,
    const float * neg_global_context,
    const float * projected_context,
    const float * neg_projected_context,
    const float * cos_phase,
    const float * sin_phase,
    const trellis_ggml_attention_policy * attention_policy);

void trellis_dit_flow_executor_free(trellis_dit_flow_executor * executor);

#endif
