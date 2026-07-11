#ifndef TRELLIS2_C_PROJECTED_DIT_FLOW_H
#define TRELLIS2_C_PROJECTED_DIT_FLOW_H

#include "trellis.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trellis_dit_flow_projection_block_weights {
    struct ggml_tensor * proj_w;
    struct ggml_tensor * proj_b;
} trellis_dit_flow_projection_block_weights;

typedef struct trellis_dit_flow_projection_sidecar {
    int enabled;
    int proj_channels;
    int n_blocks;
    trellis_dit_flow_projection_block_weights blocks[TRELLIS_DIT_FLOW_BLOCKS];
} trellis_dit_flow_projection_sidecar;

/* Internal composite used by model-aware callers.  The public
 * trellis_dit_flow_weights layout remains unchanged.
 */
typedef struct trellis_dit_flow_model {
    trellis_dit_flow_weights base;
    trellis_dit_flow_projection_sidecar projection;
} trellis_dit_flow_model;

/* Bind either a stock TRELLIS flow checkpoint or a Pixal3D projected-flow
 * checkpoint.  Pixal3D is detected from blocks.0.cross_attn.proj_linear.weight.
 */
trellis_status trellis_dit_flow_model_bind_weights(
    trellis_tensor_store * store,
    int in_channels,
    int out_channels,
    trellis_dit_flow_model * model,
    char * first_issue,
    size_t first_issue_size);

trellis_status trellis_ss_flow_model_bind_weights(
    trellis_tensor_store * store,
    trellis_dit_flow_model * model,
    char * first_issue,
    size_t first_issue_size);

trellis_status trellis_shape_slat_flow_model_bind_weights(
    trellis_tensor_store * store,
    trellis_dit_flow_model * model,
    char * first_issue,
    size_t first_issue_size);

trellis_status trellis_tex_slat_flow_model_bind_weights(
    trellis_tensor_store * store,
    trellis_dit_flow_model * model,
    char * first_issue,
    size_t first_issue_size);

/* Unified model-aware forward.  For a stock TRELLIS model, global_context is
 * the original full image context and projected_context is ignored.  A Pixal3D
 * model requires projected_context with one projected feature row per x token.
 */
struct ggml_tensor * trellis_dit_flow_forward_projected(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * timesteps,
    struct ggml_tensor * global_context,
    struct ggml_tensor * projected_context,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_dit_flow_model * model);

struct ggml_tensor * trellis_dit_flow_forward_projected_with_policy(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * timesteps,
    struct ggml_tensor * global_context,
    struct ggml_tensor * projected_context,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_dit_flow_model * model,
    const trellis_ggml_attention_policy * attention_policy);

#ifdef __cplusplus
}
#endif

#endif
