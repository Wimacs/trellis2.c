#ifndef TRELLIS2_C_GGML_LAYERS_H
#define TRELLIS2_C_GGML_LAYERS_H

#include "trellis.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Dense affine projection: applies weight * x plus an optional bias in ggml layout. */
struct ggml_tensor * trellis_ggml_linear(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * weight,
    struct ggml_tensor * bias);

/* Channel-wise layer normalization with optional affine scale and bias. */
struct ggml_tensor * trellis_ggml_layer_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    struct ggml_tensor * beta,
    float eps);

/* RMS normalization with an optional learned scale. */
struct ggml_tensor * trellis_ggml_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    float eps);

/* BF16 emulation helper used to compare C/ggml math with BF16 checkpoint behavior. */
struct ggml_tensor * trellis_ggml_bf16_roundtrip(
    struct ggml_context * ctx,
    struct ggml_tensor * x);

/* Selects flash-attention or explicit attention for trellis_ggml_sdpa. */
void trellis_ggml_set_flash_attn_enabled(int enabled);
int trellis_ggml_flash_attn_enabled(void);

/* Instance-scoped attention selection.  New model/executor code should pass a
 * policy explicitly instead of depending on the legacy process-wide switch.
 */
typedef enum trellis_ggml_attention_mode {
    TRELLIS_GGML_ATTENTION_MODE_INVALID = 0,
    TRELLIS_GGML_ATTENTION_MODE_EXPLICIT = 1,
    /* Legacy Flash Attention mode. K/V produced as F32 are narrowed to F16. */
    TRELLIS_GGML_ATTENTION_MODE_FLASH = 2,
    TRELLIS_GGML_ATTENTION_MODE_FLASH_F16 = TRELLIS_GGML_ATTENTION_MODE_FLASH,
    /* K/V produced as F32 are narrowed to BF16 before the backend Flash op. */
    TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16 = 3,
} trellis_ggml_attention_mode;

typedef struct trellis_ggml_attention_policy {
    size_t struct_size;
    trellis_ggml_attention_mode mode;
} trellis_ggml_attention_policy;

#define TRELLIS_GGML_ATTENTION_POLICY_INIT \
    { sizeof(trellis_ggml_attention_policy), TRELLIS_GGML_ATTENTION_MODE_EXPLICIT }

int trellis_ggml_attention_policy_is_valid(
    const trellis_ggml_attention_policy * policy);

/* Per-head RMS normalization for attention query/key tensors. */
struct ggml_tensor * trellis_ggml_multihead_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    float eps);

/* Transformer feed-forward MLP: linear, GELU, then linear. */
struct ggml_tensor * trellis_ggml_feed_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w1,
    struct ggml_tensor * b1,
    struct ggml_tensor * w2,
    struct ggml_tensor * b2);

/* Timestep embedding MLP used by flow models to condition denoising steps. */
struct ggml_tensor * trellis_ggml_timestep_mlp(
    struct ggml_context * ctx,
    struct ggml_tensor * timesteps,
    int frequency_dim,
    struct ggml_tensor * w1,
    struct ggml_tensor * b1,
    struct ggml_tensor * w2,
    struct ggml_tensor * b2);

/* Scaled dot-product attention over query, key, and value tensors. */
struct ggml_tensor * trellis_ggml_sdpa(
    struct ggml_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    float scale);

/* Policy-aware SDPA.  A NULL policy keeps the legacy global-switch behavior;
 * callers that need instance isolation must pass a valid explicit policy.
 */
struct ggml_tensor * trellis_ggml_sdpa_with_policy(
    struct ggml_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    float scale,
    const trellis_ggml_attention_policy * policy);

/* Applies adjacent-pair rotary position embedding to attention heads. */
struct ggml_tensor * trellis_ggml_apply_rope_adjacent(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase);

/* Multi-head self-attention without rotary position embedding. */
struct ggml_tensor * trellis_ggml_self_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int n_heads,
    struct ggml_tensor * qkv_w,
    struct ggml_tensor * qkv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b);

/* Multi-head self-attention with rotary position embedding. */
struct ggml_tensor * trellis_ggml_self_attention_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int n_heads,
    struct ggml_tensor * qkv_w,
    struct ggml_tensor * qkv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase);

/* Cross-attention that lets latent tokens attend to conditioning tokens. */
struct ggml_tensor * trellis_ggml_cross_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * context,
    int n_heads,
    struct ggml_tensor * q_w,
    struct ggml_tensor * q_b,
    struct ggml_tensor * kv_w,
    struct ggml_tensor * kv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b);

/* Pixal3D ProjectAttention: global cross-attention plus a token-aligned projection. */
struct ggml_tensor * trellis_ggml_project_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * global_context,
    struct ggml_tensor * projected_context,
    int n_heads,
    struct ggml_tensor * q_w,
    struct ggml_tensor * q_b,
    struct ggml_tensor * kv_w,
    struct ggml_tensor * kv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b,
    struct ggml_tensor * proj_w,
    struct ggml_tensor * proj_b);

typedef struct trellis_ggml_modulated_cross_block_params {
    struct ggml_tensor * block_modulation; /* [6 * channels] */
    struct ggml_tensor * norm2_gamma;      /* [channels] */
    struct ggml_tensor * norm2_beta;       /* [channels] */

    struct ggml_tensor * self_qkv_w;
    struct ggml_tensor * self_qkv_b;
    struct ggml_tensor * self_q_rms_gamma;
    struct ggml_tensor * self_k_rms_gamma;
    struct ggml_tensor * self_out_w;
    struct ggml_tensor * self_out_b;

    struct ggml_tensor * cross_q_w;
    struct ggml_tensor * cross_q_b;
    struct ggml_tensor * cross_kv_w;
    struct ggml_tensor * cross_kv_b;
    struct ggml_tensor * cross_q_rms_gamma;
    struct ggml_tensor * cross_k_rms_gamma;
    struct ggml_tensor * cross_out_w;
    struct ggml_tensor * cross_out_b;

    struct ggml_tensor * mlp_fc1_w;
    struct ggml_tensor * mlp_fc1_b;
    struct ggml_tensor * mlp_fc2_w;
    struct ggml_tensor * mlp_fc2_b;
    int debug_parts;
    int emulate_bf16;
} trellis_ggml_modulated_cross_block_params;

/* DiT block combining AdaLN modulation, self-attention, cross-attention, and MLP. */
struct ggml_tensor * trellis_ggml_modulated_cross_block(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params);

/* RoPE-enabled DiT block combining modulation, attention, and MLP. */
struct ggml_tensor * trellis_ggml_modulated_cross_block_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase);

struct ggml_tensor * trellis_ggml_modulated_cross_block_rope_with_policy(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_ggml_attention_policy * attention_policy);

/* ProjectAttention variant of the modulated DiT block. */
struct ggml_tensor * trellis_ggml_modulated_cross_block_projected(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * global_context,
    struct ggml_tensor * projected_context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * proj_w,
    struct ggml_tensor * proj_b);

/* RoPE-enabled ProjectAttention variant of the modulated DiT block. */
struct ggml_tensor * trellis_ggml_modulated_cross_block_projected_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * global_context,
    struct ggml_tensor * projected_context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * proj_w,
    struct ggml_tensor * proj_b,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase);

struct ggml_tensor * trellis_ggml_modulated_cross_block_projected_rope_with_policy(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * global_context,
    struct ggml_tensor * projected_context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * proj_w,
    struct ggml_tensor * proj_b,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_ggml_attention_policy * attention_policy);

#ifdef __cplusplus
}
#endif

#endif
