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

#ifdef __cplusplus
}
#endif

#endif
