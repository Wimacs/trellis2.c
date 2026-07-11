#ifndef TRELLIS2_C_TOKENSKIN_ARCHITECTURE_H
#define TRELLIS2_C_TOKENSKIN_ARCHITECTURE_H

#include "trellis.h"

#define TOKENSKIN_MESH_WIDTH 512
#define TOKENSKIN_MESH_HEADS 8
#define TOKENSKIN_MESH_LAYERS 8
#define TOKENSKIN_MESH_TOKENS 512
#define TOKENSKIN_MESH_FEATURES 54

#define TOKENSKIN_VAE_WIDTH 768
#define TOKENSKIN_VAE_HEADS 12
#define TOKENSKIN_VAE_COND_LAYERS 2
#define TOKENSKIN_VAE_DECODER_LAYERS 10
#define TOKENSKIN_VAE_COND_TOKENS 384
#define TOKENSKIN_VAE_LATENT_CHANNELS 512

#define TOKENSKIN_QWEN_LAYERS 28
#define TOKENSKIN_QWEN_HIDDEN 896
#define TOKENSKIN_QWEN_HEADS 16
#define TOKENSKIN_QWEN_KV_HEADS 8
#define TOKENSKIN_QWEN_HEAD_DIM 128
#define TOKENSKIN_QWEN_INTERMEDIATE 3072
#define TOKENSKIN_QWEN_VOCAB 33036
#define TOKENSKIN_QWEN_MAX_POSITIONS 3192

typedef struct tokenskin_linear_weights {
    struct ggml_tensor * weight;
    struct ggml_tensor * bias;
} tokenskin_linear_weights;

typedef struct tokenskin_norm_weights {
    struct ggml_tensor * weight;
    struct ggml_tensor * bias;
} tokenskin_norm_weights;

typedef struct tokenskin_mesh_cross_weights {
    tokenskin_linear_weights q;
    tokenskin_linear_weights kv;
    tokenskin_linear_weights out;
    tokenskin_norm_weights norm_q;
    tokenskin_norm_weights norm_context;
    tokenskin_norm_weights norm_ff;
    tokenskin_linear_weights ff_in;
    tokenskin_linear_weights ff_out;
} tokenskin_mesh_cross_weights;

typedef struct tokenskin_mesh_self_weights {
    tokenskin_linear_weights qkv;
    tokenskin_linear_weights out;
    tokenskin_norm_weights norm_attn;
    tokenskin_norm_weights norm_ff;
    tokenskin_linear_weights ff_in;
    tokenskin_linear_weights ff_out;
} tokenskin_mesh_self_weights;

typedef struct tokenskin_mesh_encoder_weights {
    tokenskin_linear_weights input;
    tokenskin_mesh_cross_weights cross;
    tokenskin_mesh_self_weights blocks[TOKENSKIN_MESH_LAYERS];
    tokenskin_norm_weights output_norm;
    tokenskin_linear_weights output_projection;
    struct ggml_tensor * output_rms_weight;
} tokenskin_mesh_encoder_weights;

typedef struct tokenskin_tripo_block_weights {
    int cross_attention;
    tokenskin_norm_weights norm_attention;
    tokenskin_norm_weights norm_context;
    tokenskin_linear_weights q;
    tokenskin_linear_weights k;
    tokenskin_linear_weights v;
    tokenskin_linear_weights out;
    tokenskin_norm_weights norm_ff;
    tokenskin_linear_weights ff_in;
    tokenskin_linear_weights ff_out;
} tokenskin_tripo_block_weights;

typedef struct tokenskin_fsq_cvae_weights {
    tokenskin_linear_weights cond_input;
    tokenskin_tripo_block_weights cond_cross;
    tokenskin_tripo_block_weights cond_self[TOKENSKIN_VAE_COND_LAYERS];
    tokenskin_norm_weights cond_output_norm;
    tokenskin_linear_weights cond_quant;

    tokenskin_linear_weights fsq_project_out;
    tokenskin_linear_weights post_quant;
    tokenskin_tripo_block_weights decoder_self[TOKENSKIN_VAE_DECODER_LAYERS];
    tokenskin_tripo_block_weights decoder_cross;
    tokenskin_linear_weights decoder_query;
    tokenskin_norm_weights decoder_output_norm;
    tokenskin_linear_weights decoder_output;
} tokenskin_fsq_cvae_weights;

typedef struct tokenskin_qwen_block_weights {
    struct ggml_tensor * input_norm;
    struct ggml_tensor * q;
    struct ggml_tensor * k;
    struct ggml_tensor * v;
    struct ggml_tensor * out;
    struct ggml_tensor * q_norm;
    struct ggml_tensor * k_norm;
    struct ggml_tensor * post_attention_norm;
    struct ggml_tensor * gate;
    struct ggml_tensor * up;
    struct ggml_tensor * down;
} tokenskin_qwen_block_weights;

typedef struct tokenskin_qwen_weights {
    struct ggml_tensor * token_embedding;
    tokenskin_qwen_block_weights blocks[TOKENSKIN_QWEN_LAYERS];
    struct ggml_tensor * output_norm;
    struct ggml_tensor * lm_head; /* tied to token_embedding */
} tokenskin_qwen_weights;

typedef struct tokenskin_qwen_executor tokenskin_qwen_executor;

trellis_status tokenskin_bind_mesh_encoder_weights(
    trellis_tensor_store * store,
    tokenskin_mesh_encoder_weights * weights,
    char * issue,
    size_t issue_size);

trellis_status tokenskin_bind_qwen_weights(
    trellis_tensor_store * store,
    tokenskin_qwen_weights * weights,
    char * issue,
    size_t issue_size);

trellis_status tokenskin_bind_fsq_cvae_weights(
    trellis_tensor_store * store,
    tokenskin_fsq_cvae_weights * weights,
    char * issue,
    size_t issue_size);

trellis_status tokenskin_qwen_executor_create(
    const trellis_backend_context * backend,
    const tokenskin_qwen_weights * weights,
    const trellis_ggml_attention_policy * attention_policy,
    int capacity,
    tokenskin_qwen_executor ** executor_out);

void tokenskin_qwen_executor_free(tokenskin_qwen_executor * executor);
void tokenskin_qwen_executor_reset(tokenskin_qwen_executor * executor);
int tokenskin_qwen_executor_length(const tokenskin_qwen_executor * executor);

trellis_status tokenskin_qwen_executor_copy_state(
    tokenskin_qwen_executor * destination,
    const tokenskin_qwen_executor * source);

trellis_status tokenskin_qwen_prefill(
    tokenskin_qwen_executor * executor,
    const float * embeddings_token_major_896,
    int n_tokens,
    float * logits_33036);

trellis_status tokenskin_qwen_decode(
    tokenskin_qwen_executor * executor,
    int token_id,
    float * logits_33036);

trellis_status tokenskin_mesh_encoder_compute(
    const trellis_backend_context * backend,
    const tokenskin_mesh_encoder_weights * weights,
    const trellis_ggml_attention_policy * attention_policy,
    const float * full_features,
    int64_t full_tokens,
    const float * query_features,
    float * output_512x896);

trellis_status tokenskin_fsq_condition_compute(
    const trellis_backend_context * backend,
    const tokenskin_fsq_cvae_weights * weights,
    const trellis_ggml_attention_policy * attention_policy,
    const float * full_features,
    int64_t full_tokens,
    const float * query_features,
    float * output_384x512);

/* Decodes one bone's four FSQ codes against the shared 384-token condition
 * latent.  Feature arrays are token-major on the host: [token][channel]. */
trellis_status tokenskin_fsq_skin_decode_compute(
    const trellis_backend_context * backend,
    const tokenskin_fsq_cvae_weights * weights,
    const trellis_ggml_attention_policy * attention_policy,
    const float * fsq_codes_4x5,
    const float * condition_384x512,
    const float * query_features,
    int64_t query_tokens,
    float * skin_probability);

/* Stage-level parity hook: returns the 388x768 decoder KV latent after the
 * requested number of self-attention blocks (0..10). */
trellis_status tokenskin_fsq_decoder_latent_compute(
    const trellis_backend_context * backend,
    const tokenskin_fsq_cvae_weights * weights,
    const trellis_ggml_attention_policy * attention_policy,
    const float * fsq_codes_4x5,
    const float * condition_384x512,
    int self_layers,
    float * latent_388x768);

struct ggml_tensor * tokenskin_attention_standard_q_interleaved_kv(
    struct ggml_context * ctx,
    struct ggml_tensor * query,
    struct ggml_tensor * context,
    int heads,
    const tokenskin_linear_weights * q_weights,
    const tokenskin_linear_weights * kv_weights,
    const tokenskin_linear_weights * out_weights,
    const trellis_ggml_attention_policy * attention_policy);

struct ggml_tensor * tokenskin_attention_interleaved_qkv(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int heads,
    const tokenskin_linear_weights * qkv_weights,
    const tokenskin_linear_weights * out_weights,
    const trellis_ggml_attention_policy * attention_policy);

struct ggml_tensor * tokenskin_tripo_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * query,
    struct ggml_tensor * context,
    int heads,
    const tokenskin_tripo_block_weights * weights,
    const trellis_ggml_attention_policy * attention_policy);

#endif
