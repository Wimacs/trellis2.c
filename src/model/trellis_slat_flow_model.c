#include "trellis.h"

/* Shared DiT flow model used by sparse_structure_flow and structured_latent_flow
 * checkpoints. The bind wrappers below select the input/output channel contract.
 */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

enum {
    TRELLIS_DIT_C = 1536,
    TRELLIS_DIT_COND_C = 1024,
    TRELLIS_DIT_T_FREQ = 256,
    TRELLIS_DIT_HEADS = 12,
    TRELLIS_DIT_HEAD_DIM = 128,
    TRELLIS_DIT_MLP = 8192,
    TRELLIS_DIT_MOD = 9216,
};

static void set_issue(char * dst, size_t dst_size, const char * fmt, ...) {
    if (dst == NULL || dst_size == 0 || dst[0] != '\0') {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(dst, dst_size, fmt, args);
    va_end(args);
}

static trellis_status bind_tensor(
    trellis_tensor_store * store,
    const char * name,
    int n_dims,
    const int64_t * ne,
    struct ggml_tensor ** out,
    char * first_issue,
    size_t first_issue_size) {
    if (store == NULL || name == NULL || ne == NULL || out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    struct ggml_tensor * t = trellis_tensor_store_get(store, name);
    if (t == NULL) {
        set_issue(first_issue, first_issue_size, "missing tensor: %s", name);
        return TRELLIS_STATUS_NOT_FOUND;
    }
    for (int i = 0; i < n_dims; ++i) {
        if (t->ne[i] != ne[i]) {
            set_issue(
                first_issue,
                first_issue_size,
                "shape mismatch: %s dim%d got %lld expected %lld",
                name,
                i,
                (long long) t->ne[i],
                (long long) ne[i]);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    for (int i = n_dims; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] != 1) {
            set_issue(
                first_issue,
                first_issue_size,
                "rank mismatch: %s dim%d got %lld expected 1",
                name,
                i,
                (long long) t->ne[i]);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    *out = t;
    return TRELLIS_STATUS_OK;
}

static trellis_status bind_vec(
    trellis_tensor_store * store,
    const char * name,
    int64_t n,
    struct ggml_tensor ** out,
    char * first_issue,
    size_t first_issue_size) {
    int64_t ne[1] = { n };
    return bind_tensor(store, name, 1, ne, out, first_issue, first_issue_size);
}

static trellis_status bind_linear(
    trellis_tensor_store * store,
    const char * w_name,
    const char * b_name,
    int64_t in,
    int64_t out,
    struct ggml_tensor ** w,
    struct ggml_tensor ** b,
    char * first_issue,
    size_t first_issue_size) {
    int64_t ne[2] = { in, out };
    trellis_status status = bind_tensor(store, w_name, 2, ne, w, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    return bind_vec(store, b_name, out, b, first_issue, first_issue_size);
}

static trellis_status bind_qk_gamma(
    trellis_tensor_store * store,
    const char * name,
    struct ggml_tensor ** out,
    char * first_issue,
    size_t first_issue_size) {
    int64_t ne[2] = { TRELLIS_DIT_HEAD_DIM, TRELLIS_DIT_HEADS };
    return bind_tensor(store, name, 2, ne, out, first_issue, first_issue_size);
}

static trellis_status bind_block(
    trellis_tensor_store * store,
    int index,
    trellis_dit_flow_block_weights * block,
    char * first_issue,
    size_t first_issue_size) {
    char name[160];
    char bias[160];
    trellis_status status;

#define BIND_VEC_FIELD(field, suffix, n) do { \
    snprintf(name, sizeof(name), "blocks.%d.%s", index, suffix); \
    status = bind_vec(store, name, n, &block->field, first_issue, first_issue_size); \
    if (status != TRELLIS_STATUS_OK) return status; \
} while (0)

#define BIND_LINEAR_FIELD(wfield, bfield, suffix, in, out) do { \
    snprintf(name, sizeof(name), "blocks.%d.%s.weight", index, suffix); \
    snprintf(bias, sizeof(bias), "blocks.%d.%s.bias", index, suffix); \
    status = bind_linear(store, name, bias, in, out, &block->wfield, &block->bfield, first_issue, first_issue_size); \
    if (status != TRELLIS_STATUS_OK) return status; \
} while (0)

#define BIND_QK_FIELD(field, suffix) do { \
    snprintf(name, sizeof(name), "blocks.%d.%s", index, suffix); \
    status = bind_qk_gamma(store, name, &block->field, first_issue, first_issue_size); \
    if (status != TRELLIS_STATUS_OK) return status; \
} while (0)

    BIND_VEC_FIELD(modulation, "modulation", TRELLIS_DIT_MOD);
    BIND_VEC_FIELD(norm2_gamma, "norm2.weight", TRELLIS_DIT_C);
    BIND_VEC_FIELD(norm2_beta, "norm2.bias", TRELLIS_DIT_C);

    BIND_LINEAR_FIELD(self_qkv_w, self_qkv_b, "self_attn.to_qkv", TRELLIS_DIT_C, 3 * TRELLIS_DIT_C);
    BIND_LINEAR_FIELD(self_out_w, self_out_b, "self_attn.to_out", TRELLIS_DIT_C, TRELLIS_DIT_C);
    BIND_QK_FIELD(self_q_rms_gamma, "self_attn.q_rms_norm.gamma");
    BIND_QK_FIELD(self_k_rms_gamma, "self_attn.k_rms_norm.gamma");

    BIND_LINEAR_FIELD(cross_q_w, cross_q_b, "cross_attn.to_q", TRELLIS_DIT_C, TRELLIS_DIT_C);
    BIND_LINEAR_FIELD(cross_kv_w, cross_kv_b, "cross_attn.to_kv", TRELLIS_DIT_COND_C, 2 * TRELLIS_DIT_C);
    BIND_LINEAR_FIELD(cross_out_w, cross_out_b, "cross_attn.to_out", TRELLIS_DIT_C, TRELLIS_DIT_C);
    BIND_QK_FIELD(cross_q_rms_gamma, "cross_attn.q_rms_norm.gamma");
    BIND_QK_FIELD(cross_k_rms_gamma, "cross_attn.k_rms_norm.gamma");

    BIND_LINEAR_FIELD(mlp_fc1_w, mlp_fc1_b, "mlp.mlp.0", TRELLIS_DIT_C, TRELLIS_DIT_MLP);
    BIND_LINEAR_FIELD(mlp_fc2_w, mlp_fc2_b, "mlp.mlp.2", TRELLIS_DIT_MLP, TRELLIS_DIT_C);

#undef BIND_VEC_FIELD
#undef BIND_LINEAR_FIELD
#undef BIND_QK_FIELD

    return TRELLIS_STATUS_OK;
}

trellis_status trellis_dit_flow_bind_weights(
    trellis_tensor_store * store,
    int in_channels,
    int out_channels,
    trellis_dit_flow_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    if (store == NULL || weights == NULL || in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (first_issue != NULL && first_issue_size != 0) {
        first_issue[0] = '\0';
    }
    memset(weights, 0, sizeof(*weights));
    weights->in_channels = in_channels;
    weights->out_channels = out_channels;
    weights->model_channels = TRELLIS_DIT_C;
    weights->cond_channels = TRELLIS_DIT_COND_C;
    weights->time_frequency_dim = TRELLIS_DIT_T_FREQ;
    weights->heads = TRELLIS_DIT_HEADS;
    weights->head_dim = TRELLIS_DIT_HEAD_DIM;
    weights->mlp_channels = TRELLIS_DIT_MLP;
    weights->mod_channels = TRELLIS_DIT_MOD;
    weights->n_blocks = TRELLIS_DIT_FLOW_BLOCKS;
    weights->debug_block_parts = -1;
    weights->debug_disable_rope = 0;
    weights->emulate_bf16_blocks = 0;
    weights->final_norm_eps = 1e-5f;

    trellis_status status = bind_linear(
        store,
        "input_layer.weight",
        "input_layer.bias",
        in_channels,
        TRELLIS_DIT_C,
        &weights->input_w,
        &weights->input_b,
        first_issue,
        first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    status = bind_linear(
        store,
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.0.bias",
        TRELLIS_DIT_T_FREQ,
        TRELLIS_DIT_C,
        &weights->t_embedder_0_w,
        &weights->t_embedder_0_b,
        first_issue,
        first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    status = bind_linear(
        store,
        "t_embedder.mlp.2.weight",
        "t_embedder.mlp.2.bias",
        TRELLIS_DIT_C,
        TRELLIS_DIT_C,
        &weights->t_embedder_2_w,
        &weights->t_embedder_2_b,
        first_issue,
        first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    status = bind_linear(
        store,
        "adaLN_modulation.1.weight",
        "adaLN_modulation.1.bias",
        TRELLIS_DIT_C,
        TRELLIS_DIT_MOD,
        &weights->adaln_w,
        &weights->adaln_b,
        first_issue,
        first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    for (int i = 0; i < TRELLIS_DIT_FLOW_BLOCKS; ++i) {
        status = bind_block(store, i, &weights->blocks[i], first_issue, first_issue_size);
        if (status != TRELLIS_STATUS_OK) {
            return status;
        }
    }

    return bind_linear(
        store,
        "out_layer.weight",
        "out_layer.bias",
        TRELLIS_DIT_C,
        out_channels,
        &weights->out_w,
        &weights->out_b,
        first_issue,
        first_issue_size);
}

trellis_status trellis_ss_flow_bind_weights(
    trellis_tensor_store * store,
    trellis_dit_flow_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    return trellis_dit_flow_bind_weights(store, 8, 8, weights, first_issue, first_issue_size);
}

trellis_status trellis_shape_slat_flow_bind_weights(
    trellis_tensor_store * store,
    trellis_dit_flow_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    return trellis_dit_flow_bind_weights(store, 32, 32, weights, first_issue, first_issue_size);
}

trellis_status trellis_tex_slat_flow_bind_weights(
    trellis_tensor_store * store,
    trellis_dit_flow_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    return trellis_dit_flow_bind_weights(store, 64, 32, weights, first_issue, first_issue_size);
}

static trellis_ggml_modulated_cross_block_params make_block_params(
    const trellis_dit_flow_block_weights * block,
    int debug_parts,
    int emulate_bf16) {
    trellis_ggml_modulated_cross_block_params params;
    memset(&params, 0, sizeof(params));
    params.block_modulation = block->modulation;
    params.norm2_gamma = block->norm2_gamma;
    params.norm2_beta = block->norm2_beta;
    params.self_qkv_w = block->self_qkv_w;
    params.self_qkv_b = block->self_qkv_b;
    params.self_q_rms_gamma = block->self_q_rms_gamma;
    params.self_k_rms_gamma = block->self_k_rms_gamma;
    params.self_out_w = block->self_out_w;
    params.self_out_b = block->self_out_b;
    params.cross_q_w = block->cross_q_w;
    params.cross_q_b = block->cross_q_b;
    params.cross_kv_w = block->cross_kv_w;
    params.cross_kv_b = block->cross_kv_b;
    params.cross_q_rms_gamma = block->cross_q_rms_gamma;
    params.cross_k_rms_gamma = block->cross_k_rms_gamma;
    params.cross_out_w = block->cross_out_w;
    params.cross_out_b = block->cross_out_b;
    params.mlp_fc1_w = block->mlp_fc1_w;
    params.mlp_fc1_b = block->mlp_fc1_b;
    params.mlp_fc2_w = block->mlp_fc2_w;
    params.mlp_fc2_b = block->mlp_fc2_b;
    params.debug_parts = debug_parts;
    params.emulate_bf16 = emulate_bf16;
    return params;
}

struct ggml_tensor * trellis_dit_flow_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * timesteps,
    struct ggml_tensor * context,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_dit_flow_weights * weights) {
    if (ctx == NULL || x == NULL || timesteps == NULL || context == NULL || weights == NULL) {
        return NULL;
    }
    if (x->ne[0] != weights->in_channels || context->ne[0] != weights->cond_channels) {
        return NULL;
    }
    if (x->ne[2] != timesteps->ne[0] || context->ne[2] != x->ne[2]) {
        return NULL;
    }
    if (weights->n_blocks < 0 || weights->n_blocks > TRELLIS_DIT_FLOW_BLOCKS ||
        weights->model_channels <= 0 || weights->heads <= 0) {
        return NULL;
    }

    struct ggml_tensor * h = trellis_ggml_linear(ctx, x, weights->input_w, weights->input_b);
    struct ggml_tensor * t_emb = trellis_ggml_timestep_mlp(
        ctx,
        timesteps,
        weights->time_frequency_dim,
        weights->t_embedder_0_w,
        weights->t_embedder_0_b,
        weights->t_embedder_2_w,
        weights->t_embedder_2_b);
    struct ggml_tensor * mod6 = trellis_ggml_linear(ctx, ggml_silu(ctx, t_emb), weights->adaln_w, weights->adaln_b);
    if (weights->emulate_bf16_blocks) {
        h = trellis_ggml_bf16_roundtrip(ctx, h);
        mod6 = trellis_ggml_bf16_roundtrip(ctx, mod6);
        context = trellis_ggml_bf16_roundtrip(ctx, context);
    }

    for (int i = 0; i < weights->n_blocks; ++i) {
        trellis_ggml_modulated_cross_block_params params =
            make_block_params(&weights->blocks[i], weights->debug_block_parts, weights->emulate_bf16_blocks);
        h = trellis_ggml_modulated_cross_block_rope(
            ctx,
            h,
            mod6,
            context,
            weights->heads,
            &params,
            weights->debug_disable_rope ? NULL : cos_phase,
            weights->debug_disable_rope ? NULL : sin_phase);
        if (h == NULL) {
            return NULL;
        }
    }

    h = trellis_ggml_layer_norm(
        ctx,
        h,
        NULL,
        NULL,
        weights->final_norm_eps > 0.0f ? weights->final_norm_eps : 1e-5f);
    return trellis_ggml_linear(ctx, h, weights->out_w, weights->out_b);
}
