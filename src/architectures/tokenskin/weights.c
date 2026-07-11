#include "tokenskin.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void set_issue(char * issue, size_t issue_size, const char * fmt, ...) {
    if (issue == NULL || issue_size == 0 || issue[0] != '\0') {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(issue, issue_size, fmt, args);
    va_end(args);
}

static struct ggml_tensor * require_tensor(
    trellis_tensor_store * store,
    const char * name,
    int64_t ne0,
    int64_t ne1,
    char * issue,
    size_t issue_size) {
    struct ggml_tensor * tensor = trellis_tensor_store_get(store, name);
    if (tensor == NULL) {
        set_issue(issue, issue_size, "missing tensor %s", name);
        return NULL;
    }
    if (tensor->ne[0] != ne0 || tensor->ne[1] != ne1 ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        set_issue(
            issue,
            issue_size,
            "tensor %s has ggml shape [%lld,%lld,%lld,%lld], expected [%lld,%lld,1,1]",
            name,
            (long long) tensor->ne[0],
            (long long) tensor->ne[1],
            (long long) tensor->ne[2],
            (long long) tensor->ne[3],
            (long long) ne0,
            (long long) ne1);
        return NULL;
    }
    if (tensor->type != GGML_TYPE_BF16) {
        set_issue(
            issue,
            issue_size,
            "tensor %s has dtype %s, expected bf16",
            name,
            ggml_type_name(tensor->type));
        return NULL;
    }
    return tensor;
}

static int bind_linear(
    trellis_tensor_store * store,
    const char * prefix,
    int64_t input,
    int64_t output,
    int has_bias,
    tokenskin_linear_weights * weights,
    char * issue,
    size_t issue_size) {
    char name[1024];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    weights->weight = require_tensor(store, name, input, output, issue, issue_size);
    if (weights->weight == NULL) return 0;
    weights->bias = NULL;
    if (has_bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        weights->bias = require_tensor(store, name, output, 1, issue, issue_size);
        if (weights->bias == NULL) return 0;
    }
    return 1;
}

static int bind_norm(
    trellis_tensor_store * store,
    const char * prefix,
    int64_t channels,
    int has_bias,
    tokenskin_norm_weights * weights,
    char * issue,
    size_t issue_size) {
    char name[1024];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    weights->weight = require_tensor(store, name, channels, 1, issue, issue_size);
    if (weights->weight == NULL) return 0;
    weights->bias = NULL;
    if (has_bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        weights->bias = require_tensor(store, name, channels, 1, issue, issue_size);
        if (weights->bias == NULL) return 0;
    }
    return 1;
}

trellis_status tokenskin_bind_mesh_encoder_weights(
    trellis_tensor_store * store,
    tokenskin_mesh_encoder_weights * weights,
    char * issue,
    size_t issue_size) {
    if (store == NULL || weights == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;
    memset(weights, 0, sizeof(*weights));
    if (issue != NULL && issue_size != 0) issue[0] = '\0';

    if (!bind_linear(store, "mesh_encoder.encoder.input_proj", 54, 512, 1, &weights->input, issue, issue_size) ||
        !bind_linear(store, "mesh_encoder.encoder.cross_attn.attn.c_q", 512, 512, 0, &weights->cross.q, issue, issue_size) ||
        !bind_linear(store, "mesh_encoder.encoder.cross_attn.attn.c_kv", 512, 1024, 0, &weights->cross.kv, issue, issue_size) ||
        !bind_linear(store, "mesh_encoder.encoder.cross_attn.attn.c_proj", 512, 512, 1, &weights->cross.out, issue, issue_size) ||
        !bind_norm(store, "mesh_encoder.encoder.cross_attn.ln_1", 512, 1, &weights->cross.norm_q, issue, issue_size) ||
        !bind_norm(store, "mesh_encoder.encoder.cross_attn.ln_2", 512, 1, &weights->cross.norm_context, issue, issue_size) ||
        !bind_norm(store, "mesh_encoder.encoder.cross_attn.ln_3", 512, 1, &weights->cross.norm_ff, issue, issue_size) ||
        !bind_linear(store, "mesh_encoder.encoder.cross_attn.mlp.c_fc", 512, 2048, 1, &weights->cross.ff_in, issue, issue_size) ||
        !bind_linear(store, "mesh_encoder.encoder.cross_attn.mlp.c_proj", 2048, 512, 1, &weights->cross.ff_out, issue, issue_size)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    for (int i = 0; i < TOKENSKIN_MESH_LAYERS; ++i) {
        char prefix[256];
        tokenskin_mesh_self_weights * block = &weights->blocks[i];
        snprintf(prefix, sizeof(prefix), "mesh_encoder.encoder.self_attn.resblocks.%d.attn.c_qkv", i);
        if (!bind_linear(store, prefix, 512, 1536, 0, &block->qkv, issue, issue_size)) return TRELLIS_STATUS_PARSE_ERROR;
        snprintf(prefix, sizeof(prefix), "mesh_encoder.encoder.self_attn.resblocks.%d.attn.c_proj", i);
        if (!bind_linear(store, prefix, 512, 512, 1, &block->out, issue, issue_size)) return TRELLIS_STATUS_PARSE_ERROR;
        snprintf(prefix, sizeof(prefix), "mesh_encoder.encoder.self_attn.resblocks.%d.ln_1", i);
        if (!bind_norm(store, prefix, 512, 1, &block->norm_attn, issue, issue_size)) return TRELLIS_STATUS_PARSE_ERROR;
        snprintf(prefix, sizeof(prefix), "mesh_encoder.encoder.self_attn.resblocks.%d.ln_2", i);
        if (!bind_norm(store, prefix, 512, 1, &block->norm_ff, issue, issue_size)) return TRELLIS_STATUS_PARSE_ERROR;
        snprintf(prefix, sizeof(prefix), "mesh_encoder.encoder.self_attn.resblocks.%d.mlp.c_fc", i);
        if (!bind_linear(store, prefix, 512, 2048, 1, &block->ff_in, issue, issue_size)) return TRELLIS_STATUS_PARSE_ERROR;
        snprintf(prefix, sizeof(prefix), "mesh_encoder.encoder.self_attn.resblocks.%d.mlp.c_proj", i);
        if (!bind_linear(store, prefix, 2048, 512, 1, &block->ff_out, issue, issue_size)) return TRELLIS_STATUS_PARSE_ERROR;
    }
    if (!bind_norm(store, "mesh_encoder.encoder.ln_post", 512, 1, &weights->output_norm, issue, issue_size) ||
        !bind_linear(store, "output_proj.0", 512, 896, 1, &weights->output_projection, issue, issue_size)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    weights->output_rms_weight = require_tensor(store, "output_proj.1.weight", 896, 1, issue, issue_size);
    return weights->output_rms_weight != NULL ? TRELLIS_STATUS_OK : TRELLIS_STATUS_PARSE_ERROR;
}

trellis_status tokenskin_bind_qwen_weights(
    trellis_tensor_store * store,
    tokenskin_qwen_weights * weights,
    char * issue,
    size_t issue_size) {
    if (store == NULL || weights == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;
    memset(weights, 0, sizeof(*weights));
    if (issue != NULL && issue_size != 0) issue[0] = '\0';
    weights->token_embedding = require_tensor(
        store, "transformer.model.embed_tokens.weight", 896, 33036, issue, issue_size);
    if (weights->token_embedding == NULL) return TRELLIS_STATUS_PARSE_ERROR;

    for (int i = 0; i < TOKENSKIN_QWEN_LAYERS; ++i) {
        char name[512];
        tokenskin_qwen_block_weights * block = &weights->blocks[i];
#define QWEN_TENSOR(field, suffix, ne0, ne1) do { \
    snprintf(name, sizeof(name), "transformer.model.layers.%d.%s", i, suffix); \
    block->field = require_tensor(store, name, ne0, ne1, issue, issue_size); \
    if (block->field == NULL) return TRELLIS_STATUS_PARSE_ERROR; \
} while (0)
        QWEN_TENSOR(input_norm, "input_layernorm.weight", 896, 1);
        QWEN_TENSOR(q, "self_attn.q_proj.weight", 896, 2048);
        QWEN_TENSOR(k, "self_attn.k_proj.weight", 896, 1024);
        QWEN_TENSOR(v, "self_attn.v_proj.weight", 896, 1024);
        QWEN_TENSOR(out, "self_attn.o_proj.weight", 2048, 896);
        QWEN_TENSOR(q_norm, "self_attn.q_norm.weight", 128, 1);
        QWEN_TENSOR(k_norm, "self_attn.k_norm.weight", 128, 1);
        QWEN_TENSOR(post_attention_norm, "post_attention_layernorm.weight", 896, 1);
        QWEN_TENSOR(gate, "mlp.gate_proj.weight", 896, 3072);
        QWEN_TENSOR(up, "mlp.up_proj.weight", 896, 3072);
        QWEN_TENSOR(down, "mlp.down_proj.weight", 3072, 896);
#undef QWEN_TENSOR
    }
    weights->output_norm = require_tensor(
        store, "transformer.model.norm.weight", 896, 1, issue, issue_size);
    if (weights->output_norm == NULL) return TRELLIS_STATUS_PARSE_ERROR;
    weights->lm_head = weights->token_embedding;
    return TRELLIS_STATUS_OK;
}

static int bind_tripo_block(
    trellis_tensor_store * store,
    const char * prefix,
    int cross,
    tokenskin_tripo_block_weights * block,
    char * issue,
    size_t issue_size) {
    char name[512];
    memset(block, 0, sizeof(*block));
    block->cross_attention = cross;
    snprintf(name, sizeof(name), "%s.%s", prefix, cross ? "norm2" : "norm1");
    if (!bind_norm(store, name, 768, 1, &block->norm_attention, issue, issue_size)) return 0;
    if (cross) {
        snprintf(name, sizeof(name), "%s.attn2.norm_cross", prefix);
        if (!bind_norm(store, name, 768, 1, &block->norm_context, issue, issue_size)) return 0;
    }
    const char * attn = cross ? "attn2" : "attn1";
    snprintf(name, sizeof(name), "%s.%s.to_q", prefix, attn);
    if (!bind_linear(store, name, 768, 768, 0, &block->q, issue, issue_size)) return 0;
    snprintf(name, sizeof(name), "%s.%s.to_k", prefix, attn);
    if (!bind_linear(store, name, 768, 768, 0, &block->k, issue, issue_size)) return 0;
    snprintf(name, sizeof(name), "%s.%s.to_v", prefix, attn);
    if (!bind_linear(store, name, 768, 768, 0, &block->v, issue, issue_size)) return 0;
    snprintf(name, sizeof(name), "%s.%s.to_out.0", prefix, attn);
    if (!bind_linear(store, name, 768, 768, 1, &block->out, issue, issue_size)) return 0;
    snprintf(name, sizeof(name), "%s.norm3", prefix);
    if (!bind_norm(store, name, 768, 1, &block->norm_ff, issue, issue_size)) return 0;
    snprintf(name, sizeof(name), "%s.ff.net.0.proj", prefix);
    if (!bind_linear(store, name, 768, 3072, 1, &block->ff_in, issue, issue_size)) return 0;
    snprintf(name, sizeof(name), "%s.ff.net.2", prefix);
    return bind_linear(store, name, 3072, 768, 1, &block->ff_out, issue, issue_size);
}

trellis_status tokenskin_bind_fsq_cvae_weights(
    trellis_tensor_store * store,
    tokenskin_fsq_cvae_weights * weights,
    char * issue,
    size_t issue_size) {
    if (store == NULL || weights == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;
    memset(weights, 0, sizeof(*weights));
    if (issue != NULL && issue_size != 0) issue[0] = '\0';

    if (!bind_linear(store, "vae.model.cond_encoder.proj_in", 54, 768, 1, &weights->cond_input, issue, issue_size) ||
        !bind_tripo_block(store, "vae.model.cond_encoder.blocks.0", 1, &weights->cond_cross, issue, issue_size)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    for (int i = 0; i < TOKENSKIN_VAE_COND_LAYERS; ++i) {
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "vae.model.cond_encoder.blocks.%d", i + 1);
        if (!bind_tripo_block(store, prefix, 0, &weights->cond_self[i], issue, issue_size)) return TRELLIS_STATUS_PARSE_ERROR;
    }
    if (!bind_norm(store, "vae.model.cond_encoder.norm_out", 768, 1, &weights->cond_output_norm, issue, issue_size) ||
        !bind_linear(store, "vae.model.cond_quant", 768, 512, 1, &weights->cond_quant, issue, issue_size) ||
        !bind_linear(store, "vae.model.FSQ.project_out", 5, 512, 1, &weights->fsq_project_out, issue, issue_size) ||
        !bind_linear(store, "vae.model.post_quant", 512, 768, 1, &weights->post_quant, issue, issue_size)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    for (int i = 0; i < TOKENSKIN_VAE_DECODER_LAYERS; ++i) {
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "vae.model.decoder.blocks.%d", i);
        if (!bind_tripo_block(store, prefix, 0, &weights->decoder_self[i], issue, issue_size)) return TRELLIS_STATUS_PARSE_ERROR;
    }
    if (!bind_tripo_block(store, "vae.model.decoder.blocks.10", 1, &weights->decoder_cross, issue, issue_size) ||
        !bind_linear(store, "vae.model.decoder.proj_query", 54, 768, 1, &weights->decoder_query, issue, issue_size) ||
        !bind_norm(store, "vae.model.decoder.norm_out", 768, 1, &weights->decoder_output_norm, issue, issue_size) ||
        !bind_linear(store, "vae.model.decoder.proj_out", 768, 1, 1, &weights->decoder_output, issue, issue_size)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    return TRELLIS_STATUS_OK;
}
