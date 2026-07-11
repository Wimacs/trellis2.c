#include "tokenskin.h"

#include <stdlib.h>

static struct ggml_tensor * bf16_boundary(
    struct ggml_context * ctx,
    struct ggml_tensor * tensor) {
    return trellis_ggml_bf16_roundtrip(ctx, tensor);
}

static struct ggml_tensor * mesh_cross_block(
    struct ggml_context * ctx,
    struct ggml_tensor * query,
    struct ggml_tensor * context,
    const tokenskin_mesh_cross_weights * weights,
    const trellis_ggml_attention_policy * attention_policy) {
    struct ggml_tensor * q_norm = trellis_ggml_layer_norm(
        ctx, query, weights->norm_q.weight, weights->norm_q.bias, 1.0e-5f);
    struct ggml_tensor * context_norm = trellis_ggml_layer_norm(
        ctx, context, weights->norm_context.weight, weights->norm_context.bias, 1.0e-5f);
    q_norm = bf16_boundary(ctx, q_norm);
    context_norm = bf16_boundary(ctx, context_norm);
    struct ggml_tensor * attention = tokenskin_attention_standard_q_interleaved_kv(
        ctx,
        q_norm,
        context_norm,
        TOKENSKIN_MESH_HEADS,
        &weights->q,
        &weights->kv,
        &weights->out,
        attention_policy);
    if (attention == NULL) return NULL;
    query = bf16_boundary(ctx, ggml_add(ctx, query, bf16_boundary(ctx, attention)));
    struct ggml_tensor * ff = trellis_ggml_layer_norm(
        ctx, query, weights->norm_ff.weight, weights->norm_ff.bias, 1.0e-5f);
    ff = bf16_boundary(ctx, ff);
    ff = trellis_ggml_linear(ctx, ff, weights->ff_in.weight, weights->ff_in.bias);
    ff = bf16_boundary(ctx, ggml_gelu(ctx, bf16_boundary(ctx, ff)));
    ff = trellis_ggml_linear(ctx, ff, weights->ff_out.weight, weights->ff_out.bias);
    return bf16_boundary(ctx, ggml_add(ctx, query, bf16_boundary(ctx, ff)));
}

static struct ggml_tensor * mesh_self_block(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    const tokenskin_mesh_self_weights * weights,
    const trellis_ggml_attention_policy * attention_policy) {
    struct ggml_tensor * norm = trellis_ggml_layer_norm(
        ctx, x, weights->norm_attn.weight, weights->norm_attn.bias, 1.0e-5f);
    norm = bf16_boundary(ctx, norm);
    struct ggml_tensor * attention = tokenskin_attention_interleaved_qkv(
        ctx, norm, TOKENSKIN_MESH_HEADS, &weights->qkv, &weights->out,
        attention_policy);
    if (attention == NULL) return NULL;
    x = bf16_boundary(ctx, ggml_add(ctx, x, bf16_boundary(ctx, attention)));
    struct ggml_tensor * ff = trellis_ggml_layer_norm(
        ctx, x, weights->norm_ff.weight, weights->norm_ff.bias, 1.0e-5f);
    ff = bf16_boundary(ctx, ff);
    ff = trellis_ggml_linear(ctx, ff, weights->ff_in.weight, weights->ff_in.bias);
    ff = bf16_boundary(ctx, ggml_gelu(ctx, bf16_boundary(ctx, ff)));
    ff = trellis_ggml_linear(ctx, ff, weights->ff_out.weight, weights->ff_out.bias);
    return bf16_boundary(ctx, ggml_add(ctx, x, bf16_boundary(ctx, ff)));
}

trellis_status tokenskin_mesh_encoder_compute(
    const trellis_backend_context * backend,
    const tokenskin_mesh_encoder_weights * weights,
    const trellis_ggml_attention_policy * attention_policy,
    const float * full_features,
    int64_t full_tokens,
    const float * query_features,
    float * output_512x896) {
    if (backend == NULL || backend->backend == NULL || weights == NULL ||
        !trellis_ggml_attention_policy_is_valid(attention_policy) ||
        full_features == NULL || full_tokens <= 0 || query_features == NULL ||
        output_512x896 == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;

    const size_t graph_nodes = 65536;
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * graph_nodes +
            ggml_graph_overhead_custom(graph_nodes, false) + 4 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    struct ggml_tensor * full = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, TOKENSKIN_MESH_FEATURES, full_tokens, 1);
    struct ggml_tensor * query_input = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, TOKENSKIN_MESH_FEATURES, TOKENSKIN_MESH_TOKENS, 1);
    ggml_set_name(full, "tokenskin_mesh_full");
    ggml_set_name(query_input, "tokenskin_mesh_query");
    struct ggml_tensor * context = trellis_ggml_linear(
        ctx, full, weights->input.weight, weights->input.bias);
    struct ggml_tensor * query = trellis_ggml_linear(
        ctx, query_input, weights->input.weight, weights->input.bias);
    context = bf16_boundary(ctx, context);
    query = bf16_boundary(ctx, query);
    struct ggml_tensor * hidden = mesh_cross_block(
        ctx, query, context, &weights->cross, attention_policy);
    for (int i = 0; hidden != NULL && i < TOKENSKIN_MESH_LAYERS; ++i) {
        hidden = mesh_self_block(ctx, hidden, &weights->blocks[i], attention_policy);
    }
    if (hidden != NULL) {
        hidden = trellis_ggml_layer_norm(
            ctx, hidden, weights->output_norm.weight, weights->output_norm.bias, 1.0e-5f);
        hidden = bf16_boundary(ctx, hidden);
        hidden = trellis_ggml_linear(
            ctx, hidden, weights->output_projection.weight, weights->output_projection.bias);
        hidden = bf16_boundary(ctx, hidden);
        hidden = trellis_ggml_rms_norm(
            ctx, hidden, weights->output_rms_weight, 1.1920928955078125e-7f);
        hidden = bf16_boundary(ctx, hidden);
    }
    if (hidden == NULL) {
        ggml_free(ctx);
        return TRELLIS_STATUS_ERROR;
    }
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    if (graph == NULL) {
        ggml_free(ctx);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_build_forward_expand(graph, hidden);
    ggml_gallocr_t allocator = trellis_backend_new_graph_allocator(backend);
    if (allocator == NULL || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (allocator != NULL) ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_backend_tensor_set(full, full_features, 0, ggml_nbytes(full));
    ggml_backend_tensor_set(query_input, query_features, 0, ggml_nbytes(query_input));
    trellis_status status = trellis_backend_compute_graph(backend, graph);
    if (status == TRELLIS_STATUS_OK) {
        ggml_backend_tensor_get(hidden, output_512x896, 0, ggml_nbytes(hidden));
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return status;
}
