#include "tokenskin.h"

static struct ggml_tensor * bf16_boundary(
    struct ggml_context * ctx,
    struct ggml_tensor * tensor) {
    return trellis_ggml_bf16_roundtrip(ctx, tensor);
}

static struct ggml_tensor * tripo_block(
    struct ggml_context * ctx,
    struct ggml_tensor * hidden,
    struct ggml_tensor * context,
    const tokenskin_tripo_block_weights * weights,
    const trellis_ggml_attention_policy * attention_policy) {
    struct ggml_tensor * query_norm = trellis_ggml_layer_norm(
        ctx,
        hidden,
        weights->norm_attention.weight,
        weights->norm_attention.bias,
        1.0e-5f);
    query_norm = bf16_boundary(ctx, query_norm);
    struct ggml_tensor * context_norm = NULL;
    if (weights->cross_attention) {
        if (context == NULL) return NULL;
        context_norm = trellis_ggml_layer_norm(
            ctx,
            context,
            weights->norm_context.weight,
            weights->norm_context.bias,
            1.0e-5f);
        context_norm = bf16_boundary(ctx, context_norm);
    }
    struct ggml_tensor * attention = tokenskin_tripo_attention(
        ctx,
        query_norm,
        context_norm,
        TOKENSKIN_VAE_HEADS,
        weights,
        attention_policy);
    if (attention == NULL) return NULL;
    hidden = bf16_boundary(ctx, ggml_add(ctx, hidden, bf16_boundary(ctx, attention)));
    struct ggml_tensor * ff = trellis_ggml_layer_norm(
        ctx,
        hidden,
        weights->norm_ff.weight,
        weights->norm_ff.bias,
        1.0e-5f);
    ff = bf16_boundary(ctx, ff);
    ff = trellis_ggml_linear(ctx, ff, weights->ff_in.weight, weights->ff_in.bias);
    ff = bf16_boundary(ctx, ggml_gelu(ctx, bf16_boundary(ctx, ff)));
    ff = trellis_ggml_linear(ctx, ff, weights->ff_out.weight, weights->ff_out.bias);
    return bf16_boundary(ctx, ggml_add(ctx, hidden, bf16_boundary(ctx, ff)));
}

trellis_status tokenskin_fsq_condition_compute(
    const trellis_backend_context * backend,
    const tokenskin_fsq_cvae_weights * weights,
    const trellis_ggml_attention_policy * attention_policy,
    const float * full_features,
    int64_t full_tokens,
    const float * query_features,
    float * output_384x512) {
    if (backend == NULL || backend->backend == NULL || weights == NULL ||
        !trellis_ggml_attention_policy_is_valid(attention_policy) ||
        full_features == NULL || full_tokens <= 0 || query_features == NULL ||
        output_384x512 == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;

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
        ctx, GGML_TYPE_F32, TOKENSKIN_MESH_FEATURES, TOKENSKIN_VAE_COND_TOKENS, 1);
    ggml_set_name(full, "tokenskin_vae_cond_full");
    ggml_set_name(query_input, "tokenskin_vae_cond_query");
    struct ggml_tensor * context = trellis_ggml_linear(
        ctx, full, weights->cond_input.weight, weights->cond_input.bias);
    struct ggml_tensor * hidden = trellis_ggml_linear(
        ctx, query_input, weights->cond_input.weight, weights->cond_input.bias);
    context = bf16_boundary(ctx, context);
    hidden = bf16_boundary(ctx, hidden);
    hidden = tripo_block(
        ctx, hidden, context, &weights->cond_cross, attention_policy);
    for (int i = 0; hidden != NULL && i < TOKENSKIN_VAE_COND_LAYERS; ++i) {
        hidden = tripo_block(
            ctx, hidden, NULL, &weights->cond_self[i], attention_policy);
    }
    if (hidden != NULL) {
        hidden = trellis_ggml_layer_norm(
            ctx,
            hidden,
            weights->cond_output_norm.weight,
            weights->cond_output_norm.bias,
            1.0e-5f);
        hidden = bf16_boundary(ctx, hidden);
        hidden = trellis_ggml_linear(
            ctx, hidden, weights->cond_quant.weight, weights->cond_quant.bias);
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
        ggml_backend_tensor_get(hidden, output_384x512, 0, ggml_nbytes(hidden));
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return status;
}

trellis_status tokenskin_fsq_skin_decode_compute(
    const trellis_backend_context * backend,
    const tokenskin_fsq_cvae_weights * weights,
    const trellis_ggml_attention_policy * attention_policy,
    const float * fsq_codes_4x5,
    const float * condition_384x512,
    const float * query_features,
    int64_t query_tokens,
    float * skin_probability) {
    if (backend == NULL || backend->backend == NULL || weights == NULL ||
        !trellis_ggml_attention_policy_is_valid(attention_policy) ||
        fsq_codes_4x5 == NULL || condition_384x512 == NULL ||
        query_features == NULL || query_tokens <= 0 || skin_probability == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    const size_t graph_nodes = 65536;
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * graph_nodes +
            ggml_graph_overhead_custom(graph_nodes, false) + 4 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;

    struct ggml_tensor * codes = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, 5, 4, 1);
    struct ggml_tensor * condition_input = ggml_new_tensor_3d(
        ctx,
        GGML_TYPE_F32,
        TOKENSKIN_VAE_LATENT_CHANNELS,
        TOKENSKIN_VAE_COND_TOKENS,
        1);
    struct ggml_tensor * query_input = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, TOKENSKIN_MESH_FEATURES, query_tokens, 1);
    ggml_set_name(codes, "tokenskin_fsq_codes");
    ggml_set_name(condition_input, "tokenskin_skin_condition");
    ggml_set_name(query_input, "tokenskin_skin_queries");

    struct ggml_tensor * latent = trellis_ggml_linear(
        ctx,
        codes,
        weights->fsq_project_out.weight,
        weights->fsq_project_out.bias);
    latent = bf16_boundary(ctx, latent);
    struct ggml_tensor * condition = bf16_boundary(ctx, condition_input);
    latent = ggml_concat(ctx, latent, condition, 1);
    latent = trellis_ggml_linear(
        ctx, latent, weights->post_quant.weight, weights->post_quant.bias);
    latent = bf16_boundary(ctx, latent);
    for (int i = 0; latent != NULL && i < TOKENSKIN_VAE_DECODER_LAYERS; ++i) {
        latent = tripo_block(
            ctx, latent, NULL, &weights->decoder_self[i], attention_policy);
    }

    struct ggml_tensor * hidden = trellis_ggml_linear(
        ctx,
        query_input,
        weights->decoder_query.weight,
        weights->decoder_query.bias);
    hidden = bf16_boundary(ctx, hidden);
    if (latent != NULL) {
        hidden = tripo_block(
            ctx,
            hidden,
            latent,
            &weights->decoder_cross,
            attention_policy);
    } else {
        hidden = NULL;
    }
    if (hidden != NULL) {
        hidden = trellis_ggml_layer_norm(
            ctx,
            hidden,
            weights->decoder_output_norm.weight,
            weights->decoder_output_norm.bias,
            1.0e-5f);
        hidden = bf16_boundary(ctx, hidden);
        hidden = trellis_ggml_linear(
            ctx,
            hidden,
            weights->decoder_output.weight,
            weights->decoder_output.bias);
        hidden = bf16_boundary(ctx, hidden);
        hidden = ggml_sigmoid(ctx, hidden);
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
    ggml_backend_tensor_set(codes, fsq_codes_4x5, 0, ggml_nbytes(codes));
    ggml_backend_tensor_set(
        condition_input,
        condition_384x512,
        0,
        ggml_nbytes(condition_input));
    ggml_backend_tensor_set(
        query_input, query_features, 0, ggml_nbytes(query_input));
    trellis_status status = trellis_backend_compute_graph(backend, graph);
    if (status == TRELLIS_STATUS_OK) {
        ggml_backend_tensor_get(
            hidden, skin_probability, 0, (size_t) query_tokens * sizeof(float));
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return status;
}

trellis_status tokenskin_fsq_decoder_latent_compute(
    const trellis_backend_context * backend,
    const tokenskin_fsq_cvae_weights * weights,
    const trellis_ggml_attention_policy * attention_policy,
    const float * fsq_codes_4x5,
    const float * condition_384x512,
    int self_layers,
    float * latent_388x768) {
    if (backend == NULL || backend->backend == NULL || weights == NULL ||
        !trellis_ggml_attention_policy_is_valid(attention_policy) ||
        fsq_codes_4x5 == NULL || condition_384x512 == NULL ||
        self_layers < 0 || self_layers > TOKENSKIN_VAE_DECODER_LAYERS ||
        latent_388x768 == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t graph_nodes = 65536;
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * graph_nodes +
            ggml_graph_overhead_custom(graph_nodes, false) + 4 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    struct ggml_tensor * codes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 4, 1);
    struct ggml_tensor * condition_input = ggml_new_tensor_3d(
        ctx,
        GGML_TYPE_F32,
        TOKENSKIN_VAE_LATENT_CHANNELS,
        TOKENSKIN_VAE_COND_TOKENS,
        1);
    struct ggml_tensor * latent = trellis_ggml_linear(
        ctx,
        codes,
        weights->fsq_project_out.weight,
        weights->fsq_project_out.bias);
    latent = bf16_boundary(ctx, latent);
    struct ggml_tensor * condition = bf16_boundary(ctx, condition_input);
    latent = ggml_concat(ctx, latent, condition, 1);
    latent = trellis_ggml_linear(
        ctx, latent, weights->post_quant.weight, weights->post_quant.bias);
    latent = bf16_boundary(ctx, latent);
    for (int i = 0; latent != NULL && i < self_layers; ++i) {
        latent = tripo_block(
            ctx, latent, NULL, &weights->decoder_self[i], attention_policy);
    }
    if (latent == NULL) {
        ggml_free(ctx);
        return TRELLIS_STATUS_ERROR;
    }
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    if (graph == NULL) {
        ggml_free(ctx);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_build_forward_expand(graph, latent);
    ggml_gallocr_t allocator = trellis_backend_new_graph_allocator(backend);
    if (allocator == NULL || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (allocator != NULL) ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_backend_tensor_set(codes, fsq_codes_4x5, 0, ggml_nbytes(codes));
    ggml_backend_tensor_set(
        condition_input,
        condition_384x512,
        0,
        ggml_nbytes(condition_input));
    trellis_status status = trellis_backend_compute_graph(backend, graph);
    if (status == TRELLIS_STATUS_OK) {
        ggml_backend_tensor_get(latent, latent_388x768, 0, ggml_nbytes(latent));
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return status;
}
