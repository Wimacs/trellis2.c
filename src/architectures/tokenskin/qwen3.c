#include "tokenskin.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tokenskin_qwen_executor {
    const trellis_backend_context * backend;
    const tokenskin_qwen_weights * weights;
    trellis_ggml_attention_policy attention_policy;

    struct ggml_context * cache_ctx;
    ggml_backend_buffer_t cache_buffer;
    struct ggml_tensor * cache_k[TOKENSKIN_QWEN_LAYERS];
    struct ggml_tensor * cache_v[TOKENSKIN_QWEN_LAYERS];

    int capacity;
    int length;
};

enum {
    TOKENSKIN_QWEN_GRAPH_NODES = 16384,
    TOKENSKIN_QWEN_KV_CHANNELS =
        TOKENSKIN_QWEN_HEAD_DIM * TOKENSKIN_QWEN_KV_HEADS,
};

static int qwen_tensor_shape(
    const struct ggml_tensor * tensor,
    int64_t ne0,
    int64_t ne1) {
    return tensor != NULL && tensor->ne[0] == ne0 && tensor->ne[1] == ne1 &&
        tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static int qwen_weights_are_valid(const tokenskin_qwen_weights * weights) {
    if (weights == NULL ||
        !qwen_tensor_shape(
            weights->token_embedding,
            TOKENSKIN_QWEN_HIDDEN,
            TOKENSKIN_QWEN_VOCAB) ||
        !qwen_tensor_shape(weights->output_norm, TOKENSKIN_QWEN_HIDDEN, 1)) {
        return 0;
    }
    for (int layer = 0; layer < TOKENSKIN_QWEN_LAYERS; ++layer) {
        const tokenskin_qwen_block_weights * block = &weights->blocks[layer];
        if (!qwen_tensor_shape(block->input_norm, TOKENSKIN_QWEN_HIDDEN, 1) ||
            !qwen_tensor_shape(
                block->q,
                TOKENSKIN_QWEN_HIDDEN,
                TOKENSKIN_QWEN_HEAD_DIM * TOKENSKIN_QWEN_HEADS) ||
            !qwen_tensor_shape(
                block->k,
                TOKENSKIN_QWEN_HIDDEN,
                TOKENSKIN_QWEN_KV_CHANNELS) ||
            !qwen_tensor_shape(
                block->v,
                TOKENSKIN_QWEN_HIDDEN,
                TOKENSKIN_QWEN_KV_CHANNELS) ||
            !qwen_tensor_shape(
                block->out,
                TOKENSKIN_QWEN_HEAD_DIM * TOKENSKIN_QWEN_HEADS,
                TOKENSKIN_QWEN_HIDDEN) ||
            !qwen_tensor_shape(block->q_norm, TOKENSKIN_QWEN_HEAD_DIM, 1) ||
            !qwen_tensor_shape(block->k_norm, TOKENSKIN_QWEN_HEAD_DIM, 1) ||
            !qwen_tensor_shape(
                block->post_attention_norm,
                TOKENSKIN_QWEN_HIDDEN,
                1) ||
            !qwen_tensor_shape(
                block->gate,
                TOKENSKIN_QWEN_HIDDEN,
                TOKENSKIN_QWEN_INTERMEDIATE) ||
            !qwen_tensor_shape(
                block->up,
                TOKENSKIN_QWEN_HIDDEN,
                TOKENSKIN_QWEN_INTERMEDIATE) ||
            !qwen_tensor_shape(
                block->down,
                TOKENSKIN_QWEN_INTERMEDIATE,
                TOKENSKIN_QWEN_HIDDEN)) {
            return 0;
        }
    }
    return 1;
}

static struct ggml_tensor * qwen_bf16_boundary(
    struct ggml_context * ctx,
    struct ggml_tensor * tensor) {
    return trellis_ggml_bf16_roundtrip(ctx, tensor);
}

static struct ggml_tensor * qwen_linear(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * weight) {
    return trellis_ggml_linear(ctx, input, weight, NULL);
}

static struct ggml_tensor * qwen_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * weight) {
    /* Qwen3RMSNorm first rounds the unit-RMS value back to the input dtype,
     * then applies its learned BF16 weight.  Multiplying gamma before that
     * round is measurably different across 28 residual blocks. */
    struct ggml_tensor * normalized = ggml_rms_norm(ctx, input, 1.0e-6f);
    normalized = qwen_bf16_boundary(ctx, normalized);
    struct ggml_tensor * gamma = weight;
    if (gamma->type != GGML_TYPE_F32) {
        gamma = ggml_cast(ctx, gamma, GGML_TYPE_F32);
    }
    gamma = ggml_repeat(ctx, gamma, normalized);
    return qwen_bf16_boundary(ctx, ggml_mul(ctx, normalized, gamma));
}

static struct ggml_tensor * qwen_apply_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * positions) {
    struct ggml_tensor * result = ggml_rope_ext(
        ctx,
        input,
        positions,
        NULL,
        TOKENSKIN_QWEN_HEAD_DIM,
        GGML_ROPE_TYPE_NEOX,
        TOKENSKIN_QWEN_MAX_POSITIONS,
        1000000.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f);
    return qwen_bf16_boundary(ctx, result);
}

static struct ggml_tensor * qwen_cache_attention_view(
    struct ggml_context * ctx,
    struct ggml_tensor * cache,
    int n_kv) {
    struct ggml_tensor * view = ggml_view_4d(
        ctx,
        cache,
        TOKENSKIN_QWEN_HEAD_DIM,
        TOKENSKIN_QWEN_KV_HEADS,
        n_kv,
        1,
        ggml_row_size(cache->type, TOKENSKIN_QWEN_HEAD_DIM),
        cache->nb[1],
        cache->nb[2],
        0);
    return ggml_permute(ctx, view, 0, 2, 1, 3);
}

static struct ggml_tensor * qwen_cache_store(
    struct ggml_context * ctx,
    struct ggml_tensor * value,
    struct ggml_tensor * cache,
    int past,
    int n_tokens) {
    value = ggml_cast(ctx, value, GGML_TYPE_BF16);
    value = ggml_reshape_2d(
        ctx,
        value,
        TOKENSKIN_QWEN_KV_CHANNELS,
        n_tokens);
    struct ggml_tensor * destination = ggml_view_2d(
        ctx,
        cache,
        TOKENSKIN_QWEN_KV_CHANNELS,
        n_tokens,
        cache->nb[1],
        (size_t) past * cache->nb[1]);
    return ggml_cpy(ctx, value, destination);
}

static struct ggml_tensor * qwen_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * query,
    struct ggml_tensor * key,
    struct ggml_tensor * value,
    struct ggml_tensor * mask,
    int n_tokens,
    const trellis_ggml_attention_policy * policy) {
    const float scale = 1.0f / sqrtf((float) TOKENSKIN_QWEN_HEAD_DIM);
    struct ggml_tensor * result = NULL;

    if (policy->mode == TRELLIS_GGML_ATTENTION_MODE_FLASH ||
        policy->mode == TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16) {
        result = ggml_flash_attn_ext(
            ctx,
            query,
            key,
            value,
            mask,
            scale,
            0.0f,
            0.0f);
        ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);
        result = ggml_reshape_2d(
            ctx,
            result,
            TOKENSKIN_QWEN_HEAD_DIM * TOKENSKIN_QWEN_HEADS,
            n_tokens);
    } else {
        struct ggml_tensor * scores = ggml_mul_mat(ctx, key, query);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);

        struct ggml_tensor * value_transposed = ggml_permute(
            ctx, value, 1, 0, 2, 3);
        value_transposed = ggml_cont_3d(
            ctx,
            value_transposed,
            value->ne[1],
            value->ne[0],
            value->ne[2]);
        result = ggml_mul_mat(ctx, value_transposed, scores);
        ggml_mul_mat_set_prec(result, GGML_PREC_F32);
        result = ggml_permute(ctx, result, 0, 2, 1, 3);
        result = ggml_cont_2d(
            ctx,
            result,
            TOKENSKIN_QWEN_HEAD_DIM * TOKENSKIN_QWEN_HEADS,
            n_tokens);
    }

    return qwen_bf16_boundary(ctx, result);
}

static void qwen_executor_destroy_cache(tokenskin_qwen_executor * executor) {
    if (executor == NULL) return;
    if (executor->cache_buffer != NULL) {
        ggml_backend_buffer_free(executor->cache_buffer);
        executor->cache_buffer = NULL;
    }
    if (executor->cache_ctx != NULL) {
        ggml_free(executor->cache_ctx);
        executor->cache_ctx = NULL;
    }
}

trellis_status tokenskin_qwen_executor_create(
    const trellis_backend_context * backend,
    const tokenskin_qwen_weights * weights,
    const trellis_ggml_attention_policy * attention_policy,
    int capacity,
    tokenskin_qwen_executor ** executor_out) {
    if (executor_out == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;
    *executor_out = NULL;
    if (backend == NULL || backend->backend == NULL ||
        !qwen_weights_are_valid(weights) || capacity <= 0 ||
        capacity > TOKENSKIN_QWEN_MAX_POSITIONS ||
        (attention_policy != NULL &&
         !trellis_ggml_attention_policy_is_valid(attention_policy))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    tokenskin_qwen_executor * executor =
        (tokenskin_qwen_executor *) calloc(1, sizeof(*executor));
    if (executor == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    executor->backend = backend;
    executor->weights = weights;
    executor->capacity = capacity;
    executor->length = 0;
    executor->attention_policy.struct_size = sizeof(executor->attention_policy);
    executor->attention_policy.mode = TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16;
    if (attention_policy != NULL) {
        executor->attention_policy.mode = attention_policy->mode;
    }

    struct ggml_init_params cache_params = {
        .mem_size = ggml_tensor_overhead() *
            (size_t) (2 * TOKENSKIN_QWEN_LAYERS + 8),
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    executor->cache_ctx = ggml_init(cache_params);
    if (executor->cache_ctx == NULL) {
        free(executor);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    for (int layer = 0; layer < TOKENSKIN_QWEN_LAYERS; ++layer) {
        char name[64];
        executor->cache_k[layer] = ggml_new_tensor_2d(
            executor->cache_ctx,
            GGML_TYPE_BF16,
            TOKENSKIN_QWEN_KV_CHANNELS,
            capacity);
        executor->cache_v[layer] = ggml_new_tensor_2d(
            executor->cache_ctx,
            GGML_TYPE_BF16,
            TOKENSKIN_QWEN_KV_CHANNELS,
            capacity);
        snprintf(name, sizeof(name), "tokenskin_qwen_cache_k_%d", layer);
        ggml_set_name(executor->cache_k[layer], name);
        snprintf(name, sizeof(name), "tokenskin_qwen_cache_v_%d", layer);
        ggml_set_name(executor->cache_v[layer], name);
    }

    executor->cache_buffer = ggml_backend_alloc_ctx_tensors(
        executor->cache_ctx,
        backend->backend);
    if (executor->cache_buffer == NULL) {
        qwen_executor_destroy_cache(executor);
        free(executor);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_backend_buffer_set_usage(
        executor->cache_buffer,
        GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_clear(executor->cache_buffer, 0);
    *executor_out = executor;
    return TRELLIS_STATUS_OK;
}

void tokenskin_qwen_executor_free(tokenskin_qwen_executor * executor) {
    if (executor == NULL) return;
    qwen_executor_destroy_cache(executor);
    free(executor);
}

void tokenskin_qwen_executor_reset(tokenskin_qwen_executor * executor) {
    if (executor == NULL) return;
    /* Values past length are never observed, so resetting the logical cursor is
     * sufficient and avoids clearing hundreds of MiB between generations. */
    executor->length = 0;
}

int tokenskin_qwen_executor_length(const tokenskin_qwen_executor * executor) {
    return executor == NULL ? 0 : executor->length;
}

trellis_status tokenskin_qwen_executor_copy_state(
    tokenskin_qwen_executor * destination,
    const tokenskin_qwen_executor * source) {
    if (destination == NULL || source == NULL || destination == source ||
        destination->backend == NULL || source->backend == NULL ||
        destination->backend->backend == NULL ||
        source->backend->backend == NULL ||
        destination->backend->backend != source->backend->backend ||
        destination->backend->kind != source->backend->kind ||
        destination->backend->device != source->backend->device ||
        source->weights == NULL ||
        destination->weights != source->weights ||
        source->capacity <= 0 ||
        destination->capacity != source->capacity ||
        destination->length < 0 ||
        destination->length > destination->capacity ||
        source->length < 0 || source->length > source->capacity ||
        (source->backend->kind != TRELLIS_BACKEND_CUDA &&
         source->backend->kind != TRELLIS_BACKEND_VULKAN)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    if (source->length == 0) {
        destination->length = 0;
        return TRELLIS_STATUS_OK;
    }

    /* A cache row is one complete 8-head K or V token.  Since rows are
     * contiguous, these views cover exactly [1024, source->length] and do not
     * copy the unused tail of the executor capacity. */
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() *
            (size_t) (4 * TOKENSKIN_QWEN_LAYERS + 8),
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;

    struct ggml_tensor * source_k[TOKENSKIN_QWEN_LAYERS];
    struct ggml_tensor * source_v[TOKENSKIN_QWEN_LAYERS];
    struct ggml_tensor * destination_k[TOKENSKIN_QWEN_LAYERS];
    struct ggml_tensor * destination_v[TOKENSKIN_QWEN_LAYERS];
    trellis_status status = TRELLIS_STATUS_OK;

    /* Initialize every backend view before submitting a copy.  Consequently,
     * a recoverable validation/allocation error cannot leave a partially
     * copied cache that appears committed through destination->length. */
    for (int layer = 0; layer < TOKENSKIN_QWEN_LAYERS; ++layer) {
        struct ggml_tensor * src_cache[2] = {
            source->cache_k[layer], source->cache_v[layer],
        };
        struct ggml_tensor * dst_cache[2] = {
            destination->cache_k[layer], destination->cache_v[layer],
        };
        struct ggml_tensor ** src_view[2] = {
            &source_k[layer], &source_v[layer],
        };
        struct ggml_tensor ** dst_view[2] = {
            &destination_k[layer], &destination_v[layer],
        };
        for (int kv = 0; kv < 2; ++kv) {
            if (src_cache[kv] == NULL || dst_cache[kv] == NULL ||
                src_cache[kv]->buffer == NULL ||
                dst_cache[kv]->buffer == NULL ||
                src_cache[kv]->type != GGML_TYPE_BF16 ||
                dst_cache[kv]->type != GGML_TYPE_BF16 ||
                src_cache[kv]->ne[0] != TOKENSKIN_QWEN_KV_CHANNELS ||
                dst_cache[kv]->ne[0] != TOKENSKIN_QWEN_KV_CHANNELS ||
                src_cache[kv]->ne[1] != source->capacity ||
                dst_cache[kv]->ne[1] != destination->capacity ||
                ggml_backend_buffer_get_type(src_cache[kv]->buffer) !=
                    ggml_backend_buffer_get_type(dst_cache[kv]->buffer)) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
                goto cleanup;
            }
            *src_view[kv] = ggml_view_2d(
                ctx,
                src_cache[kv],
                TOKENSKIN_QWEN_KV_CHANNELS,
                source->length,
                src_cache[kv]->nb[1],
                0);
            *dst_view[kv] = ggml_view_2d(
                ctx,
                dst_cache[kv],
                TOKENSKIN_QWEN_KV_CHANNELS,
                source->length,
                dst_cache[kv]->nb[1],
                0);
            if (ggml_backend_view_init(*src_view[kv]) != GGML_STATUS_SUCCESS ||
                ggml_backend_view_init(*dst_view[kv]) != GGML_STATUS_SUCCESS) {
                status = TRELLIS_STATUS_ERROR;
                goto cleanup;
            }
        }
    }

    /* CUDA and Vulkan both implement this same-backend path as a direct
     * device-buffer copy.  The public ggml operation has no status return, but
     * it orders the copies after preceding work; one synchronization below
     * commits all 56 views before exposing the new logical length. */
    for (int layer = 0; layer < TOKENSKIN_QWEN_LAYERS; ++layer) {
        ggml_backend_tensor_copy_async(
            source->backend->backend,
            destination->backend->backend,
            source_k[layer],
            destination_k[layer]);
        ggml_backend_tensor_copy_async(
            source->backend->backend,
            destination->backend->backend,
            source_v[layer],
            destination_v[layer]);
    }
    ggml_backend_synchronize(destination->backend->backend);
    destination->length = source->length;

cleanup:
    ggml_free(ctx);
    return status;
}

static trellis_status qwen_execute(
    tokenskin_qwen_executor * executor,
    const float * embeddings,
    int token_id,
    int n_tokens,
    float * logits_out) {
    if (executor == NULL || executor->backend == NULL ||
        executor->backend->backend == NULL || logits_out == NULL ||
        n_tokens <= 0 || executor->length < 0 ||
        n_tokens > executor->capacity - executor->length ||
        (embeddings == NULL &&
         (n_tokens != 1 || token_id < 0 || token_id >= TOKENSKIN_QWEN_VOCAB))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = TRELLIS_STATUS_ERROR;
    struct ggml_context * ctx = NULL;
    ggml_gallocr_t allocator = NULL;
    int32_t * host_positions = NULL;
    float * host_mask = NULL;

    const int past = executor->length;
    const int n_kv = past + n_tokens;
    const size_t graph_mem =
        ggml_tensor_overhead() * (size_t) TOKENSKIN_QWEN_GRAPH_NODES +
        ggml_graph_overhead_custom(TOKENSKIN_QWEN_GRAPH_NODES, false) +
        4 * 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size = graph_mem,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    ctx = ggml_init(params);
    if (ctx == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;

    struct ggml_cgraph * graph = ggml_new_graph_custom(
        ctx, TOKENSKIN_QWEN_GRAPH_NODES, false);
    if (graph == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    struct ggml_tensor * input = NULL;
    struct ggml_tensor * input_ids = NULL;
    if (embeddings != NULL) {
        input = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, TOKENSKIN_QWEN_HIDDEN, n_tokens);
        ggml_set_name(input, "tokenskin_qwen_embeddings");
        ggml_set_input(input);
    } else {
        input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(input_ids, "tokenskin_qwen_token_id");
        ggml_set_input(input_ids);
        input = ggml_get_rows(ctx, executor->weights->token_embedding, input_ids);
    }
    struct ggml_tensor * positions = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_kv, n_tokens);
    ggml_set_name(positions, "tokenskin_qwen_positions");
    ggml_set_name(mask, "tokenskin_qwen_causal_mask");
    ggml_set_input(positions);
    ggml_set_input(mask);
    struct ggml_tensor * attention_mask = mask;
    if (executor->attention_policy.mode == TRELLIS_GGML_ATTENTION_MODE_FLASH ||
        executor->attention_policy.mode == TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16) {
        /* All layers share the same causal mask.  Narrow it once instead of
         * inserting 28 identical mask casts into every decode graph. */
        attention_mask = ggml_cast(ctx, mask, GGML_TYPE_F16);
    }

    struct ggml_tensor * hidden = qwen_bf16_boundary(ctx, input);
    for (int layer = 0; layer < TOKENSKIN_QWEN_LAYERS; ++layer) {
        const tokenskin_qwen_block_weights * block =
            &executor->weights->blocks[layer];
        struct ggml_tensor * residual = hidden;
        struct ggml_tensor * normalized = qwen_rms_norm(
            ctx, hidden, block->input_norm);

        struct ggml_tensor * query = qwen_bf16_boundary(
            ctx, qwen_linear(ctx, normalized, block->q));
        struct ggml_tensor * key = qwen_bf16_boundary(
            ctx, qwen_linear(ctx, normalized, block->k));
        struct ggml_tensor * value = qwen_bf16_boundary(
            ctx, qwen_linear(ctx, normalized, block->v));

        query = ggml_reshape_3d(
            ctx,
            query,
            TOKENSKIN_QWEN_HEAD_DIM,
            TOKENSKIN_QWEN_HEADS,
            n_tokens);
        key = ggml_reshape_3d(
            ctx,
            key,
            TOKENSKIN_QWEN_HEAD_DIM,
            TOKENSKIN_QWEN_KV_HEADS,
            n_tokens);
        value = ggml_reshape_3d(
            ctx,
            value,
            TOKENSKIN_QWEN_HEAD_DIM,
            TOKENSKIN_QWEN_KV_HEADS,
            n_tokens);

        query = qwen_rms_norm(ctx, query, block->q_norm);
        key = qwen_rms_norm(ctx, key, block->k_norm);
        query = qwen_apply_rope(ctx, query, positions);
        key = qwen_apply_rope(ctx, key, positions);

        /* Expand cache writes before the attention node.  The cache read and
         * write are views of the same persistent tensor and therefore have no
         * ordinary ggml data-flow edge; graph order supplies that dependency,
         * matching llama.cpp's KV-cache construction. */
        ggml_build_forward_expand(graph, query);
        ggml_build_forward_expand(graph, value);
        ggml_build_forward_expand(graph, key);
        struct ggml_tensor * key_copy = qwen_cache_store(
            ctx,
            key,
            executor->cache_k[layer],
            past,
            n_tokens);
        struct ggml_tensor * value_copy = qwen_cache_store(
            ctx,
            value,
            executor->cache_v[layer],
            past,
            n_tokens);
        ggml_build_forward_expand(graph, key_copy);
        ggml_build_forward_expand(graph, value_copy);

        query = ggml_permute(ctx, query, 0, 2, 1, 3);
        query = ggml_cont_3d(
            ctx,
            query,
            TOKENSKIN_QWEN_HEAD_DIM,
            n_tokens,
            TOKENSKIN_QWEN_HEADS);
        struct ggml_tensor * cached_key = qwen_cache_attention_view(
            ctx, executor->cache_k[layer], n_kv);
        struct ggml_tensor * cached_value = qwen_cache_attention_view(
            ctx, executor->cache_v[layer], n_kv);
        struct ggml_tensor * attention = qwen_attention(
            ctx,
            query,
            cached_key,
            cached_value,
            attention_mask,
            n_tokens,
            &executor->attention_policy);
        attention = qwen_bf16_boundary(
            ctx, qwen_linear(ctx, attention, block->out));
        hidden = qwen_bf16_boundary(ctx, ggml_add(ctx, residual, attention));

        residual = hidden;
        normalized = qwen_rms_norm(
            ctx, hidden, block->post_attention_norm);
        struct ggml_tensor * gate = qwen_bf16_boundary(
            ctx, qwen_linear(ctx, normalized, block->gate));
        struct ggml_tensor * up = qwen_bf16_boundary(
            ctx, qwen_linear(ctx, normalized, block->up));
        gate = qwen_bf16_boundary(ctx, ggml_silu(ctx, gate));
        struct ggml_tensor * activated = qwen_bf16_boundary(
            ctx, ggml_mul(ctx, gate, up));
        struct ggml_tensor * down = qwen_bf16_boundary(
            ctx, qwen_linear(ctx, activated, block->down));
        hidden = qwen_bf16_boundary(ctx, ggml_add(ctx, residual, down));
    }

    hidden = qwen_rms_norm(ctx, hidden, executor->weights->output_norm);
    struct ggml_tensor * last_hidden = ggml_view_2d(
        ctx,
        hidden,
        TOKENSKIN_QWEN_HIDDEN,
        1,
        hidden->nb[1],
        (size_t) (n_tokens - 1) * hidden->nb[1]);
    struct ggml_tensor * logits = ggml_mul_mat(
        ctx, executor->weights->token_embedding, last_hidden);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    logits = qwen_bf16_boundary(ctx, logits);
    ggml_set_name(logits, "tokenskin_qwen_logits");
    ggml_build_forward_expand(graph, logits);

    allocator = trellis_backend_new_graph_allocator(executor->backend);
    if (allocator == NULL || !ggml_gallocr_alloc_graph(allocator, graph)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    host_positions = (int32_t *) malloc((size_t) n_tokens * sizeof(*host_positions));
    host_mask = (float *) malloc(
        (size_t) n_kv * (size_t) n_tokens * sizeof(*host_mask));
    if (host_positions == NULL || host_mask == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (int token = 0; token < n_tokens; ++token) {
        const int absolute_position = past + token;
        host_positions[token] = absolute_position;
        for (int key_position = 0; key_position < n_kv; ++key_position) {
            host_mask[(size_t) token * (size_t) n_kv + (size_t) key_position] =
                key_position <= absolute_position ? 0.0f : -INFINITY;
        }
    }

    if (embeddings != NULL) {
        ggml_backend_tensor_set(input, embeddings, 0, ggml_nbytes(input));
    } else {
        const int32_t id = token_id;
        ggml_backend_tensor_set(input_ids, &id, 0, sizeof(id));
    }
    ggml_backend_tensor_set(
        positions, host_positions, 0, ggml_nbytes(positions));
    ggml_backend_tensor_set(mask, host_mask, 0, ggml_nbytes(mask));

    status = trellis_backend_compute_graph(executor->backend, graph);
    if (status == TRELLIS_STATUS_OK) {
        ggml_backend_tensor_get(
            logits,
            logits_out,
            0,
            (size_t) TOKENSKIN_QWEN_VOCAB * sizeof(*logits_out));
        executor->length = n_kv;
    }

cleanup:
    free(host_mask);
    free(host_positions);
    if (allocator != NULL) ggml_gallocr_free(allocator);
    if (ctx != NULL) ggml_free(ctx);
    return status;
}

trellis_status tokenskin_qwen_prefill(
    tokenskin_qwen_executor * executor,
    const float * embeddings_token_major_896,
    int n_tokens,
    float * logits_33036) {
    if (embeddings_token_major_896 == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return qwen_execute(
        executor,
        embeddings_token_major_896,
        -1,
        n_tokens,
        logits_33036);
}

trellis_status tokenskin_qwen_decode(
    tokenskin_qwen_executor * executor,
    int token_id,
    float * logits_33036) {
    return qwen_execute(executor, NULL, token_id, 1, logits_33036);
}
