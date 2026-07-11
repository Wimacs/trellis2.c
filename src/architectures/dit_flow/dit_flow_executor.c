#include "dit_flow_executor.h"
#include "trellis_ggml_layers.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TRELLIS_CFG_BATCH_DEFAULT_MAX_ATTENTION_MIB 8192ull

static trellis_ggml_attention_policy legacy_attention_policy_snapshot(void) {
    trellis_ggml_attention_policy policy = TRELLIS_GGML_ATTENTION_POLICY_INIT;
    if (trellis_ggml_flash_attn_enabled()) {
        policy.mode = TRELLIS_GGML_ATTENTION_MODE_FLASH;
    }
    return policy;
}

static int checked_mul_size(size_t a, size_t b, size_t * out) {
    if (out == NULL) {
        return 0;
    }
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static size_t cfg_batch_attention_limit_bytes(void) {
    const char * env = getenv("TRELLIS_CFG_BATCH_MAX_ATTENTION_MIB");
    unsigned long long mib = TRELLIS_CFG_BATCH_DEFAULT_MAX_ATTENTION_MIB;
    if (env != NULL && env[0] != '\0') {
        char * end = NULL;
        unsigned long long parsed = strtoull(env, &end, 10);
        if (end != env && *end == '\0') {
            mib = parsed;
        }
    }
    if (mib == 0) {
        return 0;
    }
    const unsigned long long max_mib = (unsigned long long) (SIZE_MAX / (1024u * 1024u));
    if (mib > max_mib) {
        mib = max_mib;
    }
    return (size_t) mib * 1024u * 1024u;
}

static int checked_attention_score_bytes(
    int64_t tokens,
    int cond_tokens,
    int heads,
    int batch,
    size_t * bytes_out) {
    size_t self = 0;
    size_t cross = 0;
    size_t tmp = 0;
    if (tokens <= 0 || cond_tokens <= 0 || heads <= 0 || batch <= 0 || bytes_out == NULL) {
        return 0;
    }
    if (!checked_mul_size((size_t) tokens, (size_t) tokens, &tmp) ||
        !checked_mul_size(tmp, (size_t) heads, &tmp) ||
        !checked_mul_size(tmp, (size_t) batch, &tmp) ||
        !checked_mul_size(tmp, sizeof(float), &self)) {
        return 0;
    }
    if (!checked_mul_size((size_t) tokens, (size_t) cond_tokens, &tmp) ||
        !checked_mul_size(tmp, (size_t) heads, &tmp) ||
        !checked_mul_size(tmp, (size_t) batch, &tmp) ||
        !checked_mul_size(tmp, sizeof(float), &cross)) {
        return 0;
    }
    *bytes_out = self > cross ? self : cross;
    return 1;
}

static int cfg_batch_attention_is_reasonable(
    const trellis_dit_flow_weights * weights,
    int64_t tokens,
    int cond_tokens,
    int batch,
    const trellis_ggml_attention_policy * attention_policy) {
    if (weights == NULL ||
        !trellis_ggml_attention_policy_is_valid(attention_policy)) {
        return 0;
    }
    const int debug_parts = weights->debug_block_parts < 0 ? 3 : weights->debug_block_parts;
    if (weights->n_blocks <= 0 || debug_parts <= 0) {
        return 1;
    }
    if (attention_policy->mode == TRELLIS_GGML_ATTENTION_MODE_FLASH ||
        attention_policy->mode == TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16) {
        return 1;
    }
    size_t estimate = 0;
    if (!checked_attention_score_bytes(tokens, cond_tokens, weights->heads, batch, &estimate)) {
        TRELLIS_WARN("DiT CFG batch: attention scratch estimate overflow; using separate cond/uncond graphs");
        return 0;
    }
    const size_t limit = cfg_batch_attention_limit_bytes();
    if (limit != 0 && estimate > limit) {
        TRELLIS_WARN(
            "DiT CFG batch: estimated explicit-attention scratch %.1f MiB exceeds limit %.1f MiB; using separate cond/uncond graphs",
            (double) estimate / (1024.0 * 1024.0),
            (double) limit / (1024.0 * 1024.0));
        return 0;
    }
    return 1;
}

static int should_set_rope(const trellis_dit_flow_weights * weights) {
    if (weights == NULL || weights->n_blocks <= 0 || weights->debug_disable_rope) {
        return 0;
    }
    const int debug_parts = weights->debug_block_parts < 0 ? 3 : weights->debug_block_parts;
    return debug_parts >= 1;
}

static int should_set_context(const trellis_dit_flow_weights * weights) {
    if (weights == NULL || weights->n_blocks <= 0) {
        return 0;
    }
    const int debug_parts = weights->debug_block_parts < 0 ? 3 : weights->debug_block_parts;
    return debug_parts >= 2;
}

static int should_set_projection(const trellis_dit_flow_executor * executor) {
    if (executor == NULL || executor->model == NULL ||
        !executor->model->projection.enabled || executor->weights == NULL ||
        executor->weights->n_blocks <= 0) {
        return 0;
    }
    const int debug_parts = executor->weights->debug_block_parts < 0 ?
        3 : executor->weights->debug_block_parts;
    return debug_parts >= 2;
}

static int should_set_timestep(const trellis_dit_flow_weights * weights) {
    return weights != NULL && weights->n_blocks > 0;
}

static int set_static_inputs(
    trellis_dit_flow_executor * executor,
    const float * context,
    const float * neg_context,
    const float * projected_context,
    const float * neg_projected_context,
    const float * cos_phase_data,
    const float * sin_phase_data) {
    const trellis_dit_flow_weights * weights = executor->weights;
    if (should_set_rope(weights)) {
        if (cos_phase_data == NULL || sin_phase_data == NULL) {
            return 0;
        }
        ggml_backend_tensor_set(executor->cos_phase, cos_phase_data, 0, ggml_nbytes(executor->cos_phase));
        ggml_backend_tensor_set(executor->sin_phase, sin_phase_data, 0, ggml_nbytes(executor->sin_phase));
    }
    if (should_set_context(weights)) {
        if (context == NULL || executor->context_host == NULL) {
            return 0;
        }
        memcpy(
            executor->context_host,
            context,
            executor->single_context_count * sizeof(float));
        if (executor->batch > 1) {
            if (neg_context == NULL) {
                return 0;
            }
            memcpy(
                executor->context_host + executor->single_context_count,
                neg_context,
                executor->single_context_count * sizeof(float));
        }
        ggml_backend_tensor_set(
            executor->c,
            executor->context_host,
            0,
            ggml_nbytes(executor->c));
    }
    if (should_set_projection(executor)) {
        if (projected_context == NULL || executor->projected_host == NULL || executor->p == NULL) {
            return 0;
        }
        memcpy(
            executor->projected_host,
            projected_context,
            executor->single_projected_count * sizeof(float));
        if (executor->batch > 1) {
            if (neg_projected_context == NULL) {
                return 0;
            }
            memcpy(
                executor->projected_host + executor->single_projected_count,
                neg_projected_context,
                executor->single_projected_count * sizeof(float));
        }
        ggml_backend_tensor_set(
            executor->p,
            executor->projected_host,
            0,
            ggml_nbytes(executor->p));
    }
    return 1;
}

static trellis_status init_executor(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_weights * weights,
    const trellis_dit_flow_model * model,
    int64_t tokens,
    int cond_tokens,
    int batch,
    const float * context,
    const float * neg_context,
    const float * projected_context,
    const float * neg_projected_context,
    const float * cos_phase_data,
    const float * sin_phase_data,
    const trellis_ggml_attention_policy * attention_policy) {
    if (model != NULL) {
        weights = &model->base;
    }
    if (executor == NULL || backend == NULL || backend->backend == NULL || weights == NULL ||
        tokens <= 0 || cond_tokens <= 0 || weights->in_channels <= 0 ||
        weights->cond_channels <= 0 || weights->head_dim <= 0 || batch <= 0 ||
        !trellis_ggml_attention_policy_is_valid(attention_policy)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int projected_mode = model != NULL && model->projection.enabled;
    if (projected_mode &&
        (model->projection.proj_channels <= 0 || projected_context == NULL ||
         (batch > 1 && neg_projected_context == NULL))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    memset(executor, 0, sizeof(*executor));
    executor->backend = backend;
    executor->weights = weights;
    executor->model = model;
    executor->tokens = tokens;
    executor->cond_tokens = cond_tokens;
    executor->batch = batch;
    executor->attention_policy.struct_size = sizeof(executor->attention_policy);
    executor->attention_policy.mode = attention_policy->mode;

    if (!checked_mul_size((size_t) tokens, (size_t) weights->in_channels, &executor->single_input_count) ||
        !checked_mul_size((size_t) tokens, (size_t) weights->out_channels, &executor->single_output_count) ||
        !checked_mul_size((size_t) cond_tokens, (size_t) weights->cond_channels, &executor->single_context_count) ||
        !checked_mul_size(executor->single_input_count, (size_t) batch, &executor->input_count) ||
        !checked_mul_size(executor->single_output_count, (size_t) batch, &executor->output_count) ||
        !checked_mul_size(executor->single_context_count, (size_t) batch, &executor->context_count)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    if (projected_mode &&
        (!checked_mul_size(
             (size_t) tokens,
             (size_t) model->projection.proj_channels,
             &executor->single_projected_count) ||
         !checked_mul_size(
             executor->single_projected_count,
             (size_t) batch,
             &executor->projected_count))) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    executor->x_host = (float *) malloc(executor->input_count * sizeof(float));
    executor->y_host = (float *) malloc(executor->output_count * sizeof(float));
    executor->context_host = (float *) malloc(executor->context_count * sizeof(float));
    if (projected_mode) {
        executor->projected_host = (float *) malloc(executor->projected_count * sizeof(float));
    }
    if (executor->x_host == NULL || executor->y_host == NULL ||
        executor->context_host == NULL || (projected_mode && executor->projected_host == NULL)) {
        trellis_dit_flow_executor_free(executor);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    const size_t graph_nodes = 65536;
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * graph_nodes +
            ggml_graph_overhead_custom(graph_nodes, false) + 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    executor->ctx = ggml_init(params);
    if (executor->ctx == NULL) {
        trellis_dit_flow_executor_free(executor);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    executor->x = ggml_new_tensor_3d(
        executor->ctx,
        GGML_TYPE_F32,
        weights->in_channels,
        tokens,
        batch);
    executor->t = ggml_new_tensor_1d(executor->ctx, GGML_TYPE_F32, batch);
    executor->c = ggml_new_tensor_3d(
        executor->ctx,
        GGML_TYPE_F32,
        weights->cond_channels,
        cond_tokens,
        batch);
    if (projected_mode) {
        executor->p = ggml_new_tensor_3d(
            executor->ctx,
            GGML_TYPE_F32,
            model->projection.proj_channels,
            tokens,
            batch);
    }
    executor->cos_phase = ggml_new_tensor_4d(
        executor->ctx,
        GGML_TYPE_F32,
        1,
        weights->head_dim / 2,
        tokens,
        1);
    executor->sin_phase = ggml_new_tensor_4d(
        executor->ctx,
        GGML_TYPE_F32,
        1,
        weights->head_dim / 2,
        tokens,
        1);
    if (executor->x == NULL || executor->t == NULL || executor->c == NULL ||
        (projected_mode && executor->p == NULL) ||
        executor->cos_phase == NULL || executor->sin_phase == NULL) {
        trellis_dit_flow_executor_free(executor);
        return TRELLIS_STATUS_ERROR;
    }
    ggml_set_input(executor->x);
    ggml_set_input(executor->t);
    ggml_set_input(executor->c);
    if (executor->p != NULL) {
        ggml_set_input(executor->p);
        ggml_set_output(executor->p);
    }
    ggml_set_input(executor->cos_phase);
    ggml_set_input(executor->sin_phase);
    ggml_set_output(executor->c);
    ggml_set_output(executor->cos_phase);
    ggml_set_output(executor->sin_phase);

    if (projected_mode) {
        executor->y = trellis_dit_flow_forward_projected_with_policy(
            executor->ctx,
            executor->x,
            executor->t,
            executor->c,
            executor->p,
            executor->cos_phase,
            executor->sin_phase,
            model,
            &executor->attention_policy);
    } else {
        executor->y = trellis_dit_flow_forward_with_policy(
            executor->ctx,
            executor->x,
            executor->t,
            executor->c,
            executor->cos_phase,
            executor->sin_phase,
            weights,
            &executor->attention_policy);
    }
    if (executor->y == NULL) {
        trellis_dit_flow_executor_free(executor);
        return TRELLIS_STATUS_ERROR;
    }
    ggml_set_output(executor->y);

    executor->graph = ggml_new_graph_custom(executor->ctx, graph_nodes, false);
    if (executor->graph == NULL) {
        trellis_dit_flow_executor_free(executor);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_build_forward_expand(executor->graph, executor->y);

    executor->alloc = trellis_backend_new_graph_allocator(backend);
    if (executor->alloc == NULL || !ggml_gallocr_alloc_graph(executor->alloc, executor->graph)) {
        trellis_dit_flow_executor_free(executor);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    if (!set_static_inputs(
            executor,
            context,
            neg_context,
            projected_context,
            neg_projected_context,
            cos_phase_data,
            sin_phase_data)) {
        trellis_dit_flow_executor_free(executor);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    return TRELLIS_STATUS_OK;
}

trellis_status trellis_dit_flow_executor_init_single(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_weights * weights,
    int64_t tokens,
    int cond_tokens,
    const float * context,
    const float * cos_phase_data,
    const float * sin_phase_data) {
    const trellis_ggml_attention_policy attention_policy =
        legacy_attention_policy_snapshot();
    return trellis_dit_flow_executor_init_single_with_policy(
        executor,
        backend,
        weights,
        tokens,
        cond_tokens,
        context,
        cos_phase_data,
        sin_phase_data,
        &attention_policy);
}

trellis_status trellis_dit_flow_executor_init_single_with_policy(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_weights * weights,
    int64_t tokens,
    int cond_tokens,
    const float * context,
    const float * cos_phase_data,
    const float * sin_phase_data,
    const trellis_ggml_attention_policy * attention_policy) {
    return init_executor(
        executor,
        backend,
        weights,
        NULL,
        tokens,
        cond_tokens,
        1,
        context,
        NULL,
        NULL,
        NULL,
        cos_phase_data,
        sin_phase_data,
        attention_policy);
}

trellis_status trellis_dit_flow_executor_init_single_projected(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_model * model,
    int64_t tokens,
    int global_tokens,
    const float * global_context,
    const float * projected_context,
    const float * cos_phase_data,
    const float * sin_phase_data) {
    const trellis_ggml_attention_policy attention_policy =
        legacy_attention_policy_snapshot();
    return trellis_dit_flow_executor_init_single_projected_with_policy(
        executor,
        backend,
        model,
        tokens,
        global_tokens,
        global_context,
        projected_context,
        cos_phase_data,
        sin_phase_data,
        &attention_policy);
}

trellis_status trellis_dit_flow_executor_init_single_projected_with_policy(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_model * model,
    int64_t tokens,
    int global_tokens,
    const float * global_context,
    const float * projected_context,
    const float * cos_phase_data,
    const float * sin_phase_data,
    const trellis_ggml_attention_policy * attention_policy) {
    if (model == NULL || !model->projection.enabled) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return init_executor(
        executor,
        backend,
        &model->base,
        model,
        tokens,
        global_tokens,
        1,
        global_context,
        NULL,
        projected_context,
        NULL,
        cos_phase_data,
        sin_phase_data,
        attention_policy);
}

trellis_status trellis_dit_flow_executor_run_single(
    trellis_dit_flow_executor * executor,
    const float * latent,
    float timestep,
    float * pred) {
    if (executor == NULL || executor->backend == NULL || executor->weights == NULL ||
        executor->graph == NULL || executor->x == NULL || executor->y == NULL ||
        executor->x_host == NULL || executor->y_host == NULL ||
        executor->batch != 1 || latent == NULL || pred == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    memcpy(executor->x_host, latent, executor->input_count * sizeof(float));
    ggml_backend_tensor_set(executor->x, executor->x_host, 0, ggml_nbytes(executor->x));

    if (should_set_timestep(executor->weights)) {
        const float model_timestep = 1000.0f * timestep;
        ggml_backend_tensor_set(executor->t, &model_timestep, 0, ggml_nbytes(executor->t));
    }

    trellis_status status = trellis_backend_compute_graph(executor->backend, executor->graph);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    ggml_backend_tensor_get(executor->y, executor->y_host, 0, ggml_nbytes(executor->y));
    memcpy(pred, executor->y_host, executor->output_count * sizeof(float));
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_dit_flow_executor_init_cfg_batch(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_weights * weights,
    int64_t tokens,
    int cond_tokens,
    const float * context,
    const float * neg_context,
    const float * cos_phase_data,
    const float * sin_phase_data) {
    const trellis_ggml_attention_policy attention_policy =
        legacy_attention_policy_snapshot();
    return trellis_dit_flow_executor_init_cfg_batch_with_policy(
        executor,
        backend,
        weights,
        tokens,
        cond_tokens,
        context,
        neg_context,
        cos_phase_data,
        sin_phase_data,
        &attention_policy);
}

trellis_status trellis_dit_flow_executor_init_cfg_batch_with_policy(
    trellis_dit_flow_executor * executor,
    const trellis_backend_context * backend,
    const trellis_dit_flow_weights * weights,
    int64_t tokens,
    int cond_tokens,
    const float * context,
    const float * neg_context,
    const float * cos_phase_data,
    const float * sin_phase_data,
    const trellis_ggml_attention_policy * attention_policy) {
    if (!trellis_ggml_attention_policy_is_valid(attention_policy)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!cfg_batch_attention_is_reasonable(
            weights,
            tokens,
            cond_tokens,
            2,
            attention_policy)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    return init_executor(
        executor,
        backend,
        weights,
        NULL,
        tokens,
        cond_tokens,
        2,
        context,
        neg_context,
        NULL,
        NULL,
        cos_phase_data,
        sin_phase_data,
        attention_policy);
}

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
    const float * cos_phase_data,
    const float * sin_phase_data) {
    const trellis_ggml_attention_policy attention_policy =
        legacy_attention_policy_snapshot();
    return trellis_dit_flow_executor_init_cfg_batch_projected_with_policy(
        executor,
        backend,
        model,
        tokens,
        global_tokens,
        global_context,
        neg_global_context,
        projected_context,
        neg_projected_context,
        cos_phase_data,
        sin_phase_data,
        &attention_policy);
}

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
    const float * cos_phase_data,
    const float * sin_phase_data,
    const trellis_ggml_attention_policy * attention_policy) {
    if (model == NULL || !model->projection.enabled ||
        !trellis_ggml_attention_policy_is_valid(attention_policy) ||
        !cfg_batch_attention_is_reasonable(
            &model->base,
            tokens,
            global_tokens,
            2,
            attention_policy)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return init_executor(
        executor,
        backend,
        &model->base,
        model,
        tokens,
        global_tokens,
        2,
        global_context,
        neg_global_context,
        projected_context,
        neg_projected_context,
        cos_phase_data,
        sin_phase_data,
        attention_policy);
}

trellis_status trellis_dit_flow_executor_run_cfg_batch(
    trellis_dit_flow_executor * executor,
    const float * latent,
    float timestep,
    float * pred_pos,
    float * pred_neg) {
    if (executor == NULL || executor->backend == NULL || executor->weights == NULL ||
        executor->graph == NULL || executor->x == NULL || executor->y == NULL ||
        executor->x_host == NULL || executor->y_host == NULL || executor->batch != 2 ||
        latent == NULL || pred_pos == NULL || pred_neg == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    memcpy(executor->x_host, latent, executor->single_input_count * sizeof(float));
    memcpy(
        executor->x_host + executor->single_input_count,
        latent,
        executor->single_input_count * sizeof(float));
    ggml_backend_tensor_set(executor->x, executor->x_host, 0, ggml_nbytes(executor->x));

    if (should_set_timestep(executor->weights)) {
        const float model_timestep[2] = {
            1000.0f * timestep,
            1000.0f * timestep,
        };
        ggml_backend_tensor_set(executor->t, model_timestep, 0, ggml_nbytes(executor->t));
    }

    trellis_status status = trellis_backend_compute_graph(executor->backend, executor->graph);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    ggml_backend_tensor_get(executor->y, executor->y_host, 0, ggml_nbytes(executor->y));
    memcpy(pred_pos, executor->y_host, executor->single_output_count * sizeof(float));
    memcpy(
        pred_neg,
        executor->y_host + executor->single_output_count,
        executor->single_output_count * sizeof(float));
    return TRELLIS_STATUS_OK;
}

void trellis_dit_flow_executor_free(trellis_dit_flow_executor * executor) {
    if (executor == NULL) {
        return;
    }
    if (executor->alloc != NULL) {
        ggml_gallocr_free(executor->alloc);
    }
    if (executor->ctx != NULL) {
        ggml_free(executor->ctx);
    }
    free(executor->x_host);
    free(executor->context_host);
    free(executor->projected_host);
    free(executor->y_host);
    memset(executor, 0, sizeof(*executor));
}
