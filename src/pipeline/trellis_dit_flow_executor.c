#include "trellis_dit_flow_executor.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

static int should_set_timestep(const trellis_dit_flow_weights * weights) {
    return weights != NULL && weights->n_blocks > 0;
}

static int set_static_inputs(
    trellis_dit_flow_executor * executor,
    const float * context,
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
            executor->context_count * sizeof(float));
        ggml_backend_tensor_set(
            executor->c,
            executor->context_host,
            0,
            ggml_nbytes(executor->c));
    }
    return 1;
}

static trellis_status init_executor(
    trellis_dit_flow_executor * executor,
    const trellis_cuda_context * cuda,
    const trellis_dit_flow_weights * weights,
    int64_t tokens,
    int cond_tokens,
    const float * context,
    const float * cos_phase_data,
    const float * sin_phase_data) {
    if (executor == NULL || cuda == NULL || cuda->backend == NULL || weights == NULL ||
        tokens <= 0 || cond_tokens <= 0 || weights->in_channels <= 0 ||
        weights->cond_channels <= 0 || weights->head_dim <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    memset(executor, 0, sizeof(*executor));
    executor->cuda = cuda;
    executor->weights = weights;
    executor->tokens = tokens;
    executor->cond_tokens = cond_tokens;

    if (!checked_mul_size((size_t) tokens, (size_t) weights->in_channels, &executor->input_count) ||
        !checked_mul_size((size_t) tokens, (size_t) weights->out_channels, &executor->output_count) ||
        !checked_mul_size((size_t) cond_tokens, (size_t) weights->cond_channels, &executor->context_count)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    executor->x_host = (float *) malloc(executor->input_count * sizeof(float));
    executor->y_host = (float *) malloc(executor->output_count * sizeof(float));
    executor->context_host = (float *) malloc(executor->context_count * sizeof(float));
    if (executor->x_host == NULL || executor->y_host == NULL ||
        executor->context_host == NULL) {
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
        1);
    executor->t = ggml_new_tensor_1d(executor->ctx, GGML_TYPE_F32, 1);
    executor->c = ggml_new_tensor_3d(
        executor->ctx,
        GGML_TYPE_F32,
        weights->cond_channels,
        cond_tokens,
        1);
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
        executor->cos_phase == NULL || executor->sin_phase == NULL) {
        trellis_dit_flow_executor_free(executor);
        return TRELLIS_STATUS_ERROR;
    }
    ggml_set_input(executor->x);
    ggml_set_input(executor->t);
    ggml_set_input(executor->c);
    ggml_set_input(executor->cos_phase);
    ggml_set_input(executor->sin_phase);
    ggml_set_output(executor->c);
    ggml_set_output(executor->cos_phase);
    ggml_set_output(executor->sin_phase);

    executor->y = trellis_dit_flow_forward(
        executor->ctx,
        executor->x,
        executor->t,
        executor->c,
        executor->cos_phase,
        executor->sin_phase,
        weights);
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

    executor->alloc = trellis_cuda_new_graph_allocator(cuda);
    if (executor->alloc == NULL || !ggml_gallocr_alloc_graph(executor->alloc, executor->graph)) {
        trellis_dit_flow_executor_free(executor);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    if (!set_static_inputs(executor, context, cos_phase_data, sin_phase_data)) {
        trellis_dit_flow_executor_free(executor);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    return TRELLIS_STATUS_OK;
}

trellis_status trellis_dit_flow_executor_init_single(
    trellis_dit_flow_executor * executor,
    const trellis_cuda_context * cuda,
    const trellis_dit_flow_weights * weights,
    int64_t tokens,
    int cond_tokens,
    const float * context,
    const float * cos_phase_data,
    const float * sin_phase_data) {
    return init_executor(
        executor,
        cuda,
        weights,
        tokens,
        cond_tokens,
        context,
        cos_phase_data,
        sin_phase_data);
}

trellis_status trellis_dit_flow_executor_run_single(
    trellis_dit_flow_executor * executor,
    const float * latent,
    float timestep,
    float * pred) {
    if (executor == NULL || executor->cuda == NULL || executor->weights == NULL ||
        executor->graph == NULL || executor->x == NULL || executor->y == NULL ||
        executor->x_host == NULL || executor->y_host == NULL ||
        latent == NULL || pred == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    memcpy(executor->x_host, latent, executor->input_count * sizeof(float));
    ggml_backend_tensor_set(executor->x, executor->x_host, 0, ggml_nbytes(executor->x));

    if (should_set_timestep(executor->weights)) {
        const float model_timestep = 1000.0f * timestep;
        ggml_backend_tensor_set(executor->t, &model_timestep, 0, ggml_nbytes(executor->t));
    }

    trellis_status status = trellis_cuda_compute_graph(executor->cuda, executor->graph);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    ggml_backend_tensor_get(executor->y, executor->y_host, 0, ggml_nbytes(executor->y));
    memcpy(pred, executor->y_host, executor->output_count * sizeof(float));
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
    free(executor->y_host);
    memset(executor, 0, sizeof(*executor));
}
