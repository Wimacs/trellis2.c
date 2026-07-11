#include "trellis.h"

/* Sparse-structure VAE decoder weight binding.
 * Corresponds to TRELLIS.2 sparse_structure_vae / ss_dec_conv3d checkpoint.
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static trellis_status bind_conv3d(
    trellis_tensor_store * store,
    const char * w_name,
    const char * b_name,
    int64_t in,
    int64_t out,
    struct ggml_tensor ** w,
    struct ggml_tensor ** b,
    char * first_issue,
    size_t first_issue_size) {
    int64_t ne[4] = { 3, 3, 3, in * out };
    trellis_status status = bind_tensor(store, w_name, 4, ne, w, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    return bind_vec(store, b_name, out, b, first_issue, first_issue_size);
}

static trellis_status bind_resblock(
    trellis_tensor_store * store,
    const char * prefix,
    int channels,
    trellis_ss_decoder_resblock_weights * block,
    char * first_issue,
    size_t first_issue_size) {
    if (store == NULL || prefix == NULL || block == NULL || channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(block, 0, sizeof(*block));
    block->channels = channels;

    char name[160];
    char bias[160];
    trellis_status status;

    snprintf(name, sizeof(name), "%s.norm1.weight", prefix);
    status = bind_vec(store, name, channels, &block->norm1_gamma, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "%s.norm1.bias", prefix);
    status = bind_vec(store, name, channels, &block->norm1_beta, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "%s.conv1.weight", prefix);
    snprintf(bias, sizeof(bias), "%s.conv1.bias", prefix);
    status = bind_conv3d(store, name, bias, channels, channels, &block->conv1_w, &block->conv1_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    snprintf(name, sizeof(name), "%s.norm2.weight", prefix);
    status = bind_vec(store, name, channels, &block->norm2_gamma, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "%s.norm2.bias", prefix);
    status = bind_vec(store, name, channels, &block->norm2_beta, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "%s.conv2.weight", prefix);
    snprintf(bias, sizeof(bias), "%s.conv2.bias", prefix);
    return bind_conv3d(store, name, bias, channels, channels, &block->conv2_w, &block->conv2_b, first_issue, first_issue_size);
}

trellis_status trellis_ss_decoder_bind_weights(
    trellis_tensor_store * store,
    trellis_ss_decoder_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    if (store == NULL || weights == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (first_issue != NULL && first_issue_size != 0) {
        first_issue[0] = '\0';
    }
    memset(weights, 0, sizeof(*weights));

    trellis_status status = bind_conv3d(
        store,
        "input_layer.weight",
        "input_layer.bias",
        8,
        512,
        &weights->input_w,
        &weights->input_b,
        first_issue,
        first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    status = bind_resblock(store, "middle_block.0", 512, &weights->middle[0], first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    status = bind_resblock(store, "middle_block.1", 512, &weights->middle[1], first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    status = bind_resblock(store, "blocks.0", 512, &weights->block0, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    status = bind_resblock(store, "blocks.1", 512, &weights->block1, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    status = bind_conv3d(store, "blocks.2.conv.weight", "blocks.2.conv.bias", 512, 1024, &weights->up0_w, &weights->up0_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    status = bind_resblock(store, "blocks.3", 128, &weights->block3, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    status = bind_resblock(store, "blocks.4", 128, &weights->block4, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    status = bind_conv3d(store, "blocks.5.conv.weight", "blocks.5.conv.bias", 128, 256, &weights->up1_w, &weights->up1_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    status = bind_resblock(store, "blocks.6", 32, &weights->block6, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    status = bind_resblock(store, "blocks.7", 32, &weights->block7, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    status = bind_vec(store, "out_layer.0.weight", 32, &weights->out_norm_gamma, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    status = bind_vec(store, "out_layer.0.bias", 32, &weights->out_norm_beta, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    return bind_conv3d(store, "out_layer.2.weight", "out_layer.2.bias", 32, 1, &weights->out_w, &weights->out_b, first_issue, first_issue_size);
}

static struct ggml_tensor * ss_decoder_bias_4d(
    struct ggml_context * ctx,
    struct ggml_tensor * bias) {
    if (ctx == NULL || bias == NULL || bias->ne[0] <= 0) {
        return NULL;
    }
    const size_t row = ggml_type_size(bias->type);
    return ggml_view_4d(ctx, bias, 1, 1, 1, bias->ne[0], row, row, row, 0);
}

static struct ggml_tensor * ss_decoder_cast_param_like(
    struct ggml_context * ctx,
    struct ggml_tensor * param,
    const struct ggml_tensor * ref) {
    if (ctx == NULL || param == NULL || ref == NULL || param->type == ref->type) {
        return param;
    }
    if (ref->type == GGML_TYPE_F32 || ref->type == GGML_TYPE_F16) {
        return ggml_cast(ctx, param, ref->type);
    }
    return param;
}

static struct ggml_tensor * ss_decoder_repeat_param(
    struct ggml_context * ctx,
    struct ggml_tensor * param,
    struct ggml_tensor * ref) {
    param = ss_decoder_cast_param_like(ctx, param, ref);
    return ggml_repeat(ctx, param, ref);
}

static struct ggml_tensor * ss_decoder_conv3d_same(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w,
    struct ggml_tensor * b,
    int in_channels,
    int out_channels,
    int batch) {
    if (ctx == NULL || x == NULL || w == NULL || in_channels <= 0 || out_channels <= 0 || batch <= 0) {
        return NULL;
    }
    struct ggml_tensor * y = ggml_conv_3d_direct(
        ctx,
        w,
        x,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        in_channels,
        batch,
        out_channels);
    if (y == NULL || b == NULL) {
        return y;
    }
    struct ggml_tensor * b4 = ss_decoder_bias_4d(ctx, b);
    if (b4 == NULL) {
        return NULL;
    }
    return ggml_add(ctx, y, ss_decoder_repeat_param(ctx, b4, y));
}

static struct ggml_tensor * ss_decoder_channel_layer_norm_3d(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    struct ggml_tensor * beta,
    float eps) {
    if (ctx == NULL || x == NULL) {
        return NULL;
    }
    struct ggml_tensor * h = ggml_permute(ctx, x, 1, 2, 3, 0);
    h = ggml_cont(ctx, h);
    h = ggml_norm(ctx, h, eps);
    if (gamma != NULL) {
        h = ggml_mul(ctx, h, ss_decoder_repeat_param(ctx, gamma, h));
    }
    if (beta != NULL) {
        h = ggml_add(ctx, h, ss_decoder_repeat_param(ctx, beta, h));
    }
    h = ggml_permute(ctx, h, 3, 0, 1, 2);
    return ggml_cont(ctx, h);
}

static struct ggml_tensor * ss_decoder_resblock(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    const trellis_ss_decoder_resblock_weights * block,
    int batch) {
    if (ctx == NULL || x == NULL || block == NULL || block->channels <= 0) {
        return NULL;
    }
    struct ggml_tensor * h = ss_decoder_channel_layer_norm_3d(
        ctx,
        x,
        block->norm1_gamma,
        block->norm1_beta,
        1e-6f);
    if (h == NULL) {
        return NULL;
    }
    h = ggml_silu(ctx, h);
    h = ss_decoder_conv3d_same(
        ctx,
        h,
        block->conv1_w,
        block->conv1_b,
        block->channels,
        block->channels,
        batch);
    if (h == NULL) {
        return NULL;
    }
    h = ss_decoder_channel_layer_norm_3d(
        ctx,
        h,
        block->norm2_gamma,
        block->norm2_beta,
        1e-6f);
    if (h == NULL) {
        return NULL;
    }
    h = ggml_silu(ctx, h);
    h = ss_decoder_conv3d_same(
        ctx,
        h,
        block->conv2_w,
        block->conv2_b,
        block->channels,
        block->channels,
        batch);
    if (h == NULL) {
        return NULL;
    }
    return ggml_add(ctx, x, h);
}

static struct ggml_tensor * ss_decoder_upsample(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * conv_w,
    struct ggml_tensor * conv_b,
    int in_channels,
    int conv_out_channels,
    int batch) {
    if (ctx == NULL || x == NULL || conv_w == NULL || conv_out_channels <= 0 ||
        conv_out_channels % 8 != 0) {
        return NULL;
    }
    struct ggml_tensor * h = ss_decoder_conv3d_same(
        ctx,
        x,
        conv_w,
        conv_b,
        in_channels,
        conv_out_channels,
        batch);
    return h == NULL ? NULL : ggml_pixel_shuffle_3d(ctx, h, 2);
}

static struct ggml_tensor * ss_decoder_forward_graph(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    const trellis_ss_decoder_weights * weights,
    int batch) {
    if (ctx == NULL || x == NULL || weights == NULL) {
        return NULL;
    }
    struct ggml_tensor * h = ss_decoder_conv3d_same(ctx, x, weights->input_w, weights->input_b, 8, 512, batch);
    if (h == NULL) return NULL;
    h = ss_decoder_resblock(ctx, h, &weights->middle[0], batch);
    if (h == NULL) return NULL;
    h = ss_decoder_resblock(ctx, h, &weights->middle[1], batch);
    if (h == NULL) return NULL;
    h = ss_decoder_resblock(ctx, h, &weights->block0, batch);
    if (h == NULL) return NULL;
    h = ss_decoder_resblock(ctx, h, &weights->block1, batch);
    if (h == NULL) return NULL;
    h = ss_decoder_upsample(ctx, h, weights->up0_w, weights->up0_b, 512, 1024, batch);
    if (h == NULL) return NULL;
    h = ss_decoder_resblock(ctx, h, &weights->block3, batch);
    if (h == NULL) return NULL;
    h = ss_decoder_resblock(ctx, h, &weights->block4, batch);
    if (h == NULL) return NULL;
    h = ss_decoder_upsample(ctx, h, weights->up1_w, weights->up1_b, 128, 256, batch);
    if (h == NULL) return NULL;
    h = ss_decoder_resblock(ctx, h, &weights->block6, batch);
    if (h == NULL) return NULL;
    h = ss_decoder_resblock(ctx, h, &weights->block7, batch);
    if (h == NULL) return NULL;
    h = ss_decoder_channel_layer_norm_3d(ctx, h, weights->out_norm_gamma, weights->out_norm_beta, 1e-6f);
    if (h == NULL) return NULL;
    h = ggml_silu(ctx, h);
    return ss_decoder_conv3d_same(ctx, h, weights->out_w, weights->out_b, 32, 1, batch);
}

trellis_status trellis_ss_decoder_forward_f32_host(
    const trellis_ss_decoder_weights * weights,
    const float * latent,
    const trellis_backend_context * backend,
    int batch,
    int latent_size,
    float ** logits_out,
    int * output_size) {
    if (weights == NULL || latent == NULL || backend == NULL || backend->backend == NULL ||
        logits_out == NULL || output_size == NULL || batch != 1 || latent_size <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    *logits_out = NULL;
    *output_size = 0;

    const size_t graph_nodes = 8192;
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * graph_nodes +
            ggml_graph_overhead_custom(graph_nodes, false) + 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_status status = TRELLIS_STATUS_OK;
    ggml_gallocr_t alloc = NULL;
    float * host = NULL;

    struct ggml_tensor * x = ggml_new_tensor_4d(
        ctx,
        GGML_TYPE_F32,
        latent_size,
        latent_size,
        latent_size,
        8 * batch);
    if (x == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    ggml_set_input(x);

    struct ggml_tensor * y = ss_decoder_forward_graph(ctx, x, weights, batch);
    if (y == NULL) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    ggml_set_output(y);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    if (graph == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    ggml_build_forward_expand(graph, y);

    alloc = trellis_backend_new_graph_allocator(backend);
    if (alloc == NULL || !ggml_gallocr_alloc_graph(alloc, graph)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    ggml_backend_tensor_set(x, latent, 0, ggml_nbytes(x));
    status = trellis_backend_compute_graph(backend, graph);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }

    host = (float *) malloc(ggml_nbytes(y));
    if (host == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    ggml_backend_tensor_get(y, host, 0, ggml_nbytes(y));
    *logits_out = host;
    host = NULL;
    *output_size = (int) y->ne[0];

cleanup:
    free(host);
    if (alloc != NULL) {
        ggml_gallocr_free(alloc);
    }
    ggml_free(ctx);
    return status;
}
