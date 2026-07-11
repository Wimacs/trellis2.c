#include "trellis.h"

#include <stdarg.h>
#include <stdio.h>
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

static trellis_status make_name(char * dst, size_t dst_size, const char * prefix, const char * suffix) {
    if (dst == NULL || dst_size == 0 || prefix == NULL || suffix == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t a = strlen(prefix);
    const size_t b = strlen(suffix);
    if (a + b + 1u > dst_size) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memcpy(dst, prefix, a);
    memcpy(dst + a, suffix, b + 1u);
    return TRELLIS_STATUS_OK;
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
    const float ** out,
    char * first_issue,
    size_t first_issue_size) {
    int64_t ne[1] = { n };
    struct ggml_tensor * t = NULL;
    trellis_status status = bind_tensor(store, name, 1, ne, &t, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (t->data == NULL) {
        set_issue(first_issue, first_issue_size, "tensor has no device data: %s", name);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = (const float *) t->data;
    return TRELLIS_STATUS_OK;
}

static trellis_status bind_linear(
    trellis_tensor_store * store,
    const char * w_name,
    const char * b_name,
    int64_t in,
    int64_t out,
    const float ** w,
    const float ** b,
    char * first_issue,
    size_t first_issue_size) {
    int64_t ne[2] = { in, out };
    struct ggml_tensor * wt = NULL;
    trellis_status status = bind_tensor(store, w_name, 2, ne, &wt, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (wt->data == NULL) {
        set_issue(first_issue, first_issue_size, "tensor has no device data: %s", w_name);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *w = (const float *) wt->data;
    return bind_vec(store, b_name, out, b, first_issue, first_issue_size);
}

static trellis_status bind_sparse_conv3d_flex(
    trellis_tensor_store * store,
    const char * w_name,
    const char * b_name,
    int64_t in,
    int64_t out,
    const float ** w,
    const float ** b,
    char * first_issue,
    size_t first_issue_size) {
    /* 5D safetensors [out,3,3,3,in] are folded by the importer into
     * [in,3,3,3*out] while preserving raw contiguous storage.
     */
    int64_t ne[4] = { in, 3, 3, 3 * out };
    struct ggml_tensor * wt = NULL;
    trellis_status status = bind_tensor(store, w_name, 4, ne, &wt, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (wt->data == NULL) {
        set_issue(first_issue, first_issue_size, "tensor has no device data: %s", w_name);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *w = (const float *) wt->data;
    return bind_vec(store, b_name, out, b, first_issue, first_issue_size);
}

static trellis_status bind_convnext_block(
    trellis_tensor_store * store,
    const char * prefix,
    int channels,
    trellis_sparse_unet_vae_decoder_convnext_block_weights * block,
    char * first_issue,
    size_t first_issue_size) {
    char name[512];
    char bias[512];
    memset(block, 0, sizeof(*block));
    block->channels = channels;

    if (make_name(name, sizeof(name), prefix, ".conv.weight") != TRELLIS_STATUS_OK ||
        make_name(bias, sizeof(bias), prefix, ".conv.bias") != TRELLIS_STATUS_OK) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_status status = bind_sparse_conv3d_flex(
        store, name, bias, channels, channels, &block->conv_w, &block->conv_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    if (make_name(name, sizeof(name), prefix, ".norm.weight") != TRELLIS_STATUS_OK) return TRELLIS_STATUS_INVALID_ARGUMENT;
    status = bind_vec(store, name, channels, &block->norm_gamma, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    if (make_name(name, sizeof(name), prefix, ".norm.bias") != TRELLIS_STATUS_OK) return TRELLIS_STATUS_INVALID_ARGUMENT;
    status = bind_vec(store, name, channels, &block->norm_beta, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    if (make_name(name, sizeof(name), prefix, ".mlp.0.weight") != TRELLIS_STATUS_OK ||
        make_name(bias, sizeof(bias), prefix, ".mlp.0.bias") != TRELLIS_STATUS_OK) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    status = bind_linear(store, name, bias, channels, 4 * channels, &block->mlp0_w, &block->mlp0_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    if (make_name(name, sizeof(name), prefix, ".mlp.2.weight") != TRELLIS_STATUS_OK ||
        make_name(bias, sizeof(bias), prefix, ".mlp.2.bias") != TRELLIS_STATUS_OK) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return bind_linear(store, name, bias, 4 * channels, channels, &block->mlp2_w, &block->mlp2_b, first_issue, first_issue_size);
}

static trellis_status bind_c2s_block(
    trellis_tensor_store * store,
    const char * prefix,
    int in_channels,
    int out_channels,
    bool pred_subdiv,
    trellis_sparse_unet_vae_decoder_c2s_block_weights * block,
    char * first_issue,
    size_t first_issue_size) {
    char name[512];
    char bias[512];
    memset(block, 0, sizeof(*block));
    block->in_channels = in_channels;
    block->out_channels = out_channels;

    if (make_name(name, sizeof(name), prefix, ".norm1.weight") != TRELLIS_STATUS_OK) return TRELLIS_STATUS_INVALID_ARGUMENT;
    trellis_status status = bind_vec(store, name, in_channels, &block->norm1_gamma, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    if (make_name(name, sizeof(name), prefix, ".norm1.bias") != TRELLIS_STATUS_OK) return TRELLIS_STATUS_INVALID_ARGUMENT;
    status = bind_vec(store, name, in_channels, &block->norm1_beta, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    if (make_name(name, sizeof(name), prefix, ".conv1.weight") != TRELLIS_STATUS_OK ||
        make_name(bias, sizeof(bias), prefix, ".conv1.bias") != TRELLIS_STATUS_OK) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    status = bind_sparse_conv3d_flex(
        store, name, bias, in_channels, 8 * out_channels, &block->conv1_w, &block->conv1_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    if (make_name(name, sizeof(name), prefix, ".conv2.weight") != TRELLIS_STATUS_OK ||
        make_name(bias, sizeof(bias), prefix, ".conv2.bias") != TRELLIS_STATUS_OK) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    status = bind_sparse_conv3d_flex(
        store, name, bias, out_channels, out_channels, &block->conv2_w, &block->conv2_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    if (pred_subdiv) {
        if (make_name(name, sizeof(name), prefix, ".to_subdiv.weight") != TRELLIS_STATUS_OK ||
            make_name(bias, sizeof(bias), prefix, ".to_subdiv.bias") != TRELLIS_STATUS_OK) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        return bind_linear(store, name, bias, in_channels, 8, &block->to_subdiv_w, &block->to_subdiv_b, first_issue, first_issue_size);
    }
    block->to_subdiv_w = NULL;
    block->to_subdiv_b = NULL;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_sparse_unet_vae_decoder_bind_weights(
    trellis_tensor_store * store,
    int out_channels,
    bool pred_subdiv,
    trellis_sparse_unet_vae_decoder_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    if (store == NULL || weights == NULL || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (first_issue != NULL && first_issue_size != 0) {
        first_issue[0] = '\0';
    }
    memset(weights, 0, sizeof(*weights));

    const int channels[TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS] = {1024, 512, 256, 128, 64};
    const int blocks[TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS] = {4, 16, 8, 4, 0};
    weights->latent_channels = 32;
    weights->out_channels = out_channels;
    weights->pred_subdiv = pred_subdiv ? 1 : 0;
    weights->levels = TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS;
    for (int i = 0; i < TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS; ++i) {
        weights->channels[i] = channels[i];
        weights->blocks_per_level[i] = blocks[i];
    }

    trellis_status status = bind_linear(
        store,
        "from_latent.weight",
        "from_latent.bias",
        32,
        channels[0],
        &weights->from_latent_w,
        &weights->from_latent_b,
        first_issue,
        first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    char prefix[512];
    for (int level = 0; level < TRELLIS_SHAPE_DECODER_LEVELS; ++level) {
        for (int block = 0; block < blocks[level]; ++block) {
            snprintf(prefix, sizeof(prefix), "blocks.%d.%d", level, block);
            status = bind_convnext_block(
                store,
                prefix,
                channels[level],
                &weights->blocks[level][block],
                first_issue,
                first_issue_size);
            if (status != TRELLIS_STATUS_OK) {
                return status;
            }
        }
        if (level < TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS - 1) {
            snprintf(prefix, sizeof(prefix), "blocks.%d.%d", level, blocks[level]);
            status = bind_c2s_block(
                store,
                prefix,
                channels[level],
                channels[level + 1],
                pred_subdiv,
                &weights->up_blocks[level],
                first_issue,
                first_issue_size);
            if (status != TRELLIS_STATUS_OK) {
                return status;
            }
        }
    }

    return bind_linear(
        store,
        "output_layer.weight",
        "output_layer.bias",
        channels[TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS - 1],
        out_channels,
        &weights->output_w,
        &weights->output_b,
        first_issue,
        first_issue_size);
}

trellis_status trellis_flexi_dual_grid_vae_decoder_bind_weights(
    trellis_tensor_store * store,
    trellis_sparse_unet_vae_decoder_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    return trellis_sparse_unet_vae_decoder_bind_weights(store, 7, true, weights, first_issue, first_issue_size);
}

trellis_status trellis_tex_slat_decoder_bind_weights(
    trellis_tensor_store * store,
    trellis_sparse_unet_vae_decoder_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    return trellis_sparse_unet_vae_decoder_bind_weights(store, 6, false, weights, first_issue, first_issue_size);
}

trellis_status trellis_shape_decoder_bind_weights(
    trellis_tensor_store * store,
    trellis_shape_decoder_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    return trellis_flexi_dual_grid_vae_decoder_bind_weights(store, weights, first_issue, first_issue_size);
}
