#include "trellis.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void encoder_set_issue(char * dst, size_t dst_size, const char * fmt, ...) {
    if (dst == NULL || dst_size == 0 || dst[0] != '\0') {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(dst, dst_size, fmt, args);
    va_end(args);
}

static trellis_status encoder_bind_tensor(
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
    struct ggml_tensor * tensor = trellis_tensor_store_get(store, name);
    if (tensor == NULL) {
        encoder_set_issue(first_issue, first_issue_size, "missing tensor: %s", name);
        return TRELLIS_STATUS_NOT_FOUND;
    }
    for (int i = 0; i < n_dims; ++i) {
        if (tensor->ne[i] != ne[i]) {
            encoder_set_issue(
                first_issue,
                first_issue_size,
                "shape mismatch: %s dim%d got %lld expected %lld",
                name,
                i,
                (long long) tensor->ne[i],
                (long long) ne[i]);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    for (int i = n_dims; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] != 1) {
            encoder_set_issue(
                first_issue,
                first_issue_size,
                "rank mismatch: %s dim%d got %lld expected 1",
                name,
                i,
                (long long) tensor->ne[i]);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    if (tensor->data == NULL) {
        encoder_set_issue(first_issue, first_issue_size, "tensor has no host data: %s", name);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = tensor;
    return TRELLIS_STATUS_OK;
}

static trellis_status encoder_bind_vec(
    trellis_tensor_store * store,
    const char * name,
    int64_t n,
    const float ** out,
    char * first_issue,
    size_t first_issue_size) {
    const int64_t ne[1] = {n};
    struct ggml_tensor * tensor = NULL;
    const trellis_status status = encoder_bind_tensor(
        store, name, 1, ne, &tensor, first_issue, first_issue_size);
    if (status == TRELLIS_STATUS_OK) {
        *out = (const float *) tensor->data;
    }
    return status;
}

static trellis_status encoder_bind_linear(
    trellis_tensor_store * store,
    const char * weight_name,
    const char * bias_name,
    int64_t in_channels,
    int64_t out_channels,
    const float ** weight,
    const float ** bias,
    char * first_issue,
    size_t first_issue_size) {
    const int64_t ne[2] = {in_channels, out_channels};
    struct ggml_tensor * tensor = NULL;
    trellis_status status = encoder_bind_tensor(
        store, weight_name, 2, ne, &tensor, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    *weight = (const float *) tensor->data;
    return encoder_bind_vec(store, bias_name, out_channels, bias, first_issue, first_issue_size);
}

static trellis_status encoder_bind_sparse_conv3d(
    trellis_tensor_store * store,
    const char * weight_name,
    const char * bias_name,
    int64_t in_channels,
    int64_t out_channels,
    const float ** weight,
    const float ** bias,
    char * first_issue,
    size_t first_issue_size) {
    /* The safetensors importer folds [out,3,3,3,in] into ggml ne
     * [in,3,3,3*out] while retaining the original contiguous bytes. */
    const int64_t ne[4] = {in_channels, 3, 3, 3 * out_channels};
    struct ggml_tensor * tensor = NULL;
    trellis_status status = encoder_bind_tensor(
        store, weight_name, 4, ne, &tensor, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    *weight = (const float *) tensor->data;
    return encoder_bind_vec(store, bias_name, out_channels, bias, first_issue, first_issue_size);
}

static trellis_status encoder_bind_convnext(
    trellis_tensor_store * store,
    const char * prefix,
    int channels,
    trellis_sparse_unet_vae_encoder_convnext_block_weights * block,
    char * first_issue,
    size_t first_issue_size) {
    char weight_name[256];
    char bias_name[256];
    memset(block, 0, sizeof(*block));
    block->channels = channels;

    snprintf(weight_name, sizeof(weight_name), "%s.conv.weight", prefix);
    snprintf(bias_name, sizeof(bias_name), "%s.conv.bias", prefix);
    trellis_status status = encoder_bind_sparse_conv3d(
        store,
        weight_name,
        bias_name,
        channels,
        channels,
        &block->conv_w,
        &block->conv_b,
        first_issue,
        first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    snprintf(weight_name, sizeof(weight_name), "%s.norm.weight", prefix);
    status = encoder_bind_vec(
        store, weight_name, channels, &block->norm_gamma, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(weight_name, sizeof(weight_name), "%s.norm.bias", prefix);
    status = encoder_bind_vec(
        store, weight_name, channels, &block->norm_beta, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    snprintf(weight_name, sizeof(weight_name), "%s.mlp.0.weight", prefix);
    snprintf(bias_name, sizeof(bias_name), "%s.mlp.0.bias", prefix);
    status = encoder_bind_linear(
        store,
        weight_name,
        bias_name,
        channels,
        4 * channels,
        &block->mlp0_w,
        &block->mlp0_b,
        first_issue,
        first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    snprintf(weight_name, sizeof(weight_name), "%s.mlp.2.weight", prefix);
    snprintf(bias_name, sizeof(bias_name), "%s.mlp.2.bias", prefix);
    return encoder_bind_linear(
        store,
        weight_name,
        bias_name,
        4 * channels,
        channels,
        &block->mlp2_w,
        &block->mlp2_b,
        first_issue,
        first_issue_size);
}

static trellis_status encoder_bind_s2c(
    trellis_tensor_store * store,
    const char * prefix,
    int in_channels,
    int out_channels,
    trellis_sparse_unet_vae_encoder_s2c_block_weights * block,
    char * first_issue,
    size_t first_issue_size) {
    if (out_channels % 8 != 0 || (8 * in_channels) % out_channels != 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    char weight_name[256];
    char bias_name[256];
    memset(block, 0, sizeof(*block));
    block->in_channels = in_channels;
    block->out_channels = out_channels;

    snprintf(weight_name, sizeof(weight_name), "%s.norm1.weight", prefix);
    trellis_status status = encoder_bind_vec(
        store, weight_name, in_channels, &block->norm1_gamma, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(weight_name, sizeof(weight_name), "%s.norm1.bias", prefix);
    status = encoder_bind_vec(
        store, weight_name, in_channels, &block->norm1_beta, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    snprintf(weight_name, sizeof(weight_name), "%s.conv1.weight", prefix);
    snprintf(bias_name, sizeof(bias_name), "%s.conv1.bias", prefix);
    status = encoder_bind_sparse_conv3d(
        store,
        weight_name,
        bias_name,
        in_channels,
        out_channels / 8,
        &block->conv1_w,
        &block->conv1_b,
        first_issue,
        first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    snprintf(weight_name, sizeof(weight_name), "%s.conv2.weight", prefix);
    snprintf(bias_name, sizeof(bias_name), "%s.conv2.bias", prefix);
    return encoder_bind_sparse_conv3d(
        store,
        weight_name,
        bias_name,
        out_channels,
        out_channels,
        &block->conv2_w,
        &block->conv2_b,
        first_issue,
        first_issue_size);
}

trellis_status trellis_sparse_unet_vae_encoder_bind_weights(
    trellis_tensor_store * store,
    trellis_sparse_unet_vae_encoder_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    if (store == NULL || weights == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (first_issue != NULL && first_issue_size > 0) {
        first_issue[0] = '\0';
    }
    memset(weights, 0, sizeof(*weights));

    static const int channels[TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS] = {
        64, 128, 256, 512, 1024,
    };
    static const int blocks[TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS] = {
        0, 4, 8, 16, 4,
    };
    weights->in_channels = 6;
    weights->latent_channels = 32;
    weights->levels = TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS;
    for (int level = 0; level < TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS; ++level) {
        weights->channels[level] = channels[level];
        weights->blocks_per_level[level] = blocks[level];
    }

    trellis_status status = encoder_bind_linear(
        store,
        "input_layer.weight",
        "input_layer.bias",
        weights->in_channels,
        channels[0],
        &weights->input_w,
        &weights->input_b,
        first_issue,
        first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    char prefix[128];
    for (int level = 0; level < TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS; ++level) {
        for (int block = 0; block < blocks[level]; ++block) {
            snprintf(prefix, sizeof(prefix), "blocks.%d.%d", level, block);
            status = encoder_bind_convnext(
                store,
                prefix,
                channels[level],
                &weights->blocks[level][block],
                first_issue,
                first_issue_size);
            if (status != TRELLIS_STATUS_OK) return status;
        }
        if (level < TRELLIS_SPARSE_UNET_VAE_ENCODER_DOWN_LEVELS) {
            snprintf(prefix, sizeof(prefix), "blocks.%d.%d", level, blocks[level]);
            status = encoder_bind_s2c(
                store,
                prefix,
                channels[level],
                channels[level + 1],
                &weights->down_blocks[level],
                first_issue,
                first_issue_size);
            if (status != TRELLIS_STATUS_OK) return status;
        }
    }

    return encoder_bind_linear(
        store,
        "to_latent.weight",
        "to_latent.bias",
        channels[TRELLIS_SPARSE_UNET_VAE_ENCODER_LEVELS - 1],
        2 * weights->latent_channels,
        &weights->to_latent_w,
        &weights->to_latent_b,
        first_issue,
        first_issue_size);
}
