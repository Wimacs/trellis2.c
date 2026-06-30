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
