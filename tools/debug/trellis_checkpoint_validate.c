#include "trellis_checkpoint_validate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct expected_tensor {
    const char * name;
    trellis_dtype dtype;
    int n_dims;
    int64_t shape[TRELLIS_MAX_DIMS];
} expected_tensor;

void trellis_checkpoint_report_clear(trellis_checkpoint_report * report) {
    if (report != NULL) {
        memset(report, 0, sizeof(*report));
    }
}

static void set_first_issue(trellis_checkpoint_report * report, const char * message, const char * name) {
    if (report == NULL || report->first_issue[0] != '\0') {
        return;
    }
    if (name != NULL) {
        snprintf(report->first_issue, sizeof(report->first_issue), "%s: %s", message, name);
    } else {
        snprintf(report->first_issue, sizeof(report->first_issue), "%s", message);
    }
}

static uint64_t shape_nelements(int n_dims, const int64_t * shape) {
    uint64_t n = 1;
    for (int i = 0; i < n_dims; ++i) {
        n *= (uint64_t) shape[i];
    }
    return n;
}

static bool shape_equal(const trellis_safetensor_meta * meta, int n_dims, const int64_t * shape) {
    if (meta->n_dims != n_dims) {
        return false;
    }
    for (int i = 0; i < n_dims; ++i) {
        if (meta->shape[i] != shape[i]) {
            return false;
        }
    }
    return true;
}

static trellis_status expect_tensor(
    const trellis_safetensors * st,
    trellis_checkpoint_report * report,
    const char * name,
    trellis_dtype dtype,
    int n_dims,
    const int64_t * shape) {
    report->expected_tensors += 1;
    report->expected_elements += shape_nelements(n_dims, shape);
    report->expected_bytes += shape_nelements(n_dims, shape) * (uint64_t) trellis_dtype_size(dtype);

    const trellis_safetensor_meta * meta = trellis_safetensors_find(st, name);
    if (meta == NULL) {
        report->missing_tensors += 1;
        set_first_issue(report, "missing tensor", name);
        return TRELLIS_STATUS_NOT_FOUND;
    }

    report->found_tensors += 1;
    if (meta->dtype != dtype) {
        report->dtype_mismatches += 1;
        set_first_issue(report, "dtype mismatch", name);
    }
    if (!shape_equal(meta, n_dims, shape)) {
        report->shape_mismatches += 1;
        set_first_issue(report, "shape mismatch", name);
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status finish_report(const trellis_safetensors * st, trellis_checkpoint_report * report) {
    report->actual_tensors = st->n_tensors;
    if (report->actual_tensors > report->expected_tensors) {
        report->extra_tensors = report->actual_tensors - report->expected_tensors;
        set_first_issue(report, "checkpoint has extra tensors", NULL);
    }
    if (report->missing_tensors != 0 || report->shape_mismatches != 0 ||
        report->dtype_mismatches != 0 || report->extra_tensors != 0) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    return TRELLIS_STATUS_OK;
}

static void shape1(int64_t * s, int64_t a) {
    s[0] = a;
}

static void shape2(int64_t * s, int64_t a, int64_t b) {
    s[0] = a;
    s[1] = b;
}

static void shape5(int64_t * s, int64_t a, int64_t b, int64_t c, int64_t d, int64_t e) {
    s[0] = a;
    s[1] = b;
    s[2] = c;
    s[3] = d;
    s[4] = e;
}

static void expect_named(
    const trellis_safetensors * st,
    trellis_checkpoint_report * report,
    const char * name,
    trellis_dtype dtype,
    int n_dims,
    const int64_t * shape) {
    (void) expect_tensor(st, report, name, dtype, n_dims, shape);
}

static void expect_vec(
    const trellis_safetensors * st,
    trellis_checkpoint_report * report,
    const char * name,
    trellis_dtype dtype,
    int64_t n) {
    int64_t s[TRELLIS_MAX_DIMS] = {0};
    shape1(s, n);
    expect_named(st, report, name, dtype, 1, s);
}

static void expect_linear(
    const trellis_safetensors * st,
    trellis_checkpoint_report * report,
    const char * weight,
    const char * bias,
    trellis_dtype dtype,
    int64_t out,
    int64_t in) {
    int64_t s[TRELLIS_MAX_DIMS] = {0};
    shape2(s, out, in);
    expect_named(st, report, weight, dtype, 2, s);
    expect_vec(st, report, bias, dtype, out);
}

static void expect_conv3d(
    const trellis_safetensors * st,
    trellis_checkpoint_report * report,
    const char * weight,
    const char * bias,
    trellis_dtype dtype,
    int64_t out,
    int64_t in) {
    int64_t s[TRELLIS_MAX_DIMS] = {0};
    shape5(s, out, in, 3, 3, 3);
    expect_named(st, report, weight, dtype, 5, s);
    expect_vec(st, report, bias, dtype, out);
}

static void expect_sparse_conv3d_flex(
    const trellis_safetensors * st,
    trellis_checkpoint_report * report,
    const char * weight,
    const char * bias,
    trellis_dtype dtype,
    int64_t out,
    int64_t in) {
    int64_t s[TRELLIS_MAX_DIMS] = {0};
    shape5(s, out, 3, 3, 3, in);
    expect_named(st, report, weight, dtype, 5, s);
    expect_vec(st, report, bias, dtype, out);
}

static trellis_status validate_dit_flow_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report,
    int64_t in_channels,
    int64_t out_channels) {
    if (safetensors_path == NULL || report == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_checkpoint_report_clear(report);

    trellis_safetensors st;
    trellis_status status = trellis_safetensors_open(safetensors_path, &st);
    if (status != TRELLIS_STATUS_OK) {
        set_first_issue(report, trellis_status_string(status), safetensors_path);
        return status;
    }

    enum {
        C = 1536,
        COND_C = 1024,
        T_FREQ = 256,
        HEADS = 12,
        HEAD_DIM = 128,
        MLP = 8192,
        BLOCKS = 30,
        MOD = 9216,
    };

    expect_linear(&st, report, "input_layer.weight", "input_layer.bias", TRELLIS_DTYPE_BF16, C, in_channels);
    expect_linear(&st, report, "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias", TRELLIS_DTYPE_BF16, C, T_FREQ);
    expect_linear(&st, report, "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias", TRELLIS_DTYPE_BF16, C, C);
    expect_linear(&st, report, "adaLN_modulation.1.weight", "adaLN_modulation.1.bias", TRELLIS_DTYPE_BF16, MOD, C);

    char name[160];
    for (int i = 0; i < BLOCKS; ++i) {
        snprintf(name, sizeof(name), "blocks.%d.modulation", i);
        expect_vec(&st, report, name, TRELLIS_DTYPE_BF16, MOD);
        snprintf(name, sizeof(name), "blocks.%d.norm2.weight", i);
        expect_vec(&st, report, name, TRELLIS_DTYPE_BF16, C);
        snprintf(name, sizeof(name), "blocks.%d.norm2.bias", i);
        expect_vec(&st, report, name, TRELLIS_DTYPE_BF16, C);

        snprintf(name, sizeof(name), "blocks.%d.self_attn.to_qkv.weight", i);
        char bias[160];
        snprintf(bias, sizeof(bias), "blocks.%d.self_attn.to_qkv.bias", i);
        expect_linear(&st, report, name, bias, TRELLIS_DTYPE_BF16, 3 * C, C);
        snprintf(name, sizeof(name), "blocks.%d.self_attn.to_out.weight", i);
        snprintf(bias, sizeof(bias), "blocks.%d.self_attn.to_out.bias", i);
        expect_linear(&st, report, name, bias, TRELLIS_DTYPE_BF16, C, C);
        snprintf(name, sizeof(name), "blocks.%d.self_attn.q_rms_norm.gamma", i);
        int64_t qk_shape[TRELLIS_MAX_DIMS] = {0};
        shape2(qk_shape, HEADS, HEAD_DIM);
        expect_named(&st, report, name, TRELLIS_DTYPE_BF16, 2, qk_shape);
        snprintf(name, sizeof(name), "blocks.%d.self_attn.k_rms_norm.gamma", i);
        expect_named(&st, report, name, TRELLIS_DTYPE_BF16, 2, qk_shape);

        snprintf(name, sizeof(name), "blocks.%d.cross_attn.to_q.weight", i);
        snprintf(bias, sizeof(bias), "blocks.%d.cross_attn.to_q.bias", i);
        expect_linear(&st, report, name, bias, TRELLIS_DTYPE_BF16, C, C);
        snprintf(name, sizeof(name), "blocks.%d.cross_attn.to_kv.weight", i);
        snprintf(bias, sizeof(bias), "blocks.%d.cross_attn.to_kv.bias", i);
        expect_linear(&st, report, name, bias, TRELLIS_DTYPE_BF16, 2 * C, COND_C);
        snprintf(name, sizeof(name), "blocks.%d.cross_attn.to_out.weight", i);
        snprintf(bias, sizeof(bias), "blocks.%d.cross_attn.to_out.bias", i);
        expect_linear(&st, report, name, bias, TRELLIS_DTYPE_BF16, C, C);
        snprintf(name, sizeof(name), "blocks.%d.cross_attn.q_rms_norm.gamma", i);
        expect_named(&st, report, name, TRELLIS_DTYPE_BF16, 2, qk_shape);
        snprintf(name, sizeof(name), "blocks.%d.cross_attn.k_rms_norm.gamma", i);
        expect_named(&st, report, name, TRELLIS_DTYPE_BF16, 2, qk_shape);

        snprintf(name, sizeof(name), "blocks.%d.mlp.mlp.0.weight", i);
        snprintf(bias, sizeof(bias), "blocks.%d.mlp.mlp.0.bias", i);
        expect_linear(&st, report, name, bias, TRELLIS_DTYPE_BF16, MLP, C);
        snprintf(name, sizeof(name), "blocks.%d.mlp.mlp.2.weight", i);
        snprintf(bias, sizeof(bias), "blocks.%d.mlp.mlp.2.bias", i);
        expect_linear(&st, report, name, bias, TRELLIS_DTYPE_BF16, C, MLP);
    }

    expect_linear(&st, report, "out_layer.weight", "out_layer.bias", TRELLIS_DTYPE_BF16, out_channels, C);

    status = finish_report(&st, report);
    trellis_safetensors_close(&st);
    return status;
}

trellis_status trellis_ss_flow_validate_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report) {
    return validate_dit_flow_checkpoint(safetensors_path, report, 8, 8);
}

trellis_status trellis_shape_slat_flow_validate_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report) {
    return validate_dit_flow_checkpoint(safetensors_path, report, 32, 32);
}

static void expect_sparse_convnext_block(
    const trellis_safetensors * st,
    trellis_checkpoint_report * report,
    const char * prefix,
    int64_t channels) {
    char name[256];
    char bias[256];
    snprintf(name, sizeof(name), "%s.conv.weight", prefix);
    snprintf(bias, sizeof(bias), "%s.conv.bias", prefix);
    expect_sparse_conv3d_flex(st, report, name, bias, TRELLIS_DTYPE_F16, channels, channels);
    snprintf(name, sizeof(name), "%s.norm.weight", prefix);
    expect_vec(st, report, name, TRELLIS_DTYPE_F16, channels);
    snprintf(name, sizeof(name), "%s.norm.bias", prefix);
    expect_vec(st, report, name, TRELLIS_DTYPE_F16, channels);
    snprintf(name, sizeof(name), "%s.mlp.0.weight", prefix);
    snprintf(bias, sizeof(bias), "%s.mlp.0.bias", prefix);
    expect_linear(st, report, name, bias, TRELLIS_DTYPE_F16, channels * 4, channels);
    snprintf(name, sizeof(name), "%s.mlp.2.weight", prefix);
    snprintf(bias, sizeof(bias), "%s.mlp.2.bias", prefix);
    expect_linear(st, report, name, bias, TRELLIS_DTYPE_F16, channels, channels * 4);
}

static void expect_sparse_c2s_block(
    const trellis_safetensors * st,
    trellis_checkpoint_report * report,
    const char * prefix,
    int64_t in_channels,
    int64_t out_channels) {
    char name[256];
    char bias[256];
    snprintf(name, sizeof(name), "%s.norm1.weight", prefix);
    expect_vec(st, report, name, TRELLIS_DTYPE_F16, in_channels);
    snprintf(name, sizeof(name), "%s.norm1.bias", prefix);
    expect_vec(st, report, name, TRELLIS_DTYPE_F16, in_channels);
    snprintf(name, sizeof(name), "%s.conv1.weight", prefix);
    snprintf(bias, sizeof(bias), "%s.conv1.bias", prefix);
    expect_sparse_conv3d_flex(st, report, name, bias, TRELLIS_DTYPE_F16, out_channels * 8, in_channels);
    snprintf(name, sizeof(name), "%s.conv2.weight", prefix);
    snprintf(bias, sizeof(bias), "%s.conv2.bias", prefix);
    expect_sparse_conv3d_flex(st, report, name, bias, TRELLIS_DTYPE_F16, out_channels, out_channels);
    snprintf(name, sizeof(name), "%s.to_subdiv.weight", prefix);
    snprintf(bias, sizeof(bias), "%s.to_subdiv.bias", prefix);
    expect_linear(st, report, name, bias, TRELLIS_DTYPE_F16, 8, in_channels);
}

trellis_status trellis_shape_decoder_validate_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report) {
    if (safetensors_path == NULL || report == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_checkpoint_report_clear(report);

    trellis_safetensors st;
    trellis_status status = trellis_safetensors_open(safetensors_path, &st);
    if (status != TRELLIS_STATUS_OK) {
        set_first_issue(report, trellis_status_string(status), safetensors_path);
        return status;
    }

    const int64_t channels[] = {1024, 512, 256, 128, 64};
    const int blocks[] = {4, 16, 8, 4, 0};
    expect_linear(&st, report, "from_latent.weight", "from_latent.bias", TRELLIS_DTYPE_F16, channels[0], 32);

    char prefix[160];
    for (int level = 0; level < 5; ++level) {
        for (int block = 0; block < blocks[level]; ++block) {
            snprintf(prefix, sizeof(prefix), "blocks.%d.%d", level, block);
            expect_sparse_convnext_block(&st, report, prefix, channels[level]);
        }
        if (level < 4) {
            snprintf(prefix, sizeof(prefix), "blocks.%d.%d", level, blocks[level]);
            expect_sparse_c2s_block(&st, report, prefix, channels[level], channels[level + 1]);
        }
    }

    expect_linear(&st, report, "output_layer.weight", "output_layer.bias", TRELLIS_DTYPE_F16, 7, channels[4]);

    status = finish_report(&st, report);
    trellis_safetensors_close(&st);
    return status;
}

static void expect_resblock(
    const trellis_safetensors * st,
    trellis_checkpoint_report * report,
    const char * prefix,
    int64_t channels) {
    char name[160];
    snprintf(name, sizeof(name), "%s.norm1.weight", prefix);
    expect_vec(st, report, name, TRELLIS_DTYPE_F32, channels);
    snprintf(name, sizeof(name), "%s.norm1.bias", prefix);
    expect_vec(st, report, name, TRELLIS_DTYPE_F32, channels);
    snprintf(name, sizeof(name), "%s.conv1.weight", prefix);
    char bias[160];
    snprintf(bias, sizeof(bias), "%s.conv1.bias", prefix);
    expect_conv3d(st, report, name, bias, TRELLIS_DTYPE_F16, channels, channels);
    snprintf(name, sizeof(name), "%s.norm2.weight", prefix);
    expect_vec(st, report, name, TRELLIS_DTYPE_F32, channels);
    snprintf(name, sizeof(name), "%s.norm2.bias", prefix);
    expect_vec(st, report, name, TRELLIS_DTYPE_F32, channels);
    snprintf(name, sizeof(name), "%s.conv2.weight", prefix);
    snprintf(bias, sizeof(bias), "%s.conv2.bias", prefix);
    expect_conv3d(st, report, name, bias, TRELLIS_DTYPE_F16, channels, channels);
}

trellis_status trellis_ss_decoder_validate_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report) {
    if (safetensors_path == NULL || report == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_checkpoint_report_clear(report);

    trellis_safetensors st;
    trellis_status status = trellis_safetensors_open(safetensors_path, &st);
    if (status != TRELLIS_STATUS_OK) {
        set_first_issue(report, trellis_status_string(status), safetensors_path);
        return status;
    }

    expect_conv3d(&st, report, "input_layer.weight", "input_layer.bias", TRELLIS_DTYPE_F32, 512, 8);
    expect_resblock(&st, report, "middle_block.0", 512);
    expect_resblock(&st, report, "middle_block.1", 512);

    expect_resblock(&st, report, "blocks.0", 512);
    expect_resblock(&st, report, "blocks.1", 512);
    expect_conv3d(&st, report, "blocks.2.conv.weight", "blocks.2.conv.bias", TRELLIS_DTYPE_F16, 1024, 512);
    expect_resblock(&st, report, "blocks.3", 128);
    expect_resblock(&st, report, "blocks.4", 128);
    expect_conv3d(&st, report, "blocks.5.conv.weight", "blocks.5.conv.bias", TRELLIS_DTYPE_F16, 256, 128);
    expect_resblock(&st, report, "blocks.6", 32);
    expect_resblock(&st, report, "blocks.7", 32);

    expect_vec(&st, report, "out_layer.0.weight", TRELLIS_DTYPE_F32, 32);
    expect_vec(&st, report, "out_layer.0.bias", TRELLIS_DTYPE_F32, 32);
    expect_conv3d(&st, report, "out_layer.2.weight", "out_layer.2.bias", TRELLIS_DTYPE_F32, 1, 32);

    status = finish_report(&st, report);
    trellis_safetensors_close(&st);
    return status;
}
