#include "trellis.h"

#include "gguf.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TRELLIS_BIREFNET_PRECOMPUTE_MASKS = TRELLIS_BIREFNET_LAYERS + 1,
};

typedef struct birefnet_graph_aux {
    struct ggml_tensor * rel_pos_index;
    struct ggml_tensor * attn_masks[TRELLIS_BIREFNET_PRECOMPUTE_MASKS];
    int mask_w[TRELLIS_BIREFNET_PRECOMPUTE_MASKS];
    int mask_h[TRELLIS_BIREFNET_PRECOMPUTE_MASKS];
    int32_t * rel_pos_data;
    float * mask_data[TRELLIS_BIREFNET_PRECOMPUTE_MASKS];
} birefnet_graph_aux;

typedef struct birefnet_layer_result {
    struct ggml_tensor * x_out;
    int64_t w_out;
    int64_t h_out;
    struct ggml_tensor * x_down;
    int64_t w_down;
    int64_t h_down;
} birefnet_layer_result;

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

static int birefnet_format(char * dst, size_t dst_size, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(dst, dst_size, fmt, args);
    va_end(args);
    return n;
}

static struct ggml_tensor * weightf(
    trellis_tensor_store * store,
    const char * fmt,
    ...) {
    if (store == NULL || fmt == NULL) {
        return NULL;
    }
    char name[256];
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(name, sizeof(name), fmt, args);
    va_end(args);
    if (n < 0 || (size_t) n >= sizeof(name)) {
        return NULL;
    }
    return trellis_tensor_store_get(store, name);
}

static struct ggml_tensor * repeat_param(
    struct ggml_context * ctx,
    struct ggml_tensor * param,
    struct ggml_tensor * ref) {
    if (param == NULL || ref == NULL) {
        return NULL;
    }
    if (param->type != ref->type && (ref->type == GGML_TYPE_F32 || ref->type == GGML_TYPE_F16)) {
        param = ggml_cast(ctx, param, ref->type);
    }
    return ggml_repeat(ctx, param, ref);
}

static struct ggml_tensor * add_bias_cwhn(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * bias) {
    if (bias == NULL) {
        return x;
    }
    return ggml_add(ctx, x, repeat_param(ctx, bias, x));
}

static struct ggml_tensor * add_bias_whcn(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * bias) {
    if (bias == NULL) {
        return x;
    }
    bias = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    return ggml_add(ctx, x, repeat_param(ctx, bias, x));
}

static struct ggml_tensor * cwhn_to_whcn(struct ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
}

static struct ggml_tensor * whcn_to_cwhn(struct ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
}

static struct ggml_tensor * conv_2d_cwhn_layout(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x,
    int stride,
    int pad,
    int weight_is_cwhn) {
    struct ggml_tensor * w = weightf(store, "%s.weight", prefix);
    struct ggml_tensor * b = weightf(store, "%s.bias", prefix);
    if (w == NULL || x == NULL) {
        return NULL;
    }
    if (weight_is_cwhn) {
        w = cwhn_to_whcn(ctx, w);
    }
    struct ggml_tensor * xw = cwhn_to_whcn(ctx, x);
    struct ggml_tensor * y = ggml_conv_2d_direct(ctx, w, xw, stride, stride, pad, pad, 1, 1);
    y = add_bias_whcn(ctx, y, b);
    return whcn_to_cwhn(ctx, y);
}

static struct ggml_tensor * conv_2d_cwhn(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x,
    int stride,
    int pad) {
    return conv_2d_cwhn_layout(ctx, store, prefix, x, stride, pad, 0);
}

static struct ggml_tensor * linear_prefixed(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x) {
    struct ggml_tensor * w = weightf(store, "%s.weight", prefix);
    struct ggml_tensor * b = weightf(store, "%s.bias", prefix);
    if (w == NULL || x == NULL) {
        return NULL;
    }
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (b != NULL) {
        y = add_bias_cwhn(ctx, y, b);
    }
    return y;
}

static struct ggml_tensor * layer_norm_prefixed(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x) {
    struct ggml_tensor * gamma = weightf(store, "%s.weight", prefix);
    struct ggml_tensor * beta = weightf(store, "%s.bias", prefix);
    if (gamma == NULL || beta == NULL || x == NULL) {
        return NULL;
    }
    struct ggml_tensor * y = ggml_norm(ctx, x, 1e-5f);
    y = ggml_mul(ctx, y, repeat_param(ctx, gamma, y));
    y = ggml_add(ctx, y, repeat_param(ctx, beta, y));
    return y;
}

static struct ggml_tensor * batch_norm_prefixed(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x) {
    struct ggml_tensor * gamma = weightf(store, "%s.weight", prefix);
    struct ggml_tensor * beta = weightf(store, "%s.bias", prefix);
    if (gamma == NULL || beta == NULL || x == NULL) {
        return NULL;
    }
    x = ggml_mul(ctx, x, repeat_param(ctx, gamma, x));
    return ggml_add(ctx, x, repeat_param(ctx, beta, x));
}

static struct ggml_tensor * simple_conv(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x) {
    char p0[256];
    char p1[256];
    birefnet_format(p0, sizeof(p0), "%s.conv1", prefix);
    birefnet_format(p1, sizeof(p1), "%s.conv_out", prefix);
    x = conv_2d_cwhn(ctx, store, p0, x, 1, 1);
    return conv_2d_cwhn(ctx, store, p1, x, 1, 1);
}

static struct ggml_tensor * mean_2d_cwhn(struct ggml_context * ctx, struct ggml_tensor * x) {
    struct ggml_tensor * whcn = cwhn_to_whcn(ctx, x);
    const int64_t w = whcn->ne[0];
    const int64_t h = whcn->ne[1];
    const int64_t c = whcn->ne[2];
    const int64_t n = whcn->ne[3];
    whcn = ggml_reshape_3d(ctx, whcn, w * h, c, n);
    whcn = ggml_mean(ctx, whcn);
    return ggml_reshape_4d(ctx, whcn, c, 1, 1, n);
}

static struct ggml_tensor * upscale_to_cwhn(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * target) {
    struct ggml_tensor * whcn = cwhn_to_whcn(ctx, x);
    whcn = ggml_interpolate(
        ctx,
        whcn,
        target->ne[1],
        target->ne[2],
        whcn->ne[2],
        whcn->ne[3],
        GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
    return whcn_to_cwhn(ctx, whcn);
}

static struct ggml_tensor * upscale_to_whcn(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * target) {
    return ggml_interpolate(
        ctx,
        x,
        target->ne[0],
        target->ne[1],
        x->ne[2],
        x->ne[3],
        GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
}

static struct ggml_tensor * downscale_by_whcn(struct ggml_context * ctx, struct ggml_tensor * x, int factor) {
    return ggml_interpolate(
        ctx,
        x,
        x->ne[0] / factor,
        x->ne[1] / factor,
        x->ne[2],
        x->ne[3],
        GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
}

static struct ggml_tensor * deformable_conv_2d(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x,
    int padding) {
    char p[256];
    birefnet_format(p, sizeof(p), "%s.offset", prefix);
    struct ggml_tensor * offset = conv_2d_cwhn(ctx, store, p, x, 1, padding);
    birefnet_format(p, sizeof(p), "%s.modulator", prefix);
    struct ggml_tensor * modulator = conv_2d_cwhn(ctx, store, p, x, 1, padding);
    modulator = ggml_scale(ctx, ggml_sigmoid(ctx, modulator), 2.0f);
    struct ggml_tensor * kernel = weightf(store, "%s.conv.weight", prefix);
    if (kernel == NULL || offset == NULL || modulator == NULL) {
        return NULL;
    }

    kernel = (kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) ?
        kernel : ggml_cast(ctx, kernel, GGML_TYPE_F32);
    x = cwhn_to_whcn(ctx, x);
    offset = cwhn_to_whcn(ctx, offset);
    modulator = cwhn_to_whcn(ctx, modulator);
    x = x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
    offset = offset->type == GGML_TYPE_F32 ? offset : ggml_cast(ctx, offset, GGML_TYPE_F32);
    modulator = modulator->type == GGML_TYPE_F32 ? modulator : ggml_cast(ctx, modulator, GGML_TYPE_F32);

    struct ggml_tensor * y = ggml_conv_2d_deform(
        ctx,
        kernel,
        x,
        offset,
        modulator,
        1,
        1,
        padding,
        padding);
    return whcn_to_cwhn(ctx, y);
}

static struct ggml_tensor * aspp_module_deformable(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x,
    int padding) {
    char p[256];
    birefnet_format(p, sizeof(p), "%s.conv", prefix);
    x = deformable_conv_2d(ctx, store, p, x, padding);
    birefnet_format(p, sizeof(p), "%s.bn", prefix);
    x = batch_norm_prefixed(ctx, store, p, x);
    return ggml_relu(ctx, x);
}

static struct ggml_tensor * global_avg_pool(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x) {
    char p[256];
    x = mean_2d_cwhn(ctx, x);
    birefnet_format(p, sizeof(p), "%s.1", prefix);
    x = conv_2d_cwhn(ctx, store, p, x, 1, 0);
    return ggml_relu(ctx, x);
}

static struct ggml_tensor * aspp_deformable(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x) {
    char p[256];
    birefnet_format(p, sizeof(p), "%s.aspp1", prefix);
    struct ggml_tensor * x1 = aspp_module_deformable(ctx, store, p, x, 0);
    struct ggml_tensor * xs[5];
    xs[0] = x1;
    const int kernel_sizes[3] = { 1, 3, 7 };
    for (int i = 0; i < 3; ++i) {
        birefnet_format(p, sizeof(p), "%s.aspp_deforms.%d", prefix, i);
        xs[i + 1] = aspp_module_deformable(ctx, store, p, x, kernel_sizes[i] / 2);
    }
    birefnet_format(p, sizeof(p), "%s.global_avg_pool", prefix);
    xs[4] = upscale_to_cwhn(ctx, global_avg_pool(ctx, store, p, x), x1);
    struct ggml_tensor * y = xs[0];
    for (int i = 1; i < 5; ++i) {
        y = ggml_concat(ctx, y, xs[i], 0);
    }
    birefnet_format(p, sizeof(p), "%s.conv1", prefix);
    y = conv_2d_cwhn(ctx, store, p, y, 1, 0);
    return ggml_relu(ctx, y);
}

static struct ggml_tensor * basic_decoder_block(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x) {
    char p[256];
    birefnet_format(p, sizeof(p), "%s.conv_in", prefix);
    x = conv_2d_cwhn(ctx, store, p, x, 1, 1);
    x = ggml_relu(ctx, x);
    birefnet_format(p, sizeof(p), "%s.dec_att", prefix);
    x = aspp_deformable(ctx, store, p, x);
    birefnet_format(p, sizeof(p), "%s.conv_out", prefix);
    return conv_2d_cwhn(ctx, store, p, x, 1, 1);
}

static struct ggml_tensor * image_to_patches_whcn(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int64_t out_w,
    int64_t out_h) {
    const int64_t w = x->ne[0];
    const int64_t h = x->ne[1];
    const int64_t c = x->ne[2];
    const int64_t b = x->ne[3];
    const int64_t grid_w = w / out_w;
    const int64_t grid_h = h / out_h;
    x = ggml_reshape_4d(ctx, x, out_w, grid_w, out_h, grid_h * c * b);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    return ggml_reshape_4d(ctx, x, out_w, out_h, grid_w * grid_h * c, b);
}

static struct ggml_tensor * gdt_conv(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x) {
    char p[256];
    birefnet_format(p, sizeof(p), "%s.0", prefix);
    x = conv_2d_cwhn(ctx, store, p, x, 1, 1);
    return ggml_relu(ctx, x);
}

static struct ggml_tensor * decode(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    struct ggml_tensor * x,
    struct ggml_tensor * features[4]) {
    struct ggml_tensor * x1 = features[0];
    struct ggml_tensor * x2 = features[1];
    struct ggml_tensor * x3 = features[2];
    struct ggml_tensor * x4 = features[3];
    struct ggml_tensor * x_whcn = cwhn_to_whcn(ctx, x);
    char p[256];

    struct ggml_tensor * patches = image_to_patches_whcn(ctx, x_whcn, x4->ne[1], x4->ne[2]);
    patches = whcn_to_cwhn(ctx, patches);
    patches = simple_conv(ctx, store, "decoder.ipt_blk5", patches);
    x4 = ggml_concat(ctx, x4, patches, 0);

    struct ggml_tensor * p4 = basic_decoder_block(ctx, store, "decoder.block4", x4);
    struct ggml_tensor * p4_gdt = gdt_conv(ctx, store, "decoder.gdt_convs_4", p4);
    struct ggml_tensor * gdt_attn = conv_2d_cwhn(ctx, store, "decoder.gdt_convs_attn_4.0", p4_gdt, 1, 0);
    p4 = ggml_mul(ctx, p4, ggml_sigmoid(ctx, gdt_attn));

    x3 = conv_2d_cwhn(ctx, store, "decoder.lateral_block4.conv", x3, 1, 0);
    struct ggml_tensor * p3_in = ggml_add(ctx, upscale_to_cwhn(ctx, p4, x3), x3);
    patches = image_to_patches_whcn(ctx, x_whcn, p3_in->ne[1], p3_in->ne[2]);
    patches = whcn_to_cwhn(ctx, patches);
    patches = simple_conv(ctx, store, "decoder.ipt_blk4", patches);
    p3_in = ggml_concat(ctx, p3_in, patches, 0);

    struct ggml_tensor * p3 = basic_decoder_block(ctx, store, "decoder.block3", p3_in);
    struct ggml_tensor * p3_gdt = gdt_conv(ctx, store, "decoder.gdt_convs_3", p3);
    gdt_attn = conv_2d_cwhn(ctx, store, "decoder.gdt_convs_attn_3.0", p3_gdt, 1, 0);
    p3 = ggml_mul(ctx, p3, ggml_sigmoid(ctx, gdt_attn));

    x2 = conv_2d_cwhn(ctx, store, "decoder.lateral_block3.conv", x2, 1, 0);
    struct ggml_tensor * p2_in = ggml_add(ctx, upscale_to_cwhn(ctx, p3, x2), x2);
    patches = image_to_patches_whcn(ctx, x_whcn, p2_in->ne[1], p2_in->ne[2]);
    patches = whcn_to_cwhn(ctx, patches);
    patches = simple_conv(ctx, store, "decoder.ipt_blk3", patches);
    p2_in = ggml_concat(ctx, p2_in, patches, 0);

    struct ggml_tensor * p2 = basic_decoder_block(ctx, store, "decoder.block2", p2_in);
    struct ggml_tensor * p2_gdt = gdt_conv(ctx, store, "decoder.gdt_convs_2", p2);
    gdt_attn = conv_2d_cwhn(ctx, store, "decoder.gdt_convs_attn_2.0", p2_gdt, 1, 0);
    p2 = ggml_mul(ctx, p2, ggml_sigmoid(ctx, gdt_attn));

    x1 = conv_2d_cwhn(ctx, store, "decoder.lateral_block2.conv", x1, 1, 0);
    struct ggml_tensor * p1_in = ggml_add(ctx, upscale_to_cwhn(ctx, p2, x1), x1);
    patches = image_to_patches_whcn(ctx, x_whcn, p1_in->ne[1], p1_in->ne[2]);
    patches = whcn_to_cwhn(ctx, patches);
    patches = simple_conv(ctx, store, "decoder.ipt_blk2", patches);
    p1_in = ggml_concat(ctx, p1_in, patches, 0);

    p1_in = basic_decoder_block(ctx, store, "decoder.block1", p1_in);
    p1_in = upscale_to_cwhn(ctx, p1_in, x);
    struct ggml_tensor * p1_ipt = simple_conv(ctx, store, "decoder.ipt_blk1", x);
    p1_in = ggml_concat(ctx, p1_in, p1_ipt, 0);
    birefnet_format(p, sizeof(p), "decoder.conv_out1.0");
    return ggml_sigmoid(ctx, conv_2d_cwhn(ctx, store, p, p1_in, 1, 0));
}

static void compute_relative_position_index(int32_t * dst, int window_size) {
    const int n = window_size;
    const int n2 = n * n;
    const int n4 = n2 * n2;
    for (int i = 0; i < n4; ++i) {
        const int x0 = i % n;
        const int y0 = (i / n) % n;
        const int x1 = (i / n2) % n;
        const int y1 = (i / n2 / n) % n;
        dst[i] = (y1 - y0 + n - 1) * (2 * n - 1) + (x1 - x0 + n - 1);
    }
}

static void compute_attention_mask(float * out, int64_t w, int64_t h, int window_size) {
    const int n = window_size;
    const int n2 = n * n;
    const int n4 = n2 * n2;
    const int shift = window_size / 2;
    const int64_t nw_x = (w + n - 1) / n;
    const int64_t nw_y = (h + n - 1) / n;
    const int64_t w_pad = nw_x * n;
    const int64_t h_pad = nw_y * n;
    const size_t count = (size_t) nw_x * (size_t) nw_y * (size_t) n4;
    for (size_t i = 0; i < count; ++i) {
        out[i] = 0.0f;
    }
    for (int64_t iw_y = 0; iw_y < nw_y; ++iw_y) {
        for (int64_t iw_x = 0; iw_x < nw_x; ++iw_x) {
            if (iw_y < nw_y - 1 && iw_x < nw_x - 1) {
                continue;
            }
            const int64_t base = iw_y * nw_x * n4 + iw_x * n4;
            for (int y0 = 0; y0 < n; ++y0) {
                for (int x0 = 0; x0 < n; ++x0) {
                    for (int y1 = 0; y1 < n; ++y1) {
                        for (int x1 = 0; x1 < n; ++x1) {
                            const int yy0 = (int) iw_y * n + y0;
                            const int xx0 = (int) iw_x * n + x0;
                            const int yy1 = (int) iw_y * n + y1;
                            const int xx1 = (int) iw_x * n + x1;
                            const int match_y = (yy0 < h_pad - shift) == (yy1 < h_pad - shift);
                            const int match_x = (xx0 < w_pad - shift) == (xx1 < w_pad - shift);
                            if (!match_y || !match_x) {
                                const int64_t idx = base + (y0 * n + x0) * n2 + (y1 * n + x1);
                                out[idx] = -INFINITY;
                            }
                        }
                    }
                }
            }
        }
    }
}

static int precompute_aux(
    struct ggml_context * ctx,
    const trellis_birefnet_params * params,
    birefnet_graph_aux * aux) {
    memset(aux, 0, sizeof(*aux));
    const int n = params->window_size;
    const int n2 = n * n;
    const int n4 = n2 * n2;
    aux->rel_pos_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n4);
    if (aux->rel_pos_index == NULL) {
        return 0;
    }
    aux->rel_pos_data = (int32_t *) malloc((size_t) n4 * sizeof(int32_t));
    if (aux->rel_pos_data == NULL) {
        return 0;
    }
    compute_relative_position_index(aux->rel_pos_data, n);
    ggml_set_input(aux->rel_pos_index);

    int w = params->image_w / 4;
    int h = params->image_h / 4;
    for (int i = 0; i < TRELLIS_BIREFNET_PRECOMPUTE_MASKS; ++i) {
        const int64_t nw_x = (w + n - 1) / n;
        const int64_t nw_y = (h + n - 1) / n;
        size_t count = 0;
        if (!checked_mul_size((size_t) n2, (size_t) n2, &count) ||
            !checked_mul_size(count, (size_t) nw_x * (size_t) nw_y, &count)) {
            return 0;
        }
        aux->attn_masks[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n2, n2, nw_x * nw_y);
        aux->mask_data[i] = (float *) malloc(count * sizeof(float));
        if (aux->attn_masks[i] == NULL || aux->mask_data[i] == NULL) {
            return 0;
        }
        aux->mask_w[i] = w;
        aux->mask_h[i] = h;
        compute_attention_mask(aux->mask_data[i], w, h, n);
        ggml_set_input(aux->attn_masks[i]);
        w >>= 1;
        h >>= 1;
        if (w < 1) w = 1;
        if (h < 1) h = 1;
    }
    return 1;
}

static struct ggml_tensor * find_attention_mask(
    const birefnet_graph_aux * aux,
    int64_t w,
    int64_t h) {
    for (int i = 0; i < TRELLIS_BIREFNET_PRECOMPUTE_MASKS; ++i) {
        if (aux->mask_w[i] == (int) w && aux->mask_h[i] == (int) h) {
            return aux->attn_masks[i];
        }
    }
    return NULL;
}

static void free_aux(birefnet_graph_aux * aux) {
    if (aux == NULL) {
        return;
    }
    free(aux->rel_pos_data);
    for (int i = 0; i < TRELLIS_BIREFNET_PRECOMPUTE_MASKS; ++i) {
        free(aux->mask_data[i]);
    }
    memset(aux, 0, sizeof(*aux));
}

static struct ggml_tensor * window_partition(struct ggml_context * ctx, struct ggml_tensor * x, int window) {
    const int64_t c = x->ne[0];
    const int64_t w = x->ne[1];
    const int64_t h = x->ne[2];
    const int64_t b = x->ne[3];
    x = ggml_reshape_4d(ctx, x, c * window, w / window, window, (h / window) * b);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    return ggml_reshape_3d(ctx, x, c, window * window, (w / window) * (h / window) * b);
}

static struct ggml_tensor * window_reverse(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int64_t w,
    int64_t h,
    int window) {
    const int64_t c = x->ne[0];
    const int64_t b = x->ne[2] / (w / window) / (h / window);
    x = ggml_reshape_4d(ctx, x, c * window, window, w / window, (h / window) * b);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    return ggml_reshape_4d(ctx, x, c, w, h, b);
}

static struct ggml_tensor * split_qkv(
    struct ggml_context * ctx,
    struct ggml_tensor * qkv,
    int64_t channels,
    int64_t tokens,
    int64_t batches,
    int heads,
    int which) {
    const int64_t head_dim = channels / heads;
    const size_t offset = (size_t) which * qkv->nb[3];
    struct ggml_tensor * part = ggml_view_3d(
        ctx,
        qkv,
        head_dim,
        heads,
        tokens * batches,
        qkv->nb[1],
        qkv->nb[2],
        offset);
    return ggml_reshape_4d(ctx, part, head_dim, heads, tokens, batches);
}

static struct ggml_tensor * window_attention(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x,
    struct ggml_tensor * mask,
    struct ggml_tensor * rel_pos_index,
    int n_heads,
    int window) {
    const int64_t c = x->ne[0];
    const int64_t n = x->ne[1];
    const int64_t b = x->ne[2];
    char p[256];

    struct ggml_tensor * rel_pos_table = weightf(store, "%s.relative_position_bias_table", prefix);
    if (rel_pos_table == NULL) {
        return NULL;
    }
    struct ggml_tensor * rel = ggml_get_rows(ctx, rel_pos_table, rel_pos_index);
    rel = ggml_reshape_4d(ctx, rel, n_heads, n, n, 1);
    rel = ggml_permute(ctx, rel, 2, 0, 1, 3);

    struct ggml_tensor * attn_mask = rel;
    if (mask != NULL) {
        const int64_t n_windows = mask->ne[2];
        if (b > n_windows) {
            mask = ggml_reshape_4d(ctx, mask, n, n, n_windows, 1);
            mask = ggml_repeat_4d(ctx, mask, n, n, n_windows, b / n_windows);
        }
        mask = ggml_reshape_4d(ctx, mask, n, n, 1, b);
        mask = ggml_repeat_4d(ctx, mask, n, n, n_heads, b);
        attn_mask = ggml_add(ctx, mask, rel);
    }

    birefnet_format(p, sizeof(p), "%s.qkv", prefix);
    struct ggml_tensor * qkv = linear_prefixed(ctx, store, p, x);
    qkv = ggml_reshape_4d(ctx, qkv, c / n_heads, n_heads, 3, n * b);
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 1, 3, 2));

    struct ggml_tensor * q = split_qkv(ctx, qkv, c, n, b, n_heads, 0);
    struct ggml_tensor * k = split_qkv(ctx, qkv, c, n, b, n_heads, 1);
    struct ggml_tensor * v = split_qkv(ctx, qkv, c, n, b, n_heads, 2);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

    struct ggml_tensor * attn = ggml_mul_mat(ctx, k, q);
    attn = ggml_soft_max_ext(ctx, attn, attn_mask, 1.0f / sqrtf((float) (c / n_heads)), 0.0f);
    struct ggml_tensor * y = ggml_mul_mat(ctx, v, attn);
    y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));
    y = ggml_reshape_3d(ctx, y, c, n, b);
    birefnet_format(p, sizeof(p), "%s.proj", prefix);
    (void) window;
    return linear_prefixed(ctx, store, p, y);
}

static struct ggml_tensor * swin_mlp(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x) {
    char p[256];
    birefnet_format(p, sizeof(p), "%s.fc1", prefix);
    x = linear_prefixed(ctx, store, p, x);
    x = ggml_gelu(ctx, x);
    birefnet_format(p, sizeof(p), "%s.fc2", prefix);
    return linear_prefixed(ctx, store, p, x);
}

static struct ggml_tensor * swin_block(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x,
    struct ggml_tensor * mask,
    struct ggml_tensor * rel_pos_index,
    int n_heads,
    int window,
    int64_t w,
    int64_t h,
    int shift) {
    const int64_t c = x->ne[0];
    const int64_t n = x->ne[1];
    const int64_t b = x->ne[2];
    char p[256];
    struct ggml_tensor * shortcut = x;
    birefnet_format(p, sizeof(p), "%s.norm1", prefix);
    x = layer_norm_prefixed(ctx, store, p, x);
    x = ggml_reshape_4d(ctx, x, c, w, h, b);

    const int pad_r = (window - (int) (w % window)) % window;
    const int pad_b = (window - (int) (h % window)) % window;
    if (pad_r > 0 || pad_b > 0) {
        x = ggml_pad(ctx, x, 0, pad_r, pad_b, 0);
    }
    if (shift > 0) {
        x = ggml_roll(ctx, x, 0, -shift, -shift, 0);
    }
    x = window_partition(ctx, x, window);
    birefnet_format(p, sizeof(p), "%s.attn", prefix);
    x = window_attention(ctx, store, p, x, mask, rel_pos_index, n_heads, window);
    x = window_reverse(ctx, x, w + pad_r, h + pad_b, window);
    if (shift > 0) {
        x = ggml_roll(ctx, x, 0, shift, shift, 0);
    }
    if (pad_r > 0 || pad_b > 0) {
        x = ggml_view_4d(ctx, x, c, w, h, b, x->nb[1], x->nb[2], x->nb[3], 0);
        x = ggml_cont_4d(ctx, x, c, w, h, b);
    }
    x = ggml_reshape_3d(ctx, x, c, n, b);
    x = ggml_add(ctx, x, shortcut);

    birefnet_format(p, sizeof(p), "%s.norm2", prefix);
    struct ggml_tensor * xm = layer_norm_prefixed(ctx, store, p, x);
    birefnet_format(p, sizeof(p), "%s.mlp", prefix);
    xm = swin_mlp(ctx, store, p, xm);
    return ggml_add(ctx, x, xm);
}

static struct ggml_tensor * view_even_odd(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int off_x,
    int off_y) {
    const int64_t c = x->ne[0];
    const int64_t w = x->ne[1] / 2;
    const int64_t h = x->ne[2] / 2;
    const int64_t b = x->ne[3];
    const size_t offset = (size_t) off_x * x->nb[1] + (size_t) off_y * x->nb[2];
    return ggml_view_4d(ctx, x, c, w, h, b, x->nb[1] * 2, x->nb[2] * 2, x->nb[3], offset);
}

static struct ggml_tensor * patch_merging(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x,
    int64_t w,
    int64_t h) {
    const int64_t c = x->ne[0];
    const int64_t n = x->ne[1];
    const int64_t b = x->ne[2];
    x = ggml_reshape_4d(ctx, x, c, w, h, b);
    struct ggml_tensor * x0 = view_even_odd(ctx, x, 0, 0);
    struct ggml_tensor * x1 = view_even_odd(ctx, x, 0, 1);
    struct ggml_tensor * x2 = view_even_odd(ctx, x, 1, 0);
    struct ggml_tensor * x3 = view_even_odd(ctx, x, 1, 1);
    x = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, x0, x1, 0), x2, 0), x3, 0);
    x = ggml_reshape_3d(ctx, x, c * 4, n / 4, b);
    char p[256];
    birefnet_format(p, sizeof(p), "%s.norm", prefix);
    x = layer_norm_prefixed(ctx, store, p, x);
    birefnet_format(p, sizeof(p), "%s.reduction", prefix);
    return linear_prefixed(ctx, store, p, x);
}

static birefnet_layer_result swin_layer(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    const char * prefix,
    struct ggml_tensor * x,
    int64_t w,
    int64_t h,
    const trellis_birefnet_params * params,
    const birefnet_graph_aux * aux,
    int layer_index,
    int downsample) {
    char p[256];
    struct ggml_tensor * mask = find_attention_mask(aux, w, h);
    for (int i = 0; i < params->layer_depths[layer_index]; ++i) {
        birefnet_format(p, sizeof(p), "%s.blocks.%d", prefix, i);
        x = swin_block(
            ctx,
            store,
            p,
            x,
            mask,
            aux->rel_pos_index,
            params->layer_heads[layer_index],
            params->window_size,
            w,
            h,
            (i % 2 == 0) ? 0 : params->window_size / 2);
    }
    birefnet_layer_result r;
    r.x_out = x;
    r.w_out = w;
    r.h_out = h;
    if (downsample) {
        birefnet_format(p, sizeof(p), "%s.downsample", prefix);
        r.x_down = patch_merging(ctx, store, p, x, w, h);
        r.w_down = (w + 1) / 2;
        r.h_down = (h + 1) / 2;
    } else {
        r.x_down = x;
        r.w_down = w;
        r.h_down = h;
    }
    return r;
}

static struct ggml_tensor * patch_embed(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    struct ggml_tensor * x) {
    x = conv_2d_cwhn_layout(ctx, store, "bb.patch_embed.proj", x, 4, 0, 1);
    if (weightf(store, "bb.patch_embed.norm.weight") != NULL) {
        const int64_t c = x->ne[0];
        const int64_t w = x->ne[1];
        const int64_t h = x->ne[2];
        const int64_t b = x->ne[3];
        x = ggml_reshape_3d(ctx, x, c, w * h, b);
        x = layer_norm_prefixed(ctx, store, "bb.patch_embed.norm", x);
        x = ggml_reshape_4d(ctx, x, c, w, h, b);
    }
    return x;
}

static void swin_encode(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    struct ggml_tensor * x,
    const trellis_birefnet_params * params,
    const birefnet_graph_aux * aux,
    struct ggml_tensor * outs[4]) {
    x = patch_embed(ctx, store, x);
    const int64_t c = x->ne[0];
    const int64_t w = x->ne[1];
    const int64_t h = x->ne[2];
    const int64_t b = x->ne[3];
    x = ggml_reshape_3d(ctx, x, c, w * h, b);
    birefnet_layer_result r;
    r.x_down = x;
    r.w_down = w;
    r.h_down = h;
    for (int i = 0; i < TRELLIS_BIREFNET_LAYERS; ++i) {
        char p[256];
        birefnet_format(p, sizeof(p), "bb.layers.%d", i);
        r = swin_layer(ctx, store, p, r.x_down, r.w_down, r.h_down, params, aux, i, i < 3);
        birefnet_format(p, sizeof(p), "bb.norm%d", i);
        struct ggml_tensor * out = layer_norm_prefixed(ctx, store, p, r.x_out);
        outs[i] = ggml_reshape_4d(ctx, out, params->layer_features[i], r.w_out, r.h_out, b);
    }
}

static struct ggml_tensor * downscale_by_cwhn(struct ggml_context * ctx, struct ggml_tensor * x, int factor) {
    struct ggml_tensor * whcn = cwhn_to_whcn(ctx, x);
    whcn = downscale_by_whcn(ctx, whcn, factor);
    return whcn_to_cwhn(ctx, whcn);
}

static void birefnet_encode(
    struct ggml_context * ctx,
    trellis_tensor_store * store,
    struct ggml_tensor * x,
    const trellis_birefnet_params * params,
    const birefnet_graph_aux * aux,
    struct ggml_tensor * xs[4]) {
    struct ggml_tensor * xs_low[4];
    swin_encode(ctx, store, x, params, aux, xs);
    struct ggml_tensor * x_low = downscale_by_cwhn(ctx, x, 2);
    swin_encode(ctx, store, x_low, params, aux, xs_low);

    struct ggml_tensor * xs_whcn[4];
    struct ggml_tensor * low_whcn[4];
    for (int i = 0; i < 4; ++i) {
        xs_whcn[i] = cwhn_to_whcn(ctx, xs[i]);
        low_whcn[i] = cwhn_to_whcn(ctx, xs_low[i]);
        xs_whcn[i] = ggml_concat(ctx, xs_whcn[i], upscale_to_whcn(ctx, low_whcn[i], xs_whcn[i]), 2);
    }
    struct ggml_tensor * x4 = ggml_concat(ctx, downscale_by_whcn(ctx, xs_whcn[0], 8), downscale_by_whcn(ctx, xs_whcn[1], 4), 2);
    x4 = ggml_concat(ctx, x4, downscale_by_whcn(ctx, xs_whcn[2], 2), 2);
    xs_whcn[3] = ggml_concat(ctx, x4, xs_whcn[3], 2);
    for (int i = 0; i < 4; ++i) {
        xs[i] = whcn_to_cwhn(ctx, xs_whcn[i]);
    }
}

static struct ggml_tensor * birefnet_forward(
    struct ggml_context * ctx,
    trellis_birefnet_model * model,
    struct ggml_tensor * image,
    const birefnet_graph_aux * aux) {
    struct ggml_tensor * features[4];
    birefnet_encode(ctx, &model->store, image, &model->params, aux, features);
    features[3] = basic_decoder_block(ctx, &model->store, "squeeze_module.0", features[3]);
    return decode(ctx, &model->store, image, features);
}

static int get_gguf_i32(struct gguf_context * gguf, const char * key, int * out) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_INT32) {
        return 0;
    }
    *out = gguf_get_val_i32(gguf, id);
    return 1;
}

static int get_gguf_string(struct gguf_context * gguf, const char * key, const char ** out) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
        return 0;
    }
    *out = gguf_get_val_str(gguf, id);
    return 1;
}

static void set_swin_params(trellis_birefnet_params * params, int embed_dim) {
    params->embed_dim = embed_dim;
    if (embed_dim == 96) {
        params->window_size = 7;
        const int depths[4] = { 2, 2, 6, 2 };
        const int heads[4] = { 3, 6, 12, 24 };
        for (int i = 0; i < 4; ++i) {
            params->layer_depths[i] = depths[i];
            params->layer_heads[i] = heads[i];
            params->layer_features[i] = embed_dim << i;
        }
    } else {
        params->window_size = 12;
        const int depths[4] = { 2, 2, 18, 2 };
        const int heads[4] = { 6, 12, 24, 48 };
        for (int i = 0; i < 4; ++i) {
            params->layer_depths[i] = depths[i];
            params->layer_heads[i] = heads[i];
            params->layer_features[i] = embed_dim << i;
        }
    }
}

trellis_status trellis_birefnet_load_gguf_with_backend(
    trellis_birefnet_model * model,
    const char * gguf_path,
    trellis_backend_kind backend_kind,
    int device) {
    if (model == NULL || gguf_path == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(model, 0, sizeof(*model));

    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = NULL,
    };
    struct gguf_context * gguf = gguf_init_from_file(gguf_path, params);
    if (gguf == NULL) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    const char * arch = NULL;
    int embed_dim = 0;
    if (!get_gguf_string(gguf, "general.architecture", &arch) ||
        strcmp(arch, "birefnet") != 0 ||
        !get_gguf_i32(gguf, "birefnet.image_size", &model->params.image_size) ||
        !get_gguf_i32(gguf, "birefnet.image_multiple", &model->params.image_multiple) ||
        !get_gguf_i32(gguf, "swin.embed_dim", &embed_dim) ||
        (embed_dim != 96 && embed_dim != 192)) {
        gguf_free(gguf);
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    gguf_free(gguf);
    set_swin_params(&model->params, embed_dim);

    trellis_status status = trellis_backend_init(&model->backend, backend_kind, device);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    size_t loaded = 0;
    status = trellis_tensor_store_load_gguf(&model->store, &model->backend, gguf_path, &loaded);
    if (status != TRELLIS_STATUS_OK) {
        trellis_backend_free(&model->backend);
        return status;
    }
    TRELLIS_INFO(
        "BiRefNet: loaded %zu GGUF tensors on %s",
        loaded,
        trellis_backend_kind_name(model->backend.kind));
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_birefnet_load_gguf(
    trellis_birefnet_model * model,
    const char * gguf_path) {
    return trellis_birefnet_load_gguf_with_backend(model, gguf_path, TRELLIS_BACKEND_CPU, 0);
}

void trellis_birefnet_free(trellis_birefnet_model * model) {
    if (model == NULL) {
        return;
    }
    trellis_tensor_store_free(&model->store);
    trellis_backend_free(&model->backend);
    memset(model, 0, sizeof(*model));
}

static unsigned char sample_rgba(
    const unsigned char * rgba,
    int width,
    int height,
    int x,
    int y,
    int c) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= width) x = width - 1;
    if (y >= height) y = height - 1;
    return rgba[((size_t) y * (size_t) width + (size_t) x) * 4u + (size_t) c];
}

static void resize_rgba_to_birefnet_input(
    const unsigned char * rgba,
    int width,
    int height,
    int out_w,
    int out_h,
    float * out) {
    const float mean[3] = { 0.485f, 0.456f, 0.406f };
    const float std[3] = { 0.229f, 0.224f, 0.225f };
    for (int y = 0; y < out_h; ++y) {
        const float sy = ((float) y + 0.5f) * (float) height / (float) out_h - 0.5f;
        const int y0 = (int) floorf(sy);
        const int y1 = y0 + 1;
        const float dy = sy - (float) y0;
        for (int x = 0; x < out_w; ++x) {
            const float sx = ((float) x + 0.5f) * (float) width / (float) out_w - 0.5f;
            const int x0 = (int) floorf(sx);
            const int x1 = x0 + 1;
            const float dx = sx - (float) x0;
            for (int c = 0; c < 3; ++c) {
                const float v00 = (float) sample_rgba(rgba, width, height, x0, y0, c);
                const float v01 = (float) sample_rgba(rgba, width, height, x1, y0, c);
                const float v10 = (float) sample_rgba(rgba, width, height, x0, y1, c);
                const float v11 = (float) sample_rgba(rgba, width, height, x1, y1, c);
                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;
                const float rgb = (v0 * (1.0f - dy) + v1 * dy) / 255.0f;
                out[((size_t) y * (size_t) out_w + (size_t) x) * 3u + (size_t) c] = (rgb - mean[c]) / std[c];
            }
        }
    }
}

static float sample_mask(const float * mask, int width, int height, float x, float y) {
    const int x0 = (int) floorf(x);
    const int y0 = (int) floorf(y);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float dx = x - (float) x0;
    const float dy = y - (float) y0;
    float v00 = 0.0f;
    float v01 = 0.0f;
    float v10 = 0.0f;
    float v11 = 0.0f;
    if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) v00 = mask[(size_t) y0 * (size_t) width + (size_t) x0];
    if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) v01 = mask[(size_t) y0 * (size_t) width + (size_t) x1];
    if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) v10 = mask[(size_t) y1 * (size_t) width + (size_t) x0];
    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) v11 = mask[(size_t) y1 * (size_t) width + (size_t) x1];
    const float v0 = v00 * (1.0f - dx) + v01 * dx;
    const float v1 = v10 * (1.0f - dx) + v11 * dx;
    return v0 * (1.0f - dy) + v1 * dy;
}

trellis_status trellis_birefnet_compute_mask_u8(
    trellis_birefnet_model * model,
    const unsigned char * rgba,
    int width,
    int height,
    unsigned char ** mask_out) {
    if (model == NULL || rgba == NULL || width <= 0 || height <= 0 || mask_out == NULL ||
        model->backend.backend == NULL || model->store.ctx == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *mask_out = NULL;

    int image_w = model->params.image_size > 0 ? model->params.image_size : width;
    int image_h = model->params.image_size > 0 ? model->params.image_size : height;
    const int multiple = model->params.image_multiple > 0 ? model->params.image_multiple : 128;
    image_w = ((image_w + multiple - 1) / multiple) * multiple;
    image_h = ((image_h + multiple - 1) / multiple) * multiple;
    model->params.image_w = image_w;
    model->params.image_h = image_h;

    float * input_data = (float *) malloc((size_t) image_w * (size_t) image_h * 3u * sizeof(float));
    if (input_data == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    resize_rgba_to_birefnet_input(rgba, width, height, image_w, image_h, input_data);

    const size_t graph_nodes = 131072;
    struct ggml_init_params init = {
        .mem_size = ggml_tensor_overhead() * graph_nodes +
            ggml_graph_overhead_custom(graph_nodes, false) + 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(init);
    if (ctx == NULL) {
        free(input_data);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    birefnet_graph_aux aux;
    if (!precompute_aux(ctx, &model->params, &aux)) {
        free_aux(&aux);
        ggml_free(ctx);
        free(input_data);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    struct ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, image_w, image_h, 1);
    if (input == NULL) {
        free_aux(&aux);
        ggml_free(ctx);
        free(input_data);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_set_input(input);
    struct ggml_tensor * output = birefnet_forward(ctx, model, input, &aux);
    if (output == NULL) {
        free_aux(&aux);
        ggml_free(ctx);
        free(input_data);
        return TRELLIS_STATUS_ERROR;
    }
    ggml_set_output(output);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    if (graph == NULL) {
        free_aux(&aux);
        ggml_free(ctx);
        free(input_data);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t alloc = trellis_backend_new_graph_allocator(&model->backend);
    if (alloc == NULL || !ggml_gallocr_alloc_graph(alloc, graph)) {
        if (alloc != NULL) {
            ggml_gallocr_free(alloc);
        }
        free_aux(&aux);
        ggml_free(ctx);
        free(input_data);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    ggml_backend_tensor_set(input, input_data, 0, ggml_nbytes(input));
    ggml_backend_tensor_set(aux.rel_pos_index, aux.rel_pos_data, 0, ggml_nbytes(aux.rel_pos_index));
    for (int i = 0; i < TRELLIS_BIREFNET_PRECOMPUTE_MASKS; ++i) {
        ggml_backend_tensor_set(aux.attn_masks[i], aux.mask_data[i], 0, ggml_nbytes(aux.attn_masks[i]));
    }

    trellis_status status = trellis_backend_compute_graph(&model->backend, graph);
    float * mask_f32 = NULL;
    if (status == TRELLIS_STATUS_OK) {
        mask_f32 = (float *) malloc((size_t) image_w * (size_t) image_h * sizeof(float));
        if (mask_f32 == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else {
            ggml_backend_tensor_get(output, mask_f32, 0, ggml_nbytes(output));
        }
    }

    unsigned char * mask_u8 = NULL;
    if (status == TRELLIS_STATUS_OK) {
        mask_u8 = (unsigned char *) malloc((size_t) width * (size_t) height);
        if (mask_u8 == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else {
            for (int y = 0; y < height; ++y) {
                const float sy = ((float) y + 0.5f) * (float) image_h / (float) height - 0.5f;
                for (int x = 0; x < width; ++x) {
                    const float sx = ((float) x + 0.5f) * (float) image_w / (float) width - 0.5f;
                    float v = sample_mask(mask_f32, image_w, image_h, sx, sy);
                    if (v < 0.0f) v = 0.0f;
                    if (v > 1.0f) v = 1.0f;
                    mask_u8[(size_t) y * (size_t) width + (size_t) x] = (unsigned char) floorf(v * 255.0f + 0.5f);
                }
            }
            *mask_out = mask_u8;
        }
    }

    if (status != TRELLIS_STATUS_OK) {
        free(mask_u8);
    }
    free(mask_f32);
    ggml_gallocr_free(alloc);
    free_aux(&aux);
    ggml_free(ctx);
    free(input_data);
    return status;
}
