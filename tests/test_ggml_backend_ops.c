#include "trellis.h"
#include "trellis_ggml_layers.h"
#include "trellis_platform.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return 0; \
    } \
} while (0)

static int check_close(float got, float exp, float tol, const char * name, int idx) {
    const float diff = fabsf(got - exp);
    const float scale = fmaxf(1.0f, fmaxf(fabsf(got), fabsf(exp)));
    if (diff <= tol * scale) {
        return 1;
    }
    fprintf(stderr, "%s[%d] mismatch: got=%g expected=%g diff=%g tol=%g\n", name, idx, got, exp, diff, tol);
    return 0;
}

static double wall_clock_ms(void) {
    return (double) trellis_now_us() / 1000.0;
}

static int conv_out_size_ref(int input, int kernel, int stride, int pad, int dilation) {
    return (input + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

static void conv3d_ref(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w) {
    const int out_d = conv_out_size_ref(in_d, kernel_d, stride_d, pad_d, dilation_d);
    const int out_h = conv_out_size_ref(in_h, kernel_h, stride_h, pad_h, dilation_h);
    const int out_w = conv_out_size_ref(in_w, kernel_w, stride_w, pad_w, dilation_w);
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int od = 0; od < out_d; ++od) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        float acc = bias == NULL ? 0.0f : bias[oc];
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kd = 0; kd < kernel_d; ++kd) {
                                const int id = od * stride_d + kd * dilation_d - pad_d;
                                if (id < 0 || id >= in_d) continue;
                                for (int kh = 0; kh < kernel_h; ++kh) {
                                    const int ih = oh * stride_h + kh * dilation_h - pad_h;
                                    if (ih < 0 || ih >= in_h) continue;
                                    for (int kw = 0; kw < kernel_w; ++kw) {
                                        const int iw = ow * stride_w + kw * dilation_w - pad_w;
                                        if (iw < 0 || iw >= in_w) continue;
                                        const int64_t x_idx =
                                            (((int64_t) b * in_channels + ic) * in_d + id) * in_h * (int64_t) in_w +
                                            (int64_t) ih * in_w + iw;
                                        const int64_t w_idx =
                                            (((int64_t) oc * in_channels + ic) * kernel_d + kd) * kernel_h * (int64_t) kernel_w +
                                            (int64_t) kh * kernel_w + kw;
                                        acc += x[x_idx] * weight[w_idx];
                                    }
                                }
                            }
                        }
                        const int64_t y_idx =
                            (((int64_t) b * out_channels + oc) * out_d + od) * out_h * (int64_t) out_w +
                            (int64_t) oh * out_w + ow;
                        y[y_idx] = acc;
                    }
                }
            }
        }
    }
}

static void pixel_shuffle_3d_ref(
    const float * x,
    float * y,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int scale) {
    const int out_channels = in_channels / (scale * scale * scale);
    const int out_d = in_d * scale;
    const int out_h = in_h * scale;
    const int out_w = in_w * scale;
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int od = 0; od < out_d; ++od) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        const int id = od / scale;
                        const int ih = oh / scale;
                        const int iw = ow / scale;
                        const int rd = od - id * scale;
                        const int rh = oh - ih * scale;
                        const int rw = ow - iw * scale;
                        const int ic = (((oc * scale) + rd) * scale + rh) * scale + rw;
                        const int64_t x_idx =
                            (((int64_t) b * in_channels + ic) * in_d + id) * in_h * (int64_t) in_w +
                            (int64_t) ih * in_w + iw;
                        const int64_t y_idx =
                            (((int64_t) b * out_channels + oc) * out_d + od) * out_h * (int64_t) out_w +
                            (int64_t) oh * out_w + ow;
                        y[y_idx] = x[x_idx];
                    }
                }
            }
        }
    }
}

static struct ggml_context * make_graph_ctx(void) {
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead() + 4096,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    return ggml_init(params);
}

static int test_conv3d(trellis_backend_context * backend) {
    enum {
        B = 2, IC = 2, ID = 3, IH = 3, IW = 4,
        OC = 3, KD = 2, KH = 2, KW = 3,
        SD = 1, SH = 2, SW = 1,
        PD = 1, PH = 0, PW = 1,
        DD = 1, DH = 1, DW = 1,
        OD = (ID + 2 * PD - DD * (KD - 1) - 1) / SD + 1,
        OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1,
        OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1,
    };
    float x[B * IC * ID * IH * IW];
    float w[OC * IC * KD * KH * KW];
    float bias[OC];
    float got[B * OC * OD * OH * OW];
    float exp[B * OC * OD * OH * OW];
    for (int i = 0; i < (int) (sizeof(x) / sizeof(x[0])); ++i) {
        x[i] = sinf(0.17f * (float) i) + 0.1f * (float) (i % 5);
    }
    for (int i = 0; i < (int) (sizeof(w) / sizeof(w[0])); ++i) {
        w[i] = cosf(0.11f * (float) (i + 3)) * 0.25f;
    }
    for (int i = 0; i < OC; ++i) {
        bias[i] = 0.05f * (float) (i - 1);
    }
    conv3d_ref(x, w, bias, exp, B, IC, ID, IH, IW, OC, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW);

    struct ggml_context * ctx = make_graph_ctx();
    CHECK_TRUE(ctx != NULL);
    struct ggml_tensor * x_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IW, IH, ID, IC * B);
    struct ggml_tensor * w_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, KH, KD, IC * OC);
    struct ggml_tensor * b_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, OC);
    struct ggml_tensor * y = ggml_conv_3d_direct(ctx, w_t, x_t, SW, SH, SD, PW, PH, PD, DW, DH, DD, IC, B, OC);
    y = ggml_add(ctx, y, ggml_repeat(ctx, b_t, y));
    CHECK_TRUE(x_t != NULL && w_t != NULL && b_t != NULL && y != NULL);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_backend_new_graph_allocator(backend);
    CHECK_TRUE(alloc != NULL);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(x_t, x, 0, ggml_nbytes(x_t));
    ggml_backend_tensor_set(w_t, w, 0, ggml_nbytes(w_t));
    ggml_backend_tensor_set(b_t, bias, 0, ggml_nbytes(b_t));
    CHECK_TRUE(trellis_backend_compute_graph(backend, graph) == TRELLIS_STATUS_OK);
    ggml_backend_tensor_get(y, got, 0, ggml_nbytes(y));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    for (int i = 0; i < B * OC * OD * OH * OW; ++i) {
        CHECK_TRUE(check_close(got[i], exp[i], 1e-3f, "conv3d", i));
    }
    return 1;
}

static int test_pixel_shuffle_3d(trellis_backend_context * backend) {
    enum { B = 1, OC = 2, SCALE = 2, IC = OC * SCALE * SCALE * SCALE, ID = 2, IH = 2, IW = 1 };
    enum { OD = ID * SCALE, OH = IH * SCALE, OW = IW * SCALE };
    float x[B * IC * ID * IH * IW];
    float got[B * OC * OD * OH * OW];
    float exp[B * OC * OD * OH * OW];
    for (int i = 0; i < (int) (sizeof(x) / sizeof(x[0])); ++i) {
        x[i] = (float) (i + 1);
    }
    pixel_shuffle_3d_ref(x, exp, B, IC, ID, IH, IW, SCALE);

    struct ggml_context * ctx = make_graph_ctx();
    CHECK_TRUE(ctx != NULL);
    struct ggml_tensor * x_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IW, IH, ID, IC * B);
    struct ggml_tensor * y = ggml_pixel_shuffle_3d(ctx, x_t, SCALE);
    CHECK_TRUE(x_t != NULL && y != NULL);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_backend_new_graph_allocator(backend);
    CHECK_TRUE(alloc != NULL);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(x_t, x, 0, ggml_nbytes(x_t));
    CHECK_TRUE(trellis_backend_compute_graph(backend, graph) == TRELLIS_STATUS_OK);
    ggml_backend_tensor_get(y, got, 0, ggml_nbytes(y));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    for (int i = 0; i < B * OC * OD * OH * OW; ++i) {
        CHECK_TRUE(check_close(got[i], exp[i], 1e-6f, "pixel_shuffle_3d", i));
    }
    return 1;
}

static int test_project_attention(trellis_backend_context * backend) {
    enum {
        CHANNELS = 4,
        PROJ_IN = 3,
        GLOBAL_CHANNELS = 3,
        TOKENS = 3,
        GLOBAL_TOKENS = 2,
        HEADS = 2,
        HEAD_DIM = CHANNELS / HEADS,
        BATCHES = 2,
    };

    float x_data[CHANNELS * TOKENS * BATCHES];
    float global_data[GLOBAL_CHANNELS * GLOBAL_TOKENS * BATCHES];
    float projected_data[PROJ_IN * TOKENS * BATCHES];
    float q_w_data[CHANNELS * CHANNELS];
    float q_b_data[CHANNELS];
    float kv_w_data[2 * CHANNELS * GLOBAL_CHANNELS];
    float kv_b_data[2 * CHANNELS];
    float out_w_data[CHANNELS * CHANNELS];
    float out_b_data[CHANNELS];
    float proj_w_data[CHANNELS * PROJ_IN];
    float proj_b_data[CHANNELS];
    float got[CHANNELS * TOKENS * BATCHES];
    float exp[CHANNELS * TOKENS * BATCHES];

    for (int b = 0; b < BATCHES; ++b) {
        for (int t = 0; t < TOKENS; ++t) {
            for (int c = 0; c < CHANNELS; ++c) {
                x_data[(b * TOKENS + t) * CHANNELS + c] =
                    0.09f * (float) (b + 1) + 0.04f * (float) t - 0.025f * (float) c;
            }
            for (int c = 0; c < PROJ_IN; ++c) {
                projected_data[(b * TOKENS + t) * PROJ_IN + c] =
                    -0.08f + 0.06f * (float) b + 0.035f * (float) t + 0.02f * (float) c;
            }
        }
        for (int t = 0; t < GLOBAL_TOKENS; ++t) {
            for (int c = 0; c < GLOBAL_CHANNELS; ++c) {
                global_data[(b * GLOBAL_TOKENS + t) * GLOBAL_CHANNELS + c] =
                    0.11f * (float) (t + 1) - 0.045f * (float) b + 0.03f * (float) c;
            }
        }
    }
    for (int o = 0; o < CHANNELS; ++o) {
        q_b_data[o] = 0.012f * (float) (o - 1);
        out_b_data[o] = -0.017f + 0.009f * (float) o;
        proj_b_data[o] = 0.021f - 0.008f * (float) o;
        for (int i = 0; i < CHANNELS; ++i) {
            q_w_data[o * CHANNELS + i] = 0.035f * (float) (o + 1) - 0.012f * (float) i;
            out_w_data[o * CHANNELS + i] = 0.028f * (float) (i + 1) - 0.01f * (float) o;
        }
        for (int i = 0; i < PROJ_IN; ++i) {
            proj_w_data[o * PROJ_IN + i] = -0.03f + 0.014f * (float) (2 * o + i);
        }
    }
    for (int o = 0; o < 2 * CHANNELS; ++o) {
        kv_b_data[o] = -0.015f + 0.006f * (float) o;
        for (int i = 0; i < GLOBAL_CHANNELS; ++i) {
            kv_w_data[o * GLOBAL_CHANNELS + i] =
                0.018f * (float) (o + 1) - 0.011f * (float) i;
        }
    }

    for (int b = 0; b < BATCHES; ++b) {
        for (int t = 0; t < TOKENS; ++t) {
            float q[CHANNELS];
            float attended[CHANNELS];
            for (int o = 0; o < CHANNELS; ++o) {
                q[o] = q_b_data[o];
                attended[o] = 0.0f;
                for (int i = 0; i < CHANNELS; ++i) {
                    q[o] += q_w_data[o * CHANNELS + i] *
                        x_data[(b * TOKENS + t) * CHANNELS + i];
                }
            }
            for (int h = 0; h < HEADS; ++h) {
                float logits[GLOBAL_TOKENS];
                float values[GLOBAL_TOKENS][HEAD_DIM];
                float max_logit = -1e30f;
                for (int kt = 0; kt < GLOBAL_TOKENS; ++kt) {
                    float dot = 0.0f;
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        const int channel = h * HEAD_DIM + d;
                        float key = kv_b_data[channel];
                        float value = kv_b_data[CHANNELS + channel];
                        for (int i = 0; i < GLOBAL_CHANNELS; ++i) {
                            const float input = global_data[
                                (b * GLOBAL_TOKENS + kt) * GLOBAL_CHANNELS + i];
                            key += kv_w_data[channel * GLOBAL_CHANNELS + i] * input;
                            value += kv_w_data[(CHANNELS + channel) * GLOBAL_CHANNELS + i] * input;
                        }
                        dot += q[channel] * key;
                        values[kt][d] = value;
                    }
                    logits[kt] = dot / sqrtf((float) HEAD_DIM);
                    if (logits[kt] > max_logit) max_logit = logits[kt];
                }
                float denom = 0.0f;
                for (int kt = 0; kt < GLOBAL_TOKENS; ++kt) {
                    logits[kt] = expf(logits[kt] - max_logit);
                    denom += logits[kt];
                }
                for (int d = 0; d < HEAD_DIM; ++d) {
                    for (int kt = 0; kt < GLOBAL_TOKENS; ++kt) {
                        attended[h * HEAD_DIM + d] += logits[kt] / denom * values[kt][d];
                    }
                }
            }

            for (int o = 0; o < CHANNELS; ++o) {
                float value = out_b_data[o] + proj_b_data[o];
                for (int i = 0; i < CHANNELS; ++i) {
                    value += out_w_data[o * CHANNELS + i] * attended[i];
                }
                for (int i = 0; i < PROJ_IN; ++i) {
                    value += proj_w_data[o * PROJ_IN + i] *
                        projected_data[(b * TOKENS + t) * PROJ_IN + i];
                }
                exp[(b * TOKENS + t) * CHANNELS + o] = value;
            }
        }
    }

    struct ggml_context * ctx = make_graph_ctx();
    CHECK_TRUE(ctx != NULL);
    struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, CHANNELS, TOKENS, BATCHES);
    struct ggml_tensor * global = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, GLOBAL_CHANNELS, GLOBAL_TOKENS, BATCHES);
    struct ggml_tensor * projected = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, PROJ_IN, TOKENS, BATCHES);
    struct ggml_tensor * q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CHANNELS, CHANNELS);
    struct ggml_tensor * q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, CHANNELS);
    struct ggml_tensor * kv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GLOBAL_CHANNELS, 2 * CHANNELS);
    struct ggml_tensor * kv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * CHANNELS);
    struct ggml_tensor * out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CHANNELS, CHANNELS);
    struct ggml_tensor * out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, CHANNELS);
    struct ggml_tensor * proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PROJ_IN, CHANNELS);
    struct ggml_tensor * proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, CHANNELS);
    CHECK_TRUE(x != NULL && global != NULL && projected != NULL && q_w != NULL && q_b != NULL &&
        kv_w != NULL && kv_b != NULL && out_w != NULL && out_b != NULL && proj_w != NULL && proj_b != NULL);

    struct ggml_tensor * y = trellis_ggml_project_attention(
        ctx, x, global, projected, HEADS,
        q_w, q_b, kv_w, kv_b, NULL, NULL, out_w, out_b, proj_w, proj_b);
    CHECK_TRUE(y != NULL);
    CHECK_TRUE(y->ne[0] == CHANNELS && y->ne[1] == TOKENS && y->ne[2] == BATCHES);

    struct ggml_tensor * bad_tokens = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, PROJ_IN, TOKENS + 1, BATCHES);
    struct ggml_tensor * bad_batches = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, PROJ_IN, TOKENS, BATCHES + 1);
    struct ggml_tensor * bad_proj_w = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, PROJ_IN + 1, CHANNELS);
    CHECK_TRUE(trellis_ggml_project_attention(
        ctx, x, global, bad_tokens, HEADS,
        q_w, q_b, kv_w, kv_b, NULL, NULL, out_w, out_b, proj_w, proj_b) == NULL);
    CHECK_TRUE(trellis_ggml_project_attention(
        ctx, x, global, bad_batches, HEADS,
        q_w, q_b, kv_w, kv_b, NULL, NULL, out_w, out_b, proj_w, proj_b) == NULL);
    CHECK_TRUE(trellis_ggml_project_attention(
        ctx, x, global, projected, HEADS,
        q_w, q_b, kv_w, kv_b, NULL, NULL, out_w, out_b, bad_proj_w, proj_b) == NULL);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_backend_new_graph_allocator(backend);
    CHECK_TRUE(alloc != NULL);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(x, x_data, 0, ggml_nbytes(x));
    ggml_backend_tensor_set(global, global_data, 0, ggml_nbytes(global));
    ggml_backend_tensor_set(projected, projected_data, 0, ggml_nbytes(projected));
    ggml_backend_tensor_set(q_w, q_w_data, 0, ggml_nbytes(q_w));
    ggml_backend_tensor_set(q_b, q_b_data, 0, ggml_nbytes(q_b));
    ggml_backend_tensor_set(kv_w, kv_w_data, 0, ggml_nbytes(kv_w));
    ggml_backend_tensor_set(kv_b, kv_b_data, 0, ggml_nbytes(kv_b));
    ggml_backend_tensor_set(out_w, out_w_data, 0, ggml_nbytes(out_w));
    ggml_backend_tensor_set(out_b, out_b_data, 0, ggml_nbytes(out_b));
    ggml_backend_tensor_set(proj_w, proj_w_data, 0, ggml_nbytes(proj_w));
    ggml_backend_tensor_set(proj_b, proj_b_data, 0, ggml_nbytes(proj_b));
    CHECK_TRUE(trellis_backend_compute_graph(backend, graph) == TRELLIS_STATUS_OK);
    ggml_backend_tensor_get(y, got, 0, ggml_nbytes(y));

    int ok = 1;
    for (int i = 0; i < CHANNELS * TOKENS * BATCHES; ++i) {
        if (!check_close(got[i], exp[i], 2e-3f, "project_attention", i)) {
            ok = 0;
            break;
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

static float attention_input_value(int index, float phase, float scale) {
    return scale * (sinf((float) index * phase) + 0.5f * cosf((float) (index + 17) * (phase * 0.37f)));
}

static int check_attention_close(const float * got, const float * exp, int64_t n, float tol, const char * name) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int64_t max_abs_idx = 0;
    int64_t bad_count = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (!isfinite(got[i]) || !isfinite(exp[i])) {
            fprintf(stderr, "%s[%lld] non-finite: got=%g expected=%g\n", name, (long long) i, got[i], exp[i]);
            return 0;
        }
        const float diff = fabsf(got[i] - exp[i]);
        const float scale = fmaxf(1.0f, fabsf(exp[i]));
        const float rel = diff / scale;
        if (diff > max_abs) {
            max_abs = diff;
            max_abs_idx = i;
        }
        if (rel > max_rel) {
            max_rel = rel;
        }
        if (diff > tol * scale) {
            if (bad_count < 8) {
                fprintf(
                    stderr,
                    "%s[%lld] mismatch: got=%g expected=%g diff=%g tol=%g\n",
                    name,
                    (long long) i,
                    got[i],
                    exp[i],
                    diff,
                    tol * scale);
            }
            ++bad_count;
        }
    }
    if (bad_count != 0) {
        fprintf(
            stderr,
            "%s mismatches=%lld/%lld max_abs=%g max_rel=%g max_abs_idx=%lld\n",
            name,
            (long long) bad_count,
            (long long) n,
            max_abs,
            max_rel,
            (long long) max_abs_idx);
        return 0;
    }
    printf("%s max_abs=%g max_rel=%g\n", name, max_abs, max_rel);
    return 1;
}

static int test_sdpa_flash_attention(trellis_backend_context * backend, int tokens, int batches) {
    enum { HEAD_DIM = 128, HEADS = 12 };
    const int64_t nels = (int64_t) HEAD_DIM * tokens * HEADS * batches;
    const size_t bytes = (size_t) nels * sizeof(float);
    float * q_data = (float *) malloc(bytes);
    float * k_data = (float *) malloc(bytes);
    float * v_data = (float *) malloc(bytes);
    float * explicit_out = (float *) malloc(bytes);
    float * flash_f16_out = (float *) malloc(bytes);
    float * flash_bf16_out = (float *) malloc(bytes);
    CHECK_TRUE(q_data != NULL && k_data != NULL && v_data != NULL &&
        explicit_out != NULL && flash_f16_out != NULL && flash_bf16_out != NULL);

    for (int64_t i = 0; i < nels; ++i) {
        q_data[i] = attention_input_value((int) i, 0.0137f, 0.125f);
        k_data[i] = attention_input_value((int) i, 0.0171f, 0.125f);
        v_data[i] = attention_input_value((int) i, 0.0193f, 0.25f);
    }

    struct ggml_context * ctx = make_graph_ctx();
    CHECK_TRUE(ctx != NULL);
    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, tokens, HEADS, batches);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, tokens, HEADS, batches);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, tokens, HEADS, batches);
    CHECK_TRUE(q != NULL && k != NULL && v != NULL);
    ggml_set_name(q, "q");
    ggml_set_name(k, "k");
    ggml_set_name(v, "v");

    const float scale = 1.0f / sqrtf((float) HEAD_DIM);
    trellis_ggml_attention_policy explicit_policy = TRELLIS_GGML_ATTENTION_POLICY_INIT;
    trellis_ggml_attention_policy flash_f16_policy = TRELLIS_GGML_ATTENTION_POLICY_INIT;
    trellis_ggml_attention_policy flash_bf16_policy = TRELLIS_GGML_ATTENTION_POLICY_INIT;
    flash_f16_policy.mode = TRELLIS_GGML_ATTENTION_MODE_FLASH;
    flash_bf16_policy.mode = TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16;
    struct ggml_tensor * explicit_y = trellis_ggml_sdpa_with_policy(
        ctx, q, k, v, scale, &explicit_policy);
    struct ggml_tensor * flash_f16_y = trellis_ggml_sdpa_with_policy(
        ctx, q, k, v, scale, &flash_f16_policy);
    struct ggml_tensor * flash_bf16_y = trellis_ggml_sdpa_with_policy(
        ctx, q, k, v, scale, &flash_bf16_policy);
    CHECK_TRUE(explicit_y != NULL && flash_f16_y != NULL && flash_bf16_y != NULL);
    ggml_set_name(explicit_y, "sdpa_explicit");
    ggml_set_name(flash_f16_y, "sdpa_flash_f16");
    ggml_set_name(flash_bf16_y, "sdpa_flash_bf16");
    CHECK_TRUE(ggml_nelements(explicit_y) == ggml_nelements(flash_f16_y));
    CHECK_TRUE(ggml_nelements(explicit_y) == ggml_nelements(flash_bf16_y));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, explicit_y);
    ggml_build_forward_expand(graph, flash_f16_y);
    ggml_build_forward_expand(graph, flash_bf16_y);
    ggml_gallocr_t alloc = trellis_backend_new_graph_allocator(backend);
    CHECK_TRUE(alloc != NULL);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(q, q_data, 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k_data, 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_data, 0, ggml_nbytes(v));
    CHECK_TRUE(trellis_backend_compute_graph(backend, graph) == TRELLIS_STATUS_OK);
    ggml_backend_tensor_get(explicit_y, explicit_out, 0, ggml_nbytes(explicit_y));
    ggml_backend_tensor_get(flash_f16_y, flash_f16_out, 0, ggml_nbytes(flash_f16_y));
    ggml_backend_tensor_get(flash_bf16_y, flash_bf16_out, 0, ggml_nbytes(flash_bf16_y));

    char f16_name[128];
    char bf16_name[128];
    snprintf(f16_name, sizeof(f16_name), "sdpa_flash_f16_%s_%dtok_b%d", trellis_backend_kind_name(backend->kind), tokens, batches);
    snprintf(bf16_name, sizeof(bf16_name), "sdpa_flash_bf16_%s_%dtok_b%d", trellis_backend_kind_name(backend->kind), tokens, batches);
    int ok = check_attention_close(
        flash_f16_out,
        explicit_out,
        ggml_nelements(explicit_y),
        2e-2f,
        f16_name);
    ok = ok && check_attention_close(
        flash_bf16_out,
        explicit_out,
        ggml_nelements(explicit_y),
        3e-2f,
        bf16_name);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    free(q_data);
    free(k_data);
    free(v_data);
    free(explicit_out);
    free(flash_f16_out);
    free(flash_bf16_out);
    return ok;
}

static int test_flash_constant_attention(
    trellis_backend_context * backend,
    int query_tokens,
    int kv_tokens,
    int batches,
    float value,
    int use_bf16,
    int heads,
    int repeats) {
    enum { HEAD_DIM = 128 };
    const int64_t q_nels = (int64_t) HEAD_DIM * query_tokens * heads * batches;
    const int64_t kv_nels = (int64_t) HEAD_DIM * kv_tokens * heads * batches;
    float * q_data = (float *) calloc((size_t) q_nels, sizeof(float));
    float * k_data = (float *) calloc((size_t) kv_nels, sizeof(float));
    float * v_data = (float *) malloc((size_t) kv_nels * sizeof(float));
    float * output = (float *) malloc((size_t) q_nels * sizeof(float));
    CHECK_TRUE(q_data != NULL && k_data != NULL && v_data != NULL && output != NULL);
    for (int64_t i = 0; i < kv_nels; ++i) {
        v_data[i] = value;
    }

    struct ggml_context * ctx = make_graph_ctx();
    CHECK_TRUE(ctx != NULL);
    struct ggml_tensor * q = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, HEAD_DIM, query_tokens, heads, batches);
    struct ggml_tensor * k = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, HEAD_DIM, kv_tokens, heads, batches);
    struct ggml_tensor * v = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, HEAD_DIM, kv_tokens, heads, batches);
    CHECK_TRUE(q != NULL && k != NULL && v != NULL);

    trellis_ggml_attention_policy policy = TRELLIS_GGML_ATTENTION_POLICY_INIT;
    policy.mode = use_bf16 ?
        TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16 :
        TRELLIS_GGML_ATTENTION_MODE_FLASH_F16;
    struct ggml_tensor * y = trellis_ggml_sdpa_with_policy(
        ctx, q, k, v, 1.0f / sqrtf((float) HEAD_DIM), &policy);
    CHECK_TRUE(y != NULL);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_backend_new_graph_allocator(backend);
    CHECK_TRUE(alloc != NULL);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(q, q_data, 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k_data, 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_data, 0, ggml_nbytes(v));
    double best_ms = DBL_MAX;
    double total_ms = 0.0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const double started_ms = wall_clock_ms();
        CHECK_TRUE(trellis_backend_compute_graph(backend, graph) == TRELLIS_STATUS_OK);
        const double elapsed_ms = wall_clock_ms() - started_ms;
        best_ms = fmin(best_ms, elapsed_ms);
        total_ms += elapsed_ms;
    }
    ggml_backend_tensor_get(y, output, 0, ggml_nbytes(y));

    const float expected = use_bf16 ?
        ggml_bf16_to_fp32(ggml_fp32_to_bf16(value)) :
        ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
    const float tolerance = 2e-3f * fmaxf(1.0f, fabsf(expected));
    int ok = 1;
    for (int64_t i = 0; i < q_nels; ++i) {
        if (!isfinite(output[i]) || fabsf(output[i] - expected) > tolerance) {
            fprintf(
                stderr,
                "%s flash constant q=%d kv=%d heads=%d batch=%d value=%g output[%lld]=%g expected=%g tolerance=%g\n",
                use_bf16 ? "bf16" : "f16",
                query_tokens,
                kv_tokens,
                heads,
                batches,
                value,
                (long long) i,
                output[i],
                expected,
                tolerance);
            ok = 0;
            break;
        }
    }
    if (ok) {
        printf(
            "%s flash constant q=%d kv=%d heads=%d batch=%d value=%g output=%g best_ms=%.3f avg_ms=%.3f repeats=%d\n",
            use_bf16 ? "bf16" : "f16",
            query_tokens,
            kv_tokens,
            heads,
            batches,
            value,
            output[0],
            best_ms,
            total_ms / repeats,
            repeats);
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    free(q_data);
    free(k_data);
    free(v_data);
    free(output);
    return ok;
}

static int test_flash_bf16_mixed_v_panels(trellis_backend_context * backend) {
    enum { HEAD_DIM = 128, QUERY_TOKENS = 257, KV_TOKENS = 5 };
    static const float panel_values[HEAD_DIM / 16] = {
        70000.0f, 3.0f, -5.0f, 0.25f, 1024.0f, -2048.0f, 17.0f, -31.0f,
    };
    const int64_t q_nels = (int64_t) HEAD_DIM * QUERY_TOKENS;
    const int64_t kv_nels = (int64_t) HEAD_DIM * KV_TOKENS;
    float * q_data = (float *) calloc((size_t) q_nels, sizeof(float));
    float * k_data = (float *) calloc((size_t) kv_nels, sizeof(float));
    float * v_data = (float *) malloc((size_t) kv_nels * sizeof(float));
    float * output = (float *) malloc((size_t) q_nels * sizeof(float));
    CHECK_TRUE(q_data != NULL && k_data != NULL && v_data != NULL && output != NULL);

    for (int64_t i = 0; i < kv_nels; ++i) {
        const int channel = (int) (i % HEAD_DIM);
        v_data[i] = panel_values[channel / 16];
    }

    struct ggml_context * ctx = make_graph_ctx();
    CHECK_TRUE(ctx != NULL);
    struct ggml_tensor * q = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, HEAD_DIM, QUERY_TOKENS, 1, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, HEAD_DIM, KV_TOKENS, 1, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, HEAD_DIM, KV_TOKENS, 1, 1);
    CHECK_TRUE(q != NULL && k != NULL && v != NULL);

    trellis_ggml_attention_policy policy = TRELLIS_GGML_ATTENTION_POLICY_INIT;
    policy.mode = TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16;
    struct ggml_tensor * y = trellis_ggml_sdpa_with_policy(
        ctx, q, k, v, 1.0f / sqrtf((float) HEAD_DIM), &policy);
    CHECK_TRUE(y != NULL);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_backend_new_graph_allocator(backend);
    CHECK_TRUE(alloc != NULL);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(q, q_data, 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k_data, 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_data, 0, ggml_nbytes(v));
    CHECK_TRUE(trellis_backend_compute_graph(backend, graph) == TRELLIS_STATUS_OK);
    ggml_backend_tensor_get(y, output, 0, ggml_nbytes(y));

    int ok = 1;
    for (int64_t i = 0; i < q_nels; ++i) {
        const int channel = (int) (i % HEAD_DIM);
        const float expected = ggml_bf16_to_fp32(
            ggml_fp32_to_bf16(panel_values[channel / 16]));
        const float tolerance = 2e-3f * fmaxf(1.0f, fabsf(expected));
        if (!isfinite(output[i]) || fabsf(output[i] - expected) > tolerance) {
            fprintf(
                stderr,
                "bf16 flash mixed V panel output[%lld]=%g expected=%g tolerance=%g\n",
                (long long) i,
                output[i],
                expected,
                tolerance);
            ok = 0;
            break;
        }
    }
    if (ok) {
        printf("bf16 flash mixed V panels");
        for (int panel = 0; panel < HEAD_DIM / 16; ++panel) {
            printf(" p%d=%g", panel, output[panel * 16]);
        }
        printf("\n");
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    free(q_data);
    free(k_data);
    free(v_data);
    free(output);
    return ok;
}

static int test_flash_bf16_head128_causal_gqa(
    trellis_backend_context * backend) {
    enum {
        HEAD_DIM = 128,
        QUERY_HEADS = 16,
        KV_HEADS = 8,
        TOKENS = 17,
    };
    const int64_t q_nels =
        (int64_t) HEAD_DIM * TOKENS * QUERY_HEADS;
    const int64_t kv_nels =
        (int64_t) HEAD_DIM * TOKENS * KV_HEADS;
    const int64_t out_nels = q_nels;
    float * q_data = (float *) malloc((size_t) q_nels * sizeof(float));
    float * k_data = (float *) malloc((size_t) kv_nels * sizeof(float));
    float * v_data = (float *) malloc((size_t) kv_nels * sizeof(float));
    float * mask_data = (float *) malloc(
        (size_t) TOKENS * TOKENS * sizeof(float));
    float * output = (float *) malloc((size_t) out_nels * sizeof(float));
    float * expected = (float *) malloc((size_t) out_nels * sizeof(float));
    CHECK_TRUE(q_data != NULL && k_data != NULL && v_data != NULL &&
        mask_data != NULL && output != NULL && expected != NULL);
    for (int64_t i = 0; i < q_nels; ++i) {
        q_data[i] = attention_input_value((int) i, 0.0073f, 0.0625f);
    }
    for (int64_t i = 0; i < kv_nels; ++i) {
        k_data[i] = attention_input_value((int) i, 0.0091f, 0.0625f);
        v_data[i] = attention_input_value((int) i, 0.0117f, 0.25f);
    }
    for (int query = 0; query < TOKENS; ++query) {
        for (int key = 0; key < TOKENS; ++key) {
            mask_data[(size_t) query * TOKENS + key] =
                key <= query ? 0.0f : -INFINITY;
        }
    }

    struct ggml_context * ctx = make_graph_ctx();
    CHECK_TRUE(ctx != NULL);
    struct ggml_tensor * q = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, HEAD_DIM, TOKENS, QUERY_HEADS, 1);
    struct ggml_tensor * k_input = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, HEAD_DIM, TOKENS, KV_HEADS, 1);
    struct ggml_tensor * v_input = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, HEAD_DIM, TOKENS, KV_HEADS, 1);
    struct ggml_tensor * mask_input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, TOKENS, TOKENS);
    struct ggml_tensor * k = ggml_cast(ctx, k_input, GGML_TYPE_BF16);
    struct ggml_tensor * v = ggml_cast(ctx, v_input, GGML_TYPE_BF16);
    struct ggml_tensor * mask = ggml_cast(ctx, mask_input, GGML_TYPE_F16);
    struct ggml_tensor * y = ggml_flash_attn_ext(
        ctx,
        q,
        k,
        v,
        mask,
        1.0f / sqrtf((float) HEAD_DIM),
        0.0f,
        0.0f);
    CHECK_TRUE(y != NULL);
    ggml_flash_attn_ext_set_prec(y, GGML_PREC_F32);
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_backend_new_graph_allocator(backend);
    CHECK_TRUE(alloc != NULL && ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(q, q_data, 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k_input, k_data, 0, ggml_nbytes(k_input));
    ggml_backend_tensor_set(v_input, v_data, 0, ggml_nbytes(v_input));
    ggml_backend_tensor_set(mask_input, mask_data, 0, ggml_nbytes(mask_input));
    CHECK_TRUE(trellis_backend_compute_graph(backend, graph) == TRELLIS_STATUS_OK);
    ggml_backend_tensor_get(y, output, 0, ggml_nbytes(y));

    const float scale = 1.0f / sqrtf((float) HEAD_DIM);
    for (int query = 0; query < TOKENS; ++query) {
        for (int head = 0; head < QUERY_HEADS; ++head) {
            const int kv_head = head / (QUERY_HEADS / KV_HEADS);
            float scores[TOKENS];
            float maximum = -INFINITY;
            for (int key = 0; key <= query; ++key) {
                float dot = 0.0f;
                for (int channel = 0; channel < HEAD_DIM; ++channel) {
                    const size_t qi = (size_t) channel + HEAD_DIM *
                        ((size_t) query + TOKENS * (size_t) head);
                    const size_t ki = (size_t) channel + HEAD_DIM *
                        ((size_t) key + TOKENS * (size_t) kv_head);
                    const float rounded_k = ggml_bf16_to_fp32(
                        ggml_fp32_to_bf16(k_data[ki]));
                    dot += q_data[qi] * rounded_k;
                }
                scores[key] = dot * scale;
                if (scores[key] > maximum) maximum = scores[key];
            }
            float denominator = 0.0f;
            for (int key = 0; key <= query; ++key) {
                scores[key] = expf(scores[key] - maximum);
                denominator += scores[key];
            }
            for (int channel = 0; channel < HEAD_DIM; ++channel) {
                float value_out = 0.0f;
                for (int key = 0; key <= query; ++key) {
                    const size_t vi = (size_t) channel + HEAD_DIM *
                        ((size_t) key + TOKENS * (size_t) kv_head);
                    const float rounded_v = ggml_bf16_to_fp32(
                        ggml_fp32_to_bf16(v_data[vi]));
                    value_out += scores[key] / denominator * rounded_v;
                }
                const size_t oi = (size_t) channel + HEAD_DIM *
                    ((size_t) head + QUERY_HEADS * (size_t) query);
                expected[oi] = value_out;
            }
        }
    }
    char name[128];
    snprintf(
        name,
        sizeof(name),
        "flash_bf16_head128_causal_gqa_%s",
        trellis_backend_kind_name(backend->kind));
    const int ok = check_attention_close(
        output, expected, out_nels, 2.5e-2f, name);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    free(expected);
    free(output);
    free(mask_data);
    free(v_data);
    free(k_data);
    free(q_data);
    return ok;
}

int main(int argc, char ** argv) {
    const char * backend_name = argc > 1 ? argv[1] : TRELLIS_DEFAULT_GGML_BACKEND;
    trellis_backend_kind kind;
    trellis_status status = trellis_backend_kind_from_name(backend_name, &kind);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "invalid backend: %s\n", backend_name);
        return 1;
    }

    trellis_backend_context backend;
    status = trellis_backend_init(&backend, kind, 0);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "%s backend unavailable: %s\n", backend_name, trellis_status_string(status));
        return 77;
    }

    int ok = test_conv3d(&backend);
    ok = ok && test_pixel_shuffle_3d(&backend);
    ok = ok && test_project_attention(&backend);
    ok = ok && test_sdpa_flash_attention(&backend, 512, 1);
    ok = ok && test_sdpa_flash_attention(&backend, 1024, 1);
    ok = ok && test_sdpa_flash_attention(&backend, 512, 2);
    if (kind == TRELLIS_BACKEND_CUDA || kind == TRELLIS_BACKEND_VULKAN) {
        /* Q=K=0 makes every attention row uniform, so the exact result for a
         * constant V is that constant.  These sizes/values exercise the
         * historical F16 VKQ overflow and both aligned and short KV tails. */
        ok = ok && test_flash_constant_attention(&backend, 4730, 4730, 1, 384.0f, 1, 1, 1);
        ok = ok && test_flash_constant_attention(&backend, 8192, 8192, 1, 128.0f, 1, 1, 1);
        ok = ok && test_flash_constant_attention(&backend, 16384, 16384, 1, 64.0f, 1, 1, 1);
        ok = ok && test_flash_constant_attention(&backend, 257, 257, 2, 70000.0f, 1, 1, 1);
        ok = ok && test_flash_constant_attention(&backend, 513, 5, 1, 70000.0f, 1, 1, 1);
        ok = ok && test_flash_bf16_mixed_v_panels(&backend);
        ok = ok && test_flash_bf16_head128_causal_gqa(&backend);
        if (getenv("TRELLIS_LONG_ATTN_PERF") != NULL) {
            ok = ok && test_flash_constant_attention(&backend, 21775, 21775, 1, 1.0f, 1, 12, 5);
        }
    }
    trellis_backend_free(&backend);
    if (!ok) {
        return 1;
    }
    printf("ggml backend ops tests passed on %s\n", backend_name);
    return 0;
}
