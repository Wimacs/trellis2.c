#include "trellis.h"

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
    };
    const int OD = conv_out_size_ref(ID, KD, SD, PD, DD);
    const int OH = conv_out_size_ref(IH, KH, SH, PH, DH);
    const int OW = conv_out_size_ref(IW, KW, SW, PW, DW);
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
    trellis_backend_free(&backend);
    if (!ok) {
        return 1;
    }
    printf("ggml backend ops tests passed on %s\n", backend_name);
    return 0;
}
