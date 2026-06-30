#include "trellis.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_failures = 0;
static trellis_cuda_context g_cuda;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++g_failures; \
        return; \
    } \
} while (0)

static void check_close_at(const char * file, int line, float a, float b, float tol) {
    float diff = fabsf(a - b);
    float scale = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
    if (diff > tol * scale) {
        fprintf(stderr, "CHECK_CLOSE failed at %s:%d: actual=%g expected=%g diff=%g tol=%g\n",
                file, line, a, b, diff, tol);
        ++g_failures;
    }
}

#define CHECK_CLOSE(a, b, tol) check_close_at(__FILE__, __LINE__, (a), (b), (tol))

static struct ggml_context * make_graph_ctx(void) {
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead() + 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    return ggml_init(params);
}

static void graph_alloc_compute_get(
    struct ggml_context * ctx,
    struct ggml_tensor * out,
    float * out_data,
    size_t out_count) {
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t alloc = trellis_cuda_new_graph_allocator(&g_cuda);
    CHECK_TRUE(alloc != NULL);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    trellis_status status = trellis_cuda_compute_graph(&g_cuda, graph);
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    CHECK_TRUE(out_count == (size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data, 0, ggml_nbytes(out));
    ggml_gallocr_free(alloc);
}

static float gelu_tanh_ref(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + 0.044715f * x * x)));
}

static float silu_ref(float x) {
    return x / (1.0f + expf(-x));
}

static void test_safetensors(void) {
    char path[] = "/tmp/trellis2_c_safetensors_XXXXXX";
    int fd = mkstemp(path);
    CHECK_TRUE(fd >= 0);
    FILE * f = fdopen(fd, "wb");
    CHECK_TRUE(f != NULL);

    const char * header =
        "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
        "\"b\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[8,12]},"
        "\"c\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[12,14]},"
        "\"__metadata__\":{\"format\":\"pt\"}}";
    uint64_t hlen = (uint64_t) strlen(header);
    unsigned char hbuf[8];
    for (int i = 0; i < 8; ++i) {
        hbuf[i] = (unsigned char) ((hlen >> (8 * i)) & 0xff);
    }
    fwrite(hbuf, 1, 8, f);
    fwrite(header, 1, strlen(header), f);
    float a[2] = {1.25f, -2.0f};
    ggml_fp16_t b[2] = {ggml_fp32_to_fp16(3.0f), ggml_fp32_to_fp16(-4.0f)};
    ggml_bf16_t c[1] = {ggml_fp32_to_bf16(5.5f)};
    fwrite(a, 1, sizeof(a), f);
    fwrite(b, 1, sizeof(b), f);
    fwrite(c, 1, sizeof(c), f);
    fclose(f);

    trellis_safetensors st;
    trellis_status status = trellis_safetensors_open(path, &st);
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    CHECK_TRUE(st.n_tensors == 3);
    const trellis_safetensor_meta * ma = trellis_safetensors_find(&st, "a");
    const trellis_safetensor_meta * mb = trellis_safetensors_find(&st, "b");
    const trellis_safetensor_meta * mc = trellis_safetensors_find(&st, "c");
    CHECK_TRUE(ma != NULL && mb != NULL && mc != NULL);
    CHECK_TRUE(ma->dtype == TRELLIS_DTYPE_F32);
    CHECK_TRUE(mb->dtype == TRELLIS_DTYPE_F16);
    CHECK_TRUE(mc->dtype == TRELLIS_DTYPE_BF16);
    float out[2] = {0};
    CHECK_TRUE(trellis_safetensors_read_f32(&st, ma, out, 2) == TRELLIS_STATUS_OK);
    CHECK_CLOSE(out[0], 1.25f, 1e-6f);
    CHECK_CLOSE(out[1], -2.0f, 1e-6f);
    CHECK_TRUE(trellis_safetensors_read_f32(&st, mb, out, 2) == TRELLIS_STATUS_OK);
    CHECK_CLOSE(out[0], 3.0f, 1e-3f);
    CHECK_CLOSE(out[1], -4.0f, 1e-3f);
    CHECK_TRUE(trellis_safetensors_read_f32(&st, mc, out, 1) == TRELLIS_STATUS_OK);
    CHECK_CLOSE(out[0], 5.5f, 1e-3f);
    trellis_safetensors_close(&st);
    unlink(path);
}

static void test_tensor_store_loader_cuda(void) {
    char path[] = "/tmp/trellis2_c_weights_XXXXXX";
    int fd = mkstemp(path);
    CHECK_TRUE(fd >= 0);
    FILE * f = fdopen(fd, "wb");
    CHECK_TRUE(f != NULL);

    const char * header =
        "{\"linear.weight\":{\"dtype\":\"F32\",\"shape\":[3,2],\"data_offsets\":[0,24]},"
        "\"linear.bias\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[24,36]},"
        "\"norm.gamma\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[36,60]}}";
    uint64_t hlen = (uint64_t) strlen(header);
    unsigned char hbuf[8];
    for (int i = 0; i < 8; ++i) {
        hbuf[i] = (unsigned char) ((hlen >> (8 * i)) & 0xff);
    }
    fwrite(hbuf, 1, 8, f);
    fwrite(header, 1, strlen(header), f);
    float linear_w[6] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    float linear_b[3] = {0.25f, -0.5f, 0.75f};
    float gamma[6] = {
        10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f,
    };
    fwrite(linear_w, 1, sizeof(linear_w), f);
    fwrite(linear_b, 1, sizeof(linear_b), f);
    fwrite(gamma, 1, sizeof(gamma), f);
    fclose(f);

    trellis_tensor_store store;
    CHECK_TRUE(trellis_tensor_store_init(&store, 16, 0) == TRELLIS_STATUS_OK);
    size_t loaded = 0;
    CHECK_TRUE(trellis_tensor_store_load_safetensors_f32(&store, &g_cuda, path, true, &loaded) == TRELLIS_STATUS_OK);
    CHECK_TRUE(loaded == 3);

    struct ggml_tensor * w = trellis_tensor_store_get(&store, "linear.weight");
    struct ggml_tensor * b = trellis_tensor_store_get(&store, "linear.bias");
    struct ggml_tensor * g = trellis_tensor_store_get(&store, "norm.gamma");
    CHECK_TRUE(w != NULL && b != NULL && g != NULL);
    CHECK_TRUE(w->ne[0] == 2 && w->ne[1] == 3);
    CHECK_TRUE(b->ne[0] == 3);
    CHECK_TRUE(g->ne[0] == 3 && g->ne[1] == 2);

    float out_w[6] = {0};
    float out_b[3] = {0};
    float out_g[6] = {0};
    ggml_backend_tensor_get(w, out_w, 0, ggml_nbytes(w));
    ggml_backend_tensor_get(b, out_b, 0, ggml_nbytes(b));
    ggml_backend_tensor_get(g, out_g, 0, ggml_nbytes(g));
    for (int i = 0; i < 6; ++i) {
        CHECK_CLOSE(out_w[i], linear_w[i], 1e-6f);
        CHECK_CLOSE(out_g[i], gamma[i], 1e-6f);
    }
    for (int i = 0; i < 3; ++i) {
        CHECK_CLOSE(out_b[i], linear_b[i], 1e-6f);
    }

    trellis_tensor_store_free(&store);
    unlink(path);
}

static void test_linear_cuda(void) {
    struct ggml_context * ctx = make_graph_ctx();
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * y = trellis_ggml_linear(ctx, x, w, b);
    float x_data[] = {
        1.0f, 2.0f,
        -1.0f, 0.5f,
        3.0f, -2.0f,
    };
    float w_data[] = {
        0.5f, -1.0f,
        1.0f, 0.0f,
        -0.5f, 2.0f,
        0.25f, 0.75f,
    };
    float b_data[] = {0.1f, -0.2f, 0.3f, -0.4f};

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_cuda_new_graph_allocator(&g_cuda);
    CHECK_TRUE(alloc != NULL);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(x, x_data, 0, ggml_nbytes(x));
    ggml_backend_tensor_set(w, w_data, 0, ggml_nbytes(w));
    ggml_backend_tensor_set(b, b_data, 0, ggml_nbytes(b));
    CHECK_TRUE(trellis_cuda_compute_graph(&g_cuda, graph) == TRELLIS_STATUS_OK);

    float out[12];
    ggml_backend_tensor_get(y, out, 0, ggml_nbytes(y));
    for (int token = 0; token < 3; ++token) {
        for (int o = 0; o < 4; ++o) {
            float expected = b_data[o];
            for (int i = 0; i < 2; ++i) {
                expected += w_data[o * 2 + i] * x_data[token * 2 + i];
            }
            CHECK_CLOSE(out[token * 4 + o], expected, 1e-5f);
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

static void test_norms_cuda(void) {
    struct ggml_context * ctx = make_graph_ctx();
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    struct ggml_tensor * gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * y = trellis_ggml_layer_norm(ctx, x, gamma, beta, 1e-6f);

    float x_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -2.0f, 0.0f, 1.0f, 5.0f,
        3.0f, 3.0f, 3.0f, 3.0f,
    };
    float gamma_data[] = {1.0f, 0.5f, -1.0f, 2.0f};
    float beta_data[] = {0.0f, 1.0f, -0.5f, 0.25f};

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_cuda_new_graph_allocator(&g_cuda);
    CHECK_TRUE(alloc != NULL);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(x, x_data, 0, ggml_nbytes(x));
    ggml_backend_tensor_set(gamma, gamma_data, 0, ggml_nbytes(gamma));
    ggml_backend_tensor_set(beta, beta_data, 0, ggml_nbytes(beta));
    CHECK_TRUE(trellis_cuda_compute_graph(&g_cuda, graph) == TRELLIS_STATUS_OK);

    float out[12];
    ggml_backend_tensor_get(y, out, 0, ggml_nbytes(y));
    for (int token = 0; token < 3; ++token) {
        float mean = 0.0f;
        for (int c = 0; c < 4; ++c) mean += x_data[token * 4 + c];
        mean /= 4.0f;
        float var = 0.0f;
        for (int c = 0; c < 4; ++c) {
            float d = x_data[token * 4 + c] - mean;
            var += d * d;
        }
        var /= 4.0f;
        for (int c = 0; c < 4; ++c) {
            float expected = (x_data[token * 4 + c] - mean) / sqrtf(var + 1e-6f);
            expected = expected * gamma_data[c] + beta_data[c];
            CHECK_CLOSE(out[token * 4 + c], expected, 2e-4f);
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    ctx = make_graph_ctx();
    x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    y = trellis_ggml_rms_norm(ctx, x, gamma, 1e-6f);
    graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    alloc = trellis_cuda_new_graph_allocator(&g_cuda);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(x, x_data, 0, ggml_nbytes(x));
    ggml_backend_tensor_set(gamma, gamma_data, 0, ggml_nbytes(gamma));
    CHECK_TRUE(trellis_cuda_compute_graph(&g_cuda, graph) == TRELLIS_STATUS_OK);
    ggml_backend_tensor_get(y, out, 0, ggml_nbytes(y));
    for (int token = 0; token < 3; ++token) {
        float ms = 0.0f;
        for (int c = 0; c < 4; ++c) ms += x_data[token * 4 + c] * x_data[token * 4 + c];
        ms /= 4.0f;
        for (int c = 0; c < 4; ++c) {
            float expected = x_data[token * 4 + c] / sqrtf(ms + 1e-6f) * gamma_data[c];
            CHECK_CLOSE(out[token * 4 + c], expected, 2e-4f);
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

static void test_feed_forward_cuda(void) {
    struct ggml_context * ctx = make_graph_ctx();
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    struct ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
    struct ggml_tensor * b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    struct ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    struct ggml_tensor * b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    struct ggml_tensor * y = trellis_ggml_feed_forward(ctx, x, w1, b1, w2, b2);

    float x_data[] = {0.2f, -0.4f, 1.0f, 0.5f};
    float w1_data[] = {
        0.5f, -0.3f,
        -0.2f, 0.7f,
        0.1f, 0.4f,
    };
    float b1_data[] = {0.01f, -0.02f, 0.03f};
    float w2_data[] = {
        0.6f, -0.1f, 0.2f,
        -0.4f, 0.3f, 0.5f,
    };
    float b2_data[] = {0.05f, -0.07f};

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_cuda_new_graph_allocator(&g_cuda);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(x, x_data, 0, ggml_nbytes(x));
    ggml_backend_tensor_set(w1, w1_data, 0, ggml_nbytes(w1));
    ggml_backend_tensor_set(b1, b1_data, 0, ggml_nbytes(b1));
    ggml_backend_tensor_set(w2, w2_data, 0, ggml_nbytes(w2));
    ggml_backend_tensor_set(b2, b2_data, 0, ggml_nbytes(b2));
    CHECK_TRUE(trellis_cuda_compute_graph(&g_cuda, graph) == TRELLIS_STATUS_OK);

    float out[4];
    ggml_backend_tensor_get(y, out, 0, ggml_nbytes(y));
    for (int token = 0; token < 2; ++token) {
        float hidden[3];
        for (int h = 0; h < 3; ++h) {
            hidden[h] = b1_data[h];
            for (int i = 0; i < 2; ++i) {
                hidden[h] += w1_data[h * 2 + i] * x_data[token * 2 + i];
            }
            hidden[h] = gelu_tanh_ref(hidden[h]);
        }
        for (int o = 0; o < 2; ++o) {
            float expected = b2_data[o];
            for (int h = 0; h < 3; ++h) {
                expected += w2_data[o * 3 + h] * hidden[h];
            }
            CHECK_CLOSE(out[token * 2 + o], expected, 1e-4f);
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

static void test_flexible_dual_grid_mesh_host(void) {
    const int32_t coords[] = {
        0, 0, 0, 0,
        0, 0, 1, 0,
        0, 1, 1, 0,
        0, 1, 0, 0,
    };
    float feats[4 * 7];
    memset(feats, 0, sizeof(feats));
    feats[5] = 1.0f;

    trellis_mesh_host mesh;
    CHECK_TRUE(trellis_flexible_dual_grid_mesh_from_decoder_logits_host(
        coords, feats, 4, 7, 4, &mesh) == TRELLIS_STATUS_OK);
    CHECK_TRUE(mesh.n_vertices == 4);
    CHECK_TRUE(mesh.n_faces == 2);
    CHECK_CLOSE(mesh.vertices[0], -0.375f, 1e-6f);
    CHECK_CLOSE(mesh.vertices[1], -0.375f, 1e-6f);
    CHECK_CLOSE(mesh.vertices[2], -0.375f, 1e-6f);
    CHECK_TRUE(mesh.faces[0] == 0 && mesh.faces[1] == 1 && mesh.faces[2] == 3);
    CHECK_TRUE(mesh.faces[3] == 3 && mesh.faces[4] == 1 && mesh.faces[5] == 2);
    trellis_mesh_free(&mesh);
}

static void test_timestep_mlp_cuda(void) {
    struct ggml_context * ctx = make_graph_ctx();
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    struct ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    struct ggml_tensor * b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    struct ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    struct ggml_tensor * b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    struct ggml_tensor * y = trellis_ggml_timestep_mlp(ctx, t, 4, w1, b1, w2, b2);

    float t_data[] = {12.0f, 24.0f};
    float w1_data[] = {
        0.1f, 0.2f, -0.3f, 0.4f,
        -0.2f, 0.5f, 0.7f, -0.1f,
        0.3f, -0.4f, 0.2f, 0.6f,
    };
    float b1_data[] = {0.01f, 0.02f, -0.03f};
    float w2_data[] = {
        0.6f, -0.2f, 0.1f,
        -0.4f, 0.3f, 0.2f,
    };
    float b2_data[] = {0.05f, -0.01f};

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_cuda_new_graph_allocator(&g_cuda);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(t, t_data, 0, ggml_nbytes(t));
    ggml_backend_tensor_set(w1, w1_data, 0, ggml_nbytes(w1));
    ggml_backend_tensor_set(b1, b1_data, 0, ggml_nbytes(b1));
    ggml_backend_tensor_set(w2, w2_data, 0, ggml_nbytes(w2));
    ggml_backend_tensor_set(b2, b2_data, 0, ggml_nbytes(b2));
    CHECK_TRUE(trellis_cuda_compute_graph(&g_cuda, graph) == TRELLIS_STATUS_OK);

    float out[4];
    ggml_backend_tensor_get(y, out, 0, ggml_nbytes(y));
    for (int token = 0; token < 2; ++token) {
        float emb[4];
        const float freqs[2] = {1.0f, 0.01f};
        emb[0] = cosf(t_data[token] * freqs[0]);
        emb[1] = cosf(t_data[token] * freqs[1]);
        emb[2] = sinf(t_data[token] * freqs[0]);
        emb[3] = sinf(t_data[token] * freqs[1]);
        float hidden[3];
        for (int h = 0; h < 3; ++h) {
            hidden[h] = b1_data[h];
            for (int i = 0; i < 4; ++i) {
                hidden[h] += w1_data[h * 4 + i] * emb[i];
            }
            hidden[h] = silu_ref(hidden[h]);
        }
        for (int o = 0; o < 2; ++o) {
            float expected = b2_data[o];
            for (int h = 0; h < 3; ++h) {
                expected += w2_data[o * 3 + h] * hidden[h];
            }
            CHECK_CLOSE(out[token * 2 + o], expected, 2e-4f);
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

static void test_attention_cuda(void) {
    enum { D = 64, L = 3, H = 2, B = 2 };
    struct ggml_context * ctx = make_graph_ctx();
    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, L, H, B);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, L, H, B);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, L, H, B);
    struct ggml_tensor * y = trellis_ggml_sdpa(ctx, q, k, v, 1.0f / sqrtf((float) D));

    float q_data[D * L * H * B];
    float k_data[D * L * H * B];
    float v_data[D * L * H * B];
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < L; ++t) {
                for (int d = 0; d < D; ++d) {
                    const int idx = (((b * H + h) * L + t) * D + d);
                    q_data[idx] = sinf(0.01f * (float) (d + 3 * t + 5 * h + 7 * b));
                    k_data[idx] = cosf(0.02f * (float) (d - 2 * t + 3 * h - 4 * b));
                    v_data[idx] = 0.001f * (float) (d + 1) + 0.05f * (float) t + 0.02f * (float) h - 0.03f * (float) b;
                }
            }
        }
    }
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_cuda_new_graph_allocator(&g_cuda);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(q, q_data, 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k_data, 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_data, 0, ggml_nbytes(v));
    CHECK_TRUE(trellis_cuda_compute_graph(&g_cuda, graph) == TRELLIS_STATUS_OK);

    float out[D * L * H * B];
    ggml_backend_tensor_get(y, out, 0, ggml_nbytes(y));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int tq = 0; tq < L; ++tq) {
                float logits[L];
                float max_logit = -1e30f;
                for (int tk = 0; tk < L; ++tk) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        const int qi = (((b * H + h) * L + tq) * D + d);
                        const int ki = (((b * H + h) * L + tk) * D + d);
                        dot += q_data[qi] * k_data[ki];
                    }
                    logits[tk] = dot / sqrtf((float) D);
                    if (logits[tk] > max_logit) max_logit = logits[tk];
                }
                float denom = 0.0f;
                for (int tk = 0; tk < L; ++tk) {
                    logits[tk] = expf(logits[tk] - max_logit);
                    denom += logits[tk];
                }
                for (int d = 0; d < D; ++d) {
                    float expected = 0.0f;
                    for (int tk = 0; tk < L; ++tk) {
                        const int vi = (((b * H + h) * L + tk) * D + d);
                        expected += (logits[tk] / denom) * v_data[vi];
                    }
                    const int oi = (((b * H + h) * L + tq) * D + d);
                    CHECK_CLOSE(out[oi], expected, 2e-3f);
                }
            }
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

static void test_self_attention_rope_cuda(void) {
    enum { C = 8, L = 3, H = 2, D = C / H, HALF = D / 2, QKV = 3 * C };
    struct ggml_context * ctx = make_graph_ctx();
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, L);
    struct ggml_tensor * qkv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, QKV);
    struct ggml_tensor * qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, QKV);
    struct ggml_tensor * q_gamma = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, H);
    struct ggml_tensor * k_gamma = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, H);
    struct ggml_tensor * out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, C);
    struct ggml_tensor * out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    struct ggml_tensor * cos_phase = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HALF, L, 1);
    struct ggml_tensor * sin_phase = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HALF, L, 1);
    struct ggml_tensor * y = trellis_ggml_self_attention_rope(
        ctx, x, H, qkv_w, qkv_b, q_gamma, k_gamma, out_w, out_b, cos_phase, sin_phase);
    CHECK_TRUE(y != NULL);

    float x_data[C * L];
    float qkv_w_data[C * QKV];
    float qkv_b_data[QKV];
    float q_gamma_data[D * H];
    float k_gamma_data[D * H];
    float out_w_data[C * C];
    float out_b_data[C];
    float cos_data[L * HALF];
    float sin_data[L * HALF];
    for (int i = 0; i < C * L; ++i) x_data[i] = 0.07f * sinf((float) i);
    for (int o = 0; o < QKV; ++o) {
        qkv_b_data[o] = 0.01f * (float) ((o % 5) - 2);
        for (int i = 0; i < C; ++i) qkv_w_data[o * C + i] = 0.03f * cosf((float) (o * 3 + i));
    }
    for (int h = 0; h < H; ++h) {
        for (int d = 0; d < D; ++d) {
            q_gamma_data[h * D + d] = 0.9f + 0.02f * (float) (h * D + d);
            k_gamma_data[h * D + d] = 1.1f - 0.015f * (float) (h * D + d);
        }
    }
    for (int o = 0; o < C; ++o) {
        out_b_data[o] = -0.02f + 0.01f * (float) o;
        for (int i = 0; i < C; ++i) out_w_data[o * C + i] = 0.04f * sinf((float) (o + 2 * i));
    }
    for (int t = 0; t < L; ++t) {
        for (int p = 0; p < HALF; ++p) {
            float phase = 0.11f * (float) t + 0.07f * (float) p;
            cos_data[t * HALF + p] = cosf(phase);
            sin_data[t * HALF + p] = sinf(phase);
        }
    }

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_cuda_new_graph_allocator(&g_cuda);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(x, x_data, 0, ggml_nbytes(x));
    ggml_backend_tensor_set(qkv_w, qkv_w_data, 0, ggml_nbytes(qkv_w));
    ggml_backend_tensor_set(qkv_b, qkv_b_data, 0, ggml_nbytes(qkv_b));
    ggml_backend_tensor_set(q_gamma, q_gamma_data, 0, ggml_nbytes(q_gamma));
    ggml_backend_tensor_set(k_gamma, k_gamma_data, 0, ggml_nbytes(k_gamma));
    ggml_backend_tensor_set(out_w, out_w_data, 0, ggml_nbytes(out_w));
    ggml_backend_tensor_set(out_b, out_b_data, 0, ggml_nbytes(out_b));
    ggml_backend_tensor_set(cos_phase, cos_data, 0, ggml_nbytes(cos_phase));
    ggml_backend_tensor_set(sin_phase, sin_data, 0, ggml_nbytes(sin_phase));
    CHECK_TRUE(trellis_cuda_compute_graph(&g_cuda, graph) == TRELLIS_STATUS_OK);

    float got[C * L];
    ggml_backend_tensor_get(y, got, 0, ggml_nbytes(y));

    float q[C * L], k[C * L], v[C * L], attn_out[C * L], exp_out[C * L];
    for (int t = 0; t < L; ++t) {
        float qkv[QKV];
        for (int o = 0; o < QKV; ++o) {
            qkv[o] = qkv_b_data[o];
            for (int i = 0; i < C; ++i) qkv[o] += qkv_w_data[o * C + i] * x_data[t * C + i];
        }
        memcpy(q + t * C, qkv, C * sizeof(float));
        memcpy(k + t * C, qkv + C, C * sizeof(float));
        memcpy(v + t * C, qkv + 2 * C, C * sizeof(float));
    }
    for (int t = 0; t < L; ++t) {
        for (int h = 0; h < H; ++h) {
            float q_ms = 0.0f, k_ms = 0.0f;
            for (int d = 0; d < D; ++d) {
                q_ms += q[t * C + h * D + d] * q[t * C + h * D + d];
                k_ms += k[t * C + h * D + d] * k[t * C + h * D + d];
            }
            q_ms /= (float) D;
            k_ms /= (float) D;
            for (int d = 0; d < D; ++d) {
                q[t * C + h * D + d] = q[t * C + h * D + d] / sqrtf(q_ms) * q_gamma_data[h * D + d];
                k[t * C + h * D + d] = k[t * C + h * D + d] / sqrtf(k_ms) * k_gamma_data[h * D + d];
            }
            for (int p = 0; p < HALF; ++p) {
                int d0 = 2 * p;
                int d1 = d0 + 1;
                float c = cos_data[t * HALF + p], s = sin_data[t * HALF + p];
                float q0 = q[t * C + h * D + d0], q1 = q[t * C + h * D + d1];
                float k0 = k[t * C + h * D + d0], k1 = k[t * C + h * D + d1];
                q[t * C + h * D + d0] = q0 * c - q1 * s;
                q[t * C + h * D + d1] = q0 * s + q1 * c;
                k[t * C + h * D + d0] = k0 * c - k1 * s;
                k[t * C + h * D + d1] = k0 * s + k1 * c;
            }
        }
    }
    for (int tq = 0; tq < L; ++tq) {
        for (int h = 0; h < H; ++h) {
            float logits[L], max_logit = -1e30f;
            for (int tk = 0; tk < L; ++tk) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += q[tq * C + h * D + d] * k[tk * C + h * D + d];
                logits[tk] = dot / sqrtf((float) D);
                if (logits[tk] > max_logit) max_logit = logits[tk];
            }
            float denom = 0.0f;
            for (int tk = 0; tk < L; ++tk) {
                logits[tk] = expf(logits[tk] - max_logit);
                denom += logits[tk];
            }
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int tk = 0; tk < L; ++tk) acc += logits[tk] / denom * v[tk * C + h * D + d];
                attn_out[tq * C + h * D + d] = acc;
            }
        }
    }
    for (int t = 0; t < L; ++t) {
        for (int o = 0; o < C; ++o) {
            exp_out[t * C + o] = out_b_data[o];
            for (int i = 0; i < C; ++i) exp_out[t * C + o] += out_w_data[o * C + i] * attn_out[t * C + i];
            CHECK_CLOSE(got[t * C + o], exp_out[t * C + o], 1e-4f);
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

static int conv_out_size_ref(int in, int kernel, int stride, int pad, int dilation) {
    return (in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
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

static void channel_layer_norm_3d_ref(
    const float * x,
    const float * gamma,
    const float * beta,
    float * y,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    float eps) {
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < depth; ++d) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    float mean = 0.0f;
                    for (int c = 0; c < channels; ++c) {
                        const int64_t idx =
                            (((int64_t) b * channels + c) * depth + d) * height * (int64_t) width +
                            (int64_t) h * width + w;
                        mean += x[idx];
                    }
                    mean /= (float) channels;
                    float var = 0.0f;
                    for (int c = 0; c < channels; ++c) {
                        const int64_t idx =
                            (((int64_t) b * channels + c) * depth + d) * height * (int64_t) width +
                            (int64_t) h * width + w;
                        const float diff = x[idx] - mean;
                        var += diff * diff;
                    }
                    const float inv_std = 1.0f / sqrtf(var / (float) channels + eps);
                    for (int c = 0; c < channels; ++c) {
                        const int64_t idx =
                            (((int64_t) b * channels + c) * depth + d) * height * (int64_t) width +
                            (int64_t) h * width + w;
                        y[idx] = (x[idx] - mean) * inv_std * gamma[c] + beta[c];
                    }
                }
            }
        }
    }
}

static void dino_patch_embed_ref(
    const float * image,
    const float * weight,
    const float * bias,
    float * tokens,
    int batch,
    int image_h,
    int image_w,
    int out_channels,
    int patch_size) {
    const int patches_h = image_h / patch_size;
    const int patches_w = image_w / patch_size;
    const int n_patches = patches_h * patches_w;
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < n_patches; ++t) {
            const int py = t / patches_w;
            const int px = t - py * patches_w;
            for (int oc = 0; oc < out_channels; ++oc) {
                float acc = bias == NULL ? 0.0f : bias[oc];
                for (int ic = 0; ic < 3; ++ic) {
                    for (int ky = 0; ky < patch_size; ++ky) {
                        const int iy = py * patch_size + ky;
                        for (int kx = 0; kx < patch_size; ++kx) {
                            const int ix = px * patch_size + kx;
                            const int64_t image_idx =
                                (((int64_t) b * 3 + ic) * image_h + iy) * (int64_t) image_w + ix;
                            const int64_t weight_idx =
                                (((int64_t) oc * 3 + ic) * patch_size + ky) * (int64_t) patch_size + kx;
                            acc += image[image_idx] * weight[weight_idx];
                        }
                    }
                }
                tokens[(int64_t) b * n_patches * out_channels + (int64_t) t * out_channels + oc] = acc;
            }
        }
    }
}

static int sparse_find_coord4_ref(const int32_t * coords, int64_t n, int b, int x, int y, int z) {
    for (int64_t i = 0; i < n; ++i) {
        const int32_t * c = coords + 4 * i;
        if (c[0] == b && c[1] == x && c[2] == y && c[3] == z) {
            return (int) i;
        }
    }
    return -1;
}

static void sparse_subm_conv3d_ref(
    const int32_t * coords,
    const float * feats,
    const float * weight,
    const float * bias,
    float * out,
    int64_t n,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w) {
    const int cd = kernel_d / 2;
    const int ch = kernel_h / 2;
    const int cw = kernel_w / 2;
    for (int64_t row = 0; row < n; ++row) {
        const int32_t * center = coords + 4 * row;
        for (int oc = 0; oc < out_channels; ++oc) {
            float acc = bias == NULL ? 0.0f : bias[oc];
            for (int kd = 0; kd < kernel_d; ++kd) {
                const int nx = center[1] + (kd - cd) * dilation_d;
                for (int kh = 0; kh < kernel_h; ++kh) {
                    const int ny = center[2] + (kh - ch) * dilation_h;
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        const int nz = center[3] + (kw - cw) * dilation_w;
                        const int in_row = sparse_find_coord4_ref(coords, n, center[0], nx, ny, nz);
                        if (in_row < 0) {
                            continue;
                        }
                        const int64_t w_base =
                            ((((int64_t) oc * kernel_d + kd) * kernel_h + kh) * kernel_w + kw) * (int64_t) in_channels;
                        const int64_t f_base = (int64_t) in_row * in_channels;
                        for (int ic = 0; ic < in_channels; ++ic) {
                            acc += feats[f_base + ic] * weight[w_base + ic];
                        }
                    }
                }
            }
            out[row * (int64_t) out_channels + oc] = acc;
        }
    }
}

static void apply_rope_ref(
    const float * x,
    const float * cos_phase,
    const float * sin_phase,
    float * y,
    int batch,
    int tokens,
    int heads,
    int head_dim) {
    const int half_dim = head_dim / 2;
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int t = 0; t < tokens; ++t) {
                for (int p = 0; p < half_dim; ++p) {
                    const int d0 = 2 * p;
                    const int d1 = d0 + 1;
                    const int64_t base =
                        (int64_t) d0 + (int64_t) head_dim * (t + tokens * (h + heads * b));
                    const int64_t idx0 = base;
                    const int64_t idx1 = base + 1;
                    const float c = cos_phase[t * half_dim + p];
                    const float s = sin_phase[t * half_dim + p];
                    const float x0 = x[idx0];
                    const float x1 = x[idx1];
                    y[idx0] = x0 * c - x1 * s;
                    y[idx1] = x0 * s + x1 * c;
                }
            }
        }
    }
}

static void sparse_linear_ref(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int n,
    int in_channels,
    int out_channels) {
    for (int row = 0; row < n; ++row) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float acc = bias == NULL ? 0.0f : bias[oc];
            for (int ic = 0; ic < in_channels; ++ic) {
                acc += x[row * in_channels + ic] * weight[oc * in_channels + ic];
            }
            y[row * out_channels + oc] = acc;
        }
    }
}

static void test_custom_cuda_kernels(void) {
    {
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
        CHECK_TRUE(trellis_cuda_conv3d_f32_host(
            x, w, bias, got, 0, B, IC, ID, IH, IW, OC, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW) == TRELLIS_STATUS_OK);
        for (int i = 0; i < B * OC * OD * OH * OW; ++i) {
            CHECK_CLOSE(got[i], exp[i], 2e-5f);
        }
    }

    {
        enum { B = 1, OC = 2, SCALE = 2, IC = OC * SCALE * SCALE * SCALE, ID = 2, IH = 2, IW = 1 };
        enum { OD = ID * SCALE, OH = IH * SCALE, OW = IW * SCALE };
        float x[B * IC * ID * IH * IW];
        float got[B * OC * OD * OH * OW];
        float exp[B * OC * OD * OH * OW];
        for (int i = 0; i < (int) (sizeof(x) / sizeof(x[0])); ++i) {
            x[i] = (float) (i + 1);
        }
        pixel_shuffle_3d_ref(x, exp, B, IC, ID, IH, IW, SCALE);
        CHECK_TRUE(trellis_cuda_pixel_shuffle_3d_f32_host(x, got, 0, B, IC, ID, IH, IW, SCALE) == TRELLIS_STATUS_OK);
        for (int i = 0; i < B * OC * OD * OH * OW; ++i) {
            CHECK_CLOSE(got[i], exp[i], 1e-6f);
        }
    }

    {
        enum { B = 2, C = 3, D = 2, H = 2, W = 2, N = B * C * D * H * W };
        float x[N];
        float gamma[C] = {1.0f, 0.5f, -1.5f};
        float beta[C] = {0.1f, -0.2f, 0.3f};
        float got[N];
        float exp[N];
        for (int i = 0; i < N; ++i) {
            x[i] = sinf(0.2f * (float) i) + 0.03f * (float) (i % 7);
        }
        channel_layer_norm_3d_ref(x, gamma, beta, exp, B, C, D, H, W, 1e-6f);
        CHECK_TRUE(trellis_cuda_channel_layer_norm_3d_f32_host(x, gamma, beta, got, 0, B, C, D, H, W, 1e-6f) == TRELLIS_STATUS_OK);
        for (int i = 0; i < N; ++i) {
            CHECK_CLOSE(got[i], exp[i], 2e-5f);
        }
    }

    {
        enum { N = 5, IC = 7, OC = 4 };
        float x[N * IC];
        float weight[OC * IC];
        float bias[OC];
        float got[N * OC];
        float exp[N * OC];
        for (int i = 0; i < N * IC; ++i) {
            x[i] = 0.2f * sinf(0.19f * (float) (i + 2)) + 0.01f * (float) (i % 3);
        }
        for (int i = 0; i < OC * IC; ++i) {
            weight[i] = 0.15f * cosf(0.23f * (float) (i - 4));
        }
        for (int i = 0; i < OC; ++i) {
            bias[i] = 0.05f * (float) (i - 2);
        }
        sparse_linear_ref(x, weight, bias, exp, N, IC, OC);
        CHECK_TRUE(trellis_cuda_sparse_linear_f32_host(x, weight, bias, got, 0, N, IC, OC) == TRELLIS_STATUS_OK);
        for (int i = 0; i < N * OC; ++i) {
            CHECK_CLOSE(got[i], exp[i], 2e-5f);
        }
    }

    {
        enum { B = 1, IH = 4, IW = 4, OC = 2, P = 2, T = (IH / P) * (IW / P) };
        float image[B * 3 * IH * IW];
        float weight[OC * 3 * P * P];
        float bias[OC] = {0.25f, -0.5f};
        float got[B * T * OC];
        float exp[B * T * OC];
        for (int i = 0; i < B * 3 * IH * IW; ++i) {
            image[i] = 0.05f * (float) (i - 7);
        }
        for (int i = 0; i < OC * 3 * P * P; ++i) {
            weight[i] = sinf(0.13f * (float) (i + 1));
        }
        dino_patch_embed_ref(image, weight, bias, exp, B, IH, IW, OC, P);
        CHECK_TRUE(trellis_cuda_dino_patch_embed_f32_host(image, weight, bias, got, 0, B, IH, IW, OC, P) == TRELLIS_STATUS_OK);
        for (int i = 0; i < B * T * OC; ++i) {
            CHECK_CLOSE(got[i], exp[i], 2e-5f);
        }
    }

    {
        enum { N = 17 };
        float a[N], b[N], got[N], exp_silu[N], exp_add[N];
        for (int i = 0; i < N; ++i) {
            a[i] = 0.25f * (float) (i - 8);
            b[i] = cosf(0.3f * (float) i);
            exp_silu[i] = silu_ref(a[i]);
            exp_add[i] = a[i] + b[i];
        }
        CHECK_TRUE(trellis_cuda_silu_f32_host(a, got, 0, N) == TRELLIS_STATUS_OK);
        for (int i = 0; i < N; ++i) {
            CHECK_CLOSE(got[i], exp_silu[i], 1e-6f);
        }
        CHECK_TRUE(trellis_cuda_add_f32_host(a, b, got, 0, N) == TRELLIS_STATUS_OK);
        for (int i = 0; i < N; ++i) {
            CHECK_CLOSE(got[i], exp_add[i], 1e-6f);
        }
    }

    {
        enum { N = 6, IC = 3, OC = 2, KD = 3, KH = 3, KW = 3 };
        int32_t coords[N * 4] = {
            0, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
            0, 1, 1, 0,
            0, 3, 0, 0,
        };
        float feats[N * IC];
        float weight[OC * KD * KH * KW * IC];
        float bias[OC] = {0.15f, -0.2f};
        float got_map[N * OC];
        float got_scalar[N * OC];
        float exp[N * OC];
        for (int i = 0; i < N * IC; ++i) {
            feats[i] = 0.2f * sinf(0.31f * (float) (i + 1)) + 0.03f * (float) (i % 4);
        }
        for (int i = 0; i < OC * KD * KH * KW * IC; ++i) {
            weight[i] = 0.1f * cosf(0.07f * (float) (i - 5));
        }
        sparse_subm_conv3d_ref(coords, feats, weight, bias, exp, N, IC, OC, KD, KH, KW, 1, 1, 1);

        const char * saved_backend = getenv("TRELLIS_SPARSE_CONV_BACKEND");
        char * saved_backend_copy = saved_backend == NULL ? NULL : strdup(saved_backend);
        unsetenv("TRELLIS_SPARSE_CONV_BACKEND");
        CHECK_TRUE(trellis_cuda_sparse_subm_conv3d_f32_host(
            coords, feats, weight, bias, got_map, 0, N, IC, OC, KD, KH, KW, 1, 1, 1) == TRELLIS_STATUS_OK);
        CHECK_TRUE(setenv("TRELLIS_SPARSE_CONV_BACKEND", "scalar", 1) == 0);
        CHECK_TRUE(trellis_cuda_sparse_subm_conv3d_f32_host(
            coords, feats, weight, bias, got_scalar, 0, N, IC, OC, KD, KH, KW, 1, 1, 1) == TRELLIS_STATUS_OK);
        if (saved_backend_copy != NULL) {
            CHECK_TRUE(setenv("TRELLIS_SPARSE_CONV_BACKEND", saved_backend_copy, 1) == 0);
            free(saved_backend_copy);
        } else {
            unsetenv("TRELLIS_SPARSE_CONV_BACKEND");
        }

        for (int i = 0; i < N * OC; ++i) {
            CHECK_CLOSE(got_map[i], exp[i], 2e-5f);
            CHECK_CLOSE(got_scalar[i], exp[i], 2e-5f);
            CHECK_CLOSE(got_map[i], got_scalar[i], 1e-6f);
        }
    }

    {
        enum { B = 2, T = 3, H = 2, D = 8, HALF = D / 2 };
        float x[B * H * T * D];
        float cos_phase[T * HALF];
        float sin_phase[T * HALF];
        float got[B * H * T * D];
        float exp[B * H * T * D];
        for (int i = 0; i < B * H * T * D; ++i) {
            x[i] = 0.03f * (float) (i - 7);
        }
        for (int t = 0; t < T; ++t) {
            for (int p = 0; p < HALF; ++p) {
                const float phase = 0.2f * (float) t + 0.07f * (float) p;
                cos_phase[t * HALF + p] = cosf(phase);
                sin_phase[t * HALF + p] = sinf(phase);
            }
        }
        apply_rope_ref(x, cos_phase, sin_phase, exp, B, T, H, D);
        CHECK_TRUE(trellis_cuda_apply_rope_f32_host(x, cos_phase, sin_phase, got, 0, B, T, H, D) == TRELLIS_STATUS_OK);
        for (int i = 0; i < B * H * T * D; ++i) {
            CHECK_CLOSE(got[i], exp[i], 1e-6f);
        }
    }
}

static void test_rope_adjacent_ggml_cuda(void) {
    enum { B = 2, T = 3, H = 2, D = 8, HALF = D / 2 };
    struct ggml_context * ctx = make_graph_ctx();
    CHECK_TRUE(ctx != NULL);
    struct ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, T, H, B);
    struct ggml_tensor * cos_phase = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HALF, T, 1);
    struct ggml_tensor * sin_phase = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HALF, T, 1);
    struct ggml_tensor * y = trellis_ggml_apply_rope_adjacent(ctx, x, cos_phase, sin_phase);
    CHECK_TRUE(y != NULL);

    float x_data[B * H * T * D];
    float cos_data[T * HALF];
    float sin_data[T * HALF];
    float got[B * H * T * D];
    float exp[B * H * T * D];
    for (int i = 0; i < B * H * T * D; ++i) {
        x_data[i] = 0.013f * (float) (i + 3) - 0.2f;
    }
    for (int t = 0; t < T; ++t) {
        for (int p = 0; p < HALF; ++p) {
            const float phase = 0.17f * (float) t + 0.05f * (float) p;
            cos_data[t * HALF + p] = cosf(phase);
            sin_data[t * HALF + p] = sinf(phase);
        }
    }
    apply_rope_ref(x_data, cos_data, sin_data, exp, B, T, H, D);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_gallocr_t alloc = trellis_cuda_new_graph_allocator(&g_cuda);
    CHECK_TRUE(alloc != NULL);
    CHECK_TRUE(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(x, x_data, 0, ggml_nbytes(x));
    ggml_backend_tensor_set(cos_phase, cos_data, 0, ggml_nbytes(cos_phase));
    ggml_backend_tensor_set(sin_phase, sin_data, 0, ggml_nbytes(sin_phase));
    CHECK_TRUE(trellis_cuda_compute_graph(&g_cuda, graph) == TRELLIS_STATUS_OK);
    ggml_backend_tensor_get(y, got, 0, ggml_nbytes(y));
    for (int i = 0; i < B * H * T * D; ++i) {
        CHECK_CLOSE(got[i], exp[i], 2e-6f);
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

static void test_flow_and_sparse_host(void) {
    float x[] = {1.0f, -2.0f, 0.5f};
    float v[] = {0.2f, -0.4f, 1.0f};
    float prev[3], x0[3];
    trellis_flow_euler_step_f32(x, v, 3, 1e-5f, 0.75f, 0.5f, prev, x0);
    for (int i = 0; i < 3; ++i) {
        CHECK_CLOSE(prev[i], x[i] - 0.25f * v[i], 1e-6f);
        float sigma_t = 1e-5f + (1.0f - 1e-5f) * 0.75f;
        CHECK_CLOSE(x0[i], (1.0f - 1e-5f) * x[i] - sigma_t * v[i], 1e-6f);
    }
    float pos[] = {1, 2, 3};
    float neg[] = {-1, 0, 1};
    float cfg[3];
    trellis_flow_cfg_combine_f32(pos, neg, 3, 7.5f, cfg);
    CHECK_CLOSE(cfg[0], 7.5f * 1.0f + (1.0f - 7.5f) * -1.0f, 1e-6f);

    trellis_sparse_tensor_host s;
    s.n = 5;
    s.channels = 2;
    int32_t coords[] = {
        0,0,0,0,
        0,1,0,0,
        0,2,0,0,
        0,3,1,0,
        0,0,2,0,
    };
    float feats[] = {
        1,2,
        3,4,
        5,6,
        7,8,
        9,10,
    };
    s.coords = coords;
    s.feats = feats;
    trellis_sparse_tensor_host down;
    CHECK_TRUE(trellis_sparse_downsample_mean_host(&s, 2, &down) == TRELLIS_STATUS_OK);
    CHECK_TRUE(down.n == 3 && down.channels == 2);
    int32_t exp_coords[] = {0,0,0,0, 0,0,1,0, 0,1,0,0};
    float exp_feats[] = {2,3, 9,10, 6,7};
    for (int i = 0; i < 12; ++i) CHECK_TRUE(down.coords[i] == exp_coords[i]);
    for (int i = 0; i < 6; ++i) CHECK_CLOSE(down.feats[i], exp_feats[i], 1e-6f);
    trellis_sparse_tensor_free(&down);

    trellis_sparse_tensor_host s1;
    s1.n = 2;
    s1.channels = 1;
    int32_t coords1[] = {0,0,0,0, 0,1,0,0};
    float feats1[] = {11, 22};
    s1.coords = coords1;
    s1.feats = feats1;
    trellis_sparse_tensor_host s2c;
    CHECK_TRUE(trellis_sparse_spatial2channel_host(&s1, 2, &s2c) == TRELLIS_STATUS_OK);
    CHECK_TRUE(s2c.n == 1 && s2c.channels == 8);
    CHECK_CLOSE(s2c.feats[0], 11.0f, 1e-6f);
    CHECK_CLOSE(s2c.feats[1], 22.0f, 1e-6f);
    uint8_t subdiv[] = {1, 1, 0, 0, 0, 0, 0, 0};
    trellis_sparse_tensor_host c2s;
    CHECK_TRUE(trellis_sparse_channel2spatial_host(&s2c, subdiv, 2, &c2s) == TRELLIS_STATUS_OK);
    CHECK_TRUE(c2s.n == 2 && c2s.channels == 1);
    CHECK_CLOSE(c2s.feats[0], 11.0f, 1e-6f);
    CHECK_CLOSE(c2s.feats[1], 22.0f, 1e-6f);
    trellis_sparse_tensor_free(&s2c);
    trellis_sparse_tensor_free(&c2s);
}

static void test_stage1_sampler_math(void) {
    float pairs[4];
    CHECK_TRUE(trellis_flow_timestep_pairs_f32(2, 5.0f, pairs, 4) == TRELLIS_STATUS_OK);
    CHECK_CLOSE(pairs[0], 1.0f, 1e-6f);
    CHECK_CLOSE(pairs[1], 2.5f / 3.0f, 1e-6f);
    CHECK_CLOSE(pairs[2], 2.5f / 3.0f, 1e-6f);
    CHECK_CLOSE(pairs[3], 0.0f, 1e-6f);

    float ts[2] = {12.0f, 24.0f};
    float emb[8];
    trellis_timestep_embedding_f32(ts, 2, 4, 10000.0f, emb);
    CHECK_CLOSE(emb[0], cosf(12.0f), 1e-6f);
    CHECK_CLOSE(emb[1], cosf(0.12f), 1e-6f);
    CHECK_CLOSE(emb[2], sinf(12.0f), 1e-6f);
    CHECK_CLOSE(emb[3], sinf(0.12f), 1e-6f);
    CHECK_CLOSE(emb[4], cosf(24.0f), 1e-6f);

    float cos_phase[8 * 4];
    float sin_phase[8 * 4];
    CHECK_TRUE(trellis_rope_3d_phases_f32(2, 8, 1.0f, 10000.0f, cos_phase, sin_phase, 8 * 4) == TRELLIS_STATUS_OK);
    const int token = 5; /* meshgrid token [x=1, y=0, z=1] for resolution 2 */
    CHECK_CLOSE(cos_phase[token * 4 + 0], cosf(1.0f), 1e-6f);
    CHECK_CLOSE(sin_phase[token * 4 + 0], sinf(1.0f), 1e-6f);
    CHECK_CLOSE(cos_phase[token * 4 + 1], 1.0f, 1e-6f);
    CHECK_CLOSE(sin_phase[token * 4 + 1], 0.0f, 1e-6f);
    CHECK_CLOSE(cos_phase[token * 4 + 2], cosf(1.0f), 1e-6f);
    CHECK_CLOSE(sin_phase[token * 4 + 2], sinf(1.0f), 1e-6f);
    CHECK_CLOSE(cos_phase[token * 4 + 3], 1.0f, 1e-6f);
    CHECK_CLOSE(sin_phase[token * 4 + 3], 0.0f, 1e-6f);

    float x_t[] = {1.0f, -2.0f, 0.5f, 3.0f};
    float pos_pred[] = {0.1f, -0.2f, 0.3f, -0.4f};
    float neg_pred[] = {-0.2f, 0.1f, 0.0f, 0.2f};
    float pred[4];
    trellis_flow_cfg_rescale_combine_f32(x_t, pos_pred, neg_pred, 1, 4, 1e-5f, 0.75f, 7.5f, 0.0f, pred);
    for (int i = 0; i < 4; ++i) {
        CHECK_CLOSE(pred[i], 7.5f * pos_pred[i] + (1.0f - 7.5f) * neg_pred[i], 1e-6f);
    }
}

static int file_exists(const char * path) {
    return access(path, R_OK) == 0;
}

static const char * first_existing_path(const char ** paths, int n_paths) {
    for (int i = 0; i < n_paths; ++i) {
        if (file_exists(paths[i])) {
            return paths[i];
        }
    }
    return NULL;
}

static void test_real_stage1_checkpoint_manifests_if_present(void) {
    const char * flow_candidates[] = {
        "TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "../TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "../../TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    };
    const char * decoder_candidates[] = {
        "TRELLIS.2-4B/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        "../TRELLIS.2-4B/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        "../../TRELLIS.2-4B/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    };
    const char * flow = first_existing_path(flow_candidates, 3);
    const char * decoder = first_existing_path(decoder_candidates, 3);
    if (flow == NULL || decoder == NULL) {
        return;
    }

    trellis_checkpoint_report report;
    CHECK_TRUE(trellis_ss_flow_validate_checkpoint(flow, &report) == TRELLIS_STATUS_OK);
    CHECK_TRUE(report.expected_tensors == 640);
    CHECK_TRUE(report.actual_tensors == 640);
    CHECK_TRUE(report.missing_tensors == 0);
    CHECK_TRUE(report.shape_mismatches == 0);
    CHECK_TRUE(report.dtype_mismatches == 0);

    CHECK_TRUE(trellis_ss_decoder_validate_checkpoint(decoder, &report) == TRELLIS_STATUS_OK);
    CHECK_TRUE(report.expected_tensors == 74);
    CHECK_TRUE(report.actual_tensors == 74);
    CHECK_TRUE(report.missing_tensors == 0);
    CHECK_TRUE(report.shape_mismatches == 0);
    CHECK_TRUE(report.dtype_mismatches == 0);
}

static void test_real_stage2_checkpoint_manifests_if_present(void) {
    const char * shape_512_candidates[] = {
        "TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        "../TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        "../../TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    };
    const char * shape_1024_candidates[] = {
        "TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "../TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "../../TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    };
    const char * shape_decoder_candidates[] = {
        "TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "../TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "../../TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
    };
    const char * shape_512 = first_existing_path(shape_512_candidates, 3);
    const char * shape_1024 = first_existing_path(shape_1024_candidates, 3);
    const char * shape_decoder = first_existing_path(shape_decoder_candidates, 3);
    if (shape_512 == NULL || shape_1024 == NULL || shape_decoder == NULL) {
        return;
    }

    trellis_checkpoint_report report;
    CHECK_TRUE(trellis_shape_slat_flow_validate_checkpoint(shape_512, &report) == TRELLIS_STATUS_OK);
    CHECK_TRUE(report.expected_tensors == 640);
    CHECK_TRUE(report.actual_tensors == 640);
    CHECK_TRUE(report.missing_tensors == 0);
    CHECK_TRUE(report.shape_mismatches == 0);
    CHECK_TRUE(report.dtype_mismatches == 0);

    CHECK_TRUE(trellis_shape_slat_flow_validate_checkpoint(shape_1024, &report) == TRELLIS_STATUS_OK);
    CHECK_TRUE(report.expected_tensors == 640);
    CHECK_TRUE(report.actual_tensors == 640);
    CHECK_TRUE(report.missing_tensors == 0);
    CHECK_TRUE(report.shape_mismatches == 0);
    CHECK_TRUE(report.dtype_mismatches == 0);

    CHECK_TRUE(trellis_shape_decoder_validate_checkpoint(shape_decoder, &report) == TRELLIS_STATUS_OK);
    CHECK_TRUE(report.expected_tensors == 292);
    CHECK_TRUE(report.actual_tensors == 292);
    CHECK_TRUE(report.missing_tensors == 0);
    CHECK_TRUE(report.shape_mismatches == 0);
    CHECK_TRUE(report.dtype_mismatches == 0);
}

int main(void) {
    test_safetensors();
    test_flexible_dual_grid_mesh_host();
    test_flow_and_sparse_host();
    test_stage1_sampler_math();
    test_real_stage1_checkpoint_manifests_if_present();
    test_real_stage2_checkpoint_manifests_if_present();

    trellis_status cuda_status = trellis_cuda_init(&g_cuda, 0);
    if (cuda_status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "CUDA backend unavailable: %s\n", trellis_status_string(cuda_status));
        return 77;
    }

    test_linear_cuda();
    test_norms_cuda();
    test_feed_forward_cuda();
    test_timestep_mlp_cuda();
    test_attention_cuda();
    test_self_attention_rope_cuda();
    test_rope_adjacent_ggml_cuda();
    test_custom_cuda_kernels();
    test_tensor_store_loader_cuda();

    trellis_cuda_free(&g_cuda);
    if (g_failures != 0) {
        fprintf(stderr, "%d test failures\n", g_failures);
        return 1;
    }
    printf("trellis2.c tests passed\n");
    return 0;
}
