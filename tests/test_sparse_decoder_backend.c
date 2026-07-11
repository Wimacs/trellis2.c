#include "trellis.h"
#include "sparse/trellis_sparse_backend.h"

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

static trellis_sparse_backend_kind sparse_backend_from_name(const char * name) {
    if (strcmp(name, "cpu") == 0) {
        return TRELLIS_SPARSE_BACKEND_CPU;
    }
    if (strcmp(name, "vulkan") == 0 || strcmp(name, "vk") == 0) {
        return TRELLIS_SPARSE_BACKEND_VULKAN;
    }
    return TRELLIS_SPARSE_BACKEND_CUDA;
}

static void fill_linear(float * w, int in_channels, int out_channels, float scale, float offset) {
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            w[oc * in_channels + ic] =
                sinf((float) (oc * 17 + ic * 5 + 3)) * scale + offset * (float) (oc - ic);
        }
    }
}

static void fill_sparse_conv(float * w, int in_channels, int out_channels, float scale) {
    memset(w, 0, (size_t) out_channels * 27u * (size_t) in_channels * sizeof(float));
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            w[((oc * 27 + 13) * in_channels) + ic] = scale * (float) (1 + ((oc + ic) % 3));
            w[((oc * 27 + 4) * in_channels) + ic] = -0.25f * scale * (float) (1 + (ic % 2));
            w[((oc * 27 + 22) * in_channels) + ic] = 0.15f * scale * (float) (1 + (oc % 2));
        }
    }
}

static void fill_bias(float * b, int n, float scale) {
    for (int i = 0; i < n; ++i) {
        b[i] = scale * (float) (i - n / 2);
    }
}

static void fill_norm(float * gamma, float * beta, int n) {
    for (int i = 0; i < n; ++i) {
        gamma[i] = 0.75f + 0.05f * (float) i;
        beta[i] = -0.15f + 0.03f * (float) i;
    }
}

typedef struct tiny_sparse_weights {
    trellis_sparse_unet_vae_decoder_weights w;
    float from_latent_w[8 * 2];
    float from_latent_b[8];
    float to_subdiv_w[8 * 8];
    float to_subdiv_b[8];
    float c2s_norm1_g[8];
    float c2s_norm1_b[8];
    float c2s_conv1_w[16 * 27 * 8];
    float c2s_conv1_b[16];
    float c2s_conv2_w[2 * 27 * 2];
    float c2s_conv2_b[2];
    float block_conv_w[2 * 27 * 2];
    float block_conv_b[2];
    float block_norm_g[2];
    float block_norm_b[2];
    float block_mlp0_w[8 * 2];
    float block_mlp0_b[8];
    float block_mlp2_w[2 * 8];
    float block_mlp2_b[2];
    float output_w[2 * 2];
    float output_b[2];
} tiny_sparse_weights;

static void tiny_sparse_weights_init(tiny_sparse_weights * tw) {
    memset(tw, 0, sizeof(*tw));
    trellis_sparse_unet_vae_decoder_weights * w = &tw->w;
    w->latent_channels = 2;
    w->out_channels = 2;
    w->pred_subdiv = 1;
    w->levels = 2;
    w->channels[0] = 8;
    w->channels[1] = 2;
    w->blocks_per_level[0] = 0;
    w->blocks_per_level[1] = 1;

    fill_linear(tw->from_latent_w, 2, 8, 0.2f, 0.01f);
    fill_bias(tw->from_latent_b, 8, 0.03f);
    w->from_latent_w = tw->from_latent_w;
    w->from_latent_b = tw->from_latent_b;

    trellis_sparse_unet_vae_decoder_c2s_block_weights * up = &w->up_blocks[0];
    up->in_channels = 8;
    up->out_channels = 2;
    fill_norm(tw->c2s_norm1_g, tw->c2s_norm1_b, 8);
    fill_sparse_conv(tw->c2s_conv1_w, 8, 16, 0.035f);
    fill_bias(tw->c2s_conv1_b, 16, 0.005f);
    fill_sparse_conv(tw->c2s_conv2_w, 2, 2, 0.06f);
    fill_bias(tw->c2s_conv2_b, 2, 0.01f);
    memset(tw->to_subdiv_w, 0, sizeof(tw->to_subdiv_w));
    for (int i = 0; i < 8; ++i) {
        tw->to_subdiv_b[i] = i == 0 ? 1.0f : -1.0f;
    }
    up->norm1_gamma = tw->c2s_norm1_g;
    up->norm1_beta = tw->c2s_norm1_b;
    up->conv1_w = tw->c2s_conv1_w;
    up->conv1_b = tw->c2s_conv1_b;
    up->conv2_w = tw->c2s_conv2_w;
    up->conv2_b = tw->c2s_conv2_b;
    up->to_subdiv_w = tw->to_subdiv_w;
    up->to_subdiv_b = tw->to_subdiv_b;

    trellis_sparse_unet_vae_decoder_convnext_block_weights * block = &w->blocks[1][0];
    block->channels = 2;
    fill_sparse_conv(tw->block_conv_w, 2, 2, 0.04f);
    fill_bias(tw->block_conv_b, 2, 0.02f);
    fill_norm(tw->block_norm_g, tw->block_norm_b, 2);
    fill_linear(tw->block_mlp0_w, 2, 8, 0.08f, 0.01f);
    fill_bias(tw->block_mlp0_b, 8, 0.01f);
    fill_linear(tw->block_mlp2_w, 8, 2, 0.05f, -0.005f);
    fill_bias(tw->block_mlp2_b, 2, 0.02f);
    block->conv_w = tw->block_conv_w;
    block->conv_b = tw->block_conv_b;
    block->norm_gamma = tw->block_norm_g;
    block->norm_beta = tw->block_norm_b;
    block->mlp0_w = tw->block_mlp0_w;
    block->mlp0_b = tw->block_mlp0_b;
    block->mlp2_w = tw->block_mlp2_w;
    block->mlp2_b = tw->block_mlp2_b;

    fill_linear(tw->output_w, 2, 2, 0.11f, 0.02f);
    fill_bias(tw->output_b, 2, 0.025f);
    w->output_w = tw->output_w;
    w->output_b = tw->output_b;
}

static int run_forward(
    const trellis_sparse_unet_vae_decoder_weights * weights,
    trellis_sparse_backend_kind backend,
    int32_t ** coords_out,
    float ** feats_out,
    int64_t * n_out,
    int * channels_out,
    trellis_sparse_c2s_guides * guides_out) {
    const int32_t coords[] = {
        0, 0, 0, 0,
        0, 1, 0, 0,
    };
    const float feats[] = {
        0.25f, -0.50f,
        1.00f, 0.75f,
    };
    trellis_sparse_unet_vae_decoder_forward_options options;
    memset(&options, 0, sizeof(options));
    options.backend_kind = backend;
    options.device = 0;
    options.max_levels = 0;
    options.return_subs = guides_out;
    trellis_status status = trellis_sparse_unet_vae_decoder_forward_backend_f32_host(
        weights,
        coords,
        feats,
        2,
        &options,
        coords_out,
        feats_out,
        n_out,
        channels_out);
    if (status == TRELLIS_STATUS_NOT_IMPLEMENTED || status == TRELLIS_STATUS_CUDA_UNAVAILABLE) {
        fprintf(stderr, "sparse backend unavailable: %s\n", trellis_status_string(status));
        return 77;
    }
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    return 1;
}

static int run_cpu_trim_noop_test(void) {
    trellis_sparse_backend * backend = NULL;
    trellis_status status = trellis_sparse_cpu_backend_create(&backend);
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    size_t released_bytes = SIZE_MAX;
    status = trellis_sparse_backend_trim(
        backend,
        TRELLIS_SPARSE_TRIM_FREE_BUFFERS | TRELLIS_SPARSE_TRIM_WEIGHTS,
        &released_bytes);
    trellis_sparse_backend_destroy(backend);
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    CHECK_TRUE(released_bytes == 0);
    return 1;
}

static int run_vulkan_trim_preserves_device_guides_test(
    const tiny_sparse_weights * tw,
    const int32_t * ref_coords,
    const float * ref_feats,
    int64_t ref_n,
    int ref_channels,
    const trellis_sparse_c2s_guides * ref_guides) {
    const int32_t input_coords[] = {
        0, 0, 0, 0,
        0, 1, 0, 0,
    };
    const float input_feats[] = {
        0.25f, -0.50f,
        1.00f, 0.75f,
    };
    trellis_sparse_backend * backend = NULL;
    trellis_sparse_c2s_guides device_guides;
    int32_t * shape_coords = NULL;
    float * shape_feats = NULL;
    int32_t * guide_coords = NULL;
    int32_t * guide_parent = NULL;
    int32_t * guide_subidx = NULL;
    int32_t * guided_coords = NULL;
    float * guided_feats = NULL;
    int64_t shape_n = 0;
    int shape_channels = 0;
    int64_t guided_n = 0;
    int guided_channels = 0;
    memset(&device_guides, 0, sizeof(device_guides));

    trellis_status status = trellis_sparse_vulkan_backend_create(0, &backend);
    if (status == TRELLIS_STATUS_NOT_IMPLEMENTED || status == TRELLIS_STATUS_CUDA_UNAVAILABLE) {
        fprintf(stderr, "sparse vulkan backend unavailable: %s\n", trellis_status_string(status));
        return 77;
    }
    CHECK_TRUE(status == TRELLIS_STATUS_OK);

    trellis_sparse_unet_vae_decoder_forward_options options;
    memset(&options, 0, sizeof(options));
    options.backend_kind = TRELLIS_SPARSE_BACKEND_VULKAN;
    options.device = 0;
    options.sparse_backend = backend;
    options.return_subs = &device_guides;
    status = trellis_sparse_unet_vae_decoder_forward_backend_f32_host(
        &tw->w,
        input_coords,
        input_feats,
        2,
        &options,
        &shape_coords,
        &shape_feats,
        &shape_n,
        &shape_channels);
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    CHECK_TRUE(shape_n == ref_n && shape_channels == ref_channels);
    CHECK_TRUE(device_guides.n_levels == ref_guides->n_levels);
    CHECK_TRUE(device_guides.levels[0].device_map != NULL);
    CHECK_TRUE(device_guides.levels[0].coords_bxyz == NULL);
    for (int64_t i = 0; i < ref_n * 4; ++i) {
        CHECK_TRUE(shape_coords[i] == ref_coords[i]);
    }
    for (int64_t i = 0; i < ref_n * ref_channels; ++i) {
        CHECK_TRUE(check_close(shape_feats[i], ref_feats[i], 1e-4f, "shared_sparse_shape", (int) i));
    }

    size_t released_bytes = 0;
    status = trellis_sparse_backend_trim(
        backend,
        TRELLIS_SPARSE_TRIM_FREE_BUFFERS | TRELLIS_SPARSE_TRIM_WEIGHTS,
        &released_bytes);
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    CHECK_TRUE(released_bytes > 0);

    const int64_t guide_n = ref_guides->levels[0].n_coords;
    guide_coords = (int32_t *) malloc((size_t) guide_n * 4u * sizeof(int32_t));
    guide_parent = (int32_t *) malloc((size_t) guide_n * sizeof(int32_t));
    guide_subidx = (int32_t *) malloc((size_t) guide_n * sizeof(int32_t));
    CHECK_TRUE(guide_coords != NULL && guide_parent != NULL && guide_subidx != NULL);
    status = backend->ops->download_c2s_map(
        backend,
        (const trellis_sparse_c2s_device_map *) device_guides.levels[0].device_map,
        guide_coords,
        guide_parent,
        guide_subidx,
        guide_n);
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    for (int64_t i = 0; i < guide_n * 4; ++i) {
        CHECK_TRUE(guide_coords[i] == ref_guides->levels[0].coords_bxyz[i]);
    }
    for (int64_t i = 0; i < guide_n; ++i) {
        CHECK_TRUE(guide_parent[i] == ref_guides->levels[0].parent[i]);
        CHECK_TRUE(guide_subidx[i] == ref_guides->levels[0].subidx[i]);
    }

    trellis_sparse_unet_vae_decoder_weights guided_weights = tw->w;
    guided_weights.pred_subdiv = false;
    memset(&options, 0, sizeof(options));
    options.backend_kind = TRELLIS_SPARSE_BACKEND_VULKAN;
    options.device = 0;
    options.sparse_backend = backend;
    options.guide_subs = &device_guides;
    status = trellis_sparse_unet_vae_decoder_forward_backend_f32_host(
        &guided_weights,
        input_coords,
        input_feats,
        2,
        &options,
        &guided_coords,
        &guided_feats,
        &guided_n,
        &guided_channels);
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    CHECK_TRUE(guided_n == ref_n && guided_channels == ref_channels);
    for (int64_t i = 0; i < ref_n * 4; ++i) {
        CHECK_TRUE(guided_coords[i] == ref_coords[i]);
    }
    for (int64_t i = 0; i < ref_n * ref_channels; ++i) {
        CHECK_TRUE(check_close(guided_feats[i], ref_feats[i], 1e-4f, "trimmed_sparse_guide", (int) i));
    }

    free(guided_feats);
    free(guided_coords);
    free(guide_subidx);
    free(guide_parent);
    free(guide_coords);
    free(shape_feats);
    free(shape_coords);
    trellis_sparse_c2s_guides_free(&device_guides);
    trellis_sparse_backend_destroy(backend);
    printf("sparse vulkan trim/device-guide tests passed\n");
    return 1;
}

static float large_feat(int64_t row, int channel) {
    return sinf((float) (row % 997) * 0.013f + (float) channel * 0.17f) +
        0.001f * (float) ((row + channel * 11) % 23);
}

static int check_large_sample(
    const float * y,
    int64_t row,
    int out_channels,
    const float * expected) {
    for (int oc = 0; oc < out_channels; ++oc) {
        char name[64];
        snprintf(name, sizeof(name), "large_dispatch_row_%lld", (long long) row);
        CHECK_TRUE(check_close(
            y[row * (int64_t) out_channels + oc],
            expected[oc],
            2e-4f,
            name,
            oc));
    }
    return 1;
}

static int run_direct_sparse_conv_compare(trellis_sparse_backend_kind backend_kind) {
    const int64_t n = 8;
    const int in_channels = 3;
    const int out_channels = 2;
    const int32_t coords[] = {
        0, 1, 1, 1,
        0, 0, 1, 1,
        0, 2, 1, 1,
        0, 1, 0, 1,
        0, 1, 2, 1,
        0, 1, 1, 0,
        0, 1, 1, 2,
        0, 0, 0, 0,
    };
    float feats[8 * 3];
    float conv_w[2 * 27 * 3];
    float conv_b[2] = {0.125f, -0.075f};
    for (int64_t r = 0; r < n; ++r) {
        for (int c = 0; c < in_channels; ++c) {
            feats[r * in_channels + c] = sinf((float) r * 0.31f + (float) c * 0.17f) + 0.05f * (float) (r - c);
        }
    }
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int v = 0; v < 27; ++v) {
            for (int ic = 0; ic < in_channels; ++ic) {
                conv_w[(oc * 27 + v) * in_channels + ic] =
                    0.01f * (float) ((oc + 1) * (ic + 2)) + 0.0025f * (float) (v - 13);
            }
        }
    }

    trellis_sparse_backend * cpu = NULL;
    trellis_sparse_backend * backend = NULL;
    trellis_sparse_buffer * cpu_x = NULL;
    trellis_sparse_buffer * cpu_y = NULL;
    trellis_sparse_buffer * x = NULL;
    trellis_sparse_buffer * y = NULL;
    trellis_sparse_rulebook * cpu_rulebook = NULL;
    trellis_sparse_rulebook * rulebook = NULL;
    float ref[8 * 2];
    float got[8 * 2];
    int ok = 1;

    trellis_status status = trellis_sparse_cpu_backend_create(&cpu);
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    if (backend_kind == TRELLIS_SPARSE_BACKEND_VULKAN) {
        status = trellis_sparse_vulkan_backend_create(0, &backend);
    } else {
        status = trellis_sparse_cpu_backend_create(&backend);
    }
    if (status == TRELLIS_STATUS_NOT_IMPLEMENTED || status == TRELLIS_STATUS_CUDA_UNAVAILABLE) {
        trellis_sparse_backend_destroy(cpu);
        fprintf(stderr, "sparse backend unavailable: %s\n", trellis_status_string(status));
        return 77;
    }
    CHECK_TRUE(status == TRELLIS_STATUS_OK);

    status = cpu->ops->upload_f32(cpu, feats, (size_t) n * in_channels, &cpu_x);
    if (status == TRELLIS_STATUS_OK) status = cpu->ops->alloc_f32(cpu, (size_t) n * out_channels, &cpu_y);
    if (status == TRELLIS_STATUS_OK) status = cpu->ops->build_rulebook(cpu, coords, n, &cpu_rulebook);
    if (status == TRELLIS_STATUS_OK) status = cpu->ops->sparse_conv3d(cpu, cpu_rulebook, cpu_x, conv_w, conv_b, cpu_y, n, in_channels, out_channels);
    if (status == TRELLIS_STATUS_OK) status = cpu->ops->download_f32(cpu, cpu_y, ref, (size_t) n * out_channels);
    CHECK_TRUE(status == TRELLIS_STATUS_OK);

    status = backend->ops->upload_f32(backend, feats, (size_t) n * in_channels, &x);
    if (status == TRELLIS_STATUS_OK) status = backend->ops->alloc_f32(backend, (size_t) n * out_channels, &y);
    if (status == TRELLIS_STATUS_OK) status = backend->ops->build_rulebook(backend, coords, n, &rulebook);
    if (status == TRELLIS_STATUS_OK) status = backend->ops->sparse_conv3d(backend, rulebook, x, conv_w, conv_b, y, n, in_channels, out_channels);
    if (status == TRELLIS_STATUS_OK) status = backend->ops->download_f32(backend, y, got, (size_t) n * out_channels);
    CHECK_TRUE(status == TRELLIS_STATUS_OK);

    for (int64_t i = 0; i < n * out_channels; ++i) {
        if (!check_close(got[i], ref[i], 2e-4f, "direct_sparse_conv", (int) i)) {
            ok = 0;
            break;
        }
    }

    if (cpu != NULL && cpu->ops != NULL) {
        cpu->ops->free_rulebook(cpu, cpu_rulebook);
        cpu->ops->free_buffer(cpu, cpu_y);
        cpu->ops->free_buffer(cpu, cpu_x);
    }
    if (backend != NULL && backend->ops != NULL) {
        backend->ops->free_rulebook(backend, rulebook);
        backend->ops->free_buffer(backend, y);
        backend->ops->free_buffer(backend, x);
    }
    trellis_sparse_backend_destroy(cpu);
    trellis_sparse_backend_destroy(backend);
    CHECK_TRUE(ok);
    printf("direct sparse conv compare passed\n");
    return 1;
}

static int run_large_vulkan_dispatch_tests(void) {
    enum {
        DIM_X = 102,
        DIM_Y = 102,
        DIM_Z = 101,
        IN_CHANNELS = 3,
        OUT_CHANNELS = 2,
    };
    const int64_t n = (int64_t) DIM_X * DIM_Y * DIM_Z;
    CHECK_TRUE(n > 65535ll * 16ll);

    trellis_sparse_backend * backend = NULL;
    trellis_status status = trellis_sparse_vulkan_backend_create(0, &backend);
    if (status == TRELLIS_STATUS_NOT_IMPLEMENTED || status == TRELLIS_STATUS_CUDA_UNAVAILABLE) {
        fprintf(stderr, "sparse vulkan backend unavailable: %s\n", trellis_status_string(status));
        return 77;
    }
    CHECK_TRUE(status == TRELLIS_STATUS_OK);

    float * x = (float *) malloc((size_t) n * IN_CHANNELS * sizeof(float));
    float * y = (float *) malloc((size_t) n * OUT_CHANNELS * sizeof(float));
    int32_t * coords = (int32_t *) malloc((size_t) n * 4u * sizeof(int32_t));
    float * conv_w = (float *) calloc((size_t) OUT_CHANNELS * 27u * IN_CHANNELS, sizeof(float));
    trellis_sparse_buffer * x_buf = NULL;
    trellis_sparse_buffer * y_buf = NULL;
    trellis_sparse_rulebook * rulebook = NULL;
    int ok = 1;
    if (x == NULL || y == NULL || coords == NULL || conv_w == NULL) {
        ok = 0;
        goto cleanup;
    }

    for (int64_t row = 0; row < n; ++row) {
        for (int c = 0; c < IN_CHANNELS; ++c) {
            x[row * IN_CHANNELS + c] = large_feat(row, c);
        }
    }
    const float linear_w[OUT_CHANNELS * IN_CHANNELS] = {
        0.25f, -0.50f, 0.125f,
        -0.75f, 0.20f, 0.40f,
    };
    const float linear_b[OUT_CHANNELS] = {0.10f, -0.30f};
    status = backend->ops->upload_f32(backend, x, (size_t) n * IN_CHANNELS, &x_buf);
    if (status == TRELLIS_STATUS_OK) {
        status = backend->ops->alloc_f32(backend, (size_t) n * OUT_CHANNELS, &y_buf);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = backend->ops->linear(backend, x_buf, linear_w, linear_b, y_buf, n, IN_CHANNELS, OUT_CHANNELS);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = backend->ops->download_f32(backend, y_buf, y, (size_t) n * OUT_CHANNELS);
    }
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    const int64_t sample_rows[] = {0, 1, 65535ll * 16ll - 1ll, 65535ll * 16ll, n - 1};
    for (size_t si = 0; si < sizeof(sample_rows) / sizeof(sample_rows[0]); ++si) {
        const int64_t row = sample_rows[si];
        float expected[OUT_CHANNELS];
        for (int oc = 0; oc < OUT_CHANNELS; ++oc) {
            expected[oc] = linear_b[oc];
            for (int ic = 0; ic < IN_CHANNELS; ++ic) {
                expected[oc] += large_feat(row, ic) * linear_w[oc * IN_CHANNELS + ic];
            }
        }
        CHECK_TRUE(check_large_sample(y, row, OUT_CHANNELS, expected));
    }

    backend->ops->free_buffer(backend, y_buf);
    backend->ops->free_buffer(backend, x_buf);
    y_buf = NULL;
    x_buf = NULL;

    int64_t row = 0;
    for (int iz = 0; iz < DIM_Z; ++iz) {
        for (int iy = 0; iy < DIM_Y; ++iy) {
            for (int ix = 0; ix < DIM_X; ++ix) {
                coords[row * 4 + 0] = 0;
                coords[row * 4 + 1] = ix * 2;
                coords[row * 4 + 2] = iy * 2;
                coords[row * 4 + 3] = iz * 2;
                ++row;
            }
        }
    }
    const float conv_b[OUT_CHANNELS] = {-0.05f, 0.15f};
    for (int oc = 0; oc < OUT_CHANNELS; ++oc) {
        for (int ic = 0; ic < IN_CHANNELS; ++ic) {
            conv_w[((oc * 27 + 13) * IN_CHANNELS) + ic] = 0.07f * (float) (oc + 1) - 0.03f * (float) ic;
        }
    }
    status = backend->ops->upload_f32(backend, x, (size_t) n * IN_CHANNELS, &x_buf);
    if (status == TRELLIS_STATUS_OK) {
        status = backend->ops->alloc_f32(backend, (size_t) n * OUT_CHANNELS, &y_buf);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = backend->ops->build_rulebook(backend, coords, n, &rulebook);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = backend->ops->sparse_conv3d(backend, rulebook, x_buf, conv_w, conv_b, y_buf, n, IN_CHANNELS, OUT_CHANNELS);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = backend->ops->download_f32(backend, y_buf, y, (size_t) n * OUT_CHANNELS);
    }
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    for (size_t si = 0; si < sizeof(sample_rows) / sizeof(sample_rows[0]); ++si) {
        const int64_t r = sample_rows[si];
        float expected[OUT_CHANNELS];
        for (int oc = 0; oc < OUT_CHANNELS; ++oc) {
            expected[oc] = conv_b[oc];
            for (int ic = 0; ic < IN_CHANNELS; ++ic) {
                expected[oc] += large_feat(r, ic) * conv_w[((oc * 27 + 13) * IN_CHANNELS) + ic];
            }
        }
        CHECK_TRUE(check_large_sample(y, r, OUT_CHANNELS, expected));
    }

cleanup:
    if (backend != NULL && backend->ops != NULL) {
        backend->ops->free_rulebook(backend, rulebook);
        backend->ops->free_buffer(backend, y_buf);
        backend->ops->free_buffer(backend, x_buf);
    }
    trellis_sparse_backend_destroy(backend);
    free(conv_w);
    free(coords);
    free(y);
    free(x);
    CHECK_TRUE(ok);
    printf("large sparse vulkan dispatch tests passed\n");
    return 1;
}

int main(int argc, char ** argv) {
    const char * backend_name = argc > 1 ? argv[1] : "cpu";
    trellis_sparse_backend_kind backend = sparse_backend_from_name(backend_name);
    if (backend == TRELLIS_SPARSE_BACKEND_CUDA) {
        fprintf(stderr, "usage: %s [cpu|vulkan]\n", argv[0]);
        return 1;
    }

    tiny_sparse_weights tw;
    tiny_sparse_weights_init(&tw);

    int32_t * ref_coords = NULL;
    float * ref_feats = NULL;
    int64_t ref_n = 0;
    int ref_channels = 0;
    trellis_sparse_c2s_guides ref_guides;
    memset(&ref_guides, 0, sizeof(ref_guides));
    int ok = run_forward(&tw.w, TRELLIS_SPARSE_BACKEND_CPU, &ref_coords, &ref_feats, &ref_n, &ref_channels, &ref_guides);
    if (ok != 1) {
        return ok;
    }
    CHECK_TRUE(run_cpu_trim_noop_test() == 1);
    CHECK_TRUE(ref_n == 2);
    CHECK_TRUE(ref_channels == 2);
    CHECK_TRUE(ref_guides.n_levels == 1);
    CHECK_TRUE(ref_guides.levels[0].n_coords == 2);

    if (backend == TRELLIS_SPARSE_BACKEND_CPU) {
        free(ref_coords);
        free(ref_feats);
        trellis_sparse_c2s_guides_free(&ref_guides);
        printf("sparse decoder backend tests passed on cpu\n");
        return 0;
    }

    int32_t * got_coords = NULL;
    float * got_feats = NULL;
    int64_t got_n = 0;
    int got_channels = 0;
    trellis_sparse_c2s_guides got_guides;
    memset(&got_guides, 0, sizeof(got_guides));
    ok = run_forward(&tw.w, backend, &got_coords, &got_feats, &got_n, &got_channels, &got_guides);
    if (ok != 1) {
        free(ref_coords);
        free(ref_feats);
        trellis_sparse_c2s_guides_free(&ref_guides);
        return ok;
    }

    CHECK_TRUE(got_n == ref_n);
    CHECK_TRUE(got_channels == ref_channels);
    CHECK_TRUE(got_guides.n_levels == ref_guides.n_levels);
    for (int64_t i = 0; i < ref_n * 4; ++i) {
        CHECK_TRUE(got_coords[i] == ref_coords[i]);
    }
    for (int64_t i = 0; i < ref_n * (int64_t) ref_channels; ++i) {
        CHECK_TRUE(check_close(got_feats[i], ref_feats[i], 1e-4f, "sparse_decoder", (int) i));
    }
    for (int64_t i = 0; i < ref_guides.levels[0].n_coords * 4; ++i) {
        CHECK_TRUE(got_guides.levels[0].coords_bxyz[i] == ref_guides.levels[0].coords_bxyz[i]);
    }

    int direct_ok = run_direct_sparse_conv_compare(backend);
    if (direct_ok != 1) {
        free(ref_coords);
        free(ref_feats);
        free(got_coords);
        free(got_feats);
        trellis_sparse_c2s_guides_free(&ref_guides);
        trellis_sparse_c2s_guides_free(&got_guides);
        return direct_ok;
    }

    if (backend == TRELLIS_SPARSE_BACKEND_VULKAN) {
        int trim_ok = run_vulkan_trim_preserves_device_guides_test(
            &tw,
            ref_coords,
            ref_feats,
            ref_n,
            ref_channels,
            &ref_guides);
        if (trim_ok != 1) {
            free(ref_coords);
            free(ref_feats);
            free(got_coords);
            free(got_feats);
            trellis_sparse_c2s_guides_free(&ref_guides);
            trellis_sparse_c2s_guides_free(&got_guides);
            return trim_ok;
        }
        int large_ok = run_large_vulkan_dispatch_tests();
        if (large_ok != 1) {
            free(ref_coords);
            free(ref_feats);
            free(got_coords);
            free(got_feats);
            trellis_sparse_c2s_guides_free(&ref_guides);
            trellis_sparse_c2s_guides_free(&got_guides);
            return large_ok;
        }
    }

    free(ref_coords);
    free(ref_feats);
    free(got_coords);
    free(got_feats);
    trellis_sparse_c2s_guides_free(&ref_guides);
    trellis_sparse_c2s_guides_free(&got_guides);
    printf("sparse decoder backend tests passed on %s\n", backend_name);
    return 0;
}
