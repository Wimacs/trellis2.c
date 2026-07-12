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

typedef struct tiny_encoder_weights {
    trellis_sparse_unet_vae_encoder_weights weights;
    float input_w[2 * 2];
    float input_b[2];
    float norm_gamma[2];
    float norm_beta[2];
    float conv1_w[1 * 27 * 2];
    float conv1_b[1];
    float conv2_w[8 * 27 * 8];
    float conv2_b[8];
    float to_latent_w[2 * 8];
    float to_latent_b[2];
} tiny_encoder_weights;

static void tiny_encoder_weights_init(tiny_encoder_weights * tiny) {
    memset(tiny, 0, sizeof(*tiny));
    trellis_sparse_unet_vae_encoder_weights * weights = &tiny->weights;
    weights->in_channels = 2;
    weights->latent_channels = 1;
    weights->levels = 2;
    weights->channels[0] = 2;
    weights->channels[1] = 8;
    weights->blocks_per_level[0] = 0;
    weights->blocks_per_level[1] = 0;

    tiny->input_w[0] = 1.0f;
    tiny->input_w[3] = 1.0f;
    weights->input_w = tiny->input_w;
    weights->input_b = tiny->input_b;

    trellis_sparse_unet_vae_encoder_s2c_block_weights * down = &weights->down_blocks[0];
    down->in_channels = 2;
    down->out_channels = 8;
    tiny->norm_gamma[0] = 1.0f;
    tiny->norm_gamma[1] = 1.0f;
    down->norm1_gamma = tiny->norm_gamma;
    down->norm1_beta = tiny->norm_beta;
    down->conv1_w = tiny->conv1_w;
    down->conv1_b = tiny->conv1_b;
    down->conv2_w = tiny->conv2_w;
    down->conv2_b = tiny->conv2_b;

    tiny->to_latent_w[0] = 1.0f;
    tiny->to_latent_b[0] = 0.125f;
    weights->to_latent_w = tiny->to_latent_w;
    weights->to_latent_b = tiny->to_latent_b;
}

static float expected_latent(const float channels[8]) {
    float mean = 0.0f;
    for (int i = 0; i < 8; ++i) mean += channels[i];
    mean /= 8.0f;
    float variance = 0.0f;
    for (int i = 0; i < 8; ++i) {
        const float d = channels[i] - mean;
        variance += d * d;
    }
    variance /= 8.0f;
    return (channels[0] - mean) / sqrtf(variance + 1e-5f) + 0.125f;
}

static trellis_sparse_backend_kind backend_from_name(const char * name) {
    if (name != NULL && (strcmp(name, "vulkan") == 0 || strcmp(name, "vk") == 0)) {
        return TRELLIS_SPARSE_BACKEND_VULKAN;
    }
    return TRELLIS_SPARSE_BACKEND_CPU;
}

static int run_test(trellis_sparse_backend_kind backend_kind) {
    static const int32_t coords[] = {
        0, 3, 1, 0,
        0, 0, 0, 0,
        0, 1, 0, 0,
        0, 2, 1, 1,
        0, 0, 2, 0,
    };
    static const float feats[] = {
        2.0f, 4.0f,
        1.0f, 3.0f,
       -2.0f, 0.0f,
        6.0f, 2.0f,
        5.0f,-1.0f,
    };
    static const int32_t expected_coords[] = {
        0, 0, 0, 0,
        0, 0, 1, 0,
        0, 1, 0, 0,
    };
    static const int32_t expected_parent[] = {2, 0, 0, 2, 1};
    static const int32_t expected_subidx[] = {3, 0, 1, 6, 0};
    static const float coarse_channels[3][8] = {
        {2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {2.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f,  0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 4.0f, 0.0f},
    };

    tiny_encoder_weights tiny;
    tiny_encoder_weights_init(&tiny);
    trellis_sparse_unet_vae_encoder_forward_options options;
    memset(&options, 0, sizeof(options));
    trellis_sparse_c2s_guides guides;
    memset(&guides, 0, sizeof(guides));
    options.backend_kind = backend_kind;
    options.device = 0;
    options.return_subs = &guides;

    int32_t * coords_out = NULL;
    float * feats_out = NULL;
    int64_t n_out = 0;
    int channels_out = 0;
    const trellis_status status = trellis_sparse_unet_vae_encoder_forward_backend_f32_host(
        &tiny.weights,
        coords,
        feats,
        5,
        &options,
        &coords_out,
        &feats_out,
        &n_out,
        &channels_out);
    if (status == TRELLIS_STATUS_NOT_IMPLEMENTED || status == TRELLIS_STATUS_CUDA_UNAVAILABLE) {
        fprintf(stderr, "sparse encoder backend unavailable: %s\n", trellis_status_string(status));
        return 77;
    }
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    CHECK_TRUE(n_out == 3);
    CHECK_TRUE(channels_out == 1);
    CHECK_TRUE(guides.n_levels == 1);
    CHECK_TRUE(guides.levels[0].n_coords == 5);
    CHECK_TRUE(guides.levels[0].device_map == NULL);

    for (int i = 0; i < 3 * 4; ++i) {
        CHECK_TRUE(coords_out[i] == expected_coords[i]);
    }
    for (int i = 0; i < 5 * 4; ++i) {
        CHECK_TRUE(guides.levels[0].coords_bxyz[i] == coords[i]);
    }
    for (int i = 0; i < 5; ++i) {
        CHECK_TRUE(guides.levels[0].parent[i] == expected_parent[i]);
        CHECK_TRUE(guides.levels[0].subidx[i] == expected_subidx[i]);
    }
    for (int i = 0; i < 3; ++i) {
        const float expected = expected_latent(coarse_channels[i]);
        const float diff = fabsf(feats_out[i] - expected);
        CHECK_TRUE(diff <= 3e-4f * fmaxf(1.0f, fabsf(expected)));
    }

    free(coords_out);
    free(feats_out);
    trellis_sparse_c2s_guides_free(&guides);
    printf("sparse encoder %s test passed\n",
        backend_kind == TRELLIS_SPARSE_BACKEND_VULKAN ? "vulkan" : "cpu");
    return 1;
}

static int test_vulkan_cuda_stub_contract(void) {
    if (strcmp(TRELLIS_DEFAULT_BACKEND, "vulkan") != 0) return 1;
    trellis_sparse_c2s_guides guides;
    memset(&guides, 0x7f, sizeof(guides));
    int32_t * coords_out = (int32_t *) (uintptr_t) 1;
    float * feats_out = (float *) (uintptr_t) 1;
    int64_t n_out = 7;
    int channels_out = 7;
    const trellis_status status =
        trellis_sparse_unet_vae_encoder_forward_f32_host(
            NULL,
            NULL,
            NULL,
            0,
            0,
            0,
            &guides,
            &coords_out,
            &feats_out,
            &n_out,
            &channels_out);
    CHECK_TRUE(status == TRELLIS_STATUS_CUDA_UNAVAILABLE);
    CHECK_TRUE(guides.n_levels == 0);
    CHECK_TRUE(coords_out == NULL);
    CHECK_TRUE(feats_out == NULL);
    CHECK_TRUE(n_out == 0);
    CHECK_TRUE(channels_out == 0);
    return 1;
}

int main(int argc, char ** argv) {
    const trellis_sparse_backend_kind backend = backend_from_name(argc > 1 ? argv[1] : "cpu");
    if (!test_vulkan_cuda_stub_contract()) return 1;
    const int result = run_test(backend);
    if (result == 77) return 77;
    return result ? 0 : 1;
}
