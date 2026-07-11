#include "pixal_naf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++g_failures; \
        goto cleanup; \
    } \
} while (0)

static int close_enough(float actual, float expected, float tolerance) {
    const float scale = fmaxf(1.0f, fmaxf(fabsf(actual), fabsf(expected)));
    return fabsf(actual - expected) <= tolerance * scale;
}

int main(int argc, char ** argv) {
    if (argc > 3) {
        fprintf(stderr, "usage: %s converted-naf.safetensors [cpu|cuda|vulkan]\n", argv[0]);
        return 2;
    }
    const char * weights_path = getenv("TRELLIS_NAF_SAFETENSORS");
    trellis_backend_kind backend_kind = TRELLIS_BACKEND_CPU;
    if (argc == 2) {
        trellis_backend_kind parsed_kind;
        if (trellis_backend_kind_from_name(argv[1], &parsed_kind) == TRELLIS_STATUS_OK) {
            backend_kind = parsed_kind;
        } else {
            weights_path = argv[1];
        }
    } else if (argc == 3) {
        weights_path = argv[1];
        if (trellis_backend_kind_from_name(argv[2], &backend_kind) != TRELLIS_STATUS_OK) {
            fprintf(stderr, "unknown backend: %s\n", argv[2]);
            return 2;
        }
    }
    if (weights_path == NULL || weights_path[0] == '\0') {
        weights_path = "/tmp/naf_release.safetensors";
    }
    FILE * probe = fopen(weights_path, "rb");
    if (probe == NULL) {
        fprintf(stderr, "NAF safetensors unavailable, skipping: %s\n", weights_path);
        return 77;
    }
    fclose(probe);

    static const int query_indices[] = {
        0, 1, 15, 16, 31, 32, 47, 63, 64, 95, 127, 128,
        159, 191, 192, 223, 255, 256, 511, 767, 1024, 1536, 2047, 2303,
    };
    static const float query_golden[] = {
        -4.92498064f, 5.74215031f, 0.867199838f, -0.856079340f,
        1.48437536f, 2.10368967f, -0.00200959295f, 0.567934036f,
        6.80373859f, -0.0455352701f, -0.102560773f, 8.47960281f,
        0.0966181755f, 1.06273496f, 4.18252087f, 1.90186679f,
        2.08955789f, -4.90303230f, 2.52157807f, 2.19837475f,
        4.19805622f, 1.35163856f, 2.28353381f, 1.89639401f,
    };
    static const int key_indices[] = {
        0, 1, 15, 16, 31, 32, 47, 63, 64, 95, 127, 128, 159, 191, 192, 223, 255,
    };
    static const float key_golden[] = {
        0.273776770f, 2.03302217f, 0.172402278f, 0.0847127959f,
        0.901356459f, -0.140667811f, -0.404440343f, -0.681335509f,
        -0.137376785f, -0.355574280f, -0.756316066f, 1.08700418f,
        0.314677835f, 1.18140066f, 0.727774084f, 0.786937892f, 2.07416058f,
    };

    enum { B = 1, H = 3, W = 3, KH = 1, KW = 1, C = 256 };
    float image[B * 3 * H * W];
    float query_host[B * H * W * C];
    float key_host[B * KH * KW * C];
    float query_ggml[B * H * W * C];
    float key_ggml[B * KH * KW * C];
    for (size_t i = 0; i < sizeof(image) / sizeof(image[0]); ++i) {
        image[i] = (float) ((int) (i % 17) - 8) / 8.0f;
    }

    trellis_pixal_naf_weights host_weights = {0};
    trellis_pixal_naf_ggml_weights ggml_weights = {0};
    trellis_tensor_store store = {0};
    trellis_backend_context backend = {0};
    int store_initialized = 0;
    int backend_initialized = 0;
    char issue[256] = {0};

    CHECK_TRUE(
        trellis_pixal_naf_load_weights_f32(weights_path, &host_weights, issue, sizeof(issue)) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(host_weights.n_tensors == TRELLIS_PIXAL_NAF_WEIGHT_COUNT);
    CHECK_TRUE(trellis_pixal_naf_weight_name(0) != NULL);
    CHECK_TRUE(trellis_pixal_naf_weight_name(36) != NULL);
    float * first_host_tensor = host_weights.tensors[0].data;
    issue[0] = '\0';
    CHECK_TRUE(
        trellis_pixal_naf_load_weights_f32(
            weights_path,
            &host_weights,
            issue,
            sizeof(issue)) == TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(host_weights.n_tensors == TRELLIS_PIXAL_NAF_WEIGHT_COUNT);
    CHECK_TRUE(host_weights.tensors[0].data == first_host_tensor);
    CHECK_TRUE(issue[0] != '\0');
    issue[0] = '\0';

    CHECK_TRUE(
        trellis_pixal_naf_query_key_forward_host(
            &host_weights,
            image,
            B,
            H,
            W,
            H,
            W,
            KH,
            KW,
            query_host,
            sizeof(query_host) / sizeof(query_host[0]),
            key_host,
            sizeof(key_host) / sizeof(key_host[0])) == TRELLIS_STATUS_OK);

    for (size_t i = 0; i < sizeof(query_indices) / sizeof(query_indices[0]); ++i) {
        if (!close_enough(query_host[query_indices[i]], query_golden[i], 4e-4f)) {
            fprintf(
                stderr,
                "query golden mismatch index=%d got=%g expected=%g\n",
                query_indices[i],
                query_host[query_indices[i]],
                query_golden[i]);
            ++g_failures;
        }
    }
    for (size_t i = 0; i < sizeof(key_indices) / sizeof(key_indices[0]); ++i) {
        if (!close_enough(key_host[key_indices[i]], key_golden[i], 4e-4f)) {
            fprintf(
                stderr,
                "key golden mismatch index=%d got=%g expected=%g\n",
                key_indices[i],
                key_host[key_indices[i]],
                key_golden[i]);
            ++g_failures;
        }
    }
    CHECK_TRUE(g_failures == 0);

    CHECK_TRUE(trellis_backend_init(&backend, backend_kind, 0) == TRELLIS_STATUS_OK);
    backend_initialized = 1;
    CHECK_TRUE(trellis_tensor_store_init(&store, 64, 0) == TRELLIS_STATUS_OK);
    store_initialized = 1;
    size_t loaded = 0;
    CHECK_TRUE(
        trellis_tensor_store_load_safetensors_f32(&store, &backend, weights_path, false, &loaded) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(loaded == TRELLIS_PIXAL_NAF_WEIGHT_COUNT);
    issue[0] = '\0';
    CHECK_TRUE(
        trellis_pixal_naf_bind_ggml_weights(&store, &ggml_weights, issue, sizeof(issue)) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(
        trellis_pixal_naf_query_key_forward_ggml_host(
            &backend,
            &ggml_weights,
            image,
            B,
            H,
            W,
            H,
            W,
            KH,
            KW,
            query_ggml,
            sizeof(query_ggml) / sizeof(query_ggml[0]),
            key_ggml,
            sizeof(key_ggml) / sizeof(key_ggml[0])) == TRELLIS_STATUS_OK);

    float max_query_diff = 0.0f;
    float max_key_diff = 0.0f;
    for (size_t i = 0; i < sizeof(query_host) / sizeof(query_host[0]); ++i) {
        const float diff = fabsf(query_host[i] - query_ggml[i]);
        if (diff > max_query_diff) max_query_diff = diff;
    }
    for (size_t i = 0; i < sizeof(key_host) / sizeof(key_host[0]); ++i) {
        const float diff = fabsf(key_host[i] - key_ggml[i]);
        if (diff > max_key_diff) max_key_diff = diff;
    }
    CHECK_TRUE(max_query_diff < 2e-3f);
    CHECK_TRUE(max_key_diff < 2e-3f);
    printf(
        "Pixal NAF encoder (%s) passed: max query diff=%g key diff=%g\n",
        trellis_backend_kind_name(backend_kind),
        max_query_diff,
        max_key_diff);

    const char * smoke_size_text = getenv("TRELLIS_NAF_SMOKE_SIZE");
    if (smoke_size_text != NULL && smoke_size_text[0] != '\0') {
        const int smoke_size = atoi(smoke_size_text);
        CHECK_TRUE(smoke_size == 512 || smoke_size == 1024);
        const int smoke_key_size = smoke_size / 16;
        const size_t smoke_image_count = (size_t) 3 * smoke_size * smoke_size;
        const size_t smoke_query_count = (size_t) smoke_size * smoke_size * C;
        const size_t smoke_key_count = (size_t) smoke_key_size * smoke_key_size * C;
        float * smoke_image = (float *) malloc(smoke_image_count * sizeof(float));
        float * smoke_query = (float *) malloc(smoke_query_count * sizeof(float));
        float * smoke_key = (float *) malloc(smoke_key_count * sizeof(float));
        if (smoke_image == NULL || smoke_query == NULL || smoke_key == NULL) {
            free(smoke_image);
            free(smoke_query);
            free(smoke_key);
            fprintf(stderr, "out of memory allocating %dx%d NAF smoke buffers\n", smoke_size, smoke_size);
            ++g_failures;
            goto cleanup;
        }
        for (size_t i = 0; i < smoke_image_count; ++i) {
            smoke_image[i] = (float) ((int) (i % 251) - 125) / 125.0f;
        }
        const trellis_status smoke_status = trellis_pixal_naf_query_key_forward_ggml_host(
            &backend,
            &ggml_weights,
            smoke_image,
            1,
            smoke_size,
            smoke_size,
            smoke_size,
            smoke_size,
            smoke_key_size,
            smoke_key_size,
            smoke_query,
            smoke_query_count,
            smoke_key,
            smoke_key_count);
        if (smoke_status != TRELLIS_STATUS_OK ||
            !isfinite(smoke_query[0]) || !isfinite(smoke_query[smoke_query_count / 2]) ||
            !isfinite(smoke_key[0]) || !isfinite(smoke_key[smoke_key_count - 1])) {
            fprintf(stderr, "NAF %dx%d smoke failed: %s\n", smoke_size, smoke_size, trellis_status_string(smoke_status));
            ++g_failures;
        } else {
            printf(
                "Pixal NAF %dx%d smoke passed: query=%.1f MiB key=%.1f MiB samples=(%g,%g)\n",
                smoke_size,
                smoke_size,
                (double) (smoke_query_count * sizeof(float)) / (1024.0 * 1024.0),
                (double) (smoke_key_count * sizeof(float)) / (1024.0 * 1024.0),
                smoke_query[smoke_query_count / 2],
                smoke_key[smoke_key_count - 1]);
        }
        free(smoke_image);
        free(smoke_query);
        free(smoke_key);
        CHECK_TRUE(g_failures == 0);
    }

cleanup:
    if (store_initialized) trellis_tensor_store_free(&store);
    if (backend_initialized) trellis_backend_free(&backend);
    trellis_pixal_naf_free_weights(&host_weights);
    if (g_failures != 0 && issue[0] != '\0') {
        fprintf(stderr, "first issue: %s\n", issue);
    }
    return g_failures == 0 ? 0 : 1;
}
