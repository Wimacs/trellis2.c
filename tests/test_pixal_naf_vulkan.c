#include "sparse/trellis_sparse_backend.h"
#include "projected_attention.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static float query_value(size_t index) {
    return 0.12f * sinf((float) index * 0.013f) +
        0.03f * cosf((float) index * 0.007f);
}

static float key_value(size_t index) {
    return 0.10f * cosf((float) index * 0.017f) -
        0.02f * sinf((float) index * 0.011f);
}

static float feature_value(size_t index) {
    return 0.75f * sinf((float) index * 0.0031f) +
        0.20f * cosf((float) index * 0.0017f);
}

int main(void) {
    const trellis_pixal_naf_attention_desc desc = {
        .batch = 2,
        .query_h = 20,
        .query_w = 20,
        .feature_h = 10,
        .feature_w = 10,
        .image_resolution = 32,
        .grid_resolution = 4,
    };
    const int32_t coords[] = {
        0, 0, 0, 0,
        0, 3, 3, 3,
        0, 1, 2, 3,
        1, 0, 3, 2,
        1, 2, 1, 0,
    };
    const trellis_pixal_camera cameras[] = {
        {.camera_angle_x = 1.10f, .distance = 2.00f, .mesh_scale = 1.00f},
        {.camera_angle_x = 0.95f, .distance = 2.25f, .mesh_scale = 0.85f},
    };
    const int64_t n_coords = 5;
    const size_t query_count =
        (size_t) desc.batch * desc.query_h * desc.query_w *
        TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS;
    const size_t key_count =
        (size_t) desc.batch * desc.feature_h * desc.feature_w *
        TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS;
    const size_t value_count =
        (size_t) desc.batch * desc.feature_h * desc.feature_w *
        TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS;
    const size_t output_count =
        (size_t) n_coords * TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS;
    float * queries = (float *) malloc(query_count * sizeof(float));
    float * keys = (float *) malloc(key_count * sizeof(float));
    float * values = (float *) malloc(value_count * sizeof(float));
    float * expected = (float *) malloc(output_count * sizeof(float));
    float * actual = (float *) malloc(output_count * sizeof(float));
    if (queries == NULL || keys == NULL || values == NULL || expected == NULL || actual == NULL) {
        fprintf(stderr, "Pixal3D NAF Vulkan test allocation failed\n");
        free(actual);
        free(expected);
        free(values);
        free(keys);
        free(queries);
        return 1;
    }
    for (size_t i = 0; i < query_count; ++i) queries[i] = query_value(i);
    for (size_t i = 0; i < key_count; ++i) keys[i] = key_value(i);
    for (size_t i = 0; i < value_count; ++i) values[i] = feature_value(i);

    trellis_status status = trellis_pixal_naf_attention_project_sparse_reference_f32(
        queries,
        keys,
        values,
        coords,
        n_coords,
        cameras,
        &desc,
        expected,
        output_count);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "Pixal3D NAF CPU reference failed: %s\n", trellis_status_string(status));
        free(actual);
        free(expected);
        free(values);
        free(keys);
        free(queries);
        return 1;
    }

    trellis_sparse_backend * backend = NULL;
    status = trellis_sparse_vulkan_backend_create(0, &backend);
    if (status == TRELLIS_STATUS_NOT_IMPLEMENTED || status == TRELLIS_STATUS_CUDA_UNAVAILABLE) {
        fprintf(stderr, "Vulkan backend unavailable: %s\n", trellis_status_string(status));
        free(actual);
        free(expected);
        free(values);
        free(keys);
        free(queries);
        return 77;
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_pixal_naf_attention_project_sparse_vulkan_f32(
            backend,
            queries,
            keys,
            values,
            coords,
            n_coords,
            cameras,
            &desc,
            actual,
            output_count);
    }
    int failed = status != TRELLIS_STATUS_OK;
    if (failed) {
        fprintf(stderr, "Pixal3D NAF Vulkan operator failed: %s\n", trellis_status_string(status));
    } else {
        float max_error = 0.0f;
        size_t max_index = 0;
        for (size_t i = 0; i < output_count; ++i) {
            const float error = fabsf(actual[i] - expected[i]);
            if (error > max_error) {
                max_error = error;
                max_index = i;
            }
            const float scale = fmaxf(1.0f, fmaxf(fabsf(actual[i]), fabsf(expected[i])));
            if (error > 8.0e-4f * scale) {
                fprintf(
                    stderr,
                    "Pixal3D NAF Vulkan mismatch at %zu: got=%g expected=%g error=%g\n",
                    i,
                    actual[i],
                    expected[i],
                    error);
                failed = 1;
                break;
            }
        }
        if (!failed) {
            printf(
                "Pixal3D NAF Vulkan golden compare passed (max_error=%g at %zu)\n",
                max_error,
                max_index);
        }
    }

    trellis_sparse_backend_destroy(backend);
    free(actual);
    free(expected);
    free(values);
    free(keys);
    free(queries);
    return failed ? 1 : 0;
}
