#include "projected_attention.h"

#include <cuda_runtime_api.h>

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
    int device_count = 0;
    cudaError_t cuda_error = cudaGetDeviceCount(&device_count);
    if (cuda_error != cudaSuccess || device_count <= 0) {
        fprintf(
            stderr,
            "CUDA unavailable: %s\n",
            cuda_error == cudaSuccess ? "no CUDA devices" : cudaGetErrorString(cuda_error));
        return 77;
    }
    int caller_device = 0;
    cuda_error = cudaGetDevice(&caller_device);
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    const int execution_device =
        device_count > 1 ? (caller_device + 1) % device_count : caller_device;

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
    float * actual = (float *) malloc((output_count + 7u) * sizeof(float));
    if (queries == NULL || keys == NULL || values == NULL || expected == NULL || actual == NULL) {
        fprintf(stderr, "Pixal3D NAF CUDA test allocation failed\n");
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
    for (size_t i = 0; i < output_count + 7u; ++i) actual[i] = -123.0f;

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

    status = trellis_pixal_naf_attention_project_sparse_cuda_host_f32(
        queries,
        keys,
        values,
        coords,
        n_coords,
        cameras,
        &desc,
        execution_device,
        actual,
        output_count + 7u);
    int device_after = -1;
    cuda_error = cudaGetDevice(&device_after);
    int failed = status != TRELLIS_STATUS_OK;
    int device_restore_failed = 0;
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice after host wrapper failed: %s\n", cudaGetErrorString(cuda_error));
        failed = 1;
        device_restore_failed = 1;
    } else if (device_after != caller_device) {
        fprintf(
            stderr,
            "Pixal3D NAF CUDA host wrapper changed current device: before=%d execution=%d after=%d\n",
            caller_device,
            execution_device,
            device_after);
        failed = 1;
        device_restore_failed = 1;
    }
    if (status == TRELLIS_STATUS_CUDA_UNAVAILABLE && !device_restore_failed) {
        fprintf(stderr, "CUDA unavailable: %s\n", trellis_status_string(status));
        free(actual);
        free(expected);
        free(values);
        free(keys);
        free(queries);
        return 77;
    }
    if (failed) {
        fprintf(stderr, "Pixal3D NAF CUDA operator failed: %s\n", trellis_status_string(status));
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
                    "Pixal3D NAF CUDA mismatch at %zu: got=%g expected=%g error=%g\n",
                    i,
                    actual[i],
                    expected[i],
                    error);
                failed = 1;
                break;
            }
        }
        for (size_t i = output_count; i < output_count + 7u; ++i) {
            if (actual[i] != -123.0f) {
                fprintf(stderr, "Pixal3D NAF CUDA overwrote output tail at %zu\n", i);
                failed = 1;
            }
        }
        if (!failed) {
            printf(
                "Pixal3D NAF CUDA golden compare passed (max_error=%g at %zu)\n",
                max_error,
                max_index);
        }
    }

    free(actual);
    free(expected);
    free(values);
    free(keys);
    free(queries);
    return failed ? 1 : 0;
}
