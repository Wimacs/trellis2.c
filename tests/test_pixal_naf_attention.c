#include "pixal_projection.h"
#include "projected_attention.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static void check_close(float actual, float expected, float tolerance, int line) {
    const float scale = fmaxf(1.0f, fmaxf(fabsf(actual), fabsf(expected)));
    if (fabsf(actual - expected) > tolerance * scale) {
        fprintf(stderr, "CHECK_CLOSE failed at %s:%d: actual=%g expected=%g\n",
            __FILE__, line, actual, expected);
        ++failures;
    }
}

#define CHECK_CLOSE(actual, expected, tolerance) \
    check_close((actual), (expected), (tolerance), __LINE__)

static float * alloc_floats(size_t count) {
    float * data = (float *) calloc(count, sizeof(float));
    CHECK_TRUE(data != NULL);
    return data;
}

static size_t offset4(
    int batch,
    int y,
    int x,
    int height,
    int width,
    int channels) {
    return ((((size_t) batch * (size_t) height + (size_t) y) * (size_t) width +
             (size_t) x) *
            (size_t) channels);
}

static void test_attention_and_fused_projection(void) {
    const trellis_pixal_naf_attention_desc desc = {
        .batch = 2,
        .query_h = 20,
        .query_w = 20,
        .feature_h = 10,
        .feature_w = 10,
        .image_resolution = 32,
        .grid_resolution = 3,
    };
    const size_t query_count =
        (size_t) desc.batch * desc.query_h * desc.query_w *
        TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS;
    const size_t key_count =
        (size_t) desc.batch * desc.feature_h * desc.feature_w *
        TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS;
    const size_t value_count =
        (size_t) desc.batch * desc.feature_h * desc.feature_w *
        TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS;
    float * queries = alloc_floats(query_count);
    float * keys = alloc_floats(key_count);
    float * values = alloc_floats(value_count);
    float pixel[TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS];
    if (queries == NULL || keys == NULL || values == NULL) goto cleanup;

    for (int batch = 0; batch < desc.batch; ++batch) {
        for (int y = 0; y < desc.feature_h; ++y) {
            for (int x = 0; x < desc.feature_w; ++x) {
                const size_t base = offset4(
                    batch,
                    y,
                    x,
                    desc.feature_h,
                    desc.feature_w,
                    TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS);
                for (int channel = 0;
                     channel < TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS;
                     ++channel) {
                    values[base + (size_t) channel] =
                        1000.0f * (float) batch + 10.0f * (float) y +
                        (float) x + 0.001f * (float) channel;
                }
            }
        }
    }

    /* Zero Q/K gives an exactly uniform 9x9 softmax. At the right edge the
       NATTEN window shifts from low-resolution [0,8] to [1,9]. */
    CHECK_TRUE(trellis_pixal_naf_attention_pixel_reference_f32(
        queries, keys, values, &desc, 0, 0, 0, pixel,
        TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS) == TRELLIS_STATUS_OK);
    CHECK_CLOSE(pixel[0], 44.0f, 2e-6f);
    CHECK_CLOSE(pixel[255], 44.255f, 2e-6f);
    CHECK_CLOSE(pixel[1023], 45.023f, 2e-6f);
    CHECK_TRUE(trellis_pixal_naf_attention_pixel_reference_f32(
        queries, keys, values, &desc, 0, 19, 19, pixel,
        TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS) == TRELLIS_STATUS_OK);
    CHECK_CLOSE(pixel[0], 55.0f, 2e-6f);
    CHECK_CLOSE(pixel[1023], 56.023f, 2e-6f);

    /* Populate one scalar in each 64-D Q/K head. Head zero at (8,8) has
       score(y,x)=0.02*(10*y+x), which independently checks the 1/8 scale. */
    for (int batch = 0; batch < desc.batch; ++batch) {
        for (int y = 0; y < desc.query_h; ++y) {
            for (int x = 0; x < desc.query_w; ++x) {
                const size_t base = offset4(
                    batch,
                    y,
                    x,
                    desc.query_h,
                    desc.query_w,
                    TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS);
                for (int head = 0; head < TRELLIS_PIXAL_NAF_ATTENTION_HEADS; ++head) {
                    queries[base + (size_t) head *
                        TRELLIS_PIXAL_NAF_ATTENTION_QUERY_HEAD_DIM] =
                        0.25f * (float) (head + 1) +
                        0.001f * (float) (y * desc.query_w + x);
                }
            }
        }
        for (int y = 0; y < desc.feature_h; ++y) {
            for (int x = 0; x < desc.feature_w; ++x) {
                const size_t base = offset4(
                    batch,
                    y,
                    x,
                    desc.feature_h,
                    desc.feature_w,
                    TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS);
                for (int head = 0; head < TRELLIS_PIXAL_NAF_ATTENTION_HEADS; ++head) {
                    keys[base + (size_t) head *
                        TRELLIS_PIXAL_NAF_ATTENTION_QUERY_HEAD_DIM] =
                        0.02f * (float) (10 * y + x) / (float) (head + 1);
                }
            }
        }
        queries[offset4(
            batch,
            8,
            8,
            desc.query_h,
            desc.query_w,
            TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS)] = 8.0f;
    }
    CHECK_TRUE(trellis_pixal_naf_attention_pixel_reference_f32(
        queries, keys, values, &desc, 0, 8, 8, pixel,
        TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS) == TRELLIS_STATUS_OK);
    float expected_numerator = 0.0f;
    float expected_denominator = 0.0f;
    for (int y = 0; y < 9; ++y) {
        for (int x = 0; x < 9; ++x) {
            const float spatial = (float) (10 * y + x);
            const float weight = expf(0.02f * spatial - 1.76f);
            expected_numerator += weight * spatial;
            expected_denominator += weight;
        }
    }
    CHECK_CLOSE(pixel[0], expected_numerator / expected_denominator, 2e-6f);

    /* Materialize the CPU-oracle HR map only in the test, then prove the
       fused path matches the established projection sampler. */
    const size_t hr_count =
        (size_t) desc.query_h * desc.query_w *
        TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS;
    float * hr = alloc_floats(hr_count);
    if (hr == NULL) goto cleanup;
    for (int y = 0; y < desc.query_h; ++y) {
        for (int x = 0; x < desc.query_w; ++x) {
            CHECK_TRUE(trellis_pixal_naf_attention_pixel_reference_f32(
                queries,
                keys,
                values,
                &desc,
                0,
                y,
                x,
                hr + ((size_t) y * desc.query_w + (size_t) x) *
                    TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS,
                TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS) == TRELLIS_STATUS_OK);
        }
    }
    const int32_t coords[] = {
        0, 0, 0, 0,
        0, 1, 2, 1,
        0, 2, 1, 2,
        1, 0, 0, 0,
    };
    const int32_t batch_zero_coords[] = {
        0, 0, 0, 0,
        0, 1, 2, 1,
        0, 2, 1, 2,
    };
    const trellis_pixal_camera cameras[] = {
        {.camera_angle_x = 1.2f, .distance = 2.0f, .mesh_scale = 1.0f},
        {.camera_angle_x = 1.2f, .distance = 2.0f, .mesh_scale = 1.0f},
    };
    float * fused = alloc_floats(
        4u * TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS);
    float * materialized = alloc_floats(
        3u * TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS);
    if (fused != NULL && materialized != NULL) {
        CHECK_TRUE(trellis_pixal_naf_attention_project_sparse_reference_f32(
            queries,
            keys,
            values,
            coords,
            4,
            cameras,
            &desc,
            fused,
            4u * TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS) == TRELLIS_STATUS_OK);
        CHECK_TRUE(trellis_pixal_project_patch_features_sparse_f32(
            hr,
            desc.query_h,
            desc.query_w,
            TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS,
            desc.image_resolution,
            desc.grid_resolution,
            cameras,
            batch_zero_coords,
            3,
            materialized,
            3u * TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS) == TRELLIS_STATUS_OK);
        for (int token = 0; token < 3; ++token) {
            for (int channel = 0;
                 channel < TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS;
                 ++channel) {
                CHECK_CLOSE(
                    fused[(size_t) token * TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS +
                        (size_t) channel],
                    materialized[(size_t) token *
                        TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS +
                        (size_t) channel],
                    2e-5f);
            }
        }
        for (int channel = 0;
             channel < TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS;
             ++channel) {
            CHECK_CLOSE(
                fused[3u * TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS +
                    (size_t) channel],
                fused[channel] + 1000.0f,
                2e-5f);
        }
    }
    free(materialized);
    free(fused);
    free(hr);

    trellis_pixal_naf_attention_desc invalid_desc = desc;
    invalid_desc.feature_h = 8;
    invalid_desc.query_h = 16;
    CHECK_TRUE(trellis_pixal_naf_attention_pixel_reference_f32(
        queries, keys, values, &invalid_desc, 0, 0, 0, pixel,
        TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS) == TRELLIS_STATUS_INVALID_ARGUMENT);

cleanup:
    free(values);
    free(keys);
    free(queries);
}

int main(void) {
    test_attention_and_fused_projection();
    if (failures != 0) {
        fprintf(stderr, "%d Pixal3D NAF attention test failures\n", failures);
        return 1;
    }
    printf("Pixal3D NAF attention tests passed\n");
    return 0;
}
