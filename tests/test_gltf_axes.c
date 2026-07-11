#include "image_to_3d_internal.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++g_failures; \
        return; \
    } \
} while (0)

static int vectors_equal(const float * actual, const float * expected, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (fabsf(actual[i] - expected[i]) > 1e-7f) {
            return 0;
        }
    }
    return 1;
}

static void test_trellis_basis(void) {
    float positions[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    float normals[9];
    memcpy(normals, positions, sizeof(normals));
    const float expected[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f,
        0.0f, 1.0f, 0.0f,
    };

    CHECK_TRUE(trellis_pipeline_transform_gltf_vectors(
        TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_TRELLIS,
        positions,
        normals,
        3) == TRELLIS_STATUS_OK);
    CHECK_TRUE(vectors_equal(positions, expected, 9));
    CHECK_TRUE(vectors_equal(normals, expected, 9));
}

static void test_pixal3d_basis(void) {
    float positions[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    float normals[9];
    memcpy(normals, positions, sizeof(normals));
    const float expected[9] = {
        -1.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f,
         0.0f, 0.0f, -1.0f,
    };

    CHECK_TRUE(trellis_pipeline_transform_gltf_vectors(
        TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_PIXAL3D,
        positions,
        normals,
        3) == TRELLIS_STATUS_OK);
    CHECK_TRUE(vectors_equal(positions, expected, 9));
    CHECK_TRUE(vectors_equal(normals, expected, 9));
}

static void test_invalid_transform(void) {
    float positions[3] = {1.0f, 2.0f, 3.0f};
    float normals[3] = {4.0f, 5.0f, 6.0f};
    const float expected_positions[3] = {1.0f, 2.0f, 3.0f};
    const float expected_normals[3] = {4.0f, 5.0f, 6.0f};

    CHECK_TRUE(trellis_pipeline_transform_gltf_vectors(
        (trellis_pipeline_gltf_coordinate_transform) 99,
        positions,
        normals,
        1) == TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(vectors_equal(positions, expected_positions, 3));
    CHECK_TRUE(vectors_equal(normals, expected_normals, 3));
}

int main(void) {
    test_trellis_basis();
    test_pixal3d_basis();
    test_invalid_transform();
    if (g_failures != 0) {
        fprintf(stderr, "%d glTF axis test(s) failed\n", g_failures);
        return 1;
    }
    printf("glTF axis tests passed\n");
    return 0;
}
