#include "labels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
        return; \
    } \
} while (0)

static void test_transfers_and_clusters_voxel_colors(void) {
    const float positions[] = {
        -0.40f, -0.40f, 0.0f,
        -0.20f, -0.40f, 0.0f,
        -0.40f, -0.20f, 0.0f,
         0.20f,  0.20f, 0.0f,
         0.40f,  0.20f, 0.0f,
         0.20f,  0.40f, 0.0f,
    };
    const uint32_t triangles[] = {0, 1, 2, 3, 4, 5};
    const int32_t coords[] = {
        0, 1, 1, 5,
        0, 3, 1, 5,
        0, 1, 3, 5,
        0, 7, 7, 5,
        0, 9, 7, 5,
        0, 7, 9, 5,
    };
    const float attrs[] = {
        0.98f, 0.03f, 0.02f,
        1.00f, 0.02f, 0.01f,
        0.96f, 0.04f, 0.03f,
        0.02f, 0.04f, 0.98f,
        0.03f, 0.02f, 1.00f,
        0.01f, 0.05f, 0.96f,
    };
    uint32_t * labels = NULL;
    size_t semantic_count = 0;
    CHECK_TRUE(
        trellis_mesh_segmentation_labels_from_voxels(
            positions, 6, triangles, 2, coords, attrs, 6, 3, 10,
            1, 32.0f / 255.0f, &labels, &semantic_count) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(semantic_count == 2);
    CHECK_TRUE(labels[0] != labels[1]);
    free(labels);
}

static void test_merges_nearby_palette_bins(void) {
    const float positions[] = {
        -0.4f, -0.4f, 0.0f,
        -0.3f, -0.4f, 0.0f,
        -0.4f, -0.3f, 0.0f,
    };
    const uint32_t triangles[] = {0, 1, 2};
    const int32_t coords[] = {0, 1, 1, 5, 0, 2, 1, 5, 0, 1, 2, 5};
    const float attrs[] = {
        0.50f, 0.50f, 0.50f,
        0.56f, 0.50f, 0.50f,
        0.51f, 0.49f, 0.50f,
    };
    uint32_t * labels = NULL;
    size_t semantic_count = 0;
    CHECK_TRUE(
        trellis_mesh_segmentation_labels_from_voxels(
            positions, 3, triangles, 1, coords, attrs, 3, 3, 10,
            1, 0.10f, &labels, &semantic_count) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(semantic_count == 1);
    CHECK_TRUE(labels[0] == 0);
    free(labels);
}

static void test_rejects_non_finite_decoder_colors(void) {
    const float positions[] = {
        -0.4f, -0.4f, 0.0f,
        -0.3f, -0.4f, 0.0f,
        -0.4f, -0.3f, 0.0f,
    };
    const uint32_t triangles[] = {0, 1, 2};
    const int32_t coords[] = {0, 1, 1, 5, 0, 2, 1, 5, 0, 1, 2, 5};
    float attrs[] = {
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
    };
    uint32_t * labels = (uint32_t *) (uintptr_t) 1u;
    size_t semantic_count = 99;
    attrs[4] = NAN;
    CHECK_TRUE(
        trellis_mesh_segmentation_labels_from_voxels(
            positions, 3, triangles, 1, coords, attrs, 3, 3, 10,
            1, 0.10f, &labels, &semantic_count) ==
        TRELLIS_STATUS_PARSE_ERROR);
    CHECK_TRUE(labels == NULL);
    CHECK_TRUE(semantic_count == 0);
}

int main(void) {
    test_transfers_and_clusters_voxel_colors();
    test_merges_nearby_palette_bins();
    test_rejects_non_finite_decoder_colors();
    if (failures != 0) return 1;
    printf("voxel-to-face segmentation labels passed\n");
    return 0;
}
