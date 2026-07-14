#include "partition.h"

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

static void test_semantic_and_instance_partitioning(void) {
    /* Two connected left faces, two connected right faces, plus a disconnected
     * triangle that repeats the left semantic label. */
    const uint32_t triangles[] = {
        0, 1, 2, 1, 3, 2,
        1, 4, 3, 4, 5, 3,
        6, 7, 8,
    };
    const uint32_t labels[] = {10, 10, 20, 20, 10};
    uint32_t * parts = NULL;
    size_t part_count = 0;
    CHECK_TRUE(
        trellis_mesh_segmentation_partition_faces(
            triangles, 5, 9, labels, 1, &parts, &part_count) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(part_count == 3);
    CHECK_TRUE(parts[0] == parts[1]);
    CHECK_TRUE(parts[2] == parts[3]);
    CHECK_TRUE(parts[0] != parts[2]);
    CHECK_TRUE(parts[4] != parts[0]);
    free(parts);
}

static void test_absorbs_tiny_edge_island(void) {
    const uint32_t triangles[] = {
        0, 1, 2,
        1, 3, 2,
        1, 4, 3,
    };
    const uint32_t labels[] = {1, 1, 99};
    uint32_t * parts = NULL;
    size_t part_count = 0;
    CHECK_TRUE(
        trellis_mesh_segmentation_partition_faces(
            triangles, 3, 5, labels, 2, &parts, &part_count) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(part_count == 1);
    CHECK_TRUE(parts[0] == 0 && parts[1] == 0 && parts[2] == 0);
    free(parts);
}

static void test_rejects_invalid_indices(void) {
    const uint32_t triangles[] = {0, 1, 9};
    const uint32_t labels[] = {0};
    uint32_t * parts = NULL;
    size_t part_count = 0;
    CHECK_TRUE(
        trellis_mesh_segmentation_partition_faces(
            triangles, 1, 3, labels, 1, &parts, &part_count) ==
        TRELLIS_STATUS_PARSE_ERROR);
    CHECK_TRUE(parts == NULL && part_count == 0);
}

static void test_geometric_welding_connects_uv_seams(void) {
    /* The diagonal is duplicated as two independent vertex pairs, as happens
     * at glTF UV/material seams. Geometric welding must recover one surface. */
    const float positions[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    const uint32_t triangles[] = {0, 1, 2, 3, 4, 5};
    const uint32_t labels[] = {7, 7};
    uint32_t * parts = NULL;
    size_t part_count = 0;
    CHECK_TRUE(
        trellis_mesh_segmentation_partition_faces_geometric(
            positions,
            triangles,
            2,
            6,
            labels,
            1,
            1e-5f,
            &parts,
            &part_count) == TRELLIS_STATUS_OK);
    CHECK_TRUE(part_count == 1);
    CHECK_TRUE(parts[0] == parts[1]);
    free(parts);
}

static void test_geometric_welding_preserves_collapsed_faces(void) {
    /* Cover each pairwise collapse and a face where all three distinct source
     * vertices weld together. None may become a parse error or be discarded. */
    const float positions[] = {
        0.0f, 0.0f, 0.0f,
        5e-6f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        10.0f, 0.0f, 0.0f,
        10.0f, 1.0f, 0.0f,
        10.000005f, 1.0f, 0.0f,
        20.0f, 0.0f, 0.0f,
        20.0f, 1.0f, 0.0f,
        20.000005f, 0.0f, 0.0f,
        30.0f, 0.0f, 0.0f,
        30.000003f, 0.0f, 0.0f,
        30.000006f, 0.0f, 0.0f,
    };
    const uint32_t triangles[] = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11,
    };
    const uint32_t labels[] = {7, 7, 7, 7};
    uint32_t * parts = NULL;
    size_t part_count = 0;
    CHECK_TRUE(
        trellis_mesh_segmentation_partition_faces_geometric(
            positions,
            triangles,
            4,
            12,
            labels,
            1,
            1e-5f,
            &parts,
            &part_count) == TRELLIS_STATUS_OK);
    CHECK_TRUE(part_count == 4);
    CHECK_TRUE(parts != NULL);
    CHECK_TRUE(parts[0] != parts[1]);
    CHECK_TRUE(parts[0] != parts[2]);
    CHECK_TRUE(parts[0] != parts[3]);
    CHECK_TRUE(parts[1] != parts[2]);
    CHECK_TRUE(parts[1] != parts[3]);
    CHECK_TRUE(parts[2] != parts[3]);
    free(parts);
}

static void test_collapsed_face_keeps_unaffected_adjacency(void) {
    /* The first face collapses vertices 0 and 1, but its unaffected 0-2 edge
     * must still join the UV-seam duplicates 3-4 on the neighboring face. */
    const float positions[] = {
        0.0f, 0.0f, 0.0f,
        5e-6f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,
    };
    const uint32_t triangles[] = {0, 1, 2, 3, 4, 5};
    const uint32_t labels[] = {7, 7};
    uint32_t * parts = NULL;
    size_t part_count = 0;
    CHECK_TRUE(
        trellis_mesh_segmentation_partition_faces_geometric(
            positions,
            triangles,
            2,
            6,
            labels,
            1,
            1e-5f,
            &parts,
            &part_count) == TRELLIS_STATUS_OK);
    CHECK_TRUE(part_count == 1);
    CHECK_TRUE(parts != NULL && parts[0] == parts[1]);
    free(parts);
}

static void test_geometric_welding_rejects_source_self_edges(void) {
    const float positions[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    const uint32_t triangles[][3] = {
        {0, 0, 2},
        {0, 1, 1},
        {0, 1, 0},
    };
    const uint32_t labels[] = {7};
    for (size_t case_index = 0;
         case_index < sizeof(triangles) / sizeof(triangles[0]);
         ++case_index) {
        uint32_t * parts = NULL;
        size_t part_count = 0;
        CHECK_TRUE(
            trellis_mesh_segmentation_partition_faces_geometric(
                positions,
                triangles[case_index],
                1,
                3,
                labels,
                1,
                1e-5f,
                &parts,
                &part_count) == TRELLIS_STATUS_PARSE_ERROR);
        CHECK_TRUE(parts == NULL && part_count == 0);
    }
}

int main(void) {
    test_semantic_and_instance_partitioning();
    test_absorbs_tiny_edge_island();
    test_rejects_invalid_indices();
    test_geometric_welding_connects_uv_seams();
    test_geometric_welding_preserves_collapsed_faces();
    test_collapsed_face_keeps_unaffected_adjacency();
    test_geometric_welding_rejects_source_self_edges();
    if (failures != 0) return 1;
    printf("mesh semantic-to-instance partitioning passed\n");
    return 0;
}
