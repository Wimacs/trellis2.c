#include "partition.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void test_small_shell_modes_and_scale_gates(void) {
    /* A large two-face body, a low-poly but model-scale triangle, and one
     * genuinely microscopic disconnected shell. Only the last shell passes
     * both the 1% diagonal and 1e-5 area gates. */
    const float positions[] = {
        0.0f, 0.0f, 0.0f,
        100.0f, 0.0f, 0.0f,
        0.0f, 100.0f, 0.0f,
        100.0f, 100.0f, 0.0f,
        200.0f, 0.0f, 0.0f,
        210.0f, 0.0f, 0.0f,
        200.0f, 10.0f, 0.0f,
        1.0f, 1.0f, 0.1f,
        1.001f, 1.0f, 0.1f,
        1.0f, 1.001f, 0.1f,
    };
    const uint32_t triangles[] = {
        0, 1, 2,
        1, 3, 2,
        4, 5, 6,
        7, 8, 9,
    };
    const uint32_t labels[] = {1, 1, 2, 1};
    uint32_t * legacy = NULL;
    uint32_t * keep = NULL;
    uint32_t * merge = NULL;
    uint32_t * discard = NULL;
    size_t legacy_count = 0;
    size_t keep_count = 0;
    size_t merge_count = 0;
    size_t discard_count = 0;
    trellis_mesh_segmentation_small_part_stats keep_stats;
    trellis_mesh_segmentation_small_part_stats merge_stats;
    trellis_mesh_segmentation_small_part_stats discard_stats;

    CHECK_TRUE(trellis_mesh_segmentation_partition_faces_geometric(
        positions, triangles, 4, 10, labels, 2, 1e-5f,
        &legacy, &legacy_count) == TRELLIS_STATUS_OK);
    CHECK_TRUE(trellis_mesh_segmentation_partition_faces_geometric_ex(
        positions, triangles, 4, 10, labels, 2, 1e-5f,
        TRELLIS_MESH_SEGMENTATION_SMALL_PART_KEEP,
        &keep, &keep_count, &keep_stats) == TRELLIS_STATUS_OK);
    CHECK_TRUE(legacy_count == 3 && keep_count == legacy_count);
    CHECK_TRUE(memcmp(legacy, keep, 4u * sizeof(uint32_t)) == 0);
    CHECK_TRUE(keep_stats.input_part_count == 3);
    CHECK_TRUE(keep_stats.output_part_count == 3);
    CHECK_TRUE(keep_stats.candidate_shell_count == 0);

    CHECK_TRUE(trellis_mesh_segmentation_partition_faces_geometric_ex(
        positions, triangles, 4, 10, labels, 2, 1e-5f,
        TRELLIS_MESH_SEGMENTATION_SMALL_PART_MERGE,
        &merge, &merge_count, &merge_stats) == TRELLIS_STATUS_OK);
    CHECK_TRUE(merge_count == 2);
    CHECK_TRUE(merge[0] == merge[1] && merge[3] == merge[0]);
    CHECK_TRUE(merge[2] != merge[0]);
    CHECK_TRUE(merge_stats.geometric_shell_count == 3);
    CHECK_TRUE(merge_stats.candidate_shell_count == 1);
    CHECK_TRUE(merge_stats.candidate_part_count == 1);
    CHECK_TRUE(merge_stats.affected_face_count == 1);
    CHECK_TRUE(merge_stats.input_part_count == 3);
    CHECK_TRUE(merge_stats.output_part_count == 2);

    CHECK_TRUE(trellis_mesh_segmentation_partition_faces_geometric_ex(
        positions, triangles, 4, 10, labels, 2, 1e-5f,
        TRELLIS_MESH_SEGMENTATION_SMALL_PART_DISCARD,
        &discard, &discard_count, &discard_stats) == TRELLIS_STATUS_OK);
    CHECK_TRUE(discard_count == 2);
    CHECK_TRUE(discard[0] == discard[1]);
    CHECK_TRUE(discard[2] != discard[0]);
    CHECK_TRUE(discard[3] == UINT32_MAX);
    CHECK_TRUE(discard_stats.candidate_shell_count == 1);
    CHECK_TRUE(discard_stats.affected_face_count == 1);

    free(discard);
    free(merge);
    free(keep);
    free(legacy);
}

static void test_small_shell_degenerate_scale_falls_back_to_keep(void) {
    /* With zero surface area there is no meaningful relative area scale. */
    const float positions[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f,
        10.0f, 0.0f, 0.0f,
        11.0f, 0.0f, 0.0f,
        12.0f, 0.0f, 0.0f,
    };
    const uint32_t triangles[] = {0, 1, 2, 3, 4, 5};
    const uint32_t labels[] = {1, 1};
    uint32_t * parts = NULL;
    size_t part_count = 0;
    trellis_mesh_segmentation_small_part_stats stats;
    CHECK_TRUE(trellis_mesh_segmentation_partition_faces_geometric_ex(
        positions, triangles, 2, 6, labels, 16, 1e-5f,
        TRELLIS_MESH_SEGMENTATION_SMALL_PART_DISCARD,
        &parts, &part_count, &stats) == TRELLIS_STATUS_OK);
    CHECK_TRUE(part_count == 2);
    CHECK_TRUE(parts[0] != UINT32_MAX && parts[1] != UINT32_MAX);
    CHECK_TRUE(stats.candidate_shell_count == 0);
    CHECK_TRUE(stats.affected_face_count == 0);
    free(parts);
}

static void test_multisemantic_micro_shell_moves_as_one(void) {
    /* The ten-face microscopic strip has semantic component sizes 4/1/1/4.
     * The legacy edge-neighbor absorption leaves two logical parts, but shell
     * classification ignores those labels and merges every strip face to the
     * same retained body part. */
    float positions[16u * 3u] = {0};
    uint32_t triangles[12u * 3u] = {0};
    uint32_t labels[12u] = {0};
    positions[0] = 0.0f; positions[1] = 0.0f;
    positions[3] = 100.0f; positions[4] = 0.0f;
    positions[6] = 0.0f; positions[7] = 100.0f;
    positions[9] = 100.0f; positions[10] = 100.0f;
    triangles[0] = 0; triangles[1] = 1; triangles[2] = 2;
    triangles[3] = 1; triangles[4] = 3; triangles[5] = 2;
    labels[0] = labels[1] = 1;
    for (size_t vertex = 0; vertex < 12u; ++vertex) {
        const size_t output = (vertex + 4u) * 3u;
        positions[output] = 1.0f + (float) vertex * 0.0001f;
        positions[output + 1u] = 1.0f +
            (vertex % 2u != 0 ? 0.0001f : 0.0f);
        positions[output + 2u] = 0.1f;
    }
    for (size_t face = 0; face < 10u; ++face) {
        triangles[(face + 2u) * 3u] = (uint32_t) face + 4u;
        triangles[(face + 2u) * 3u + 1u] = (uint32_t) face + 5u;
        triangles[(face + 2u) * 3u + 2u] = (uint32_t) face + 6u;
        labels[face + 2u] = face < 4u ? 1u :
            face == 4u ? 2u : face == 5u ? 3u : 4u;
    }

    uint32_t * parts = NULL;
    size_t part_count = 0;
    trellis_mesh_segmentation_small_part_stats stats;
    CHECK_TRUE(trellis_mesh_segmentation_partition_faces_geometric_ex(
        positions, triangles, 12, 16, labels, 11, 1e-5f,
        TRELLIS_MESH_SEGMENTATION_SMALL_PART_MERGE,
        &parts, &part_count, &stats) == TRELLIS_STATUS_OK);
    CHECK_TRUE(part_count == 1);
    for (size_t face = 0; face < 12u; ++face) {
        CHECK_TRUE(parts[face] == 0u);
    }
    CHECK_TRUE(stats.geometric_shell_count == 2);
    CHECK_TRUE(stats.candidate_shell_count == 1);
    CHECK_TRUE(stats.candidate_part_count == 2);
    CHECK_TRUE(stats.affected_face_count == 10);
    free(parts);
}

static void test_connected_semantic_patch_is_not_discarded(void) {
    const float positions[] = {
        0.0f, 0.0f, 0.0f,
        100.0f, 0.0f, 0.0f,
        0.0f, 100.0f, 0.0f,
        100.0f, 100.0f, 0.0f,
        101.0f, 100.0f, 0.0f,
    };
    const uint32_t triangles[] = {
        0, 1, 2,
        1, 3, 2,
        1, 4, 3,
    };
    const uint32_t labels[] = {1, 1, 99};
    uint32_t * parts = NULL;
    size_t part_count = 0;
    trellis_mesh_segmentation_small_part_stats stats;
    CHECK_TRUE(trellis_mesh_segmentation_partition_faces_geometric_ex(
        positions, triangles, 3, 5, labels, 2, 1e-5f,
        TRELLIS_MESH_SEGMENTATION_SMALL_PART_DISCARD,
        &parts, &part_count, &stats) == TRELLIS_STATUS_OK);
    CHECK_TRUE(part_count == 1);
    CHECK_TRUE(parts[0] == 0u && parts[1] == 0u && parts[2] == 0u);
    CHECK_TRUE(stats.geometric_shell_count == 1);
    CHECK_TRUE(stats.candidate_shell_count == 0);
    CHECK_TRUE(stats.affected_face_count == 0);
    free(parts);
}

static void test_small_shell_merge_uses_distance_before_semantic_label(void) {
    /* The tiny label-1 shell is spatially adjacent to the label-2 quad and
     * very far from another label-1 quad. Nearest is the primary contract;
     * semantic compatibility may only break an exact distance tie. */
    const float positions[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        100.0f, 0.0f, 0.0f,
        101.0f, 0.0f, 0.0f,
        100.0f, 1.0f, 0.0f,
        101.0f, 1.0f, 0.0f,
        2.0f, 0.5f, 0.1f,
        2.0001f, 0.5f, 0.1f,
        2.0f, 0.5001f, 0.1f,
    };
    const uint32_t triangles[] = {
        0, 1, 2,
        1, 3, 2,
        4, 5, 6,
        5, 7, 6,
        8, 9, 10,
    };
    const uint32_t labels[] = {2, 2, 1, 1, 1};
    uint32_t * first = NULL;
    uint32_t * second = NULL;
    size_t first_count = 0;
    size_t second_count = 0;
    trellis_mesh_segmentation_small_part_stats first_stats;
    trellis_mesh_segmentation_small_part_stats second_stats;

    CHECK_TRUE(trellis_mesh_segmentation_partition_faces_geometric_ex(
        positions, triangles, 5, 11, labels, 2, 1e-5f,
        TRELLIS_MESH_SEGMENTATION_SMALL_PART_MERGE,
        &first, &first_count, &first_stats) == TRELLIS_STATUS_OK);
    CHECK_TRUE(trellis_mesh_segmentation_partition_faces_geometric_ex(
        positions, triangles, 5, 11, labels, 2, 1e-5f,
        TRELLIS_MESH_SEGMENTATION_SMALL_PART_MERGE,
        &second, &second_count, &second_stats) == TRELLIS_STATUS_OK);
    CHECK_TRUE(first_count == 2 && second_count == 2);
    CHECK_TRUE(first[0] == first[1]);
    CHECK_TRUE(first[2] == first[3]);
    CHECK_TRUE(first[0] != first[2]);
    CHECK_TRUE(first[4] == first[0]); /* near different-label target wins */
    CHECK_TRUE(memcmp(first, second, 5u * sizeof(uint32_t)) == 0);
    CHECK_TRUE(memcmp(&first_stats, &second_stats, sizeof(first_stats)) == 0);
    free(second);
    free(first);
}

static void test_small_shell_rejects_invalid_mode(void) {
    const float positions[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    const uint32_t triangles[] = {0, 1, 2};
    const uint32_t labels[] = {1};
    uint32_t * parts = NULL;
    size_t part_count = 0;
    CHECK_TRUE(trellis_mesh_segmentation_partition_faces_geometric_ex(
        positions, triangles, 1, 3, labels, 16, 1e-5f,
        (trellis_mesh_segmentation_small_part_mode) 99,
        &parts, &part_count, NULL) == TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(parts == NULL && part_count == 0);
}

int main(void) {
    test_semantic_and_instance_partitioning();
    test_absorbs_tiny_edge_island();
    test_rejects_invalid_indices();
    test_geometric_welding_connects_uv_seams();
    test_geometric_welding_preserves_collapsed_faces();
    test_collapsed_face_keeps_unaffected_adjacency();
    test_geometric_welding_rejects_source_self_edges();
    test_small_shell_modes_and_scale_gates();
    test_small_shell_degenerate_scale_falls_back_to_keep();
    test_multisemantic_micro_shell_moves_as_one();
    test_connected_semantic_patch_is_not_discarded();
    test_small_shell_merge_uses_distance_before_semantic_label();
    test_small_shell_rejects_invalid_mode();
    if (failures != 0) return 1;
    printf("mesh semantic-to-instance partitioning passed\n");
    return 0;
}
