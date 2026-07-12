#include "trellis.h"

#include <math.h>
#include <stdint.h>
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

static void check_close(float actual, float expected, float tolerance, int line) {
    if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
        fprintf(stderr,
            "CHECK_CLOSE failed at %s:%d: actual=%.9g expected=%.9g tolerance=%.9g\n",
            __FILE__, line, actual, expected, tolerance);
        ++g_failures;
    }
}

#define CHECK_CLOSE(actual, expected, tolerance) \
    check_close((actual), (expected), (tolerance), __LINE__)

static const float k_triangle_vertices[] = {
    0.2f, 0.2f, 0.5f,
    0.8f, 0.2f, 0.5f,
    0.2f, 0.8f, 0.5f,
};

static const int32_t k_triangle_faces[] = {0, 1, 2};

static trellis_flexible_dual_grid_options triangle_options(void) {
    trellis_flexible_dual_grid_options options;
    trellis_flexible_dual_grid_options_default(&options);
    for (int axis = 0; axis < 3; ++axis) {
        options.grid_size[axis] = 4;
        options.aabb_min[axis] = 0.0f;
        options.aabb_max[axis] = 1.0f;
    }
    return options;
}

static void test_defaults(void) {
    trellis_flexible_dual_grid_options options;
    memset(&options, 0xff, sizeof(options));
    trellis_flexible_dual_grid_options_default(&options);
    for (int axis = 0; axis < 3; ++axis) {
        CHECK_TRUE(options.grid_size[axis] == 512);
        CHECK_CLOSE(options.aabb_min[axis], -0.5f, 0.0f);
        CHECK_CLOSE(options.aabb_max[axis], 0.5f, 0.0f);
    }
    CHECK_CLOSE(options.face_weight, 1.0f, 0.0f);
    CHECK_CLOSE(options.boundary_weight, 0.2f, 0.0f);
    CHECK_CLOSE(options.regularization_weight, 0.01f, 0.0f);
}

static void test_triangle_matches_o_voxel_golden(void) {
    static const int32_t expected_xyz[][3] = {
        {0, 0, 2}, {0, 1, 2}, {1, 0, 2}, {1, 1, 2},
        {2, 0, 2}, {2, 1, 2}, {3, 0, 2}, {3, 1, 2},
        {0, 2, 2}, {1, 2, 2}, {2, 2, 2}, {0, 3, 2}, {1, 3, 2},
    };
    static const float expected_dual[][3] = {
        {0.20238103f, 0.20238097f, 0.50000000f},
        {0.20454526f, 0.37500003f, 0.50000000f},
        {0.37500262f, 0.20454547f, 0.50000000f},
        {0.37500262f, 0.37500000f, 0.50000000f},
        {0.62500238f, 0.20454547f, 0.50000000f},
        {0.61956668f, 0.36956388f, 0.50000000f},
        {0.79149330f, 0.20435703f, 0.50000000f},
        {0.75000000f, 0.25000030f, 0.50000000f},
        {0.20454526f, 0.62500000f, 0.50000000f},
        {0.36956334f, 0.61956656f, 0.50000000f},
        {0.50000000f, 0.50000000f, 0.50000000f},
        {0.20435698f, 0.79149371f, 0.50000000f},
        {0.25000000f, 0.75000000f, 0.50000000f},
    };
    static const uint8_t expected_intersected[][3] = {
        {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
        {0, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
        {0, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
    };

    trellis_flexible_dual_grid_options options = triangle_options();
    trellis_flexible_dual_grid grid = {0};
    CHECK_TRUE(trellis_mesh_to_flexible_dual_grid_host(
        k_triangle_vertices, 3, k_triangle_faces, 1, &options, &grid) == TRELLIS_STATUS_OK);
    CHECK_TRUE(grid.n == 13);
    CHECK_TRUE(grid.coords != NULL);
    CHECK_TRUE(grid.dual_vertices != NULL);
    CHECK_TRUE(grid.intersected != NULL);

    for (int64_t i = 0; i < grid.n; ++i) {
        CHECK_TRUE(grid.coords[i * 4 + 0] == 0);
        for (int axis = 0; axis < 3; ++axis) {
            CHECK_TRUE(grid.coords[i * 4 + 1 + axis] == expected_xyz[i][axis]);
            CHECK_CLOSE(grid.dual_vertices[i * 3 + axis], expected_dual[i][axis], 2e-5f);
            CHECK_TRUE(grid.intersected[i * 3 + axis] == expected_intersected[i][axis]);
        }
    }

    trellis_flexible_dual_grid second = {0};
    CHECK_TRUE(trellis_mesh_to_flexible_dual_grid_host(
        k_triangle_vertices, 3, k_triangle_faces, 1, &options, &second) == TRELLIS_STATUS_OK);
    CHECK_TRUE(second.n == grid.n);
    CHECK_TRUE(memcmp(second.coords, grid.coords, (size_t) grid.n * 4u * sizeof(int32_t)) == 0);
    CHECK_TRUE(memcmp(second.dual_vertices, grid.dual_vertices, (size_t) grid.n * 3u * sizeof(float)) == 0);
    CHECK_TRUE(memcmp(second.intersected, grid.intersected, (size_t) grid.n * 3u) == 0);

    trellis_flexible_dual_grid_free(&second);
    trellis_flexible_dual_grid_free(&grid);
    CHECK_TRUE(grid.coords == NULL && grid.dual_vertices == NULL && grid.intersected == NULL && grid.n == 0);
}

static void test_empty_and_invalid_inputs(void) {
    trellis_flexible_dual_grid_options options = triangle_options();
    trellis_flexible_dual_grid grid = {0};
    CHECK_TRUE(trellis_mesh_to_flexible_dual_grid_host(
        NULL, 0, NULL, 0, &options, &grid) == TRELLIS_STATUS_OK);
    CHECK_TRUE(grid.n == 0 && grid.coords == NULL && grid.dual_vertices == NULL && grid.intersected == NULL);

    const int32_t invalid_face[] = {0, 1, 3};
    CHECK_TRUE(trellis_mesh_to_flexible_dual_grid_host(
        k_triangle_vertices, 3, invalid_face, 1, &options, &grid) == TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(grid.n == 0 && grid.coords == NULL && grid.dual_vertices == NULL && grid.intersected == NULL);

    options.grid_size[1] = 0;
    CHECK_TRUE(trellis_mesh_to_flexible_dual_grid_host(
        k_triangle_vertices, 3, k_triangle_faces, 1, &options, &grid) == TRELLIS_STATUS_INVALID_ARGUMENT);
}

static void test_qef_weights_are_configurable(void) {
    trellis_flexible_dual_grid_options options = triangle_options();
    options.boundary_weight = 0.0f;
    trellis_flexible_dual_grid grid = {0};
    CHECK_TRUE(trellis_mesh_to_flexible_dual_grid_host(
        k_triangle_vertices, 3, k_triangle_faces, 1, &options, &grid) == TRELLIS_STATUS_OK);
    CHECK_TRUE(grid.n == 13);
    CHECK_CLOSE(grid.dual_vertices[0], 0.25f, 2e-5f);
    CHECK_CLOSE(grid.dual_vertices[1], 0.25f, 2e-5f);
    CHECK_CLOSE(grid.dual_vertices[2], 0.50f, 2e-5f);
    trellis_flexible_dual_grid_free(&grid);
}

int main(void) {
    test_defaults();
    test_triangle_matches_o_voxel_golden();
    test_empty_and_invalid_inputs();
    test_qef_weights_are_configurable();
    if (g_failures != 0) {
        fprintf(stderr, "%d mesh-to-Flexible-Dual-Grid test(s) failed\n", g_failures);
        return 1;
    }
    printf("mesh-to-Flexible-Dual-Grid tests passed\n");
    return 0;
}
