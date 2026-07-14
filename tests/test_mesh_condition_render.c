#include "tasks/mesh_segmentation/condition_render.h"

#include "trellis_platform.h"

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

static size_t count_opaque_pixels(
    const trellis_mesh_segmentation_condition_image * image) {
    size_t opaque = 0;
    const size_t pixels = (size_t) image->width * (size_t) image->height;
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
        if (image->rgba[pixel * 4u + 3u] != 0u) ++opaque;
    }
    return opaque;
}

static size_t count_partial_alpha_pixels(
    const trellis_mesh_segmentation_condition_image * image) {
    size_t partial = 0;
    const size_t pixels = (size_t) image->width * (size_t) image->height;
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
        const uint8_t alpha = image->rgba[pixel * 4u + 3u];
        if (alpha != 0u && alpha != 255u) ++partial;
    }
    return partial;
}

static void make_camera_facing_triangle(
    float positions[9],
    float normals[9]) {
    static const float right[3] = {
        0.8819212913513184f, -0.4713967740535736f, 0.0f,
    };
    static const float up[3] = {
        0.06494797021150589f, 0.12150891125202179f, 0.9904633164405823f,
    };
    static const float toward_camera[3] = {
        -0.46690112352371216f, -0.8735106587409973f, 0.13777753710746765f,
    };
    const float coordinates[3][2] = {
        { -0.45f, -0.40f },
        {  0.45f, -0.40f },
        {  0.00f,  0.45f },
    };
    for (int vertex = 0; vertex < 3; ++vertex) {
        float blender_position[3];
        for (int axis = 0; axis < 3; ++axis) {
            blender_position[axis] =
                coordinates[vertex][0] * right[axis] +
                coordinates[vertex][1] * up[axis];
        }
        /* Inverse of the renderer's glTF Y-up -> Blender Z-up mapping. */
        positions[vertex * 3] = blender_position[0];
        positions[vertex * 3 + 1] = blender_position[2];
        positions[vertex * 3 + 2] = -blender_position[1];
        normals[vertex * 3] = toward_camera[0];
        normals[vertex * 3 + 1] = toward_camera[2];
        normals[vertex * 3 + 2] = -toward_camera[1];
    }
}

static void test_triangle_is_visible_on_transparent_background(void) {
    float positions[9];
    float normals[9];
    make_camera_facing_triangle(positions, normals);
    uint32_t triangles[3] = { 0, 1, 2 };
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    asset.positions = positions;
    asset.normals = normals;
    asset.triangles = triangles;
    asset.vertex_count = 3;
    asset.triangle_count = 1;

    trellis_mesh_segmentation_condition_image image =
        TRELLIS_MESH_SEGMENTATION_CONDITION_IMAGE_INIT;
    CHECK_TRUE(
        trellis_mesh_segmentation_render_condition_rgba(&asset, 128, &image) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(image.width == 128 && image.height == 128);
    CHECK_TRUE(image.rgba != NULL);
    CHECK_TRUE(count_opaque_pixels(&image) > 1000u);
    CHECK_TRUE(count_partial_alpha_pixels(&image) > 0u);
    CHECK_TRUE(image.rgba[3] == 0u);
    CHECK_TRUE(image.rgba[((size_t) 127 * 128u + 127u) * 4u + 3u] == 0u);
    CHECK_TRUE(image.rgba[((size_t) 64 * 128u + 64u) * 4u + 3u] == 255u);
    trellis_mesh_segmentation_condition_image_free(&image);
    CHECK_TRUE(image.rgba == NULL && image.width == 0 && image.height == 0);
}

static void test_cube_is_deterministic_and_depth_shaded(void) {
    float positions[] = {
        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,
    };
    uint32_t triangles[] = {
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        3, 7, 6, 3, 6, 2,
        0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
    };
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    asset.positions = positions;
    asset.triangles = triangles;
    asset.vertex_count = 8;
    asset.triangle_count = 12;

    trellis_mesh_segmentation_condition_image first =
        TRELLIS_MESH_SEGMENTATION_CONDITION_IMAGE_INIT;
    trellis_mesh_segmentation_condition_image second =
        TRELLIS_MESH_SEGMENTATION_CONDITION_IMAGE_INIT;
    CHECK_TRUE(
        trellis_mesh_segmentation_render_condition_rgba(&asset, 160, &first) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(
        trellis_mesh_segmentation_render_condition_rgba(&asset, 160, &second) ==
        TRELLIS_STATUS_OK);
    const size_t byte_count = 160u * 160u * 4u;
    CHECK_TRUE(memcmp(first.rgba, second.rgba, byte_count) == 0);
    CHECK_TRUE(count_opaque_pixels(&first) > 4000u);

    uint32_t first_color = 0;
    int found_first = 0;
    int found_second = 0;
    for (size_t pixel = 0; pixel < 160u * 160u; ++pixel) {
        const uint8_t * value = first.rgba + pixel * 4u;
        if (value[3] == 0u) continue;
        const uint32_t color =
            (uint32_t) value[0] |
            ((uint32_t) value[1] << 8u) |
            ((uint32_t) value[2] << 16u);
        if (!found_first) {
            first_color = color;
            found_first = 1;
        } else if (color != first_color) {
            found_second = 1;
            break;
        }
    }
    CHECK_TRUE(found_first && found_second);
    trellis_mesh_segmentation_condition_image_free(&first);
    trellis_mesh_segmentation_condition_image_free(&second);
}

static void test_prepared_image_owns_temporary_png(void) {
    float positions[9];
    float normals[9];
    make_camera_facing_triangle(positions, normals);
    uint32_t triangles[3] = { 0, 1, 2 };
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    asset.positions = positions;
    asset.normals = normals;
    asset.triangles = triangles;
    asset.vertex_count = 3;
    asset.triangle_count = 1;

    trellis_prepared_condition_image prepared;
    memset(&prepared, 0, sizeof(prepared));
    CHECK_TRUE(
        trellis_mesh_segmentation_render_condition(
            &asset, 96, &prepared, NULL) == TRELLIS_STATUS_OK);
    CHECK_TRUE(prepared.source_path[0] != '\0');
    CHECK_TRUE(strcmp(prepared.source_path, prepared.converted_path) == 0);
    CHECK_TRUE(trellis_access_read(prepared.source_path));
    char temporary_path[4096];
    snprintf(temporary_path, sizeof(temporary_path), "%s", prepared.source_path);
    trellis_pipeline_prepared_condition_image_free(&prepared);
    CHECK_TRUE(!trellis_access_read(temporary_path));
}

int main(void) {
    test_triangle_is_visible_on_transparent_background();
    test_cube_is_deterministic_and_depth_shaded();
    test_prepared_image_owns_temporary_png();
    if (failures != 0) return 1;
    printf("mesh condition renderer passed\n");
    return 0;
}
