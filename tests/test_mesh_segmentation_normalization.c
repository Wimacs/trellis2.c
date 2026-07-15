#include "normalization.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK_TRUE(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
            return 1; \
        } \
    } while (0)

#define CHECK_CLOSE(actual, expected, tolerance) \
    CHECK_TRUE(fabsf((actual) - (expected)) <= (tolerance))

static void fill_asset(
    trellis_mesh_rigging_asset * asset,
    float positions[9],
    uint32_t triangles[3]) {
    memset(asset, 0, sizeof(*asset));
    asset->positions = positions;
    asset->triangles = triangles;
    asset->vertex_count = 3;
    asset->triangle_count = 1;
    triangles[0] = 0;
    triangles[1] = 1;
    triangles[2] = 2;
}

static trellis_shape_latent_cache_info decoder_anchor(void) {
    trellis_shape_latent_cache_info info;
    memset(&info, 0, sizeof(info));
    info.version = 1;
    info.resolution = 512;
    info.channels = 32;
    info.n_coords = 1;
    info.anchor_aabb_min[0] = -0.4f;
    info.anchor_aabb_min[1] = -0.5f;
    info.anchor_aabb_min[2] = -0.2f;
    info.anchor_aabb_max[0] = 0.4f;
    info.anchor_aabb_max[1] = 0.5f;
    info.anchor_aabb_max[2] = 0.2f;
    return info;
}

static int verify_mesh(
    const trellis_mesh_host * mesh,
    const float expected[9]) {
    CHECK_TRUE(mesh->n_vertices == 3);
    CHECK_TRUE(mesh->n_faces == 1);
    for (int index = 0; index < 9; ++index) {
        CHECK_CLOSE(mesh->vertices[index], expected[index], 1e-6f);
    }
    CHECK_TRUE(mesh->faces[0] == 0 && mesh->faces[1] == 1 && mesh->faces[2] == 2);
    return 0;
}

int main(void) {
    const float decoder_positions[9] = {
        -0.4f, -0.5f, -0.2f,
         0.4f, -0.5f, -0.2f,
         0.0f,  0.5f,  0.2f,
    };
    /* Forward TRELLIS export transform: (x,y,z) -> (x,z,-y). */
    float trellis_gltf_positions[9] = {
        -0.4f, -0.2f,  0.5f,
         0.4f, -0.2f,  0.5f,
         0.0f,  0.2f, -0.5f,
    };
    uint32_t triangles[3];
    trellis_mesh_rigging_asset asset;
    fill_asset(&asset, trellis_gltf_positions, triangles);
    trellis_shape_latent_cache_info anchor = decoder_anchor();
    trellis_mesh_host normalized;
    memset(&normalized, 0, sizeof(normalized));
    CHECK_TRUE(trellis_mesh_segmentation_normalize_asset_mesh(
        &asset, &anchor, &normalized) == TRELLIS_STATUS_OK);
    CHECK_TRUE(verify_mesh(&normalized, decoder_positions) == 0);
    trellis_mesh_free(&normalized);

    float direct_positions[9];
    memcpy(direct_positions, decoder_positions, sizeof(direct_positions));
    fill_asset(&asset, direct_positions, triangles);
    CHECK_TRUE(trellis_mesh_segmentation_normalize_asset_mesh(
        &asset, &anchor, &normalized) == TRELLIS_STATUS_OK);
    CHECK_TRUE(verify_mesh(&normalized, decoder_positions) == 0);
    trellis_mesh_free(&normalized);

    anchor.anchor_aabb_min[1] = -0.500209689f;
    anchor.anchor_aabb_max[1] = 0.499790311f;
    float boundary_positions[9];
    memcpy(boundary_positions, decoder_positions, sizeof(boundary_positions));
    fill_asset(&asset, boundary_positions, triangles);
    CHECK_TRUE(trellis_mesh_segmentation_normalize_asset_mesh(
        &asset, &anchor, &normalized) == TRELLIS_STATUS_OK);
    CHECK_TRUE(normalized.vertices[1] >=
        -trellis_shape_latent_decoder_aabb_limit(512));
    trellis_mesh_free(&normalized);

    puts("mesh segmentation normalization tests passed");
    return 0;
}
