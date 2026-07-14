#include "parts_glb.h"
#include "trellis_platform.h"

#include "cgltf.h"

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
        goto cleanup; \
    } \
} while (0)

static int close_float(float a, float b) {
    return isfinite(a) && fabsf(a - b) <= 1e-6f;
}

static int write_bytes(const char * path, const void * bytes, size_t size) {
    FILE * file = fopen(path, "wb");
    if (file == NULL) return 0;
    const int wrote = fwrite(bytes, 1, size, file) == size;
    return fclose(file) == 0 && wrote;
}

static int file_equals(const char * path, const void * expected, size_t size) {
    FILE * file = fopen(path, "rb");
    if (file == NULL) return 0;
    unsigned char buffer[64];
    const int ok = size <= sizeof(buffer) &&
        fread(buffer, 1, size, file) == size &&
        fgetc(file) == EOF &&
        memcmp(buffer, expected, size) == 0;
    fclose(file);
    return ok;
}

static int run_parts_round_trip(void) {
    static float positions[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        2.0f, 0.0f, 0.0f,
        2.0f, 1.0f, 0.0f,
    };
    static uint32_t triangles[] = {
        0u, 1u, 2u,
        1u, 3u, 2u,
        1u, 4u, 3u,
        4u, 5u, 3u,
    };
    static const uint32_t parts[] = {0u, 0u, 1u, 1u};
    static const char * names[] = {"seat", "back"};
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    asset.positions = positions;
    asset.vertex_count = 6;
    asset.triangles = triangles;
    asset.triangle_count = 4;
    char path[4096] = {0};
    char error[512] = {0};
    cgltf_data * data = NULL;
    int ok = 0;

    CHECK_TRUE(trellis_make_temp_path(path, sizeof(path), "trellis_parts", ".glb"));
    CHECK_TRUE(
        trellis_mesh_segmentation_write_parts_glb(
            path, &asset, parts, 4, 2, names, error, sizeof(error)) ==
        TRELLIS_STATUS_OK);

    cgltf_options options;
    memset(&options, 0, sizeof(options));
    CHECK_TRUE(cgltf_parse_file(&options, path, &data) == cgltf_result_success);
    CHECK_TRUE(cgltf_load_buffers(&options, data, path) == cgltf_result_success);
    CHECK_TRUE(cgltf_validate(data) == cgltf_result_success);
    CHECK_TRUE(data->scene != NULL && data->scene->nodes_count == 1);
    const cgltf_node * root = data->scene->nodes[0];
    CHECK_TRUE(root != NULL && root->children_count == 2);
    CHECK_TRUE(data->meshes_count == 2 && data->materials_count == 2);
    CHECK_TRUE(strcmp(root->children[0]->name, "seat") == 0);
    CHECK_TRUE(strcmp(root->children[1]->name, "back") == 0);
    CHECK_TRUE(root->children[0]->extras.data != NULL);
    CHECK_TRUE(strstr(root->children[0]->extras.data, "trellis_part_id") != NULL);

    size_t total_faces = 0;
    size_t total_vertices = 0;
    for (size_t part = 0; part < 2; ++part) {
        const cgltf_node * node = root->children[part];
        CHECK_TRUE(node != NULL && node->mesh != NULL);
        CHECK_TRUE(node->mesh->primitives_count == 1);
        const cgltf_primitive * primitive = &node->mesh->primitives[0];
        CHECK_TRUE(primitive->indices != NULL && primitive->indices->count == 6);
        total_faces += (size_t) primitive->indices->count / 3u;
        const cgltf_accessor * position =
            cgltf_find_accessor(primitive, cgltf_attribute_type_position, 0);
        CHECK_TRUE(position != NULL && position->count == 4);
        CHECK_TRUE(position->has_min && position->has_max);
        CHECK_TRUE(close_float(position->min[0], part == 0 ? 0.0f : 1.0f));
        CHECK_TRUE(close_float(position->max[0], part == 0 ? 1.0f : 2.0f));
        total_vertices += (size_t) position->count;
        for (cgltf_size index = 0; index < primitive->indices->count; ++index) {
            CHECK_TRUE(cgltf_accessor_read_index(primitive->indices, index) < position->count);
        }
    }
    CHECK_TRUE(total_faces == 4);
    CHECK_TRUE(total_vertices == 8); /* shared seam vertices were duplicated */
    CHECK_TRUE(
        data->materials[0].pbr_metallic_roughness.base_color_factor[0] !=
        data->materials[1].pbr_metallic_roughness.base_color_factor[0]);
    ok = 1;

cleanup:
    if (data != NULL) cgltf_free(data);
    if (path[0] != '\0') remove(path);
    return ok;
}

static int run_validation_checks(void) {
    static float positions[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    static uint32_t triangles[] = {0u, 1u, 2u};
    static const uint32_t invalid_part[] = {2u};
    static const uint32_t part_zero[] = {0u};
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    asset.positions = positions;
    asset.vertex_count = 3;
    asset.triangles = triangles;
    asset.triangle_count = 1;
    char path[4096] = {0};
    char error[256] = {0};
    int ok = 0;
    CHECK_TRUE(trellis_make_temp_path(path, sizeof(path), "trellis_parts_invalid", ".glb"));
    CHECK_TRUE(
        trellis_mesh_segmentation_write_parts_glb(
            path, &asset, part_zero, 0, 1, NULL, error, sizeof(error)) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(
        trellis_mesh_segmentation_write_parts_glb(
            path, &asset, invalid_part, 1, 2, NULL, error, sizeof(error)) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(
        trellis_mesh_segmentation_write_parts_glb(
            "not-a-glb.gltf", &asset, part_zero, 1, 1, NULL, error, sizeof(error)) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    ok = 1;
cleanup:
    if (path[0] != '\0') remove(path);
    return ok;
}

static int run_atomic_output_checks(void) {
    static float positions[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    static uint32_t triangles[] = {0u, 1u, 2u};
    static uint32_t invalid_triangles[] = {0u, 1u, 99u};
    static const uint32_t parts[] = {0u};
    static const unsigned char sentinel[] = "existing output must survive";
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    asset.positions = positions;
    asset.vertex_count = 3;
    asset.triangles = triangles;
    asset.triangle_count = 1;
    char path[4096] = {0};
    char directory_path[4096] = {0};
    char temporary_path[4176] = {0};
    char error[256] = {0};
    int ok = 0;

    CHECK_TRUE(trellis_make_temp_path(path, sizeof(path), "trellis_parts_atomic", ".glb"));
    CHECK_TRUE(write_bytes(path, sentinel, sizeof(sentinel)));

    asset.triangles = invalid_triangles;
    CHECK_TRUE(
        trellis_mesh_segmentation_write_parts_glb(
            path, &asset, parts, 1, 1, NULL, error, sizeof(error)) ==
        TRELLIS_STATUS_PARSE_ERROR);
    CHECK_TRUE(file_equals(path, sentinel, sizeof(sentinel)));

    asset.triangles = triangles;
    CHECK_TRUE(
        trellis_mesh_segmentation_write_parts_glb(
            path, &asset, parts, 1, 1, NULL, error, sizeof(error)) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(!file_equals(path, sentinel, sizeof(sentinel)));

    CHECK_TRUE(trellis_make_temp_path(
        directory_path, sizeof(directory_path), "trellis_parts_replace_failure", ".glb"));
    CHECK_TRUE(trellis_mkdir_one(directory_path));
    CHECK_TRUE(
        trellis_mesh_segmentation_write_parts_glb(
            directory_path, &asset, parts, 1, 1, NULL, error, sizeof(error)) ==
        TRELLIS_STATUS_IO_ERROR);
    CHECK_TRUE(trellis_access_exists(directory_path));
    for (int attempt = 0; attempt < 100; ++attempt) {
        CHECK_TRUE(snprintf(
            temporary_path,
            sizeof(temporary_path),
            "%s.trellis-tmp-%ld-%d.glb",
            directory_path,
            trellis_getpid(),
            attempt) > 0);
        CHECK_TRUE(!trellis_access_exists(temporary_path));
    }
    ok = 1;

cleanup:
    if (path[0] != '\0') trellis_unlink(path);
    if (directory_path[0] != '\0') {
#ifdef _WIN32
        _rmdir(directory_path);
#else
        rmdir(directory_path);
#endif
    }
    return ok;
}

int main(void) {
    run_parts_round_trip();
    run_validation_checks();
    run_atomic_output_checks();
    if (failures != 0) return 1;
    printf("mesh parts GLB round-trip passed\n");
    return 0;
}
