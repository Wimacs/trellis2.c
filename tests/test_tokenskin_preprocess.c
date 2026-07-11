#include "gltf_io.h"
#include "preprocess.h"
#include "trellis_platform.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;

#define CHECK_TRUE(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        ++g_failures; \
        return 0; \
    } \
} while (0)

static int close_float(float actual, float expected, float tolerance) {
    return isfinite(actual) && fabsf(actual - expected) <= tolerance;
}

static int write_all(FILE * file, const void * data, size_t size) {
    return size == 0 || fwrite(data, 1, size, file) == size;
}

static int write_u32_le(FILE * file, uint32_t value) {
    const unsigned char bytes[4] = {
        (unsigned char) (value & 0xffu),
        (unsigned char) ((value >> 8u) & 0xffu),
        (unsigned char) ((value >> 16u) & 0xffu),
        (unsigned char) ((value >> 24u) & 0xffu),
    };
    return write_all(file, bytes, sizeof(bytes));
}

static int write_positions(const char * path) {
    static const float positions[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 1.0f,
    };
    FILE * file = fopen(path, "wb");
    if (file == NULL) return 0;
    const int ok = write_all(file, positions, sizeof(positions)) && fclose(file) == 0;
    return ok;
}

static int format_document(char * output, size_t output_size, const char * buffer_uri) {
    const char * buffer_format = buffer_uri != NULL ?
        "{\"byteLength\":72,\"uri\":\"%s\"}" :
        "{\"byteLength\":72}";
    char buffer[512];
    const int buffer_length = buffer_uri != NULL ?
        snprintf(buffer, sizeof(buffer), buffer_format, buffer_uri) :
        snprintf(buffer, sizeof(buffer), "%s", buffer_format);
    if (buffer_length < 0 || (size_t) buffer_length >= sizeof(buffer)) return 0;

    const int length = snprintf(
        output,
        output_size,
        "{"
        "\"asset\":{\"version\":\"2.0\"},"
        "\"scene\":0,"
        "\"scenes\":[{\"nodes\":[0,1]}],"
        "\"nodes\":["
          "{\"mesh\":0,\"translation\":[10,0,0]},"
          "{\"mesh\":0,\"translation\":[0,20,0],\"scale\":[2,2,2]}"
        "],"
        "\"meshes\":[{\"primitives\":["
          "{\"attributes\":{\"POSITION\":0}},"
          "{\"attributes\":{\"POSITION\":1}}"
        "]}],"
        "\"buffers\":[%s],"
        "\"bufferViews\":["
          "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36},"
          "{\"buffer\":0,\"byteOffset\":36,\"byteLength\":36}"
        "],"
        "\"accessors\":["
          "{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\","
            "\"min\":[0,0,0],\"max\":[1,1,0]},"
          "{\"bufferView\":1,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\","
            "\"min\":[0,0,1],\"max\":[1,1,1]}"
        "]"
        "}",
        buffer);
    return length > 0 && (size_t) length < output_size;
}

static int write_synthetic_gltf(const char * gltf_path, const char * bin_path) {
    if (!write_positions(bin_path)) return 0;
    const char * base = trellis_path_last_sep_const(bin_path);
    base = base != NULL ? base + 1 : bin_path;
    char document[4096];
    if (!format_document(document, sizeof(document), base)) return 0;
    FILE * file = fopen(gltf_path, "wb");
    if (file == NULL) return 0;
    const int ok = write_all(file, document, strlen(document)) && fclose(file) == 0;
    return ok;
}

static int write_synthetic_glb(const char * glb_path, const char * bin_path) {
    FILE * bin = fopen(bin_path, "rb");
    if (bin == NULL) return 0;
    unsigned char binary[72];
    const int read_ok = fread(binary, 1, sizeof(binary), bin) == sizeof(binary);
    fclose(bin);
    if (!read_ok) return 0;

    char document[4096];
    if (!format_document(document, sizeof(document), NULL)) return 0;
    const size_t json_length = strlen(document);
    const size_t padded_json_length = (json_length + 3u) & ~(size_t) 3u;
    const uint32_t total_length = (uint32_t) (
        12u + 8u + padded_json_length + 8u + sizeof(binary));
    FILE * file = fopen(glb_path, "wb");
    if (file == NULL) return 0;
    int ok =
        write_u32_le(file, 0x46546c67u) &&
        write_u32_le(file, 2u) &&
        write_u32_le(file, total_length) &&
        write_u32_le(file, (uint32_t) padded_json_length) &&
        write_u32_le(file, 0x4e4f534au) &&
        write_all(file, document, json_length);
    for (size_t i = json_length; i < padded_json_length && ok; ++i) {
        const unsigned char space = 0x20u;
        ok = write_all(file, &space, 1u);
    }
    ok = ok &&
        write_u32_le(file, (uint32_t) sizeof(binary)) &&
        write_u32_le(file, 0x004e4942u) &&
        write_all(file, binary, sizeof(binary));
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int check_asset(const trellis_mesh_rigging_asset * asset) {
    CHECK_TRUE(asset->vertex_count == 12u);
    CHECK_TRUE(asset->triangle_count == 4u);
    CHECK_TRUE(asset->primitive_count == 4u);
    CHECK_TRUE(close_float(asset->aabb_min[0], 0.0f, 1e-6f));
    CHECK_TRUE(close_float(asset->aabb_min[1], 0.0f, 1e-6f));
    CHECK_TRUE(close_float(asset->aabb_min[2], 0.0f, 1e-6f));
    CHECK_TRUE(close_float(asset->aabb_max[0], 11.0f, 1e-6f));
    CHECK_TRUE(close_float(asset->aabb_max[1], 22.0f, 1e-6f));
    CHECK_TRUE(close_float(asset->aabb_max[2], 2.0f, 1e-6f));
    CHECK_TRUE(asset->primitives[0].source_node_index == 0u);
    CHECK_TRUE(asset->primitives[0].source_primitive_index == 0u);
    CHECK_TRUE(asset->primitives[1].source_primitive_index == 1u);
    CHECK_TRUE(asset->primitives[2].source_node_index == 1u);
    CHECK_TRUE(asset->primitives[3].first_vertex == 9u);
    CHECK_TRUE(asset->primitives[3].first_triangle == 3u);
    for (size_t i = 0; i < asset->vertex_count; ++i) {
        const float * normal = asset->normals + i * 3u;
        CHECK_TRUE(close_float(normal[0], 0.0f, 1e-6f));
        CHECK_TRUE(close_float(normal[1], 0.0f, 1e-6f));
        CHECK_TRUE(close_float(normal[2], 1.0f, 1e-6f));
    }
    return 1;
}

static int indices_are_unique(const uint32_t * indices, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = i + 1u; j < count; ++j) {
            if (indices[i] == indices[j]) return 0;
        }
    }
    return 1;
}

static int check_preprocessed(
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_rigging_preprocessed * preprocessed) {
    CHECK_TRUE(preprocessed->vertex_count == asset->vertex_count);
    CHECK_TRUE(preprocessed->sample_count == 2048u);
    CHECK_TRUE(preprocessed->mesh_candidate_count == 2048u);
    CHECK_TRUE(preprocessed->mesh_fps_count == 512u);
    CHECK_TRUE(preprocessed->skin_candidate_count == 1536u);
    CHECK_TRUE(preprocessed->skin_fps_count == 384u);
    CHECK_TRUE(close_float(preprocessed->normalization.center[0], 5.5f, 1e-6f));
    CHECK_TRUE(close_float(preprocessed->normalization.center[1], 11.0f, 1e-6f));
    CHECK_TRUE(close_float(preprocessed->normalization.center[2], 1.0f, 1e-6f));
    CHECK_TRUE(close_float(preprocessed->normalization.scale, 1.0f / 11.0f, 1e-7f));
    CHECK_TRUE(close_float(preprocessed->normalization.inverse_scale, 11.0f, 1e-6f));
    CHECK_TRUE(close_float(preprocessed->normalization.normalized_from_world[13], -1.0f, 1e-6f));
    CHECK_TRUE(close_float(preprocessed->normalization.world_from_normalized[13], 11.0f, 1e-6f));

    for (size_t vertex = 0; vertex < asset->vertex_count; ++vertex) {
        float world[3];
        trellis_mesh_rigging_denormalize_point(
            &preprocessed->normalization,
            preprocessed->normalized_vertices + vertex * 3u,
            world);
        for (int axis = 0; axis < 3; ++axis) {
            CHECK_TRUE(close_float(world[axis], asset->positions[vertex * 3u + (size_t) axis], 2e-5f));
            CHECK_TRUE(fabsf(preprocessed->normalized_vertices[vertex * 3u + (size_t) axis]) <= 1.00001f);
        }
    }

    size_t large_instance_samples = 0;
    for (size_t sample = 0; sample < preprocessed->sample_count; ++sample) {
        CHECK_TRUE(preprocessed->sample_triangles[sample] < asset->triangle_count);
        if (preprocessed->sample_triangles[sample] >= 2u) ++large_instance_samples;
        const float * barycentric = preprocessed->sample_barycentric + sample * 3u;
        CHECK_TRUE(barycentric[0] >= 0.0f && barycentric[1] >= 0.0f && barycentric[2] >= 0.0f);
        CHECK_TRUE(close_float(barycentric[0] + barycentric[1] + barycentric[2], 1.0f, 2e-6f));
        const float * normal = preprocessed->sample_normals + sample * 3u;
        CHECK_TRUE(close_float(normal[0], 0.0f, 1e-6f));
        CHECK_TRUE(close_float(normal[1], 0.0f, 1e-6f));
        CHECK_TRUE(close_float(normal[2], 1.0f, 1e-6f));
    }
    /* Node 1 uses scale 2, so its two triangles contribute four times the area
     * of node 0's two triangles. */
    const double large_ratio = (double) large_instance_samples / (double) preprocessed->sample_count;
    CHECK_TRUE(large_ratio > 0.75 && large_ratio < 0.85);

    for (size_t i = 0; i < preprocessed->mesh_candidate_count; ++i) {
        CHECK_TRUE(preprocessed->mesh_candidate_indices[i] < preprocessed->sample_count);
    }
    for (size_t i = 0; i < preprocessed->skin_candidate_count; ++i) {
        CHECK_TRUE(preprocessed->skin_candidate_indices[i] < preprocessed->sample_count);
    }
    for (size_t i = 0; i < preprocessed->mesh_fps_count; ++i) {
        CHECK_TRUE(preprocessed->mesh_fps_indices[i] < preprocessed->sample_count);
    }
    for (size_t i = 0; i < preprocessed->skin_fps_count; ++i) {
        CHECK_TRUE(preprocessed->skin_fps_indices[i] < preprocessed->sample_count);
    }
    CHECK_TRUE(indices_are_unique(preprocessed->mesh_candidate_indices, preprocessed->mesh_candidate_count));
    CHECK_TRUE(indices_are_unique(preprocessed->skin_candidate_indices, preprocessed->skin_candidate_count));
    CHECK_TRUE(indices_are_unique(preprocessed->mesh_fps_indices, preprocessed->mesh_fps_count));
    CHECK_TRUE(indices_are_unique(preprocessed->skin_fps_indices, preprocessed->skin_fps_count));
    return 1;
}

static int test_synthetic(const char * gltf_path, const char * glb_path) {
    char error[512];
    trellis_mesh_rigging_asset gltf_asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    trellis_mesh_rigging_asset glb_asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    CHECK_TRUE(trellis_mesh_rigging_gltf_load(
        gltf_path, &gltf_asset, error, sizeof(error)) == TRELLIS_STATUS_OK);
    CHECK_TRUE(check_asset(&gltf_asset));
    CHECK_TRUE(trellis_mesh_rigging_gltf_load(
        glb_path, &glb_asset, error, sizeof(error)) == TRELLIS_STATUS_OK);
    CHECK_TRUE(check_asset(&glb_asset));
    CHECK_TRUE(memcmp(
        gltf_asset.positions,
        glb_asset.positions,
        gltf_asset.vertex_count * 3u * sizeof(float)) == 0);

    trellis_mesh_rigging_preprocess_options options =
        TRELLIS_MESH_RIGGING_PREPROCESS_OPTIONS_INIT;
    options.surface_sample_count = 2047u;
    options.seed = UINT64_C(0x123456789abcdef0);
    trellis_mesh_rigging_preprocessed first = TRELLIS_MESH_RIGGING_PREPROCESSED_INIT;
    trellis_mesh_rigging_preprocessed second = TRELLIS_MESH_RIGGING_PREPROCESSED_INIT;
    trellis_mesh_rigging_preprocessed different = TRELLIS_MESH_RIGGING_PREPROCESSED_INIT;
    CHECK_TRUE(trellis_mesh_rigging_preprocess(
        &glb_asset, &options, &first, error, sizeof(error)) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    options.surface_sample_count = 2048u;
    uint32_t * invalid_triangles = (uint32_t *) malloc(
        glb_asset.triangle_count * 3u * sizeof(uint32_t));
    CHECK_TRUE(invalid_triangles != NULL);
    memcpy(
        invalid_triangles,
        glb_asset.triangles,
        glb_asset.triangle_count * 3u * sizeof(uint32_t));
    invalid_triangles[0] = (uint32_t) glb_asset.vertex_count;
    trellis_mesh_rigging_asset invalid_asset = glb_asset;
    invalid_asset.triangles = invalid_triangles;
    CHECK_TRUE(trellis_mesh_rigging_preprocess(
        &invalid_asset, &options, &first, error, sizeof(error)) ==
        TRELLIS_STATUS_INVALID_ARGUMENT);
    free(invalid_triangles);
    invalid_triangles = NULL;
    CHECK_TRUE(trellis_mesh_rigging_preprocess(
        &glb_asset, &options, &first, error, sizeof(error)) == TRELLIS_STATUS_OK);
    CHECK_TRUE(check_preprocessed(&glb_asset, &first));
    CHECK_TRUE(trellis_mesh_rigging_preprocess(
        &glb_asset, &options, &second, error, sizeof(error)) == TRELLIS_STATUS_OK);
    CHECK_TRUE(memcmp(first.sample_positions, second.sample_positions,
        first.sample_count * 3u * sizeof(float)) == 0);
    CHECK_TRUE(memcmp(first.mesh_candidate_indices, second.mesh_candidate_indices,
        first.mesh_candidate_count * sizeof(uint32_t)) == 0);
    CHECK_TRUE(memcmp(first.mesh_fps_indices, second.mesh_fps_indices,
        first.mesh_fps_count * sizeof(uint32_t)) == 0);
    CHECK_TRUE(memcmp(first.skin_fps_indices, second.skin_fps_indices,
        first.skin_fps_count * sizeof(uint32_t)) == 0);

    ++options.seed;
    CHECK_TRUE(trellis_mesh_rigging_preprocess(
        &glb_asset, &options, &different, error, sizeof(error)) == TRELLIS_STATUS_OK);
    CHECK_TRUE(memcmp(first.sample_positions, different.sample_positions,
        first.sample_count * 3u * sizeof(float)) != 0);

    trellis_mesh_rigging_preprocessed_free(&different);
    trellis_mesh_rigging_preprocessed_free(&second);
    trellis_mesh_rigging_preprocessed_free(&first);
    trellis_mesh_rigging_asset_free(&glb_asset);
    trellis_mesh_rigging_asset_free(&gltf_asset);
    return 1;
}

static int test_real_glb(const char * path) {
    char error[512];
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    const trellis_status status = trellis_mesh_rigging_gltf_load(
        path, &asset, error, sizeof(error));
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "failed to load real GLB %s: %s\n", path, error);
        ++g_failures;
        return 0;
    }
    CHECK_TRUE(asset.vertex_count > 1000u);
    CHECK_TRUE(asset.triangle_count > 1000u);
    CHECK_TRUE(asset.primitive_count >= 1u);
    trellis_mesh_rigging_preprocess_options options =
        TRELLIS_MESH_RIGGING_PREPROCESS_OPTIONS_INIT;
    options.surface_sample_count = 2048u;
    options.seed = 7u;
    trellis_mesh_rigging_preprocessed preprocessed = TRELLIS_MESH_RIGGING_PREPROCESSED_INIT;
    if (trellis_mesh_rigging_preprocess(
            &asset, &options, &preprocessed, error, sizeof(error)) != TRELLIS_STATUS_OK) {
        fprintf(stderr, "failed to preprocess real GLB %s: %s\n", path, error);
        trellis_mesh_rigging_asset_free(&asset);
        ++g_failures;
        return 0;
    }
    CHECK_TRUE(preprocessed.mesh_fps_count == 512u);
    CHECK_TRUE(preprocessed.skin_fps_count == 384u);
    printf(
        "real GLB preprocessing passed: %s (%zu vertices, %zu triangles, %zu primitive instances)\n",
        path,
        asset.vertex_count,
        asset.triangle_count,
        asset.primitive_count);
    trellis_mesh_rigging_preprocessed_free(&preprocessed);
    trellis_mesh_rigging_asset_free(&asset);
    return 1;
}

int main(int argc, char ** argv) {
    char base[PATH_MAX];
    char gltf_path[PATH_MAX];
    char glb_path[PATH_MAX];
    char bin_path[PATH_MAX];
    if (!trellis_make_temp_path(base, sizeof(base), "tokenskin_preprocess", NULL) ||
        snprintf(gltf_path, sizeof(gltf_path), "%s.gltf", base) < 0 ||
        snprintf(glb_path, sizeof(glb_path), "%s.glb", base) < 0 ||
        snprintf(bin_path, sizeof(bin_path), "%s.bin", base) < 0 ||
        !write_synthetic_gltf(gltf_path, bin_path) ||
        !write_synthetic_glb(glb_path, bin_path)) {
        fprintf(stderr, "failed to create synthetic glTF fixtures\n");
        return 1;
    }

    (void) test_synthetic(gltf_path, glb_path);
    if (argc > 1) {
        (void) test_real_glb(argv[1]);
    }

    trellis_unlink(glb_path);
    trellis_unlink(gltf_path);
    trellis_unlink(bin_path);
    if (g_failures != 0) {
        fprintf(stderr, "%d TokenSkin preprocessing test(s) failed\n", g_failures);
        return 1;
    }
    puts("TokenSkin preprocessing tests passed");
    return 0;
}
