#include "material_features.h"
#include "condition_render.h"
#include "trellis_platform.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern unsigned char * stbi_write_png_to_mem(
    const unsigned char * pixels,
    int stride_bytes,
    int x,
    int y,
    int n,
    int * out_len);

static int failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
        goto cleanup; \
    } \
} while (0)

static int close_float(float actual, float expected) {
    return isfinite(actual) && fabsf(actual - expected) <= 1e-5f;
}

static void write_u32_le(unsigned char * destination, uint32_t value) {
    destination[0] = (unsigned char) value;
    destination[1] = (unsigned char) (value >> 8u);
    destination[2] = (unsigned char) (value >> 16u);
    destination[3] = (unsigned char) (value >> 24u);
}

static int write_fixture_glb(
    const char * path,
    const float positions[18],
    const float texcoords[12]) {
    static const uint16_t indices[6] = { 0u, 1u, 2u, 0u, 1u, 2u };
    static const unsigned char pixels[16] = {
        128u, 64u, 192u, 128u,
        255u, 0u, 0u, 255u,
        0u, 255u, 0u, 255u,
        0u, 0u, 255u, 255u,
    };
    int png_length = 0;
    unsigned char * png = stbi_write_png_to_mem(pixels, 2 * 4, 2, 2, 4, &png_length);
    if (png == NULL || png_length <= 0) {
        free(png);
        return 0;
    }

    const size_t png_offset = sizeof(float) * 30u + sizeof(indices);
    const size_t bin_size = png_offset + (size_t) png_length;
    const size_t bin_padded = (bin_size + 3u) & ~(size_t) 3u;
    unsigned char * bin = (unsigned char *) calloc(bin_padded, 1u);
    if (bin == NULL) {
        free(png);
        return 0;
    }
    memcpy(bin, positions, sizeof(float) * 18u);
    memcpy(bin + sizeof(float) * 18u, texcoords, sizeof(float) * 12u);
    memcpy(bin + sizeof(float) * 30u, indices, sizeof(indices));
    memcpy(bin + png_offset, png, (size_t) png_length);
    free(png);

    char json[8192];
    const int json_length_raw = snprintf(
        json,
        sizeof(json),
        "{"
        "\"asset\":{\"version\":\"2.0\"},"
        "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
        "\"nodes\":[{\"mesh\":0}],"
        "\"buffers\":[{\"byteLength\":%zu}],"
        "\"bufferViews\":["
          "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":72,\"byteStride\":12,\"target\":34962},"
          "{\"buffer\":0,\"byteOffset\":72,\"byteLength\":48,\"byteStride\":8,\"target\":34962},"
          "{\"buffer\":0,\"byteOffset\":120,\"byteLength\":12,\"target\":34963},"
          "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%d}"
        "],"
        "\"accessors\":["
          "{\"bufferView\":0,\"byteOffset\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
          "{\"bufferView\":0,\"byteOffset\":36,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
          "{\"bufferView\":1,\"byteOffset\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC2\"},"
          "{\"bufferView\":1,\"byteOffset\":24,\"componentType\":5126,\"count\":3,\"type\":\"VEC2\"},"
          "{\"bufferView\":2,\"byteOffset\":0,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"},"
          "{\"bufferView\":2,\"byteOffset\":6,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"}"
        "],"
        "\"samplers\":[{\"magFilter\":9728,\"minFilter\":9728,\"wrapS\":33648,\"wrapT\":10497}],"
        "\"images\":[{\"bufferView\":3,\"mimeType\":\"image/png\"}],"
        "\"textures\":[{\"sampler\":0,\"source\":0}],"
        "\"materials\":["
          "{\"alphaMode\":\"MASK\",\"alphaCutoff\":0.5,\"pbrMetallicRoughness\":{"
            "\"baseColorFactor\":[0.25,0.5,0.75,0.4],"
            "\"metallicFactor\":0.2,\"roughnessFactor\":0.6}},"
          "{\"alphaMode\":\"BLEND\",\"pbrMetallicRoughness\":{"
            "\"baseColorFactor\":[0.5,0.25,1.0,0.5],"
            "\"baseColorTexture\":{\"index\":0},"
            "\"metallicFactor\":0.8,\"roughnessFactor\":0.4,"
            "\"metallicRoughnessTexture\":{\"index\":0}}}"
        "],"
        "\"meshes\":[{\"primitives\":["
          "{\"attributes\":{\"POSITION\":0,\"TEXCOORD_0\":2},\"indices\":4,\"material\":0},"
          "{\"attributes\":{\"POSITION\":1,\"TEXCOORD_0\":3},\"indices\":5,\"material\":1}"
        "]}]"
        "}",
        bin_size,
        png_offset,
        png_length);
    if (json_length_raw <= 0 || (size_t) json_length_raw >= sizeof(json)) {
        free(bin);
        return 0;
    }
    const size_t json_length = (size_t) json_length_raw;
    const size_t json_padded = (json_length + 3u) & ~(size_t) 3u;
    const size_t total_size = 12u + 8u + json_padded + 8u + bin_padded;
    if (total_size > UINT32_MAX) {
        free(bin);
        return 0;
    }
    unsigned char * glb = (unsigned char *) malloc(total_size);
    if (glb == NULL) {
        free(bin);
        return 0;
    }
    write_u32_le(glb + 0u, UINT32_C(0x46546c67));
    write_u32_le(glb + 4u, 2u);
    write_u32_le(glb + 8u, (uint32_t) total_size);
    write_u32_le(glb + 12u, (uint32_t) json_padded);
    write_u32_le(glb + 16u, UINT32_C(0x4e4f534a));
    memcpy(glb + 20u, json, json_length);
    memset(glb + 20u + json_length, ' ', json_padded - json_length);
    const size_t bin_header = 20u + json_padded;
    write_u32_le(glb + bin_header, (uint32_t) bin_padded);
    write_u32_le(glb + bin_header + 4u, UINT32_C(0x004e4942));
    memcpy(glb + bin_header + 8u, bin, bin_padded);
    free(bin);

    FILE * file = fopen(path, "wb");
    if (file == NULL) {
        free(glb);
        return 0;
    }
    const int wrote = fwrite(glb, 1u, total_size, file) == total_size;
    const int closed = fclose(file) == 0;
    const int ok = wrote && closed;
    free(glb);
    return ok;
}

static void test_factor_texture_wrap_and_face_mapping(void) {
    static float positions[18] = {
        -0.45f, -0.10f, 0.0f,
        -0.25f, -0.10f, 0.0f,
        -0.45f,  0.10f, 0.0f,
         0.25f, -0.10f, 0.0f,
         0.45f, -0.10f, 0.0f,
         0.25f,  0.10f, 0.0f,
    };
    static float texcoords[12] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        1.75f, 2.25f, 1.75f, 2.25f, 1.75f, 2.25f,
    };
    static int32_t coords[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    static float dual_vertices[6];
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    trellis_mesh_host mesh;
    trellis_flexible_dual_grid grid;
    float * features = NULL;
    float * repeated = NULL;
    trellis_mesh_segmentation_material_sampler sampler =
        TRELLIS_MESH_SEGMENTATION_MATERIAL_SAMPLER_INIT;
    trellis_mesh_segmentation_condition_image condition =
        TRELLIS_MESH_SEGMENTATION_CONDITION_IMAGE_INIT;
    char path[4096] = {0};
    char error[512] = {0};

    memset(&mesh, 0, sizeof(mesh));
    memset(&grid, 0, sizeof(grid));
    grid.coords = coords;
    grid.dual_vertices = dual_vertices;
    grid.n = 2;
    for (int row = 0; row < 2; ++row) {
        for (int axis = 0; axis < 3; ++axis) {
            const int base = row * 3;
            dual_vertices[base + axis] =
                (positions[base * 3 + axis] +
                 positions[(base + 1) * 3 + axis] +
                 positions[(base + 2) * 3 + axis]) / 3.0f + 0.5f;
        }
    }

    CHECK_TRUE(trellis_make_temp_path(path, sizeof(path), "trellis_material_features", ".glb"));
    CHECK_TRUE(write_fixture_glb(path, positions, texcoords));
    CHECK_TRUE(
        trellis_mesh_rigging_gltf_load(
            path, &asset, error, sizeof(error)) == TRELLIS_STATUS_OK);
    CHECK_TRUE(asset.vertex_count == 6u && asset.triangle_count == 2u);
    CHECK_TRUE(asset.primitive_count == 2u && asset.texcoord_set_count == 1u);
    mesh.vertices = asset.positions;
    mesh.faces = (int32_t *) asset.triangles;
    mesh.n_vertices = (int64_t) asset.vertex_count;
    mesh.n_faces = (int64_t) asset.triangle_count;
    CHECK_TRUE(
        trellis_mesh_segmentation_material_features(
            path, &asset, &mesh, &grid, &features) == TRELLIS_STATUS_OK);
    CHECK_TRUE(features != NULL);
    CHECK_TRUE(close_float(features[0], -0.5f));
    CHECK_TRUE(close_float(features[1], 0.0f));
    CHECK_TRUE(close_float(features[2], 0.5f));
    CHECK_TRUE(close_float(features[3], -0.6f));
    CHECK_TRUE(close_float(features[4], 0.2f));
    CHECK_TRUE(close_float(features[5], -0.2f));

    const float texture[4] = {
        128.0f / 255.0f,
        64.0f / 255.0f,
        192.0f / 255.0f,
        128.0f / 255.0f,
    };
    CHECK_TRUE(close_float(features[6], 2.0f * (0.5f * texture[0]) - 1.0f));
    CHECK_TRUE(close_float(features[7], 2.0f * (0.25f * texture[1]) - 1.0f));
    CHECK_TRUE(close_float(features[8], 2.0f * texture[2] - 1.0f));
    CHECK_TRUE(close_float(features[9], 2.0f * (0.8f * texture[2]) - 1.0f));
    CHECK_TRUE(close_float(features[10], 2.0f * (0.4f * texture[1]) - 1.0f));
    CHECK_TRUE(close_float(features[11], 2.0f * (0.5f * texture[3]) - 1.0f));

    CHECK_TRUE(
        trellis_mesh_segmentation_material_sampler_open(
            path, &asset, &sampler) == TRELLIS_STATUS_OK);
    const float barycentric[3] = {
        1.0f / 3.0f,
        1.0f / 3.0f,
        1.0f / 3.0f,
    };
    trellis_mesh_segmentation_surface_sample surface;
    memset(&surface, 0, sizeof(surface));
    CHECK_TRUE(
        trellis_mesh_segmentation_material_sampler_sample(
            &sampler, 0u, barycentric, &surface) == TRELLIS_STATUS_OK);
    CHECK_TRUE(close_float(surface.base_color[0], 0.25f));
    CHECK_TRUE(close_float(surface.base_color[1], 0.5f));
    CHECK_TRUE(close_float(surface.base_color[2], 0.75f));
    CHECK_TRUE(close_float(surface.alpha, 0.0f));
    CHECK_TRUE(surface.unlit == 0);
    CHECK_TRUE(
        trellis_mesh_segmentation_material_sampler_sample(
            &sampler, 1u, barycentric, &surface) == TRELLIS_STATUS_OK);
    const float red_linear = powf(
        ((texture[0] + 0.055f) / 1.055f), 2.4f);
    const float green_linear = powf(
        ((texture[1] + 0.055f) / 1.055f), 2.4f);
    const float blue_linear = powf(
        ((texture[2] + 0.055f) / 1.055f), 2.4f);
    CHECK_TRUE(close_float(surface.base_color[0], 0.5f * red_linear));
    CHECK_TRUE(close_float(surface.base_color[1], 0.25f * green_linear));
    CHECK_TRUE(close_float(surface.base_color[2], blue_linear));
    CHECK_TRUE(close_float(surface.alpha, 0.5f * texture[3]));
    trellis_mesh_segmentation_material_sampler_close(&sampler);
    CHECK_TRUE(sampler.implementation == NULL);

    CHECK_TRUE(
        trellis_mesh_segmentation_render_condition_rgba(
            &asset, 128, &condition) == TRELLIS_STATUS_OK);
    size_t covered_pixels = 0u;
    uint8_t maximum_alpha = 0u;
    for (size_t pixel = 0u; pixel < 128u * 128u; ++pixel) {
        const uint8_t alpha = condition.rgba[pixel * 4u + 3u];
        if (alpha != 0u) ++covered_pixels;
        if (alpha > maximum_alpha) maximum_alpha = alpha;
    }
    CHECK_TRUE(covered_pixels > 10u);
    /* The masked face has alpha 0; the blend face's 0.5 * 128/255
     * coverage survives the renderer and its premultiplied SSAA resolve. */
    CHECK_TRUE(maximum_alpha >= 63u && maximum_alpha <= 65u);
    trellis_mesh_segmentation_condition_image_free(&condition);

    CHECK_TRUE(
        trellis_mesh_segmentation_material_features(
            path, &asset, &mesh, &grid, &repeated) == TRELLIS_STATUS_OK);
    CHECK_TRUE(repeated != NULL && memcmp(features, repeated, 12u * sizeof(float)) == 0);

cleanup:
    trellis_mesh_segmentation_condition_image_free(&condition);
    trellis_mesh_segmentation_material_sampler_close(&sampler);
    free(repeated);
    free(features);
    trellis_mesh_rigging_asset_free(&asset);
    if (path[0] != '\0') remove(path);
}

int main(void) {
    test_factor_texture_wrap_and_face_mapping();
    if (failures != 0) {
        fprintf(stderr, "%d mesh material feature test(s) failed\n", failures);
        return 1;
    }
    printf("mesh material feature tests passed\n");
    return 0;
}
