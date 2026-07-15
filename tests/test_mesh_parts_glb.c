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

static int write_u32_le(FILE * file, uint32_t value) {
    const unsigned char bytes[4] = {
        (unsigned char) value,
        (unsigned char) (value >> 8u),
        (unsigned char) (value >> 16u),
        (unsigned char) (value >> 24u),
    };
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}

static void store_f32_le(unsigned char * output, float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    output[0] = (unsigned char) bits;
    output[1] = (unsigned char) (bits >> 8u);
    output[2] = (unsigned char) (bits >> 16u);
    output[3] = (unsigned char) (bits >> 24u);
}

static void store_u16_le(unsigned char * output, uint16_t value) {
    output[0] = (unsigned char) value;
    output[1] = (unsigned char) (value >> 8u);
}

static void store_float_array(
    unsigned char * output,
    const float * values,
    size_t count) {
    for (size_t i = 0; i < count; ++i) store_f32_le(output + i * 4u, values[i]);
}

static int write_appearance_source_glb(
    const char * path,
    const char * scale_json,
    const char * root_fragment) {
    static const float positions0[] = {
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
    };
    static const float positions1[] = {
        2.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f,
        2.0f, 1.0f, 0.0f, 3.0f, 1.0f, 0.0f,
    };
    static const float normals[] = {
        0.0f, 0.7071067812f, 0.7071067812f,
        0.0f, 0.7071067812f, 0.7071067812f,
        0.0f, 0.7071067812f, 0.7071067812f,
        0.0f, 0.7071067812f, 0.7071067812f,
    };
    static const float tangents[] = {
        1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    };
    static const float texcoords[] = {
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
    };
    static const float colors0[] = {
        1.0f, 0.0f, 0.0f, 0.25f, 0.0f, 1.0f, 0.0f, 0.5f,
        0.0f, 0.0f, 1.0f, 0.75f, 1.0f, 1.0f, 1.0f, 1.0f,
    };
    static const float colors1[] = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f, 1.0f, 0.9f, 0.8f,
    };
    static const uint16_t indices[] = {0u, 1u, 2u, 1u, 3u, 2u};
    static const unsigned char image_bytes[] = {
        0x89u, 0x50u, 0x4eu, 0x47u, 0x0du, 0x0au, 0x1au, 0x0au,
    };
    unsigned char binary[528] = {0};
    store_float_array(binary + 0u, positions0, 12u);
    store_float_array(binary + 48u, normals, 12u);
    store_float_array(binary + 96u, tangents, 16u);
    store_float_array(binary + 160u, texcoords, 8u);
    store_float_array(binary + 192u, colors0, 16u);
    for (size_t i = 0; i < 6u; ++i) store_u16_le(binary + 256u + i * 2u, indices[i]);
    store_float_array(binary + 268u, positions1, 12u);
    store_float_array(binary + 316u, normals, 12u);
    store_float_array(binary + 364u, tangents, 16u);
    store_float_array(binary + 428u, texcoords, 8u);
    store_float_array(binary + 460u, colors1, 12u);
    for (size_t i = 0; i < 6u; ++i) store_u16_le(binary + 508u + i * 2u, indices[i]);
    memcpy(binary + 520u, image_bytes, sizeof(image_bytes));

    static const char document_template[] =
        "{"
        "\"asset\":{\"version\":\"2.0\"},"
        "\"extensionsUsed\":[\"KHR_materials_clearcoat\",\"KHR_texture_transform\","
          "\"KHR_materials_variants\"],"
        "%s"
        "\"extensions\":{\"KHR_materials_variants\":{"
          "\"variants\":[{\"name\":\"alternate\"}]}},"
        "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
        "\"nodes\":[{\"mesh\":0,\"scale\":%s}],"
        "\"meshes\":[{\"primitives\":["
          "{\"attributes\":{\"POSITION\":0,\"NORMAL\":1,\"TANGENT\":2,"
            "\"TEXCOORD_0\":3,\"COLOR_0\":4},\"indices\":5,\"material\":0,"
            "\"extensions\":{\"KHR_materials_variants\":{\"mappings\":[{"
              "\"material\":1,\"variants\":[0]}]}}},"
          "{\"attributes\":{\"POSITION\":6,\"NORMAL\":7,\"TANGENT\":8,"
            "\"TEXCOORD_0\":9,\"COLOR_0\":10},\"indices\":11,\"material\":1}"
        "]}],"
        "\"materials\":["
          "{\"name\":\"paint\",\"doubleSided\":true,\"alphaMode\":\"MASK\","
            "\"alphaCutoff\":0.25,\"emissiveFactor\":[0.1,0.2,0.3],"
            "\"normalTexture\":{\"index\":0,\"scale\":0.7},"
            "\"occlusionTexture\":{\"index\":1,\"strength\":0.4},"
            "\"emissiveTexture\":{\"index\":0},"
            "\"extensions\":{\"KHR_materials_clearcoat\":{"
              "\"clearcoatFactor\":0.6,\"clearcoatTexture\":{\"index\":0}}},"
            "\"pbrMetallicRoughness\":{\"baseColorFactor\":[0.2,0.3,0.4,0.5],"
              "\"metallicFactor\":0.35,\"roughnessFactor\":0.65,"
              "\"baseColorTexture\":{\"index\":0,\"texCoord\":0,"
                "\"extensions\":{\"KHR_texture_transform\":{"
                  "\"offset\":[0.25,0.5],\"scale\":[0.5,0.75]}}},"
              "\"metallicRoughnessTexture\":{\"index\":1}}},"
          "{\"name\":\"glass\",\"pbrMetallicRoughness\":{"
            "\"baseColorFactor\":[0.8,0.7,0.6,1],\"metallicFactor\":0.1,"
            "\"roughnessFactor\":0.2}}"
        "],"
        "\"samplers\":[{\"magFilter\":9729,\"minFilter\":9987,"
          "\"wrapS\":33071,\"wrapT\":33648}],"
        "\"images\":[{\"name\":\"embedded paint\",\"bufferView\":12,"
          "\"mimeType\":\"image/png\"},{\"name\":\"data paint\","
          "\"uri\":\"data:image/png;base64,iVBORw0KGgo=\"}],"
        "\"textures\":[{\"name\":\"base\",\"sampler\":0,\"source\":0},"
          "{\"name\":\"packed\",\"sampler\":0,\"source\":1}],"
        "\"buffers\":[{\"byteLength\":528}],"
        "\"bufferViews\":["
          "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":48},"
          "{\"buffer\":0,\"byteOffset\":48,\"byteLength\":48},"
          "{\"buffer\":0,\"byteOffset\":96,\"byteLength\":64},"
          "{\"buffer\":0,\"byteOffset\":160,\"byteLength\":32},"
          "{\"buffer\":0,\"byteOffset\":192,\"byteLength\":64},"
          "{\"buffer\":0,\"byteOffset\":256,\"byteLength\":12},"
          "{\"buffer\":0,\"byteOffset\":268,\"byteLength\":48},"
          "{\"buffer\":0,\"byteOffset\":316,\"byteLength\":48},"
          "{\"buffer\":0,\"byteOffset\":364,\"byteLength\":64},"
          "{\"buffer\":0,\"byteOffset\":428,\"byteLength\":32},"
          "{\"buffer\":0,\"byteOffset\":460,\"byteLength\":48},"
          "{\"buffer\":0,\"byteOffset\":508,\"byteLength\":12},"
          "{\"buffer\":0,\"byteOffset\":520,\"byteLength\":8}"
        "],"
        "\"accessors\":["
          "{\"bufferView\":0,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\"},"
          "{\"bufferView\":1,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\"},"
          "{\"bufferView\":2,\"componentType\":5126,\"count\":4,\"type\":\"VEC4\"},"
          "{\"bufferView\":3,\"componentType\":5126,\"count\":4,\"type\":\"VEC2\"},"
          "{\"bufferView\":4,\"componentType\":5126,\"count\":4,\"type\":\"VEC4\"},"
          "{\"bufferView\":5,\"componentType\":5123,\"count\":6,\"type\":\"SCALAR\"},"
          "{\"bufferView\":6,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\"},"
          "{\"bufferView\":7,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\"},"
          "{\"bufferView\":8,\"componentType\":5126,\"count\":4,\"type\":\"VEC4\"},"
          "{\"bufferView\":9,\"componentType\":5126,\"count\":4,\"type\":\"VEC2\"},"
          "{\"bufferView\":10,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\"},"
          "{\"bufferView\":11,\"componentType\":5123,\"count\":6,\"type\":\"SCALAR\"}"
        "]}"
        ;
    char document[8192];
    const int document_length = snprintf(
        document, sizeof(document), document_template, root_fragment, scale_json);
    if (document_length <= 0 || (size_t) document_length >= sizeof(document)) return 0;
    const size_t json_size = strlen(document);
    const size_t padded_json_size = (json_size + 3u) & ~(size_t) 3u;
    const uint32_t total_size = (uint32_t) (
        12u + 8u + padded_json_size + 8u + sizeof(binary));
    FILE * file = fopen(path, "wb");
    if (file == NULL) return 0;
    int ok = write_u32_le(file, UINT32_C(0x46546c67)) &&
        write_u32_le(file, 2u) && write_u32_le(file, total_size) &&
        write_u32_le(file, (uint32_t) padded_json_size) &&
        write_u32_le(file, UINT32_C(0x4e4f534a)) &&
        fwrite(document, 1, json_size, file) == json_size;
    for (size_t i = json_size; i < padded_json_size && ok; ++i) {
        const unsigned char space = 0x20u;
        ok = fwrite(&space, 1, 1u, file) == 1u;
    }
    ok = ok && write_u32_le(file, (uint32_t) sizeof(binary)) &&
        write_u32_le(file, UINT32_C(0x004e4942)) &&
        fwrite(binary, 1, sizeof(binary), file) == sizeof(binary);
    if (fclose(file) != 0) ok = 0;
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
    CHECK_TRUE(data->meshes_count == 2 && data->materials_count == 1);
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
    CHECK_TRUE(strcmp(data->materials[0].name, "trellis_default_material") == 0);
    CHECK_TRUE(close_float(
        data->materials[0].pbr_metallic_roughness.base_color_factor[0], 1.0f));
    CHECK_TRUE(close_float(
        data->materials[0].pbr_metallic_roughness.metallic_factor, 1.0f));
    CHECK_TRUE(close_float(
        data->materials[0].pbr_metallic_roughness.roughness_factor, 1.0f));
    CHECK_TRUE(
        root->children[0]->mesh->primitives[0].material ==
        root->children[1]->mesh->primitives[0].material);
    ok = 1;

cleanup:
    if (data != NULL) cgltf_free(data);
    if (path[0] != '\0') remove(path);
    return ok;
}

static int run_source_appearance_round_trip(void) {
    static const uint32_t parts[] = {0u, 1u, 0u, 1u};
    static const unsigned char image_bytes[] = {
        0x89u, 0x50u, 0x4eu, 0x47u, 0x0du, 0x0au, 0x1au, 0x0au,
    };
    char source_path[4096] = {0};
    char output_path[4096] = {0};
    char error[512] = {0};
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    cgltf_data * data = NULL;
    int ok = 0;

    CHECK_TRUE(trellis_make_temp_path(
        source_path, sizeof(source_path), "trellis_parts_source", ".glb"));
    CHECK_TRUE(trellis_make_temp_path(
        output_path, sizeof(output_path), "trellis_parts_output", ".glb"));
    CHECK_TRUE(write_appearance_source_glb(source_path, "[-2,3,4]", ""));
    CHECK_TRUE(
        trellis_mesh_rigging_gltf_load(
            source_path, &asset, error, sizeof(error)) == TRELLIS_STATUS_OK);
    CHECK_TRUE(asset.primitive_count == 2u && asset.triangle_count == 4u);
    CHECK_TRUE(asset.primitives[0].has_source_normal);
    CHECK_TRUE(asset.primitives[0].has_source_tangent);
    CHECK_TRUE(asset.primitives[0].texcoord_mask == UINT64_C(1));
    CHECK_TRUE(asset.primitives[0].color_mask == UINT64_C(1));
    CHECK_TRUE(asset.primitives[0].color_vec4_mask == UINT64_C(1));
    CHECK_TRUE(asset.primitives[1].color_vec4_mask == 0u);
    CHECK_TRUE(close_float(asset.normals[1], 0.8f));
    CHECK_TRUE(close_float(asset.normals[2], 0.6f));
    CHECK_TRUE(close_float(asset.tangents[0], -1.0f));
    CHECK_TRUE(close_float(asset.tangents[1], 0.0f));
    CHECK_TRUE(close_float(asset.tangents[2], 0.0f));
    CHECK_TRUE(close_float(asset.tangents[3], -1.0f));

    CHECK_TRUE(
        trellis_mesh_segmentation_write_parts_glb(
            output_path,
            &asset,
            parts,
            4u,
            2u,
            NULL,
            error,
            sizeof(error)) == TRELLIS_STATUS_OK);
    cgltf_options options;
    memset(&options, 0, sizeof(options));
    CHECK_TRUE(cgltf_parse_file(&options, output_path, &data) == cgltf_result_success);
    CHECK_TRUE(cgltf_load_buffers(&options, data, output_path) == cgltf_result_success);
    CHECK_TRUE(cgltf_validate(data) == cgltf_result_success);
    CHECK_TRUE(data->materials_count == 2u);
    CHECK_TRUE(data->variants_count == 1u);
    CHECK_TRUE(strcmp(data->variants[0].name, "alternate") == 0);
    CHECK_TRUE(data->meshes_count == 2u);
    CHECK_TRUE(data->scene != NULL && data->scene->nodes_count == 1u);
    CHECK_TRUE(data->scene->nodes[0]->children_count == 2u);
    CHECK_TRUE(data->images_count == 2u && data->textures_count == 2u);
    CHECK_TRUE(data->samplers_count == 1u);
    CHECK_TRUE(data->images[0].uri == NULL && data->images[0].buffer_view != NULL);
    CHECK_TRUE(data->images[0].mime_type != NULL &&
        strcmp(data->images[0].mime_type, "image/png") == 0);
    CHECK_TRUE(data->images[0].buffer_view->size == sizeof(image_bytes));
    CHECK_TRUE(memcmp(
        cgltf_buffer_view_data(data->images[0].buffer_view),
        image_bytes,
        sizeof(image_bytes)) == 0);
    CHECK_TRUE(data->images[1].uri == NULL && data->images[1].buffer_view != NULL);
    CHECK_TRUE(data->images[1].buffer_view->size == sizeof(image_bytes));
    CHECK_TRUE(memcmp(
        cgltf_buffer_view_data(data->images[1].buffer_view),
        image_bytes,
        sizeof(image_bytes)) == 0);
    CHECK_TRUE(data->textures[0].sampler == &data->samplers[0]);
    CHECK_TRUE(data->textures[0].image == &data->images[0]);
    CHECK_TRUE(data->textures[1].image == &data->images[1]);
    CHECK_TRUE(data->samplers[0].mag_filter == 9729);
    CHECK_TRUE(data->samplers[0].min_filter == 9987);
    CHECK_TRUE(data->samplers[0].wrap_s == 33071);
    CHECK_TRUE(data->samplers[0].wrap_t == 33648);

    const cgltf_material * paint = &data->materials[0];
    CHECK_TRUE(strcmp(paint->name, "paint") == 0);
    CHECK_TRUE(paint->has_pbr_metallic_roughness);
    CHECK_TRUE(close_float(
        paint->pbr_metallic_roughness.base_color_factor[0], 0.2f));
    CHECK_TRUE(close_float(
        paint->pbr_metallic_roughness.metallic_factor, 0.35f));
    CHECK_TRUE(close_float(
        paint->pbr_metallic_roughness.roughness_factor, 0.65f));
    CHECK_TRUE(paint->pbr_metallic_roughness.base_color_texture.texture ==
        &data->textures[0]);
    CHECK_TRUE(paint->pbr_metallic_roughness.base_color_texture.has_transform);
    CHECK_TRUE(close_float(
        paint->pbr_metallic_roughness.base_color_texture.transform.offset[0], 0.25f));
    CHECK_TRUE(paint->pbr_metallic_roughness.metallic_roughness_texture.texture ==
        &data->textures[1]);
    CHECK_TRUE(paint->normal_texture.texture == &data->textures[0]);
    CHECK_TRUE(paint->occlusion_texture.texture == &data->textures[1]);
    CHECK_TRUE(paint->emissive_texture.texture == &data->textures[0]);
    CHECK_TRUE(close_float(paint->normal_texture.scale, 0.7f));
    CHECK_TRUE(close_float(paint->occlusion_texture.scale, 0.4f));
    CHECK_TRUE(close_float(paint->emissive_factor[2], 0.3f));
    CHECK_TRUE(paint->has_clearcoat);
    CHECK_TRUE(close_float(paint->clearcoat.clearcoat_factor, 0.6f));
    CHECK_TRUE(paint->clearcoat.clearcoat_texture.texture == &data->textures[0]);

    size_t total_faces = 0;
    for (size_t part = 0; part < 2u; ++part) {
        const cgltf_mesh * mesh = data->scene->nodes[0]->children[part]->mesh;
        CHECK_TRUE(mesh != NULL && mesh->primitives_count == 2u);
        for (size_t source_primitive = 0; source_primitive < 2u; ++source_primitive) {
            const cgltf_primitive * primitive = &mesh->primitives[source_primitive];
            CHECK_TRUE(primitive->material == &data->materials[source_primitive]);
            if (source_primitive == 0u) {
                CHECK_TRUE(primitive->mappings_count == 1u);
                CHECK_TRUE(primitive->mappings[0].variant == 0u);
                CHECK_TRUE(primitive->mappings[0].material == &data->materials[1]);
            } else {
                CHECK_TRUE(primitive->mappings_count == 0u);
            }
            CHECK_TRUE(primitive->indices != NULL && primitive->indices->count == 3u);
            total_faces += (size_t) primitive->indices->count / 3u;
            const cgltf_accessor * position = cgltf_find_accessor(
                primitive, cgltf_attribute_type_position, 0);
            const cgltf_accessor * normal = cgltf_find_accessor(
                primitive, cgltf_attribute_type_normal, 0);
            const cgltf_accessor * tangent = cgltf_find_accessor(
                primitive, cgltf_attribute_type_tangent, 0);
            const cgltf_accessor * uv = cgltf_find_accessor(
                primitive, cgltf_attribute_type_texcoord, 0);
            const cgltf_accessor * color = cgltf_find_accessor(
                primitive, cgltf_attribute_type_color, 0);
            CHECK_TRUE(position != NULL && normal != NULL && tangent != NULL);
            CHECK_TRUE(uv != NULL && color != NULL);
            CHECK_TRUE(position->count == 3u && normal->count == 3u);
            CHECK_TRUE(tangent->type == cgltf_type_vec4);
            CHECK_TRUE(uv->type == cgltf_type_vec2);
            CHECK_TRUE(color->type == (
                source_primitive == 0u ? cgltf_type_vec4 : cgltf_type_vec3));
            float normal_value[3] = {0};
            float tangent_value[4] = {0};
            float uv_value[2] = {0};
            float color_value[4] = {0};
            CHECK_TRUE(cgltf_accessor_read_float(normal, 0, normal_value, 3u));
            CHECK_TRUE(cgltf_accessor_read_float(tangent, 0, tangent_value, 4u));
            CHECK_TRUE(cgltf_accessor_read_float(uv, 0, uv_value, 2u));
            CHECK_TRUE(cgltf_accessor_read_float(
                color, 0, color_value, source_primitive == 0u ? 4u : 3u));
            CHECK_TRUE(close_float(normal_value[1], 0.8f));
            CHECK_TRUE(close_float(normal_value[2], 0.6f));
            CHECK_TRUE(close_float(tangent_value[0], -1.0f));
            CHECK_TRUE(close_float(tangent_value[3], -1.0f));
            CHECK_TRUE(isfinite(uv_value[0]) && isfinite(uv_value[1]));
            CHECK_TRUE(isfinite(color_value[0]) && isfinite(color_value[1]));
        }
    }
    CHECK_TRUE(total_faces == 4u);
    ok = 1;

cleanup:
    if (data != NULL) cgltf_free(data);
    trellis_mesh_rigging_asset_free(&asset);
    if (source_path[0] != '\0') trellis_unlink(source_path);
    if (output_path[0] != '\0') trellis_unlink(output_path);
    return ok;
}

static int run_discarded_face_round_trip(void) {
    static const uint32_t parts[] = {UINT32_MAX, 0u, UINT32_MAX, 0u};
    char source_path[4096] = {0};
    char output_path[4096] = {0};
    char error[512] = {0};
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    cgltf_data * data = NULL;
    int ok = 0;

    CHECK_TRUE(trellis_make_temp_path(
        source_path, sizeof(source_path), "trellis_parts_discard_source", ".glb"));
    CHECK_TRUE(trellis_make_temp_path(
        output_path, sizeof(output_path), "trellis_parts_discard_output", ".glb"));
    CHECK_TRUE(write_appearance_source_glb(source_path, "[1,1,1]", ""));
    CHECK_TRUE(trellis_mesh_rigging_gltf_load(
        source_path, &asset, error, sizeof(error)) == TRELLIS_STATUS_OK);
    CHECK_TRUE(asset.primitive_count == 2u && asset.triangle_count == 4u);
    CHECK_TRUE(trellis_mesh_segmentation_write_parts_glb(
        output_path, &asset, parts, 4u, 1u, NULL, error, sizeof(error)) ==
        TRELLIS_STATUS_OK);

    cgltf_options options;
    memset(&options, 0, sizeof(options));
    CHECK_TRUE(cgltf_parse_file(&options, output_path, &data) ==
        cgltf_result_success);
    CHECK_TRUE(cgltf_load_buffers(&options, data, output_path) ==
        cgltf_result_success);
    CHECK_TRUE(cgltf_validate(data) == cgltf_result_success);
    CHECK_TRUE(data->scene != NULL && data->scene->nodes_count == 1u);
    CHECK_TRUE(data->scene->nodes[0]->children_count == 1u);
    const cgltf_mesh * mesh = data->scene->nodes[0]->children[0]->mesh;
    CHECK_TRUE(mesh != NULL && mesh->primitives_count == 2u);
    size_t total_faces = 0;
    for (size_t primitive = 0; primitive < 2u; ++primitive) {
        CHECK_TRUE(mesh->primitives[primitive].indices != NULL);
        CHECK_TRUE(mesh->primitives[primitive].indices->count == 3u);
        CHECK_TRUE(mesh->primitives[primitive].material ==
            &data->materials[primitive]);
        total_faces += (size_t) mesh->primitives[primitive].indices->count / 3u;
    }
    CHECK_TRUE(total_faces == 2u);
    ok = 1;

cleanup:
    if (data != NULL) cgltf_free(data);
    trellis_mesh_rigging_asset_free(&asset);
    if (source_path[0] != '\0') trellis_unlink(source_path);
    if (output_path[0] != '\0') trellis_unlink(output_path);
    return ok;
}

static int run_tiny_uniform_scale_preservation(void) {
    char source_path[4096] = {0};
    char error[512] = {0};
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    int ok = 0;
    CHECK_TRUE(trellis_make_temp_path(
        source_path, sizeof(source_path), "trellis_parts_tiny_scale", ".glb"));
    CHECK_TRUE(write_appearance_source_glb(
        source_path, "[1e-7,1e-7,1e-7]", ""));
    CHECK_TRUE(
        trellis_mesh_rigging_gltf_load(
            source_path, &asset, error, sizeof(error)) == TRELLIS_STATUS_OK);
    CHECK_TRUE(close_float(asset.normals[0], 0.0f));
    CHECK_TRUE(close_float(asset.normals[1], 0.7071067812f));
    CHECK_TRUE(close_float(asset.normals[2], 0.7071067812f));
    CHECK_TRUE(close_float(asset.tangents[0], 1.0f));
    CHECK_TRUE(close_float(asset.tangents[3], 1.0f));
    ok = 1;
cleanup:
    trellis_mesh_rigging_asset_free(&asset);
    if (source_path[0] != '\0') trellis_unlink(source_path);
    return ok;
}

static int run_required_extension_rejection(void) {
    static const uint32_t parts[] = {0u, 0u, 0u, 0u};
    static const unsigned char sentinel[] = "required extension must preserve output";
    char source_path[4096] = {0};
    char output_path[4096] = {0};
    char error[512] = {0};
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    int ok = 0;
    CHECK_TRUE(trellis_make_temp_path(
        source_path, sizeof(source_path), "trellis_parts_required", ".glb"));
    CHECK_TRUE(trellis_make_temp_path(
        output_path, sizeof(output_path), "trellis_parts_required_output", ".glb"));
    CHECK_TRUE(write_appearance_source_glb(
        source_path,
        "[1,1,1]",
        "\"extensionsRequired\":[\"KHR_materials_clearcoat\"],"));
    CHECK_TRUE(
        trellis_mesh_rigging_gltf_load(
            source_path, &asset, error, sizeof(error)) == TRELLIS_STATUS_OK);
    CHECK_TRUE(write_bytes(output_path, sentinel, sizeof(sentinel)));
    CHECK_TRUE(
        trellis_mesh_segmentation_write_parts_glb(
            output_path,
            &asset,
            parts,
            4u,
            1u,
            NULL,
            error,
            sizeof(error)) == TRELLIS_STATUS_NOT_IMPLEMENTED);
    CHECK_TRUE(strstr(error, "required") != NULL);
    CHECK_TRUE(file_equals(output_path, sentinel, sizeof(sentinel)));
    ok = 1;
cleanup:
    trellis_mesh_rigging_asset_free(&asset);
    if (source_path[0] != '\0') trellis_unlink(source_path);
    if (output_path[0] != '\0') trellis_unlink(output_path);
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
    static const uint32_t discarded[] = {UINT32_MAX};
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
    CHECK_TRUE(
        trellis_mesh_segmentation_write_parts_glb(
            path, &asset, discarded, 1, 1, NULL, error, sizeof(error)) ==
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
    run_source_appearance_round_trip();
    run_discarded_face_round_trip();
    run_tiny_uniform_scale_preservation();
    run_required_extension_rejection();
    run_validation_checks();
    run_atomic_output_checks();
    if (failures != 0) return 1;
    printf("mesh parts GLB round-trip passed\n");
    return 0;
}
