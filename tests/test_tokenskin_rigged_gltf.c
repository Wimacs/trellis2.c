#include "rigged_gltf.h"

#ifdef TRELLIS_TOKENSKIN_TEST_STANDALONE
#define CGLTF_WRITE_IMPLEMENTATION
#include "cgltf_write.h"
#else
#include "cgltf.h"
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "test requirement failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
            goto fail; \
        } \
    } while (0)

static int nearly_equal(float left, float right, float tolerance) {
    return fabsf(left - right) <= tolerance;
}

static void make_asset(trellis_mesh_rigging_asset * asset) {
    static float positions[] = {
        8.0f, 18.0f, 30.0f,
        9.0f, 18.0f, 30.0f,
        8.0f, 19.0f, 30.0f,
        8.5f, 18.5f, 30.0f,
        12.0f, 22.0f, 31.0f,
        13.0f, 22.0f, 31.0f,
        12.0f, 23.0f, 31.0f,
    };
    static float normals[] = {
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
    };
    static float face_normals[] = {
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
    };
    static uint32_t triangles[] = { 0, 1, 2, 4, 5, 6 };
    static trellis_mesh_rigging_primitive_range primitives[] = {
        { 0, 4, 0, 1, 4, 2, 0, 7, SIZE_MAX },
        { 4, 3, 1, 1, 5, 3, 1, 8, 6 },
    };
    memset(asset, 0, sizeof(*asset));
    asset->positions = positions;
    asset->normals = normals;
    asset->face_normals = face_normals;
    asset->triangles = triangles;
    asset->primitives = primitives;
    asset->vertex_count = 7;
    asset->triangle_count = 2;
    asset->primitive_count = 2;
    asset->aabb_min[0] = 8.0f;
    asset->aabb_min[1] = 18.0f;
    asset->aabb_min[2] = 30.0f;
    asset->aabb_max[0] = 13.0f;
    asset->aabb_max[1] = 23.0f;
    asset->aabb_max[2] = 31.0f;
}

static void make_normalization(trellis_mesh_rigging_normalization * normalization) {
    memset(normalization, 0, sizeof(*normalization));
    normalization->center[0] = 10.0f;
    normalization->center[1] = 20.0f;
    normalization->center[2] = 30.0f;
    normalization->scale = 0.5f;
    normalization->inverse_scale = 2.0f;
    normalization->normalized_from_world[0] = 0.5f;
    normalization->normalized_from_world[5] = 0.5f;
    normalization->normalized_from_world[10] = 0.5f;
    normalization->normalized_from_world[15] = 1.0f;
    normalization->normalized_from_world[12] = -5.0f;
    normalization->normalized_from_world[13] = -10.0f;
    normalization->normalized_from_world[14] = -15.0f;
    normalization->world_from_normalized[0] = 2.0f;
    normalization->world_from_normalized[5] = 2.0f;
    normalization->world_from_normalized[10] = 2.0f;
    normalization->world_from_normalized[15] = 1.0f;
    normalization->world_from_normalized[12] = 10.0f;
    normalization->world_from_normalized[13] = 20.0f;
    normalization->world_from_normalized[14] = 30.0f;
}

static void make_skeleton(trellis_tokenskin_skeleton * skeleton) {
    static float joints[] = {
        0.0f, 0.0f, 0.0f,
        0.5f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.0f,
        0.5f, 0.5f, 0.0f,
        1.0f, 0.5f, 0.0f,
    };
    static float parent_joints[] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.5f, 0.0f, 0.0f,
        0.5f, 0.0f, 0.0f,
    };
    static int32_t parents[] = { -1, 0, 0, 1, 1 };
    memset(skeleton, 0, sizeof(*skeleton));
    skeleton->joints_xyz = joints;
    skeleton->parent_joints_xyz = parent_joints;
    skeleton->parents = parents;
    skeleton->joint_count = 5;
}

static void expected_direct_influences(
    size_t vertex,
    uint32_t joints[4],
    float weights[4]) {
    static const uint32_t expected_joints[7][4] = {
        { 4, 3, 2, 1 },
        { 0, 0, 0, 0 },
        { 0, 1, 2, 3 },
        { 0, 1, 2, 3 },
        { 0, 1, 2, 3 },
        { 4, 3, 1, 0 },
        { 3, 0, 1, 2 },
    };
    static const float expected_weights[7][4] = {
        { 0.4f, 0.3f, 0.2f, 0.1f },
        { 1.0f, 0.0f, 0.0f, 0.0f },
        { 0.25f, 0.25f, 0.25f, 0.25f },
        { 1.0f, 0.0f, 0.0f, 0.0f },
        { 1.0f, 0.0f, 0.0f, 0.0f },
        { 0.7f, 0.2f, 0.1f, 0.0f },
        { 1.0f, 0.0f, 0.0f, 0.0f },
    };
    memcpy(joints, expected_joints[vertex], sizeof(expected_joints[vertex]));
    memcpy(weights, expected_weights[vertex], sizeof(expected_weights[vertex]));
}

static void expected_sampled_influences(
    size_t vertex,
    uint32_t joints[4],
    float weights[4]) {
    static const uint32_t first_joints[4] = { 4, 3, 2, 1 };
    static const float first_weights[4] = {
        5.0f / 14.0f, 4.0f / 14.0f, 3.0f / 14.0f, 2.0f / 14.0f,
    };
    static const uint32_t second_joints[4] = { 0, 1, 2, 3 };
    static const float second_weights[4] = {
        5.0f / 14.0f, 4.0f / 14.0f, 3.0f / 14.0f, 2.0f / 14.0f,
    };
    if (vertex < 4u) {
        memcpy(joints, first_joints, sizeof(first_joints));
        memcpy(weights, first_weights, sizeof(first_weights));
    } else {
        memcpy(joints, second_joints, sizeof(second_joints));
        memcpy(weights, second_weights, sizeof(second_weights));
    }
}

static int validate_glb(
    const char * path,
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_rigging_normalization * normalization,
    const trellis_tokenskin_skeleton * skeleton,
    int sampled_weights) {
    cgltf_options options;
    cgltf_data * data = NULL;
    memset(&options, 0, sizeof(options));
    TEST_REQUIRE(cgltf_parse_file(&options, path, &data) == cgltf_result_success);
    TEST_REQUIRE(cgltf_load_buffers(&options, data, path) == cgltf_result_success);
    TEST_REQUIRE(cgltf_validate(data) == cgltf_result_success);
    TEST_REQUIRE(data->file_type == cgltf_file_type_glb);
    TEST_REQUIRE(data->meshes_count == 1u);
    TEST_REQUIRE(data->meshes[0].primitives_count == asset->primitive_count);
    TEST_REQUIRE(data->materials_count == 1u);
    TEST_REQUIRE(data->materials[0].has_pbr_metallic_roughness);
    TEST_REQUIRE(nearly_equal(
        data->materials[0].pbr_metallic_roughness.base_color_factor[0], 1.0f, 1e-6f));
    TEST_REQUIRE(nearly_equal(
        data->materials[0].pbr_metallic_roughness.base_color_factor[1], 1.0f, 1e-6f));
    TEST_REQUIRE(nearly_equal(
        data->materials[0].pbr_metallic_roughness.base_color_factor[2], 1.0f, 1e-6f));
    TEST_REQUIRE(nearly_equal(
        data->materials[0].pbr_metallic_roughness.base_color_factor[3], 1.0f, 1e-6f));
    TEST_REQUIRE(nearly_equal(
        data->materials[0].pbr_metallic_roughness.metallic_factor, 0.0f, 1e-6f));
    TEST_REQUIRE(nearly_equal(
        data->materials[0].pbr_metallic_roughness.roughness_factor, 1.0f, 1e-6f));
    TEST_REQUIRE(data->skins_count == 1u);
    TEST_REQUIRE(data->skins[0].joints_count == skeleton->joint_count);
    TEST_REQUIRE(data->skins[0].inverse_bind_matrices != NULL);
    TEST_REQUIRE(data->skins[0].skeleton == data->skins[0].joints[0]);
    TEST_REQUIRE(data->scenes_count == 1u);
    TEST_REQUIRE(data->scene != NULL);
    TEST_REQUIRE(data->scene->nodes_count == 2u);

    cgltf_node * mesh_node = NULL;
    for (cgltf_size node = 0; node < data->nodes_count; ++node) {
        if (data->nodes[node].mesh != NULL) {
            TEST_REQUIRE(mesh_node == NULL);
            mesh_node = &data->nodes[node];
        }
    }
    TEST_REQUIRE(mesh_node != NULL);
    TEST_REQUIRE(mesh_node->mesh == &data->meshes[0]);
    TEST_REQUIRE(mesh_node->skin == &data->skins[0]);
    TEST_REQUIRE(mesh_node->parent == NULL);

    const cgltf_accessor * inverse_binds = data->skins[0].inverse_bind_matrices;
    TEST_REQUIRE(inverse_binds->component_type == cgltf_component_type_r_32f);
    TEST_REQUIRE(inverse_binds->type == cgltf_type_mat4);
    TEST_REQUIRE(inverse_binds->count == skeleton->joint_count);
    for (size_t joint = 0; joint < skeleton->joint_count; ++joint) {
        const float expected_world[3] = {
            skeleton->joints_xyz[joint * 3u + 0u] * normalization->inverse_scale +
                normalization->center[0],
            skeleton->joints_xyz[joint * 3u + 1u] * normalization->inverse_scale +
                normalization->center[1],
            skeleton->joints_xyz[joint * 3u + 2u] * normalization->inverse_scale +
                normalization->center[2],
        };
        const int32_t parent = skeleton->parents[joint];
        cgltf_node * node = data->skins[0].joints[joint];
        TEST_REQUIRE(node != NULL);
        TEST_REQUIRE(node->has_translation);
        TEST_REQUIRE(node->parent == (parent < 0 ? NULL : data->skins[0].joints[parent]));
        for (size_t axis = 0; axis < 3u; ++axis) {
            const float expected_local = parent < 0 ? expected_world[axis] :
                expected_world[axis] -
                    (skeleton->joints_xyz[(size_t) parent * 3u + axis] *
                        normalization->inverse_scale + normalization->center[axis]);
            TEST_REQUIRE(nearly_equal(node->translation[axis], expected_local, 1e-6f));
        }
        float world_matrix[16];
        cgltf_node_transform_world(node, world_matrix);
        TEST_REQUIRE(nearly_equal(world_matrix[12], expected_world[0], 1e-6f));
        TEST_REQUIRE(nearly_equal(world_matrix[13], expected_world[1], 1e-6f));
        TEST_REQUIRE(nearly_equal(world_matrix[14], expected_world[2], 1e-6f));

        float inverse_bind[16];
        TEST_REQUIRE(cgltf_accessor_read_float(
            inverse_binds, joint, inverse_bind, 16u));
        for (size_t element = 0; element < 16u; ++element) {
            float expected = 0.0f;
            if (element == 0u || element == 5u || element == 10u || element == 15u) {
                expected = 1.0f;
            } else if (element == 12u) {
                expected = -expected_world[0];
            } else if (element == 13u) {
                expected = -expected_world[1];
            } else if (element == 14u) {
                expected = -expected_world[2];
            }
            TEST_REQUIRE(nearly_equal(inverse_bind[element], expected, 1e-6f));
        }
    }

    for (size_t primitive_index = 0; primitive_index < asset->primitive_count; ++primitive_index) {
        const trellis_mesh_rigging_primitive_range * range = &asset->primitives[primitive_index];
        const cgltf_primitive * primitive = &data->meshes[0].primitives[primitive_index];
        TEST_REQUIRE(primitive->type == cgltf_primitive_type_triangles);
        TEST_REQUIRE(primitive->material == &data->materials[0]);
        TEST_REQUIRE(primitive->indices != NULL);
        TEST_REQUIRE(primitive->indices->component_type == cgltf_component_type_r_32u);
        TEST_REQUIRE(primitive->indices->type == cgltf_type_scalar);
        TEST_REQUIRE(primitive->indices->count == range->triangle_count * 3u);

        const cgltf_accessor * position =
            cgltf_find_accessor(primitive, cgltf_attribute_type_position, 0);
        const cgltf_accessor * normal =
            cgltf_find_accessor(primitive, cgltf_attribute_type_normal, 0);
        const cgltf_accessor * joints =
            cgltf_find_accessor(primitive, cgltf_attribute_type_joints, 0);
        const cgltf_accessor * weights =
            cgltf_find_accessor(primitive, cgltf_attribute_type_weights, 0);
        TEST_REQUIRE(position != NULL && normal != NULL && joints != NULL && weights != NULL);
        TEST_REQUIRE(position->component_type == cgltf_component_type_r_32f);
        TEST_REQUIRE(position->type == cgltf_type_vec3);
        TEST_REQUIRE(position->count == range->vertex_count);
        TEST_REQUIRE(position->has_min && position->has_max);
        TEST_REQUIRE(normal->component_type == cgltf_component_type_r_32f);
        TEST_REQUIRE(normal->type == cgltf_type_vec3);
        TEST_REQUIRE(normal->count == range->vertex_count);
        TEST_REQUIRE(joints->component_type == cgltf_component_type_r_16u);
        TEST_REQUIRE(joints->type == cgltf_type_vec4);
        TEST_REQUIRE(joints->count == range->vertex_count);
        TEST_REQUIRE(weights->component_type == cgltf_component_type_r_32f);
        TEST_REQUIRE(weights->type == cgltf_type_vec4);
        TEST_REQUIRE(weights->count == range->vertex_count);

        float expected_min[3] = { INFINITY, INFINITY, INFINITY };
        float expected_max[3] = { -INFINITY, -INFINITY, -INFINITY };
        for (size_t local_vertex = 0; local_vertex < range->vertex_count; ++local_vertex) {
            const size_t global_vertex = range->first_vertex + local_vertex;
            float loaded_position[3];
            float loaded_normal[3];
            cgltf_uint loaded_joints[4];
            float loaded_weights[4];
            TEST_REQUIRE(cgltf_accessor_read_float(
                position, local_vertex, loaded_position, 3u));
            TEST_REQUIRE(cgltf_accessor_read_float(
                normal, local_vertex, loaded_normal, 3u));
            TEST_REQUIRE(cgltf_accessor_read_uint(
                joints, local_vertex, loaded_joints, 4u));
            TEST_REQUIRE(cgltf_accessor_read_float(
                weights, local_vertex, loaded_weights, 4u));
            uint32_t expected_joints[4];
            float expected_weights[4];
            if (sampled_weights) {
                expected_sampled_influences(global_vertex, expected_joints, expected_weights);
            } else {
                expected_direct_influences(global_vertex, expected_joints, expected_weights);
            }
            float weight_sum = 0.0f;
            for (size_t axis = 0; axis < 3u; ++axis) {
                const float expected = asset->positions[global_vertex * 3u + axis];
                TEST_REQUIRE(nearly_equal(loaded_position[axis], expected, 1e-6f));
                TEST_REQUIRE(nearly_equal(
                    loaded_normal[axis], asset->normals[global_vertex * 3u + axis], 1e-6f));
                if (expected < expected_min[axis]) expected_min[axis] = expected;
                if (expected > expected_max[axis]) expected_max[axis] = expected;
            }
            for (size_t influence = 0; influence < 4u; ++influence) {
                TEST_REQUIRE(loaded_joints[influence] == expected_joints[influence]);
                TEST_REQUIRE(loaded_joints[influence] < skeleton->joint_count);
                TEST_REQUIRE(loaded_weights[influence] >= 0.0f);
                TEST_REQUIRE(nearly_equal(
                    loaded_weights[influence], expected_weights[influence], 2e-6f));
                weight_sum += loaded_weights[influence];
            }
            TEST_REQUIRE(nearly_equal(weight_sum, 1.0f, 2e-6f));
        }
        for (size_t axis = 0; axis < 3u; ++axis) {
            TEST_REQUIRE(nearly_equal(position->min[axis], expected_min[axis], 1e-6f));
            TEST_REQUIRE(nearly_equal(position->max[axis], expected_max[axis], 1e-6f));
        }

        for (size_t triangle = 0; triangle < range->triangle_count; ++triangle) {
            uint32_t local_indices[3];
            float triangle_positions[3][3];
            for (size_t corner = 0; corner < 3u; ++corner) {
                const size_t element = triangle * 3u + corner;
                local_indices[corner] = (uint32_t)
                    cgltf_accessor_read_index(primitive->indices, element);
                const uint32_t expected_global =
                    asset->triangles[(range->first_triangle + triangle) * 3u + corner];
                TEST_REQUIRE(local_indices[corner] == expected_global - range->first_vertex);
                TEST_REQUIRE(cgltf_accessor_read_float(
                    position, local_indices[corner], triangle_positions[corner], 3u));
            }
            const float ax = triangle_positions[1][0] - triangle_positions[0][0];
            const float ay = triangle_positions[1][1] - triangle_positions[0][1];
            const float bx = triangle_positions[2][0] - triangle_positions[0][0];
            const float by = triangle_positions[2][1] - triangle_positions[0][1];
            TEST_REQUIRE(ax * by - ay * bx > 0.0f);
        }
        uint32_t expected_index_min = UINT32_MAX;
        uint32_t expected_index_max = 0u;
        for (size_t triangle = 0; triangle < range->triangle_count; ++triangle) {
            for (size_t corner = 0; corner < 3u; ++corner) {
                const uint32_t local = asset->triangles[
                    (range->first_triangle + triangle) * 3u + corner] -
                    (uint32_t) range->first_vertex;
                if (local < expected_index_min) expected_index_min = local;
                if (local > expected_index_max) expected_index_max = local;
            }
        }
        TEST_REQUIRE(primitive->indices->has_min);
        TEST_REQUIRE(primitive->indices->has_max);
        TEST_REQUIRE(nearly_equal(
            primitive->indices->min[0], (float) expected_index_min, 0.0f));
        TEST_REQUIRE(nearly_equal(
            primitive->indices->max[0], (float) expected_index_max, 0.0f));
    }

    cgltf_free(data);
    return 1;

fail:
    cgltf_free(data);
    return 0;
}

static int test_invalid_inputs(
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_rigging_normalization * normalization,
    const trellis_tokenskin_skeleton * skeleton,
    const trellis_mesh_rigging_dense_skin_weights * weights) {
    char error[512];
    trellis_tokenskin_skeleton invalid_skeleton = *skeleton;
    int32_t cyclic_parents[] = { 4, 0, 0, 1, 1 };
    invalid_skeleton.parents = cyclic_parents;
    TEST_REQUIRE(trellis_mesh_rigging_write_rigged_glb(
        "test_tokenskin_invalid.glb",
        asset,
        normalization,
        &invalid_skeleton,
        weights,
        error,
        sizeof(error)) == TRELLIS_STATUS_INVALID_ARGUMENT);
    TEST_REQUIRE(strstr(error, "cycle") != NULL);

    float invalid_values[35];
    memcpy(invalid_values, weights->values, sizeof(invalid_values));
    invalid_values[7] = NAN;
    trellis_mesh_rigging_dense_skin_weights invalid_weights = *weights;
    invalid_weights.values = invalid_values;
    TEST_REQUIRE(trellis_mesh_rigging_write_rigged_glb(
        "test_tokenskin_invalid.glb",
        asset,
        normalization,
        skeleton,
        &invalid_weights,
        error,
        sizeof(error)) == TRELLIS_STATUS_INVALID_ARGUMENT);
    TEST_REQUIRE(strstr(error, "non-finite") != NULL);

    TEST_REQUIRE(trellis_mesh_rigging_write_rigged_glb(
        "test_tokenskin_invalid.gltf",
        asset,
        normalization,
        skeleton,
        weights,
        error,
        sizeof(error)) == TRELLIS_STATUS_INVALID_ARGUMENT);
    TEST_REQUIRE(strstr(error, ".glb") != NULL);
    remove("test_tokenskin_invalid.glb");
    return 1;

fail:
    remove("test_tokenskin_invalid.glb");
    return 0;
}

int main(void) {
    const char * direct_path = "test_tokenskin_rigged_direct.glb";
    const char * sampled_path = "test_tokenskin_rigged_sampled.glb";
    trellis_mesh_rigging_asset asset;
    trellis_mesh_rigging_normalization normalization;
    trellis_tokenskin_skeleton skeleton;
    make_asset(&asset);
    make_normalization(&normalization);
    make_skeleton(&skeleton);

    static const float direct_values[7][5] = {
        { -1.0f, 1.0f, 2.0f, 3.0f, 4.0f },
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f },
        { 5.0f, 0.0f, 0.0f, 0.0f, 0.0f },
        { 0.0f, 0.1f, 0.0f, 0.2f, 0.7f },
        { -2.0f, -1.0f, 0.0f, 4.0f, 0.0f },
    };
    trellis_mesh_rigging_dense_skin_weights direct_weights =
        TRELLIS_MESH_RIGGING_DENSE_SKIN_WEIGHTS_INIT;
    direct_weights.values = &direct_values[0][0];
    direct_weights.point_count = 7;
    direct_weights.joint_count = 5;

    char error[512];
    trellis_status status = trellis_mesh_rigging_write_rigged_glb(
        direct_path,
        &asset,
        &normalization,
        &skeleton,
        &direct_weights,
        error,
        sizeof(error));
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "direct rigged GLB write failed: %s\n", error);
        goto fail;
    }
    if (!validate_glb(
            direct_path, &asset, &normalization, &skeleton, 0)) {
        goto fail;
    }
    if (!test_invalid_inputs(
            &asset, &normalization, &skeleton, &direct_weights)) {
        goto fail;
    }

    static const float sample_positions[2][3] = {
        { -1.0f, -1.0f, 0.0f },
        { 1.0f, 1.0f, 0.5f },
    };
    static const float sampled_values[2][5] = {
        { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f },
        { 5.0f, 4.0f, 3.0f, 2.0f, 1.0f },
    };
    trellis_mesh_rigging_dense_skin_weights sampled_weights =
        TRELLIS_MESH_RIGGING_DENSE_SKIN_WEIGHTS_INIT;
    sampled_weights.values = &sampled_values[0][0];
    sampled_weights.sample_positions = &sample_positions[0][0];
    sampled_weights.point_count = 2;
    sampled_weights.joint_count = 5;
    sampled_weights.interpolation_neighbors = 1;
    status = trellis_mesh_rigging_write_rigged_glb(
        sampled_path,
        &asset,
        &normalization,
        &skeleton,
        &sampled_weights,
        error,
        sizeof(error));
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "sampled rigged GLB write failed: %s\n", error);
        goto fail;
    }
    if (!validate_glb(
            sampled_path, &asset, &normalization, &skeleton, 1)) {
        goto fail;
    }

    remove(sampled_path);
    remove(direct_path);
    puts("TokenSkin rigged GLB tests passed");
    return 0;

fail:
    remove(sampled_path);
    remove(direct_path);
    return 1;
}
