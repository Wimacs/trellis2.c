#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include "gltf_io.h"

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char * output, size_t output_size, const char * format, ...) {
    if (output == NULL || output_size == 0) {
        return;
    }
    va_list args;
    va_start(args, format);
    vsnprintf(output, output_size, format, args);
    va_end(args);
    output[output_size - 1] = '\0';
}

static char * duplicate_string(const char * value) {
    if (value == NULL) {
        return NULL;
    }
    const size_t length = strlen(value) + 1u;
    char * copy = (char *) malloc(length);
    if (copy != NULL) {
        memcpy(copy, value, length);
    }
    return copy;
}

static int add_size(size_t a, size_t b, size_t * result) {
    if (a > SIZE_MAX - b) {
        return 0;
    }
    *result = a + b;
    return 1;
}

static int multiply_size(size_t a, size_t b, size_t * result) {
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *result = a * b;
    return 1;
}

static const char * cgltf_result_name(cgltf_result result) {
    switch (result) {
        case cgltf_result_success: return "success";
        case cgltf_result_data_too_short: return "data too short";
        case cgltf_result_unknown_format: return "unknown format";
        case cgltf_result_invalid_json: return "invalid JSON";
        case cgltf_result_invalid_gltf: return "invalid glTF";
        case cgltf_result_invalid_options: return "invalid options";
        case cgltf_result_file_not_found: return "file not found";
        case cgltf_result_io_error: return "I/O error";
        case cgltf_result_out_of_memory: return "out of memory";
        case cgltf_result_legacy_gltf: return "legacy glTF";
        default: return "unknown error";
    }
}

static trellis_status status_from_cgltf(cgltf_result result) {
    if (result == cgltf_result_out_of_memory) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    if (result == cgltf_result_file_not_found || result == cgltf_result_io_error) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    return TRELLIS_STATUS_PARSE_ERROR;
}

static int primitive_is_triangular(cgltf_primitive_type type) {
    return type == cgltf_primitive_type_triangles ||
        type == cgltf_primitive_type_triangle_strip ||
        type == cgltf_primitive_type_triangle_fan;
}

static int node_is_in_selected_scene(const cgltf_data * data, const cgltf_node * node) {
    const cgltf_scene * scene = data->scene;
    if (scene == NULL && data->scenes_count == 1) {
        scene = &data->scenes[0];
    }
    if (scene == NULL) {
        return 1;
    }
    const cgltf_node * root = node;
    while (root->parent != NULL) {
        root = root->parent;
    }
    for (cgltf_size i = 0; i < scene->nodes_count; ++i) {
        if (scene->nodes[i] == root) {
            return 1;
        }
    }
    return 0;
}

static trellis_status primitive_counts(
    const cgltf_primitive * primitive,
    size_t * vertex_count,
    size_t * triangle_count,
    char * error,
    size_t error_size) {
    const cgltf_accessor * position =
        cgltf_find_accessor(primitive, cgltf_attribute_type_position, 0);
    if (position == NULL) {
        set_error(error, error_size, "triangle primitive has no POSITION accessor");
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    if (position->type != cgltf_type_vec3 || position->count == 0) {
        set_error(error, error_size, "POSITION accessor must be a non-empty VEC3");
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    const size_t indices = primitive->indices != NULL ?
        (size_t) primitive->indices->count : (size_t) position->count;
    size_t triangles = 0;
    if (primitive->type == cgltf_primitive_type_triangles) {
        if (indices == 0 || indices % 3u != 0) {
            set_error(error, error_size, "triangle primitive index count %zu is not divisible by 3", indices);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        triangles = indices / 3u;
    } else {
        if (indices < 3u) {
            set_error(error, error_size, "triangle strip/fan needs at least three indices");
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        triangles = indices - 2u;
    }
    *vertex_count = (size_t) position->count;
    *triangle_count = triangles;
    return TRELLIS_STATUS_OK;
}

static int matrix_is_finite(const float matrix[16]) {
    for (int i = 0; i < 16; ++i) {
        if (!isfinite(matrix[i])) {
            return 0;
        }
    }
    return 1;
}

static void transform_position(const float matrix[16], const float input[3], float output[3]) {
    const float x = input[0];
    const float y = input[1];
    const float z = input[2];
    output[0] = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    output[1] = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    output[2] = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
}

static float matrix_linear_determinant(const float matrix[16]) {
    return
        matrix[0] * (matrix[5] * matrix[10] - matrix[9] * matrix[6]) -
        matrix[4] * (matrix[1] * matrix[10] - matrix[9] * matrix[2]) +
        matrix[8] * (matrix[1] * matrix[6] - matrix[5] * matrix[2]);
}

static trellis_status read_primitive_index(
    const cgltf_primitive * primitive,
    size_t position_count,
    size_t element,
    uint32_t * index_out,
    char * error,
    size_t error_size) {
    const size_t index = primitive->indices != NULL ?
        (size_t) cgltf_accessor_read_index(primitive->indices, (cgltf_size) element) : element;
    if (index >= position_count || index > UINT32_MAX) {
        set_error(
            error,
            error_size,
            "primitive index %zu at element %zu exceeds POSITION count %zu",
            index,
            element,
            position_count);
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    *index_out = (uint32_t) index;
    return TRELLIS_STATUS_OK;
}

static trellis_status append_primitive_triangles(
    const cgltf_primitive * primitive,
    size_t position_count,
    size_t vertex_base,
    int flip_winding,
    uint32_t * triangles,
    size_t triangle_base,
    size_t triangle_count,
    char * error,
    size_t error_size) {
    for (size_t triangle = 0; triangle < triangle_count; ++triangle) {
        size_t elements[3];
        if (primitive->type == cgltf_primitive_type_triangles) {
            elements[0] = triangle * 3u;
            elements[1] = elements[0] + 1u;
            elements[2] = elements[0] + 2u;
        } else if (primitive->type == cgltf_primitive_type_triangle_strip) {
            elements[0] = triangle;
            elements[1] = triangle + 1u;
            elements[2] = triangle + 2u;
            if ((triangle & 1u) != 0) {
                const size_t temporary = elements[0];
                elements[0] = elements[1];
                elements[1] = temporary;
            }
        } else {
            elements[0] = 0;
            elements[1] = triangle + 1u;
            elements[2] = triangle + 2u;
        }

        uint32_t local[3];
        for (int corner = 0; corner < 3; ++corner) {
            trellis_status status = read_primitive_index(
                primitive,
                position_count,
                elements[corner],
                &local[corner],
                error,
                error_size);
            if (status != TRELLIS_STATUS_OK) {
                return status;
            }
        }
        if (flip_winding) {
            const uint32_t temporary = local[1];
            local[1] = local[2];
            local[2] = temporary;
        }
        for (int corner = 0; corner < 3; ++corner) {
            const size_t global = vertex_base + (size_t) local[corner];
            if (global > UINT32_MAX) {
                set_error(error, error_size, "flattened vertex index exceeds uint32 range");
                return TRELLIS_STATUS_PARSE_ERROR;
            }
            triangles[(triangle_base + triangle) * 3u + (size_t) corner] = (uint32_t) global;
        }
    }
    return TRELLIS_STATUS_OK;
}

static void compute_normals_and_bounds(trellis_mesh_rigging_asset * asset) {
    for (int axis = 0; axis < 3; ++axis) {
        asset->aabb_min[axis] = FLT_MAX;
        asset->aabb_max[axis] = -FLT_MAX;
    }
    for (size_t vertex = 0; vertex < asset->vertex_count; ++vertex) {
        const float * position = asset->positions + vertex * 3u;
        for (int axis = 0; axis < 3; ++axis) {
            if (position[axis] < asset->aabb_min[axis]) asset->aabb_min[axis] = position[axis];
            if (position[axis] > asset->aabb_max[axis]) asset->aabb_max[axis] = position[axis];
        }
    }

    memset(asset->normals, 0, asset->vertex_count * 3u * sizeof(float));
    for (size_t triangle = 0; triangle < asset->triangle_count; ++triangle) {
        const uint32_t * indices = asset->triangles + triangle * 3u;
        const float * a = asset->positions + (size_t) indices[0] * 3u;
        const float * b = asset->positions + (size_t) indices[1] * 3u;
        const float * c = asset->positions + (size_t) indices[2] * 3u;
        const float ab[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
        const float ac[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
        const float cross[3] = {
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        };
        const float length = sqrtf(
            cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
        float * face_normal = asset->face_normals + triangle * 3u;
        if (length > 1e-20f && isfinite(length)) {
            face_normal[0] = cross[0] / length;
            face_normal[1] = cross[1] / length;
            face_normal[2] = cross[2] / length;
        } else {
            face_normal[0] = 0.0f;
            face_normal[1] = 1.0f;
            face_normal[2] = 0.0f;
        }
        for (int corner = 0; corner < 3; ++corner) {
            float * normal = asset->normals + (size_t) indices[corner] * 3u;
            normal[0] += cross[0];
            normal[1] += cross[1];
            normal[2] += cross[2];
        }
    }
    for (size_t vertex = 0; vertex < asset->vertex_count; ++vertex) {
        float * normal = asset->normals + vertex * 3u;
        const float length = sqrtf(
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        if (length > 1e-20f && isfinite(length)) {
            normal[0] /= length;
            normal[1] /= length;
            normal[2] /= length;
        } else {
            normal[0] = 0.0f;
            normal[1] = 1.0f;
            normal[2] = 0.0f;
        }
    }
}

void trellis_mesh_rigging_asset_free(trellis_mesh_rigging_asset * asset) {
    if (asset == NULL) {
        return;
    }
    free(asset->source_path);
    free(asset->positions);
    free(asset->normals);
    free(asset->face_normals);
    free(asset->triangles);
    free(asset->primitives);
    *asset = (trellis_mesh_rigging_asset) TRELLIS_MESH_RIGGING_ASSET_INIT;
}

trellis_status trellis_mesh_rigging_gltf_load(
    const char * path,
    trellis_mesh_rigging_asset * asset_out,
    char * error_out,
    size_t error_size) {
    if (error_out != NULL && error_size > 0) {
        error_out[0] = '\0';
    }
    if (path == NULL || path[0] == '\0' || asset_out == NULL) {
        set_error(error_out, error_size, "path and asset_out are required");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    cgltf_options options;
    memset(&options, 0, sizeof(options));
    cgltf_data * data = NULL;
    cgltf_result result = cgltf_parse_file(&options, path, &data);
    if (result != cgltf_result_success || data == NULL) {
        set_error(error_out, error_size, "failed to parse '%s': %s", path, cgltf_result_name(result));
        cgltf_free(data);
        return status_from_cgltf(result);
    }
    result = cgltf_load_buffers(&options, data, path);
    if (result != cgltf_result_success) {
        set_error(error_out, error_size, "failed to load buffers for '%s': %s", path, cgltf_result_name(result));
        cgltf_free(data);
        return status_from_cgltf(result);
    }
    result = cgltf_validate(data);
    if (result != cgltf_result_success) {
        set_error(error_out, error_size, "invalid glTF '%s': %s", path, cgltf_result_name(result));
        cgltf_free(data);
        return status_from_cgltf(result);
    }

    size_t vertex_count = 0;
    size_t triangle_count = 0;
    size_t primitive_count = 0;
    trellis_status status = TRELLIS_STATUS_OK;
    for (size_t node_index = 0; node_index < (size_t) data->nodes_count; ++node_index) {
        const cgltf_node * node = &data->nodes[node_index];
        if (node->mesh == NULL || !node_is_in_selected_scene(data, node)) {
            continue;
        }
        for (size_t primitive_index = 0;
             primitive_index < (size_t) node->mesh->primitives_count;
             ++primitive_index) {
            const cgltf_primitive * primitive = &node->mesh->primitives[primitive_index];
            if (!primitive_is_triangular(primitive->type)) {
                continue;
            }
            size_t primitive_vertices = 0;
            size_t primitive_triangles = 0;
            status = primitive_counts(
                primitive,
                &primitive_vertices,
                &primitive_triangles,
                error_out,
                error_size);
            if (status != TRELLIS_STATUS_OK ||
                !add_size(vertex_count, primitive_vertices, &vertex_count) ||
                !add_size(triangle_count, primitive_triangles, &triangle_count) ||
                !add_size(primitive_count, 1u, &primitive_count)) {
                if (status == TRELLIS_STATUS_OK) {
                    status = TRELLIS_STATUS_OUT_OF_MEMORY;
                    set_error(error_out, error_size, "flattened mesh counts overflow size_t");
                }
                cgltf_free(data);
                return status;
            }
        }
    }
    if (vertex_count == 0 || triangle_count == 0 || primitive_count == 0) {
        set_error(error_out, error_size, "'%s' has no node-instanced triangle mesh", path);
        cgltf_free(data);
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    if (vertex_count > UINT32_MAX) {
        set_error(error_out, error_size, "flattened mesh exceeds supported uint32 index range");
        cgltf_free(data);
        return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }

    size_t position_values = 0;
    size_t triangle_values = 0;
    if (!multiply_size(vertex_count, 3u, &position_values) ||
        !multiply_size(triangle_count, 3u, &triangle_values)) {
        set_error(error_out, error_size, "mesh allocation size overflow");
        cgltf_free(data);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    asset.source_path = duplicate_string(path);
    asset.positions = (float *) malloc(position_values * sizeof(float));
    asset.normals = (float *) malloc(position_values * sizeof(float));
    asset.face_normals = (float *) malloc(triangle_values * sizeof(float));
    asset.triangles = (uint32_t *) malloc(triangle_values * sizeof(uint32_t));
    asset.primitives = (trellis_mesh_rigging_primitive_range *) calloc(
        primitive_count,
        sizeof(*asset.primitives));
    asset.vertex_count = vertex_count;
    asset.triangle_count = triangle_count;
    asset.primitive_count = primitive_count;
    if (asset.source_path == NULL || asset.positions == NULL || asset.normals == NULL ||
        asset.face_normals == NULL || asset.triangles == NULL || asset.primitives == NULL) {
        set_error(error_out, error_size, "out of memory allocating flattened mesh");
        trellis_mesh_rigging_asset_free(&asset);
        cgltf_free(data);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    size_t vertex_cursor = 0;
    size_t triangle_cursor = 0;
    size_t primitive_cursor = 0;
    for (size_t node_index = 0; node_index < (size_t) data->nodes_count; ++node_index) {
        const cgltf_node * node = &data->nodes[node_index];
        if (node->mesh == NULL || !node_is_in_selected_scene(data, node)) {
            continue;
        }
        float world[16];
        cgltf_node_transform_world(node, world);
        if (!matrix_is_finite(world)) {
            set_error(error_out, error_size, "node %zu has a non-finite world transform", node_index);
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        const int flip_winding = matrix_linear_determinant(world) < 0.0f;
        for (size_t primitive_index = 0;
             primitive_index < (size_t) node->mesh->primitives_count;
             ++primitive_index) {
            const cgltf_primitive * primitive = &node->mesh->primitives[primitive_index];
            if (!primitive_is_triangular(primitive->type)) {
                continue;
            }
            const cgltf_accessor * position =
                cgltf_find_accessor(primitive, cgltf_attribute_type_position, 0);
            size_t primitive_vertices = 0;
            size_t primitive_triangles = 0;
            status = primitive_counts(
                primitive,
                &primitive_vertices,
                &primitive_triangles,
                error_out,
                error_size);
            if (status != TRELLIS_STATUS_OK) {
                goto cleanup;
            }
            for (size_t vertex = 0; vertex < primitive_vertices; ++vertex) {
                float local[3];
                if (!cgltf_accessor_read_float(position, (cgltf_size) vertex, local, 3)) {
                    set_error(
                        error_out,
                        error_size,
                        "failed to read POSITION %zu from node %zu primitive %zu",
                        vertex,
                        node_index,
                        primitive_index);
                    status = TRELLIS_STATUS_PARSE_ERROR;
                    goto cleanup;
                }
                float * transformed = asset.positions + (vertex_cursor + vertex) * 3u;
                transform_position(world, local, transformed);
                if (!isfinite(transformed[0]) || !isfinite(transformed[1]) || !isfinite(transformed[2])) {
                    set_error(error_out, error_size, "transformed POSITION is non-finite");
                    status = TRELLIS_STATUS_PARSE_ERROR;
                    goto cleanup;
                }
            }
            status = append_primitive_triangles(
                primitive,
                primitive_vertices,
                vertex_cursor,
                flip_winding,
                asset.triangles,
                triangle_cursor,
                primitive_triangles,
                error_out,
                error_size);
            if (status != TRELLIS_STATUS_OK) {
                goto cleanup;
            }

            trellis_mesh_rigging_primitive_range * range =
                &asset.primitives[primitive_cursor++];
            range->first_vertex = vertex_cursor;
            range->vertex_count = primitive_vertices;
            range->first_triangle = triangle_cursor;
            range->triangle_count = primitive_triangles;
            range->source_node_index = node_index;
            range->source_mesh_index = (size_t) (node->mesh - data->meshes);
            range->source_primitive_index = primitive_index;
            range->source_position_accessor_index = (size_t) (position - data->accessors);
            range->source_material_index = primitive->material != NULL ?
                (size_t) (primitive->material - data->materials) : SIZE_MAX;
            vertex_cursor += primitive_vertices;
            triangle_cursor += primitive_triangles;
        }
    }
    if (vertex_cursor != vertex_count || triangle_cursor != triangle_count ||
        primitive_cursor != primitive_count) {
        set_error(error_out, error_size, "internal flattened mesh count mismatch");
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }

    compute_normals_and_bounds(&asset);
    *asset_out = asset;
    cgltf_free(data);
    return TRELLIS_STATUS_OK;

cleanup:
    trellis_mesh_rigging_asset_free(&asset);
    cgltf_free(data);
    return status;
}
