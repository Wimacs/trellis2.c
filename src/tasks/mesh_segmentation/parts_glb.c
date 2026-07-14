#include "parts_glb.h"

#include "cgltf.h"
#include "cgltf_write.h"
#include "trellis_platform.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct segmentation_part_geometry {
    float * positions;
    float * normals;
    uint32_t * indices;
    size_t vertex_count;
    size_t index_count;
    float bounds_min[3];
    float bounds_max[3];
} segmentation_part_geometry;

static void parts_set_error(
    char * output,
    size_t output_size,
    const char * message) {
    if (output == NULL || output_size == 0) return;
    snprintf(output, output_size, "%s", message != NULL ? message : "unknown error");
}

static int parts_has_glb_extension(const char * path) {
    if (path == NULL) return 0;
    const size_t length = strlen(path);
    if (length < 4) return 0;
    const char * extension = path + length - 4;
    return extension[0] == '.' &&
        (extension[1] == 'g' || extension[1] == 'G') &&
        (extension[2] == 'l' || extension[2] == 'L') &&
        (extension[3] == 'b' || extension[3] == 'B');
}

/* Open a unique file beside the destination so the final rename cannot cross
 * filesystems. Keep the exclusive handle open through the entire write. */
static trellis_status parts_create_sibling_temporary(
    const char * path,
    char ** temporary_out,
    FILE ** file_out,
    char * error_out,
    size_t error_size) {
    if (path == NULL || temporary_out == NULL || file_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *temporary_out = NULL;
    *file_out = NULL;
    const size_t path_length = strlen(path);
    if (path_length > SIZE_MAX - 80u) {
        parts_set_error(error_out, error_size, "parts GLB output path is too long");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t capacity = path_length + 80u;
    char * temporary = (char *) malloc(capacity);
    if (temporary == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;

    for (int attempt = 0; attempt < 100; ++attempt) {
        const int written = snprintf(
            temporary,
            capacity,
            "%s.trellis-tmp-%ld-%d.glb",
            path,
            trellis_getpid(),
            attempt);
        if (written < 0 || (size_t) written >= capacity) {
            free(temporary);
            parts_set_error(error_out, error_size, "parts GLB temporary path is too long");
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
#ifdef _WIN32
        const int descriptor = _open(
            temporary,
            _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY,
            _S_IREAD | _S_IWRITE);
#else
        const int descriptor = open(temporary, O_CREAT | O_EXCL | O_RDWR, 0666);
#endif
        if (descriptor >= 0) {
#ifdef _WIN32
            FILE * file = _fdopen(descriptor, "wb");
#else
            FILE * file = fdopen(descriptor, "wb");
#endif
            if (file == NULL) {
#ifdef _WIN32
                _close(descriptor);
#else
                close(descriptor);
#endif
                trellis_unlink(temporary);
                free(temporary);
                parts_set_error(error_out, error_size, "failed to open parts GLB temporary stream");
                return TRELLIS_STATUS_IO_ERROR;
            }
            *temporary_out = temporary;
            *file_out = file;
            return TRELLIS_STATUS_OK;
        }
        if (errno != EEXIST) {
            free(temporary);
            parts_set_error(error_out, error_size, "failed to create parts GLB temporary file");
            return TRELLIS_STATUS_IO_ERROR;
        }
    }
    free(temporary);
    parts_set_error(error_out, error_size, "could not reserve a parts GLB temporary file");
    return TRELLIS_STATUS_IO_ERROR;
}

static int parts_atomic_replace(const char * temporary, const char * path) {
#ifdef _WIN32
    return MoveFileExA(
        temporary,
        path,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(temporary, path) == 0;
#endif
}

static int add_size(size_t a, size_t b, size_t * output);

static int parts_write_exact(FILE * file, const void * data, size_t size) {
    return size == 0 || fwrite(data, 1, size, file) == size;
}

static int parts_write_u32_le(FILE * file, uint32_t value) {
    const unsigned char bytes[4] = {
        (unsigned char) value,
        (unsigned char) (value >> 8u),
        (unsigned char) (value >> 16u),
        (unsigned char) (value >> 24u),
    };
    return parts_write_exact(file, bytes, sizeof(bytes));
}

static trellis_status parts_write_glb_stream(
    FILE * file,
    const cgltf_options * options,
    const cgltf_data * data) {
    if (file == NULL || options == NULL || data == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const cgltf_size expected = cgltf_write(options, NULL, 0, data);
    if (expected == 0) return TRELLIS_STATUS_ERROR;
    char * json = (char *) malloc((size_t) expected);
    if (json == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    const cgltf_size actual = cgltf_write(options, json, expected, data);
    if (actual != expected || actual == 0) {
        free(json);
        return TRELLIS_STATUS_ERROR;
    }
    const size_t json_size = (size_t) actual - 1u;
    const size_t binary_size = (size_t) data->bin_size;
    const size_t json_padding = (4u - json_size % 4u) % 4u;
    const size_t binary_padding = (4u - binary_size % 4u) % 4u;
    size_t total_size = 12u;
    size_t json_chunk_size = 0;
    size_t binary_chunk_size = 0;
    int sizes_ok =
        add_size(json_size, json_padding, &json_chunk_size) &&
        add_size(total_size, 8u, &total_size) &&
        add_size(total_size, json_chunk_size, &total_size);
    if (sizes_ok && data->bin != NULL && binary_size > 0) {
        sizes_ok = add_size(binary_size, binary_padding, &binary_chunk_size) &&
            add_size(total_size, 8u, &total_size) &&
            add_size(total_size, binary_chunk_size, &total_size);
    }
    if (!sizes_ok || total_size > UINT32_MAX ||
        json_chunk_size > UINT32_MAX || binary_chunk_size > UINT32_MAX) {
        free(json);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    static const unsigned char glb_magic[4] = {'g', 'l', 'T', 'F'};
    static const unsigned char json_magic[4] = {'J', 'S', 'O', 'N'};
    static const unsigned char binary_magic[4] = {'B', 'I', 'N', 0};
    static const unsigned char json_pad[3] = {' ', ' ', ' '};
    static const unsigned char binary_pad[3] = {0, 0, 0};
    int ok = parts_write_exact(file, glb_magic, sizeof(glb_magic)) &&
        parts_write_u32_le(file, 2u) &&
        parts_write_u32_le(file, (uint32_t) total_size) &&
        parts_write_u32_le(file, (uint32_t) json_chunk_size) &&
        parts_write_exact(file, json_magic, sizeof(json_magic)) &&
        parts_write_exact(file, json, json_size) &&
        parts_write_exact(file, json_pad, json_padding);
    if (ok && data->bin != NULL && binary_size > 0) {
        ok = parts_write_u32_le(file, (uint32_t) binary_chunk_size) &&
            parts_write_exact(file, binary_magic, sizeof(binary_magic)) &&
            parts_write_exact(file, data->bin, binary_size) &&
            parts_write_exact(file, binary_pad, binary_padding);
    }
    free(json);
    if (!ok || fflush(file) != 0) return TRELLIS_STATUS_IO_ERROR;
    return TRELLIS_STATUS_OK;
}

static int add_size(size_t a, size_t b, size_t * output) {
    if (output == NULL || a > SIZE_MAX - b) return 0;
    *output = a + b;
    return 1;
}

static int multiply_size(size_t a, size_t b, size_t * output) {
    if (output == NULL || (a != 0 && b > SIZE_MAX / a)) return 0;
    *output = a * b;
    return 1;
}

static int align4_size(size_t value, size_t * output) {
    if (value > SIZE_MAX - 3u) return 0;
    *output = (value + 3u) & ~(size_t) 3u;
    return 1;
}

static void part_geometry_free(segmentation_part_geometry * geometry) {
    if (geometry == NULL) return;
    free(geometry->positions);
    free(geometry->normals);
    free(geometry->indices);
    memset(geometry, 0, sizeof(*geometry));
}

static void normalize_part_normals(segmentation_part_geometry * geometry) {
    for (size_t vertex = 0; vertex < geometry->vertex_count; ++vertex) {
        float * normal = geometry->normals + vertex * 3u;
        const float length = sqrtf(
            normal[0] * normal[0] +
            normal[1] * normal[1] +
            normal[2] * normal[2]);
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

static trellis_status build_part_geometry(
    const trellis_mesh_rigging_asset * asset,
    const uint32_t * face_part_ids,
    uint32_t part_id,
    size_t part_faces,
    uint32_t * vertex_map,
    uint32_t * vertex_stamps,
    uint32_t stamp,
    segmentation_part_geometry * geometry) {
    if (asset == NULL || face_part_ids == NULL || vertex_map == NULL ||
        vertex_stamps == NULL || geometry == NULL || part_faces == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(geometry, 0, sizeof(*geometry));
    size_t max_vertices = 0;
    size_t vertex_values = 0;
    size_t vertex_bytes = 0;
    size_t index_bytes = 0;
    if (!multiply_size(part_faces, 3u, &max_vertices) ||
        !multiply_size(max_vertices, 3u, &vertex_values) ||
        !multiply_size(vertex_values, sizeof(float), &vertex_bytes) ||
        !multiply_size(max_vertices, sizeof(uint32_t), &index_bytes) ||
        max_vertices > UINT32_MAX) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    geometry->positions = (float *) malloc(vertex_bytes);
    geometry->normals = (float *) calloc(vertex_values, sizeof(float));
    geometry->indices = (uint32_t *) malloc(index_bytes);
    if (geometry->positions == NULL || geometry->normals == NULL ||
        geometry->indices == NULL) {
        part_geometry_free(geometry);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    for (size_t face = 0; face < asset->triangle_count; ++face) {
        if (face_part_ids[face] != part_id) continue;
        uint32_t local[3];
        for (size_t corner = 0; corner < 3; ++corner) {
            const uint32_t source_vertex = asset->triangles[face * 3u + corner];
            if ((size_t) source_vertex >= asset->vertex_count) {
                part_geometry_free(geometry);
                return TRELLIS_STATUS_PARSE_ERROR;
            }
            if (vertex_stamps[source_vertex] != stamp) {
                if (geometry->vertex_count >= max_vertices ||
                    geometry->vertex_count > UINT32_MAX) {
                    part_geometry_free(geometry);
                    return TRELLIS_STATUS_OUT_OF_MEMORY;
                }
                vertex_stamps[source_vertex] = stamp;
                vertex_map[source_vertex] = (uint32_t) geometry->vertex_count;
                float * destination = geometry->positions + geometry->vertex_count * 3u;
                const float * source = asset->positions + (size_t) source_vertex * 3u;
                memcpy(destination, source, 3u * sizeof(float));
                for (int axis = 0; axis < 3; ++axis) {
                    if (!isfinite(destination[axis])) {
                        part_geometry_free(geometry);
                        return TRELLIS_STATUS_PARSE_ERROR;
                    }
                    if (geometry->vertex_count == 0 ||
                        destination[axis] < geometry->bounds_min[axis]) {
                        geometry->bounds_min[axis] = destination[axis];
                    }
                    if (geometry->vertex_count == 0 ||
                        destination[axis] > geometry->bounds_max[axis]) {
                        geometry->bounds_max[axis] = destination[axis];
                    }
                }
                ++geometry->vertex_count;
            }
            local[corner] = vertex_map[source_vertex];
            geometry->indices[geometry->index_count++] = local[corner];
        }
        const float * a = geometry->positions + (size_t) local[0] * 3u;
        const float * b = geometry->positions + (size_t) local[1] * 3u;
        const float * c = geometry->positions + (size_t) local[2] * 3u;
        const float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const float ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const float normal[3] = {
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        };
        for (size_t corner = 0; corner < 3; ++corner) {
            float * destination = geometry->normals + (size_t) local[corner] * 3u;
            destination[0] += normal[0];
            destination[1] += normal[1];
            destination[2] += normal[2];
        }
    }
    if (geometry->index_count != part_faces * 3u || geometry->vertex_count == 0) {
        part_geometry_free(geometry);
        return TRELLIS_STATUS_ERROR;
    }
    normalize_part_normals(geometry);
    return TRELLIS_STATUS_OK;
}

static float hue_channel(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 0.5f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}

static void part_color(size_t part, float output[4]) {
    const float hue = fmodf(0.07f + (float) part * 0.61803398875f, 1.0f);
    const float saturation = 0.66f;
    const float lightness = 0.58f;
    const float q = lightness < 0.5f ?
        lightness * (1.0f + saturation) :
        lightness + saturation - lightness * saturation;
    const float p = 2.0f * lightness - q;
    output[0] = hue_channel(p, q, hue + 1.0f / 3.0f);
    output[1] = hue_channel(p, q, hue);
    output[2] = hue_channel(p, q, hue - 1.0f / 3.0f);
    output[3] = 1.0f;
}

static trellis_status verify_parts_glb(
    const char * path,
    size_t expected_parts,
    size_t expected_faces,
    char * error_out,
    size_t error_size) {
    cgltf_options options;
    memset(&options, 0, sizeof(options));
    cgltf_data * data = NULL;
    cgltf_result result = cgltf_parse_file(&options, path, &data);
    if (result != cgltf_result_success || data == NULL) {
        parts_set_error(error_out, error_size, "failed to parse written parts GLB");
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    result = cgltf_load_buffers(&options, data, path);
    if (result == cgltf_result_success) result = cgltf_validate(data);
    if (result != cgltf_result_success || data->scene == NULL ||
        data->meshes_count != expected_parts ||
        data->scene->nodes_count != 1 ||
        data->scene->nodes[0] == NULL ||
        data->scene->nodes[0]->children_count != expected_parts) {
        cgltf_free(data);
        parts_set_error(error_out, error_size, "written parts GLB failed structural validation");
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    size_t faces = 0;
    for (size_t part = 0; part < expected_parts; ++part) {
        const cgltf_node * node = data->scene->nodes[0]->children[part];
        if (node == NULL || node->mesh == NULL ||
            node->mesh->primitives_count != 1 ||
            node->mesh->primitives[0].indices == NULL ||
            node->mesh->primitives[0].indices->count % 3u != 0) {
            cgltf_free(data);
            parts_set_error(error_out, error_size, "written GLB contains an invalid part node");
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        const size_t part_faces =
            (size_t) node->mesh->primitives[0].indices->count / 3u;
        if (!add_size(faces, part_faces, &faces)) {
            cgltf_free(data);
            parts_set_error(error_out, error_size, "written GLB face count overflowed");
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    cgltf_free(data);
    if (faces != expected_faces) {
        parts_set_error(error_out, error_size, "written GLB lost or duplicated faces");
        return TRELLIS_STATUS_ERROR;
    }
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_mesh_segmentation_write_parts_glb(
    const char * path,
    const trellis_mesh_rigging_asset * asset,
    const uint32_t * face_part_ids,
    size_t face_count,
    size_t part_count,
    const char * const * part_names,
    char * error_out,
    size_t error_size) {
    if (error_out != NULL && error_size > 0) error_out[0] = '\0';
    if (!parts_has_glb_extension(path) || asset == NULL ||
        asset->positions == NULL || asset->triangles == NULL ||
        asset->vertex_count == 0 || asset->triangle_count == 0 ||
        face_part_ids == NULL || face_count != asset->triangle_count ||
        part_count == 0 || part_count >= UINT32_MAX ||
        part_count > SIZE_MAX / sizeof(size_t) ||
        part_count > SIZE_MAX / sizeof(segmentation_part_geometry) ||
        part_count > SIZE_MAX / (3u * sizeof(size_t)) ||
        asset->vertex_count > SIZE_MAX / sizeof(uint32_t)) {
        parts_set_error(error_out, error_size, "invalid parts GLB request");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    size_t triple_parts = 0;
    size_t double_parts = 0;
    size_t node_count = 0;
    size_t allocation_bytes = 0;
    if (!multiply_size(part_count, 3u, &triple_parts) ||
        !multiply_size(part_count, 2u, &double_parts) ||
        !add_size(part_count, 1u, &node_count) ||
        !multiply_size(triple_parts, sizeof(cgltf_buffer_view), &allocation_bytes) ||
        !multiply_size(triple_parts, sizeof(cgltf_accessor), &allocation_bytes) ||
        !multiply_size(double_parts, sizeof(cgltf_attribute), &allocation_bytes) ||
        !multiply_size(part_count, sizeof(cgltf_primitive), &allocation_bytes) ||
        !multiply_size(part_count, sizeof(cgltf_mesh), &allocation_bytes) ||
        !multiply_size(part_count, sizeof(cgltf_material), &allocation_bytes) ||
        !multiply_size(node_count, sizeof(cgltf_node), &allocation_bytes) ||
        !multiply_size(part_count, sizeof(cgltf_node *), &allocation_bytes) ||
        !multiply_size(part_count, 64u, &allocation_bytes)) {
        parts_set_error(error_out, error_size, "parts GLB allocation size overflowed");
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_status status = TRELLIS_STATUS_ERROR;
    size_t * part_faces = (size_t *) calloc(part_count, sizeof(size_t));
    segmentation_part_geometry * geometries =
        (segmentation_part_geometry *) calloc(part_count, sizeof(*geometries));
    uint32_t * vertex_map =
        (uint32_t *) malloc(asset->vertex_count * sizeof(uint32_t));
    uint32_t * vertex_stamps =
        (uint32_t *) calloc(asset->vertex_count, sizeof(uint32_t));
    if (part_faces == NULL || geometries == NULL || vertex_map == NULL ||
        vertex_stamps == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (size_t face = 0; face < face_count; ++face) {
        if ((size_t) face_part_ids[face] >= part_count) {
            parts_set_error(error_out, error_size, "face part id exceeds part_count");
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
        ++part_faces[face_part_ids[face]];
    }
    for (size_t part = 0; part < part_count; ++part) {
        if (part_faces[part] == 0) {
            parts_set_error(error_out, error_size, "part ids must be dense and non-empty");
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
        status = build_part_geometry(
            asset,
            face_part_ids,
            (uint32_t) part,
            part_faces[part],
            vertex_map,
            vertex_stamps,
            (uint32_t) part + 1u,
            &geometries[part]);
        if (status != TRELLIS_STATUS_OK) {
            parts_set_error(error_out, error_size, "failed to build compact part geometry");
            goto cleanup;
        }
    }

    size_t binary_size = 0;
    size_t * offsets = (size_t *) calloc(triple_parts, sizeof(size_t));
    size_t * sizes = (size_t *) calloc(triple_parts, sizeof(size_t));
    if (offsets == NULL || sizes == NULL) {
        free(offsets);
        free(sizes);
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (size_t part = 0; part < part_count; ++part) {
        size_t index_bytes = 0;
        size_t vertex_bytes = 0;
        size_t vertex_values = 0;
        if (!multiply_size(geometries[part].index_count, sizeof(uint32_t), &index_bytes) ||
            !multiply_size(geometries[part].vertex_count, 3u, &vertex_values) ||
            !multiply_size(vertex_values, sizeof(float), &vertex_bytes)) {
            free(offsets);
            free(sizes);
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        size_t aligned = 0;
        if (!align4_size(binary_size, &aligned)) {
            free(offsets);
            free(sizes);
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        binary_size = aligned;
        offsets[part * 3u] = binary_size;
        sizes[part * 3u] = index_bytes;
        if (!add_size(binary_size, index_bytes, &binary_size) ||
            !align4_size(binary_size, &binary_size)) {
            free(offsets);
            free(sizes);
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        offsets[part * 3u + 1u] = binary_size;
        sizes[part * 3u + 1u] = vertex_bytes;
        if (!add_size(binary_size, vertex_bytes, &binary_size) ||
            !align4_size(binary_size, &binary_size)) {
            free(offsets);
            free(sizes);
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        offsets[part * 3u + 2u] = binary_size;
        sizes[part * 3u + 2u] = vertex_bytes;
        if (!add_size(binary_size, vertex_bytes, &binary_size)) {
            free(offsets);
            free(sizes);
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
    }
    if (!align4_size(binary_size, &binary_size) || binary_size == 0) {
        free(offsets);
        free(sizes);
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    unsigned char * binary = (unsigned char *) calloc(binary_size, 1u);
    if (binary == NULL) {
        free(offsets);
        free(sizes);
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (size_t part = 0; part < part_count; ++part) {
        memcpy(binary + offsets[part * 3u], geometries[part].indices, sizes[part * 3u]);
        memcpy(binary + offsets[part * 3u + 1u], geometries[part].positions, sizes[part * 3u + 1u]);
        memcpy(binary + offsets[part * 3u + 2u], geometries[part].normals, sizes[part * 3u + 2u]);
    }

    cgltf_buffer_view * views = (cgltf_buffer_view *) calloc(triple_parts, sizeof(*views));
    cgltf_accessor * accessors = (cgltf_accessor *) calloc(triple_parts, sizeof(*accessors));
    cgltf_attribute * attributes = (cgltf_attribute *) calloc(double_parts, sizeof(*attributes));
    cgltf_primitive * primitives = (cgltf_primitive *) calloc(part_count, sizeof(*primitives));
    cgltf_mesh * meshes = (cgltf_mesh *) calloc(part_count, sizeof(*meshes));
    cgltf_material * materials = (cgltf_material *) calloc(part_count, sizeof(*materials));
    cgltf_node * nodes = (cgltf_node *) calloc(node_count, sizeof(*nodes));
    cgltf_node ** children = (cgltf_node **) calloc(part_count, sizeof(*children));
    char (*names)[64] = (char (*)[64]) calloc(part_count, sizeof(*names));
    char (*extras)[64] = (char (*)[64]) calloc(part_count, sizeof(*extras));
    if (views == NULL || accessors == NULL || attributes == NULL ||
        primitives == NULL || meshes == NULL || materials == NULL ||
        nodes == NULL || children == NULL || names == NULL || extras == NULL) {
        free(binary); free(offsets); free(sizes); free(views); free(accessors);
        free(attributes); free(primitives); free(meshes); free(materials);
        free(nodes); free(children); free(names); free(extras);
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    cgltf_data data;
    memset(&data, 0, sizeof(data));
    data.file_type = cgltf_file_type_glb;
    data.asset.version = (char *) "2.0";
    data.asset.generator = (char *) "trellis2.c SegviGen parts";
    data.bin = binary;
    data.bin_size = binary_size;
    cgltf_buffer buffer;
    memset(&buffer, 0, sizeof(buffer));
    buffer.size = binary_size;
    data.buffers = &buffer;
    data.buffers_count = 1;

    for (size_t part = 0; part < part_count; ++part) {
        for (size_t item = 0; item < 3; ++item) {
            const size_t index = part * 3u + item;
            views[index].buffer = &buffer;
            views[index].offset = offsets[index];
            views[index].size = sizes[index];
            views[index].type = item == 0 ?
                cgltf_buffer_view_type_indices : cgltf_buffer_view_type_vertices;
            if (item != 0) views[index].stride = 3u * sizeof(float);
        }
        cgltf_accessor * index_accessor = &accessors[part * 3u];
        index_accessor->buffer_view = &views[part * 3u];
        index_accessor->component_type = cgltf_component_type_r_32u;
        index_accessor->type = cgltf_type_scalar;
        index_accessor->count = geometries[part].index_count;
        cgltf_accessor * position_accessor = &accessors[part * 3u + 1u];
        position_accessor->buffer_view = &views[part * 3u + 1u];
        position_accessor->component_type = cgltf_component_type_r_32f;
        position_accessor->type = cgltf_type_vec3;
        position_accessor->count = geometries[part].vertex_count;
        position_accessor->has_min = position_accessor->has_max = 1;
        memcpy(position_accessor->min, geometries[part].bounds_min, 3u * sizeof(float));
        memcpy(position_accessor->max, geometries[part].bounds_max, 3u * sizeof(float));
        cgltf_accessor * normal_accessor = &accessors[part * 3u + 2u];
        normal_accessor->buffer_view = &views[part * 3u + 2u];
        normal_accessor->component_type = cgltf_component_type_r_32f;
        normal_accessor->type = cgltf_type_vec3;
        normal_accessor->count = geometries[part].vertex_count;

        attributes[part * 2u].name = (char *) "POSITION";
        attributes[part * 2u].type = cgltf_attribute_type_position;
        attributes[part * 2u].data = position_accessor;
        attributes[part * 2u + 1u].name = (char *) "NORMAL";
        attributes[part * 2u + 1u].type = cgltf_attribute_type_normal;
        attributes[part * 2u + 1u].data = normal_accessor;

        snprintf(names[part], sizeof(names[part]), "part_%04u", (unsigned) part);
        snprintf(extras[part], sizeof(extras[part]), "{\"trellis_part_id\":%u}", (unsigned) part);
        materials[part].name = names[part];
        materials[part].has_pbr_metallic_roughness = 1;
        part_color(part, materials[part].pbr_metallic_roughness.base_color_factor);
        materials[part].pbr_metallic_roughness.metallic_factor = 0.0f;
        materials[part].pbr_metallic_roughness.roughness_factor = 0.72f;
        materials[part].alpha_mode = cgltf_alpha_mode_opaque;
        materials[part].double_sided = 1;

        primitives[part].type = cgltf_primitive_type_triangles;
        primitives[part].indices = index_accessor;
        primitives[part].material = &materials[part];
        primitives[part].attributes = &attributes[part * 2u];
        primitives[part].attributes_count = 2;
        meshes[part].name = names[part];
        meshes[part].primitives = &primitives[part];
        meshes[part].primitives_count = 1;
        nodes[part + 1u].name = (char *) (
            part_names != NULL && part_names[part] != NULL && part_names[part][0] != '\0' ?
                part_names[part] : names[part]);
        nodes[part + 1u].mesh = &meshes[part];
        nodes[part + 1u].parent = &nodes[0];
        nodes[part + 1u].extras.data = extras[part];
        children[part] = &nodes[part + 1u];
    }
    nodes[0].name = (char *) "trellis_parts";
    nodes[0].children = children;
    nodes[0].children_count = part_count;
    data.buffer_views = views;
    data.buffer_views_count = triple_parts;
    data.accessors = accessors;
    data.accessors_count = triple_parts;
    data.materials = materials;
    data.materials_count = part_count;
    data.meshes = meshes;
    data.meshes_count = part_count;
    data.nodes = nodes;
    data.nodes_count = node_count;
    cgltf_node * scene_nodes[1] = {&nodes[0]};
    cgltf_scene scene;
    memset(&scene, 0, sizeof(scene));
    scene.name = (char *) "SegviGen Parts";
    scene.nodes = scene_nodes;
    scene.nodes_count = 1;
    data.scenes = &scene;
    data.scenes_count = 1;
    data.scene = &scene;

    char * temporary = NULL;
    FILE * temporary_file = NULL;
    status = parts_create_sibling_temporary(
        path, &temporary, &temporary_file, error_out, error_size);
    if (status == TRELLIS_STATUS_OK) {
        cgltf_options write_options;
        memset(&write_options, 0, sizeof(write_options));
        write_options.type = cgltf_file_type_glb;
        status = parts_write_glb_stream(
            temporary_file, &write_options, &data);
        if (fclose(temporary_file) != 0 && status == TRELLIS_STATUS_OK) {
            status = TRELLIS_STATUS_IO_ERROR;
        }
        temporary_file = NULL;
        if (status != TRELLIS_STATUS_OK) {
            parts_set_error(error_out, error_size, "cgltf failed to write parts GLB");
        } else {
            status = verify_parts_glb(
                temporary, part_count, face_count, error_out, error_size);
            if (status == TRELLIS_STATUS_OK &&
                !parts_atomic_replace(temporary, path)) {
                parts_set_error(error_out, error_size, "failed to atomically replace parts GLB");
                status = TRELLIS_STATUS_IO_ERROR;
            }
        }
        if (status != TRELLIS_STATUS_OK) trellis_unlink(temporary);
        free(temporary);
    }

    free(binary); free(offsets); free(sizes); free(views); free(accessors);
    free(attributes); free(primitives); free(meshes); free(materials);
    free(nodes); free(children); free(names); free(extras);

cleanup:
    if (geometries != NULL) {
        for (size_t part = 0; part < part_count; ++part) {
            part_geometry_free(&geometries[part]);
        }
    }
    free(part_faces);
    free(geometries);
    free(vertex_map);
    free(vertex_stamps);
    return status;
}
