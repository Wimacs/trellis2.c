#include "parts_glb.h"

#include "cgltf.h"
#include "cgltf_write.h"
#include "source_appearance.h"
#include "trellis_platform.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct segmentation_attribute_data {
    cgltf_attribute_type type;
    cgltf_type value_type;
    int index;
    size_t components;
    size_t source_stride;
    const float * source;
    float * values;
    size_t offset;
    size_t size;
} segmentation_attribute_data;

typedef struct segmentation_part_group {
    uint32_t part_id;
    size_t range_index;
    size_t face_count;
    uint32_t first_face;
    uint32_t last_face;
    uint32_t * indices;
    size_t index_count;
    size_t vertex_count;
    segmentation_attribute_data * attributes;
    size_t attribute_count;
    float bounds_min[3];
    float bounds_max[3];
    size_t index_offset;
    size_t index_size;
} segmentation_part_group;

typedef struct segmentation_group_table {
    segmentation_part_group * groups;
    size_t count;
    size_t capacity;
    size_t * slots; /* group index + 1; zero is empty. */
    size_t slot_count;
} segmentation_group_table;

static void parts_set_error(
    char * output,
    size_t output_size,
    const char * message) {
    if (output == NULL || output_size == 0) return;
    snprintf(output, output_size, "%s", message != NULL ? message : "unknown error");
}

static void parts_set_error_extension(
    char * output,
    size_t output_size,
    const char * owner,
    size_t index,
    const char * extension) {
    if (output == NULL || output_size == 0) return;
    snprintf(
        output,
        output_size,
        "%s %zu uses unsupported raw glTF extension '%s'",
        owner,
        index,
        extension != NULL ? extension : "unknown");
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

static int layout_region(size_t * cursor, size_t size, size_t * offset) {
    size_t aligned = 0;
    if (!align4_size(*cursor, &aligned) || !add_size(aligned, size, cursor)) return 0;
    *offset = aligned;
    return 1;
}

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

static size_t hash_pair(uint32_t part, size_t range) {
    uint64_t value = (uint64_t) part * UINT64_C(0x9e3779b185ebca87) ^
        (uint64_t) range * UINT64_C(0xc2b2ae3d27d4eb4f);
    value ^= value >> 33u;
    value *= UINT64_C(0xff51afd7ed558ccd);
    value ^= value >> 33u;
    return (size_t) value;
}

static void group_table_free(segmentation_group_table * table) {
    if (table == NULL) return;
    free(table->slots);
    free(table->groups);
    memset(table, 0, sizeof(*table));
}

static int group_table_rehash(segmentation_group_table * table, size_t slot_count) {
    size_t bytes = 0;
    if (slot_count < 16u || (slot_count & (slot_count - 1u)) != 0 ||
        !multiply_size(slot_count, sizeof(size_t), &bytes)) return 0;
    size_t * slots = (size_t *) calloc(slot_count, sizeof(size_t));
    if (slots == NULL) return 0;
    for (size_t group = 0; group < table->count; ++group) {
        size_t slot = hash_pair(
            table->groups[group].part_id,
            table->groups[group].range_index) & (slot_count - 1u);
        while (slots[slot] != 0) slot = (slot + 1u) & (slot_count - 1u);
        slots[slot] = group + 1u;
    }
    free(table->slots);
    table->slots = slots;
    table->slot_count = slot_count;
    return 1;
}

static int group_table_grow_groups(segmentation_group_table * table) {
    const size_t capacity = table->capacity == 0 ? 16u : table->capacity * 2u;
    if (capacity < table->capacity || capacity > SIZE_MAX / sizeof(*table->groups)) return 0;
    void * resized = realloc(table->groups, capacity * sizeof(*table->groups));
    if (resized == NULL) return 0;
    table->groups = (segmentation_part_group *) resized;
    table->capacity = capacity;
    return 1;
}

static int group_table_find_or_insert(
    segmentation_group_table * table,
    uint32_t part,
    size_t range,
    uint32_t face,
    uint32_t * face_next) {
    if (table->slot_count == 0 && !group_table_rehash(table, 16u)) return 0;
    if ((table->count + 1u) * 10u >= table->slot_count * 7u) {
        if (table->slot_count > SIZE_MAX / 2u ||
            !group_table_rehash(table, table->slot_count * 2u)) return 0;
    }
    size_t slot = hash_pair(part, range) & (table->slot_count - 1u);
    for (;;) {
        if (table->slots[slot] == 0) {
            if (table->count == table->capacity && !group_table_grow_groups(table)) return 0;
            const size_t group_index = table->count++;
            segmentation_part_group * group = &table->groups[group_index];
            memset(group, 0, sizeof(*group));
            group->part_id = part;
            group->range_index = range;
            group->face_count = 1u;
            group->first_face = face;
            group->last_face = face;
            table->slots[slot] = group_index + 1u;
            return 1;
        }
        segmentation_part_group * group = &table->groups[table->slots[slot] - 1u];
        if (group->part_id == part && group->range_index == range) {
            face_next[group->last_face] = face;
            group->last_face = face;
            ++group->face_count;
            return 1;
        }
        slot = (slot + 1u) & (table->slot_count - 1u);
    }
}

static int compare_groups(const void * left_value, const void * right_value) {
    const segmentation_part_group * left =
        (const segmentation_part_group *) left_value;
    const segmentation_part_group * right =
        (const segmentation_part_group *) right_value;
    if (left->part_id < right->part_id) return -1;
    if (left->part_id > right->part_id) return 1;
    if (left->range_index < right->range_index) return -1;
    if (left->range_index > right->range_index) return 1;
    return 0;
}

static unsigned count_bits64(uint64_t value) {
    unsigned count = 0;
    while (value != 0) {
        value &= value - 1u;
        ++count;
    }
    return count;
}

static void group_geometry_free(segmentation_part_group * group) {
    if (group == NULL) return;
    if (group->attributes != NULL) {
        for (size_t attribute = 0; attribute < group->attribute_count; ++attribute) {
            free(group->attributes[attribute].values);
        }
    }
    free(group->attributes);
    free(group->indices);
    group->attributes = NULL;
    group->indices = NULL;
}

static int finite_attribute_values(const float * values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static trellis_status configure_group_attributes(
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_rigging_primitive_range * range,
    segmentation_part_group * group,
    size_t max_vertices,
    char * error,
    size_t error_size) {
    const uint64_t texcoord_mask = range != NULL ? range->texcoord_mask : 0;
    const uint64_t color_mask = range != NULL ? range->color_mask : 0;
    const int has_tangent = range != NULL && range->has_source_tangent;
    size_t attribute_count = 2u + (has_tangent ? 1u : 0u) +
        count_bits64(texcoord_mask) + count_bits64(color_mask);
    group->attributes = (segmentation_attribute_data *) calloc(
        attribute_count, sizeof(*group->attributes));
    if (group->attributes == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    group->attribute_count = attribute_count;
    size_t cursor = 0;

#define CONFIGURE_ATTRIBUTE(kind, value_kind, semantic_index, component_count, stride, source_ptr) do { \
        segmentation_attribute_data * attribute = &group->attributes[cursor++]; \
        attribute->type = (kind); \
        attribute->value_type = (value_kind); \
        attribute->index = (int) (semantic_index); \
        attribute->components = (component_count); \
        attribute->source_stride = (stride); \
        attribute->source = (source_ptr); \
        size_t value_count = 0; \
        size_t byte_count = 0; \
        if (!multiply_size(max_vertices, attribute->components, &value_count) || \
            !multiply_size(value_count, sizeof(float), &byte_count)) \
            return TRELLIS_STATUS_OUT_OF_MEMORY; \
        attribute->values = (float *) calloc(byte_count, 1u); \
        if (attribute->values == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY; \
    } while (0)

    CONFIGURE_ATTRIBUTE(
        cgltf_attribute_type_position, cgltf_type_vec3, 0, 3u, 3u, asset->positions);
    CONFIGURE_ATTRIBUTE(
        cgltf_attribute_type_normal, cgltf_type_vec3, 0, 3u, 3u, asset->normals);
    if (has_tangent) {
        if (asset->tangents == NULL) {
            parts_set_error(error, error_size, "source TANGENT metadata has no flattened values");
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        CONFIGURE_ATTRIBUTE(
            cgltf_attribute_type_tangent, cgltf_type_vec4, 0, 4u, 4u, asset->tangents);
    }
    for (size_t set = 0; set < 64u; ++set) {
        if ((texcoord_mask & (UINT64_C(1) << set)) == 0) continue;
        if (set >= asset->texcoord_set_count || asset->texcoords == NULL ||
            asset->texcoords[set] == NULL) {
            parts_set_error(error, error_size, "source TEXCOORD metadata has no flattened values");
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        CONFIGURE_ATTRIBUTE(
            cgltf_attribute_type_texcoord, cgltf_type_vec2, set, 2u, 2u,
            asset->texcoords[set]);
    }
    for (size_t set = 0; set < 64u; ++set) {
        if ((color_mask & (UINT64_C(1) << set)) == 0) continue;
        if (set >= asset->color_set_count || asset->colors == NULL ||
            asset->colors[set] == NULL) {
            parts_set_error(error, error_size, "source COLOR metadata has no flattened values");
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        const int vec4 = (range->color_vec4_mask & (UINT64_C(1) << set)) != 0;
        CONFIGURE_ATTRIBUTE(
            cgltf_attribute_type_color,
            vec4 ? cgltf_type_vec4 : cgltf_type_vec3,
            set,
            vec4 ? 4u : 3u,
            4u,
            asset->colors[set]);
    }
#undef CONFIGURE_ATTRIBUTE
    return cursor == attribute_count ? TRELLIS_STATUS_OK : TRELLIS_STATUS_ERROR;
}

static void normalize_generated_normals(segmentation_attribute_data * normal, size_t vertex_count) {
    for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
        float * value = normal->values + vertex * 3u;
        const float length = sqrtf(
            value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
        if (length > 1e-20f && isfinite(length)) {
            value[0] /= length;
            value[1] /= length;
            value[2] /= length;
        } else {
            value[0] = 0.0f;
            value[1] = 1.0f;
            value[2] = 0.0f;
        }
    }
}

static trellis_status build_group_geometry(
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_rigging_primitive_range * range,
    const uint32_t * face_next,
    uint32_t * vertex_map,
    uint32_t * vertex_stamps,
    uint32_t stamp,
    segmentation_part_group * group,
    char * error,
    size_t error_size) {
    size_t max_vertices = 0;
    size_t index_bytes = 0;
    if (!multiply_size(group->face_count, 3u, &max_vertices) ||
        max_vertices > UINT32_MAX ||
        !multiply_size(max_vertices, sizeof(uint32_t), &index_bytes)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    group->indices = (uint32_t *) malloc(index_bytes);
    if (group->indices == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    trellis_status status = configure_group_attributes(
        asset, range, group, max_vertices, error, error_size);
    if (status != TRELLIS_STATUS_OK) return status;
    const int generate_normals = asset->normals == NULL;

    uint32_t face = group->first_face;
    for (size_t face_in_group = 0; face_in_group < group->face_count; ++face_in_group) {
        if ((size_t) face >= asset->triangle_count ||
            (range != NULL &&
             ((size_t) face < range->first_triangle ||
              (size_t) face >= range->first_triangle + range->triangle_count))) {
            parts_set_error(error, error_size, "part/source-primitive face grouping is inconsistent");
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        uint32_t local[3];
        for (size_t corner = 0; corner < 3u; ++corner) {
            const uint32_t source_vertex = asset->triangles[(size_t) face * 3u + corner];
            if ((size_t) source_vertex >= asset->vertex_count ||
                (range != NULL &&
                 ((size_t) source_vertex < range->first_vertex ||
                  (size_t) source_vertex >= range->first_vertex + range->vertex_count))) {
                parts_set_error(error, error_size, "source triangle vertex mapping is inconsistent");
                return TRELLIS_STATUS_PARSE_ERROR;
            }
            if (vertex_stamps[source_vertex] != stamp) {
                if (group->vertex_count >= max_vertices ||
                    group->vertex_count > UINT32_MAX) return TRELLIS_STATUS_OUT_OF_MEMORY;
                vertex_stamps[source_vertex] = stamp;
                vertex_map[source_vertex] = (uint32_t) group->vertex_count;
                for (size_t attribute_index = 0;
                     attribute_index < group->attribute_count;
                     ++attribute_index) {
                    segmentation_attribute_data * attribute =
                        &group->attributes[attribute_index];
                    float * destination = attribute->values +
                        group->vertex_count * attribute->components;
                    if (attribute->source != NULL) {
                        const float * source = attribute->source +
                            (size_t) source_vertex * attribute->source_stride;
                        memcpy(
                            destination,
                            source,
                            attribute->components * sizeof(float));
                        if (!finite_attribute_values(destination, attribute->components)) {
                            parts_set_error(error, error_size, "flattened vertex attribute is non-finite");
                            return TRELLIS_STATUS_PARSE_ERROR;
                        }
                    }
                }
                const float * position = group->attributes[0].values +
                    group->vertex_count * 3u;
                for (size_t axis = 0; axis < 3u; ++axis) {
                    if (group->vertex_count == 0 || position[axis] < group->bounds_min[axis]) {
                        group->bounds_min[axis] = position[axis];
                    }
                    if (group->vertex_count == 0 || position[axis] > group->bounds_max[axis]) {
                        group->bounds_max[axis] = position[axis];
                    }
                }
                ++group->vertex_count;
            }
            local[corner] = vertex_map[source_vertex];
            group->indices[group->index_count++] = local[corner];
        }
        if (generate_normals) {
            const float * a = group->attributes[0].values + (size_t) local[0] * 3u;
            const float * b = group->attributes[0].values + (size_t) local[1] * 3u;
            const float * c = group->attributes[0].values + (size_t) local[2] * 3u;
            const float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
            const float ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
            const float n[3] = {
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            };
            for (size_t corner = 0; corner < 3u; ++corner) {
                float * destination = group->attributes[1].values + (size_t) local[corner] * 3u;
                destination[0] += n[0];
                destination[1] += n[1];
                destination[2] += n[2];
            }
        }
        face = face_next[face];
    }
    if (group->index_count != group->face_count * 3u || group->vertex_count == 0 ||
        face != UINT32_MAX) {
        parts_set_error(error, error_size, "part/source-primitive face chain is inconsistent");
        return TRELLIS_STATUS_ERROR;
    }
    if (generate_normals) normalize_generated_normals(&group->attributes[1], group->vertex_count);
    /* The first pass must reserve the worst case (three unique vertices per
     * face), but most groups reuse vertices heavily. Release that slack before
     * allocating the final GLB binary so large meshes do not carry both full
     * worst-case attribute storage and the output payload at once. A failed
     * shrinking realloc leaves the original, still-valid allocation intact. */
    for (size_t attribute_index = 0;
         attribute_index < group->attribute_count;
         ++attribute_index) {
        segmentation_attribute_data * attribute = &group->attributes[attribute_index];
        size_t value_count = 0;
        size_t byte_count = 0;
        if (multiply_size(group->vertex_count, attribute->components, &value_count) &&
            multiply_size(value_count, sizeof(float), &byte_count) &&
            byte_count != 0) {
            void * shrunk = realloc(attribute->values, byte_count);
            if (shrunk != NULL) attribute->values = (float *) shrunk;
        }
    }
    return TRELLIS_STATUS_OK;
}

static const cgltf_primitive * source_primitive_for_range(
    const trellis_gltf_source_appearance * appearance,
    const trellis_mesh_rigging_primitive_range * range) {
    if (appearance == NULL || appearance->data == NULL || range == NULL ||
        range->source_mesh_index >= (size_t) appearance->data->meshes_count ||
        range->source_primitive_index >= (size_t)
            appearance->data->meshes[range->source_mesh_index].primitives_count) return NULL;
    return &appearance->data->meshes[range->source_mesh_index].primitives[
        range->source_primitive_index];
}

static trellis_status validate_preservable_source(
    const trellis_mesh_rigging_asset * asset,
    const trellis_gltf_source_appearance * appearance,
    char * error,
    size_t error_size) {
    if (appearance->data == NULL) return TRELLIS_STATUS_OK;
    const cgltf_data * data = appearance->data;
    if (data->animations_count != 0) {
        parts_set_error(error, error_size, "animated source documents cannot yet be losslessly split");
        return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }
    if (data->data_extensions_count != 0) {
        parts_set_error_extension(
            error, error_size, "document", 0u, data->data_extensions[0].name);
        return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }
#define REJECT_RAW_EXTENSIONS(array_name, count_name, owner_name) do { \
        for (size_t i = 0; i < (size_t) data->count_name; ++i) { \
            if (data->array_name[i].extensions_count != 0) { \
                parts_set_error_extension( \
                    error, error_size, owner_name, i, data->array_name[i].extensions[0].name); \
                return TRELLIS_STATUS_NOT_IMPLEMENTED; \
            } \
        } \
    } while (0)
    REJECT_RAW_EXTENSIONS(materials, materials_count, "material");
    REJECT_RAW_EXTENSIONS(textures, textures_count, "texture");
    REJECT_RAW_EXTENSIONS(images, images_count, "image");
    REJECT_RAW_EXTENSIONS(samplers, samplers_count, "sampler");
    REJECT_RAW_EXTENSIONS(accessors, accessors_count, "accessor");
    REJECT_RAW_EXTENSIONS(buffer_views, buffer_views_count, "bufferView");
    REJECT_RAW_EXTENSIONS(buffers, buffers_count, "buffer");
#undef REJECT_RAW_EXTENSIONS

    for (size_t range_index = 0; range_index < asset->primitive_count; ++range_index) {
        const trellis_mesh_rigging_primitive_range * range = &asset->primitives[range_index];
        const cgltf_primitive * primitive = source_primitive_for_range(
            appearance, range);
        if (primitive == NULL) {
            parts_set_error(error, error_size, "source primitive mapping is inconsistent");
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        const cgltf_mesh * mesh = &data->meshes[range->source_mesh_index];
        const cgltf_node * node = range->source_node_index < (size_t) data->nodes_count ?
            &data->nodes[range->source_node_index] : NULL;
        if (mesh->extensions_count != 0 || primitive->extensions_count != 0 ||
            (node != NULL && node->extensions_count != 0)) {
            const cgltf_extension * extension = mesh->extensions_count != 0 ?
                &mesh->extensions[0] : primitive->extensions_count != 0 ?
                    &primitive->extensions[0] : &node->extensions[0];
            parts_set_error_extension(
                error, error_size, "source primitive", range_index, extension->name);
            return TRELLIS_STATUS_NOT_IMPLEMENTED;
        }
        if (node == NULL || node->has_mesh_gpu_instancing || node->skin != NULL ||
            node->weights_count != 0 || mesh->weights_count != 0 ||
            mesh->target_names_count != 0) {
            parts_set_error(
                error,
                error_size,
                "skinned, animated, instanced, or morph-weighted source nodes cannot be losslessly split");
            return TRELLIS_STATUS_NOT_IMPLEMENTED;
        }
        if (primitive->targets_count != 0) {
            parts_set_error(error, error_size, "morph-target primitives cannot yet be losslessly split");
            return TRELLIS_STATUS_NOT_IMPLEMENTED;
        }
        uint64_t texcoord_semantics = 0;
        uint64_t color_semantics = 0;
        uint64_t color_vec4_semantics = 0;
        unsigned position_count = 0;
        unsigned normal_count = 0;
        unsigned tangent_count = 0;
        for (size_t attribute = 0;
             attribute < (size_t) primitive->attributes_count;
             ++attribute) {
            const cgltf_attribute_type type = primitive->attributes[attribute].type;
            if (type != cgltf_attribute_type_position &&
                type != cgltf_attribute_type_normal &&
                type != cgltf_attribute_type_tangent &&
                type != cgltf_attribute_type_texcoord &&
                type != cgltf_attribute_type_color) {
                parts_set_error(
                    error,
                    error_size,
                    "source primitive contains unsupported skinning/custom vertex attributes");
                return TRELLIS_STATUS_NOT_IMPLEMENTED;
            }
            const int semantic_index = primitive->attributes[attribute].index;
            if ((type == cgltf_attribute_type_position ||
                 type == cgltf_attribute_type_normal ||
                 type == cgltf_attribute_type_tangent) && semantic_index != 0) {
                parts_set_error(
                    error,
                    error_size,
                    "indexed POSITION/NORMAL/TANGENT semantics cannot be losslessly split");
                return TRELLIS_STATUS_NOT_IMPLEMENTED;
            }
            if (type == cgltf_attribute_type_position) ++position_count;
            if (type == cgltf_attribute_type_normal) ++normal_count;
            if (type == cgltf_attribute_type_tangent) ++tangent_count;
            if (type == cgltf_attribute_type_texcoord || type == cgltf_attribute_type_color) {
                if (semantic_index < 0 || semantic_index >= 64) {
                    parts_set_error(error, error_size, "vertex attribute set exceeds the 64-set preservation limit");
                    return TRELLIS_STATUS_NOT_IMPLEMENTED;
                }
                uint64_t * semantics = type == cgltf_attribute_type_texcoord ?
                    &texcoord_semantics : &color_semantics;
                const uint64_t bit = UINT64_C(1) << (unsigned) semantic_index;
                if ((*semantics & bit) != 0) {
                    parts_set_error(error, error_size, "duplicate vertex attribute semantics cannot be preserved");
                    return TRELLIS_STATUS_NOT_IMPLEMENTED;
                }
                *semantics |= bit;
                if (type == cgltf_attribute_type_color) {
                    const cgltf_accessor * accessor = primitive->attributes[attribute].data;
                    if (accessor == NULL ||
                        (accessor->type != cgltf_type_vec3 &&
                         accessor->type != cgltf_type_vec4)) {
                        parts_set_error(error, error_size, "source COLOR accessor type changed after flattening");
                        return TRELLIS_STATUS_PARSE_ERROR;
                    }
                    if (accessor->type == cgltf_type_vec4) color_vec4_semantics |= bit;
                }
            }
        }
        if (position_count != 1u || normal_count > 1u || tangent_count > 1u) {
            parts_set_error(error, error_size, "duplicate fixed vertex semantics cannot be preserved");
            return TRELLIS_STATUS_NOT_IMPLEMENTED;
        }
        if ((normal_count != 0u) != (range->has_source_normal != 0) ||
            (tangent_count != 0u) != (range->has_source_tangent != 0) ||
            texcoord_semantics != range->texcoord_mask ||
            color_semantics != range->color_mask ||
            color_vec4_semantics != range->color_vec4_mask) {
            parts_set_error(error, error_size, "source vertex attributes changed after flattening");
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    return TRELLIS_STATUS_OK;
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
        if (node == NULL || node->mesh == NULL || node->mesh->primitives_count == 0) {
            cgltf_free(data);
            parts_set_error(error_out, error_size, "written GLB contains an invalid part node");
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        for (size_t primitive_index = 0;
             primitive_index < (size_t) node->mesh->primitives_count;
             ++primitive_index) {
            const cgltf_primitive * primitive = &node->mesh->primitives[primitive_index];
            if (primitive->indices == NULL || primitive->material == NULL ||
                primitive->indices->count % 3u != 0 ||
                cgltf_find_accessor(primitive, cgltf_attribute_type_position, 0) == NULL ||
                cgltf_find_accessor(primitive, cgltf_attribute_type_normal, 0) == NULL) {
                cgltf_free(data);
                parts_set_error(error_out, error_size, "written GLB contains an invalid part primitive");
                return TRELLIS_STATUS_PARSE_ERROR;
            }
            const size_t primitive_faces = (size_t) primitive->indices->count / 3u;
            if (!add_size(faces, primitive_faces, &faces)) {
                cgltf_free(data);
                parts_set_error(error_out, error_size, "written GLB face count overflowed");
                return TRELLIS_STATUS_PARSE_ERROR;
            }
        }
    }
    cgltf_free(data);
    if (faces != expected_faces) {
        parts_set_error(error_out, error_size, "written GLB lost or duplicated faces");
        return TRELLIS_STATUS_ERROR;
    }
    return TRELLIS_STATUS_OK;
}

static const char * attribute_name(
    const segmentation_attribute_data * attribute,
    char * generated,
    size_t generated_size) {
    switch (attribute->type) {
        case cgltf_attribute_type_position: return "POSITION";
        case cgltf_attribute_type_normal: return "NORMAL";
        case cgltf_attribute_type_tangent: return "TANGENT";
        case cgltf_attribute_type_texcoord:
            snprintf(generated, generated_size, "TEXCOORD_%d", attribute->index);
            return generated;
        case cgltf_attribute_type_color:
            snprintf(generated, generated_size, "COLOR_%d", attribute->index);
            return generated;
        default: return NULL;
    }
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
        face_count > UINT32_MAX || part_count == 0 || part_count >= UINT32_MAX ||
        asset->vertex_count > SIZE_MAX / sizeof(uint32_t) ||
        (asset->primitive_count != 0 && asset->primitives == NULL)) {
        parts_set_error(error_out, error_size, "invalid parts GLB request");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = TRELLIS_STATUS_ERROR;
    trellis_gltf_source_appearance appearance;
    memset(&appearance, 0, sizeof(appearance));
    segmentation_group_table table;
    memset(&table, 0, sizeof(table));
    size_t * part_faces = NULL;
    uint32_t * face_next = NULL;
    uint32_t * vertex_map = NULL;
    uint32_t * vertex_stamps = NULL;
    unsigned char * binary = NULL;
    cgltf_buffer_view * views = NULL;
    cgltf_accessor * accessors = NULL;
    cgltf_attribute * attributes = NULL;
    cgltf_primitive * primitives = NULL;
    cgltf_material_mapping * mappings = NULL;
    cgltf_mesh * meshes = NULL;
    cgltf_material * materials = NULL;
    cgltf_node * nodes = NULL;
    cgltf_node ** children = NULL;
    char (*names)[64] = NULL;
    char (*extras)[64] = NULL;
    char (*attribute_names)[24] = NULL;
    size_t * image_offsets = NULL;
    size_t * part_group_offsets = NULL;

    status = trellis_gltf_source_appearance_load(
        asset, &appearance, error_out, error_size);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    status = validate_preservable_source(
        asset, &appearance, error_out, error_size);
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    for (size_t range_index = 0; range_index < asset->primitive_count; ++range_index) {
        const trellis_mesh_rigging_primitive_range * range = &asset->primitives[range_index];
        if (range->first_triangle > asset->triangle_count ||
            range->triangle_count > asset->triangle_count - range->first_triangle ||
            range->first_vertex > asset->vertex_count ||
            range->vertex_count > asset->vertex_count - range->first_vertex) {
            parts_set_error(error_out, error_size, "source primitive range exceeds flattened mesh bounds");
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
    }
    size_t face_next_bytes = 0;
    if (!multiply_size(face_count, sizeof(uint32_t), &face_next_bytes)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    part_faces = (size_t *) calloc(part_count, sizeof(size_t));
    face_next = (uint32_t *) malloc(face_next_bytes);
    if (part_faces == NULL || face_next == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (size_t face = 0; face < face_count; ++face) face_next[face] = UINT32_MAX;

    size_t range_cursor = 0;
    size_t retained_face_count = 0;
    for (size_t face = 0; face < face_count; ++face) {
        if (asset->primitive_count != 0) {
            while (range_cursor < asset->primitive_count &&
                face >= asset->primitives[range_cursor].first_triangle +
                    asset->primitives[range_cursor].triangle_count) {
                ++range_cursor;
            }
            if (range_cursor >= asset->primitive_count ||
                face < asset->primitives[range_cursor].first_triangle) {
                parts_set_error(error_out, error_size, "primitive ranges do not cover every source face");
                status = TRELLIS_STATUS_PARSE_ERROR;
                goto cleanup;
            }
        }
        if (face_part_ids[face] == UINT32_MAX) continue;
        if ((size_t) face_part_ids[face] >= part_count) {
            parts_set_error(error_out, error_size, "face part id exceeds part_count");
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
        ++part_faces[face_part_ids[face]];
        ++retained_face_count;
        if (!group_table_find_or_insert(
                &table,
                face_part_ids[face],
                asset->primitive_count != 0 ? range_cursor : 0u,
                (uint32_t) face,
                face_next)) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
    }
    if (retained_face_count == 0 || table.count == 0 ||
        table.count >= UINT32_MAX) {
        parts_set_error(error_out, error_size, "parts GLB cannot discard every source face");
        status = retained_face_count == 0 ?
            TRELLIS_STATUS_INVALID_ARGUMENT : TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (size_t part = 0; part < part_count; ++part) {
        if (part_faces[part] == 0) {
            parts_set_error(error_out, error_size, "part ids must be dense and non-empty");
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
    }
    qsort(table.groups, table.count, sizeof(*table.groups), compare_groups);

    vertex_map = (uint32_t *) malloc(asset->vertex_count * sizeof(uint32_t));
    vertex_stamps = (uint32_t *) calloc(asset->vertex_count, sizeof(uint32_t));
    if (vertex_map == NULL || vertex_stamps == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (size_t group_index = 0; group_index < table.count; ++group_index) {
        const trellis_mesh_rigging_primitive_range * range =
            asset->primitive_count != 0 ?
                &asset->primitives[table.groups[group_index].range_index] : NULL;
        status = build_group_geometry(
            asset,
            range,
            face_next,
            vertex_map,
            vertex_stamps,
            (uint32_t) group_index + 1u,
            &table.groups[group_index],
            error_out,
            error_size);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
    }

    size_t geometry_attribute_count = 0;
    size_t mapping_count = 0;
    for (size_t group = 0; group < table.count; ++group) {
        if (!add_size(
                geometry_attribute_count,
                table.groups[group].attribute_count,
                &geometry_attribute_count)) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        if (asset->primitive_count != 0 && appearance.data != NULL) {
            const cgltf_primitive * source_primitive = source_primitive_for_range(
                &appearance, &asset->primitives[table.groups[group].range_index]);
            if (source_primitive == NULL ||
                !add_size(mapping_count, source_primitive->mappings_count, &mapping_count)) {
                status = TRELLIS_STATUS_OUT_OF_MEMORY;
                goto cleanup;
            }
        }
    }
    size_t geometry_view_count = 0;
    if (!add_size(table.count, geometry_attribute_count, &geometry_view_count)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    const size_t image_count = appearance.data != NULL ?
        (size_t) appearance.data->images_count : 0u;
    size_t view_count = 0;
    if (!add_size(geometry_view_count, image_count, &view_count)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    size_t binary_size = 0;
    for (size_t group_index = 0; group_index < table.count; ++group_index) {
        segmentation_part_group * group = &table.groups[group_index];
        if (!multiply_size(group->index_count, sizeof(uint32_t), &group->index_size) ||
            !layout_region(&binary_size, group->index_size, &group->index_offset)) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        for (size_t attribute_index = 0;
             attribute_index < group->attribute_count;
             ++attribute_index) {
            segmentation_attribute_data * attribute = &group->attributes[attribute_index];
            size_t value_count = 0;
            if (!multiply_size(group->vertex_count, attribute->components, &value_count) ||
                !multiply_size(value_count, sizeof(float), &attribute->size) ||
                !layout_region(&binary_size, attribute->size, &attribute->offset)) {
                status = TRELLIS_STATUS_OUT_OF_MEMORY;
                goto cleanup;
            }
        }
    }
    image_offsets = image_count != 0 ?
        (size_t *) calloc(image_count, sizeof(size_t)) : NULL;
    if (image_count != 0 && image_offsets == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (size_t image = 0; image < image_count; ++image) {
        if (!layout_region(
                &binary_size,
                appearance.images[image].size,
                &image_offsets[image])) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
    }
    if (!align4_size(binary_size, &binary_size) ||
        binary_size == 0 || binary_size > UINT32_MAX) {
        parts_set_error(error_out, error_size, "parts GLB binary chunk exceeds the 32-bit GLB limit");
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    binary = (unsigned char *) calloc(binary_size, 1u);
    if (binary == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (size_t group_index = 0; group_index < table.count; ++group_index) {
        const segmentation_part_group * group = &table.groups[group_index];
        memcpy(binary + group->index_offset, group->indices, group->index_size);
        for (size_t attribute_index = 0;
             attribute_index < group->attribute_count;
             ++attribute_index) {
            const segmentation_attribute_data * attribute = &group->attributes[attribute_index];
            memcpy(binary + attribute->offset, attribute->values, attribute->size);
        }
    }
    for (size_t image = 0; image < image_count; ++image) {
        memcpy(
            binary + image_offsets[image],
            appearance.images[image].bytes,
            appearance.images[image].size);
    }

    views = (cgltf_buffer_view *) calloc(view_count, sizeof(*views));
    accessors = (cgltf_accessor *) calloc(geometry_view_count, sizeof(*accessors));
    attributes = (cgltf_attribute *) calloc(geometry_attribute_count, sizeof(*attributes));
    primitives = (cgltf_primitive *) calloc(table.count, sizeof(*primitives));
    mappings = mapping_count != 0 ?
        (cgltf_material_mapping *) calloc(mapping_count, sizeof(*mappings)) : NULL;
    meshes = (cgltf_mesh *) calloc(part_count, sizeof(*meshes));
    const size_t source_material_count = appearance.data != NULL ?
        (size_t) appearance.data->materials_count : 0u;
    int needs_fallback_material = appearance.data == NULL || asset->primitive_count == 0;
    if (!needs_fallback_material) {
        for (size_t range = 0; range < asset->primitive_count; ++range) {
            if (asset->primitives[range].source_material_index == SIZE_MAX) {
                needs_fallback_material = 1;
                break;
            }
        }
    }
    const size_t material_count = source_material_count +
        (needs_fallback_material ? 1u : 0u);
    materials = (cgltf_material *) calloc(material_count, sizeof(*materials));
    size_t node_count = 0;
    if (!add_size(part_count, 1u, &node_count)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    nodes = (cgltf_node *) calloc(node_count, sizeof(*nodes));
    children = (cgltf_node **) calloc(part_count, sizeof(*children));
    names = (char (*)[64]) calloc(part_count, sizeof(*names));
    extras = (char (*)[64]) calloc(part_count, sizeof(*extras));
    attribute_names = geometry_attribute_count != 0 ?
        (char (*)[24]) calloc(geometry_attribute_count, sizeof(*attribute_names)) : NULL;
    part_group_offsets = (size_t *) calloc(part_count + 1u, sizeof(size_t));
    if (views == NULL || accessors == NULL || attributes == NULL ||
        primitives == NULL || (mapping_count != 0 && mappings == NULL) ||
        meshes == NULL || materials == NULL || nodes == NULL || children == NULL ||
        names == NULL || extras == NULL || attribute_names == NULL ||
        part_group_offsets == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    if (source_material_count != 0) {
        memcpy(
            materials,
            appearance.data->materials,
            source_material_count * sizeof(*materials));
    }
    if (needs_fallback_material) {
        cgltf_material * fallback = &materials[source_material_count];
        fallback->name = (char *) "trellis_default_material";
        fallback->has_pbr_metallic_roughness = 1;
        for (size_t channel = 0; channel < 4u; ++channel) {
            fallback->pbr_metallic_roughness.base_color_factor[channel] = 1.0f;
        }
        /* glTF's implicit material is white, fully metallic, roughness 1. */
        fallback->pbr_metallic_roughness.metallic_factor = 1.0f;
        fallback->pbr_metallic_roughness.roughness_factor = 1.0f;
        fallback->alpha_mode = cgltf_alpha_mode_opaque;
        fallback->alpha_cutoff = 0.5f;
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

    size_t view_cursor = 0;
    size_t accessor_cursor = 0;
    size_t attribute_cursor = 0;
    size_t mapping_cursor = 0;
    for (size_t group_index = 0; group_index < table.count; ++group_index) {
        const segmentation_part_group * group = &table.groups[group_index];
        cgltf_buffer_view * index_view = &views[view_cursor];
        index_view->buffer = &buffer;
        index_view->offset = group->index_offset;
        index_view->size = group->index_size;
        index_view->type = cgltf_buffer_view_type_indices;
        cgltf_accessor * index_accessor = &accessors[accessor_cursor];
        index_accessor->buffer_view = index_view;
        index_accessor->component_type = cgltf_component_type_r_32u;
        index_accessor->type = cgltf_type_scalar;
        index_accessor->count = group->index_count;
        ++view_cursor;
        ++accessor_cursor;

        cgltf_primitive * primitive = &primitives[group_index];
        primitive->type = cgltf_primitive_type_triangles;
        primitive->indices = index_accessor;
        primitive->attributes = &attributes[attribute_cursor];
        primitive->attributes_count = group->attribute_count;
        for (size_t group_attribute = 0;
             group_attribute < group->attribute_count;
             ++group_attribute) {
            const segmentation_attribute_data * source_attribute =
                &group->attributes[group_attribute];
            cgltf_buffer_view * view = &views[view_cursor++];
            view->buffer = &buffer;
            view->offset = source_attribute->offset;
            view->size = source_attribute->size;
            view->stride = source_attribute->components * sizeof(float);
            view->type = cgltf_buffer_view_type_vertices;
            cgltf_accessor * accessor = &accessors[accessor_cursor++];
            accessor->buffer_view = view;
            accessor->component_type = cgltf_component_type_r_32f;
            accessor->type = source_attribute->value_type;
            accessor->count = group->vertex_count;
            if (source_attribute->type == cgltf_attribute_type_position) {
                accessor->has_min = accessor->has_max = 1;
                memcpy(accessor->min, group->bounds_min, 3u * sizeof(float));
                memcpy(accessor->max, group->bounds_max, 3u * sizeof(float));
            }
            cgltf_attribute * attribute = &attributes[attribute_cursor];
            attribute->name = (char *) attribute_name(
                source_attribute,
                attribute_names[attribute_cursor],
                sizeof(attribute_names[attribute_cursor]));
            attribute->type = source_attribute->type;
            attribute->index = source_attribute->index;
            attribute->data = accessor;
            ++attribute_cursor;
        }

        const trellis_mesh_rigging_primitive_range * range =
            asset->primitive_count != 0 ?
                &asset->primitives[group->range_index] : NULL;
        const cgltf_primitive * source_primitive = source_primitive_for_range(
            &appearance, range);
        if (appearance.data != NULL && range != NULL &&
            range->source_material_index != SIZE_MAX &&
            range->source_material_index < source_material_count) {
            primitive->material = &materials[range->source_material_index];
        } else {
            primitive->material = &materials[source_material_count];
        }
        if (source_primitive != NULL) {
            primitive->extras = source_primitive->extras;
            primitive->mappings = source_primitive->mappings_count != 0 ?
                &mappings[mapping_cursor] : NULL;
            primitive->mappings_count = source_primitive->mappings_count;
            for (size_t mapping = 0;
                 mapping < (size_t) source_primitive->mappings_count;
                 ++mapping) {
                mappings[mapping_cursor] = source_primitive->mappings[mapping];
                const size_t material_index = (size_t)
                    (source_primitive->mappings[mapping].material -
                     appearance.data->materials);
                if (material_index >= source_material_count) {
                    parts_set_error(error_out, error_size, "material-variant mapping is invalid");
                    status = TRELLIS_STATUS_PARSE_ERROR;
                    goto cleanup;
                }
                mappings[mapping_cursor].material = &materials[material_index];
                ++mapping_cursor;
            }
        }
        ++part_group_offsets[group->part_id + 1u];
    }
    if (view_cursor != geometry_view_count ||
        accessor_cursor != geometry_view_count ||
        attribute_cursor != geometry_attribute_count ||
        mapping_cursor != mapping_count) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }

    for (size_t image = 0; image < image_count; ++image) {
        cgltf_buffer_view * view = &views[view_cursor++];
        view->buffer = &buffer;
        view->offset = image_offsets[image];
        view->size = appearance.images[image].size;
        view->name = appearance.data->images[image].name;
        appearance.data->images[image].uri = NULL;
        appearance.data->images[image].buffer_view = view;
        appearance.data->images[image].mime_type =
            (char *) appearance.images[image].embedded_mime_type;
    }
    if (view_cursor != view_count) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    for (size_t part = 0; part < part_count; ++part) {
        part_group_offsets[part + 1u] += part_group_offsets[part];
        snprintf(names[part], sizeof(names[part]), "part_%04u", (unsigned) part);
        snprintf(extras[part], sizeof(extras[part]), "{\"trellis_part_id\":%u}", (unsigned) part);
        meshes[part].name = names[part];
        meshes[part].primitives = &primitives[part_group_offsets[part]];
        meshes[part].primitives_count =
            part_group_offsets[part + 1u] - part_group_offsets[part];
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
    data.buffer_views_count = view_count;
    data.accessors = accessors;
    data.accessors_count = geometry_view_count;
    data.materials = materials;
    data.materials_count = material_count;
    data.meshes = meshes;
    data.meshes_count = part_count;
    data.nodes = nodes;
    data.nodes_count = node_count;
    if (appearance.data != NULL) {
        data.images = appearance.data->images;
        data.images_count = appearance.data->images_count;
        data.samplers = appearance.data->samplers;
        data.samplers_count = appearance.data->samplers_count;
        data.textures = appearance.data->textures;
        data.textures_count = appearance.data->textures_count;
        data.variants = appearance.data->variants;
        data.variants_count = appearance.data->variants_count;
    }
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
        status = parts_write_glb_stream(temporary_file, &write_options, &data);
        if (fclose(temporary_file) != 0 && status == TRELLIS_STATUS_OK) {
            status = TRELLIS_STATUS_IO_ERROR;
        }
        temporary_file = NULL;
        if (status != TRELLIS_STATUS_OK) {
            parts_set_error(
                error_out,
                error_size,
                status == TRELLIS_STATUS_OUT_OF_MEMORY ?
                    "parts GLB exceeds the 32-bit GLB size limit" :
                    "cgltf failed to write parts GLB");
        } else {
            status = verify_parts_glb(
                temporary, part_count, retained_face_count, error_out, error_size);
            if (status == TRELLIS_STATUS_OK && !parts_atomic_replace(temporary, path)) {
                parts_set_error(error_out, error_size, "failed to atomically replace parts GLB");
                status = TRELLIS_STATUS_IO_ERROR;
            }
        }
        if (status != TRELLIS_STATUS_OK) trellis_unlink(temporary);
        free(temporary);
    }

cleanup:
    if (table.groups != NULL) {
        for (size_t group = 0; group < table.count; ++group) {
            group_geometry_free(&table.groups[group]);
        }
    }
    trellis_gltf_source_appearance_free(&appearance);
    group_table_free(&table);
    free(part_faces);
    free(face_next);
    free(vertex_map);
    free(vertex_stamps);
    free(binary);
    free(views);
    free(accessors);
    free(attributes);
    free(primitives);
    free(mappings);
    free(meshes);
    free(materials);
    free(nodes);
    free(children);
    free(names);
    free(extras);
    free(attribute_names);
    free(image_offsets);
    free(part_group_offsets);
    return status;
}
