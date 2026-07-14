#include "partition.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct segmentation_edge {
    uint32_t low;
    uint32_t high;
    uint32_t face;
} segmentation_edge;

static int compare_edges(const void * left_ptr, const void * right_ptr) {
    const segmentation_edge * left = (const segmentation_edge *) left_ptr;
    const segmentation_edge * right = (const segmentation_edge *) right_ptr;
    if (left->low != right->low) return left->low < right->low ? -1 : 1;
    if (left->high != right->high) return left->high < right->high ? -1 : 1;
    if (left->face != right->face) return left->face < right->face ? -1 : 1;
    return 0;
}

static uint32_t partition_find(uint32_t * parent, uint32_t value) {
    uint32_t root = value;
    while (parent[root] != root) root = parent[root];
    while (parent[value] != value) {
        const uint32_t next = parent[value];
        parent[value] = root;
        value = next;
    }
    return root;
}

static uint32_t partition_union(
    uint32_t * parent,
    size_t * sizes,
    uint32_t left,
    uint32_t right) {
    left = partition_find(parent, left);
    right = partition_find(parent, right);
    if (left == right) return left;
    if (sizes[left] < sizes[right] ||
        (sizes[left] == sizes[right] && left > right)) {
        const uint32_t temporary = left;
        left = right;
        right = temporary;
    }
    parent[right] = left;
    sizes[left] += sizes[right];
    sizes[right] = 0;
    return left;
}

static int edge_same_key(
    const segmentation_edge * left,
    const segmentation_edge * right) {
    return left->low == right->low && left->high == right->high;
}

trellis_status trellis_mesh_segmentation_partition_faces(
    const uint32_t * triangles,
    size_t face_count,
    size_t vertex_count,
    const uint32_t * semantic_labels,
    size_t min_component_faces,
    uint32_t ** part_ids_out,
    size_t * part_count_out) {
    if (part_ids_out != NULL) *part_ids_out = NULL;
    if (part_count_out != NULL) *part_count_out = 0;
    if (triangles == NULL || semantic_labels == NULL || face_count == 0 ||
        vertex_count == 0 || part_ids_out == NULL || part_count_out == NULL ||
        face_count > UINT32_MAX ||
        face_count > SIZE_MAX / (3u * sizeof(segmentation_edge)) ||
        face_count > SIZE_MAX / sizeof(uint32_t) ||
        face_count > SIZE_MAX / sizeof(size_t)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t edge_count = face_count * 3u;
    segmentation_edge * edges =
        (segmentation_edge *) malloc(edge_count * sizeof(*edges));
    uint32_t * parent = (uint32_t *) malloc(face_count * sizeof(uint32_t));
    size_t * sizes = (size_t *) malloc(face_count * sizeof(size_t));
    uint32_t * best_neighbor = (uint32_t *) malloc(face_count * sizeof(uint32_t));
    size_t * best_size = (size_t *) calloc(face_count, sizeof(size_t));
    uint32_t * dense_root = (uint32_t *) malloc(face_count * sizeof(uint32_t));
    uint32_t * output = (uint32_t *) malloc(face_count * sizeof(uint32_t));
    if (edges == NULL || parent == NULL || sizes == NULL ||
        best_neighbor == NULL || best_size == NULL || dense_root == NULL ||
        output == NULL) {
        free(edges); free(parent); free(sizes); free(best_neighbor);
        free(best_size); free(dense_root); free(output);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (size_t face = 0; face < face_count; ++face) {
        parent[face] = (uint32_t) face;
        sizes[face] = 1;
        best_neighbor[face] = UINT32_MAX;
        dense_root[face] = UINT32_MAX;
        const uint32_t a = triangles[face * 3u];
        const uint32_t b = triangles[face * 3u + 1u];
        const uint32_t c = triangles[face * 3u + 2u];
        if ((size_t) a >= vertex_count || (size_t) b >= vertex_count ||
            (size_t) c >= vertex_count || a == b || b == c || c == a) {
            free(edges); free(parent); free(sizes); free(best_neighbor);
            free(best_size); free(dense_root); free(output);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        const uint32_t vertices[3] = {a, b, c};
        for (size_t edge = 0; edge < 3; ++edge) {
            const uint32_t first = vertices[edge];
            const uint32_t second = vertices[(edge + 1u) % 3u];
            segmentation_edge * destination = &edges[face * 3u + edge];
            destination->low = first < second ? first : second;
            destination->high = first < second ? second : first;
            destination->face = (uint32_t) face;
        }
    }
    qsort(edges, edge_count, sizeof(*edges), compare_edges);

    size_t begin = 0;
    while (begin < edge_count) {
        size_t end = begin + 1u;
        while (end < edge_count && edge_same_key(&edges[begin], &edges[end])) ++end;
        for (size_t left = begin; left < end; ++left) {
            for (size_t right = left + 1u; right < end; ++right) {
                const uint32_t left_face = edges[left].face;
                const uint32_t right_face = edges[right].face;
                if (semantic_labels[left_face] == semantic_labels[right_face]) {
                    partition_union(parent, sizes, left_face, right_face);
                }
            }
        }
        begin = end;
    }

    if (min_component_faces > 1) {
        begin = 0;
        while (begin < edge_count) {
            size_t end = begin + 1u;
            while (end < edge_count && edge_same_key(&edges[begin], &edges[end])) ++end;
            for (size_t left = begin; left < end; ++left) {
                for (size_t right = left + 1u; right < end; ++right) {
                    uint32_t left_root = partition_find(parent, edges[left].face);
                    uint32_t right_root = partition_find(parent, edges[right].face);
                    if (left_root == right_root) continue;
                    if (sizes[left_root] < min_component_faces &&
                        (best_neighbor[left_root] == UINT32_MAX ||
                         sizes[right_root] > best_size[left_root] ||
                         (sizes[right_root] == best_size[left_root] &&
                          right_root < best_neighbor[left_root]))) {
                        best_neighbor[left_root] = right_root;
                        best_size[left_root] = sizes[right_root];
                    }
                    if (sizes[right_root] < min_component_faces &&
                        (best_neighbor[right_root] == UINT32_MAX ||
                         sizes[left_root] > best_size[right_root] ||
                         (sizes[left_root] == best_size[right_root] &&
                          left_root < best_neighbor[right_root]))) {
                        best_neighbor[right_root] = left_root;
                        best_size[right_root] = sizes[left_root];
                    }
                }
            }
            begin = end;
        }
        for (size_t face = 0; face < face_count; ++face) {
            if (parent[face] != face || sizes[face] >= min_component_faces ||
                best_neighbor[face] == UINT32_MAX) {
                continue;
            }
            const uint32_t neighbor = partition_find(parent, best_neighbor[face]);
            if (neighbor != face) partition_union(parent, sizes, (uint32_t) face, neighbor);
        }
    }

    size_t part_count = 0;
    for (size_t face = 0; face < face_count; ++face) {
        const uint32_t root = partition_find(parent, (uint32_t) face);
        if (dense_root[root] == UINT32_MAX) {
            if (part_count > UINT32_MAX) {
                free(edges); free(parent); free(sizes); free(best_neighbor);
                free(best_size); free(dense_root); free(output);
                return TRELLIS_STATUS_OUT_OF_MEMORY;
            }
            dense_root[root] = (uint32_t) part_count++;
        }
        output[face] = dense_root[root];
    }

    free(edges); free(parent); free(sizes); free(best_neighbor);
    free(best_size); free(dense_root);
    *part_ids_out = output;
    *part_count_out = part_count;
    return TRELLIS_STATUS_OK;
}

static uint64_t segmentation_mix_u64(uint64_t value) {
    value ^= value >> 30u;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27u;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

static size_t segmentation_cell_bucket(
    int64_t x,
    int64_t y,
    int64_t z,
    size_t mask) {
    uint64_t hash = segmentation_mix_u64((uint64_t) x);
    hash ^= segmentation_mix_u64((uint64_t) y + UINT64_C(0x9e3779b97f4a7c15));
    hash ^= segmentation_mix_u64((uint64_t) z + UINT64_C(0x243f6a8885a308d3));
    return (size_t) hash & mask;
}

trellis_status trellis_mesh_segmentation_partition_faces_geometric(
    const float * positions,
    const uint32_t * triangles,
    size_t face_count,
    size_t vertex_count,
    const uint32_t * semantic_labels,
    size_t min_component_faces,
    float weld_tolerance,
    uint32_t ** part_ids_out,
    size_t * part_count_out) {
    if (part_ids_out != NULL) *part_ids_out = NULL;
    if (part_count_out != NULL) *part_count_out = 0;
    if (positions == NULL || triangles == NULL || semantic_labels == NULL ||
        face_count == 0 || vertex_count == 0 || vertex_count > UINT32_MAX ||
        face_count > SIZE_MAX / (3u * sizeof(uint32_t)) ||
        vertex_count > SIZE_MAX / sizeof(uint32_t) ||
        vertex_count > SIZE_MAX / (3u * sizeof(int64_t)) ||
        part_ids_out == NULL || part_count_out == NULL ||
        !isfinite(weld_tolerance) || !(weld_tolerance > 0.0f)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    size_t table_capacity = 16u;
    while (table_capacity < vertex_count) {
        if (table_capacity > SIZE_MAX / 2u) return TRELLIS_STATUS_OUT_OF_MEMORY;
        table_capacity *= 2u;
    }
    if (table_capacity <= vertex_count) {
        if (table_capacity > SIZE_MAX / 2u) return TRELLIS_STATUS_OUT_OF_MEMORY;
        table_capacity *= 2u;
    }
    if (table_capacity > SIZE_MAX / sizeof(uint32_t)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    uint32_t * buckets =
        (uint32_t *) malloc(table_capacity * sizeof(uint32_t));
    uint32_t * next = (uint32_t *) malloc(vertex_count * sizeof(uint32_t));
    uint32_t * representative =
        (uint32_t *) malloc(vertex_count * sizeof(uint32_t));
    uint32_t * vertex_to_welded =
        (uint32_t *) malloc(vertex_count * sizeof(uint32_t));
    int64_t * cells = (int64_t *) malloc(vertex_count * 3u * sizeof(int64_t));
    uint32_t * welded_triangles =
        (uint32_t *) malloc(face_count * 3u * sizeof(uint32_t));
    if (buckets == NULL || next == NULL || representative == NULL ||
        vertex_to_welded == NULL || cells == NULL || welded_triangles == NULL) {
        free(buckets); free(next); free(representative);
        free(vertex_to_welded); free(cells); free(welded_triangles);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (size_t bucket = 0; bucket < table_capacity; ++bucket) {
        buckets[bucket] = UINT32_MAX;
    }

    const double inverse_tolerance = 1.0 / (double) weld_tolerance;
    const double tolerance_squared =
        (double) weld_tolerance * (double) weld_tolerance;
    const size_t mask = table_capacity - 1u;
    uint32_t welded_count = 0;
    trellis_status status = TRELLIS_STATUS_OK;
    for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
        const float * position = positions + vertex * 3u;
        int64_t cell[3];
        for (int axis = 0; axis < 3; ++axis) {
            if (!isfinite(position[axis])) {
                status = TRELLIS_STATUS_PARSE_ERROR;
                goto cleanup;
            }
            const double scaled = floor((double) position[axis] * inverse_tolerance);
            if (scaled < (double) INT64_MIN + 2.0 ||
                scaled > (double) INT64_MAX - 2.0) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
                goto cleanup;
            }
            cell[axis] = (int64_t) scaled;
        }

        uint32_t match = UINT32_MAX;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    const int64_t neighbor_x = cell[0] + dx;
                    const int64_t neighbor_y = cell[1] + dy;
                    const int64_t neighbor_z = cell[2] + dz;
                    const size_t bucket = segmentation_cell_bucket(
                        neighbor_x, neighbor_y, neighbor_z, mask);
                    for (uint32_t candidate = buckets[bucket];
                         candidate != UINT32_MAX;
                         candidate = next[candidate]) {
                        const int64_t * candidate_cell = cells + (size_t) candidate * 3u;
                        if (candidate_cell[0] != neighbor_x ||
                            candidate_cell[1] != neighbor_y ||
                            candidate_cell[2] != neighbor_z) continue;
                        const float * candidate_position =
                            positions + (size_t) representative[candidate] * 3u;
                        const double difference_x =
                            (double) position[0] - candidate_position[0];
                        const double difference_y =
                            (double) position[1] - candidate_position[1];
                        const double difference_z =
                            (double) position[2] - candidate_position[2];
                        const double distance_squared =
                            difference_x * difference_x +
                            difference_y * difference_y +
                            difference_z * difference_z;
                        if (distance_squared <= tolerance_squared &&
                            (match == UINT32_MAX || candidate < match)) {
                            match = candidate;
                        }
                    }
                }
            }
        }
        if (match == UINT32_MAX) {
            if (welded_count == UINT32_MAX) {
                status = TRELLIS_STATUS_OUT_OF_MEMORY;
                goto cleanup;
            }
            match = welded_count++;
            representative[match] = (uint32_t) vertex;
            cells[(size_t) match * 3u] = cell[0];
            cells[(size_t) match * 3u + 1u] = cell[1];
            cells[(size_t) match * 3u + 2u] = cell[2];
            const size_t bucket = segmentation_cell_bucket(
                cell[0], cell[1], cell[2], mask);
            next[match] = buckets[bucket];
            buckets[bucket] = match;
        }
        vertex_to_welded[vertex] = match;
    }
    for (size_t index = 0; index < face_count * 3u; ++index) {
        if ((size_t) triangles[index] >= vertex_count) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        welded_triangles[index] = vertex_to_welded[triangles[index]];
    }
    status = trellis_mesh_segmentation_partition_faces(
        welded_triangles,
        face_count,
        welded_count,
        semantic_labels,
        min_component_faces,
        part_ids_out,
        part_count_out);

cleanup:
    free(welded_triangles);
    free(cells);
    free(vertex_to_welded);
    free(representative);
    free(next);
    free(buckets);
    return status;
}
