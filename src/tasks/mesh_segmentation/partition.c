#include "partition.h"

#include <float.h>
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

static trellis_status partition_faces_internal(
    const uint32_t * triangles,
    size_t face_count,
    size_t vertex_count,
    const uint32_t * semantic_labels,
    size_t min_component_faces,
    uint32_t ** part_ids_out,
    size_t * part_count_out,
    uint32_t ** geometric_shell_ids_out,
    size_t * geometric_shell_count_out) {
    if (part_ids_out != NULL) *part_ids_out = NULL;
    if (part_count_out != NULL) *part_count_out = 0;
    if (geometric_shell_ids_out != NULL) *geometric_shell_ids_out = NULL;
    if (geometric_shell_count_out != NULL) *geometric_shell_count_out = 0;
    const int collect_shells = geometric_shell_ids_out != NULL &&
        geometric_shell_count_out != NULL;
    if (triangles == NULL || semantic_labels == NULL || face_count == 0 ||
        vertex_count == 0 || part_ids_out == NULL || part_count_out == NULL ||
        ((geometric_shell_ids_out == NULL) !=
         (geometric_shell_count_out == NULL)) ||
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
    uint32_t * shell_parent = collect_shells ?
        (uint32_t *) malloc(face_count * sizeof(uint32_t)) : NULL;
    size_t * shell_sizes = collect_shells ?
        (size_t *) malloc(face_count * sizeof(size_t)) : NULL;
    uint32_t * shell_dense = collect_shells ?
        (uint32_t *) malloc(face_count * sizeof(uint32_t)) : NULL;
    uint32_t * shell_output = collect_shells ?
        (uint32_t *) malloc(face_count * sizeof(uint32_t)) : NULL;
    if (edges == NULL || parent == NULL || sizes == NULL ||
        best_neighbor == NULL || best_size == NULL || dense_root == NULL ||
        output == NULL ||
        (collect_shells && (shell_parent == NULL || shell_sizes == NULL ||
                            shell_dense == NULL || shell_output == NULL))) {
        free(edges); free(parent); free(sizes); free(best_neighbor);
        free(best_size); free(dense_root); free(output); free(shell_parent);
        free(shell_sizes); free(shell_dense); free(shell_output);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (size_t face = 0; face < face_count; ++face) {
        parent[face] = (uint32_t) face;
        sizes[face] = 1;
        best_neighbor[face] = UINT32_MAX;
        dense_root[face] = UINT32_MAX;
        if (collect_shells) {
            shell_parent[face] = (uint32_t) face;
            shell_sizes[face] = 1u;
            shell_dense[face] = UINT32_MAX;
        }
        const uint32_t a = triangles[face * 3u];
        const uint32_t b = triangles[face * 3u + 1u];
        const uint32_t c = triangles[face * 3u + 2u];
        if ((size_t) a >= vertex_count || (size_t) b >= vertex_count ||
            (size_t) c >= vertex_count || a == b || b == c || c == a) {
            free(edges); free(parent); free(sizes); free(best_neighbor);
            free(best_size); free(dense_root); free(output); free(shell_parent);
            free(shell_sizes); free(shell_dense); free(shell_output);
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
                if (collect_shells) {
                    partition_union(
                        shell_parent, shell_sizes, left_face, right_face);
                }
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
            if (part_count >= UINT32_MAX) {
                free(edges); free(parent); free(sizes); free(best_neighbor);
                free(best_size); free(dense_root); free(output);
                free(shell_parent); free(shell_sizes); free(shell_dense);
                free(shell_output);
                return TRELLIS_STATUS_OUT_OF_MEMORY;
            }
            dense_root[root] = (uint32_t) part_count++;
        }
        output[face] = dense_root[root];
    }

    size_t shell_count = 0;
    if (collect_shells) {
        for (size_t face = 0; face < face_count; ++face) {
            const uint32_t root = partition_find(shell_parent, (uint32_t) face);
            if (shell_dense[root] == UINT32_MAX) {
                if (shell_count >= UINT32_MAX) {
                    free(edges); free(parent); free(sizes); free(best_neighbor);
                    free(best_size); free(dense_root); free(output);
                    free(shell_parent); free(shell_sizes); free(shell_dense);
                    free(shell_output);
                    return TRELLIS_STATUS_OUT_OF_MEMORY;
                }
                shell_dense[root] = (uint32_t) shell_count++;
            }
            shell_output[face] = shell_dense[root];
        }
    }

    free(edges); free(parent); free(sizes); free(best_neighbor);
    free(best_size); free(dense_root); free(shell_parent); free(shell_sizes);
    free(shell_dense);
    *part_ids_out = output;
    *part_count_out = part_count;
    if (collect_shells) {
        *geometric_shell_ids_out = shell_output;
        *geometric_shell_count_out = shell_count;
    }
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_mesh_segmentation_partition_faces(
    const uint32_t * triangles,
    size_t face_count,
    size_t vertex_count,
    const uint32_t * semantic_labels,
    size_t min_component_faces,
    uint32_t ** part_ids_out,
    size_t * part_count_out) {
    return partition_faces_internal(
        triangles,
        face_count,
        vertex_count,
        semantic_labels,
        min_component_faces,
        part_ids_out,
        part_count_out,
        NULL,
        NULL);
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

/* A shell must pass both scale gates. The values intentionally sit above the
 * observed tail of disconnected tessellation specks while remaining far below
 * a low-poly but model-scale component. They are ratios, so source units do
 * not affect the decision. */
#define SEGMENTATION_SMALL_SHELL_MAX_DIAGONAL_RATIO 0.01
#define SEGMENTATION_SMALL_SHELL_MAX_AREA_RATIO 1e-5

typedef struct segmentation_shell_stats {
    size_t face_count;
    double area;
    double centroid[3];
    double bounds_min[3];
    double bounds_max[3];
    uint32_t dominant_part;
    size_t dominant_part_faces;
    int candidate;
} segmentation_shell_stats;

typedef struct segmentation_part_stats {
    size_t face_count;
    uint32_t shell_id;
    uint32_t semantic_label;
    double bounds_min[3];
    double bounds_max[3];
} segmentation_part_stats;

static void segmentation_bounds_init(double bounds_min[3], double bounds_max[3]) {
    for (size_t axis = 0; axis < 3u; ++axis) {
        bounds_min[axis] = DBL_MAX;
        bounds_max[axis] = -DBL_MAX;
    }
}

static void segmentation_bounds_include(
    double bounds_min[3],
    double bounds_max[3],
    const float * point) {
    for (size_t axis = 0; axis < 3u; ++axis) {
        const double value = point[axis];
        if (value < bounds_min[axis]) bounds_min[axis] = value;
        if (value > bounds_max[axis]) bounds_max[axis] = value;
    }
}

static double segmentation_bounds_diagonal_squared(
    const double bounds_min[3],
    const double bounds_max[3]) {
    double result = 0.0;
    for (size_t axis = 0; axis < 3u; ++axis) {
        const double extent = bounds_max[axis] - bounds_min[axis];
        result += extent * extent;
    }
    return result;
}

static double segmentation_point_aabb_distance_squared(
    const double point[3],
    const double bounds_min[3],
    const double bounds_max[3]) {
    double result = 0.0;
    for (size_t axis = 0; axis < 3u; ++axis) {
        double distance = 0.0;
        if (point[axis] < bounds_min[axis]) {
            distance = bounds_min[axis] - point[axis];
        } else if (point[axis] > bounds_max[axis]) {
            distance = point[axis] - bounds_max[axis];
        }
        result += distance * distance;
    }
    return result;
}

static trellis_status segmentation_postprocess_small_shells(
    const float * positions,
    const uint32_t * triangles,
    size_t face_count,
    size_t vertex_count,
    const uint32_t * semantic_labels,
    size_t min_component_faces,
    trellis_mesh_segmentation_small_part_mode mode,
    const uint32_t * shell_ids,
    size_t shell_count,
    uint32_t * part_ids,
    size_t * part_count_io,
    trellis_mesh_segmentation_small_part_stats * stats_out) {
    const size_t input_part_count = part_count_io != NULL ? *part_count_io : 0u;
    trellis_mesh_segmentation_small_part_stats stats;
    memset(&stats, 0, sizeof(stats));
    stats.input_part_count = input_part_count;
    stats.output_part_count = input_part_count;
    stats.geometric_shell_count = shell_count;
    if (positions == NULL || triangles == NULL || semantic_labels == NULL ||
        shell_ids == NULL || part_ids == NULL || part_count_io == NULL ||
        face_count == 0 || vertex_count == 0 || shell_count == 0 ||
        input_part_count == 0 ||
        (mode != TRELLIS_MESH_SEGMENTATION_SMALL_PART_MERGE &&
         mode != TRELLIS_MESH_SEGMENTATION_SMALL_PART_DISCARD)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (shell_count > SIZE_MAX / sizeof(segmentation_shell_stats) ||
        input_part_count > SIZE_MAX / sizeof(segmentation_part_stats) ||
        input_part_count > SIZE_MAX / sizeof(uint32_t)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    segmentation_shell_stats * shells = (segmentation_shell_stats *)
        calloc(shell_count, sizeof(*shells));
    segmentation_part_stats * parts = (segmentation_part_stats *)
        calloc(input_part_count, sizeof(*parts));
    uint32_t * dense_part = (uint32_t *)
        malloc(input_part_count * sizeof(*dense_part));
    if (shells == NULL || parts == NULL || dense_part == NULL) {
        free(shells); free(parts); free(dense_part);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (size_t shell = 0; shell < shell_count; ++shell) {
        segmentation_bounds_init(
            shells[shell].bounds_min, shells[shell].bounds_max);
        shells[shell].dominant_part = UINT32_MAX;
    }
    for (size_t part = 0; part < input_part_count; ++part) {
        segmentation_bounds_init(parts[part].bounds_min, parts[part].bounds_max);
        parts[part].shell_id = UINT32_MAX;
        dense_part[part] = UINT32_MAX;
    }
    double global_bounds_min[3];
    double global_bounds_max[3];
    segmentation_bounds_init(global_bounds_min, global_bounds_max);
    double global_area = 0.0;
    trellis_status status = TRELLIS_STATUS_OK;

    for (size_t face = 0; face < face_count; ++face) {
        const uint32_t shell_id = shell_ids[face];
        const uint32_t part_id = part_ids[face];
        if ((size_t) shell_id >= shell_count ||
            (size_t) part_id >= input_part_count) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        segmentation_shell_stats * shell = &shells[shell_id];
        segmentation_part_stats * part = &parts[part_id];
        if (part->face_count == 0) {
            part->shell_id = shell_id;
            part->semantic_label = semantic_labels[face];
        } else if (part->shell_id != shell_id) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        ++shell->face_count;
        ++part->face_count;

        const uint32_t ia = triangles[face * 3u];
        const uint32_t ib = triangles[face * 3u + 1u];
        const uint32_t ic = triangles[face * 3u + 2u];
        if ((size_t) ia >= vertex_count || (size_t) ib >= vertex_count ||
            (size_t) ic >= vertex_count) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        const float * a = positions + (size_t) ia * 3u;
        const float * b = positions + (size_t) ib * 3u;
        const float * c = positions + (size_t) ic * 3u;
        for (size_t corner = 0; corner < 3u; ++corner) {
            const float * point = corner == 0 ? a : corner == 1 ? b : c;
            if (!isfinite(point[0]) || !isfinite(point[1]) ||
                !isfinite(point[2])) {
                status = TRELLIS_STATUS_PARSE_ERROR;
                goto cleanup;
            }
            segmentation_bounds_include(
                shell->bounds_min, shell->bounds_max, point);
            segmentation_bounds_include(
                part->bounds_min, part->bounds_max, point);
            segmentation_bounds_include(
                global_bounds_min, global_bounds_max, point);
        }
        const double ab[3] = {
            (double) b[0] - a[0],
            (double) b[1] - a[1],
            (double) b[2] - a[2],
        };
        const double ac[3] = {
            (double) c[0] - a[0],
            (double) c[1] - a[1],
            (double) c[2] - a[2],
        };
        const double cross[3] = {
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        };
        const double area = 0.5 * sqrt(
            cross[0] * cross[0] + cross[1] * cross[1] +
            cross[2] * cross[2]);
        const double updated_shell_area = shell->area + area;
        if (!isfinite(area) || !isfinite(global_area + area) ||
            !isfinite(updated_shell_area)) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        if (area > 0.0) {
            const double weight = area / updated_shell_area;
            for (size_t axis = 0; axis < 3u; ++axis) {
                const double face_centroid =
                    ((double) a[axis] + b[axis] + c[axis]) / 3.0;
                const double updated_centroid = shell->centroid[axis] +
                    (face_centroid - shell->centroid[axis]) * weight;
                if (!isfinite(updated_centroid)) {
                    status = TRELLIS_STATUS_PARSE_ERROR;
                    goto cleanup;
                }
                shell->centroid[axis] = updated_centroid;
            }
        }
        shell->area = updated_shell_area;
        global_area += area;
    }

    for (size_t part_id = 0; part_id < input_part_count; ++part_id) {
        const segmentation_part_stats * part = &parts[part_id];
        if (part->face_count == 0 || (size_t) part->shell_id >= shell_count) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        segmentation_shell_stats * shell = &shells[part->shell_id];
        if (shell->dominant_part == UINT32_MAX ||
            part->face_count > shell->dominant_part_faces ||
            (part->face_count == shell->dominant_part_faces &&
             part_id < shell->dominant_part)) {
            shell->dominant_part = (uint32_t) part_id;
            shell->dominant_part_faces = part->face_count;
        }
    }

    size_t protected_shell = 0;
    for (size_t shell_id = 0; shell_id < shell_count; ++shell_id) {
        const segmentation_shell_stats * shell = &shells[shell_id];
        if (shell->face_count == 0 || shell->dominant_part == UINT32_MAX) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        const segmentation_shell_stats * protected_stats =
            &shells[protected_shell];
        if (shell_id == 0 || shell->area > protected_stats->area ||
            (shell->area == protected_stats->area &&
             shell->face_count > protected_stats->face_count) ||
            (shell->area == protected_stats->area &&
             shell->face_count == protected_stats->face_count &&
             shell_id < protected_shell)) {
            protected_shell = shell_id;
        }
    }

    const double global_diagonal_squared = segmentation_bounds_diagonal_squared(
        global_bounds_min, global_bounds_max);
    /* Degenerate or non-finite source scale cannot support a meaningful
     * relative-size classification. Fall back to KEEP, never "everything is
     * tiny", which also guarantees DISCARD cannot erase the complete mesh. */
    if (min_component_faces <= 1u || !isfinite(global_area) ||
        !(global_area > 0.0) || !isfinite(global_diagonal_squared) ||
        !(global_diagonal_squared > 0.0)) {
        goto cleanup;
    }
    const double diagonal_limit_squared = global_diagonal_squared *
        SEGMENTATION_SMALL_SHELL_MAX_DIAGONAL_RATIO *
        SEGMENTATION_SMALL_SHELL_MAX_DIAGONAL_RATIO;
    const double area_limit =
        global_area * SEGMENTATION_SMALL_SHELL_MAX_AREA_RATIO;
    for (size_t shell_id = 0; shell_id < shell_count; ++shell_id) {
        segmentation_shell_stats * shell = &shells[shell_id];
        const double diagonal_squared = segmentation_bounds_diagonal_squared(
            shell->bounds_min, shell->bounds_max);
        shell->candidate = shell_id != protected_shell &&
            shell->face_count < min_component_faces &&
            isfinite(diagonal_squared) &&
            diagonal_squared <= diagonal_limit_squared &&
            shell->area <= area_limit;
        if (!shell->candidate) continue;
        ++stats.candidate_shell_count;
        stats.affected_face_count += shell->face_count;
    }
    if (stats.candidate_shell_count == 0) goto cleanup;
    for (size_t part_id = 0; part_id < input_part_count; ++part_id) {
        if (shells[parts[part_id].shell_id].candidate) {
            ++stats.candidate_part_count;
        }
    }

    if (mode == TRELLIS_MESH_SEGMENTATION_SMALL_PART_MERGE) {
        for (size_t shell_id = 0; shell_id < shell_count; ++shell_id) {
            segmentation_shell_stats * shell = &shells[shell_id];
            if (!shell->candidate) continue;
            double centroid[3];
            for (size_t axis = 0; axis < 3u; ++axis) {
                centroid[axis] = shell->area > 0.0 ?
                    shell->centroid[axis] :
                    (shell->bounds_min[axis] + shell->bounds_max[axis]) * 0.5;
            }
            const uint32_t preferred_label =
                parts[shell->dominant_part].semantic_label;
            uint32_t best_part = UINT32_MAX;
            double best_distance = DBL_MAX;
            int best_matches_label = 0;
            for (size_t part_id = 0; part_id < input_part_count; ++part_id) {
                const segmentation_part_stats * target = &parts[part_id];
                if (shells[target->shell_id].candidate) continue;
                const int matches_label =
                    target->semantic_label == preferred_label;
                const double distance =
                    segmentation_point_aabb_distance_squared(
                        centroid, target->bounds_min, target->bounds_max);
                if (best_part == UINT32_MAX || distance < best_distance ||
                    (distance == best_distance &&
                     (matches_label > best_matches_label ||
                      (matches_label == best_matches_label &&
                       part_id < best_part)))) {
                    best_part = (uint32_t) part_id;
                    best_distance = distance;
                    best_matches_label = matches_label;
                }
            }
            if (best_part == UINT32_MAX) {
                status = TRELLIS_STATUS_ERROR;
                goto cleanup;
            }
            /* Reuse dominant_part after classification as the immutable merge
             * anchor. Targets are always retained, so chains and cycles are
             * impossible and every face in this shell receives one target. */
            shell->dominant_part = best_part;
        }
    }

    for (size_t face = 0; face < face_count; ++face) {
        segmentation_shell_stats * shell = &shells[shell_ids[face]];
        if (!shell->candidate) continue;
        part_ids[face] = mode == TRELLIS_MESH_SEGMENTATION_SMALL_PART_MERGE ?
            shell->dominant_part : UINT32_MAX;
    }
    size_t output_part_count = 0;
    for (size_t face = 0; face < face_count; ++face) {
        const uint32_t old_part = part_ids[face];
        if (old_part == UINT32_MAX) continue;
        if ((size_t) old_part >= input_part_count) {
            status = TRELLIS_STATUS_ERROR;
            goto cleanup;
        }
        if (dense_part[old_part] == UINT32_MAX) {
            if (output_part_count >= UINT32_MAX) {
                status = TRELLIS_STATUS_OUT_OF_MEMORY;
                goto cleanup;
            }
            dense_part[old_part] = (uint32_t) output_part_count++;
        }
        part_ids[face] = dense_part[old_part];
    }
    if (output_part_count == 0) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    *part_count_io = output_part_count;
    stats.output_part_count = output_part_count;

cleanup:
    if (stats_out != NULL && status == TRELLIS_STATUS_OK) *stats_out = stats;
    free(dense_part);
    free(parts);
    free(shells);
    return status;
}

trellis_status trellis_mesh_segmentation_partition_faces_geometric_ex(
    const float * positions,
    const uint32_t * triangles,
    size_t face_count,
    size_t vertex_count,
    const uint32_t * semantic_labels,
    size_t min_component_faces,
    float weld_tolerance,
    trellis_mesh_segmentation_small_part_mode small_part_mode,
    uint32_t ** part_ids_out,
    size_t * part_count_out,
    trellis_mesh_segmentation_small_part_stats * stats_out) {
    if (part_ids_out != NULL) *part_ids_out = NULL;
    if (part_count_out != NULL) *part_count_out = 0;
    if (stats_out != NULL) memset(stats_out, 0, sizeof(*stats_out));
    if (positions == NULL || triangles == NULL || semantic_labels == NULL ||
        face_count == 0 || vertex_count == 0 || vertex_count > UINT32_MAX ||
        face_count > UINT32_MAX ||
        face_count > SIZE_MAX / (3u * sizeof(uint32_t)) ||
        vertex_count > SIZE_MAX / sizeof(uint32_t) ||
        vertex_count > SIZE_MAX / (3u * sizeof(int64_t)) ||
        part_ids_out == NULL || part_count_out == NULL ||
        (small_part_mode != TRELLIS_MESH_SEGMENTATION_SMALL_PART_KEEP &&
         small_part_mode != TRELLIS_MESH_SEGMENTATION_SMALL_PART_MERGE &&
         small_part_mode != TRELLIS_MESH_SEGMENTATION_SMALL_PART_DISCARD) ||
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
    uint32_t * shell_ids = NULL;
    size_t shell_count = 0;
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
    size_t adjacency_vertex_count = (size_t) welded_count;
    for (size_t face = 0; face < face_count; ++face) {
        const uint32_t original[3] = {
            triangles[face * 3u],
            triangles[face * 3u + 1u],
            triangles[face * 3u + 2u],
        };
        if ((size_t) original[0] >= vertex_count ||
            (size_t) original[1] >= vertex_count ||
            (size_t) original[2] >= vertex_count ||
            original[0] == original[1] || original[1] == original[2] ||
            original[2] == original[0]) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        const uint32_t welded[3] = {
            vertex_to_welded[original[0]],
            vertex_to_welded[original[1]],
            vertex_to_welded[original[2]],
        };
        for (size_t corner = 0; corner < 3u; ++corner) {
            int duplicates_earlier_corner = 0;
            for (size_t earlier = 0; earlier < corner; ++earlier) {
                if (welded[corner] == welded[earlier]) {
                    duplicates_earlier_corner = 1;
                    break;
                }
            }
            if (duplicates_earlier_corner) {
                /* Keep tolerance-collapsed corners from turning an otherwise
                 * valid source face into a self-edge. Each synthetic ID is
                 * private to one corner, while all other geometric welding is
                 * retained for cross-face adjacency. */
                if (adjacency_vertex_count == SIZE_MAX ||
                    adjacency_vertex_count > (size_t) UINT32_MAX) {
                    status = TRELLIS_STATUS_OUT_OF_MEMORY;
                    goto cleanup;
                }
                welded_triangles[face * 3u + corner] =
                    (uint32_t) adjacency_vertex_count++;
            } else {
                welded_triangles[face * 3u + corner] = welded[corner];
            }
        }
    }
    status = partition_faces_internal(
        welded_triangles,
        face_count,
        adjacency_vertex_count,
        semantic_labels,
        min_component_faces,
        part_ids_out,
        part_count_out,
        small_part_mode == TRELLIS_MESH_SEGMENTATION_SMALL_PART_KEEP ?
            NULL : &shell_ids,
        small_part_mode == TRELLIS_MESH_SEGMENTATION_SMALL_PART_KEEP ?
            NULL : &shell_count);
    if (status == TRELLIS_STATUS_OK &&
        small_part_mode == TRELLIS_MESH_SEGMENTATION_SMALL_PART_KEEP) {
        if (stats_out != NULL) {
            stats_out->input_part_count = *part_count_out;
            stats_out->output_part_count = *part_count_out;
        }
    } else if (status == TRELLIS_STATUS_OK) {
        status = segmentation_postprocess_small_shells(
            positions,
            triangles,
            face_count,
            vertex_count,
            semantic_labels,
            min_component_faces,
            small_part_mode,
            shell_ids,
            shell_count,
            *part_ids_out,
            part_count_out,
            stats_out);
    }

cleanup:
    if (status != TRELLIS_STATUS_OK) {
        free(*part_ids_out);
        *part_ids_out = NULL;
        *part_count_out = 0;
        if (stats_out != NULL) memset(stats_out, 0, sizeof(*stats_out));
    }
    free(shell_ids);
    free(welded_triangles);
    free(cells);
    free(vertex_to_welded);
    free(representative);
    free(next);
    free(buckets);
    return status;
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
    return trellis_mesh_segmentation_partition_faces_geometric_ex(
        positions,
        triangles,
        face_count,
        vertex_count,
        semantic_labels,
        min_component_faces,
        weld_tolerance,
        TRELLIS_MESH_SEGMENTATION_SMALL_PART_KEEP,
        part_ids_out,
        part_count_out,
        NULL);
}
