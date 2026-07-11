#include "rigged_gltf.h"

#include "cgltf_write.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    RIGGED_VIEW_POSITIONS = 0,
    RIGGED_VIEW_NORMALS,
    RIGGED_VIEW_JOINTS,
    RIGGED_VIEW_WEIGHTS,
    RIGGED_VIEW_INDICES,
    RIGGED_VIEW_INVERSE_BINDS,
    RIGGED_VIEW_COUNT,
};

enum {
    RIGGED_ACCESSORS_PER_PRIMITIVE = 5,
    RIGGED_ATTRIBUTES_PER_PRIMITIVE = 4,
    RIGGED_MAX_INTERPOLATION_NEIGHBORS = 8,
};

typedef struct rigged_kd_node {
    uint32_t point;
    size_t left;
    size_t right;
    unsigned char axis;
} rigged_kd_node;

typedef struct rigged_kd_tree {
    const float * points;
    uint32_t * order;
    rigged_kd_node * nodes;
    size_t count;
    size_t root;
} rigged_kd_tree;

typedef struct rigged_neighbors {
    uint32_t points[RIGGED_MAX_INTERPOLATION_NEIGHBORS];
    double squared_distances[RIGGED_MAX_INTERPOLATION_NEIGHBORS];
    size_t count;
    size_t capacity;
} rigged_neighbors;

static void rigged_set_error(
    char * output,
    size_t output_size,
    const char * format,
    ...) {
    if (output == NULL || output_size == 0) {
        return;
    }
    va_list args;
    va_start(args, format);
    vsnprintf(output, output_size, format, args);
    va_end(args);
    output[output_size - 1u] = '\0';
}

static int rigged_add_size(size_t a, size_t b, size_t * result) {
    if (a > SIZE_MAX - b) {
        return 0;
    }
    *result = a + b;
    return 1;
}

static int rigged_multiply_size(size_t a, size_t b, size_t * result) {
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *result = a * b;
    return 1;
}

static int rigged_path_has_glb_extension(const char * path) {
    if (path == NULL) {
        return 0;
    }
    const size_t length = strlen(path);
    if (length < 4u) {
        return 0;
    }
    const char * extension = path + length - 4u;
    return extension[0] == '.' &&
        tolower((unsigned char) extension[1]) == 'g' &&
        tolower((unsigned char) extension[2]) == 'l' &&
        tolower((unsigned char) extension[3]) == 'b';
}

static int rigged_finite3(const float value[3]) {
    return isfinite(value[0]) && isfinite(value[1]) && isfinite(value[2]);
}

static int rigged_point_less(
    const float * points,
    uint32_t left,
    uint32_t right,
    unsigned axis) {
    const float a = points[(size_t) left * 3u + axis];
    const float b = points[(size_t) right * 3u + axis];
    return a < b || (a == b && left < right);
}

static void rigged_swap_u32(uint32_t * left, uint32_t * right) {
    const uint32_t temporary = *left;
    *left = *right;
    *right = temporary;
}

static size_t rigged_kd_partition(
    uint32_t * order,
    size_t first,
    size_t last,
    size_t pivot,
    unsigned axis,
    const float * points) {
    const uint32_t pivot_point = order[pivot];
    rigged_swap_u32(&order[pivot], &order[last - 1u]);
    size_t store = first;
    for (size_t i = first; i + 1u < last; ++i) {
        if (rigged_point_less(points, order[i], pivot_point, axis)) {
            rigged_swap_u32(&order[store], &order[i]);
            ++store;
        }
    }
    rigged_swap_u32(&order[store], &order[last - 1u]);
    return store;
}

static void rigged_kd_select(
    uint32_t * order,
    size_t first,
    size_t last,
    size_t selected,
    unsigned axis,
    const float * points) {
    while (last - first > 1u) {
        const size_t pivot = first + (last - first) / 2u;
        const size_t partition = rigged_kd_partition(
            order, first, last, pivot, axis, points);
        if (partition == selected) {
            return;
        }
        if (selected < partition) {
            last = partition;
        } else {
            first = partition + 1u;
        }
    }
}

static size_t rigged_kd_build_range(
    rigged_kd_tree * tree,
    size_t first,
    size_t last) {
    if (first == last) {
        return SIZE_MAX;
    }
    float minimum[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float maximum[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (size_t i = first; i < last; ++i) {
        const float * point = tree->points + (size_t) tree->order[i] * 3u;
        for (unsigned axis = 0; axis < 3u; ++axis) {
            if (point[axis] < minimum[axis]) minimum[axis] = point[axis];
            if (point[axis] > maximum[axis]) maximum[axis] = point[axis];
        }
    }
    unsigned axis = 0;
    if (maximum[1] - minimum[1] > maximum[axis] - minimum[axis]) axis = 1;
    if (maximum[2] - minimum[2] > maximum[axis] - minimum[axis]) axis = 2;

    const size_t middle = first + (last - first) / 2u;
    rigged_kd_select(tree->order, first, last, middle, axis, tree->points);
    tree->nodes[middle].point = tree->order[middle];
    tree->nodes[middle].axis = (unsigned char) axis;
    tree->nodes[middle].left = rigged_kd_build_range(tree, first, middle);
    tree->nodes[middle].right = rigged_kd_build_range(tree, middle + 1u, last);
    return middle;
}

static trellis_status rigged_kd_tree_init(
    const float * points,
    size_t count,
    rigged_kd_tree * tree,
    char * error,
    size_t error_size) {
    memset(tree, 0, sizeof(*tree));
    tree->root = SIZE_MAX;
    if (count == 0 || count > UINT32_MAX) {
        rigged_set_error(error, error_size, "sample point count %zu is unsupported", count);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    size_t order_bytes = 0;
    size_t node_bytes = 0;
    if (!rigged_multiply_size(count, sizeof(*tree->order), &order_bytes) ||
        !rigged_multiply_size(count, sizeof(*tree->nodes), &node_bytes)) {
        rigged_set_error(error, error_size, "sample spatial index size overflow");
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    tree->order = (uint32_t *) malloc(order_bytes);
    tree->nodes = (rigged_kd_node *) malloc(node_bytes);
    if (tree->order == NULL || tree->nodes == NULL) {
        free(tree->nodes);
        free(tree->order);
        memset(tree, 0, sizeof(*tree));
        tree->root = SIZE_MAX;
        rigged_set_error(error, error_size, "out of memory building sample spatial index");
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    tree->points = points;
    tree->count = count;
    for (size_t i = 0; i < count; ++i) {
        tree->order[i] = (uint32_t) i;
    }
    tree->root = rigged_kd_build_range(tree, 0, count);
    return TRELLIS_STATUS_OK;
}

static void rigged_kd_tree_free(rigged_kd_tree * tree) {
    if (tree == NULL) return;
    free(tree->nodes);
    free(tree->order);
    memset(tree, 0, sizeof(*tree));
    tree->root = SIZE_MAX;
}

static int rigged_neighbor_better(
    double left_distance,
    uint32_t left_point,
    double right_distance,
    uint32_t right_point) {
    return left_distance < right_distance ||
        (left_distance == right_distance && left_point < right_point);
}

static void rigged_neighbors_insert(
    rigged_neighbors * neighbors,
    uint32_t point,
    double squared_distance) {
    size_t position = 0;
    while (position < neighbors->count &&
        !rigged_neighbor_better(
            squared_distance,
            point,
            neighbors->squared_distances[position],
            neighbors->points[position])) {
        ++position;
    }
    if (neighbors->count == neighbors->capacity && position == neighbors->count) {
        return;
    }
    size_t new_count = neighbors->count;
    if (new_count < neighbors->capacity) {
        ++new_count;
    }
    for (size_t i = new_count; i > position + 1u; --i) {
        neighbors->points[i - 1u] = neighbors->points[i - 2u];
        neighbors->squared_distances[i - 1u] =
            neighbors->squared_distances[i - 2u];
    }
    if (position < neighbors->capacity) {
        neighbors->points[position] = point;
        neighbors->squared_distances[position] = squared_distance;
        neighbors->count = new_count;
    }
}

static void rigged_kd_query_node(
    const rigged_kd_tree * tree,
    size_t node_index,
    const float query[3],
    rigged_neighbors * neighbors) {
    if (node_index == SIZE_MAX) {
        return;
    }
    const rigged_kd_node * node = &tree->nodes[node_index];
    const float * point = tree->points + (size_t) node->point * 3u;
    const double dx = (double) query[0] - point[0];
    const double dy = (double) query[1] - point[1];
    const double dz = (double) query[2] - point[2];
    const double squared_distance = dx * dx + dy * dy + dz * dz;
    rigged_neighbors_insert(neighbors, node->point, squared_distance);

    const double difference = (double) query[node->axis] - point[node->axis];
    const size_t near_node = difference <= 0.0 ? node->left : node->right;
    const size_t far_node = difference <= 0.0 ? node->right : node->left;
    rigged_kd_query_node(tree, near_node, query, neighbors);
    const double worst = neighbors->count < neighbors->capacity ?
        DBL_MAX : neighbors->squared_distances[neighbors->count - 1u];
    if (difference * difference <= worst) {
        rigged_kd_query_node(tree, far_node, query, neighbors);
    }
}

static void rigged_kd_query(
    const rigged_kd_tree * tree,
    const float query[3],
    size_t count,
    rigged_neighbors * neighbors) {
    memset(neighbors, 0, sizeof(*neighbors));
    neighbors->capacity = count < tree->count ? count : tree->count;
    rigged_kd_query_node(tree, tree->root, query, neighbors);
}

static void rigged_top4_insert(
    double value,
    uint16_t joint,
    double top_values[4],
    uint16_t top_joints[4]) {
    size_t position = 0;
    while (position < 4u &&
        (value < top_values[position] ||
         (value == top_values[position] && joint >= top_joints[position]))) {
        ++position;
    }
    if (position == 4u) {
        return;
    }
    for (size_t i = 3u; i > position; --i) {
        top_values[i] = top_values[i - 1u];
        top_joints[i] = top_joints[i - 1u];
    }
    top_values[position] = value;
    top_joints[position] = joint;
}

static void rigged_finish_top4(
    const double top_values[4],
    const uint16_t top_joints[4],
    uint16_t output_joints[4],
    float output_weights[4]) {
    double sum = 0.0;
    for (size_t i = 0; i < 4u; ++i) {
        if (top_values[i] > 0.0) {
            sum += top_values[i];
        }
    }
    if (!(sum > 0.0) || !isfinite(sum)) {
        output_joints[0] = 0;
        output_joints[1] = 0;
        output_joints[2] = 0;
        output_joints[3] = 0;
        output_weights[0] = 1.0f;
        output_weights[1] = 0.0f;
        output_weights[2] = 0.0f;
        output_weights[3] = 0.0f;
        return;
    }
    float float_sum = 0.0f;
    for (size_t i = 0; i < 4u; ++i) {
        output_joints[i] = top_values[i] >= 0.0 ? top_joints[i] : 0;
        output_weights[i] = top_values[i] > 0.0 ?
            (float) (top_values[i] / sum) : 0.0f;
        float_sum += output_weights[i];
    }
    if (float_sum > 0.0f && isfinite(float_sum)) {
        for (size_t i = 0; i < 4u; ++i) {
            output_weights[i] /= float_sum;
        }
    } else {
        output_joints[0] = 0;
        output_joints[1] = 0;
        output_joints[2] = 0;
        output_joints[3] = 0;
        output_weights[0] = 1.0f;
        output_weights[1] = 0.0f;
        output_weights[2] = 0.0f;
        output_weights[3] = 0.0f;
    }
}

static trellis_status rigged_validate_inputs(
    const char * path,
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_rigging_normalization * normalization,
    const trellis_tokenskin_skeleton * skeleton,
    const trellis_mesh_rigging_dense_skin_weights * dense_weights,
    size_t * root_count_out,
    char * error,
    size_t error_size) {
    if (!rigged_path_has_glb_extension(path)) {
        rigged_set_error(error, error_size, "rigged output path must end in .glb");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (asset == NULL || normalization == NULL || skeleton == NULL ||
        dense_weights == NULL || asset->positions == NULL || asset->normals == NULL ||
        asset->triangles == NULL || asset->primitives == NULL ||
        skeleton->joints_xyz == NULL || skeleton->parents == NULL ||
        dense_weights->values == NULL) {
        rigged_set_error(error, error_size, "rigged GLB input contains a null required field");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (asset->vertex_count == 0 || asset->triangle_count == 0 || asset->primitive_count == 0) {
        rigged_set_error(error, error_size, "rigged GLB needs non-empty geometry and primitive ranges");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (skeleton->joint_count == 0 || skeleton->joint_count > (size_t) UINT16_MAX + 1u) {
        rigged_set_error(
            error,
            error_size,
            "joint count %zu cannot be represented by JOINTS_0 UNSIGNED_SHORT",
            skeleton->joint_count);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (dense_weights->joint_count != skeleton->joint_count ||
        dense_weights->point_count == 0) {
        rigged_set_error(
            error,
            error_size,
            "dense weights shape %zux%zu does not match %zu skeleton joints",
            dense_weights->point_count,
            dense_weights->joint_count,
            skeleton->joint_count);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (dense_weights->sample_positions == NULL &&
        dense_weights->point_count != asset->vertex_count) {
        rigged_set_error(
            error,
            error_size,
            "vertex weights have %zu rows but flattened asset has %zu vertices",
            dense_weights->point_count,
            asset->vertex_count);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (dense_weights->sample_positions != NULL &&
        dense_weights->interpolation_neighbors > RIGGED_MAX_INTERPOLATION_NEIGHBORS) {
        rigged_set_error(
            error,
            error_size,
            "sample interpolation supports at most %u neighbours",
            RIGGED_MAX_INTERPOLATION_NEIGHBORS);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!rigged_finite3(normalization->center) ||
        !isfinite(normalization->scale) || normalization->scale <= 0.0f ||
        !isfinite(normalization->inverse_scale) || normalization->inverse_scale <= 0.0f) {
        rigged_set_error(error, error_size, "normalization has non-finite or non-positive scale data");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    size_t expected_vertex = 0;
    size_t expected_triangle = 0;
    for (size_t primitive = 0; primitive < asset->primitive_count; ++primitive) {
        const trellis_mesh_rigging_primitive_range * range = &asset->primitives[primitive];
        if (range->first_vertex != expected_vertex ||
            range->first_triangle != expected_triangle ||
            range->vertex_count == 0 || range->triangle_count == 0 ||
            !rigged_add_size(expected_vertex, range->vertex_count, &expected_vertex) ||
            !rigged_add_size(expected_triangle, range->triangle_count, &expected_triangle) ||
            expected_vertex > asset->vertex_count || expected_triangle > asset->triangle_count ||
            range->vertex_count > UINT32_MAX) {
            rigged_set_error(
                error,
                error_size,
                "primitive range %zu is empty, unordered, or outside flattened geometry",
                primitive);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        for (size_t triangle = range->first_triangle;
             triangle < range->first_triangle + range->triangle_count;
             ++triangle) {
            for (size_t corner = 0; corner < 3u; ++corner) {
                const uint32_t index = asset->triangles[triangle * 3u + corner];
                if ((size_t) index < range->first_vertex ||
                    (size_t) index >= range->first_vertex + range->vertex_count) {
                    rigged_set_error(
                        error,
                        error_size,
                        "triangle %zu index %u crosses primitive range %zu",
                        triangle,
                        index,
                        primitive);
                    return TRELLIS_STATUS_INVALID_ARGUMENT;
                }
            }
        }
    }
    if (expected_vertex != asset->vertex_count || expected_triangle != asset->triangle_count) {
        rigged_set_error(error, error_size, "primitive ranges do not cover all flattened geometry");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (size_t vertex = 0; vertex < asset->vertex_count; ++vertex) {
        if (!rigged_finite3(asset->positions + vertex * 3u) ||
            !rigged_finite3(asset->normals + vertex * 3u)) {
            rigged_set_error(error, error_size, "vertex %zu contains non-finite geometry", vertex);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }

    unsigned char * parent_state = (unsigned char *) calloc(skeleton->joint_count, 1u);
    if (parent_state == NULL) {
        rigged_set_error(error, error_size, "out of memory validating skeleton hierarchy");
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    size_t root_count = 0;
    for (size_t joint = 0; joint < skeleton->joint_count; ++joint) {
        if (!rigged_finite3(skeleton->joints_xyz + joint * 3u)) {
            free(parent_state);
            rigged_set_error(error, error_size, "joint %zu has a non-finite normalized position", joint);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        const int32_t parent = skeleton->parents[joint];
        if (parent < -1 || (parent >= 0 && (size_t) parent >= skeleton->joint_count)) {
            free(parent_state);
            rigged_set_error(error, error_size, "joint %zu has invalid parent %d", joint, parent);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        if (parent == -1) ++root_count;
    }
    for (size_t start = 0; start < skeleton->joint_count; ++start) {
        size_t current = start;
        while (parent_state[current] == 0u) {
            parent_state[current] = 1u;
            const int32_t parent = skeleton->parents[current];
            if (parent < 0) break;
            current = (size_t) parent;
        }
        if (parent_state[current] == 1u && skeleton->parents[current] >= 0) {
            free(parent_state);
            rigged_set_error(error, error_size, "skeleton parent graph contains a cycle");
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        current = start;
        while (parent_state[current] == 1u) {
            parent_state[current] = 2u;
            const int32_t parent = skeleton->parents[current];
            if (parent < 0) break;
            current = (size_t) parent;
        }
    }
    free(parent_state);
    if (root_count == 0) {
        rigged_set_error(error, error_size, "skeleton has no root joint");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    size_t weight_elements = 0;
    if (!rigged_multiply_size(
            dense_weights->point_count,
            dense_weights->joint_count,
            &weight_elements)) {
        rigged_set_error(error, error_size, "dense weight shape overflows address space");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (size_t element = 0; element < weight_elements; ++element) {
        if (!isfinite(dense_weights->values[element])) {
            rigged_set_error(error, error_size, "dense skin weight %zu is non-finite", element);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }
    if (dense_weights->sample_positions != NULL) {
        if (dense_weights->point_count > UINT32_MAX) {
            rigged_set_error(error, error_size, "too many sampled skin rows for spatial interpolation");
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        for (size_t point = 0; point < dense_weights->point_count; ++point) {
            if (!rigged_finite3(dense_weights->sample_positions + point * 3u)) {
                rigged_set_error(error, error_size, "skin sample position %zu is non-finite", point);
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    *root_count_out = root_count;
    return TRELLIS_STATUS_OK;
}

static trellis_status rigged_build_vertex_influences(
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_rigging_normalization * normalization,
    const trellis_mesh_rigging_dense_skin_weights * dense_weights,
    uint16_t ** joints_out,
    float ** weights_out,
    char * error,
    size_t error_size) {
    size_t influence_count = 0;
    size_t joints_bytes = 0;
    size_t weights_bytes = 0;
    if (!rigged_multiply_size(asset->vertex_count, 4u, &influence_count) ||
        !rigged_multiply_size(influence_count, sizeof(uint16_t), &joints_bytes) ||
        !rigged_multiply_size(influence_count, sizeof(float), &weights_bytes)) {
        rigged_set_error(error, error_size, "vertex influence allocation size overflow");
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    uint16_t * output_joints = (uint16_t *) malloc(joints_bytes);
    float * output_weights = (float *) malloc(weights_bytes);
    if (output_joints == NULL || output_weights == NULL) {
        free(output_weights);
        free(output_joints);
        rigged_set_error(error, error_size, "out of memory reducing dense skin weights");
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    rigged_kd_tree tree;
    memset(&tree, 0, sizeof(tree));
    tree.root = SIZE_MAX;
    size_t neighbor_count = dense_weights->interpolation_neighbors;
    if (neighbor_count == 0) neighbor_count = RIGGED_MAX_INTERPOLATION_NEIGHBORS;
    trellis_status status = TRELLIS_STATUS_OK;
    if (dense_weights->sample_positions != NULL) {
        status = rigged_kd_tree_init(
            dense_weights->sample_positions,
            dense_weights->point_count,
            &tree,
            error,
            error_size);
    }

    for (size_t vertex = 0; status == TRELLIS_STATUS_OK && vertex < asset->vertex_count; ++vertex) {
        double top_values[4] = { -1.0, -1.0, -1.0, -1.0 };
        uint16_t top_joints[4] = { UINT16_MAX, UINT16_MAX, UINT16_MAX, UINT16_MAX };
        if (dense_weights->sample_positions == NULL) {
            const float * row = dense_weights->values + vertex * dense_weights->joint_count;
            for (size_t joint = 0; joint < dense_weights->joint_count; ++joint) {
                const double value = row[joint] > 0.0f ? row[joint] : 0.0;
                rigged_top4_insert(value, (uint16_t) joint, top_values, top_joints);
            }
        } else {
            float query[3];
            trellis_mesh_rigging_normalize_point(
                normalization, asset->positions + vertex * 3u, query);
            rigged_neighbors neighbors;
            rigged_kd_query(&tree, query, neighbor_count, &neighbors);
            if (neighbors.count == 0) {
                status = TRELLIS_STATUS_ERROR;
                rigged_set_error(error, error_size, "sample spatial index returned no neighbours");
                break;
            }
            double interpolation_weights[RIGGED_MAX_INTERPOLATION_NEIGHBORS];
            for (size_t neighbor = 0; neighbor < neighbors.count; ++neighbor) {
                interpolation_weights[neighbor] =
                    1.0 / (sqrt(neighbors.squared_distances[neighbor]) + 1e-8);
            }
            for (size_t joint = 0; joint < dense_weights->joint_count; ++joint) {
                double value = 0.0;
                for (size_t neighbor = 0; neighbor < neighbors.count; ++neighbor) {
                    const size_t row = neighbors.points[neighbor];
                    const float source =
                        dense_weights->values[row * dense_weights->joint_count + joint];
                    if (source > 0.0f) {
                        value += interpolation_weights[neighbor] * source;
                    }
                }
                rigged_top4_insert(value, (uint16_t) joint, top_values, top_joints);
            }
        }
        if (status == TRELLIS_STATUS_OK) {
            rigged_finish_top4(
                top_values,
                top_joints,
                output_joints + vertex * 4u,
                output_weights + vertex * 4u);
        }
    }
    rigged_kd_tree_free(&tree);
    if (status != TRELLIS_STATUS_OK) {
        free(output_weights);
        free(output_joints);
        return status;
    }
    *joints_out = output_joints;
    *weights_out = output_weights;
    return TRELLIS_STATUS_OK;
}

static void rigged_write_u16_le(unsigned char * output, uint16_t value) {
    output[0] = (unsigned char) (value & 0xffu);
    output[1] = (unsigned char) ((value >> 8u) & 0xffu);
}

static void rigged_write_u32_le(unsigned char * output, uint32_t value) {
    output[0] = (unsigned char) (value & 0xffu);
    output[1] = (unsigned char) ((value >> 8u) & 0xffu);
    output[2] = (unsigned char) ((value >> 16u) & 0xffu);
    output[3] = (unsigned char) ((value >> 24u) & 0xffu);
}

static void rigged_write_f32_le(unsigned char * output, float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    rigged_write_u32_le(output, bits);
}

static int rigged_layout_region(
    size_t * cursor,
    size_t size,
    size_t * offset_out) {
    const size_t aligned = (*cursor + 3u) & ~(size_t) 3u;
    if (aligned < *cursor || aligned > SIZE_MAX - size) {
        return 0;
    }
    *offset_out = aligned;
    *cursor = aligned + size;
    return 1;
}

static const char * rigged_cgltf_result_name(cgltf_result result) {
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

static trellis_status rigged_verify_written_glb(
    const char * path,
    char * error,
    size_t error_size) {
    cgltf_options options;
    memset(&options, 0, sizeof(options));
    cgltf_data * data = NULL;
    cgltf_result result = cgltf_parse_file(&options, path, &data);
    if (result == cgltf_result_success) {
        result = cgltf_load_buffers(&options, data, path);
    }
    if (result == cgltf_result_success) {
        result = cgltf_validate(data);
    }
    cgltf_free(data);
    if (result == cgltf_result_success) {
        return TRELLIS_STATUS_OK;
    }
    rigged_set_error(
        error,
        error_size,
        "written GLB failed cgltf round-trip validation: %s",
        rigged_cgltf_result_name(result));
    return result == cgltf_result_out_of_memory ?
        TRELLIS_STATUS_OUT_OF_MEMORY : TRELLIS_STATUS_IO_ERROR;
}

trellis_status trellis_mesh_rigging_write_rigged_glb(
    const char * path,
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_rigging_normalization * normalization,
    const trellis_tokenskin_skeleton * skeleton,
    const trellis_mesh_rigging_dense_skin_weights * dense_weights,
    char * error_out,
    size_t error_size) {
    if (error_out != NULL && error_size > 0) error_out[0] = '\0';
    size_t root_count = 0;
    trellis_status status = rigged_validate_inputs(
        path,
        asset,
        normalization,
        skeleton,
        dense_weights,
        &root_count,
        error_out,
        error_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (sizeof(float) != 4u) {
        rigged_set_error(error_out, error_size, "GLB writer requires 32-bit IEEE float storage");
        return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }

    uint16_t * vertex_joints = NULL;
    float * vertex_weights = NULL;
    status = rigged_build_vertex_influences(
        asset,
        normalization,
        dense_weights,
        &vertex_joints,
        &vertex_weights,
        error_out,
        error_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    unsigned char * binary = NULL;
    cgltf_accessor * accessors = NULL;
    cgltf_primitive * primitives = NULL;
    cgltf_attribute * attributes = NULL;
    cgltf_node * nodes = NULL;
    cgltf_node ** skin_joints = NULL;
    cgltf_node ** child_links = NULL;
    cgltf_node ** scene_nodes = NULL;
    size_t * child_counts = NULL;
    size_t * child_offsets = NULL;
    size_t * child_fill = NULL;
    float * world_joints = NULL;
    char * joint_names = NULL;

    size_t position_bytes = 0;
    size_t normal_bytes = 0;
    size_t joint_bytes = 0;
    size_t weight_bytes = 0;
    size_t index_count = 0;
    size_t index_bytes = 0;
    size_t inverse_bind_bytes = 0;
    if (!rigged_multiply_size(asset->vertex_count, 3u * sizeof(float), &position_bytes) ||
        !rigged_multiply_size(asset->vertex_count, 3u * sizeof(float), &normal_bytes) ||
        !rigged_multiply_size(asset->vertex_count, 4u * sizeof(uint16_t), &joint_bytes) ||
        !rigged_multiply_size(asset->vertex_count, 4u * sizeof(float), &weight_bytes) ||
        !rigged_multiply_size(asset->triangle_count, 3u, &index_count) ||
        !rigged_multiply_size(index_count, sizeof(uint32_t), &index_bytes) ||
        !rigged_multiply_size(skeleton->joint_count, 16u * sizeof(float), &inverse_bind_bytes)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        rigged_set_error(error_out, error_size, "GLB binary layout size overflow");
        goto cleanup;
    }

    size_t view_offsets[RIGGED_VIEW_COUNT] = { 0 };
    size_t binary_size = 0;
    if (!rigged_layout_region(&binary_size, position_bytes, &view_offsets[RIGGED_VIEW_POSITIONS]) ||
        !rigged_layout_region(&binary_size, normal_bytes, &view_offsets[RIGGED_VIEW_NORMALS]) ||
        !rigged_layout_region(&binary_size, joint_bytes, &view_offsets[RIGGED_VIEW_JOINTS]) ||
        !rigged_layout_region(&binary_size, weight_bytes, &view_offsets[RIGGED_VIEW_WEIGHTS]) ||
        !rigged_layout_region(&binary_size, index_bytes, &view_offsets[RIGGED_VIEW_INDICES]) ||
        !rigged_layout_region(&binary_size, inverse_bind_bytes, &view_offsets[RIGGED_VIEW_INVERSE_BINDS]) ||
        binary_size > UINT32_MAX) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        rigged_set_error(error_out, error_size, "GLB binary chunk exceeds the 32-bit GLB limit");
        goto cleanup;
    }
    binary = (unsigned char *) calloc(binary_size, 1u);
    if (binary == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        rigged_set_error(error_out, error_size, "out of memory allocating GLB binary chunk");
        goto cleanup;
    }
    for (size_t element = 0; element < asset->vertex_count * 3u; ++element) {
        rigged_write_f32_le(
            binary + view_offsets[RIGGED_VIEW_POSITIONS] + element * sizeof(float),
            asset->positions[element]);
        rigged_write_f32_le(
            binary + view_offsets[RIGGED_VIEW_NORMALS] + element * sizeof(float),
            asset->normals[element]);
    }
    for (size_t element = 0; element < asset->vertex_count * 4u; ++element) {
        rigged_write_u16_le(
            binary + view_offsets[RIGGED_VIEW_JOINTS] + element * sizeof(uint16_t),
            vertex_joints[element]);
        rigged_write_f32_le(
            binary + view_offsets[RIGGED_VIEW_WEIGHTS] + element * sizeof(float),
            vertex_weights[element]);
    }
    for (size_t primitive = 0; primitive < asset->primitive_count; ++primitive) {
        const trellis_mesh_rigging_primitive_range * range = &asset->primitives[primitive];
        for (size_t triangle = 0; triangle < range->triangle_count; ++triangle) {
            for (size_t corner = 0; corner < 3u; ++corner) {
                const size_t source_triangle = range->first_triangle + triangle;
                const uint32_t global_index = asset->triangles[source_triangle * 3u + corner];
                const uint32_t local_index = (uint32_t) ((size_t) global_index - range->first_vertex);
                const size_t element = source_triangle * 3u + corner;
                rigged_write_u32_le(
                    binary + view_offsets[RIGGED_VIEW_INDICES] + element * sizeof(uint32_t),
                    local_index);
            }
        }
    }

    size_t world_joint_elements = 0;
    size_t world_joint_bytes = 0;
    if (!rigged_multiply_size(skeleton->joint_count, 3u, &world_joint_elements) ||
        !rigged_multiply_size(world_joint_elements, sizeof(float), &world_joint_bytes)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        rigged_set_error(error_out, error_size, "joint world-position allocation size overflow");
        goto cleanup;
    }
    world_joints = (float *) malloc(world_joint_bytes);
    if (world_joints == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        rigged_set_error(error_out, error_size, "out of memory allocating world-space joints");
        goto cleanup;
    }
    for (size_t joint = 0; joint < skeleton->joint_count; ++joint) {
        trellis_mesh_rigging_denormalize_point(
            normalization,
            skeleton->joints_xyz + joint * 3u,
            world_joints + joint * 3u);
        if (!rigged_finite3(world_joints + joint * 3u)) {
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
            rigged_set_error(error_out, error_size, "joint %zu overflows during world-space conversion", joint);
            goto cleanup;
        }
        float inverse_bind[16] = { 0 };
        inverse_bind[0] = 1.0f;
        inverse_bind[5] = 1.0f;
        inverse_bind[10] = 1.0f;
        inverse_bind[15] = 1.0f;
        inverse_bind[12] = -world_joints[joint * 3u + 0u];
        inverse_bind[13] = -world_joints[joint * 3u + 1u];
        inverse_bind[14] = -world_joints[joint * 3u + 2u];
        for (size_t element = 0; element < 16u; ++element) {
            rigged_write_f32_le(
                binary + view_offsets[RIGGED_VIEW_INVERSE_BINDS] +
                    (joint * 16u + element) * sizeof(float),
                inverse_bind[element]);
        }
    }

    size_t accessor_count = 0;
    size_t attribute_count = 0;
    if (!rigged_multiply_size(
            asset->primitive_count,
            RIGGED_ACCESSORS_PER_PRIMITIVE,
            &accessor_count) ||
        !rigged_add_size(accessor_count, 1u, &accessor_count) ||
        !rigged_multiply_size(
            asset->primitive_count,
            RIGGED_ATTRIBUTES_PER_PRIMITIVE,
            &attribute_count)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        rigged_set_error(error_out, error_size, "GLB metadata allocation size overflow");
        goto cleanup;
    }
    accessors = (cgltf_accessor *) calloc(accessor_count, sizeof(*accessors));
    primitives = (cgltf_primitive *) calloc(asset->primitive_count, sizeof(*primitives));
    attributes = (cgltf_attribute *) calloc(attribute_count, sizeof(*attributes));
    nodes = (cgltf_node *) calloc(skeleton->joint_count + 1u, sizeof(*nodes));
    skin_joints = (cgltf_node **) calloc(skeleton->joint_count, sizeof(*skin_joints));
    child_counts = (size_t *) calloc(skeleton->joint_count, sizeof(*child_counts));
    child_offsets = (size_t *) calloc(skeleton->joint_count + 1u, sizeof(*child_offsets));
    child_fill = (size_t *) calloc(skeleton->joint_count, sizeof(*child_fill));
    scene_nodes = (cgltf_node **) calloc(root_count + 1u, sizeof(*scene_nodes));
    joint_names = (char *) calloc(skeleton->joint_count, 32u);
    if (accessors == NULL || primitives == NULL || attributes == NULL || nodes == NULL ||
        skin_joints == NULL || child_counts == NULL || child_offsets == NULL ||
        child_fill == NULL || scene_nodes == NULL || joint_names == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        rigged_set_error(error_out, error_size, "out of memory allocating GLB metadata");
        goto cleanup;
    }
    if (skeleton->joint_count > root_count) {
        child_links = (cgltf_node **) calloc(
            skeleton->joint_count - root_count, sizeof(*child_links));
        if (child_links == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            rigged_set_error(error_out, error_size, "out of memory allocating skeleton child links");
            goto cleanup;
        }
    }

    cgltf_buffer buffer;
    cgltf_buffer_view views[RIGGED_VIEW_COUNT];
    cgltf_material material;
    cgltf_mesh mesh;
    cgltf_skin skin;
    cgltf_scene scene;
    cgltf_data data;
    memset(&buffer, 0, sizeof(buffer));
    memset(views, 0, sizeof(views));
    memset(&material, 0, sizeof(material));
    memset(&mesh, 0, sizeof(mesh));
    memset(&skin, 0, sizeof(skin));
    memset(&scene, 0, sizeof(scene));
    memset(&data, 0, sizeof(data));

    buffer.name = (char *) "TokenSkin binary";
    buffer.size = binary_size;
    buffer.data = binary;
    const size_t view_sizes[RIGGED_VIEW_COUNT] = {
        position_bytes,
        normal_bytes,
        joint_bytes,
        weight_bytes,
        index_bytes,
        inverse_bind_bytes,
    };
    for (size_t view = 0; view < RIGGED_VIEW_COUNT; ++view) {
        views[view].buffer = &buffer;
        views[view].offset = view_offsets[view];
        views[view].size = view_sizes[view];
    }
    views[RIGGED_VIEW_POSITIONS].name = (char *) "positions";
    views[RIGGED_VIEW_NORMALS].name = (char *) "normals";
    views[RIGGED_VIEW_JOINTS].name = (char *) "joints";
    views[RIGGED_VIEW_WEIGHTS].name = (char *) "weights";
    views[RIGGED_VIEW_INDICES].name = (char *) "indices";
    views[RIGGED_VIEW_INVERSE_BINDS].name = (char *) "inverse bind matrices";
    views[RIGGED_VIEW_POSITIONS].type = cgltf_buffer_view_type_vertices;
    views[RIGGED_VIEW_NORMALS].type = cgltf_buffer_view_type_vertices;
    views[RIGGED_VIEW_JOINTS].type = cgltf_buffer_view_type_vertices;
    views[RIGGED_VIEW_WEIGHTS].type = cgltf_buffer_view_type_vertices;
    views[RIGGED_VIEW_INDICES].type = cgltf_buffer_view_type_indices;

    material.name = (char *) "TokenSkin default PBR";
    material.has_pbr_metallic_roughness = 1;
    material.pbr_metallic_roughness.base_color_factor[0] = 1.0f;
    material.pbr_metallic_roughness.base_color_factor[1] = 1.0f;
    material.pbr_metallic_roughness.base_color_factor[2] = 1.0f;
    material.pbr_metallic_roughness.base_color_factor[3] = 1.0f;
    material.pbr_metallic_roughness.metallic_factor = 0.0f;
    material.pbr_metallic_roughness.roughness_factor = 1.0f;
    material.alpha_mode = cgltf_alpha_mode_opaque;

    for (size_t primitive = 0; primitive < asset->primitive_count; ++primitive) {
        const trellis_mesh_rigging_primitive_range * range = &asset->primitives[primitive];
        cgltf_accessor * position = &accessors[primitive * RIGGED_ACCESSORS_PER_PRIMITIVE + 0u];
        cgltf_accessor * normal = &accessors[primitive * RIGGED_ACCESSORS_PER_PRIMITIVE + 1u];
        cgltf_accessor * joints = &accessors[primitive * RIGGED_ACCESSORS_PER_PRIMITIVE + 2u];
        cgltf_accessor * weights = &accessors[primitive * RIGGED_ACCESSORS_PER_PRIMITIVE + 3u];
        cgltf_accessor * indices = &accessors[primitive * RIGGED_ACCESSORS_PER_PRIMITIVE + 4u];

        position->buffer_view = &views[RIGGED_VIEW_POSITIONS];
        position->offset = range->first_vertex * 3u * sizeof(float);
        position->component_type = cgltf_component_type_r_32f;
        position->type = cgltf_type_vec3;
        position->count = range->vertex_count;
        position->has_min = 1;
        position->has_max = 1;
        for (size_t axis = 0; axis < 3u; ++axis) {
            position->min[axis] = FLT_MAX;
            position->max[axis] = -FLT_MAX;
        }
        for (size_t vertex = range->first_vertex;
             vertex < range->first_vertex + range->vertex_count;
             ++vertex) {
            for (size_t axis = 0; axis < 3u; ++axis) {
                const float value = asset->positions[vertex * 3u + axis];
                if (value < position->min[axis]) position->min[axis] = value;
                if (value > position->max[axis]) position->max[axis] = value;
            }
        }

        normal->buffer_view = &views[RIGGED_VIEW_NORMALS];
        normal->offset = range->first_vertex * 3u * sizeof(float);
        normal->component_type = cgltf_component_type_r_32f;
        normal->type = cgltf_type_vec3;
        normal->count = range->vertex_count;

        joints->buffer_view = &views[RIGGED_VIEW_JOINTS];
        joints->offset = range->first_vertex * 4u * sizeof(uint16_t);
        joints->component_type = cgltf_component_type_r_16u;
        joints->type = cgltf_type_vec4;
        joints->count = range->vertex_count;

        weights->buffer_view = &views[RIGGED_VIEW_WEIGHTS];
        weights->offset = range->first_vertex * 4u * sizeof(float);
        weights->component_type = cgltf_component_type_r_32f;
        weights->type = cgltf_type_vec4;
        weights->count = range->vertex_count;

        indices->buffer_view = &views[RIGGED_VIEW_INDICES];
        indices->offset = range->first_triangle * 3u * sizeof(uint32_t);
        indices->component_type = cgltf_component_type_r_32u;
        indices->type = cgltf_type_scalar;
        indices->count = range->triangle_count * 3u;
        indices->has_min = 1;
        indices->has_max = 1;
        uint32_t minimum_index = UINT32_MAX;
        uint32_t maximum_index = 0u;
        for (size_t triangle = 0; triangle < range->triangle_count; ++triangle) {
            const size_t source_triangle = range->first_triangle + triangle;
            for (size_t corner = 0; corner < 3u; ++corner) {
                const uint32_t global_index =
                    asset->triangles[source_triangle * 3u + corner];
                const uint32_t local_index =
                    (uint32_t) ((size_t) global_index - range->first_vertex);
                if (local_index < minimum_index) minimum_index = local_index;
                if (local_index > maximum_index) maximum_index = local_index;
            }
        }
        indices->min[0] = (float) minimum_index;
        indices->max[0] = (float) maximum_index;

        cgltf_attribute * primitive_attributes =
            attributes + primitive * RIGGED_ATTRIBUTES_PER_PRIMITIVE;
        primitive_attributes[0].name = (char *) "POSITION";
        primitive_attributes[0].type = cgltf_attribute_type_position;
        primitive_attributes[0].data = position;
        primitive_attributes[1].name = (char *) "NORMAL";
        primitive_attributes[1].type = cgltf_attribute_type_normal;
        primitive_attributes[1].data = normal;
        primitive_attributes[2].name = (char *) "JOINTS_0";
        primitive_attributes[2].type = cgltf_attribute_type_joints;
        primitive_attributes[2].index = 0;
        primitive_attributes[2].data = joints;
        primitive_attributes[3].name = (char *) "WEIGHTS_0";
        primitive_attributes[3].type = cgltf_attribute_type_weights;
        primitive_attributes[3].index = 0;
        primitive_attributes[3].data = weights;

        primitives[primitive].type = cgltf_primitive_type_triangles;
        primitives[primitive].indices = indices;
        primitives[primitive].material = &material;
        primitives[primitive].attributes = primitive_attributes;
        primitives[primitive].attributes_count = RIGGED_ATTRIBUTES_PER_PRIMITIVE;
    }

    cgltf_accessor * inverse_binds = &accessors[accessor_count - 1u];
    inverse_binds->name = (char *) "inverse bind matrices";
    inverse_binds->buffer_view = &views[RIGGED_VIEW_INVERSE_BINDS];
    inverse_binds->component_type = cgltf_component_type_r_32f;
    inverse_binds->type = cgltf_type_mat4;
    inverse_binds->count = skeleton->joint_count;

    mesh.name = (char *) "TokenSkin flattened mesh";
    mesh.primitives = primitives;
    mesh.primitives_count = asset->primitive_count;

    nodes[0].name = (char *) "TokenSkin mesh";
    nodes[0].mesh = &mesh;
    for (size_t joint = 0; joint < skeleton->joint_count; ++joint) {
        cgltf_node * node = &nodes[joint + 1u];
        snprintf(joint_names + joint * 32u, 32u, "joint_%zu", joint);
        node->name = joint_names + joint * 32u;
        node->has_translation = 1;
        const int32_t parent = skeleton->parents[joint];
        for (size_t axis = 0; axis < 3u; ++axis) {
            node->translation[axis] = parent < 0 ?
                world_joints[joint * 3u + axis] :
                world_joints[joint * 3u + axis] -
                    world_joints[(size_t) parent * 3u + axis];
        }
        skin_joints[joint] = node;
        if (parent >= 0) ++child_counts[(size_t) parent];
    }
    for (size_t joint = 0; joint < skeleton->joint_count; ++joint) {
        child_offsets[joint + 1u] = child_offsets[joint] + child_counts[joint];
        child_fill[joint] = child_offsets[joint];
        if (child_counts[joint] != 0) {
            nodes[joint + 1u].children = child_links + child_offsets[joint];
            nodes[joint + 1u].children_count = child_counts[joint];
        }
    }
    size_t scene_root = 1u;
    scene_nodes[0] = &nodes[0];
    for (size_t joint = 0; joint < skeleton->joint_count; ++joint) {
        const int32_t parent = skeleton->parents[joint];
        if (parent < 0) {
            scene_nodes[scene_root++] = &nodes[joint + 1u];
        } else {
            cgltf_node * child = &nodes[joint + 1u];
            cgltf_node * parent_node = &nodes[(size_t) parent + 1u];
            child_links[child_fill[(size_t) parent]++] = child;
            child->parent = parent_node;
        }
    }

    skin.name = (char *) "TokenSkin skin";
    skin.joints = skin_joints;
    skin.joints_count = skeleton->joint_count;
    skin.skeleton = root_count == 1u ? scene_nodes[1] : NULL;
    skin.inverse_bind_matrices = inverse_binds;
    nodes[0].skin = &skin;

    scene.name = (char *) "TokenSkin rigged scene";
    scene.nodes = scene_nodes;
    scene.nodes_count = root_count + 1u;

    data.file_type = cgltf_file_type_glb;
    data.asset.version = (char *) "2.0";
    data.asset.generator = (char *) "trellis2.c TokenSkin rigged GLB writer";
    data.meshes = &mesh;
    data.meshes_count = 1;
    data.materials = &material;
    data.materials_count = 1;
    data.accessors = accessors;
    data.accessors_count = accessor_count;
    data.buffer_views = views;
    data.buffer_views_count = RIGGED_VIEW_COUNT;
    data.buffers = &buffer;
    data.buffers_count = 1;
    data.skins = &skin;
    data.skins_count = 1;
    data.nodes = nodes;
    data.nodes_count = skeleton->joint_count + 1u;
    data.scenes = &scene;
    data.scenes_count = 1;
    data.scene = &scene;
    data.bin = binary;
    data.bin_size = binary_size;

    cgltf_options write_options;
    memset(&write_options, 0, sizeof(write_options));
    write_options.type = cgltf_file_type_glb;
    const cgltf_size json_with_terminator = cgltf_write(&write_options, NULL, 0, &data);
    if (json_with_terminator == 0) {
        status = TRELLIS_STATUS_ERROR;
        rigged_set_error(error_out, error_size, "cgltf could not size rigged GLB JSON");
        goto cleanup;
    }
    const size_t json_size = (size_t) json_with_terminator - 1u;
    const size_t json_padding = (4u - json_size % 4u) % 4u;
    const size_t binary_padding = (4u - binary_size % 4u) % 4u;
    size_t glb_size = 12u + 8u;
    if (!rigged_add_size(glb_size, json_size, &glb_size) ||
        !rigged_add_size(glb_size, json_padding, &glb_size) ||
        !rigged_add_size(glb_size, 8u, &glb_size) ||
        !rigged_add_size(glb_size, binary_size, &glb_size) ||
        !rigged_add_size(glb_size, binary_padding, &glb_size) ||
        glb_size > UINT32_MAX) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        rigged_set_error(error_out, error_size, "rigged GLB exceeds the 32-bit container size limit");
        goto cleanup;
    }
    const cgltf_result write_result = cgltf_write_file(&write_options, path, &data);
    if (write_result != cgltf_result_success) {
        status = write_result == cgltf_result_out_of_memory ?
            TRELLIS_STATUS_OUT_OF_MEMORY : TRELLIS_STATUS_IO_ERROR;
        rigged_set_error(
            error_out,
            error_size,
            "failed to write rigged GLB: %s",
            rigged_cgltf_result_name(write_result));
        goto cleanup;
    }
    status = rigged_verify_written_glb(path, error_out, error_size);
    if (status != TRELLIS_STATUS_OK) {
        remove(path);
    }

cleanup:
    free(joint_names);
    free(world_joints);
    free(child_fill);
    free(child_offsets);
    free(child_counts);
    free(scene_nodes);
    free(child_links);
    free(skin_joints);
    free(nodes);
    free(attributes);
    free(primitives);
    free(accessors);
    free(binary);
    free(vertex_weights);
    free(vertex_joints);
    return status;
}
