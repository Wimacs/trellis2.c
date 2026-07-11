#include "interpolate.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct kd_node {
    uint32_t point;
    int32_t left;
    int32_t right;
    uint8_t axis;
} kd_node;

typedef struct kd_tree {
    const float * points;
    kd_node * nodes;
    uint32_t * permutation;
    size_t node_count;
} kd_tree;

typedef struct nearest_heap {
    float distance2[8];
    uint32_t point[8];
    int count;
    int capacity;
} nearest_heap;

static float coordinate(const kd_tree * tree, uint32_t point, int axis) {
    return tree->points[(size_t) point * 3u + (size_t) axis];
}

static void swap_indices(uint32_t * a, uint32_t * b) {
    const uint32_t value = *a;
    *a = *b;
    *b = value;
}

static size_t partition(
    kd_tree * tree,
    size_t begin,
    size_t end,
    size_t pivot,
    int axis) {
    const float pivot_value = coordinate(tree, tree->permutation[pivot], axis);
    swap_indices(&tree->permutation[pivot], &tree->permutation[end - 1u]);
    size_t store = begin;
    for (size_t i = begin; i + 1u < end; ++i) {
        const uint32_t point = tree->permutation[i];
        const float value = coordinate(tree, point, axis);
        if (value < pivot_value ||
            (value == pivot_value && point < tree->permutation[end - 1u])) {
            swap_indices(&tree->permutation[store++], &tree->permutation[i]);
        }
    }
    swap_indices(&tree->permutation[store], &tree->permutation[end - 1u]);
    return store;
}

static void select_nth(
    kd_tree * tree,
    size_t begin,
    size_t end,
    size_t nth,
    int axis) {
    while (end - begin > 1u) {
        size_t pivot = begin + (end - begin) / 2u;
        pivot = partition(tree, begin, end, pivot, axis);
        if (pivot == nth) return;
        if (nth < pivot) end = pivot;
        else begin = pivot + 1u;
    }
}

static int choose_axis(const kd_tree * tree, size_t begin, size_t end) {
    float minimum[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float maximum[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (size_t i = begin; i < end; ++i) {
        const float * point = tree->points + (size_t) tree->permutation[i] * 3u;
        for (int axis = 0; axis < 3; ++axis) {
            if (point[axis] < minimum[axis]) minimum[axis] = point[axis];
            if (point[axis] > maximum[axis]) maximum[axis] = point[axis];
        }
    }
    int axis = 0;
    if (maximum[1] - minimum[1] > maximum[axis] - minimum[axis]) axis = 1;
    if (maximum[2] - minimum[2] > maximum[axis] - minimum[axis]) axis = 2;
    return axis;
}

static int32_t build_subtree(kd_tree * tree, size_t begin, size_t end) {
    if (begin >= end) return -1;
    const int axis = choose_axis(tree, begin, end);
    const size_t middle = begin + (end - begin) / 2u;
    select_nth(tree, begin, end, middle, axis);
    const int32_t node_index = (int32_t) tree->node_count++;
    kd_node * node = &tree->nodes[node_index];
    node->point = tree->permutation[middle];
    node->axis = (uint8_t) axis;
    node->left = build_subtree(tree, begin, middle);
    node->right = build_subtree(tree, middle + 1u, end);
    return node_index;
}

static void heap_swap(nearest_heap * heap, int a, int b) {
    const float distance = heap->distance2[a];
    const uint32_t point = heap->point[a];
    heap->distance2[a] = heap->distance2[b];
    heap->point[a] = heap->point[b];
    heap->distance2[b] = distance;
    heap->point[b] = point;
}

static int farther_than(float da, uint32_t pa, float db, uint32_t pb) {
    return da > db || (da == db && pa > pb);
}

static void heap_sift_up(nearest_heap * heap, int index) {
    while (index > 0) {
        const int parent = (index - 1) / 2;
        if (!farther_than(
                heap->distance2[index], heap->point[index],
                heap->distance2[parent], heap->point[parent])) break;
        heap_swap(heap, index, parent);
        index = parent;
    }
}

static void heap_sift_down(nearest_heap * heap, int index) {
    for (;;) {
        const int left = index * 2 + 1;
        const int right = left + 1;
        int farther = index;
        if (left < heap->count && farther_than(
                heap->distance2[left], heap->point[left],
                heap->distance2[farther], heap->point[farther])) farther = left;
        if (right < heap->count && farther_than(
                heap->distance2[right], heap->point[right],
                heap->distance2[farther], heap->point[farther])) farther = right;
        if (farther == index) break;
        heap_swap(heap, index, farther);
        index = farther;
    }
}

static void heap_offer(nearest_heap * heap, float distance2, uint32_t point) {
    if (heap->count < heap->capacity) {
        const int index = heap->count++;
        heap->distance2[index] = distance2;
        heap->point[index] = point;
        heap_sift_up(heap, index);
    } else if (distance2 < heap->distance2[0] ||
               (distance2 == heap->distance2[0] && point < heap->point[0])) {
        heap->distance2[0] = distance2;
        heap->point[0] = point;
        heap_sift_down(heap, 0);
    }
}

static void query_subtree(
    const kd_tree * tree,
    int32_t node_index,
    const float query[3],
    nearest_heap * heap) {
    if (node_index < 0) return;
    const kd_node * node = &tree->nodes[node_index];
    const float * point = tree->points + (size_t) node->point * 3u;
    const float dx = query[0] - point[0];
    const float dy = query[1] - point[1];
    const float dz = query[2] - point[2];
    const float distance2 = dx * dx + dy * dy + dz * dz;
    heap_offer(heap, distance2, node->point);

    const float delta = query[node->axis] - point[node->axis];
    const int32_t near_node = delta <= 0.0f ? node->left : node->right;
    const int32_t far_node = delta <= 0.0f ? node->right : node->left;
    query_subtree(tree, near_node, query, heap);
    if (heap->count < heap->capacity || delta * delta <= heap->distance2[0]) {
        query_subtree(tree, far_node, query, heap);
    }
}

trellis_status trellis_tokenskin_interpolate_skin_8nn(
    const float * sampled_positions,
    const float * sampled_skin,
    size_t sample_count,
    size_t joint_count,
    const float * query_positions,
    size_t query_count,
    float ** query_skin_out) {
    if (sampled_positions == NULL || sampled_skin == NULL || sample_count == 0 ||
        sample_count > INT32_MAX ||
        sample_count > SIZE_MAX / sizeof(kd_node) ||
        sample_count > SIZE_MAX / sizeof(uint32_t) ||
        joint_count == 0 || query_positions == NULL ||
        query_count == 0 || query_skin_out == NULL ||
        query_count > SIZE_MAX / joint_count ||
        query_count * joint_count > SIZE_MAX / sizeof(float)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    kd_tree tree = {sampled_positions, NULL, NULL, 0};
    tree.nodes = (kd_node *) malloc(sample_count * sizeof(kd_node));
    tree.permutation = (uint32_t *) malloc(sample_count * sizeof(uint32_t));
    float * output = (float *) malloc(query_count * joint_count * sizeof(float));
    if (tree.nodes == NULL || tree.permutation == NULL || output == NULL) {
        free(output);
        free(tree.permutation);
        free(tree.nodes);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < sample_count; ++i) tree.permutation[i] = (uint32_t) i;
    const int32_t root = build_subtree(&tree, 0, sample_count);
    const int neighbors = sample_count < 8u ? (int) sample_count : 8;
    for (size_t query_index = 0; query_index < query_count; ++query_index) {
        nearest_heap heap = {{0}, {0}, 0, neighbors};
        query_subtree(
            &tree,
            root,
            query_positions + query_index * 3u,
            &heap);
        float weight_sum = 0.0f;
        float weights[8];
        for (int k = 0; k < heap.count; ++k) {
            weights[k] = 1.0f / (sqrtf(heap.distance2[k]) + 1.0e-8f);
            weight_sum += weights[k];
        }
        float * destination = output + query_index * joint_count;
        for (size_t joint = 0; joint < joint_count; ++joint) {
            float value = 0.0f;
            for (int k = 0; k < heap.count; ++k) {
                value += weights[k] * sampled_skin[(size_t) heap.point[k] * joint_count + joint];
            }
            destination[joint] = value / weight_sum;
        }
    }
    free(tree.permutation);
    free(tree.nodes);
    *query_skin_out = output;
    return TRELLIS_STATUS_OK;
}
