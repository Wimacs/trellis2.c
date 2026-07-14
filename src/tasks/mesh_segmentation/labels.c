#include "labels.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SEGMENTATION_PALETTE_BINS 4096u
#define SEGMENTATION_MAX_PALETTE 256u
#define SEGMENTATION_HASH_EMPTY UINT64_C(0)

typedef struct segmentation_color_bin {
    uint32_t bin;
    size_t count;
    float color[3];
} segmentation_color_bin;

typedef struct segmentation_voxel_hash {
    uint64_t * keys;
    int64_t * values;
    size_t capacity;
} segmentation_voxel_hash;

static float clamp_unit(float value) {
    if (!isfinite(value)) return 0.0f;
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

static uint64_t voxel_key(int x, int y, int z) {
    return ((((uint64_t) (uint32_t) x) << 42u) |
            (((uint64_t) (uint32_t) y) << 21u) |
            (uint64_t) (uint32_t) z) + UINT64_C(1);
}

static size_t hash_slot(uint64_t key, size_t mask) {
    key ^= key >> 30u;
    key *= UINT64_C(0xbf58476d1ce4e5b9);
    key ^= key >> 27u;
    key *= UINT64_C(0x94d049bb133111eb);
    key ^= key >> 31u;
    return (size_t) key & mask;
}

static void voxel_hash_free(segmentation_voxel_hash * hash) {
    if (hash == NULL) return;
    free(hash->keys);
    free(hash->values);
    memset(hash, 0, sizeof(*hash));
}

static trellis_status voxel_hash_build(
    const int32_t * coords,
    int64_t count,
    int resolution,
    segmentation_voxel_hash * hash) {
    if (coords == NULL || count <= 0 || resolution <= 0 || hash == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(hash, 0, sizeof(*hash));
    if ((uint64_t) count > SIZE_MAX / 2u) return TRELLIS_STATUS_OUT_OF_MEMORY;
    size_t capacity = 16u;
    while (capacity < (size_t) count * 2u) {
        if (capacity > SIZE_MAX / 2u) return TRELLIS_STATUS_OUT_OF_MEMORY;
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX / sizeof(uint64_t) ||
        capacity > SIZE_MAX / sizeof(int64_t)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    hash->keys = (uint64_t *) calloc(capacity, sizeof(uint64_t));
    hash->values = (int64_t *) malloc(capacity * sizeof(int64_t));
    if (hash->keys == NULL || hash->values == NULL) {
        voxel_hash_free(hash);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    hash->capacity = capacity;
    const size_t mask = capacity - 1u;
    for (int64_t voxel = 0; voxel < count; ++voxel) {
        const int32_t * coord = coords + (size_t) voxel * 4u;
        if (coord[0] != 0 || coord[1] < 0 || coord[1] >= resolution ||
            coord[2] < 0 || coord[2] >= resolution ||
            coord[3] < 0 || coord[3] >= resolution) {
            voxel_hash_free(hash);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        const uint64_t key = voxel_key(coord[1], coord[2], coord[3]);
        size_t slot = hash_slot(key, mask);
        while (hash->keys[slot] != SEGMENTATION_HASH_EMPTY &&
               hash->keys[slot] != key) {
            slot = (slot + 1u) & mask;
        }
        hash->keys[slot] = key;
        hash->values[slot] = voxel;
    }
    return TRELLIS_STATUS_OK;
}

static int64_t voxel_hash_find(
    const segmentation_voxel_hash * hash,
    int x,
    int y,
    int z) {
    const uint64_t key = voxel_key(x, y, z);
    const size_t mask = hash->capacity - 1u;
    size_t slot = hash_slot(key, mask);
    while (hash->keys[slot] != SEGMENTATION_HASH_EMPTY) {
        if (hash->keys[slot] == key) return hash->values[slot];
        slot = (slot + 1u) & mask;
    }
    return -1;
}

static int64_t nearest_voxel(
    const segmentation_voxel_hash * hash,
    const int32_t * coords,
    int64_t count,
    int resolution,
    const float position[3]) {
    int target[3];
    for (int axis = 0; axis < 3; ++axis) {
        const float coordinate =
            (clamp_unit(position[axis] + 0.5f)) * (float) resolution;
        int value = (int) floorf(coordinate);
        if (value < 0) value = 0;
        if (value >= resolution) value = resolution - 1;
        target[axis] = value;
    }
    int64_t best = -1;
    int64_t best_distance = INT64_MAX;
    for (int radius = 0; radius <= 8; ++radius) {
        const int min_x = target[0] - radius < 0 ? 0 : target[0] - radius;
        const int max_x = target[0] + radius >= resolution ? resolution - 1 : target[0] + radius;
        const int min_y = target[1] - radius < 0 ? 0 : target[1] - radius;
        const int max_y = target[1] + radius >= resolution ? resolution - 1 : target[1] + radius;
        const int min_z = target[2] - radius < 0 ? 0 : target[2] - radius;
        const int max_z = target[2] + radius >= resolution ? resolution - 1 : target[2] + radius;
        for (int x = min_x; x <= max_x; ++x) {
            for (int y = min_y; y <= max_y; ++y) {
                for (int z = min_z; z <= max_z; ++z) {
                    if (radius > 0 && x > min_x && x < max_x &&
                        y > min_y && y < max_y && z > min_z && z < max_z) {
                        continue;
                    }
                    const int64_t voxel = voxel_hash_find(hash, x, y, z);
                    if (voxel < 0) continue;
                    const int64_t dx = (int64_t) x - target[0];
                    const int64_t dy = (int64_t) y - target[1];
                    const int64_t dz = (int64_t) z - target[2];
                    const int64_t distance = dx * dx + dy * dy + dz * dz;
                    if (distance < best_distance) {
                        best_distance = distance;
                        best = voxel;
                    }
                }
            }
        }
        if (best >= 0 &&
            (int64_t) (radius + 1) * (radius + 1) > best_distance) {
            return best;
        }
    }
    /* A source vertex should lie next to a decoded surface voxel. Keep an
     * exact fallback for unusual decoder topology rather than dropping it. */
    for (int64_t voxel = 0; voxel < count; ++voxel) {
        const int32_t * coord = coords + (size_t) voxel * 4u + 1u;
        const int64_t dx = (int64_t) coord[0] - target[0];
        const int64_t dy = (int64_t) coord[1] - target[1];
        const int64_t dz = (int64_t) coord[2] - target[2];
        const int64_t distance = dx * dx + dy * dy + dz * dz;
        if (distance < best_distance) {
            best_distance = distance;
            best = voxel;
        }
    }
    return best;
}

static int compare_color_bins(const void * left_ptr, const void * right_ptr) {
    const segmentation_color_bin * left =
        (const segmentation_color_bin *) left_ptr;
    const segmentation_color_bin * right =
        (const segmentation_color_bin *) right_ptr;
    if (left->count != right->count) return left->count > right->count ? -1 : 1;
    if (left->bin != right->bin) return left->bin < right->bin ? -1 : 1;
    return 0;
}

static float color_distance_squared(const float left[3], const float right[3]) {
    const float r = left[0] - right[0];
    const float g = left[1] - right[1];
    const float b = left[2] - right[2];
    return r * r + g * g + b * b;
}

static uint32_t nearest_palette_color(
    const float color[3],
    const float palette[SEGMENTATION_MAX_PALETTE][3],
    size_t palette_count) {
    uint32_t best = 0;
    float best_distance = FLT_MAX;
    for (size_t index = 0; index < palette_count; ++index) {
        const float distance = color_distance_squared(color, palette[index]);
        if (distance < best_distance) {
            best_distance = distance;
            best = (uint32_t) index;
        }
    }
    return best;
}

trellis_status trellis_mesh_segmentation_labels_from_voxels(
    const float * normalized_positions,
    size_t vertex_count,
    const uint32_t * triangles,
    size_t face_count,
    const int32_t * voxel_coords_bxyz,
    const float * voxel_attrs,
    int64_t n_voxels,
    int voxel_channels,
    int resolution,
    size_t min_palette_voxels,
    float palette_merge_distance,
    uint32_t ** semantic_labels_out,
    size_t * semantic_count_out) {
    if (semantic_labels_out != NULL) *semantic_labels_out = NULL;
    if (semantic_count_out != NULL) *semantic_count_out = 0;
    if (normalized_positions == NULL || vertex_count == 0 || triangles == NULL ||
        face_count == 0 || voxel_coords_bxyz == NULL || voxel_attrs == NULL ||
        n_voxels <= 0 || voxel_channels < 3 || resolution <= 0 ||
        resolution > 0x1fffff ||
        semantic_labels_out == NULL || semantic_count_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if ((uint64_t) n_voxels > SIZE_MAX / 4u ||
        (size_t) n_voxels > SIZE_MAX / (size_t) voxel_channels ||
        (size_t) n_voxels * (size_t) voxel_channels > SIZE_MAX / sizeof(float) ||
        vertex_count > SIZE_MAX / (3u * sizeof(float)) ||
        vertex_count > SIZE_MAX / sizeof(uint32_t) ||
        face_count > SIZE_MAX / (3u * sizeof(uint32_t)) ||
        face_count > SIZE_MAX / sizeof(uint32_t)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    if (min_palette_voxels == 0) min_palette_voxels = 4;
    if (!(palette_merge_distance > 0.0f) || !isfinite(palette_merge_distance)) {
        palette_merge_distance = 32.0f / 255.0f;
    }
    for (int64_t voxel = 0; voxel < n_voxels; ++voxel) {
        const float * color =
            voxel_attrs + (size_t) voxel * (size_t) voxel_channels;
        if (!isfinite(color[0]) || !isfinite(color[1]) ||
            !isfinite(color[2])) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }

    segmentation_voxel_hash hash;
    trellis_status status = voxel_hash_build(
        voxel_coords_bxyz, n_voxels, resolution, &hash);
    if (status != TRELLIS_STATUS_OK) return status;
    float * vertex_colors = (float *) malloc(vertex_count * 3u * sizeof(float));
    uint32_t * vertex_labels = (uint32_t *) malloc(vertex_count * sizeof(uint32_t));
    uint32_t * face_labels = (uint32_t *) malloc(face_count * sizeof(uint32_t));
    if (vertex_colors == NULL || vertex_labels == NULL || face_labels == NULL) {
        voxel_hash_free(&hash);
        free(vertex_colors); free(vertex_labels); free(face_labels);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
        const float * position = normalized_positions + vertex * 3u;
        if (!isfinite(position[0]) || !isfinite(position[1]) || !isfinite(position[2])) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        const int64_t voxel = nearest_voxel(
            &hash, voxel_coords_bxyz, n_voxels, resolution, position);
        if (voxel < 0) {
            status = TRELLIS_STATUS_ERROR;
            goto cleanup;
        }
        for (int channel = 0; channel < 3; ++channel) {
            vertex_colors[vertex * 3u + (size_t) channel] = clamp_unit(
                voxel_attrs[(size_t) voxel * (size_t) voxel_channels + (size_t) channel]);
        }
    }

    size_t counts[SEGMENTATION_PALETTE_BINS] = {0};
    double sums[SEGMENTATION_PALETTE_BINS][3];
    memset(sums, 0, sizeof(sums));
    for (int64_t voxel = 0; voxel < n_voxels; ++voxel) {
        float color[3];
        uint32_t quantized[3];
        for (int channel = 0; channel < 3; ++channel) {
            color[channel] = clamp_unit(
                voxel_attrs[(size_t) voxel * (size_t) voxel_channels + (size_t) channel]);
            quantized[channel] = (uint32_t) floorf(color[channel] * 15.0f + 0.5f);
            if (quantized[channel] > 15u) quantized[channel] = 15u;
        }
        const uint32_t bin = (quantized[0] << 8u) | (quantized[1] << 4u) | quantized[2];
        ++counts[bin];
        sums[bin][0] += color[0];
        sums[bin][1] += color[1];
        sums[bin][2] += color[2];
    }
    segmentation_color_bin candidates[SEGMENTATION_PALETTE_BINS];
    size_t candidate_count = 0;
    for (uint32_t bin = 0; bin < SEGMENTATION_PALETTE_BINS; ++bin) {
        if (counts[bin] < min_palette_voxels) continue;
        segmentation_color_bin * candidate = &candidates[candidate_count++];
        candidate->bin = bin;
        candidate->count = counts[bin];
        candidate->color[0] = (float) (sums[bin][0] / (double) counts[bin]);
        candidate->color[1] = (float) (sums[bin][1] / (double) counts[bin]);
        candidate->color[2] = (float) (sums[bin][2] / (double) counts[bin]);
    }
    if (candidate_count == 0) {
        for (uint32_t bin = 0; bin < SEGMENTATION_PALETTE_BINS; ++bin) {
            if (counts[bin] == 0) continue;
            segmentation_color_bin * candidate = &candidates[candidate_count++];
            candidate->bin = bin;
            candidate->count = counts[bin];
            candidate->color[0] = (float) (sums[bin][0] / (double) counts[bin]);
            candidate->color[1] = (float) (sums[bin][1] / (double) counts[bin]);
            candidate->color[2] = (float) (sums[bin][2] / (double) counts[bin]);
        }
    }
    qsort(candidates, candidate_count, sizeof(candidates[0]), compare_color_bins);
    float palette[SEGMENTATION_MAX_PALETTE][3];
    size_t palette_weights[SEGMENTATION_MAX_PALETTE] = {0};
    size_t palette_count = 0;
    const float merge_distance_squared =
        palette_merge_distance * palette_merge_distance;
    for (size_t candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
        const segmentation_color_bin * candidate = &candidates[candidate_index];
        size_t merge = SIZE_MAX;
        float best_distance = FLT_MAX;
        for (size_t palette_index = 0; palette_index < palette_count; ++palette_index) {
            const float distance = color_distance_squared(
                candidate->color, palette[palette_index]);
            if (distance <= merge_distance_squared && distance < best_distance) {
                merge = palette_index;
                best_distance = distance;
            }
        }
        if (merge == SIZE_MAX) {
            if (palette_count >= SEGMENTATION_MAX_PALETTE) continue;
            merge = palette_count++;
            memcpy(palette[merge], candidate->color, 3u * sizeof(float));
            palette_weights[merge] = candidate->count;
        } else {
            const size_t total = palette_weights[merge] + candidate->count;
            for (int channel = 0; channel < 3; ++channel) {
                palette[merge][channel] =
                    (palette[merge][channel] * (float) palette_weights[merge] +
                     candidate->color[channel] * (float) candidate->count) /
                    (float) total;
            }
            palette_weights[merge] = total;
        }
    }
    if (palette_count == 0) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
        vertex_labels[vertex] = nearest_palette_color(
            vertex_colors + vertex * 3u, palette, palette_count);
    }
    for (size_t face = 0; face < face_count; ++face) {
        const uint32_t a = triangles[face * 3u];
        const uint32_t b = triangles[face * 3u + 1u];
        const uint32_t c = triangles[face * 3u + 2u];
        if ((size_t) a >= vertex_count || (size_t) b >= vertex_count ||
            (size_t) c >= vertex_count) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        const uint32_t la = vertex_labels[a];
        const uint32_t lb = vertex_labels[b];
        const uint32_t lc = vertex_labels[c];
        if (la == lb || la == lc) {
            face_labels[face] = la;
        } else if (lb == lc) {
            face_labels[face] = lb;
        } else {
            float average[3];
            for (int channel = 0; channel < 3; ++channel) {
                average[channel] = (
                    vertex_colors[(size_t) a * 3u + (size_t) channel] +
                    vertex_colors[(size_t) b * 3u + (size_t) channel] +
                    vertex_colors[(size_t) c * 3u + (size_t) channel]) / 3.0f;
            }
            face_labels[face] = nearest_palette_color(average, palette, palette_count);
        }
    }
    *semantic_labels_out = face_labels;
    face_labels = NULL;
    *semantic_count_out = palette_count;
    status = TRELLIS_STATUS_OK;

cleanup:
    voxel_hash_free(&hash);
    free(vertex_colors);
    free(vertex_labels);
    free(face_labels);
    return status;
}
