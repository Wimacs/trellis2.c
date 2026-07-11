#include "preprocess.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct trellis_mesh_rigging_rng {
    uint64_t state;
    uint64_t increment;
} trellis_mesh_rigging_rng;

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

static int multiply_size(size_t a, size_t b, size_t * result) {
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *result = a * b;
    return 1;
}

/* PCG-XSH-RR gives an explicitly specified stream on every supported C11
 * platform.  This avoids libc rand(), whose sequence and bit width vary by
 * platform, and makes preprocessing artifacts reproducible from one seed. */
static uint32_t rng_next_u32(trellis_mesh_rigging_rng * rng) {
    const uint64_t old_state = rng->state;
    rng->state = old_state * UINT64_C(6364136223846793005) + rng->increment;
    const uint32_t xor_shifted = (uint32_t) (((old_state >> 18u) ^ old_state) >> 27u);
    const uint32_t rotation = (uint32_t) (old_state >> 59u);
    return (xor_shifted >> rotation) | (xor_shifted << ((0u - rotation) & 31u));
}

static void rng_seed(trellis_mesh_rigging_rng * rng, uint64_t seed) {
    rng->state = 0u;
    rng->increment = (UINT64_C(0xda3e39cb94b95bdb) << 1u) | 1u;
    (void) rng_next_u32(rng);
    rng->state += seed;
    (void) rng_next_u32(rng);
}

static double rng_unit_f64(trellis_mesh_rigging_rng * rng) {
    const uint64_t high = (uint64_t) (rng_next_u32(rng) >> 5u);
    const uint64_t low = (uint64_t) (rng_next_u32(rng) >> 6u);
    return (double) (high * UINT64_C(67108864) + low) / 9007199254740992.0;
}

static uint32_t rng_bounded_u32(trellis_mesh_rigging_rng * rng, uint32_t bound) {
    if (bound == 0u) {
        return 0u;
    }
    const uint32_t threshold = (uint32_t) (0u - bound) % bound;
    for (;;) {
        const uint32_t value = rng_next_u32(rng);
        if (value >= threshold) {
            return value % bound;
        }
    }
}

static void matrix_identity(float matrix[16]) {
    memset(matrix, 0, 16u * sizeof(float));
    matrix[0] = 1.0f;
    matrix[5] = 1.0f;
    matrix[10] = 1.0f;
    matrix[15] = 1.0f;
}

static trellis_status build_normalization(
    const trellis_mesh_rigging_asset * asset,
    trellis_mesh_rigging_normalization * normalization,
    char * error,
    size_t error_size) {
    float maximum_extent = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        const float minimum = asset->aabb_min[axis];
        const float maximum = asset->aabb_max[axis];
        if (!isfinite(minimum) || !isfinite(maximum) || maximum < minimum) {
            set_error(error, error_size, "asset has an invalid AABB");
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        normalization->source_min[axis] = minimum;
        normalization->source_max[axis] = maximum;
        normalization->center[axis] = minimum + (maximum - minimum) * 0.5f;
        const float extent = maximum - minimum;
        if (extent > maximum_extent) {
            maximum_extent = extent;
        }
    }
    if (!(maximum_extent > 1e-12f) || !isfinite(maximum_extent)) {
        set_error(error, error_size, "asset AABB is degenerate");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    normalization->scale = 2.0f / maximum_extent;
    normalization->inverse_scale = maximum_extent * 0.5f;
    matrix_identity(normalization->normalized_from_world);
    matrix_identity(normalization->world_from_normalized);
    normalization->normalized_from_world[0] = normalization->scale;
    normalization->normalized_from_world[5] = normalization->scale;
    normalization->normalized_from_world[10] = normalization->scale;
    normalization->normalized_from_world[12] = -normalization->center[0] * normalization->scale;
    normalization->normalized_from_world[13] = -normalization->center[1] * normalization->scale;
    normalization->normalized_from_world[14] = -normalization->center[2] * normalization->scale;
    normalization->world_from_normalized[0] = normalization->inverse_scale;
    normalization->world_from_normalized[5] = normalization->inverse_scale;
    normalization->world_from_normalized[10] = normalization->inverse_scale;
    normalization->world_from_normalized[12] = normalization->center[0];
    normalization->world_from_normalized[13] = normalization->center[1];
    normalization->world_from_normalized[14] = normalization->center[2];
    return TRELLIS_STATUS_OK;
}

void trellis_mesh_rigging_normalize_point(
    const trellis_mesh_rigging_normalization * normalization,
    const float world[3],
    float normalized[3]) {
    if (normalization == NULL || world == NULL || normalized == NULL) {
        return;
    }
    const float x = world[0];
    const float y = world[1];
    const float z = world[2];
    normalized[0] = (x - normalization->center[0]) * normalization->scale;
    normalized[1] = (y - normalization->center[1]) * normalization->scale;
    normalized[2] = (z - normalization->center[2]) * normalization->scale;
}

void trellis_mesh_rigging_denormalize_point(
    const trellis_mesh_rigging_normalization * normalization,
    const float normalized[3],
    float world[3]) {
    if (normalization == NULL || normalized == NULL || world == NULL) {
        return;
    }
    const float x = normalized[0];
    const float y = normalized[1];
    const float z = normalized[2];
    world[0] = x * normalization->inverse_scale + normalization->center[0];
    world[1] = y * normalization->inverse_scale + normalization->center[1];
    world[2] = z * normalization->inverse_scale + normalization->center[2];
}

static size_t find_area_interval(const double * cumulative, size_t count, double value) {
    size_t first = 0;
    size_t last = count;
    while (first < last) {
        const size_t middle = first + (last - first) / 2u;
        if (value < cumulative[middle]) {
            last = middle;
        } else {
            first = middle + 1u;
        }
    }
    return first < count ? first : count - 1u;
}

static trellis_status sample_surface(
    const trellis_mesh_rigging_asset * asset,
    const float * normalized_vertices,
    size_t sample_count,
    trellis_mesh_rigging_rng * rng,
    float * positions,
    float * normals,
    uint32_t * sampled_triangles,
    float * barycentric,
    char * error,
    size_t error_size) {
    double * cumulative = (double *) malloc(asset->triangle_count * sizeof(double));
    if (cumulative == NULL) {
        set_error(error, error_size, "out of memory allocating face-area distribution");
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    double total_area = 0.0;
    for (size_t triangle = 0; triangle < asset->triangle_count; ++triangle) {
        const uint32_t * indices = asset->triangles + triangle * 3u;
        const float * a = normalized_vertices + (size_t) indices[0] * 3u;
        const float * b = normalized_vertices + (size_t) indices[1] * 3u;
        const float * c = normalized_vertices + (size_t) indices[2] * 3u;
        const double abx = (double) b[0] - a[0];
        const double aby = (double) b[1] - a[1];
        const double abz = (double) b[2] - a[2];
        const double acx = (double) c[0] - a[0];
        const double acy = (double) c[1] - a[1];
        const double acz = (double) c[2] - a[2];
        const double cross_x = aby * acz - abz * acy;
        const double cross_y = abz * acx - abx * acz;
        const double cross_z = abx * acy - aby * acx;
        const double double_area = sqrt(
            cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
        if (isfinite(double_area) && double_area > 0.0) {
            total_area += double_area;
        }
        cumulative[triangle] = total_area;
    }
    if (!(total_area > 0.0) || !isfinite(total_area)) {
        free(cumulative);
        set_error(error, error_size, "mesh has no finite, non-degenerate triangle area");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    for (size_t sample = 0; sample < sample_count; ++sample) {
        const double area_pick = rng_unit_f64(rng) * total_area;
        const size_t triangle = find_area_interval(cumulative, asset->triangle_count, area_pick);
        const uint32_t * indices = asset->triangles + triangle * 3u;
        const double root = sqrt(rng_unit_f64(rng));
        const double split = rng_unit_f64(rng);
        const float weights[3] = {
            (float) (1.0 - root),
            (float) (root * (1.0 - split)),
            (float) (root * split),
        };
        float * point = positions + sample * 3u;
        for (int axis = 0; axis < 3; ++axis) {
            point[axis] =
                weights[0] * normalized_vertices[(size_t) indices[0] * 3u + (size_t) axis] +
                weights[1] * normalized_vertices[(size_t) indices[1] * 3u + (size_t) axis] +
                weights[2] * normalized_vertices[(size_t) indices[2] * 3u + (size_t) axis];
        }
        memcpy(normals + sample * 3u, asset->face_normals + triangle * 3u, 3u * sizeof(float));
        memcpy(barycentric + sample * 3u, weights, sizeof(weights));
        sampled_triangles[sample] = (uint32_t) triangle;
    }
    free(cumulative);
    return TRELLIS_STATUS_OK;
}

static trellis_status choose_candidates(
    size_t population,
    size_t candidate_count,
    trellis_mesh_rigging_rng * rng,
    uint32_t * candidates,
    char * error,
    size_t error_size) {
    if (population == 0 || population > UINT32_MAX || candidate_count == 0) {
        set_error(error, error_size, "invalid FPS candidate population/count");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (candidate_count > population) {
        for (size_t i = 0; i < candidate_count; ++i) {
            candidates[i] = rng_bounded_u32(rng, (uint32_t) population);
        }
        return TRELLIS_STATUS_OK;
    }

    uint32_t * pool = (uint32_t *) malloc(population * sizeof(uint32_t));
    if (pool == NULL) {
        set_error(error, error_size, "out of memory allocating FPS candidate permutation");
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < population; ++i) {
        pool[i] = (uint32_t) i;
    }
    for (size_t i = 0; i < candidate_count; ++i) {
        const uint32_t remaining = (uint32_t) (population - i);
        const size_t selected = i + (size_t) rng_bounded_u32(rng, remaining);
        const uint32_t temporary = pool[i];
        pool[i] = pool[selected];
        pool[selected] = temporary;
        candidates[i] = pool[i];
    }
    free(pool);
    return TRELLIS_STATUS_OK;
}

static trellis_status farthest_point_sample(
    const float * points,
    const uint32_t * candidates,
    size_t candidate_count,
    size_t output_count,
    uint32_t * output,
    char * error,
    size_t error_size) {
    if (points == NULL || candidates == NULL || output == NULL ||
        candidate_count == 0 || output_count == 0 || output_count > candidate_count) {
        set_error(error, error_size, "invalid farthest-point sampling inputs");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    float * minimum_distance = (float *) malloc(candidate_count * sizeof(float));
    if (minimum_distance == NULL) {
        set_error(error, error_size, "out of memory allocating FPS distances");
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < candidate_count; ++i) {
        minimum_distance[i] = INFINITY;
    }

    size_t farthest = 0;
    for (size_t selected = 0; selected < output_count; ++selected) {
        const uint32_t sample_index = candidates[farthest];
        output[selected] = sample_index;
        const float * centroid = points + (size_t) sample_index * 3u;
        float maximum_distance = -1.0f;
        size_t next_farthest = 0;
        for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
            const float * point = points + (size_t) candidates[candidate] * 3u;
            const float dx = point[0] - centroid[0];
            const float dy = point[1] - centroid[1];
            const float dz = point[2] - centroid[2];
            const float distance = dx * dx + dy * dy + dz * dz;
            if (distance < minimum_distance[candidate]) {
                minimum_distance[candidate] = distance;
            }
            if (minimum_distance[candidate] > maximum_distance) {
                maximum_distance = minimum_distance[candidate];
                next_farthest = candidate;
            }
        }
        farthest = next_farthest;
    }
    free(minimum_distance);
    return TRELLIS_STATUS_OK;
}

static trellis_status build_fps_set(
    const float * points,
    size_t population,
    size_t token_count,
    size_t candidate_multiplier,
    trellis_mesh_rigging_rng * rng,
    uint32_t ** candidates_out,
    size_t * candidate_count_out,
    uint32_t ** indices_out,
    char * error,
    size_t error_size) {
    size_t candidate_count = 0;
    if (!multiply_size(token_count, candidate_multiplier, &candidate_count) ||
        candidate_count == 0) {
        set_error(error, error_size, "FPS candidate count overflows size_t");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    uint32_t * candidates = (uint32_t *) malloc(candidate_count * sizeof(uint32_t));
    uint32_t * indices = (uint32_t *) malloc(token_count * sizeof(uint32_t));
    if (candidates == NULL || indices == NULL) {
        free(candidates);
        free(indices);
        set_error(error, error_size, "out of memory allocating FPS indices");
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    trellis_status status = choose_candidates(
        population,
        candidate_count,
        rng,
        candidates,
        error,
        error_size);
    if (status == TRELLIS_STATUS_OK) {
        status = farthest_point_sample(
            points,
            candidates,
            candidate_count,
            token_count,
            indices,
            error,
            error_size);
    }
    if (status != TRELLIS_STATUS_OK) {
        free(candidates);
        free(indices);
        return status;
    }
    *candidates_out = candidates;
    *candidate_count_out = candidate_count;
    *indices_out = indices;
    return TRELLIS_STATUS_OK;
}

void trellis_mesh_rigging_preprocessed_free(
    trellis_mesh_rigging_preprocessed * output) {
    if (output == NULL) {
        return;
    }
    free(output->normalized_vertices);
    free(output->sample_positions);
    free(output->sample_normals);
    free(output->sample_triangles);
    free(output->sample_barycentric);
    free(output->mesh_candidate_indices);
    free(output->mesh_fps_indices);
    free(output->skin_candidate_indices);
    free(output->skin_fps_indices);
    *output = (trellis_mesh_rigging_preprocessed) TRELLIS_MESH_RIGGING_PREPROCESSED_INIT;
}

trellis_status trellis_mesh_rigging_preprocess(
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_rigging_preprocess_options * options,
    trellis_mesh_rigging_preprocessed * output,
    char * error_out,
    size_t error_size) {
    if (error_out != NULL && error_size > 0) {
        error_out[0] = '\0';
    }
    if (asset == NULL || options == NULL || output == NULL ||
        options->struct_size < sizeof(*options) ||
        asset->positions == NULL || asset->normals == NULL || asset->face_normals == NULL ||
        asset->triangles == NULL || asset->vertex_count == 0 || asset->triangle_count == 0 ||
        asset->vertex_count > UINT32_MAX || asset->triangle_count > UINT32_MAX ||
        asset->triangle_count > SIZE_MAX / 3u ||
        options->surface_sample_count == 0 || options->surface_sample_count > UINT32_MAX ||
        options->mesh_token_count == 0 || options->skin_token_count == 0 ||
        options->fps_candidate_multiplier == 0) {
        set_error(error_out, error_size, "invalid mesh-rigging preprocessing inputs/options");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    size_t mesh_candidates = 0;
    size_t skin_candidates = 0;
    if (!multiply_size(
            options->mesh_token_count,
            options->fps_candidate_multiplier,
            &mesh_candidates) ||
        !multiply_size(
            options->skin_token_count,
            options->fps_candidate_multiplier,
            &skin_candidates) ||
        options->surface_sample_count < mesh_candidates ||
        options->surface_sample_count < skin_candidates) {
        set_error(
            error_out,
            error_size,
            "surface samples must cover each unique FPS candidate pool");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (size_t element = 0; element < asset->triangle_count * 3u; ++element) {
        if ((size_t) asset->triangles[element] >= asset->vertex_count) {
            set_error(
                error_out,
                error_size,
                "triangle index %u is outside %zu vertices",
                asset->triangles[element],
                asset->vertex_count);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }

    size_t vertex_values = 0;
    size_t sample_values = 0;
    if (!multiply_size(asset->vertex_count, 3u, &vertex_values) ||
        !multiply_size(options->surface_sample_count, 3u, &sample_values) ||
        vertex_values > SIZE_MAX / sizeof(float) || sample_values > SIZE_MAX / sizeof(float)) {
        set_error(error_out, error_size, "preprocessing allocation size overflow");
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_mesh_rigging_preprocessed result = TRELLIS_MESH_RIGGING_PREPROCESSED_INIT;
    result.vertex_count = asset->vertex_count;
    result.sample_count = options->surface_sample_count;
    result.normalized_vertices = (float *) malloc(vertex_values * sizeof(float));
    result.sample_positions = (float *) malloc(sample_values * sizeof(float));
    result.sample_normals = (float *) malloc(sample_values * sizeof(float));
    result.sample_triangles = (uint32_t *) malloc(result.sample_count * sizeof(uint32_t));
    result.sample_barycentric = (float *) malloc(sample_values * sizeof(float));
    if (result.normalized_vertices == NULL || result.sample_positions == NULL ||
        result.sample_normals == NULL || result.sample_triangles == NULL ||
        result.sample_barycentric == NULL) {
        set_error(error_out, error_size, "out of memory allocating preprocessed mesh");
        trellis_mesh_rigging_preprocessed_free(&result);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_status status = build_normalization(
        asset,
        &result.normalization,
        error_out,
        error_size);
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_rigging_preprocessed_free(&result);
        return status;
    }
    for (size_t vertex = 0; vertex < asset->vertex_count; ++vertex) {
        trellis_mesh_rigging_normalize_point(
            &result.normalization,
            asset->positions + vertex * 3u,
            result.normalized_vertices + vertex * 3u);
    }

    trellis_mesh_rigging_rng rng;
    rng_seed(&rng, options->seed);
    status = sample_surface(
        asset,
        result.normalized_vertices,
        result.sample_count,
        &rng,
        result.sample_positions,
        result.sample_normals,
        result.sample_triangles,
        result.sample_barycentric,
        error_out,
        error_size);
    if (status == TRELLIS_STATUS_OK) {
        status = build_fps_set(
            result.sample_positions,
            result.sample_count,
            options->mesh_token_count,
            options->fps_candidate_multiplier,
            &rng,
            &result.mesh_candidate_indices,
            &result.mesh_candidate_count,
            &result.mesh_fps_indices,
            error_out,
            error_size);
        result.mesh_fps_count = status == TRELLIS_STATUS_OK ? options->mesh_token_count : 0;
    }
    if (status == TRELLIS_STATUS_OK) {
        status = build_fps_set(
            result.sample_positions,
            result.sample_count,
            options->skin_token_count,
            options->fps_candidate_multiplier,
            &rng,
            &result.skin_candidate_indices,
            &result.skin_candidate_count,
            &result.skin_fps_indices,
            error_out,
            error_size);
        result.skin_fps_count = status == TRELLIS_STATUS_OK ? options->skin_token_count : 0;
    }
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_rigging_preprocessed_free(&result);
        return status;
    }

    *output = result;
    return TRELLIS_STATUS_OK;
}
