#include "projected_attention.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int checked_mul_size(size_t a, size_t b, size_t * out) {
    if (out == NULL || (a != 0 && b > SIZE_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int checked_tensor_extent(
    int batch,
    int height,
    int width,
    int channels) {
    size_t count = 0;
    return batch > 0 && height > 0 && width > 0 && channels > 0 &&
        checked_mul_size((size_t) batch, (size_t) height, &count) &&
        checked_mul_size(count, (size_t) width, &count) &&
        checked_mul_size(count, (size_t) channels, &count);
}

static int valid_desc(const trellis_pixal_naf_attention_desc * desc) {
    if (desc == NULL || desc->batch <= 0 || desc->query_h <= 0 || desc->query_w <= 0 ||
        desc->feature_h < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE ||
        desc->feature_w < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE ||
        desc->image_resolution <= 0 || desc->grid_resolution <= 0 ||
        desc->query_h % desc->feature_h != 0 || desc->query_w % desc->feature_w != 0) {
        return 0;
    }
    const int dilation_h = desc->query_h / desc->feature_h;
    const int dilation_w = desc->query_w / desc->feature_w;
    if (dilation_h <= 0 || dilation_h != dilation_w) {
        return 0;
    }
    return checked_tensor_extent(
               desc->batch,
               desc->query_h,
               desc->query_w,
               TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS) &&
        checked_tensor_extent(
               desc->batch,
               desc->feature_h,
               desc->feature_w,
               TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS) &&
        checked_tensor_extent(
               desc->batch,
               desc->feature_h,
               desc->feature_w,
               TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS);
}

static int valid_camera(const trellis_pixal_camera * camera) {
    static const float pi = 3.14159265358979323846f;
    return camera != NULL && isfinite(camera->camera_angle_x) &&
        camera->camera_angle_x > 0.0f && camera->camera_angle_x < pi &&
        isfinite(camera->distance) && isfinite(camera->mesh_scale) &&
        camera->mesh_scale != 0.0f;
}

static size_t bhwc_offset(
    int batch,
    int y,
    int x,
    int height,
    int width,
    int channels) {
    return ((((size_t) batch * (size_t) height + (size_t) y) * (size_t) width +
             (size_t) x) *
            (size_t) channels);
}

static int clamp_int(int value, int lower, int upper) {
    if (value < lower) return lower;
    if (value > upper) return upper;
    return value;
}

/*
 * NATTEN applies dilation by splitting an axis into residue groups. The
 * ordinary shifted-boundary window is then evaluated in group coordinates.
 * With the exact integer Pixal3D resize ratio, nearest-exact maps every
 * enlarged member of the group directly to the same low-resolution index.
 */
static int low_resolution_window_start(
    int query_index,
    int dilation,
    int feature_length) {
    const int group_coordinate = query_index / dilation;
    return clamp_int(
        group_coordinate - TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE / 2,
        0,
        feature_length - TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE);
}

static void attention_pixel_accumulate(
    const float * queries,
    const float * keys,
    const float * values,
    const trellis_pixal_naf_attention_desc * desc,
    int batch_index,
    int query_y,
    int query_x,
    float output_weight,
    float * output) {
    const int dilation = desc->query_h / desc->feature_h;
    const int start_y = low_resolution_window_start(query_y, dilation, desc->feature_h);
    const int start_x = low_resolution_window_start(query_x, dilation, desc->feature_w);
    const size_t query_base = bhwc_offset(
        batch_index,
        query_y,
        query_x,
        desc->query_h,
        desc->query_w,
        TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS);

    for (int head = 0; head < TRELLIS_PIXAL_NAF_ATTENTION_HEADS; ++head) {
        float weights[
            TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE *
            TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE];
        float maximum = -INFINITY;
        int neighbor = 0;
        for (int ky = 0; ky < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE; ++ky) {
            const int feature_y = start_y + ky;
            for (int kx = 0; kx < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE; ++kx) {
                const int feature_x = start_x + kx;
                const size_t key_base = bhwc_offset(
                    batch_index,
                    feature_y,
                    feature_x,
                    desc->feature_h,
                    desc->feature_w,
                    TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS);
                float dot = 0.0f;
                for (int channel = 0;
                     channel < TRELLIS_PIXAL_NAF_ATTENTION_QUERY_HEAD_DIM;
                     ++channel) {
                    const int head_channel =
                        head * TRELLIS_PIXAL_NAF_ATTENTION_QUERY_HEAD_DIM + channel;
                    dot += queries[query_base + (size_t) head_channel] *
                        keys[key_base + (size_t) head_channel];
                }
                const float score = dot * 0.125f;
                weights[neighbor++] = score;
                if (score > maximum) maximum = score;
            }
        }

        float denominator = 0.0f;
        for (int i = 0; i < neighbor; ++i) {
            weights[i] = expf(weights[i] - maximum);
            denominator += weights[i];
        }
        const float normalization = output_weight / denominator;
        const int output_head = head * TRELLIS_PIXAL_NAF_ATTENTION_VALUE_HEAD_DIM;
        for (int channel = 0;
             channel < TRELLIS_PIXAL_NAF_ATTENTION_VALUE_HEAD_DIM;
             ++channel) {
            float value_sum = 0.0f;
            neighbor = 0;
            for (int ky = 0; ky < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE; ++ky) {
                const int feature_y = start_y + ky;
                for (int kx = 0; kx < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE; ++kx) {
                    const int feature_x = start_x + kx;
                    const size_t value_base = bhwc_offset(
                        batch_index,
                        feature_y,
                        feature_x,
                        desc->feature_h,
                        desc->feature_w,
                        TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS);
                    value_sum += weights[neighbor++] *
                        values[value_base + (size_t) (output_head + channel)];
                }
            }
            output[output_head + channel] += value_sum * normalization;
        }
    }
}

trellis_status trellis_pixal_naf_attention_pixel_reference_f32(
    const float * queries,
    const float * keys,
    const float * values,
    const trellis_pixal_naf_attention_desc * desc,
    int batch_index,
    int query_y,
    int query_x,
    float * output,
    size_t output_count) {
    if (queries == NULL || keys == NULL || values == NULL || output == NULL ||
        !valid_desc(desc) || batch_index < 0 || batch_index >= desc->batch ||
        query_y < 0 || query_y >= desc->query_h ||
        query_x < 0 || query_x >= desc->query_w ||
        output_count < TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int channel = 0;
         channel < TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS;
         ++channel) {
        output[channel] = 0.0f;
    }
    attention_pixel_accumulate(
        queries,
        keys,
        values,
        desc,
        batch_index,
        query_y,
        query_x,
        1.0f,
        output);
    return TRELLIS_STATUS_OK;
}

static float grid_coordinate(int index, int resolution) {
    if (resolution == 1) return 0.0f;
    return -1.0f + 2.0f * (float) index / (float) (resolution - 1);
}

static void projected_query_coordinate(
    const trellis_pixal_naf_attention_desc * desc,
    const trellis_pixal_camera * camera,
    int grid_x,
    int grid_y,
    int grid_z,
    float * sample_x,
    float * sample_y) {
    const float inv_scale_2 = 0.5f / camera->mesh_scale;
    const float world_x = grid_coordinate(grid_x, desc->grid_resolution) * inv_scale_2;
    const float camera_y = grid_coordinate(grid_y, desc->grid_resolution) * inv_scale_2;
    const float depth =
        camera->distance - grid_coordinate(grid_z, desc->grid_resolution) * inv_scale_2;
    const float focal_pixels =
        0.5f * (float) desc->image_resolution /
        tanf(0.5f * camera->camera_angle_x);
    const float denominator = depth + 1e-8f;
    const float pixel_x =
        focal_pixels * world_x / denominator + 0.5f * (float) desc->image_resolution;
    const float pixel_y =
        -focal_pixels * camera_y / denominator + 0.5f * (float) desc->image_resolution;

    float x =
        (pixel_x + 0.5f) * (float) desc->query_w /
            (float) desc->image_resolution -
        0.5f;
    float y =
        (pixel_y + 0.5f) * (float) desc->query_h /
            (float) desc->image_resolution -
        0.5f;
    if (!isfinite(x)) x = x < 0.0f ? 0.0f : (float) (desc->query_w - 1);
    if (!isfinite(y)) y = y < 0.0f ? 0.0f : (float) (desc->query_h - 1);
    *sample_x = fminf(fmaxf(x, 0.0f), (float) (desc->query_w - 1));
    *sample_y = fminf(fmaxf(y, 0.0f), (float) (desc->query_h - 1));
}

trellis_status trellis_pixal_naf_attention_project_sparse_reference_f32(
    const float * queries,
    const float * keys,
    const float * values,
    const int32_t * coords_bxyz,
    int64_t n_coords,
    const trellis_pixal_camera * cameras,
    const trellis_pixal_naf_attention_desc * desc,
    float * projected_out,
    size_t projected_count) {
    if (queries == NULL || keys == NULL || values == NULL || coords_bxyz == NULL ||
        cameras == NULL || projected_out == NULL || n_coords < 0 || !valid_desc(desc)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    size_t needed = 0;
    if ((uint64_t) n_coords > (uint64_t) SIZE_MAX ||
        !checked_mul_size(
            (size_t) n_coords,
            TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS,
            &needed) ||
        projected_count < needed) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int batch = 0; batch < desc->batch; ++batch) {
        if (!valid_camera(cameras + batch)) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }
    /* Validate all rows before writing any output. */
    for (int64_t i = 0; i < n_coords; ++i) {
        const int32_t batch = coords_bxyz[(size_t) i * 4u + 0u];
        const int32_t x = coords_bxyz[(size_t) i * 4u + 1u];
        const int32_t y = coords_bxyz[(size_t) i * 4u + 2u];
        const int32_t z = coords_bxyz[(size_t) i * 4u + 3u];
        if (batch < 0 || batch >= desc->batch || x < 0 || y < 0 || z < 0 ||
            x >= desc->grid_resolution || y >= desc->grid_resolution ||
            z >= desc->grid_resolution) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }

    for (int64_t i = 0; i < n_coords; ++i) {
        const int batch = coords_bxyz[(size_t) i * 4u + 0u];
        const int grid_x = coords_bxyz[(size_t) i * 4u + 1u];
        const int grid_y = coords_bxyz[(size_t) i * 4u + 2u];
        const int grid_z = coords_bxyz[(size_t) i * 4u + 3u];
        float * output = projected_out +
            (size_t) i * TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS;
        for (int channel = 0;
             channel < TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS;
             ++channel) {
            output[channel] = 0.0f;
        }

        float sample_x = 0.0f;
        float sample_y = 0.0f;
        projected_query_coordinate(
            desc,
            cameras + batch,
            grid_x,
            grid_y,
            grid_z,
            &sample_x,
            &sample_y);
        const int x0 = (int) floorf(sample_x);
        const int y0 = (int) floorf(sample_y);
        const int x1 = x0 + 1 < desc->query_w ? x0 + 1 : x0;
        const int y1 = y0 + 1 < desc->query_h ? y0 + 1 : y0;
        const float wx = sample_x - (float) x0;
        const float wy = sample_y - (float) y0;
        const float weights[4] = {
            (1.0f - wx) * (1.0f - wy),
            wx * (1.0f - wy),
            (1.0f - wx) * wy,
            wx * wy,
        };
        const int xs[4] = {x0, x1, x0, x1};
        const int ys[4] = {y0, y0, y1, y1};
        for (int corner = 0; corner < 4; ++corner) {
            if (weights[corner] == 0.0f) continue;
            attention_pixel_accumulate(
                queries,
                keys,
                values,
                desc,
                batch,
                ys[corner],
                xs[corner],
                weights[corner],
                output);
        }
    }
    return TRELLIS_STATUS_OK;
}
