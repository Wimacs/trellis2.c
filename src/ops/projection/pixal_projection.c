#include "pixal_projection.h"

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

trellis_status trellis_pixal_resolve_camera_distance(
    float camera_angle_x,
    float mesh_scale,
    float requested_distance,
    float * resolved_distance) {
    static const float pi = 3.14159265358979323846f;
    if (resolved_distance == NULL || !isfinite(camera_angle_x) ||
        camera_angle_x <= 0.0f || camera_angle_x >= pi ||
        !isfinite(mesh_scale) || mesh_scale <= 0.0f ||
        !isfinite(requested_distance)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (requested_distance > 0.0f) {
        *resolved_distance = requested_distance;
        return TRELLIS_STATUS_OK;
    }

    const float distance =
        1.0f / (2.0f * mesh_scale * tanf(0.5f * camera_angle_x));
    if (!isfinite(distance) || distance <= 0.0f) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *resolved_distance = distance;
    return TRELLIS_STATUS_OK;
}

static int valid_projection_args(
    const float * patch_features,
    int patch_h,
    int patch_w,
    int channels,
    int image_resolution,
    int grid_resolution,
    const trellis_pixal_camera * camera,
    const float * projected_out) {
    return patch_features != NULL && projected_out != NULL && camera != NULL &&
        patch_h > 0 && patch_w > 0 && channels > 0 && image_resolution > 0 &&
        grid_resolution > 0 && isfinite(camera->camera_angle_x) &&
        camera->camera_angle_x > 0.0f && camera->camera_angle_x < 3.14159265358979323846f &&
        isfinite(camera->distance) && isfinite(camera->mesh_scale) && camera->mesh_scale != 0.0f;
}

static float grid_coord(int index, int resolution) {
    if (resolution == 1) {
        return 0.0f;
    }
    return -1.0f + 2.0f * (float) index / (float) (resolution - 1);
}

static void project_one(
    const float * patch_features,
    int patch_h,
    int patch_w,
    int channels,
    int image_resolution,
    int grid_resolution,
    const trellis_pixal_camera * camera,
    int gx,
    int gy,
    int gz,
    float * out) {
    const float inv_scale_2 = 0.5f / camera->mesh_scale;
    const float x_world = grid_coord(gx, grid_resolution) * inv_scale_2;
    const float y_camera = grid_coord(gy, grid_resolution) * inv_scale_2;
    const float depth = camera->distance - grid_coord(gz, grid_resolution) * inv_scale_2;
    const float focal_pixels =
        0.5f * (float) image_resolution / tanf(0.5f * camera->camera_angle_x);

    /* The fixed Blender front-view transform makes camera x=grid x and
       camera y=grid y after Pixal3D's axis rotation. The source computes a
       validity mask but intentionally samples border values without using it. */
    const float denom = depth + 1e-8f;
    const float pixel_x = focal_pixels * x_world / denom + 0.5f * (float) image_resolution;
    const float pixel_y = -focal_pixels * y_camera / denom + 0.5f * (float) image_resolution;

    /* Exact composition of Pixal3D's pixel -> normalized grid conversion and
       align_corners=false grid_sample coordinate transform. */
    float sample_x = (pixel_x + 0.5f) * (float) patch_w / (float) image_resolution - 0.5f;
    float sample_y = (pixel_y + 0.5f) * (float) patch_h / (float) image_resolution - 0.5f;
    if (!isfinite(sample_x)) sample_x = sample_x < 0.0f ? 0.0f : (float) (patch_w - 1);
    if (!isfinite(sample_y)) sample_y = sample_y < 0.0f ? 0.0f : (float) (patch_h - 1);
    sample_x = fminf(fmaxf(sample_x, 0.0f), (float) (patch_w - 1));
    sample_y = fminf(fmaxf(sample_y, 0.0f), (float) (patch_h - 1));

    const int x0 = (int) floorf(sample_x);
    const int y0 = (int) floorf(sample_y);
    const int x1 = x0 + 1 < patch_w ? x0 + 1 : x0;
    const int y1 = y0 + 1 < patch_h ? y0 + 1 : y0;
    const float wx = sample_x - (float) x0;
    const float wy = sample_y - (float) y0;
    const float w00 = (1.0f - wx) * (1.0f - wy);
    const float w01 = wx * (1.0f - wy);
    const float w10 = (1.0f - wx) * wy;
    const float w11 = wx * wy;
    const float * p00 = patch_features + ((size_t) y0 * (size_t) patch_w + (size_t) x0) * (size_t) channels;
    const float * p01 = patch_features + ((size_t) y0 * (size_t) patch_w + (size_t) x1) * (size_t) channels;
    const float * p10 = patch_features + ((size_t) y1 * (size_t) patch_w + (size_t) x0) * (size_t) channels;
    const float * p11 = patch_features + ((size_t) y1 * (size_t) patch_w + (size_t) x1) * (size_t) channels;
    for (int c = 0; c < channels; ++c) {
        out[c] = w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
    }
}

trellis_status trellis_pixal_project_patch_features_dense_f32(
    const float * patch_features,
    int patch_h,
    int patch_w,
    int channels,
    int image_resolution,
    int grid_resolution,
    const trellis_pixal_camera * camera,
    float * projected_out,
    size_t projected_count) {
    if (!valid_projection_args(
            patch_features, patch_h, patch_w, channels, image_resolution,
            grid_resolution, camera, projected_out)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    size_t grid_count = 0;
    size_t needed = 0;
    if (!checked_mul_size((size_t) grid_resolution, (size_t) grid_resolution, &grid_count) ||
        !checked_mul_size(grid_count, (size_t) grid_resolution, &grid_count) ||
        !checked_mul_size(grid_count, (size_t) channels, &needed) || projected_count < needed) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    size_t token = 0;
    for (int x = 0; x < grid_resolution; ++x) {
        for (int y = 0; y < grid_resolution; ++y) {
            for (int z = 0; z < grid_resolution; ++z, ++token) {
                project_one(
                    patch_features, patch_h, patch_w, channels, image_resolution,
                    grid_resolution, camera, x, y, z,
                    projected_out + token * (size_t) channels);
            }
        }
    }
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pixal_project_patch_features_sparse_f32(
    const float * patch_features,
    int patch_h,
    int patch_w,
    int channels,
    int image_resolution,
    int grid_resolution,
    const trellis_pixal_camera * camera,
    const int32_t * coords_bxyz,
    int64_t n_coords,
    float * projected_out,
    size_t projected_count) {
    if (!valid_projection_args(
            patch_features, patch_h, patch_w, channels, image_resolution,
            grid_resolution, camera, projected_out) || coords_bxyz == NULL || n_coords < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    size_t needed = 0;
    if ((uint64_t) n_coords > (uint64_t) SIZE_MAX ||
        !checked_mul_size((size_t) n_coords, (size_t) channels, &needed) || projected_count < needed) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int64_t i = 0; i < n_coords; ++i) {
        const int32_t batch = coords_bxyz[(size_t) i * 4u + 0u];
        const int32_t x = coords_bxyz[(size_t) i * 4u + 1u];
        const int32_t y = coords_bxyz[(size_t) i * 4u + 2u];
        const int32_t z = coords_bxyz[(size_t) i * 4u + 3u];
        if (batch != 0 || x < 0 || y < 0 || z < 0 ||
            x >= grid_resolution || y >= grid_resolution || z >= grid_resolution) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        project_one(
            patch_features, patch_h, patch_w, channels, image_resolution,
            grid_resolution, camera, x, y, z,
            projected_out + (size_t) i * (size_t) channels);
    }
    return TRELLIS_STATUS_OK;
}
