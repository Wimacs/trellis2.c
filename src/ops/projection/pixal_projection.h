#ifndef TRELLIS2_C_OPS_PIXAL_PROJECTION_H
#define TRELLIS2_C_OPS_PIXAL_PROJECTION_H

#include "trellis.h"

typedef struct trellis_pixal_camera {
    float camera_angle_x;
    float distance;
    float mesh_scale;
} trellis_pixal_camera;

/* A non-positive distance selects the Pixal3D reference camera fit.
 * Positive distances are preserved verbatim. */
trellis_status trellis_pixal_resolve_camera_distance(
    float camera_angle_x,
    float mesh_scale,
    float requested_distance,
    float * resolved_distance);

/*
 * Projects an x-major R^3 grid into a row-major image feature map and samples
 * it with PyTorch grid_sample(..., align_corners=false, padding_mode=border)
 * semantics. Feature vectors are contiguous in the last dimension.
 */
trellis_status trellis_pixal_project_patch_features_dense_f32(
    const float * patch_features,
    int patch_h,
    int patch_w,
    int channels,
    int image_resolution,
    int grid_resolution,
    const trellis_pixal_camera * camera,
    float * projected_out,
    size_t projected_count);

/* Same projection, gathered in the exact order of sparse [batch,x,y,z] rows. */
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
    size_t projected_count);

#endif
