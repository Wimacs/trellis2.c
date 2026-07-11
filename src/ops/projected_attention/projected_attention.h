#ifndef TRELLIS2_C_PROJECTED_ATTENTION_H
#define TRELLIS2_C_PROJECTED_ATTENTION_H

#include "pixal_projection.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    TRELLIS_PIXAL_NAF_ATTENTION_HEADS = 4,
    TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS = 256,
    TRELLIS_PIXAL_NAF_ATTENTION_QUERY_HEAD_DIM = 64,
    TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS = 1024,
    TRELLIS_PIXAL_NAF_ATTENTION_VALUE_HEAD_DIM = 256,
    TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE = 9,
};

/*
 * Shared shape contract for the fused Pixal3D NAF attention/projection
 * operator. All feature tensors are contiguous F32 with channels last:
 *
 *   queries: [batch, query_h,   query_w,   256]
 *   keys:    [batch, feature_h, feature_w, 256]
 *   values:  [batch, feature_h, feature_w, 1024]
 *   output:  [n_coords, 1024]
 *
 * query_h / feature_h and query_w / feature_w must be the same positive
 * integer. This ratio is the NATTEN dilation used after Pixal3D's logical
 * nearest-exact resize of keys and values. The feature dimensions must each
 * be at least the 9-element neighborhood size.
 */
typedef struct trellis_pixal_naf_attention_desc {
    int batch;
    int query_h;
    int query_w;
    int feature_h;
    int feature_w;
    int image_resolution;
    int grid_resolution;
} trellis_pixal_naf_attention_desc;

/*
 * CPU correctness oracle for one integer pixel of the high-resolution NAF
 * output. It implements four heads, 9x9 shifted-boundary neighborhoods,
 * scale=1/sqrt(64), and a per-head softmax. No high-resolution K/V tensor is
 * materialized.
 */
trellis_status trellis_pixal_naf_attention_pixel_reference_f32(
    const float * queries,
    const float * keys,
    const float * values,
    const trellis_pixal_naf_attention_desc * desc,
    int batch_index,
    int query_y,
    int query_x,
    float * output,
    size_t output_count);

/*
 * CPU correctness oracle for the complete fused operator. coords_bxyz is a
 * contiguous [n_coords,4] array. cameras is a [batch] array indexed by the
 * coordinate's batch field. For each sparse coordinate this computes only
 * the four integer NAF pixels needed by Pixal3D's bilinear
 * grid_sample(..., align_corners=false, padding_mode=border), then blends
 * them directly into one 1024-channel token.
 */
trellis_status trellis_pixal_naf_attention_project_sparse_reference_f32(
    const float * queries,
    const float * keys,
    const float * values,
    const int32_t * coords_bxyz,
    int64_t n_coords,
    const trellis_pixal_camera * cameras,
    const trellis_pixal_naf_attention_desc * desc,
    float * projected_out,
    size_t projected_count);

struct trellis_sparse_backend;

/* Vulkan fused implementation using an existing sparse Vulkan backend. The
 * host-side tensor contract is identical to the reference function above. */
trellis_status trellis_pixal_naf_attention_project_sparse_vulkan_f32(
    struct trellis_sparse_backend * backend,
    const float * queries,
    const float * keys,
    const float * values,
    const int32_t * coords_bxyz,
    int64_t n_coords,
    const trellis_pixal_camera * cameras,
    const trellis_pixal_naf_attention_desc * desc,
    float * projected_out,
    size_t projected_count);

/* CUDA device-pointer variant. The current CUDA device and default stream are
 * used; all tensor, coordinate, camera, and output pointers must reside on
 * that device. Coordinate and camera contents must satisfy the same contract
 * as the CPU reference. */
trellis_status trellis_pixal_naf_attention_project_sparse_cuda_f32(
    const float * queries_dev,
    const float * keys_dev,
    const float * values_dev,
    const int32_t * coords_bxyz_dev,
    int64_t n_coords,
    const trellis_pixal_camera * cameras_dev,
    const trellis_pixal_naf_attention_desc * desc,
    float * projected_out_dev,
    size_t projected_count);

/* Convenience wrapper for host pointers. It selects device, uploads the
 * inputs, runs the same fused kernel, and downloads [n_coords,1024]. */
trellis_status trellis_pixal_naf_attention_project_sparse_cuda_host_f32(
    const float * queries,
    const float * keys,
    const float * values,
    const int32_t * coords_bxyz,
    int64_t n_coords,
    const trellis_pixal_camera * cameras,
    const trellis_pixal_naf_attention_desc * desc,
    int device,
    float * projected_out,
    size_t projected_count);

#ifdef __cplusplus
}
#endif

#endif
