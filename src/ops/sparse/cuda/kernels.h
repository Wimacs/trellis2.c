#ifndef TRELLIS2_C_OPS_SPARSE_CUDA_KERNELS_H
#define TRELLIS2_C_OPS_SPARSE_CUDA_KERNELS_H

#include "trellis.h"

#include <cuda_runtime.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

trellis_status trellis_cuda_silu_f32(
    const float * x_dev,
    float * y_dev,
    size_t n);

trellis_status trellis_cuda_silu_f32_host(
    const float * x,
    float * y,
    int device,
    size_t n);

trellis_status trellis_cuda_add_f32(
    const float * a_dev,
    const float * b_dev,
    float * y_dev,
    size_t n);

trellis_status trellis_cuda_add_f32_host(
    const float * a,
    const float * b,
    float * y,
    int device,
    size_t n);

trellis_status trellis_cuda_sparse_linear_f32_host(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int device,
    int64_t n,
    int in_channels,
    int out_channels);

trellis_status trellis_cuda_sparse_subm_conv3d_f32(
    const int32_t * coords_dev,
    const float * feats_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * out_dev,
    int64_t n,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w);

trellis_status trellis_cuda_sparse_subm_conv3d_f32_host(
    const int32_t * coords,
    const float * feats,
    const float * weight,
    const float * bias,
    float * out,
    int device,
    int64_t n,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w);

typedef struct sparse_neighbor_map_device {
    int32_t * indices;
    int32_t * src_rows;
    int32_t * dst_rows;
    int32_t * offset_counts_dev;
    int32_t * offset_starts_dev;
    int32_t * offset_counts_host;
    int32_t * offset_starts_host;
    int32_t * tile_valid_offsets;
    int32_t * tile_valid_counts;
    int64_t n;
    int64_t total_pairs;
    int fixed_rulebook;
    int k_volume;
    int kernel_d;
    int kernel_h;
    int kernel_w;
    int dilation_d;
    int dilation_h;
    int dilation_w;
} sparse_neighbor_map_device;

typedef struct sparse_subm_conv3d_device {
    sparse_neighbor_map_device neighbor_map;
} sparse_subm_conv3d_device;

typedef struct sparse_c2s_map_device {
    int32_t * coords;
    int32_t * parent;
    int32_t * subidx;
    int64_t n;
} sparse_c2s_map_device;

void sparse_neighbor_map_free(sparse_neighbor_map_device * map);

void sparse_subm_conv3d_device_free(sparse_subm_conv3d_device * conv);

trellis_status sparse_subm_conv3d_device_build(
    const int32_t * coords_dev,
    int64_t n,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    sparse_subm_conv3d_device * conv);

trellis_status sparse_subm_conv3d_device_forward_f32(
    const sparse_subm_conv3d_device * conv,
    const float * feats_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * out_dev,
    int64_t n,
    int in_channels,
    int out_channels);

void sparse_c2s_map_device_free(sparse_c2s_map_device * map);

trellis_status sparse_c2s_map_build_device(
    const int32_t * coords_dev,
    const float * subdiv_logits_dev,
    int64_t n,
    sparse_c2s_map_device * map);

trellis_status sparse_neighbor_map_build_device(
    const int32_t * coords_dev,
    int64_t n,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int build_rulebook,
    sparse_neighbor_map_device * map);

trellis_status sparse_linear_device_cublas(
    const float * x_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * y_dev,
    int64_t n,
    int in_channels,
    int out_channels);

trellis_status row_layer_norm_device(
    const float * x_dev,
    const float * gamma_dev,
    const float * beta_dev,
    float * y_dev,
    int64_t n,
    int channels,
    float eps);

trellis_status sparse_c2s_gather_device(
    const float * x_dev,
    const int32_t * parent_dev,
    const int32_t * subidx_dev,
    float * y_dev,
    int64_t n_out,
    int out_channels);

trellis_status sparse_c2s_skip_repeat_device(
    const float * x_dev,
    const int32_t * parent_dev,
    const int32_t * subidx_dev,
    float * y_dev,
    int64_t n_out,
    int in_channels,
    int out_channels);

/* Sparse-to-channel is the inverse layout of C2S.  Each fine row is written
 * into the sub-voxel channel slice of its coarse parent; missing children stay
 * zero.  The output is [n_out, 8 * in_channels]. */
trellis_status sparse_s2c_scatter_device(
    const float * x_dev,
    const int32_t * parent_dev,
    const int32_t * subidx_dev,
    float * y_dev,
    int64_t n_in,
    int64_t n_out,
    int in_channels);

/* Implements S2C(x).reshape(n_out, out_channels, -1).mean(-1) without
 * materializing the intermediate 8 * in_channels tensor. */
trellis_status sparse_s2c_skip_mean_device(
    const float * x_dev,
    const int32_t * parent_dev,
    const int32_t * subidx_dev,
    float * y_dev,
    int64_t n_in,
    int64_t n_out,
    int in_channels,
    int out_channels);

#ifdef __cplusplus
}
#endif

#endif
