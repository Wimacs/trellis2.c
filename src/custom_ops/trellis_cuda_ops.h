#ifndef TRELLIS2_C_CUSTOM_OPS_TRELLIS_CUDA_OPS_H
#define TRELLIS2_C_CUSTOM_OPS_TRELLIS_CUDA_OPS_H

#include "trellis.h"

#include <cuda_runtime.h>

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sparse_neighbor_map_device {
    int32_t * indices;
    int32_t * src_rows;
    int32_t * dst_rows;
    int32_t * offset_counts_dev;
    int32_t * offset_starts_dev;
    int32_t * offset_counts_host;
    int32_t * offset_starts_host;
    int64_t n;
    int64_t total_pairs;
    int k_volume;
    int kernel_d;
    int kernel_h;
    int kernel_w;
    int dilation_d;
    int dilation_h;
    int dilation_w;
} sparse_neighbor_map_device;

struct sparse_rulebook_matmul_workspace;

typedef struct sparse_rulebook_matmul_workspace_cache {
    struct sparse_rulebook_matmul_workspace * head;
} sparse_rulebook_matmul_workspace_cache;

void sparse_neighbor_map_free(sparse_neighbor_map_device * map);

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

void sparse_rulebook_matmul_workspace_cache_free(
    sparse_rulebook_matmul_workspace_cache * cache);

trellis_status sparse_subm_conv3d_dispatch_f32(
    ggml_backend_t backend,
    sparse_rulebook_matmul_workspace_cache * workspace_cache,
    const sparse_neighbor_map_device * map,
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

trellis_status sparse_linear_device_scalar(
    const float * x_dev,
    const struct ggml_tensor * w,
    const struct ggml_tensor * b,
    float * y_dev,
    int64_t n,
    int in_channels,
    int out_channels);

trellis_status sparse_linear_device(
    ggml_backend_t backend,
    const float * x_dev,
    const struct ggml_tensor * w,
    const struct ggml_tensor * b,
    float * y_dev,
    int64_t n,
    int in_channels,
    int out_channels);

trellis_status row_layer_norm_device(
    const float * x_dev,
    const struct ggml_tensor * gamma,
    const struct ggml_tensor * beta,
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

#ifdef __cplusplus
}
#endif

#endif
