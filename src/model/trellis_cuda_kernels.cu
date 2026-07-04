#include "trellis_cuda_kernels.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static trellis_status cuda_status_to_trellis(cudaError_t err) {
    return err == cudaSuccess ? TRELLIS_STATUS_OK : TRELLIS_STATUS_ERROR;
}

static trellis_status cublas_status_to_trellis(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return TRELLIS_STATUS_OK;
    }
    if (status == CUBLAS_STATUS_ALLOC_FAILED) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    return TRELLIS_STATUS_ERROR;
}

static const char * cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

static trellis_status trellis_cublas_handle(cublasHandle_t * handle_out) {
    enum { TRELLIS_CUBLAS_MAX_DEVICES = 16 };
    static cublasHandle_t handles[TRELLIS_CUBLAS_MAX_DEVICES];
    if (handle_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }
    if (device < 0 || device >= TRELLIS_CUBLAS_MAX_DEVICES) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (handles[device] == NULL) {
        cublasStatus_t status = cublasCreate(&handles[device]);
        if (status != CUBLAS_STATUS_SUCCESS) {
            handles[device] = NULL;
            return cublas_status_to_trellis(status);
        }
#ifdef CUBLAS_TF32_TENSOR_OP_MATH
        status = cublasSetMathMode(handles[device], CUBLAS_TF32_TENSOR_OP_MATH);
        if (status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handles[device]);
            handles[device] = NULL;
            return cublas_status_to_trellis(status);
        }
#endif
    }
    cublasStatus_t status = cublasSetStream(handles[device], 0);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return cublas_status_to_trellis(status);
    }
    *handle_out = handles[device];
    return TRELLIS_STATUS_OK;
}

static trellis_status malloc_copy_to_device(const float * src, size_t count, float ** dst) {
    *dst = NULL;
    cudaError_t err = cudaMalloc((void **) dst, count * sizeof(float));
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    err = cudaMemcpy(*dst, src, count * sizeof(float), cudaMemcpyHostToDevice);
    return cuda_status_to_trellis(err);
}

static trellis_status malloc_copy_i32_to_device(const int32_t * src, size_t count, int32_t ** dst) {
    *dst = NULL;
    cudaError_t err = cudaMalloc((void **) dst, count * sizeof(int32_t));
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    err = cudaMemcpy(*dst, src, count * sizeof(int32_t), cudaMemcpyHostToDevice);
    return cuda_status_to_trellis(err);
}

static trellis_status cuda_malloc_f32(size_t count, float ** dst) {
    *dst = NULL;
    cudaError_t err = cudaMalloc((void **) dst, count * sizeof(float));
    return err == cudaSuccess ? TRELLIS_STATUS_OK : TRELLIS_STATUS_OUT_OF_MEMORY;
}

static bool elementwise_args_valid(const float * x, float * y, size_t n) {
    return x != NULL && y != NULL && n > 0;
}

static bool add_args_valid(const float * a, const float * b, float * y, size_t n) {
    return a != NULL && b != NULL && y != NULL && n > 0;
}

static bool sparse_subm_conv3d_args_valid(
    const int32_t * coords,
    const float * feats,
    const float * weight,
    float * out,
    int64_t n,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w) {
    return coords != NULL && feats != NULL && weight != NULL && out != NULL &&
        n > 0 && in_channels > 0 && out_channels > 0 &&
        kernel_d > 0 && kernel_h > 0 && kernel_w > 0 &&
        (kernel_d & 1) == 1 && (kernel_h & 1) == 1 && (kernel_w & 1) == 1 &&
        dilation_d > 0 && dilation_h > 0 && dilation_w > 0;
}

static int64_t next_power_of_two_i64(int64_t x) {
    int64_t v = 1;
    while (v < x && v < (INT64_MAX / 2)) {
        v <<= 1;
    }
    return v;
}

static __device__ __forceinline__ unsigned long long trellis_pack_coord4(int b, int x, int y, int z) {
    const unsigned long long raw =
        ((unsigned long long) ((uint32_t) b & 0xffffu) << 48) |
        ((unsigned long long) ((uint32_t) x & 0xffffu) << 32) |
        ((unsigned long long) ((uint32_t) y & 0xffffu) << 16) |
        ((unsigned long long) ((uint32_t) z & 0xffffu));
    return raw + 1ull;
}

static __device__ __forceinline__ unsigned long long trellis_hash_u64(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

__global__ void trellis_sparse_hash_insert_kernel(
    const int32_t * __restrict__ coords,
    unsigned long long * __restrict__ keys,
    int32_t * __restrict__ values,
    int64_t n,
    int64_t table_mask) {
    const int64_t row = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (row >= n) {
        return;
    }
    const int32_t * c = coords + 4 * row;
    const unsigned long long key = trellis_pack_coord4(c[0], c[1], c[2], c[3]);
    unsigned long long slot = trellis_hash_u64(key) & (unsigned long long) table_mask;
    for (int64_t probe = 0; probe <= table_mask; ++probe) {
        const unsigned long long old = atomicCAS(&keys[slot], 0ull, key);
        if (old == 0ull || old == key) {
            values[slot] = (int32_t) row;
            return;
        }
        slot = (slot + 1ull) & (unsigned long long) table_mask;
    }
}

static __device__ __forceinline__ int trellis_sparse_hash_find(
    const unsigned long long * __restrict__ keys,
    const int32_t * __restrict__ values,
    int64_t table_mask,
    unsigned long long key) {
    unsigned long long slot = trellis_hash_u64(key) & (unsigned long long) table_mask;
    for (int64_t probe = 0; probe <= table_mask; ++probe) {
        const unsigned long long found = keys[slot];
        if (found == key) {
            return values[slot];
        }
        if (found == 0ull) {
            return -1;
        }
        slot = (slot + 1ull) & (unsigned long long) table_mask;
    }
    return -1;
}

__global__ void trellis_sparse_conv_fill_bias_f32_kernel(
    float * __restrict__ out,
    const float * __restrict__ bias,
    int64_t total,
    int out_channels) {
    const int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int oc = (int) (linear % out_channels);
    out[linear] = bias == NULL ? 0.0f : bias[oc];
}

__global__ void trellis_sparse_build_neighbor_map_kernel(
    const int32_t * __restrict__ coords,
    const unsigned long long * __restrict__ keys,
    const int32_t * __restrict__ values,
    int32_t * __restrict__ map,
    int64_t n,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int64_t table_mask,
    int64_t total) {
    const int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int64_t row = linear % n;
    const int k_index = (int) (linear / n);
    const int kw = k_index % kernel_w;
    const int kh = (k_index / kernel_w) % kernel_h;
    const int kd = k_index / (kernel_w * kernel_h);
    const int32_t * c = coords + 4 * row;
    const int nd = c[1] + (kd - kernel_d / 2) * dilation_d;
    const int nh = c[2] + (kh - kernel_h / 2) * dilation_h;
    const int nw = c[3] + (kw - kernel_w / 2) * dilation_w;
    const unsigned long long key = trellis_pack_coord4(c[0], nd, nh, nw);
    map[linear] = trellis_sparse_hash_find(keys, values, table_mask, key);
}

__global__ void trellis_sparse_rulebook_count_kernel(
    const int32_t * __restrict__ neighbor_map,
    int32_t * __restrict__ counts,
    int64_t n) {
    const int k = (int) blockIdx.x;
    const int tid = (int) threadIdx.x;
    __shared__ int scratch[256];
    int count = 0;
    for (int64_t row = tid; row < n; row += blockDim.x) {
        count += neighbor_map[(int64_t) k * n + row] >= 0 ? 1 : 0;
    }
    scratch[tid] = count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        counts[k] = scratch[0];
    }
}

__global__ void trellis_sparse_rulebook_fill_kernel(
    const int32_t * __restrict__ neighbor_map,
    int32_t * __restrict__ counters,
    int32_t * __restrict__ src_rows,
    int32_t * __restrict__ dst_rows,
    int64_t n,
    int64_t total) {
    const int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int64_t dst = linear % n;
    const int k = (int) (linear / n);
    const int32_t src = neighbor_map[linear];
    if (src < 0) {
        return;
    }
    const int32_t pos = atomicAdd(&counters[k], 1);
    src_rows[pos] = src;
    dst_rows[pos] = (int32_t) dst;
}

__global__ void trellis_sparse_subm_conv3d_implicit_gemm_f32_kernel(
    const float * __restrict__ feats,
    const float * __restrict__ weight,
    const int32_t * __restrict__ src_rows,
    const int32_t * __restrict__ dst_rows,
    float * __restrict__ out,
    int64_t pair_count,
    int in_channels,
    int out_channels,
    int k_volume,
    int offset) {
    enum {
        TILE_M = 16,
        TILE_N = 16,
        TILE_K = 32,
    };
    __shared__ float x_tile[TILE_M][TILE_K];
    __shared__ float w_tile[TILE_N][TILE_K];

    const int local_n = (int) threadIdx.x;
    const int local_m = (int) threadIdx.y;
    const int thread_linear = local_m * TILE_N + local_n;
    const int64_t pair_tile = ((int64_t) blockIdx.z * (int64_t) gridDim.y) + (int64_t) blockIdx.y;
    const int64_t pair = pair_tile * TILE_M + local_m;
    const int oc = (int) blockIdx.x * TILE_N + local_n;
    const int valid = pair < pair_count && oc < out_channels;
    float acc = 0.0f;

    for (int c0 = 0; c0 < in_channels; c0 += TILE_K) {
        for (int load = thread_linear; load < TILE_M * TILE_K; load += TILE_M * TILE_N) {
            const int m = load / TILE_K;
            const int kc = load - m * TILE_K;
            const int64_t p = pair_tile * TILE_M + m;
            const int c = c0 + kc;
            float v = 0.0f;
            if (p < pair_count && c < in_channels) {
                const int32_t src = src_rows[p];
                v = feats[(int64_t) src * in_channels + c];
            }
            x_tile[m][kc] = v;
        }
        for (int load = thread_linear; load < TILE_N * TILE_K; load += TILE_M * TILE_N) {
            const int n = load / TILE_K;
            const int kc = load - n * TILE_K;
            const int out_c = (int) blockIdx.x * TILE_N + n;
            const int c = c0 + kc;
            float v = 0.0f;
            if (out_c < out_channels && c < in_channels) {
                v = weight[((int64_t) out_c * k_volume + offset) * (int64_t) in_channels + c];
            }
            w_tile[n][kc] = v;
        }
        __syncthreads();
        if (valid) {
            #pragma unroll
            for (int kc = 0; kc < TILE_K; ++kc) {
                acc += x_tile[local_m][kc] * w_tile[local_n][kc];
            }
        }
        __syncthreads();
    }

    if (valid) {
        const int32_t dst = dst_rows[pair];
        out[(int64_t) dst * out_channels + oc] += acc;
    }
}

__global__ void trellis_silu_f32_kernel(
    const float * __restrict__ x,
    float * __restrict__ y,
    int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (i < n) {
        y[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

__global__ void trellis_add_f32_kernel(
    const float * __restrict__ a,
    const float * __restrict__ b,
    float * __restrict__ y,
    int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (i < n) {
        y[i] = a[i] + b[i];
    }
}

__global__ void trellis_linear_add_bias_f32_kernel(
    float * __restrict__ y,
    const float * __restrict__ bias,
    int out_channels,
    int64_t total) {
    const int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int oc = (int) (linear % out_channels);
    y[linear] += bias[oc];
}

__global__ void trellis_row_layer_norm_f32_kernel(
    const float * __restrict__ x,
    const float * __restrict__ gamma,
    const float * __restrict__ beta,
    float * __restrict__ y,
    int64_t n,
    int channels,
    float eps) {
    const int64_t row = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (row >= n) {
        return;
    }
    const int64_t base = row * (int64_t) channels;
    float mean = 0.0f;
    for (int c = 0; c < channels; ++c) {
        mean += x[base + c];
    }
    mean /= (float) channels;
    float var = 0.0f;
    for (int c = 0; c < channels; ++c) {
        const float diff = x[base + c] - mean;
        var += diff * diff;
    }
    const float inv_std = rsqrtf(var / (float) channels + eps);
    for (int c = 0; c < channels; ++c) {
        float v = (x[base + c] - mean) * inv_std;
        if (gamma != NULL) {
            v *= gamma[c];
        }
        if (beta != NULL) {
            v += beta[c];
        }
        y[base + c] = v;
    }
}

__global__ void trellis_sparse_c2s_gather_f32_kernel(
    const float * __restrict__ x,
    const int32_t * __restrict__ parent,
    const int32_t * __restrict__ subidx,
    float * __restrict__ y,
    int64_t n_out,
    int out_channels,
    int64_t total) {
    int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int c = (int) (linear % out_channels);
    const int64_t row = linear / out_channels;
    const int p = parent[row];
    const int s = subidx[row];
    y[row * (int64_t) out_channels + c] =
        x[((int64_t) p * 8 + s) * (int64_t) out_channels + c];
}

__global__ void trellis_sparse_c2s_skip_repeat_f32_kernel(
    const float * __restrict__ x,
    const int32_t * __restrict__ parent,
    const int32_t * __restrict__ subidx,
    float * __restrict__ y,
    int64_t n_out,
    int in_channels,
    int out_channels,
    int repeat,
    int64_t total) {
    int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int c = (int) (linear % out_channels);
    const int64_t row = linear / out_channels;
    const int p = parent[row];
    const int s = subidx[row];
    const int repeated_channel = s * out_channels + c;
    int ic = repeated_channel / repeat;
    if (ic >= in_channels) {
        ic = in_channels - 1;
    }
    y[row * (int64_t) out_channels + c] = x[(int64_t) p * in_channels + ic];
}

__global__ void trellis_sparse_c2s_count_kernel(
    const float * __restrict__ subdiv_logits,
    int32_t * __restrict__ counts,
    int64_t n) {
    const int64_t row = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (row >= n) {
        return;
    }
    int count = 0;
    const float * logits = subdiv_logits + row * 8;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        count += logits[s] > 0.0f ? 1 : 0;
    }
    counts[row] = count;
}

__global__ void trellis_sparse_c2s_fill_kernel(
    const int32_t * __restrict__ coords,
    const float * __restrict__ subdiv_logits,
    const int32_t * __restrict__ inclusive_counts,
    const int32_t * __restrict__ counts,
    int32_t * __restrict__ out_coords,
    int32_t * __restrict__ parent,
    int32_t * __restrict__ subidx,
    int64_t n) {
    const int64_t row = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (row >= n) {
        return;
    }
    const int count = counts[row];
    if (count <= 0) {
        return;
    }
    int32_t dst = inclusive_counts[row] - count;
    const int32_t * c = coords + row * 4;
    const float * logits = subdiv_logits + row * 8;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (logits[s] <= 0.0f) {
            continue;
        }
        out_coords[4 * (int64_t) dst + 0] = c[0];
        out_coords[4 * (int64_t) dst + 1] = c[1] * 2 + (s & 1);
        out_coords[4 * (int64_t) dst + 2] = c[2] * 2 + ((s >> 1) & 1);
        out_coords[4 * (int64_t) dst + 3] = c[3] * 2 + ((s >> 2) & 1);
        parent[dst] = (int32_t) row;
        subidx[dst] = (int32_t) s;
        ++dst;
    }
}

static trellis_status sparse_linear_raw_device_cublas(
    const float * x_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * y_dev,
    int64_t n,
    int in_channels,
    int out_channels) {
    if (x_dev == NULL || weight_dev == NULL || y_dev == NULL ||
        n <= 0 || in_channels <= 0 || out_channels <= 0 ||
        n > (int64_t) INT_MAX || in_channels > INT_MAX || out_channels > INT_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    cublasHandle_t handle = NULL;
    trellis_status status = trellis_cublas_handle(&handle);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_channels,
        (int) n,
        in_channels,
        &alpha,
        weight_dev,
        in_channels,
        x_dev,
        in_channels,
        &beta,
        y_dev,
        out_channels);
    status = cublas_status_to_trellis(cublas_status);
    if (status != TRELLIS_STATUS_OK || bias_dev == NULL) {
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "sparse_linear_device_cublas: cublasSgemm failed: %s n=%lld in=%d out=%d",
                cublas_status_name(cublas_status),
                (long long) n,
                in_channels,
                out_channels);
        }
        return status;
    }

    const int64_t total = n * (int64_t) out_channels;
    const int block = 256;
    const int grid = (int) ((total + block - 1) / block);
    trellis_linear_add_bias_f32_kernel<<<grid, block>>>(y_dev, bias_dev, out_channels, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TRELLIS_ERROR(
            "sparse_linear_device_cublas: add_bias launch failed: %s total=%lld out=%d",
            cudaGetErrorString(err),
            (long long) total,
            out_channels);
    }
    return cuda_status_to_trellis(err);
}

trellis_status sparse_linear_device_cublas(
    const float * x_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * y_dev,
    int64_t n,
    int in_channels,
    int out_channels) {
    return sparse_linear_raw_device_cublas(
        x_dev,
        weight_dev,
        bias_dev,
        y_dev,
        n,
        in_channels,
        out_channels);
}

trellis_status row_layer_norm_device(
    const float * x_dev,
    const float * gamma_dev,
    const float * beta_dev,
    float * y_dev,
    int64_t n,
    int channels,
    float eps) {
    if (x_dev == NULL || y_dev == NULL || n <= 0 || channels <= 0 || eps <= 0.0f) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int block = 128;
    const int grid = (int) ((n + block - 1) / block);
    trellis_row_layer_norm_f32_kernel<<<grid, block>>>(
        x_dev,
        gamma_dev,
        beta_dev,
        y_dev,
        n,
        channels,
        eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TRELLIS_ERROR(
            "row_layer_norm_device: launch failed: %s rows=%lld channels=%d grid=%d",
            cudaGetErrorString(err),
            (long long) n,
            channels,
            grid);
    }
    return cuda_status_to_trellis(err);
}

trellis_status sparse_c2s_gather_device(
    const float * x_dev,
    const int32_t * parent_dev,
    const int32_t * subidx_dev,
    float * y_dev,
    int64_t n_out,
    int out_channels) {
    if (x_dev == NULL || parent_dev == NULL || subidx_dev == NULL || y_dev == NULL ||
        n_out <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int64_t total = n_out * (int64_t) out_channels;
    const int block = 256;
    const int grid = (int) ((total + block - 1) / block);
    trellis_sparse_c2s_gather_f32_kernel<<<grid, block>>>(
        x_dev,
        parent_dev,
        subidx_dev,
        y_dev,
        n_out,
        out_channels,
        total);
    return cuda_status_to_trellis(cudaGetLastError());
}

trellis_status sparse_c2s_skip_repeat_device(
    const float * x_dev,
    const int32_t * parent_dev,
    const int32_t * subidx_dev,
    float * y_dev,
    int64_t n_out,
    int in_channels,
    int out_channels) {
    if (x_dev == NULL || parent_dev == NULL || subidx_dev == NULL || y_dev == NULL ||
        n_out <= 0 || in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if ((in_channels % 8) != 0 || (out_channels % (in_channels / 8)) != 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int repeat = out_channels / (in_channels / 8);
    const int64_t total = n_out * (int64_t) out_channels;
    const int block = 256;
    const int grid = (int) ((total + block - 1) / block);
    trellis_sparse_c2s_skip_repeat_f32_kernel<<<grid, block>>>(
        x_dev,
        parent_dev,
        subidx_dev,
        y_dev,
        n_out,
        in_channels,
        out_channels,
        repeat,
        total);
    return cuda_status_to_trellis(cudaGetLastError());
}

void sparse_c2s_map_device_free(sparse_c2s_map_device * map) {
    if (map == NULL) {
        return;
    }
    cudaFree(map->coords);
    cudaFree(map->parent);
    cudaFree(map->subidx);
    memset(map, 0, sizeof(*map));
}

trellis_status sparse_c2s_map_build_device(
    const int32_t * coords_dev,
    const float * subdiv_logits_dev,
    int64_t n,
    sparse_c2s_map_device * map) {
    if (coords_dev == NULL || subdiv_logits_dev == NULL || map == NULL ||
        n <= 0 || n > (int64_t) INT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(map, 0, sizeof(*map));

    int32_t * counts_dev = NULL;
    int32_t * inclusive_counts_dev = NULL;
    void * scan_tmp = NULL;
    size_t scan_tmp_bytes = 0;
    int32_t m32 = 0;
    trellis_status status = TRELLIS_STATUS_OK;
    cudaError_t err = cudaMalloc((void **) &counts_dev, (size_t) n * sizeof(int32_t));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    err = cudaMalloc((void **) &inclusive_counts_dev, (size_t) n * sizeof(int32_t));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    {
        const int block = 256;
        const int grid = (int) ((n + block - 1) / block);
        trellis_sparse_c2s_count_kernel<<<grid, block>>>(subdiv_logits_dev, counts_dev, n);
        status = cuda_status_to_trellis(cudaGetLastError());
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
    }

    err = cub::DeviceScan::InclusiveSum(
        scan_tmp,
        scan_tmp_bytes,
        counts_dev,
        inclusive_counts_dev,
        (int) n);
    if (err != cudaSuccess) {
        status = cuda_status_to_trellis(err);
        goto cleanup;
    }
    err = cudaMalloc(&scan_tmp, scan_tmp_bytes);
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    err = cub::DeviceScan::InclusiveSum(
        scan_tmp,
        scan_tmp_bytes,
        counts_dev,
        inclusive_counts_dev,
        (int) n);
    status = cuda_status_to_trellis(err);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }

    status = cuda_status_to_trellis(cudaMemcpy(
        &m32,
        inclusive_counts_dev + n - 1,
        sizeof(int32_t),
        cudaMemcpyDeviceToHost));
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    if (m32 <= 0) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    map->n = (int64_t) m32;
    err = cudaMalloc((void **) &map->coords, (size_t) map->n * 4u * sizeof(int32_t));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    err = cudaMalloc((void **) &map->parent, (size_t) map->n * sizeof(int32_t));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    err = cudaMalloc((void **) &map->subidx, (size_t) map->n * sizeof(int32_t));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    {
        const int block = 256;
        const int grid = (int) ((n + block - 1) / block);
        trellis_sparse_c2s_fill_kernel<<<grid, block>>>(
            coords_dev,
            subdiv_logits_dev,
            inclusive_counts_dev,
            counts_dev,
            map->coords,
            map->parent,
            map->subidx,
            n);
        status = cuda_status_to_trellis(cudaGetLastError());
    }

cleanup:
    cudaFree(scan_tmp);
    cudaFree(counts_dev);
    cudaFree(inclusive_counts_dev);
    if (status != TRELLIS_STATUS_OK) {
        sparse_c2s_map_device_free(map);
    }
    return status;
}

extern "C" trellis_status trellis_cuda_silu_f32(
    const float * x_dev,
    float * y_dev,
    size_t n) {
    if (!elementwise_args_valid(x_dev, y_dev, n)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int block = 256;
    const int grid = (int) (((int64_t) n + block - 1) / block);
    trellis_silu_f32_kernel<<<grid, block>>>(x_dev, y_dev, (int64_t) n);
    return cuda_status_to_trellis(cudaGetLastError());
}

extern "C" trellis_status trellis_cuda_add_f32(
    const float * a_dev,
    const float * b_dev,
    float * y_dev,
    size_t n) {
    if (!add_args_valid(a_dev, b_dev, y_dev, n)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int block = 256;
    const int grid = (int) (((int64_t) n + block - 1) / block);
    trellis_add_f32_kernel<<<grid, block>>>(a_dev, b_dev, y_dev, (int64_t) n);
    return cuda_status_to_trellis(cudaGetLastError());
}

void sparse_neighbor_map_free(sparse_neighbor_map_device * map) {
    if (map == NULL) {
        return;
    }
    cudaFree(map->indices);
    cudaFree(map->src_rows);
    cudaFree(map->dst_rows);
    cudaFree(map->offset_counts_dev);
    cudaFree(map->offset_starts_dev);
    free(map->offset_counts_host);
    free(map->offset_starts_host);
    memset(map, 0, sizeof(*map));
}

static int sparse_neighbor_map_matches(
    const sparse_neighbor_map_device * map,
    int64_t n,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w) {
    return map != NULL && map->indices != NULL && map->n == n &&
        map->kernel_d == kernel_d && map->kernel_h == kernel_h && map->kernel_w == kernel_w &&
        map->dilation_d == dilation_d && map->dilation_h == dilation_h && map->dilation_w == dilation_w;
}

static int sparse_neighbor_map_has_rulebook(const sparse_neighbor_map_device * map) {
    return map != NULL && map->src_rows != NULL && map->dst_rows != NULL &&
        map->offset_counts_host != NULL && map->offset_starts_host != NULL;
}

static trellis_status sparse_neighbor_rulebook_build_device(sparse_neighbor_map_device * map) {
    if (map == NULL || map->indices == NULL || map->n <= 0 || map->k_volume <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int k_volume = map->k_volume;
    const int64_t map_total = map->n * (int64_t) k_volume;
    if (map_total <= 0 || map_total > (int64_t) INT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    map->offset_counts_host = (int32_t *) malloc((size_t) k_volume * sizeof(int32_t));
    map->offset_starts_host = (int32_t *) malloc((size_t) k_volume * sizeof(int32_t));
    if (map->offset_counts_host == NULL || map->offset_starts_host == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_status status = TRELLIS_STATUS_OK;
    int32_t * counters_dev = NULL;
    int64_t total_pairs = 0;
    cudaError_t err = cudaMalloc((void **) &map->offset_counts_dev, (size_t) k_volume * sizeof(int32_t));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    err = cudaMalloc((void **) &map->offset_starts_dev, (size_t) k_volume * sizeof(int32_t));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    trellis_sparse_rulebook_count_kernel<<<k_volume, 256>>>(map->indices, map->offset_counts_dev, map->n);
    status = cuda_status_to_trellis(cudaGetLastError());
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    status = cuda_status_to_trellis(cudaMemcpy(
        map->offset_counts_host,
        map->offset_counts_dev,
        (size_t) k_volume * sizeof(int32_t),
        cudaMemcpyDeviceToHost));
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }

    for (int k = 0; k < k_volume; ++k) {
        map->offset_starts_host[k] = (int32_t) total_pairs;
        total_pairs += map->offset_counts_host[k];
        if (total_pairs > (int64_t) INT32_MAX) {
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
    }
    map->total_pairs = total_pairs;
    status = cuda_status_to_trellis(cudaMemcpy(
        map->offset_starts_dev,
        map->offset_starts_host,
        (size_t) k_volume * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    if (total_pairs <= 0) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }

    err = cudaMalloc((void **) &map->src_rows, (size_t) total_pairs * sizeof(int32_t));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    err = cudaMalloc((void **) &map->dst_rows, (size_t) total_pairs * sizeof(int32_t));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    err = cudaMalloc((void **) &counters_dev, (size_t) k_volume * sizeof(int32_t));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    status = cuda_status_to_trellis(cudaMemcpy(
        counters_dev,
        map->offset_starts_host,
        (size_t) k_volume * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }

    {
        const int block = 256;
        const int grid = (int) ((map_total + block - 1) / block);
        trellis_sparse_rulebook_fill_kernel<<<grid, block>>>(
            map->indices,
            counters_dev,
            map->src_rows,
            map->dst_rows,
            map->n,
            map_total);
        status = cuda_status_to_trellis(cudaGetLastError());
    }

cleanup:
    cudaFree(counters_dev);
    if (status != TRELLIS_STATUS_OK) {
        cudaFree(map->src_rows);
        cudaFree(map->dst_rows);
        cudaFree(map->offset_counts_dev);
        cudaFree(map->offset_starts_dev);
        free(map->offset_counts_host);
        free(map->offset_starts_host);
        map->src_rows = NULL;
        map->dst_rows = NULL;
        map->offset_counts_dev = NULL;
        map->offset_starts_dev = NULL;
        map->offset_counts_host = NULL;
        map->offset_starts_host = NULL;
        map->total_pairs = 0;
    }
    return status;
}

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
    sparse_neighbor_map_device * map) {
    if (coords_dev == NULL || map == NULL || n <= 0 ||
        kernel_d <= 0 || kernel_h <= 0 || kernel_w <= 0 ||
        dilation_d <= 0 || dilation_h <= 0 || dilation_w <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(map, 0, sizeof(*map));
    const int k_volume = kernel_d * kernel_h * kernel_w;
    if (k_volume <= 0 || (uint64_t) n > (uint64_t) SIZE_MAX / (uint64_t) k_volume / sizeof(int32_t)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    const int64_t table_size = next_power_of_two_i64(n * 4);
    if (table_size <= n || table_size > (int64_t) INT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int64_t table_mask = table_size - 1;
    const int64_t map_total = n * (int64_t) k_volume;
    const int block = 256;

    unsigned long long * keys_dev = NULL;
    int32_t * values_dev = NULL;
    trellis_status status = TRELLIS_STATUS_OK;

    cudaError_t err = cudaMalloc((void **) &keys_dev, (size_t) table_size * sizeof(unsigned long long));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    err = cudaMalloc((void **) &values_dev, (size_t) table_size * sizeof(int32_t));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    err = cudaMalloc((void **) &map->indices, (size_t) map_total * sizeof(int32_t));
    if (err != cudaSuccess) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    status = cuda_status_to_trellis(cudaMemset(keys_dev, 0, (size_t) table_size * sizeof(unsigned long long)));
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    {
        const int grid_insert = (int) ((n + block - 1) / block);
        trellis_sparse_hash_insert_kernel<<<grid_insert, block>>>(
            coords_dev,
            keys_dev,
            values_dev,
            n,
            table_mask);
        status = cuda_status_to_trellis(cudaGetLastError());
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
    }
    {
        const int grid_map = (int) ((map_total + block - 1) / block);
        trellis_sparse_build_neighbor_map_kernel<<<grid_map, block>>>(
            coords_dev,
            keys_dev,
            values_dev,
            map->indices,
            n,
            kernel_d,
            kernel_h,
            kernel_w,
            dilation_d,
            dilation_h,
            dilation_w,
            table_mask,
            map_total);
        status = cuda_status_to_trellis(cudaGetLastError());
    }
    if (status == TRELLIS_STATUS_OK) {
        map->n = n;
        map->k_volume = k_volume;
        map->kernel_d = kernel_d;
        map->kernel_h = kernel_h;
        map->kernel_w = kernel_w;
        map->dilation_d = dilation_d;
        map->dilation_h = dilation_h;
        map->dilation_w = dilation_w;
        if (build_rulebook) {
            status = sparse_neighbor_rulebook_build_device(map);
        }
    }

cleanup:
    cudaFree(keys_dev);
    cudaFree(values_dev);
    if (status != TRELLIS_STATUS_OK) {
        sparse_neighbor_map_free(map);
    }
    return status;
}

static trellis_status sparse_subm_conv3d_with_map_f32(
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
    int dilation_w) {
    if (!sparse_neighbor_map_matches(map, n, kernel_d, kernel_h, kernel_w, dilation_d, dilation_h, dilation_w) ||
        !sparse_neighbor_map_has_rulebook(map) ||
        feats_dev == NULL || weight_dev == NULL || out_dev == NULL || in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int64_t total = n * (int64_t) out_channels;
    const int block = 256;
    const int grid = (int) ((total + block - 1) / block);
    trellis_sparse_conv_fill_bias_f32_kernel<<<grid, block>>>(out_dev, bias_dev, total, out_channels);
    cudaError_t fill_err = cudaGetLastError();
    trellis_status status = cuda_status_to_trellis(fill_err);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "sparse_subm_conv3d: fill_bias launch failed: %s n=%lld out=%d total=%lld grid=%d",
            cudaGetErrorString(fill_err),
            (long long) n,
            out_channels,
            (long long) total,
            grid);
        return status;
    }

    const dim3 gemm_block(16, 16);
    for (int k = 0; status == TRELLIS_STATUS_OK && k < map->k_volume; ++k) {
        const int64_t pair_count = map->offset_counts_host[k];
        if (pair_count <= 0) {
            continue;
        }
        const int64_t start = map->offset_starts_host[k];
        const uint64_t pair_tiles = (uint64_t) ((pair_count + 15) / 16);
        const unsigned int grid_y = pair_tiles > 65535ull ? 65535u : (unsigned int) pair_tiles;
        const unsigned int grid_z = (unsigned int) ((pair_tiles + (uint64_t) grid_y - 1ull) / (uint64_t) grid_y);
        if (grid_y == 0 || grid_z == 0 || grid_z > 65535u) {
            TRELLIS_ERROR(
                "sparse_subm_conv3d: implicit_gemm grid too large k=%d pairs=%lld tiles=%llu grid_y=%u grid_z=%u",
                k,
                (long long) pair_count,
                (unsigned long long) pair_tiles,
                grid_y,
                grid_z);
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
            break;
        }
        const dim3 gemm_grid(
            (unsigned int) ((out_channels + 15) / 16),
            grid_y,
            grid_z);
        trellis_sparse_subm_conv3d_implicit_gemm_f32_kernel<<<gemm_grid, gemm_block>>>(
            feats_dev,
            weight_dev,
            map->src_rows + start,
            map->dst_rows + start,
            out_dev,
            pair_count,
            in_channels,
            out_channels,
            map->k_volume,
            k);
        cudaError_t err = cudaGetLastError();
        status = cuda_status_to_trellis(err);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "sparse_subm_conv3d: implicit_gemm launch failed: %s k=%d pairs=%lld in=%d out=%d grid=(%u,%u,%u)",
                cudaGetErrorString(err),
                k,
                (long long) pair_count,
                in_channels,
                out_channels,
                gemm_grid.x,
                gemm_grid.y,
                gemm_grid.z);
        }
    }
    return status;
}

void sparse_subm_conv3d_device_free(sparse_subm_conv3d_device * conv) {
    if (conv == NULL) {
        return;
    }
    sparse_neighbor_map_free(&conv->neighbor_map);
}

trellis_status sparse_subm_conv3d_device_build(
    const int32_t * coords_dev,
    int64_t n,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    sparse_subm_conv3d_device * conv) {
    if (conv == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(conv, 0, sizeof(*conv));
    return sparse_neighbor_map_build_device(
        coords_dev,
        n,
        kernel_d,
        kernel_h,
        kernel_w,
        dilation_d,
        dilation_h,
        dilation_w,
        1,
        &conv->neighbor_map);
}

trellis_status sparse_subm_conv3d_device_forward_f32(
    const sparse_subm_conv3d_device * conv,
    const float * feats_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * out_dev,
    int64_t n,
    int in_channels,
    int out_channels) {
    if (conv == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const sparse_neighbor_map_device * map = &conv->neighbor_map;
    return sparse_subm_conv3d_with_map_f32(
        map,
        feats_dev,
        weight_dev,
        bias_dev,
        out_dev,
        n,
        in_channels,
        out_channels,
        map->kernel_d,
        map->kernel_h,
        map->kernel_w,
        map->dilation_d,
        map->dilation_h,
        map->dilation_w);
}

extern "C" trellis_status trellis_cuda_sparse_subm_conv3d_f32(
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
    int dilation_w) {
    if (!sparse_subm_conv3d_args_valid(
            coords_dev, feats_dev, weight_dev, out_dev, n, in_channels, out_channels,
            kernel_d, kernel_h, kernel_w, dilation_d, dilation_h, dilation_w)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    sparse_neighbor_map_device map;
    memset(&map, 0, sizeof(map));
    trellis_status status = sparse_neighbor_map_build_device(
        coords_dev,
        n,
        kernel_d,
        kernel_h,
        kernel_w,
        dilation_d,
        dilation_h,
        dilation_w,
        1,
        &map);
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_subm_conv3d_with_map_f32(
            &map,
            feats_dev,
            weight_dev,
            bias_dev,
            out_dev,
            n,
            in_channels,
            out_channels,
            kernel_d,
            kernel_h,
            kernel_w,
            dilation_d,
            dilation_h,
            dilation_w);
    }
    sparse_neighbor_map_free(&map);
    return status;
}

extern "C" trellis_status trellis_cuda_silu_f32_host(
    const float * x,
    float * y,
    int device,
    size_t n) {
    if (!elementwise_args_valid(x, y, n) || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }
    float * x_dev = NULL;
    float * y_dev = NULL;
    trellis_status status = malloc_copy_to_device(x, n, &x_dev);
    if (status == TRELLIS_STATUS_OK) {
        err = cudaMalloc((void **) &y_dev, n * sizeof(float));
        status = err == cudaSuccess ? TRELLIS_STATUS_OK : TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_silu_f32(x_dev, y_dev, n);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaMemcpy(y, y_dev, n * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaFree(x_dev);
    cudaFree(y_dev);
    return status;
}

extern "C" trellis_status trellis_cuda_add_f32_host(
    const float * a,
    const float * b,
    float * y,
    int device,
    size_t n) {
    if (!add_args_valid(a, b, y, n) || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }
    float * a_dev = NULL;
    float * b_dev = NULL;
    float * y_dev = NULL;
    trellis_status status = malloc_copy_to_device(a, n, &a_dev);
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(b, n, &b_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        err = cudaMalloc((void **) &y_dev, n * sizeof(float));
        status = err == cudaSuccess ? TRELLIS_STATUS_OK : TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_add_f32(a_dev, b_dev, y_dev, n);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaMemcpy(y, y_dev, n * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(y_dev);
    return status;
}

extern "C" trellis_status trellis_cuda_sparse_linear_f32_host(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int device,
    int64_t n,
    int in_channels,
    int out_channels) {
    if (x == NULL || weight == NULL || y == NULL || device < 0 ||
        n <= 0 || in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if ((uint64_t) n > (uint64_t) SIZE_MAX / (uint64_t) in_channels / sizeof(float) ||
        (uint64_t) n > (uint64_t) SIZE_MAX / (uint64_t) out_channels / sizeof(float)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    const size_t input_count = (size_t) n * (size_t) in_channels;
    const size_t output_count = (size_t) n * (size_t) out_channels;
    const size_t weight_count = (size_t) out_channels * (size_t) in_channels;
    trellis_status status = TRELLIS_STATUS_OK;
    float * x_dev = NULL;
    float * weight_dev = NULL;
    float * bias_dev = NULL;
    float * y_dev = NULL;
    status = malloc_copy_to_device(x, input_count, &x_dev);
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(weight, weight_count, &weight_dev);
    }
    if (status == TRELLIS_STATUS_OK && bias != NULL) {
        status = malloc_copy_to_device(bias, (size_t) out_channels, &bias_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_malloc_f32(output_count, &y_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_linear_raw_device_cublas(
            x_dev,
            weight_dev,
            bias_dev,
            y_dev,
            n,
            in_channels,
            out_channels);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaMemcpy(y, y_dev, output_count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaFree(x_dev);
    cudaFree(weight_dev);
    cudaFree(bias_dev);
    cudaFree(y_dev);
    return status;
}

extern "C" trellis_status trellis_cuda_sparse_subm_conv3d_f32_host(
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
    int dilation_w) {
    if (!sparse_subm_conv3d_args_valid(
            coords, feats, weight, out, n, in_channels, out_channels,
            kernel_d, kernel_h, kernel_w, dilation_d, dilation_h, dilation_w) || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    int32_t * coords_dev = NULL;
    float * feats_dev = NULL;
    float * weight_dev = NULL;
    float * bias_dev = NULL;
    float * out_dev = NULL;
    const size_t coords_count = (size_t) n * 4u;
    const size_t feats_count = (size_t) n * (size_t) in_channels;
    const size_t weight_count =
        (size_t) out_channels * (size_t) kernel_d * (size_t) kernel_h * (size_t) kernel_w * (size_t) in_channels;
    const size_t out_count = (size_t) n * (size_t) out_channels;

    trellis_status status = malloc_copy_i32_to_device(coords, coords_count, &coords_dev);
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(feats, feats_count, &feats_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(weight, weight_count, &weight_dev);
    }
    if (status == TRELLIS_STATUS_OK && bias != NULL) {
        status = malloc_copy_to_device(bias, (size_t) out_channels, &bias_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_malloc_f32(out_count, &out_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_sparse_subm_conv3d_f32(
            coords_dev,
            feats_dev,
            weight_dev,
            bias_dev,
            out_dev,
            n,
            in_channels,
            out_channels,
            kernel_d,
            kernel_h,
            kernel_w,
            dilation_d,
            dilation_h,
            dilation_w);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaMemcpy(out, out_dev, out_count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaFree(coords_dev);
    cudaFree(feats_dev);
    cudaFree(weight_dev);
    cudaFree(bias_dev);
    cudaFree(out_dev);
    return status;
}
