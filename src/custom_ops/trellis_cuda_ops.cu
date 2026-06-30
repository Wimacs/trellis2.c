#include "trellis_cuda_ops.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static trellis_status cuda_status_to_trellis(cudaError_t err) {
    return err == cudaSuccess ? TRELLIS_STATUS_OK : TRELLIS_STATUS_ERROR;
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

static int conv_out_size(int in, int kernel, int stride, int pad, int dilation) {
    return (in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

static bool conv3d_args_valid(
    const float * x,
    const float * weight,
    float * y,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w) {
    return x != NULL && weight != NULL && y != NULL &&
        batch > 0 && in_channels > 0 && in_d > 0 && in_h > 0 && in_w > 0 &&
        out_channels > 0 && kernel_d > 0 && kernel_h > 0 && kernel_w > 0 &&
        stride_d > 0 && stride_h > 0 && stride_w > 0 &&
        pad_d >= 0 && pad_h >= 0 && pad_w >= 0 &&
        dilation_d > 0 && dilation_h > 0 && dilation_w > 0;
}

static bool pixel_shuffle_args_valid(
    const float * x,
    float * y,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int scale) {
    const int cube = scale * scale * scale;
    return x != NULL && y != NULL && batch > 0 && in_channels > 0 &&
        in_d > 0 && in_h > 0 && in_w > 0 && scale > 0 && in_channels % cube == 0;
}

static bool rope_args_valid(
    const float * x,
    const float * cos_phase,
    const float * sin_phase,
    float * y,
    int batch,
    int tokens,
    int heads,
    int head_dim) {
    return x != NULL && cos_phase != NULL && sin_phase != NULL && y != NULL &&
        batch > 0 && tokens > 0 && heads > 0 && head_dim > 0 && (head_dim & 1) == 0;
}

static bool channel_layer_norm_3d_args_valid(
    const float * x,
    float * y,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    float eps) {
    return x != NULL && y != NULL && batch > 0 && channels > 0 &&
        depth > 0 && height > 0 && width > 0 && eps > 0.0f;
}

static bool elementwise_args_valid(const float * x, float * y, size_t n) {
    return x != NULL && y != NULL && n > 0;
}

static bool add_args_valid(const float * a, const float * b, float * y, size_t n) {
    return a != NULL && b != NULL && y != NULL && n > 0;
}

static bool dino_patch_embed_args_valid(
    const float * image,
    const float * weight,
    float * tokens,
    int batch,
    int image_h,
    int image_w,
    int out_channels,
    int patch_size) {
    return image != NULL && weight != NULL && tokens != NULL &&
        batch > 0 && image_h > 0 && image_w > 0 && out_channels > 0 && patch_size > 0 &&
        image_h % patch_size == 0 && image_w % patch_size == 0;
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

__global__ void trellis_sparse_subm_conv3d_f32_kernel(
    const int32_t * __restrict__ coords,
    const float * __restrict__ feats,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    const unsigned long long * __restrict__ keys,
    const int32_t * __restrict__ values,
    float * __restrict__ out,
    int64_t n,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int64_t table_mask,
    int64_t total) {
    int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int oc = (int) (linear % out_channels);
    const int64_t row = linear / out_channels;
    const int32_t * center = coords + 4 * row;
    const int center_d = kernel_d / 2;
    const int center_h = kernel_h / 2;
    const int center_w = kernel_w / 2;
    float acc = bias == NULL ? 0.0f : bias[oc];

    for (int kd = 0; kd < kernel_d; ++kd) {
        const int nx = center[1] + (kd - center_d) * dilation_d;
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int ny = center[2] + (kh - center_h) * dilation_h;
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int nz = center[3] + (kw - center_w) * dilation_w;
                const unsigned long long key = trellis_pack_coord4(center[0], nx, ny, nz);
                const int in_row = trellis_sparse_hash_find(keys, values, table_mask, key);
                if (in_row < 0) {
                    continue;
                }
                const int64_t w_base =
                    ((((int64_t) oc * kernel_d + kd) * kernel_h + kh) * kernel_w + kw) * (int64_t) in_channels;
                const int64_t f_base = (int64_t) in_row * in_channels;
                for (int ic = 0; ic < in_channels; ++ic) {
                    acc += feats[f_base + ic] * weight[w_base + ic];
                }
            }
        }
    }
    out[row * (int64_t) out_channels + oc] = acc;
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

__global__ void trellis_sparse_subm_conv3d_f32_map_kernel(
    const float * __restrict__ feats,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    const int32_t * __restrict__ neighbor_map,
    float * __restrict__ out,
    int64_t n,
    int in_channels,
    int out_channels,
    int k_volume,
    int64_t total) {
    const int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int oc = (int) (linear % out_channels);
    const int64_t row = linear / out_channels;
    float acc = bias == NULL ? 0.0f : bias[oc];

    for (int k = 0; k < k_volume; ++k) {
        const int in_row = neighbor_map[(int64_t) k * n + row];
        if (in_row < 0) {
            continue;
        }
        const int64_t f_base = (int64_t) in_row * in_channels;
        const int64_t w_base = ((int64_t) oc * k_volume + k) * (int64_t) in_channels;
        for (int ic = 0; ic < in_channels; ++ic) {
            acc += feats[f_base + ic] * weight[w_base + ic];
        }
    }
    out[row * (int64_t) out_channels + oc] = acc;
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

__global__ void trellis_sparse_gather_pairs_f32_kernel(
    const float * __restrict__ feats,
    const int32_t * __restrict__ src_rows,
    float * __restrict__ gathered,
    int64_t pair_count,
    int channels,
    int64_t total) {
    const int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int c = (int) (linear % channels);
    const int64_t pair = linear / channels;
    const int32_t src = src_rows[pair];
    gathered[linear] = feats[(int64_t) src * channels + c];
}

__global__ void trellis_sparse_pack_all_weights_f32_kernel(
    const float * __restrict__ weight,
    float * __restrict__ packed,
    int in_channels,
    int out_channels,
    int k_volume,
    int64_t total) {
    const int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int ic = (int) (linear % in_channels);
    const int oc = (int) ((linear / in_channels) % out_channels);
    const int k = (int) (linear / ((int64_t) in_channels * out_channels));
    packed[linear] = weight[((int64_t) oc * k_volume + k) * (int64_t) in_channels + ic];
}

__global__ void trellis_sparse_scatter_add_pairs_f32_kernel(
    const float * __restrict__ partial,
    const int32_t * __restrict__ dst_rows,
    float * __restrict__ out,
    int64_t pair_count,
    int out_channels,
    int64_t total) {
    const int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int oc = (int) (linear % out_channels);
    const int64_t pair = linear / out_channels;
    const int32_t dst = dst_rows[pair];
    out[(int64_t) dst * out_channels + oc] += partial[linear];
}

__global__ void trellis_conv3d_f32_kernel(
    const float * __restrict__ x,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float * __restrict__ y,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int out_d,
    int out_h,
    int out_w,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int64_t total) {
    int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }

    const int ow = (int) (linear % out_w);
    linear /= out_w;
    const int oh = (int) (linear % out_h);
    linear /= out_h;
    const int od = (int) (linear % out_d);
    linear /= out_d;
    const int oc = (int) (linear % out_channels);
    linear /= out_channels;
    const int b = (int) linear;

    float acc = bias == NULL ? 0.0f : bias[oc];
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            const int id = od * stride_d + kd * dilation_d - pad_d;
            if ((unsigned) id >= (unsigned) in_d) {
                continue;
            }
            for (int kh = 0; kh < kernel_h; ++kh) {
                const int ih = oh * stride_h + kh * dilation_h - pad_h;
                if ((unsigned) ih >= (unsigned) in_h) {
                    continue;
                }
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int iw = ow * stride_w + kw * dilation_w - pad_w;
                    if ((unsigned) iw >= (unsigned) in_w) {
                        continue;
                    }
                    const int64_t x_idx =
                        (((int64_t) b * in_channels + ic) * in_d + id) * in_h * (int64_t) in_w +
                        (int64_t) ih * in_w + iw;
                    const int64_t w_idx =
                        (((int64_t) oc * in_channels + ic) * kernel_d + kd) * kernel_h * (int64_t) kernel_w +
                        (int64_t) kh * kernel_w + kw;
                    acc += x[x_idx] * weight[w_idx];
                }
            }
        }
    }

    const int64_t y_idx =
        (((int64_t) b * out_channels + oc) * out_d + od) * out_h * (int64_t) out_w +
        (int64_t) oh * out_w + ow;
    y[y_idx] = acc;
}

__global__ void trellis_pixel_shuffle_3d_f32_kernel(
    const float * __restrict__ x,
    float * __restrict__ y,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int out_d,
    int out_h,
    int out_w,
    int scale,
    int64_t total) {
    int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }

    const int ow = (int) (linear % out_w);
    linear /= out_w;
    const int oh = (int) (linear % out_h);
    linear /= out_h;
    const int od = (int) (linear % out_d);
    linear /= out_d;
    const int oc = (int) (linear % out_channels);
    linear /= out_channels;
    const int b = (int) linear;

    const int id = od / scale;
    const int ih = oh / scale;
    const int iw = ow / scale;
    const int rd = od - id * scale;
    const int rh = oh - ih * scale;
    const int rw = ow - iw * scale;
    const int ic = (((oc * scale) + rd) * scale + rh) * scale + rw;

    const int64_t x_idx =
        (((int64_t) b * in_channels + ic) * in_d + id) * in_h * (int64_t) in_w +
        (int64_t) ih * in_w + iw;
    const int64_t y_idx =
        (((int64_t) b * out_channels + oc) * out_d + od) * out_h * (int64_t) out_w +
        (int64_t) oh * out_w + ow;
    y[y_idx] = x[x_idx];
}

__global__ void trellis_apply_rope_f32_kernel(
    const float * __restrict__ x,
    const float * __restrict__ cos_phase,
    const float * __restrict__ sin_phase,
    float * __restrict__ y,
    int batch,
    int tokens,
    int heads,
    int head_dim,
    int half_dim,
    int64_t total_pairs) {
    int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total_pairs) {
        return;
    }

    const int pair = (int) (linear % half_dim);
    linear /= half_dim;
    const int token = (int) (linear % tokens);
    linear /= tokens;
    const int head = (int) (linear % heads);
    linear /= heads;
    const int b = (int) linear;

    const int d0 = pair * 2;
    const int d1 = d0 + 1;
    const int64_t base = (int64_t) token * head_dim + (int64_t) head * tokens * head_dim +
        (int64_t) b * heads * tokens * head_dim;
    const int64_t idx0 = base + d0;
    const int64_t idx1 = base + d1;
    const float c = cos_phase[(int64_t) token * half_dim + pair];
    const float s = sin_phase[(int64_t) token * half_dim + pair];
    const float x0 = x[idx0];
    const float x1 = x[idx1];
    y[idx0] = x0 * c - x1 * s;
    y[idx1] = x0 * s + x1 * c;
}

__global__ void trellis_channel_layer_norm_3d_f32_kernel(
    const float * __restrict__ x,
    const float * __restrict__ gamma,
    const float * __restrict__ beta,
    float * __restrict__ y,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    float eps,
    int64_t total_voxels) {
    int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total_voxels) {
        return;
    }

    const int w = (int) (linear % width);
    linear /= width;
    const int h = (int) (linear % height);
    linear /= height;
    const int d = (int) (linear % depth);
    linear /= depth;
    const int b = (int) linear;

    float mean = 0.0f;
    for (int c = 0; c < channels; ++c) {
        const int64_t idx =
            (((int64_t) b * channels + c) * depth + d) * height * (int64_t) width +
            (int64_t) h * width + w;
        mean += x[idx];
    }
    mean /= (float) channels;

    float var = 0.0f;
    for (int c = 0; c < channels; ++c) {
        const int64_t idx =
            (((int64_t) b * channels + c) * depth + d) * height * (int64_t) width +
            (int64_t) h * width + w;
        const float diff = x[idx] - mean;
        var += diff * diff;
    }
    const float inv_std = rsqrtf(var / (float) channels + eps);

    for (int c = 0; c < channels; ++c) {
        const int64_t idx =
            (((int64_t) b * channels + c) * depth + d) * height * (int64_t) width +
            (int64_t) h * width + w;
        float v = (x[idx] - mean) * inv_std;
        if (gamma != NULL) {
            v *= gamma[c];
        }
        if (beta != NULL) {
            v += beta[c];
        }
        y[idx] = v;
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

__global__ void trellis_dino_patch_embed_f32_kernel(
    const float * __restrict__ image,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float * __restrict__ tokens,
    int batch,
    int image_h,
    int image_w,
    int out_channels,
    int patch_size,
    int patches_h,
    int patches_w,
    int n_patches,
    int64_t total) {
    int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int oc = (int) (linear % out_channels);
    linear /= out_channels;
    const int token = (int) (linear % n_patches);
    linear /= n_patches;
    const int b = (int) linear;
    const int py = token / patches_w;
    const int px = token - py * patches_w;

    float acc = bias == NULL ? 0.0f : bias[oc];
    for (int ic = 0; ic < 3; ++ic) {
        for (int ky = 0; ky < patch_size; ++ky) {
            const int iy = py * patch_size + ky;
            for (int kx = 0; kx < patch_size; ++kx) {
                const int ix = px * patch_size + kx;
                const int64_t image_idx =
                    (((int64_t) b * 3 + ic) * image_h + iy) * (int64_t) image_w + ix;
                const int64_t weight_idx =
                    (((int64_t) oc * 3 + ic) * patch_size + ky) * (int64_t) patch_size + kx;
                acc += image[image_idx] * weight[weight_idx];
            }
        }
    }
    tokens[(int64_t) b * n_patches * out_channels + (int64_t) token * out_channels + oc] = acc;
}

__global__ void trellis_sparse_linear_f32_kernel(
    const float * __restrict__ x,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float * __restrict__ y,
    int64_t n,
    int in_channels,
    int out_channels,
    int64_t total) {
    int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int oc = (int) (linear % out_channels);
    const int64_t row = linear / out_channels;
    float acc = bias == NULL ? 0.0f : bias[oc];
    const int64_t x_base = row * (int64_t) in_channels;
    const int64_t w_base = (int64_t) oc * in_channels;
    for (int ic = 0; ic < in_channels; ++ic) {
        acc += x[x_base + ic] * weight[w_base + ic];
    }
    y[row * (int64_t) out_channels + oc] = acc;
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

static int sparse_linear_force_scalar(void) {
    const char * backend = getenv("TRELLIS_SPARSE_LINEAR_BACKEND");
    return backend != NULL && strcmp(backend, "scalar") == 0;
}

trellis_status sparse_linear_device_scalar(
    const float * x_dev,
    const struct ggml_tensor * w,
    const struct ggml_tensor * b,
    float * y_dev,
    int64_t n,
    int in_channels,
    int out_channels) {
    if (x_dev == NULL || w == NULL || y_dev == NULL || n <= 0 || in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int64_t total = n * (int64_t) out_channels;
    const int block = 256;
    const int grid = (int) ((total + block - 1) / block);
    trellis_sparse_linear_f32_kernel<<<grid, block>>>(
        x_dev,
        (const float *) w->data,
        b == NULL ? NULL : (const float *) b->data,
        y_dev,
        n,
        in_channels,
        out_channels,
        total);
    return cuda_status_to_trellis(cudaGetLastError());
}

trellis_status sparse_linear_device(
    ggml_backend_t backend,
    const float * x_dev,
    const struct ggml_tensor * w,
    const struct ggml_tensor * b,
    float * y_dev,
    int64_t n,
    int in_channels,
    int out_channels) {
    if (x_dev == NULL || w == NULL || y_dev == NULL || n <= 0 || in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (w->data == NULL || w->ne[0] != in_channels || w->ne[1] != out_channels ||
        (b != NULL && (b->data == NULL || b->ne[0] != out_channels))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (backend == NULL || sparse_linear_force_scalar()) {
        return sparse_linear_device_scalar(x_dev, w, b, y_dev, n, in_channels, out_channels);
    }
    if ((uint64_t) n > (uint64_t) SIZE_MAX / (uint64_t) in_channels / sizeof(float) ||
        (uint64_t) n > (uint64_t) SIZE_MAX / (uint64_t) out_channels / sizeof(float)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    const size_t input_bytes = (size_t) n * (size_t) in_channels * sizeof(float);
    const size_t output_bytes = (size_t) n * (size_t) out_channels * sizeof(float);
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false) + 4096,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_status status = TRELLIS_STATUS_OK;
    ggml_gallocr_t alloc = NULL;
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_channels, n);
    struct ggml_tensor * y = ggml_mul_mat(ctx, (struct ggml_tensor *) w, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    if (b != NULL) {
        y = ggml_add(ctx, y, ggml_repeat(ctx, (struct ggml_tensor *) b, y));
    }
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    if (x == NULL || y == NULL || graph == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    ggml_build_forward_expand(graph, y);
    alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (alloc == NULL || !ggml_gallocr_alloc_graph(alloc, graph)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    status = cuda_status_to_trellis(cudaDeviceSynchronize());
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    status = cuda_status_to_trellis(cudaMemcpy(x->data, x_dev, input_bytes, cudaMemcpyDeviceToDevice));
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    status = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS ?
        TRELLIS_STATUS_OK : TRELLIS_STATUS_ERROR;
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    ggml_backend_synchronize(backend);
    status = cuda_status_to_trellis(cudaMemcpy(y_dev, y->data, output_bytes, cudaMemcpyDeviceToDevice));

cleanup:
    if (alloc != NULL) {
        ggml_gallocr_free(alloc);
    }
    ggml_free(ctx);
    return status;
}

trellis_status row_layer_norm_device(
    const float * x_dev,
    const struct ggml_tensor * gamma,
    const struct ggml_tensor * beta,
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
        gamma == NULL ? NULL : (const float *) gamma->data,
        beta == NULL ? NULL : (const float *) beta->data,
        y_dev,
        n,
        channels,
        eps);
    return cuda_status_to_trellis(cudaGetLastError());
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

extern "C" trellis_status trellis_cuda_conv3d_f32(
    const float * x_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * y_dev,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w) {
    if (!conv3d_args_valid(x_dev, weight_dev, y_dev, batch, in_channels, in_d, in_h, in_w,
            out_channels, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int out_d = conv_out_size(in_d, kernel_d, stride_d, pad_d, dilation_d);
    const int out_h = conv_out_size(in_h, kernel_h, stride_h, pad_h, dilation_h);
    const int out_w = conv_out_size(in_w, kernel_w, stride_w, pad_w, dilation_w);
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int64_t total = (int64_t) batch * out_channels * out_d * out_h * out_w;
    const int block = 256;
    const int grid = (int) ((total + block - 1) / block);
    trellis_conv3d_f32_kernel<<<grid, block>>>(
        x_dev,
        weight_dev,
        bias_dev,
        y_dev,
        batch,
        in_channels,
        in_d,
        in_h,
        in_w,
        out_channels,
        out_d,
        out_h,
        out_w,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation_d,
        dilation_h,
        dilation_w,
        total);
    return cuda_status_to_trellis(cudaGetLastError());
}

extern "C" trellis_status trellis_cuda_pixel_shuffle_3d_f32(
    const float * x_dev,
    float * y_dev,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int scale) {
    if (!pixel_shuffle_args_valid(x_dev, y_dev, batch, in_channels, in_d, in_h, in_w, scale)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int out_channels = in_channels / (scale * scale * scale);
    const int out_d = in_d * scale;
    const int out_h = in_h * scale;
    const int out_w = in_w * scale;
    const int64_t total = (int64_t) batch * out_channels * out_d * out_h * out_w;
    const int block = 256;
    const int grid = (int) ((total + block - 1) / block);
    trellis_pixel_shuffle_3d_f32_kernel<<<grid, block>>>(
        x_dev,
        y_dev,
        batch,
        in_channels,
        in_d,
        in_h,
        in_w,
        out_channels,
        out_d,
        out_h,
        out_w,
        scale,
        total);
    return cuda_status_to_trellis(cudaGetLastError());
}

extern "C" trellis_status trellis_cuda_apply_rope_f32(
    const float * x_dev,
    const float * cos_dev,
    const float * sin_dev,
    float * y_dev,
    int batch,
    int tokens,
    int heads,
    int head_dim) {
    if (!rope_args_valid(x_dev, cos_dev, sin_dev, y_dev, batch, tokens, heads, head_dim)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int half_dim = head_dim / 2;
    const int64_t total_pairs = (int64_t) batch * heads * tokens * half_dim;
    const int block = 256;
    const int grid = (int) ((total_pairs + block - 1) / block);
    trellis_apply_rope_f32_kernel<<<grid, block>>>(
        x_dev,
        cos_dev,
        sin_dev,
        y_dev,
        batch,
        tokens,
        heads,
        head_dim,
        half_dim,
        total_pairs);
    return cuda_status_to_trellis(cudaGetLastError());
}

extern "C" trellis_status trellis_cuda_channel_layer_norm_3d_f32(
    const float * x_dev,
    const float * gamma_dev,
    const float * beta_dev,
    float * y_dev,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    float eps) {
    if (!channel_layer_norm_3d_args_valid(x_dev, y_dev, batch, channels, depth, height, width, eps)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int64_t total_voxels = (int64_t) batch * depth * height * width;
    const int block = 128;
    const int grid = (int) ((total_voxels + block - 1) / block);
    trellis_channel_layer_norm_3d_f32_kernel<<<grid, block>>>(
        x_dev,
        gamma_dev,
        beta_dev,
        y_dev,
        batch,
        channels,
        depth,
        height,
        width,
        eps,
        total_voxels);
    return cuda_status_to_trellis(cudaGetLastError());
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

extern "C" trellis_status trellis_cuda_dino_patch_embed_f32(
    const float * image_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * tokens_dev,
    int batch,
    int image_h,
    int image_w,
    int out_channels,
    int patch_size) {
    if (!dino_patch_embed_args_valid(image_dev, weight_dev, tokens_dev, batch, image_h, image_w, out_channels, patch_size)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int patches_h = image_h / patch_size;
    const int patches_w = image_w / patch_size;
    const int n_patches = patches_h * patches_w;
    const int64_t total = (int64_t) batch * n_patches * out_channels;
    const int block = 256;
    const int grid = (int) ((total + block - 1) / block);
    trellis_dino_patch_embed_f32_kernel<<<grid, block>>>(
        image_dev,
        weight_dev,
        bias_dev,
        tokens_dev,
        batch,
        image_h,
        image_w,
        out_channels,
        patch_size,
        patches_h,
        patches_w,
        n_patches,
        total);
    return cuda_status_to_trellis(cudaGetLastError());
}

static int sparse_conv_force_scalar(void) {
    const char * backend = getenv("TRELLIS_SPARSE_CONV_BACKEND");
    return backend != NULL && strcmp(backend, "scalar") == 0;
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
        feats_dev == NULL || weight_dev == NULL || out_dev == NULL || in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int64_t total = n * (int64_t) out_channels;
    const int block = 256;
    const int grid = (int) ((total + block - 1) / block);
    trellis_sparse_subm_conv3d_f32_map_kernel<<<grid, block>>>(
        feats_dev,
        weight_dev,
        bias_dev,
        map->indices,
        out_dev,
        n,
        in_channels,
        out_channels,
        map->k_volume,
        total);
    return cuda_status_to_trellis(cudaGetLastError());
}

typedef struct sparse_conv_weight_pack_cache {
    const float * source_weight_dev;
    float * packed_dev;
    int in_channels;
    int out_channels;
    int k_volume;
    int device;
    struct sparse_conv_weight_pack_cache * next;
} sparse_conv_weight_pack_cache;

static sparse_conv_weight_pack_cache * g_sparse_conv_weight_pack_cache = NULL;

static trellis_status sparse_conv_weight_pack_cache_get(
    const float * weight_dev,
    int in_channels,
    int out_channels,
    int k_volume,
    const float ** packed_dev_out) {
    if (weight_dev == NULL || packed_dev_out == NULL ||
        in_channels <= 0 || out_channels <= 0 || k_volume <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *packed_dev_out = NULL;
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return cuda_status_to_trellis(err);
    }

    for (sparse_conv_weight_pack_cache * cur = g_sparse_conv_weight_pack_cache; cur != NULL; cur = cur->next) {
        if (cur->source_weight_dev == weight_dev &&
            cur->in_channels == in_channels &&
            cur->out_channels == out_channels &&
            cur->k_volume == k_volume &&
            cur->device == device) {
            *packed_dev_out = cur->packed_dev;
            return TRELLIS_STATUS_OK;
        }
    }

    sparse_conv_weight_pack_cache * entry = (sparse_conv_weight_pack_cache *) calloc(1, sizeof(*entry));
    if (entry == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    const int64_t total = (int64_t) k_volume * (int64_t) out_channels * (int64_t) in_channels;
    if (total <= 0 || (uint64_t) total > (uint64_t) SIZE_MAX / sizeof(float)) {
        free(entry);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    err = cudaMalloc((void **) &entry->packed_dev, (size_t) total * sizeof(float));
    if (err != cudaSuccess) {
        free(entry);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    const int block = 256;
    const int grid = (int) ((total + block - 1) / block);
    trellis_sparse_pack_all_weights_f32_kernel<<<grid, block>>>(
        weight_dev,
        entry->packed_dev,
        in_channels,
        out_channels,
        k_volume,
        total);
    trellis_status status = cuda_status_to_trellis(cudaGetLastError());
    if (status != TRELLIS_STATUS_OK) {
        cudaFree(entry->packed_dev);
        free(entry);
        return status;
    }

    entry->source_weight_dev = weight_dev;
    entry->in_channels = in_channels;
    entry->out_channels = out_channels;
    entry->k_volume = k_volume;
    entry->device = device;
    entry->next = g_sparse_conv_weight_pack_cache;
    g_sparse_conv_weight_pack_cache = entry;
    *packed_dev_out = entry->packed_dev;
    return TRELLIS_STATUS_OK;
}

typedef struct sparse_rulebook_matmul_workspace {
    ggml_backend_t backend;
    int in_channels;
    int out_channels;
    int64_t pair_count;
    struct ggml_context * ctx;
    ggml_gallocr_t alloc;
    struct ggml_tensor * x;
    struct ggml_tensor * w;
    struct ggml_tensor * y;
    struct ggml_cgraph * graph;
    struct sparse_rulebook_matmul_workspace * next;
} sparse_rulebook_matmul_workspace;

static void sparse_rulebook_matmul_workspace_free(sparse_rulebook_matmul_workspace * ws) {
    if (ws == NULL) {
        return;
    }
    if (ws->alloc != NULL) {
        ggml_gallocr_free(ws->alloc);
    }
    if (ws->ctx != NULL) {
        ggml_free(ws->ctx);
    }
    free(ws);
}

void sparse_rulebook_matmul_workspace_cache_free(sparse_rulebook_matmul_workspace_cache * cache) {
    if (cache == NULL) {
        return;
    }
    sparse_rulebook_matmul_workspace * cur = cache->head;
    while (cur != NULL) {
        sparse_rulebook_matmul_workspace * next = cur->next;
        sparse_rulebook_matmul_workspace_free(cur);
        cur = next;
    }
    cache->head = NULL;
}

static trellis_status sparse_rulebook_matmul_workspace_cache_get(
    sparse_rulebook_matmul_workspace_cache * cache,
    ggml_backend_t backend,
    int in_channels,
    int out_channels,
    int64_t pair_count,
    sparse_rulebook_matmul_workspace ** ws_out) {
    if (cache == NULL || backend == NULL || ws_out == NULL ||
        in_channels <= 0 || out_channels <= 0 || pair_count <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *ws_out = NULL;
    for (sparse_rulebook_matmul_workspace * cur = cache->head; cur != NULL; cur = cur->next) {
        if (cur->backend == backend &&
            cur->in_channels == in_channels &&
            cur->out_channels == out_channels &&
            cur->pair_count == pair_count) {
            *ws_out = cur;
            return TRELLIS_STATUS_OK;
        }
    }

    sparse_rulebook_matmul_workspace * ws =
        (sparse_rulebook_matmul_workspace *) calloc(1, sizeof(*ws));
    if (ws == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false) + 4096,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    ws->ctx = ggml_init(params);
    if (ws->ctx == NULL) {
        sparse_rulebook_matmul_workspace_free(ws);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ws->x = ggml_new_tensor_2d(ws->ctx, GGML_TYPE_F32, in_channels, pair_count);
    ws->w = ggml_new_tensor_2d(ws->ctx, GGML_TYPE_F32, in_channels, out_channels);
    if (ws->x == NULL || ws->w == NULL) {
        sparse_rulebook_matmul_workspace_free(ws);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ws->y = ggml_mul_mat(ws->ctx, ws->w, ws->x);
    if (ws->y == NULL) {
        sparse_rulebook_matmul_workspace_free(ws);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_mul_mat_set_prec(ws->y, GGML_PREC_F32);
    ws->graph = ggml_new_graph_custom(ws->ctx, 16, false);
    if (ws->graph == NULL) {
        sparse_rulebook_matmul_workspace_free(ws);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_build_forward_expand(ws->graph, ws->y);
    ws->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (ws->alloc == NULL || !ggml_gallocr_alloc_graph(ws->alloc, ws->graph)) {
        sparse_rulebook_matmul_workspace_free(ws);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    ws->backend = backend;
    ws->in_channels = in_channels;
    ws->out_channels = out_channels;
    ws->pair_count = pair_count;
    ws->next = cache->head;
    cache->head = ws;
    *ws_out = ws;
    return TRELLIS_STATUS_OK;
}

static trellis_status sparse_rulebook_offset_matmul_scatter_ggml_f32(
    ggml_backend_t backend,
    sparse_rulebook_matmul_workspace_cache * workspace_cache,
    const float * feats_dev,
    const float * packed_weight_offset_dev,
    float * out_dev,
    const int32_t * src_rows_dev,
    const int32_t * dst_rows_dev,
    int64_t pair_count,
    int in_channels,
    int out_channels) {
    if (backend == NULL ||
        feats_dev == NULL || packed_weight_offset_dev == NULL || out_dev == NULL ||
        src_rows_dev == NULL || dst_rows_dev == NULL || pair_count <= 0 ||
        in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if ((uint64_t) pair_count > (uint64_t) SIZE_MAX / (uint64_t) in_channels / sizeof(float) ||
        (uint64_t) pair_count > (uint64_t) SIZE_MAX / (uint64_t) out_channels / sizeof(float)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_status status = TRELLIS_STATUS_OK;
    sparse_rulebook_matmul_workspace * ws = NULL;
    sparse_rulebook_matmul_workspace_cache temp_cache;
    memset(&temp_cache, 0, sizeof(temp_cache));
    sparse_rulebook_matmul_workspace_cache * active_cache =
        workspace_cache == NULL ? &temp_cache : workspace_cache;
    status = sparse_rulebook_matmul_workspace_cache_get(
        active_cache,
        backend,
        in_channels,
        out_channels,
        pair_count,
        &ws);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }

    {
        const int block = 256;
        const int64_t gather_total = pair_count * (int64_t) in_channels;
        const int gather_grid = (int) ((gather_total + block - 1) / block);
        trellis_sparse_gather_pairs_f32_kernel<<<gather_grid, block>>>(
            feats_dev,
            src_rows_dev,
            (float *) ws->x->data,
            pair_count,
            in_channels,
            gather_total);
        status = cuda_status_to_trellis(cudaGetLastError());
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }

        const int64_t weight_total = (int64_t) out_channels * (int64_t) in_channels;
        status = cuda_status_to_trellis(cudaMemcpy(
            ws->w->data,
            packed_weight_offset_dev,
            (size_t) weight_total * sizeof(float),
            cudaMemcpyDeviceToDevice));
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
    }

    status = cuda_status_to_trellis(cudaDeviceSynchronize());
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    status = ggml_backend_graph_compute(backend, ws->graph) == GGML_STATUS_SUCCESS ?
        TRELLIS_STATUS_OK : TRELLIS_STATUS_ERROR;
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    ggml_backend_synchronize(backend);

    {
        const int block = 256;
        const int64_t scatter_total = pair_count * (int64_t) out_channels;
        const int scatter_grid = (int) ((scatter_total + block - 1) / block);
        trellis_sparse_scatter_add_pairs_f32_kernel<<<scatter_grid, block>>>(
            (const float *) ws->y->data,
            dst_rows_dev,
            out_dev,
            pair_count,
            out_channels,
            scatter_total);
        status = cuda_status_to_trellis(cudaGetLastError());
    }
cleanup:
    if (workspace_cache == NULL) {
        sparse_rulebook_matmul_workspace_cache_free(&temp_cache);
    }
    return status;
}

static int sparse_conv_force_map_kernel(void) {
    const char * backend = getenv("TRELLIS_SPARSE_CONV_BACKEND");
    return backend != NULL &&
        (strcmp(backend, "map") == 0 || strcmp(backend, "neighbor_map") == 0);
}

static trellis_status sparse_subm_conv3d_rulebook_ggml_f32(
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
    int dilation_w) {
    if (backend == NULL ||
        !sparse_neighbor_map_matches(map, n, kernel_d, kernel_h, kernel_w, dilation_d, dilation_h, dilation_w) ||
        !sparse_neighbor_map_has_rulebook(map) ||
        feats_dev == NULL || weight_dev == NULL || out_dev == NULL || in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    const int64_t total = n * (int64_t) out_channels;
    const int block = 256;
    const int grid = (int) ((total + block - 1) / block);
    trellis_sparse_conv_fill_bias_f32_kernel<<<grid, block>>>(out_dev, bias_dev, total, out_channels);
    trellis_status status = cuda_status_to_trellis(cudaGetLastError());
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    const float * packed_weight_dev = NULL;
    status = sparse_conv_weight_pack_cache_get(
        weight_dev,
        in_channels,
        out_channels,
        map->k_volume,
        &packed_weight_dev);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    const int64_t weight_offset_stride = (int64_t) out_channels * (int64_t) in_channels;
    for (int k = 0; status == TRELLIS_STATUS_OK && k < map->k_volume; ++k) {
        const int64_t pair_count = map->offset_counts_host[k];
        if (pair_count <= 0) {
            continue;
        }
        const int64_t start = map->offset_starts_host[k];
        status = sparse_rulebook_offset_matmul_scatter_ggml_f32(
            backend,
            workspace_cache,
            feats_dev,
            packed_weight_dev + (int64_t) k * weight_offset_stride,
            out_dev,
            map->src_rows + start,
            map->dst_rows + start,
            pair_count,
            in_channels,
            out_channels);
    }
    return status;
}

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
    int dilation_w) {
    if (backend != NULL && !sparse_conv_force_map_kernel() && sparse_neighbor_map_has_rulebook(map)) {
        return sparse_subm_conv3d_rulebook_ggml_f32(
            backend,
            workspace_cache,
            map,
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
    return sparse_subm_conv3d_with_map_f32(
        map,
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
    if (!sparse_conv_force_scalar()) {
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
            0,
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

    const int64_t table_size = next_power_of_two_i64(n * 4);
    if (table_size <= n || table_size > (int64_t) INT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int64_t table_mask = table_size - 1;
    unsigned long long * keys_dev = NULL;
    int32_t * values_dev = NULL;
    cudaError_t err = cudaMalloc((void **) &keys_dev, (size_t) table_size * sizeof(unsigned long long));
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    err = cudaMalloc((void **) &values_dev, (size_t) table_size * sizeof(int32_t));
    if (err != cudaSuccess) {
        cudaFree(keys_dev);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    err = cudaMemset(keys_dev, 0, (size_t) table_size * sizeof(unsigned long long));
    trellis_status status = cuda_status_to_trellis(err);
    const int block = 256;
    if (status == TRELLIS_STATUS_OK) {
        const int grid_insert = (int) ((n + block - 1) / block);
        trellis_sparse_hash_insert_kernel<<<grid_insert, block>>>(
            coords_dev,
            keys_dev,
            values_dev,
            n,
            table_mask);
        status = cuda_status_to_trellis(cudaGetLastError());
    }
    if (status == TRELLIS_STATUS_OK) {
        const int64_t total = n * (int64_t) out_channels;
        const int grid_conv = (int) ((total + block - 1) / block);
        trellis_sparse_subm_conv3d_f32_kernel<<<grid_conv, block>>>(
            coords_dev,
            feats_dev,
            weight_dev,
            bias_dev,
            keys_dev,
            values_dev,
            out_dev,
            n,
            in_channels,
            out_channels,
            kernel_d,
            kernel_h,
            kernel_w,
            dilation_d,
            dilation_h,
            dilation_w,
            table_mask,
            total);
        status = cuda_status_to_trellis(cudaGetLastError());
    }
    cudaFree(keys_dev);
    cudaFree(values_dev);
    return status;
}

extern "C" trellis_status trellis_cuda_conv3d_f32_host(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int device,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w) {
    if (!conv3d_args_valid(x, weight, y, batch, in_channels, in_d, in_h, in_w,
            out_channels, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w) || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int out_d = conv_out_size(in_d, kernel_d, stride_d, pad_d, dilation_d);
    const int out_h = conv_out_size(in_h, kernel_h, stride_h, pad_h, dilation_h);
    const int out_w = conv_out_size(in_w, kernel_w, stride_w, pad_w, dilation_w);
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    float * x_dev = NULL;
    float * weight_dev = NULL;
    float * bias_dev = NULL;
    float * y_dev = NULL;
    const size_t x_count = (size_t) batch * (size_t) in_channels * (size_t) in_d * (size_t) in_h * (size_t) in_w;
    const size_t weight_count = (size_t) out_channels * (size_t) in_channels * (size_t) kernel_d * (size_t) kernel_h * (size_t) kernel_w;
    const size_t y_count = (size_t) batch * (size_t) out_channels * (size_t) out_d * (size_t) out_h * (size_t) out_w;

    trellis_status status = malloc_copy_to_device(x, x_count, &x_dev);
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(weight, weight_count, &weight_dev);
    }
    if (status == TRELLIS_STATUS_OK && bias != NULL) {
        status = malloc_copy_to_device(bias, (size_t) out_channels, &bias_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        err = cudaMalloc((void **) &y_dev, y_count * sizeof(float));
        status = err == cudaSuccess ? TRELLIS_STATUS_OK : TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_conv3d_f32(
            x_dev, weight_dev, bias_dev, y_dev,
            batch, in_channels, in_d, in_h, in_w,
            out_channels, kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            dilation_d, dilation_h, dilation_w);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaMemcpy(y, y_dev, y_count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaFree(x_dev);
    cudaFree(weight_dev);
    cudaFree(bias_dev);
    cudaFree(y_dev);
    return status;
}

extern "C" trellis_status trellis_cuda_pixel_shuffle_3d_f32_host(
    const float * x,
    float * y,
    int device,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int scale) {
    if (!pixel_shuffle_args_valid(x, y, batch, in_channels, in_d, in_h, in_w, scale) || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    const int out_channels = in_channels / (scale * scale * scale);
    const int out_d = in_d * scale;
    const int out_h = in_h * scale;
    const int out_w = in_w * scale;
    const size_t x_count = (size_t) batch * (size_t) in_channels * (size_t) in_d * (size_t) in_h * (size_t) in_w;
    const size_t y_count = (size_t) batch * (size_t) out_channels * (size_t) out_d * (size_t) out_h * (size_t) out_w;

    float * x_dev = NULL;
    float * y_dev = NULL;
    trellis_status status = malloc_copy_to_device(x, x_count, &x_dev);
    if (status == TRELLIS_STATUS_OK) {
        err = cudaMalloc((void **) &y_dev, y_count * sizeof(float));
        status = err == cudaSuccess ? TRELLIS_STATUS_OK : TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_pixel_shuffle_3d_f32(x_dev, y_dev, batch, in_channels, in_d, in_h, in_w, scale);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaMemcpy(y, y_dev, y_count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaFree(x_dev);
    cudaFree(y_dev);
    return status;
}

extern "C" trellis_status trellis_cuda_channel_layer_norm_3d_f32_host(
    const float * x,
    const float * gamma,
    const float * beta,
    float * y,
    int device,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    float eps) {
    if (!channel_layer_norm_3d_args_valid(x, y, batch, channels, depth, height, width, eps) || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    const size_t count = (size_t) batch * (size_t) channels * (size_t) depth * (size_t) height * (size_t) width;
    float * x_dev = NULL;
    float * gamma_dev = NULL;
    float * beta_dev = NULL;
    float * y_dev = NULL;
    trellis_status status = malloc_copy_to_device(x, count, &x_dev);
    if (status == TRELLIS_STATUS_OK && gamma != NULL) {
        status = malloc_copy_to_device(gamma, (size_t) channels, &gamma_dev);
    }
    if (status == TRELLIS_STATUS_OK && beta != NULL) {
        status = malloc_copy_to_device(beta, (size_t) channels, &beta_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        err = cudaMalloc((void **) &y_dev, count * sizeof(float));
        status = err == cudaSuccess ? TRELLIS_STATUS_OK : TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_channel_layer_norm_3d_f32(
            x_dev, gamma_dev, beta_dev, y_dev, batch, channels, depth, height, width, eps);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaMemcpy(y, y_dev, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaFree(x_dev);
    cudaFree(gamma_dev);
    cudaFree(beta_dev);
    cudaFree(y_dev);
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

    ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
    ggml_backend_load_all();
    backend = ggml_backend_cuda_init(device);
#endif
    if (backend == NULL) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    const size_t input_count = (size_t) n * (size_t) in_channels;
    const size_t output_count = (size_t) n * (size_t) out_channels;
    const size_t weight_count = (size_t) out_channels * (size_t) in_channels;
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * 4 + 4096,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        ggml_backend_free(backend);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_status status = TRELLIS_STATUS_OK;
    ggml_backend_buffer_t buffer = NULL;
    float * x_dev = NULL;
    float * y_dev = NULL;
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_channels, out_channels);
    struct ggml_tensor * b = bias == NULL ? NULL : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
    if (w == NULL || (bias != NULL && b == NULL)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    ggml_backend_tensor_set(w, weight, 0, weight_count * sizeof(float));
    if (b != NULL) {
        ggml_backend_tensor_set(b, bias, 0, (size_t) out_channels * sizeof(float));
    }
    status = malloc_copy_to_device(x, input_count, &x_dev);
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_malloc_f32(output_count, &y_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_linear_device(backend, x_dev, w, b, y_dev, n, in_channels, out_channels);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaMemcpy(y, y_dev, output_count * sizeof(float), cudaMemcpyDeviceToHost));
    }

cleanup:
    cudaFree(x_dev);
    cudaFree(y_dev);
    if (buffer != NULL) {
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    ggml_backend_free(backend);
    return status;
}

extern "C" trellis_status trellis_cuda_dino_patch_embed_f32_host(
    const float * image,
    const float * weight,
    const float * bias,
    float * tokens,
    int device,
    int batch,
    int image_h,
    int image_w,
    int out_channels,
    int patch_size) {
    if (!dino_patch_embed_args_valid(image, weight, tokens, batch, image_h, image_w, out_channels, patch_size) || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }
    const int n_patches = (image_h / patch_size) * (image_w / patch_size);
    const size_t image_count = (size_t) batch * 3u * (size_t) image_h * (size_t) image_w;
    const size_t weight_count = (size_t) out_channels * 3u * (size_t) patch_size * (size_t) patch_size;
    const size_t token_count = (size_t) batch * (size_t) n_patches * (size_t) out_channels;
    float * image_dev = NULL;
    float * weight_dev = NULL;
    float * bias_dev = NULL;
    float * tokens_dev = NULL;
    trellis_status status = malloc_copy_to_device(image, image_count, &image_dev);
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(weight, weight_count, &weight_dev);
    }
    if (status == TRELLIS_STATUS_OK && bias != NULL) {
        status = malloc_copy_to_device(bias, (size_t) out_channels, &bias_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_malloc_f32(token_count, &tokens_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_dino_patch_embed_f32(
            image_dev,
            weight_dev,
            bias_dev,
            tokens_dev,
            batch,
            image_h,
            image_w,
            out_channels,
            patch_size);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaMemcpy(tokens, tokens_dev, token_count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaFree(image_dev);
    cudaFree(weight_dev);
    cudaFree(bias_dev);
    cudaFree(tokens_dev);
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

extern "C" trellis_status trellis_cuda_apply_rope_f32_host(
    const float * x,
    const float * cos_phase,
    const float * sin_phase,
    float * y,
    int device,
    int batch,
    int tokens,
    int heads,
    int head_dim) {
    if (!rope_args_valid(x, cos_phase, sin_phase, y, batch, tokens, heads, head_dim) || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    const int half_dim = head_dim / 2;
    const size_t x_count = (size_t) batch * (size_t) heads * (size_t) tokens * (size_t) head_dim;
    const size_t phase_count = (size_t) tokens * (size_t) half_dim;

    float * x_dev = NULL;
    float * cos_dev = NULL;
    float * sin_dev = NULL;
    float * y_dev = NULL;
    trellis_status status = malloc_copy_to_device(x, x_count, &x_dev);
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(cos_phase, phase_count, &cos_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(sin_phase, phase_count, &sin_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        err = cudaMalloc((void **) &y_dev, x_count * sizeof(float));
        status = err == cudaSuccess ? TRELLIS_STATUS_OK : TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_apply_rope_f32(x_dev, cos_dev, sin_dev, y_dev, batch, tokens, heads, head_dim);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaMemcpy(y, y_dev, x_count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaFree(x_dev);
    cudaFree(cos_dev);
    cudaFree(sin_dev);
    cudaFree(y_dev);
    return status;
}
