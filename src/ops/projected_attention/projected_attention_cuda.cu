#include "projected_attention.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static trellis_status cuda_error_status(cudaError_t error) {
    if (error == cudaSuccess) return TRELLIS_STATUS_OK;
    if (error == cudaErrorMemoryAllocation) return TRELLIS_STATUS_OUT_OF_MEMORY;
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
        error == cudaErrorInitializationError) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }
    return TRELLIS_STATUS_ERROR;
}

static int checked_mul_size(size_t a, size_t b, size_t * out) {
    if (out == NULL || (a != 0 && b > SIZE_MAX / a)) return 0;
    *out = a * b;
    return 1;
}

static int tensor_count4(
    int batch,
    int height,
    int width,
    int channels,
    size_t * count) {
    size_t value = 0;
    return batch > 0 && height > 0 && width > 0 && channels > 0 &&
        checked_mul_size((size_t) batch, (size_t) height, &value) &&
        checked_mul_size(value, (size_t) width, &value) &&
        checked_mul_size(value, (size_t) channels, count);
}

static int validate_shape_and_counts(
    const trellis_pixal_naf_attention_desc * desc,
    int64_t n_coords,
    size_t projected_count,
    size_t * query_count,
    size_t * key_count,
    size_t * value_count,
    size_t * coord_count,
    size_t * needed_output) {
    if (desc == NULL || n_coords <= 0 || (uint64_t) n_coords > (uint64_t) SIZE_MAX ||
        desc->batch <= 0 || desc->query_h <= 0 || desc->query_w <= 0 ||
        desc->feature_h < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE ||
        desc->feature_w < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE ||
        desc->image_resolution <= 0 || desc->grid_resolution <= 0 ||
        desc->query_h % desc->feature_h != 0 ||
        desc->query_w % desc->feature_w != 0 ||
        desc->query_h / desc->feature_h != desc->query_w / desc->feature_w) {
        return 0;
    }
    return tensor_count4(
               desc->batch,
               desc->query_h,
               desc->query_w,
               TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS,
               query_count) &&
        tensor_count4(
               desc->batch,
               desc->feature_h,
               desc->feature_w,
               TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS,
               key_count) &&
        tensor_count4(
               desc->batch,
               desc->feature_h,
               desc->feature_w,
               TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS,
               value_count) &&
        checked_mul_size((size_t) n_coords, 4u, coord_count) &&
        checked_mul_size(
               (size_t) n_coords,
               TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS,
               needed_output) &&
        projected_count >= *needed_output;
}

static int valid_camera(const trellis_pixal_camera * camera) {
    static const float pi = 3.14159265358979323846f;
    return camera != NULL && isfinite(camera->camera_angle_x) &&
        camera->camera_angle_x > 0.0f && camera->camera_angle_x < pi &&
        isfinite(camera->distance) && isfinite(camera->mesh_scale) &&
        camera->mesh_scale != 0.0f;
}

static __device__ __forceinline__ float device_grid_coordinate(
    int index,
    int resolution) {
    if (resolution == 1) return 0.0f;
    return -1.0f + 2.0f * (float) index / (float) (resolution - 1);
}

static __device__ __forceinline__ size_t device_bhwc_offset(
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

static __global__ void pixal_naf_attention_project_kernel(
    const float * __restrict__ queries,
    const float * __restrict__ keys,
    const float * __restrict__ values,
    const int32_t * __restrict__ coords_bxyz,
    const trellis_pixal_camera * __restrict__ cameras,
    trellis_pixal_naf_attention_desc desc,
    int64_t n_coords,
    float * __restrict__ output) {
    const int64_t token_head = (int64_t) blockIdx.x;
    if (token_head >= n_coords * TRELLIS_PIXAL_NAF_ATTENTION_HEADS) return;
    const int token = (int) (token_head / TRELLIS_PIXAL_NAF_ATTENTION_HEADS);
    const int head = (int) (token_head % TRELLIS_PIXAL_NAF_ATTENTION_HEADS);
    const int lane = (int) threadIdx.x;
    const int batch = coords_bxyz[(size_t) token * 4u + 0u];

    __shared__ float query_vector[TRELLIS_PIXAL_NAF_ATTENTION_QUERY_HEAD_DIM];
    __shared__ float scores[
        TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE *
        TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE];
    __shared__ int sample_x[4];
    __shared__ int sample_y[4];
    __shared__ float sample_weight[4];

    if (lane == 0) {
        const int grid_x = coords_bxyz[(size_t) token * 4u + 1u];
        const int grid_y = coords_bxyz[(size_t) token * 4u + 2u];
        const int grid_z = coords_bxyz[(size_t) token * 4u + 3u];
        const trellis_pixal_camera camera = cameras[batch];
        const float inv_scale_2 = 0.5f / camera.mesh_scale;
        const float world_x =
            device_grid_coordinate(grid_x, desc.grid_resolution) * inv_scale_2;
        const float camera_y =
            device_grid_coordinate(grid_y, desc.grid_resolution) * inv_scale_2;
        const float depth = camera.distance -
            device_grid_coordinate(grid_z, desc.grid_resolution) * inv_scale_2;
        const float focal_pixels =
            0.5f * (float) desc.image_resolution /
            tanf(0.5f * camera.camera_angle_x);
        const float denominator = depth + 1e-8f;
        const float pixel_x = focal_pixels * world_x / denominator +
            0.5f * (float) desc.image_resolution;
        const float pixel_y = -focal_pixels * camera_y / denominator +
            0.5f * (float) desc.image_resolution;
        float x =
            (pixel_x + 0.5f) * (float) desc.query_w /
                (float) desc.image_resolution -
            0.5f;
        float y =
            (pixel_y + 0.5f) * (float) desc.query_h /
                (float) desc.image_resolution -
            0.5f;
        if (!isfinite(x)) x = x < 0.0f ? 0.0f : (float) (desc.query_w - 1);
        if (!isfinite(y)) y = y < 0.0f ? 0.0f : (float) (desc.query_h - 1);
        x = fminf(fmaxf(x, 0.0f), (float) (desc.query_w - 1));
        y = fminf(fmaxf(y, 0.0f), (float) (desc.query_h - 1));
        const int x0 = (int) floorf(x);
        const int y0 = (int) floorf(y);
        const int x1 = x0 + 1 < desc.query_w ? x0 + 1 : x0;
        const int y1 = y0 + 1 < desc.query_h ? y0 + 1 : y0;
        const float wx = x - (float) x0;
        const float wy = y - (float) y0;
        sample_x[0] = x0;
        sample_y[0] = y0;
        sample_weight[0] = (1.0f - wx) * (1.0f - wy);
        sample_x[1] = x1;
        sample_y[1] = y0;
        sample_weight[1] = wx * (1.0f - wy);
        sample_x[2] = x0;
        sample_y[2] = y1;
        sample_weight[2] = (1.0f - wx) * wy;
        sample_x[3] = x1;
        sample_y[3] = y1;
        sample_weight[3] = wx * wy;
    }
    __syncthreads();

    const int dilation = desc.query_h / desc.feature_h;
    float result = 0.0f;
    for (int corner = 0; corner < 4; ++corner) {
        const int query_y = sample_y[corner];
        const int query_x = sample_x[corner];
        const int start_y = max(
            0,
            min(query_y / dilation - 4,
                desc.feature_h - TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE));
        const int start_x = max(
            0,
            min(query_x / dilation - 4,
                desc.feature_w - TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE));
        if (lane < TRELLIS_PIXAL_NAF_ATTENTION_QUERY_HEAD_DIM) {
            const size_t query_base = device_bhwc_offset(
                batch,
                query_y,
                query_x,
                desc.query_h,
                desc.query_w,
                TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS);
            query_vector[lane] = queries[
                query_base + (size_t) head *
                    TRELLIS_PIXAL_NAF_ATTENTION_QUERY_HEAD_DIM +
                (size_t) lane];
        }
        __syncthreads();

        if (lane < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE *
                TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE) {
            const int kernel_y = lane / TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE;
            const int kernel_x = lane % TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE;
            const size_t key_base = device_bhwc_offset(
                batch,
                start_y + kernel_y,
                start_x + kernel_x,
                desc.feature_h,
                desc.feature_w,
                TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS);
            float dot = 0.0f;
            for (int channel = 0;
                 channel < TRELLIS_PIXAL_NAF_ATTENTION_QUERY_HEAD_DIM;
                 ++channel) {
                dot += query_vector[channel] * keys[
                    key_base + (size_t) head *
                        TRELLIS_PIXAL_NAF_ATTENTION_QUERY_HEAD_DIM +
                    (size_t) channel];
            }
            scores[lane] = dot * 0.125f;
        }
        __syncthreads();

        if (lane == 0) {
            float maximum = scores[0];
            for (int i = 1;
                 i < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE *
                     TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE;
                 ++i) {
                if (scores[i] > maximum) maximum = scores[i];
            }
            float denominator = 0.0f;
            for (int i = 0;
                 i < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE *
                     TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE;
                 ++i) {
                scores[i] = expf(scores[i] - maximum);
                denominator += scores[i];
            }
            const float inverse_denominator = 1.0f / denominator;
            for (int i = 0;
                 i < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE *
                     TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE;
                 ++i) {
                scores[i] *= inverse_denominator;
            }
        }
        __syncthreads();

        if (lane < TRELLIS_PIXAL_NAF_ATTENTION_VALUE_HEAD_DIM) {
            float value_sum = 0.0f;
            for (int neighbor = 0;
                 neighbor < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE *
                     TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE;
                 ++neighbor) {
                const int kernel_y =
                    neighbor / TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE;
                const int kernel_x =
                    neighbor % TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE;
                const size_t value_base = device_bhwc_offset(
                    batch,
                    start_y + kernel_y,
                    start_x + kernel_x,
                    desc.feature_h,
                    desc.feature_w,
                    TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS);
                value_sum += scores[neighbor] * values[
                    value_base + (size_t) head *
                        TRELLIS_PIXAL_NAF_ATTENTION_VALUE_HEAD_DIM +
                    (size_t) lane];
            }
            result += sample_weight[corner] * value_sum;
        }
        __syncthreads();
    }

    if (lane < TRELLIS_PIXAL_NAF_ATTENTION_VALUE_HEAD_DIM) {
        output[(size_t) token * TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS +
            (size_t) head * TRELLIS_PIXAL_NAF_ATTENTION_VALUE_HEAD_DIM +
            (size_t) lane] = result;
    }
}

extern "C" trellis_status trellis_pixal_naf_attention_project_sparse_cuda_f32(
    const float * queries_dev,
    const float * keys_dev,
    const float * values_dev,
    const int32_t * coords_bxyz_dev,
    int64_t n_coords,
    const trellis_pixal_camera * cameras_dev,
    const trellis_pixal_naf_attention_desc * desc,
    float * projected_out_dev,
    size_t projected_count) {
    size_t query_count = 0;
    size_t key_count = 0;
    size_t value_count = 0;
    size_t coord_count = 0;
    size_t needed_output = 0;
    if (queries_dev == NULL || keys_dev == NULL || values_dev == NULL ||
        coords_bxyz_dev == NULL || cameras_dev == NULL || projected_out_dev == NULL ||
        !validate_shape_and_counts(
            desc,
            n_coords,
            projected_count,
            &query_count,
            &key_count,
            &value_count,
            &coord_count,
            &needed_output)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    (void) query_count;
    (void) key_count;
    (void) value_count;
    (void) coord_count;
    (void) needed_output;

    int device = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) return cuda_error_status(error);
    cudaDeviceProp properties;
    error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return cuda_error_status(error);
    const uint64_t blocks =
        (uint64_t) n_coords * TRELLIS_PIXAL_NAF_ATTENTION_HEADS;
    if (blocks == 0 || blocks > (uint64_t) properties.maxGridSize[0]) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    pixal_naf_attention_project_kernel<<<(unsigned int) blocks, 256>>>(
        queries_dev,
        keys_dev,
        values_dev,
        coords_bxyz_dev,
        cameras_dev,
        *desc,
        n_coords,
        projected_out_dev);
    return cuda_error_status(cudaGetLastError());
}

static trellis_status malloc_copy_to_device(
    const void * source,
    size_t count,
    size_t element_size,
    void ** destination) {
    *destination = NULL;
    size_t bytes = 0;
    if (source == NULL || !checked_mul_size(count, element_size, &bytes)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t error = cudaMalloc(destination, bytes);
    if (error != cudaSuccess) return cuda_error_status(error);
    error = cudaMemcpy(*destination, source, bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(*destination);
        *destination = NULL;
    }
    return cuda_error_status(error);
}

extern "C" trellis_status trellis_pixal_naf_attention_project_sparse_cuda_host_f32(
    const float * queries,
    const float * keys,
    const float * values,
    const int32_t * coords_bxyz,
    int64_t n_coords,
    const trellis_pixal_camera * cameras,
    const trellis_pixal_naf_attention_desc * desc,
    int device,
    float * projected_out,
    size_t projected_count) {
    size_t query_count = 0;
    size_t key_count = 0;
    size_t value_count = 0;
    size_t coord_count = 0;
    size_t needed_output = 0;
    if (queries == NULL || keys == NULL || values == NULL || coords_bxyz == NULL ||
        cameras == NULL || projected_out == NULL || device < 0 ||
        !validate_shape_and_counts(
            desc,
            n_coords,
            projected_count,
            &query_count,
            &key_count,
            &value_count,
            &coord_count,
            &needed_output)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int batch = 0; batch < desc->batch; ++batch) {
        if (!valid_camera(cameras + batch)) return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int64_t row = 0; row < n_coords; ++row) {
        const int32_t batch = coords_bxyz[(size_t) row * 4u + 0u];
        const int32_t x = coords_bxyz[(size_t) row * 4u + 1u];
        const int32_t y = coords_bxyz[(size_t) row * 4u + 2u];
        const int32_t z = coords_bxyz[(size_t) row * 4u + 3u];
        if (batch < 0 || batch >= desc->batch || x < 0 || y < 0 || z < 0 ||
            x >= desc->grid_resolution || y >= desc->grid_resolution ||
            z >= desc->grid_resolution) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }

    int previous_device = -1;
    cudaError_t error = cudaGetDevice(&previous_device);
    if (error != cudaSuccess) return cuda_error_status(error);
    const int switched_device = previous_device != device;
    if (switched_device) {
        error = cudaSetDevice(device);
        if (error != cudaSuccess) return cuda_error_status(error);
    }
    float * queries_dev = NULL;
    float * keys_dev = NULL;
    float * values_dev = NULL;
    int32_t * coords_dev = NULL;
    trellis_pixal_camera * cameras_dev = NULL;
    float * output_dev = NULL;
    trellis_status status = malloc_copy_to_device(
        queries,
        query_count,
        sizeof(float),
        (void **) &queries_dev);
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(
            keys, key_count, sizeof(float), (void **) &keys_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(
            values, value_count, sizeof(float), (void **) &values_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(
            coords_bxyz, coord_count, sizeof(int32_t), (void **) &coords_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(
            cameras,
            (size_t) desc->batch,
            sizeof(trellis_pixal_camera),
            (void **) &cameras_dev);
    }
    size_t output_bytes = 0;
    if (status == TRELLIS_STATUS_OK &&
        !checked_mul_size(needed_output, sizeof(float), &output_bytes)) {
        status = TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (status == TRELLIS_STATUS_OK) {
        error = cudaMalloc((void **) &output_dev, output_bytes);
        status = cuda_error_status(error);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_pixal_naf_attention_project_sparse_cuda_f32(
            queries_dev,
            keys_dev,
            values_dev,
            coords_dev,
            n_coords,
            cameras_dev,
            desc,
            output_dev,
            needed_output);
    }
    if (status == TRELLIS_STATUS_OK) {
        error = cudaMemcpy(
            projected_out,
            output_dev,
            output_bytes,
            cudaMemcpyDeviceToHost);
        status = cuda_error_status(error);
    }

    if (output_dev != NULL) cudaFree(output_dev);
    if (cameras_dev != NULL) cudaFree(cameras_dev);
    if (coords_dev != NULL) cudaFree(coords_dev);
    if (values_dev != NULL) cudaFree(values_dev);
    if (keys_dev != NULL) cudaFree(keys_dev);
    if (queries_dev != NULL) cudaFree(queries_dev);
    if (switched_device) {
        const cudaError_t restore_error = cudaSetDevice(previous_device);
        if (status == TRELLIS_STATUS_OK && restore_error != cudaSuccess) {
            status = cuda_error_status(restore_error);
        }
    }
    return status;
}
