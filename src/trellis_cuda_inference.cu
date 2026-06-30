#include "trellis.h"
#include "custom_ops/trellis_cuda_ops.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Network execution lives here; CUDA kernels and reusable op wrappers live in
 * src/custom_ops. The shape decoder entry point copies slat+coords to CUDA,
 * projects features, builds sparse neighbor maps, runs ConvNeXt/C2S levels,
 * then applies the final norm+linear head. The SS decoder entry point runs the
 * dense 3D VAE decode blocks and pixel-shuffle upsampling.
 */

static trellis_status cuda_status_to_trellis(cudaError_t err) {
    return err == cudaSuccess ? TRELLIS_STATUS_OK : TRELLIS_STATUS_ERROR;
}

static int trellis_profile_enabled(void) {
    const char * enabled = getenv("TRELLIS_PROFILE");
    return enabled != NULL && enabled[0] != '\0' && strcmp(enabled, "0") != 0;
}

static double trellis_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1000.0 + (double) ts.tv_nsec / 1000000.0;
}

static void trellis_profile_cuda_log(const char * label, double start_ms) {
    if (!trellis_profile_enabled()) {
        return;
    }
    cudaDeviceSynchronize();
    fprintf(stderr, "trellis_profile\t%s\t%.3f ms\n", label, trellis_now_ms() - start_ms);
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

typedef struct shape_decoder_debug_writer {
    const char * dump_dir;
    FILE * manifest;
    int next_index;
} shape_decoder_debug_writer;

static int shape_debug_enabled(const shape_decoder_debug_writer * debug) {
    return debug != NULL && debug->dump_dir != NULL && debug->dump_dir[0] != '\0' && debug->manifest != NULL;
}

static int shape_debug_should_dump(const shape_decoder_debug_writer * debug, const char * name) {
    if (!shape_debug_enabled(debug)) {
        return 0;
    }
    const char * filters = getenv("TRELLIS_SHAPE_DEBUG_FILTER");
    if (filters == NULL || filters[0] == '\0') {
        return 1;
    }

    const char * p = filters;
    while (*p != '\0') {
        while (*p == ',' || *p == ' ' || *p == '\t') {
            ++p;
        }
        const char * start = p;
        while (*p != '\0' && *p != ',') {
            ++p;
        }
        const char * end = p;
        while (end > start && (end[-1] == ' ' || end[-1] == '\t')) {
            --end;
        }
        const size_t len = (size_t) (end - start);
        if (len == 1 && start[0] == '*') {
            return 1;
        }
        if (len > 0 && len < 128 && name != NULL) {
            char token[128];
            memcpy(token, start, len);
            token[len] = '\0';
            if (strstr(name, token) != NULL) {
                return 1;
            }
        }
    }
    return 0;
}

static trellis_status shape_debug_make_path(
    shape_decoder_debug_writer * debug,
    const char * name,
    const char * ext,
    char * rel_path,
    size_t rel_path_size,
    char * abs_path,
    size_t abs_path_size) {
    if (!shape_debug_enabled(debug) || name == NULL || ext == NULL ||
        rel_path == NULL || abs_path == NULL || rel_path_size == 0 || abs_path_size == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    int n = snprintf(rel_path, rel_path_size, "%03d_%s.%s", debug->next_index++, name, ext);
    if (n < 0 || (size_t) n >= rel_path_size) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    n = snprintf(abs_path, abs_path_size, "%s/%s", debug->dump_dir, rel_path);
    if (n < 0 || (size_t) n >= abs_path_size) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status shape_debug_write_manifest(
    shape_decoder_debug_writer * debug,
    const char * name,
    int64_t rows,
    int cols,
    const char * dtype,
    const char * rel_path) {
    if (!shape_debug_enabled(debug)) {
        return TRELLIS_STATUS_OK;
    }
    if (name == NULL || dtype == NULL || rel_path == NULL || rows < 0 || cols <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    fprintf(debug->manifest, "%s\t%lldx%d\t%s\t%s\n",
        name,
        (long long) rows,
        cols,
        dtype,
        rel_path);
    return ferror(debug->manifest) ? TRELLIS_STATUS_ERROR : TRELLIS_STATUS_OK;
}

static trellis_status shape_debug_dump_host_f32(
    shape_decoder_debug_writer * debug,
    const char * name,
    const float * data,
    int64_t rows,
    int cols) {
    if (!shape_debug_should_dump(debug, name)) {
        return TRELLIS_STATUS_OK;
    }
    if (name == NULL || data == NULL || rows < 0 || cols <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    char rel_path[512];
    char abs_path[4096];
    trellis_status status = shape_debug_make_path(debug, name, "f32", rel_path, sizeof(rel_path), abs_path, sizeof(abs_path));
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    FILE * f = fopen(abs_path, "wb");
    if (f == NULL) {
        return TRELLIS_STATUS_ERROR;
    }
    const size_t count = (size_t) rows * (size_t) cols;
    size_t wrote = fwrite(data, sizeof(float), count, f);
    int close_rc = fclose(f);
    if (wrote != count || close_rc != 0) {
        return TRELLIS_STATUS_ERROR;
    }
    return shape_debug_write_manifest(debug, name, rows, cols, "f32", rel_path);
}

static trellis_status shape_debug_dump_host_i32(
    shape_decoder_debug_writer * debug,
    const char * name,
    const int32_t * data,
    int64_t rows,
    int cols) {
    if (!shape_debug_should_dump(debug, name)) {
        return TRELLIS_STATUS_OK;
    }
    if (name == NULL || data == NULL || rows < 0 || cols <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    char rel_path[512];
    char abs_path[4096];
    trellis_status status = shape_debug_make_path(debug, name, "i32", rel_path, sizeof(rel_path), abs_path, sizeof(abs_path));
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    FILE * f = fopen(abs_path, "wb");
    if (f == NULL) {
        return TRELLIS_STATUS_ERROR;
    }
    const size_t count = (size_t) rows * (size_t) cols;
    size_t wrote = fwrite(data, sizeof(int32_t), count, f);
    int close_rc = fclose(f);
    if (wrote != count || close_rc != 0) {
        return TRELLIS_STATUS_ERROR;
    }
    return shape_debug_write_manifest(debug, name, rows, cols, "i32", rel_path);
}

static trellis_status shape_debug_dump_device_f32(
    shape_decoder_debug_writer * debug,
    const char * name,
    const float * data_dev,
    int64_t rows,
    int cols) {
    if (!shape_debug_should_dump(debug, name)) {
        return TRELLIS_STATUS_OK;
    }
    if (data_dev == NULL || rows < 0 || cols <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t count = (size_t) rows * (size_t) cols;
    float * host = (float *) malloc(count * sizeof(float));
    if (host == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    trellis_status status = cuda_status_to_trellis(cudaMemcpy(host, data_dev, count * sizeof(float), cudaMemcpyDeviceToHost));
    if (status == TRELLIS_STATUS_OK) {
        status = shape_debug_dump_host_f32(debug, name, host, rows, cols);
    }
    free(host);
    return status;
}

static int sparse_linear_force_scalar(void) {
    const char * backend = getenv("TRELLIS_SPARSE_LINEAR_BACKEND");
    return backend != NULL && strcmp(backend, "scalar") == 0;
}

static int c2s_linear_force_ggml(void) {
    const char * backend = getenv("TRELLIS_C2S_LINEAR_BACKEND");
    return backend != NULL &&
        (strcmp(backend, "ggml") == 0 || strcmp(backend, "matmul") == 0);
}

static trellis_status make_c2s_mapping_host(
    const int32_t * coords,
    const float * subdiv_logits,
    int64_t n,
    int32_t ** coords_out,
    int32_t ** parent_out,
    int32_t ** subidx_out,
    int64_t * n_out) {
    if (coords == NULL || subdiv_logits == NULL || coords_out == NULL ||
        parent_out == NULL || subidx_out == NULL || n_out == NULL || n <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *coords_out = NULL;
    *parent_out = NULL;
    *subidx_out = NULL;
    *n_out = 0;

    int64_t m = 0;
    for (int64_t i = 0; i < n; ++i) {
        for (int s = 0; s < 8; ++s) {
            if (subdiv_logits[i * 8 + s] > 0.0f) {
                ++m;
            }
        }
    }
    if (m <= 0) {
        return TRELLIS_STATUS_ERROR;
    }
    int32_t * new_coords = (int32_t *) malloc((size_t) m * 4u * sizeof(int32_t));
    int32_t * parent = (int32_t *) malloc((size_t) m * sizeof(int32_t));
    int32_t * subidx = (int32_t *) malloc((size_t) m * sizeof(int32_t));
    if (new_coords == NULL || parent == NULL || subidx == NULL) {
        free(new_coords);
        free(parent);
        free(subidx);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    int64_t row = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t * c = coords + 4 * i;
        for (int s = 0; s < 8; ++s) {
            if (subdiv_logits[i * 8 + s] <= 0.0f) {
                continue;
            }
            new_coords[4 * row + 0] = c[0];
            new_coords[4 * row + 1] = c[1] * 2 + (s & 1);
            new_coords[4 * row + 2] = c[2] * 2 + ((s >> 1) & 1);
            new_coords[4 * row + 3] = c[3] * 2 + ((s >> 2) & 1);
            parent[row] = (int32_t) i;
            subidx[row] = (int32_t) s;
            ++row;
        }
    }

    *coords_out = new_coords;
    *parent_out = parent;
    *subidx_out = subidx;
    *n_out = m;
    return TRELLIS_STATUS_OK;
}

static trellis_status shape_convnext_block_device(
    ggml_backend_t matmul_backend,
    sparse_rulebook_matmul_workspace_cache * sparse_conv_cache,
    const trellis_shape_decoder_convnext_block_weights * block,
    const sparse_neighbor_map_device * neighbor_map,
    const float * h_dev,
    int64_t n,
    shape_decoder_debug_writer * debug,
    int level,
    int block_index,
    float ** out_dev) {
    if (block == NULL || neighbor_map == NULL || h_dev == NULL || out_dev == NULL ||
        n <= 0 || block->channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out_dev = NULL;
    const int c = block->channels;
    const size_t count = (size_t) n * (size_t) c;
    float * conv = NULL;
    float * norm = NULL;
    float * mlp1 = NULL;
    float * mlp2 = NULL;
    float * out = NULL;
    char dump_name[128];
    trellis_status status = cuda_malloc_f32(count, &conv);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32(count, &norm);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) n * (size_t) c * 4u, &mlp1);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32(count, &mlp2);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32(count, &out);
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_subm_conv3d_dispatch_f32(
            matmul_backend,
            sparse_conv_cache,
            neighbor_map,
            h_dev,
            (const float *) block->conv_w->data,
            (const float *) block->conv_b->data,
            conv,
            n,
            c,
            c,
            3,
            3,
            3,
            1,
            1,
            1);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_b%02d_conv", level, block_index);
        status = shape_debug_dump_device_f32(debug, dump_name, conv, n, c);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = row_layer_norm_device(conv, block->norm_gamma, block->norm_beta, norm, n, c, 1e-6f);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_b%02d_norm", level, block_index);
        status = shape_debug_dump_device_f32(debug, dump_name, norm, n, c);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_linear_device(matmul_backend, norm, block->mlp0_w, block->mlp0_b, mlp1, n, c, 4 * c);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_b%02d_mlp0", level, block_index);
        status = shape_debug_dump_device_f32(debug, dump_name, mlp1, n, 4 * c);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_silu_f32(mlp1, mlp1, (size_t) n * (size_t) c * 4u);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_b%02d_mlp0_silu", level, block_index);
        status = shape_debug_dump_device_f32(debug, dump_name, mlp1, n, 4 * c);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_linear_device(matmul_backend, mlp1, block->mlp2_w, block->mlp2_b, mlp2, n, 4 * c, c);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_b%02d_mlp2", level, block_index);
        status = shape_debug_dump_device_f32(debug, dump_name, mlp2, n, c);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_add_f32(h_dev, mlp2, out, count);
    }
    cudaFree(conv);
    cudaFree(norm);
    cudaFree(mlp1);
    cudaFree(mlp2);
    if (status == TRELLIS_STATUS_OK) {
        *out_dev = out;
    } else {
        cudaFree(out);
    }
    return status;
}

static trellis_status shape_c2s_block_device(
    ggml_backend_t matmul_backend,
    const trellis_shape_decoder_c2s_block_weights * block,
    const int32_t * coords_host,
    const int32_t * coords_dev,
    const sparse_neighbor_map_device * cur_neighbor_map,
    const float * h_dev,
    int64_t n,
    shape_decoder_debug_writer * debug,
    int level,
    int32_t ** coords_host_out,
    int32_t ** coords_dev_out,
    float ** h_dev_out,
    sparse_neighbor_map_device * next_neighbor_map_out,
    int64_t * n_out) {
    if (block == NULL || coords_host == NULL || coords_dev == NULL || cur_neighbor_map == NULL || h_dev == NULL ||
        coords_host_out == NULL || coords_dev_out == NULL || h_dev_out == NULL ||
        next_neighbor_map_out == NULL || n_out == NULL ||
        n <= 0 || block->in_channels <= 0 || block->out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *coords_host_out = NULL;
    *coords_dev_out = NULL;
    *h_dev_out = NULL;
    memset(next_neighbor_map_out, 0, sizeof(*next_neighbor_map_out));
    *n_out = 0;

    const int ci = block->in_channels;
    const int co = block->out_channels;
    sparse_neighbor_map_device next_neighbor_map;
    memset(&next_neighbor_map, 0, sizeof(next_neighbor_map));
    float * subdiv_dev = NULL;
    float * subdiv_host = NULL;
    int32_t * next_coords_host = NULL;
    int32_t * parent_host = NULL;
    int32_t * subidx_host = NULL;
    int32_t * parent_dev = NULL;
    int32_t * subidx_dev = NULL;
    int32_t * next_coords_dev = NULL;
    float * norm1 = NULL;
    float * conv1 = NULL;
    float * h_up = NULL;
    float * skip = NULL;
    float * norm2 = NULL;
    float * conv2 = NULL;
    float * out = NULL;
    int64_t m = 0;
    char dump_name[128];

    trellis_status status = cuda_malloc_f32((size_t) n * 8u, &subdiv_dev);
    if (status == TRELLIS_STATUS_OK) {
        if (c2s_linear_force_ggml()) {
            status = sparse_linear_device(matmul_backend, h_dev, block->to_subdiv_w, block->to_subdiv_b, subdiv_dev, n, ci, 8);
        } else {
            status = sparse_linear_device_scalar(h_dev, block->to_subdiv_w, block->to_subdiv_b, subdiv_dev, n, ci, 8);
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        subdiv_host = (float *) malloc((size_t) n * 8u * sizeof(float));
        if (subdiv_host == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaMemcpy(
            subdiv_host,
            subdiv_dev,
            (size_t) n * 8u * sizeof(float),
            cudaMemcpyDeviceToHost));
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_subdiv_logits", level);
        status = shape_debug_dump_host_f32(debug, dump_name, subdiv_host, n, 8);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = make_c2s_mapping_host(
            coords_host,
            subdiv_host,
            n,
            &next_coords_host,
            &parent_host,
            &subidx_host,
            &m);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_coords", level);
        status = shape_debug_dump_host_i32(debug, dump_name, next_coords_host, m, 4);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_parent", level);
        status = shape_debug_dump_host_i32(debug, dump_name, parent_host, m, 1);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_subidx", level);
        status = shape_debug_dump_host_i32(debug, dump_name, subidx_host, m, 1);
    }
    if (status == TRELLIS_STATUS_OK) status = malloc_copy_i32_to_device(next_coords_host, (size_t) m * 4u, &next_coords_dev);
    if (status == TRELLIS_STATUS_OK) status = malloc_copy_i32_to_device(parent_host, (size_t) m, &parent_dev);
    if (status == TRELLIS_STATUS_OK) status = malloc_copy_i32_to_device(subidx_host, (size_t) m, &subidx_dev);
    if (status == TRELLIS_STATUS_OK) {
        const double t0 = trellis_now_ms();
        status = sparse_neighbor_map_build_device(next_coords_dev, m, 3, 3, 3, 1, 1, 1, 1, &next_neighbor_map);
        trellis_profile_cuda_log("shape.c2s.next_neighbor_map", t0);
    }

    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) n * (size_t) ci, &norm1);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) n * (size_t) co * 8u, &conv1);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) m * (size_t) co, &h_up);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) m * (size_t) co, &skip);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) m * (size_t) co, &norm2);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) m * (size_t) co, &conv2);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) m * (size_t) co, &out);

    if (status == TRELLIS_STATUS_OK) {
        status = row_layer_norm_device(h_dev, block->norm1_gamma, block->norm1_beta, norm1, n, ci, 1e-6f);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_silu_f32(norm1, norm1, (size_t) n * (size_t) ci);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_norm1_silu", level);
        status = shape_debug_dump_device_f32(debug, dump_name, norm1, n, ci);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_subm_conv3d_dispatch_f32(
            matmul_backend,
            NULL,
            cur_neighbor_map,
            norm1,
            (const float *) block->conv1_w->data,
            (const float *) block->conv1_b->data,
            conv1,
            n,
            ci,
            8 * co,
            3,
            3,
            3,
            1,
            1,
            1);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_conv1", level);
        status = shape_debug_dump_device_f32(debug, dump_name, conv1, n, 8 * co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_c2s_gather_device(conv1, parent_dev, subidx_dev, h_up, m, co);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_h_up", level);
        status = shape_debug_dump_device_f32(debug, dump_name, h_up, m, co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_c2s_skip_repeat_device(h_dev, parent_dev, subidx_dev, skip, m, ci, co);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_skip", level);
        status = shape_debug_dump_device_f32(debug, dump_name, skip, m, co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = row_layer_norm_device(h_up, NULL, NULL, norm2, m, co, 1e-6f);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_silu_f32(norm2, norm2, (size_t) m * (size_t) co);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_norm2_silu", level);
        status = shape_debug_dump_device_f32(debug, dump_name, norm2, m, co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_subm_conv3d_dispatch_f32(
            matmul_backend,
            NULL,
            &next_neighbor_map,
            norm2,
            (const float *) block->conv2_w->data,
            (const float *) block->conv2_b->data,
            conv2,
            m,
            co,
            co,
            3,
            3,
            3,
            1,
            1,
            1);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_conv2", level);
        status = shape_debug_dump_device_f32(debug, dump_name, conv2, m, co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_add_f32(conv2, skip, out, (size_t) m * (size_t) co);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_out", level);
        status = shape_debug_dump_device_f32(debug, dump_name, out, m, co);
    }

    cudaFree(subdiv_dev);
    free(subdiv_host);
    free(parent_host);
    free(subidx_host);
    cudaFree(parent_dev);
    cudaFree(subidx_dev);
    cudaFree(norm1);
    cudaFree(conv1);
    cudaFree(h_up);
    cudaFree(skip);
    cudaFree(norm2);
    cudaFree(conv2);
    if (status == TRELLIS_STATUS_OK) {
        *coords_host_out = next_coords_host;
        *coords_dev_out = next_coords_dev;
        *h_dev_out = out;
        *next_neighbor_map_out = next_neighbor_map;
        memset(&next_neighbor_map, 0, sizeof(next_neighbor_map));
        *n_out = m;
    } else {
        free(next_coords_host);
        cudaFree(next_coords_dev);
        cudaFree(out);
        sparse_neighbor_map_free(&next_neighbor_map);
    }
    return status;
}

static trellis_status decoder_conv3d_same(
    const float * x,
    const struct ggml_tensor * w,
    const struct ggml_tensor * b,
    float * y,
    int batch,
    int in_channels,
    int size,
    int out_channels) {
    return trellis_cuda_conv3d_f32(
        x,
        (const float *) w->data,
        b == NULL ? NULL : (const float *) b->data,
        y,
        batch,
        in_channels,
        size,
        size,
        size,
        out_channels,
        3,
        3,
        3,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1);
}

static trellis_status decoder_resblock(
    const trellis_ss_decoder_resblock_weights * block,
    float ** x_io,
    int batch,
    int size) {
    if (block == NULL || x_io == NULL || *x_io == NULL || batch <= 0 || size <= 0 || block->channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int channels = block->channels;
    const size_t count = (size_t) batch * (size_t) channels * (size_t) size * (size_t) size * (size_t) size;
    float * tmp_a = NULL;
    float * tmp_b = NULL;
    trellis_status status = cuda_malloc_f32(count, &tmp_a);
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_malloc_f32(count, &tmp_b);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_channel_layer_norm_3d_f32(
            *x_io,
            (const float *) block->norm1_gamma->data,
            (const float *) block->norm1_beta->data,
            tmp_a,
            batch,
            channels,
            size,
            size,
            size,
            1e-6f);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_silu_f32(tmp_a, tmp_a, count);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = decoder_conv3d_same(*x_io == tmp_a ? *x_io : tmp_a, block->conv1_w, block->conv1_b, tmp_b, batch, channels, size, channels);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_channel_layer_norm_3d_f32(
            tmp_b,
            (const float *) block->norm2_gamma->data,
            (const float *) block->norm2_beta->data,
            tmp_a,
            batch,
            channels,
            size,
            size,
            size,
            1e-6f);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_silu_f32(tmp_a, tmp_a, count);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = decoder_conv3d_same(tmp_a, block->conv2_w, block->conv2_b, tmp_b, batch, channels, size, channels);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_add_f32(*x_io, tmp_b, tmp_a, count);
    }
    if (status == TRELLIS_STATUS_OK) {
        cudaFree(*x_io);
        *x_io = tmp_a;
        tmp_a = NULL;
    }
    cudaFree(tmp_a);
    cudaFree(tmp_b);
    return status;
}

static trellis_status decoder_upsample(
    float ** x_io,
    const struct ggml_tensor * conv_w,
    const struct ggml_tensor * conv_b,
    int batch,
    int in_channels,
    int * size_io,
    int conv_out_channels) {
    if (x_io == NULL || *x_io == NULL || conv_w == NULL || conv_b == NULL || size_io == NULL ||
        batch <= 0 || in_channels <= 0 || *size_io <= 0 || conv_out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int size = *size_io;
    const size_t conv_count = (size_t) batch * (size_t) conv_out_channels * (size_t) size * (size_t) size * (size_t) size;
    float * conv = NULL;
    trellis_status status = cuda_malloc_f32(conv_count, &conv);
    if (status == TRELLIS_STATUS_OK) {
        status = decoder_conv3d_same(*x_io, conv_w, conv_b, conv, batch, in_channels, size, conv_out_channels);
    }
    const int out_channels = conv_out_channels / 8;
    const int out_size = size * 2;
    const size_t out_count = (size_t) batch * (size_t) out_channels * (size_t) out_size * (size_t) out_size * (size_t) out_size;
    float * out = NULL;
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_malloc_f32(out_count, &out);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_pixel_shuffle_3d_f32(conv, out, batch, conv_out_channels, size, size, size, 2);
    }
    if (status == TRELLIS_STATUS_OK) {
        cudaFree(*x_io);
        *x_io = out;
        *size_io = out_size;
        out = NULL;
    }
    cudaFree(conv);
    cudaFree(out);
    return status;
}

extern "C" trellis_status trellis_ss_decoder_forward_f32_host(
    const trellis_ss_decoder_weights * weights,
    const float * latent,
    int device,
    int batch,
    int latent_size,
    float ** logits_out,
    int * output_size) {
    if (weights == NULL || latent == NULL || logits_out == NULL || output_size == NULL ||
        device < 0 || batch <= 0 || latent_size <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *logits_out = NULL;
    *output_size = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    int size = latent_size;
    const size_t latent_count = (size_t) batch * 8u * (size_t) size * (size_t) size * (size_t) size;
    float * cur = NULL;
    trellis_status status = malloc_copy_to_device(latent, latent_count, &cur);

    float * next = NULL;
    const size_t input_count = (size_t) batch * 512u * (size_t) size * (size_t) size * (size_t) size;
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_malloc_f32(input_count, &next);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = decoder_conv3d_same(cur, weights->input_w, weights->input_b, next, batch, 8, size, 512);
    }
    if (status == TRELLIS_STATUS_OK) {
        cudaFree(cur);
        cur = next;
        next = NULL;
    }

    if (status == TRELLIS_STATUS_OK) status = decoder_resblock(&weights->middle[0], &cur, batch, size);
    if (status == TRELLIS_STATUS_OK) status = decoder_resblock(&weights->middle[1], &cur, batch, size);
    if (status == TRELLIS_STATUS_OK) status = decoder_resblock(&weights->block0, &cur, batch, size);
    if (status == TRELLIS_STATUS_OK) status = decoder_resblock(&weights->block1, &cur, batch, size);
    if (status == TRELLIS_STATUS_OK) status = decoder_upsample(&cur, weights->up0_w, weights->up0_b, batch, 512, &size, 1024);
    if (status == TRELLIS_STATUS_OK) status = decoder_resblock(&weights->block3, &cur, batch, size);
    if (status == TRELLIS_STATUS_OK) status = decoder_resblock(&weights->block4, &cur, batch, size);
    if (status == TRELLIS_STATUS_OK) status = decoder_upsample(&cur, weights->up1_w, weights->up1_b, batch, 128, &size, 256);
    if (status == TRELLIS_STATUS_OK) status = decoder_resblock(&weights->block6, &cur, batch, size);
    if (status == TRELLIS_STATUS_OK) status = decoder_resblock(&weights->block7, &cur, batch, size);

    const size_t tail_count = (size_t) batch * 32u * (size_t) size * (size_t) size * (size_t) size;
    float * tail = NULL;
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_malloc_f32(tail_count, &tail);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_channel_layer_norm_3d_f32(
            cur,
            (const float *) weights->out_norm_gamma->data,
            (const float *) weights->out_norm_beta->data,
            tail,
            batch,
            32,
            size,
            size,
            size,
            1e-6f);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_silu_f32(tail, tail, tail_count);
    }

    const size_t logits_count = (size_t) batch * (size_t) size * (size_t) size * (size_t) size;
    float * logits_dev = NULL;
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_malloc_f32(logits_count, &logits_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = decoder_conv3d_same(tail, weights->out_w, weights->out_b, logits_dev, batch, 32, size, 1);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        float * host = (float *) malloc(logits_count * sizeof(float));
        if (host == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else {
            status = cuda_status_to_trellis(cudaMemcpy(host, logits_dev, logits_count * sizeof(float), cudaMemcpyDeviceToHost));
            if (status == TRELLIS_STATUS_OK) {
                *logits_out = host;
                *output_size = size;
                host = NULL;
            }
            free(host);
        }
    }

    cudaFree(cur);
    cudaFree(next);
    cudaFree(tail);
    cudaFree(logits_dev);
    return status;
}

extern "C" trellis_status trellis_shape_decoder_forward_f32_host_debug(
    const trellis_shape_decoder_weights * weights,
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int device,
    int max_levels,
    const trellis_shape_decoder_debug_options * debug_options,
    int32_t ** coords_out,
    float ** feats_out,
    int64_t * n_out,
    int * channels_out) {
    if (weights == NULL || coords == NULL || feats == NULL || coords_out == NULL ||
        feats_out == NULL || n_out == NULL || channels_out == NULL || n <= 0 || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *coords_out = NULL;
    *feats_out = NULL;
    *n_out = 0;
    *channels_out = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    const int levels = weights->levels <= 0 ? TRELLIS_SHAPE_DECODER_LEVELS : weights->levels;
    int levels_to_run = max_levels <= 0 || max_levels > levels ? levels : max_levels;
    if (levels_to_run <= 0 || levels_to_run > TRELLIS_SHAPE_DECODER_LEVELS) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    shape_decoder_debug_writer debug;
    memset(&debug, 0, sizeof(debug));
    if (debug_options != NULL && debug_options->dump_dir != NULL && debug_options->dump_dir[0] != '\0') {
        char manifest_path[4096];
        int n_path = snprintf(manifest_path, sizeof(manifest_path), "%s/shape_decoder_intermediates.tsv", debug_options->dump_dir);
        if (n_path < 0 || (size_t) n_path >= sizeof(manifest_path)) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        debug.dump_dir = debug_options->dump_dir;
        debug.manifest = fopen(manifest_path, "w");
        if (debug.manifest == NULL) {
            return TRELLIS_STATUS_ERROR;
        }
        fprintf(debug.manifest, "name\tshape\tdtype\tfile\n");
        if (ferror(debug.manifest)) {
            fclose(debug.manifest);
            return TRELLIS_STATUS_ERROR;
        }
    }

    int32_t * cur_coords_host = (int32_t *) malloc((size_t) n * 4u * sizeof(int32_t));
    if (cur_coords_host == NULL) {
        if (debug.manifest != NULL) {
            fclose(debug.manifest);
        }
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    memcpy(cur_coords_host, coords, (size_t) n * 4u * sizeof(int32_t));

    ggml_backend_t matmul_backend = NULL;
#ifdef GGML_USE_CUDA
    if (!sparse_linear_force_scalar()) {
        ggml_backend_load_all();
        matmul_backend = ggml_backend_cuda_init(device);
        if (matmul_backend == NULL) {
            free(cur_coords_host);
            if (debug.manifest != NULL) {
                fclose(debug.manifest);
            }
            return TRELLIS_STATUS_CUDA_UNAVAILABLE;
        }
    }
#endif

    int32_t * cur_coords_dev = NULL;
    float * in_dev = NULL;
    float * cur_h = NULL;
    trellis_status status = shape_debug_dump_host_i32(&debug, "input_coords", coords, n, 4);
    if (status == TRELLIS_STATUS_OK) {
        status = shape_debug_dump_host_f32(&debug, "input_slat", feats, n, 32);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_i32_to_device(coords, (size_t) n * 4u, &cur_coords_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(feats, (size_t) n * 32u, &in_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_malloc_f32((size_t) n * (size_t) weights->channels[0], &cur_h);
    }
    if (status == TRELLIS_STATUS_OK) {
        const double t0 = trellis_now_ms();
        status = sparse_linear_device(
            matmul_backend,
            in_dev,
            weights->from_latent_w,
            weights->from_latent_b,
            cur_h,
            n,
            32,
            weights->channels[0]);
        trellis_profile_cuda_log("shape.from_latent", t0);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = shape_debug_dump_device_f32(&debug, "from_latent", cur_h, n, weights->channels[0]);
    }
    cudaFree(in_dev);
    in_dev = NULL;

    int cur_channels = weights->channels[0];
    int64_t cur_n = n;
    sparse_neighbor_map_device cur_neighbor_map;
    memset(&cur_neighbor_map, 0, sizeof(cur_neighbor_map));
    sparse_rulebook_matmul_workspace_cache sparse_conv_cache;
    memset(&sparse_conv_cache, 0, sizeof(sparse_conv_cache));
    if (status == TRELLIS_STATUS_OK) {
        const double t0 = trellis_now_ms();
        status = sparse_neighbor_map_build_device(cur_coords_dev, cur_n, 3, 3, 3, 1, 1, 1, 1, &cur_neighbor_map);
        trellis_profile_cuda_log("shape.initial_neighbor_map", t0);
    }
    char dump_name[128];
    for (int level = 0; status == TRELLIS_STATUS_OK && level < levels_to_run; ++level) {
        cur_channels = weights->channels[level];
        for (int block = 0; status == TRELLIS_STATUS_OK && block < weights->blocks_per_level[level]; ++block) {
            float * next_h = NULL;
            const double t0 = trellis_now_ms();
            status = shape_convnext_block_device(
                matmul_backend,
                &sparse_conv_cache,
                &weights->blocks[level][block],
                &cur_neighbor_map,
                cur_h,
                cur_n,
                &debug,
                level,
                block,
                &next_h);
            if (trellis_profile_enabled()) {
                char profile_name[64];
                snprintf(profile_name, sizeof(profile_name), "shape.l%02d.b%02d", level, block);
                trellis_profile_cuda_log(profile_name, t0);
            }
            if (status == TRELLIS_STATUS_OK) {
                cudaFree(cur_h);
                cur_h = next_h;
                snprintf(dump_name, sizeof(dump_name), "l%02d_b%02d", level, block);
                status = shape_debug_dump_device_f32(&debug, dump_name, cur_h, cur_n, cur_channels);
            }
        }
        if (status != TRELLIS_STATUS_OK || level >= levels_to_run - 1 || level >= levels - 1) {
            break;
        }

        int32_t * next_coords_host = NULL;
        int32_t * next_coords_dev = NULL;
        float * next_h = NULL;
        sparse_neighbor_map_device next_neighbor_map;
        memset(&next_neighbor_map, 0, sizeof(next_neighbor_map));
        int64_t next_n = 0;
        const double t0 = trellis_now_ms();
        status = shape_c2s_block_device(
            matmul_backend,
            &weights->up_blocks[level],
            cur_coords_host,
            cur_coords_dev,
            &cur_neighbor_map,
            cur_h,
            cur_n,
            &debug,
            level,
            &next_coords_host,
            &next_coords_dev,
            &next_h,
            &next_neighbor_map,
            &next_n);
        if (trellis_profile_enabled()) {
            char profile_name[64];
            snprintf(profile_name, sizeof(profile_name), "shape.l%02d.c2s", level);
            trellis_profile_cuda_log(profile_name, t0);
        }
        if (status == TRELLIS_STATUS_OK) {
            free(cur_coords_host);
            cudaFree(cur_coords_dev);
            cudaFree(cur_h);
            sparse_neighbor_map_free(&cur_neighbor_map);
            sparse_rulebook_matmul_workspace_cache_free(&sparse_conv_cache);
            cur_coords_host = next_coords_host;
            cur_coords_dev = next_coords_dev;
            cur_h = next_h;
            cur_neighbor_map = next_neighbor_map;
            memset(&next_neighbor_map, 0, sizeof(next_neighbor_map));
            cur_n = next_n;
            cur_channels = weights->channels[level + 1];
        } else {
            sparse_neighbor_map_free(&next_neighbor_map);
        }
    }
    sparse_rulebook_matmul_workspace_cache_free(&sparse_conv_cache);

    if (status == TRELLIS_STATUS_OK && levels_to_run == levels) {
        float * norm = NULL;
        float * out = NULL;
        status = cuda_malloc_f32((size_t) cur_n * (size_t) cur_channels, &norm);
        if (status == TRELLIS_STATUS_OK) {
            status = cuda_malloc_f32((size_t) cur_n * 7u, &out);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = row_layer_norm_device(cur_h, NULL, NULL, norm, cur_n, cur_channels, 1e-5f);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = shape_debug_dump_device_f32(&debug, "output_norm", norm, cur_n, cur_channels);
        }
        if (status == TRELLIS_STATUS_OK) {
            const double t0 = trellis_now_ms();
            status = sparse_linear_device(matmul_backend, norm, weights->output_w, weights->output_b, out, cur_n, cur_channels, 7);
            trellis_profile_cuda_log("shape.output_linear", t0);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = shape_debug_dump_device_f32(&debug, "output_logits", out, cur_n, 7);
        }
        cudaFree(norm);
        if (status == TRELLIS_STATUS_OK) {
            cudaFree(cur_h);
            cur_h = out;
            cur_channels = 7;
        } else {
            cudaFree(out);
        }
    }

    float * host_feats = NULL;
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        host_feats = (float *) malloc((size_t) cur_n * (size_t) cur_channels * sizeof(float));
        if (host_feats == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        const double t0 = trellis_now_ms();
        status = cuda_status_to_trellis(cudaMemcpy(
            host_feats,
            cur_h,
            (size_t) cur_n * (size_t) cur_channels * sizeof(float),
            cudaMemcpyDeviceToHost));
        trellis_profile_cuda_log("shape.copy_output_to_host", t0);
    }
    if (status == TRELLIS_STATUS_OK) {
        *coords_out = cur_coords_host;
        *feats_out = host_feats;
        *n_out = cur_n;
        *channels_out = cur_channels;
        cur_coords_host = NULL;
        host_feats = NULL;
    }

    free(host_feats);
    free(cur_coords_host);
    sparse_neighbor_map_free(&cur_neighbor_map);
    cudaFree(cur_coords_dev);
    cudaFree(cur_h);
    if (matmul_backend != NULL) {
        ggml_backend_free(matmul_backend);
    }
    if (debug.manifest != NULL) {
        fclose(debug.manifest);
    }
    return status;
}

extern "C" trellis_status trellis_shape_decoder_forward_f32_host(
    const trellis_shape_decoder_weights * weights,
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int device,
    int max_levels,
    int32_t ** coords_out,
    float ** feats_out,
    int64_t * n_out,
    int * channels_out) {
    return trellis_shape_decoder_forward_f32_host_debug(
        weights,
        coords,
        feats,
        n,
        device,
        max_levels,
        NULL,
        coords_out,
        feats_out,
        n_out,
        channels_out);
}

extern "C" trellis_status trellis_dino_patch_embed_f32_host(
    const trellis_dino_vit_weights * weights,
    const float * image,
    int device,
    int batch,
    int image_h,
    int image_w,
    float ** tokens_out,
    int * n_patches_out) {
    if (weights == NULL || image == NULL || tokens_out == NULL || n_patches_out == NULL ||
        device < 0 || batch <= 0 || image_h <= 0 || image_w <= 0 ||
        image_h % weights->patch_size != 0 || image_w % weights->patch_size != 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *tokens_out = NULL;
    *n_patches_out = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    const int n_patches = (image_h / weights->patch_size) * (image_w / weights->patch_size);
    const size_t image_count = (size_t) batch * 3u * (size_t) image_h * (size_t) image_w;
    const size_t token_count = (size_t) batch * (size_t) n_patches * (size_t) weights->hidden_size;
    float * image_dev = NULL;
    float * tokens_dev = NULL;
    trellis_status status = malloc_copy_to_device(image, image_count, &image_dev);
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_malloc_f32(token_count, &tokens_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_dino_patch_embed_f32(
            image_dev,
            (const float *) weights->patch_w->data,
            (const float *) weights->patch_b->data,
            tokens_dev,
            batch,
            image_h,
            image_w,
            weights->hidden_size,
            weights->patch_size);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        float * host = (float *) malloc(token_count * sizeof(float));
        if (host == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else {
            status = cuda_status_to_trellis(cudaMemcpy(host, tokens_dev, token_count * sizeof(float), cudaMemcpyDeviceToHost));
            if (status == TRELLIS_STATUS_OK) {
                *tokens_out = host;
                *n_patches_out = n_patches;
                host = NULL;
            }
            free(host);
        }
    }
    cudaFree(image_dev);
    cudaFree(tokens_dev);
    return status;
}
