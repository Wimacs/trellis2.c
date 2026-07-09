#include "trellis.h"
#include "trellis_platform.h"
#include "trellis_cuda_kernels.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Network execution lives here; CUDA kernels and reusable op wrappers live in
 * src/model/trellis_cuda_kernels.cu. The SparseUnetVaeDecoder entry point copies slat+coords
 * to CUDA, projects features, builds sparse neighbor maps, runs ConvNeXt/C2S levels,
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
    return (double) trellis_now_us() / 1000.0;
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

static trellis_status shape_debug_dump_device_i32(
    shape_decoder_debug_writer * debug,
    const char * name,
    const int32_t * data_dev,
    int64_t rows,
    int cols) {
    if (!shape_debug_should_dump(debug, name)) {
        return TRELLIS_STATUS_OK;
    }
    if (data_dev == NULL || rows < 0 || cols <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t count = (size_t) rows * (size_t) cols;
    int32_t * host = (int32_t *) malloc(count * sizeof(int32_t));
    if (host == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    trellis_status status = cuda_status_to_trellis(cudaMemcpy(host, data_dev, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (status == TRELLIS_STATUS_OK) {
        status = shape_debug_dump_host_i32(debug, name, host, rows, cols);
    }
    free(host);
    return status;
}

static trellis_status shape_convnext_block_device(
    const trellis_shape_decoder_convnext_block_weights * block,
    const sparse_subm_conv3d_device * conv3d,
    const float * h_dev,
    int64_t n,
    shape_decoder_debug_writer * debug,
    int level,
    int block_index,
    float ** out_dev) {
    if (block == NULL || conv3d == NULL || h_dev == NULL || out_dev == NULL ||
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
        status = sparse_subm_conv3d_device_forward_f32(
            conv3d,
            h_dev,
            block->conv_w,
            block->conv_b,
            conv,
            n,
            c,
            c);
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
        status = sparse_linear_device_cublas(norm, block->mlp0_w, block->mlp0_b, mlp1, n, c, 4 * c);
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
        status = sparse_linear_device_cublas(mlp1, block->mlp2_w, block->mlp2_b, mlp2, n, 4 * c, c);
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
    const trellis_shape_decoder_c2s_block_weights * block,
    const int32_t * coords_dev,
    const sparse_subm_conv3d_device * cur_conv3d,
    const float * h_dev,
    int64_t n,
    shape_decoder_debug_writer * debug,
    int level,
    const trellis_sparse_c2s_guide_level * guide_in,
    trellis_sparse_c2s_guide_level * guide_out,
    int32_t ** coords_dev_out,
    float ** h_dev_out,
    sparse_subm_conv3d_device * next_conv3d_out,
    int64_t * n_out) {
    if (block == NULL || coords_dev == NULL || cur_conv3d == NULL || h_dev == NULL ||
        coords_dev_out == NULL || h_dev_out == NULL || next_conv3d_out == NULL || n_out == NULL ||
        n <= 0 || block->in_channels <= 0 || block->out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *coords_dev_out = NULL;
    *h_dev_out = NULL;
    memset(next_conv3d_out, 0, sizeof(*next_conv3d_out));
    *n_out = 0;

    const int ci = block->in_channels;
    const int co = block->out_channels;
    sparse_c2s_map_device c2s_map;
    sparse_subm_conv3d_device next_conv3d;
    memset(&c2s_map, 0, sizeof(c2s_map));
    memset(&next_conv3d, 0, sizeof(next_conv3d));
    float * subdiv_dev = NULL;
    float * norm1 = NULL;
    float * conv1 = NULL;
    float * h_up = NULL;
    float * skip = NULL;
    float * norm2 = NULL;
    float * conv2 = NULL;
    float * out = NULL;
    int64_t m = 0;
    char dump_name[128];
    const char * fail_step = NULL;

    trellis_status status = TRELLIS_STATUS_OK;
    if (guide_out != NULL) {
        memset(guide_out, 0, sizeof(*guide_out));
    }
    if (guide_in != NULL) {
        if (guide_in->coords_bxyz == NULL || guide_in->parent == NULL ||
            guide_in->subidx == NULL || guide_in->n_coords <= 0) {
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        if (status == TRELLIS_STATUS_OK) {
            m = guide_in->n_coords;
            status = malloc_copy_i32_to_device(guide_in->coords_bxyz, (size_t) m * 4u, &c2s_map.coords);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = malloc_copy_i32_to_device(guide_in->parent, (size_t) m, &c2s_map.parent);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = malloc_copy_i32_to_device(guide_in->subidx, (size_t) m, &c2s_map.subidx);
        }
        if (status == TRELLIS_STATUS_OK) {
            c2s_map.n = m;
        }
        if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
            fail_step = "load guide map";
        }
    } else {
        if (block->to_subdiv_w == NULL || block->to_subdiv_b == NULL) {
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        if (status == TRELLIS_STATUS_OK) {
            status = cuda_malloc_f32((size_t) n * 8u, &subdiv_dev);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = sparse_linear_device_cublas(h_dev, block->to_subdiv_w, block->to_subdiv_b, subdiv_dev, n, ci, 8);
        }
        if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
            snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_subdiv_logits", level);
            status = shape_debug_dump_device_f32(debug, dump_name, subdiv_dev, n, 8);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = sparse_c2s_map_build_device(coords_dev, subdiv_dev, n, &c2s_map);
        }
        if (status == TRELLIS_STATUS_OK) {
            m = c2s_map.n;
        }
        if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
            fail_step = "predict/build c2s map";
        }
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_coords", level);
        status = shape_debug_dump_device_i32(debug, dump_name, c2s_map.coords, m, 4);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_parent", level);
        status = shape_debug_dump_device_i32(debug, dump_name, c2s_map.parent, m, 1);
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_subidx", level);
        status = shape_debug_dump_device_i32(debug, dump_name, c2s_map.subidx, m, 1);
    }
    if (status == TRELLIS_STATUS_OK && guide_out != NULL) {
        guide_out->coords_bxyz = (int32_t *) malloc((size_t) m * 4u * sizeof(int32_t));
        guide_out->parent = (int32_t *) malloc((size_t) m * sizeof(int32_t));
        guide_out->subidx = (int32_t *) malloc((size_t) m * sizeof(int32_t));
        if (guide_out->coords_bxyz == NULL || guide_out->parent == NULL || guide_out->subidx == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        }
        if (status == TRELLIS_STATUS_OK) {
            guide_out->n_coords = m;
            status = cuda_status_to_trellis(cudaMemcpy(
                guide_out->coords_bxyz,
                c2s_map.coords,
                (size_t) m * 4u * sizeof(int32_t),
                cudaMemcpyDeviceToHost));
        }
        if (status == TRELLIS_STATUS_OK) {
            status = cuda_status_to_trellis(cudaMemcpy(
                guide_out->parent,
                c2s_map.parent,
                (size_t) m * sizeof(int32_t),
                cudaMemcpyDeviceToHost));
        }
        if (status == TRELLIS_STATUS_OK) {
            status = cuda_status_to_trellis(cudaMemcpy(
                guide_out->subidx,
                c2s_map.subidx,
                (size_t) m * sizeof(int32_t),
                cudaMemcpyDeviceToHost));
        }
        if (status != TRELLIS_STATUS_OK) {
            if (fail_step == NULL) {
                fail_step = "copy guide to host";
            }
            free(guide_out->coords_bxyz);
            free(guide_out->parent);
            free(guide_out->subidx);
            memset(guide_out, 0, sizeof(*guide_out));
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        const double t0 = trellis_now_ms();
        status = sparse_subm_conv3d_device_build(c2s_map.coords, m, 3, 3, 3, 1, 1, 1, &next_conv3d);
        if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
            fail_step = "build next sparse conv map";
        }
        trellis_profile_cuda_log("shape.c2s.next_neighbor_map", t0);
    }

    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) n * (size_t) ci, &norm1);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) n * (size_t) co * 8u, &conv1);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) m * (size_t) co, &h_up);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) m * (size_t) co, &skip);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) m * (size_t) co, &norm2);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) m * (size_t) co, &conv2);
    if (status == TRELLIS_STATUS_OK) status = cuda_malloc_f32((size_t) m * (size_t) co, &out);
    if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
        fail_step = "allocate c2s temporaries";
    }

    if (status == TRELLIS_STATUS_OK) {
        status = row_layer_norm_device(h_dev, block->norm1_gamma, block->norm1_beta, norm1, n, ci, 1e-6f);
        if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
            fail_step = "norm1";
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_silu_f32(norm1, norm1, (size_t) n * (size_t) ci);
        if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
            fail_step = "silu norm1";
        }
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_norm1_silu", level);
        status = shape_debug_dump_device_f32(debug, dump_name, norm1, n, ci);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_subm_conv3d_device_forward_f32(
            cur_conv3d,
            norm1,
            block->conv1_w,
            block->conv1_b,
            conv1,
            n,
            ci,
            8 * co);
        if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
            fail_step = "conv1";
        }
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_conv1", level);
        status = shape_debug_dump_device_f32(debug, dump_name, conv1, n, 8 * co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_c2s_gather_device(conv1, c2s_map.parent, c2s_map.subidx, h_up, m, co);
        if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
            fail_step = "gather";
        }
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_h_up", level);
        status = shape_debug_dump_device_f32(debug, dump_name, h_up, m, co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_c2s_skip_repeat_device(h_dev, c2s_map.parent, c2s_map.subidx, skip, m, ci, co);
        if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
            fail_step = "skip repeat";
        }
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_skip", level);
        status = shape_debug_dump_device_f32(debug, dump_name, skip, m, co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = row_layer_norm_device(h_up, NULL, NULL, norm2, m, co, 1e-6f);
        if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
            fail_step = "norm2";
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_silu_f32(norm2, norm2, (size_t) m * (size_t) co);
        if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
            fail_step = "silu norm2";
        }
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_norm2_silu", level);
        status = shape_debug_dump_device_f32(debug, dump_name, norm2, m, co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = sparse_subm_conv3d_device_forward_f32(
            &next_conv3d,
            norm2,
            block->conv2_w,
            block->conv2_b,
            conv2,
            m,
            co,
            co);
        if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
            fail_step = "conv2";
        }
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_conv2", level);
        status = shape_debug_dump_device_f32(debug, dump_name, conv2, m, co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_cuda_add_f32(conv2, skip, out, (size_t) m * (size_t) co);
        if (status != TRELLIS_STATUS_OK && fail_step == NULL) {
            fail_step = "residual add";
        }
    }
    if (status == TRELLIS_STATUS_OK && shape_debug_enabled(debug)) {
        snprintf(dump_name, sizeof(dump_name), "l%02d_c2s_out", level);
        status = shape_debug_dump_device_f32(debug, dump_name, out, m, co);
    }

    cudaFree(subdiv_dev);
    cudaFree(norm1);
    cudaFree(conv1);
    cudaFree(h_up);
    cudaFree(skip);
    cudaFree(norm2);
    cudaFree(conv2);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "SparseUnetVaeDecoder: C2S level %d failed at %s: %s input_tokens=%lld output_tokens=%lld in=%d out=%d",
            level,
            fail_step != NULL ? fail_step : "unknown",
            trellis_status_string(status),
            (long long) n,
            (long long) m,
            ci,
            co);
    }
    if (status == TRELLIS_STATUS_OK) {
        *coords_dev_out = c2s_map.coords;
        c2s_map.coords = NULL;
        *h_dev_out = out;
        *next_conv3d_out = next_conv3d;
        memset(&next_conv3d, 0, sizeof(next_conv3d));
        *n_out = m;
    } else {
        cudaFree(out);
    }
    sparse_c2s_map_device_free(&c2s_map);
    sparse_subm_conv3d_device_free(&next_conv3d);
    return status;
}

static trellis_status sparse_unet_vae_decoder_forward_f32_host_debug(
    const trellis_shape_decoder_weights * weights,
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int device,
    int max_levels,
    const trellis_sparse_c2s_guides * guide_subs,
    trellis_sparse_c2s_guides * return_subs,
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
    if (return_subs != NULL) {
        memset(return_subs, 0, sizeof(*return_subs));
    }
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    const int levels = weights->levels <= 0 ? TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS : weights->levels;
    int levels_to_run = max_levels <= 0 || max_levels > levels ? levels : max_levels;
    if (levels_to_run <= 0 || levels_to_run > TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS ||
        weights->latent_channels <= 0 || weights->out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!weights->pred_subdiv && guide_subs == NULL && levels_to_run > 1) {
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

    int32_t * cur_coords_dev = NULL;
    float * in_dev = NULL;
    float * cur_h = NULL;
    trellis_status status = shape_debug_dump_host_i32(&debug, "input_coords", coords, n, 4);
    if (status == TRELLIS_STATUS_OK) {
        status = shape_debug_dump_host_f32(&debug, "input_slat", feats, n, weights->latent_channels);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_i32_to_device(coords, (size_t) n * 4u, &cur_coords_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = malloc_copy_to_device(feats, (size_t) n * (size_t) weights->latent_channels, &in_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_malloc_f32((size_t) n * (size_t) weights->channels[0], &cur_h);
    }
    if (status == TRELLIS_STATUS_OK) {
        const double t0 = trellis_now_ms();
        status = sparse_linear_device_cublas(
            in_dev,
            weights->from_latent_w,
            weights->from_latent_b,
            cur_h,
            n,
            weights->latent_channels,
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
    sparse_subm_conv3d_device cur_conv3d;
    memset(&cur_conv3d, 0, sizeof(cur_conv3d));
    if (status == TRELLIS_STATUS_OK) {
        const double t0 = trellis_now_ms();
        status = sparse_subm_conv3d_device_build(cur_coords_dev, cur_n, 3, 3, 3, 1, 1, 1, &cur_conv3d);
        trellis_profile_cuda_log("shape.initial_neighbor_map", t0);
    }
    char dump_name[128];
    for (int level = 0; status == TRELLIS_STATUS_OK && level < levels_to_run; ++level) {
        cur_channels = weights->channels[level];
        for (int block = 0; status == TRELLIS_STATUS_OK && block < weights->blocks_per_level[level]; ++block) {
            float * next_h = NULL;
            const double t0 = trellis_now_ms();
            status = shape_convnext_block_device(
                &weights->blocks[level][block],
                &cur_conv3d,
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

        int32_t * next_coords_dev = NULL;
        float * next_h = NULL;
        sparse_subm_conv3d_device next_conv3d;
        memset(&next_conv3d, 0, sizeof(next_conv3d));
        int64_t next_n = 0;
        const trellis_sparse_c2s_guide_level * guide_in = NULL;
        trellis_sparse_c2s_guide_level * guide_out = NULL;
        if (!weights->pred_subdiv) {
            if (guide_subs == NULL || guide_subs->n_levels <= level) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
                break;
            }
            guide_in = &guide_subs->levels[level];
        } else if (return_subs != NULL) {
            guide_out = &return_subs->levels[level];
        }
        const double t0 = trellis_now_ms();
        status = shape_c2s_block_device(
            &weights->up_blocks[level],
            cur_coords_dev,
            &cur_conv3d,
            cur_h,
            cur_n,
            &debug,
            level,
            guide_in,
            guide_out,
            &next_coords_dev,
            &next_h,
            &next_conv3d,
            &next_n);
        if (trellis_profile_enabled()) {
            char profile_name[64];
            snprintf(profile_name, sizeof(profile_name), "shape.l%02d.c2s", level);
            trellis_profile_cuda_log(profile_name, t0);
        }
        if (status == TRELLIS_STATUS_OK) {
            cudaFree(cur_coords_dev);
            cudaFree(cur_h);
            sparse_subm_conv3d_device_free(&cur_conv3d);
            cur_coords_dev = next_coords_dev;
            cur_h = next_h;
            cur_conv3d = next_conv3d;
            memset(&next_conv3d, 0, sizeof(next_conv3d));
            cur_n = next_n;
            cur_channels = weights->channels[level + 1];
            if (return_subs != NULL && weights->pred_subdiv && return_subs->n_levels < level + 1) {
                return_subs->n_levels = level + 1;
            }
        } else {
            sparse_subm_conv3d_device_free(&next_conv3d);
        }
    }

    if (status == TRELLIS_STATUS_OK && levels_to_run == levels) {
        float * norm = NULL;
        float * out = NULL;
        TRELLIS_INFO(
            "SparseUnetVaeDecoder: output head tokens=%lld channels=%d out=%d",
            (long long) cur_n,
            cur_channels,
            weights->out_channels);
        status = cuda_malloc_f32((size_t) cur_n * (size_t) cur_channels, &norm);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "SparseUnetVaeDecoder: output norm allocation failed tokens=%lld channels=%d",
                (long long) cur_n,
                cur_channels);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = cuda_malloc_f32((size_t) cur_n * (size_t) weights->out_channels, &out);
            if (status != TRELLIS_STATUS_OK) {
                TRELLIS_ERROR(
                    "SparseUnetVaeDecoder: output allocation failed tokens=%lld channels=%d",
                    (long long) cur_n,
                    weights->out_channels);
            }
        }
        if (status == TRELLIS_STATUS_OK) {
            status = row_layer_norm_device(cur_h, NULL, NULL, norm, cur_n, cur_channels, 1e-5f);
            if (status != TRELLIS_STATUS_OK) {
                TRELLIS_ERROR(
                    "SparseUnetVaeDecoder: output layer norm failed tokens=%lld channels=%d",
                    (long long) cur_n,
                    cur_channels);
            }
        }
        if (status == TRELLIS_STATUS_OK) {
            status = shape_debug_dump_device_f32(&debug, "output_norm", norm, cur_n, cur_channels);
        }
        if (status == TRELLIS_STATUS_OK) {
            const double t0 = trellis_now_ms();
            status = sparse_linear_device_cublas(norm, weights->output_w, weights->output_b, out, cur_n, cur_channels, weights->out_channels);
            trellis_profile_cuda_log("shape.output_linear", t0);
            if (status != TRELLIS_STATUS_OK) {
                TRELLIS_ERROR(
                    "SparseUnetVaeDecoder: output linear failed tokens=%lld in=%d out=%d",
                    (long long) cur_n,
                    cur_channels,
                    weights->out_channels);
            }
        }
        if (status == TRELLIS_STATUS_OK) {
            status = shape_debug_dump_device_f32(&debug, "output_logits", out, cur_n, weights->out_channels);
        }
        cudaFree(norm);
        if (status == TRELLIS_STATUS_OK) {
            cudaFree(cur_h);
            cur_h = out;
            cur_channels = weights->out_channels;
        } else {
            cudaFree(out);
        }
    }

    int32_t * host_coords = NULL;
    float * host_feats = NULL;
    if (status == TRELLIS_STATUS_OK) {
        status = cuda_status_to_trellis(cudaDeviceSynchronize());
    }
    if (status == TRELLIS_STATUS_OK) {
        host_coords = (int32_t *) malloc((size_t) cur_n * 4u * sizeof(int32_t));
        host_feats = (float *) malloc((size_t) cur_n * (size_t) cur_channels * sizeof(float));
        if (host_coords == NULL || host_feats == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        const double t0 = trellis_now_ms();
        status = cuda_status_to_trellis(cudaMemcpy(
            host_coords,
            cur_coords_dev,
            (size_t) cur_n * 4u * sizeof(int32_t),
            cudaMemcpyDeviceToHost));
        trellis_profile_cuda_log("shape.copy_coords_to_host", t0);
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
        *coords_out = host_coords;
        *feats_out = host_feats;
        *n_out = cur_n;
        *channels_out = cur_channels;
        host_coords = NULL;
        host_feats = NULL;
    }

    free(host_coords);
    free(host_feats);
    sparse_subm_conv3d_device_free(&cur_conv3d);
    cudaFree(cur_coords_dev);
    cudaFree(cur_h);
    if (debug.manifest != NULL) {
        fclose(debug.manifest);
    }
    if (status != TRELLIS_STATUS_OK && return_subs != NULL) {
        trellis_sparse_c2s_guides_free(return_subs);
    }
    return status;
}

extern "C" trellis_status trellis_sparse_unet_vae_decoder_forward_f32_host(
    const trellis_sparse_unet_vae_decoder_weights * weights,
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int device,
    int max_levels,
    const trellis_sparse_c2s_guides * guide_subs,
    trellis_sparse_c2s_guides * return_subs,
    int32_t ** coords_out,
    float ** feats_out,
    int64_t * n_out,
    int * channels_out) {
    return sparse_unet_vae_decoder_forward_f32_host_debug(
        weights,
        coords,
        feats,
        n,
        device,
        max_levels,
        guide_subs,
        return_subs,
        NULL,
        coords_out,
        feats_out,
        n_out,
        channels_out);
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
    return sparse_unet_vae_decoder_forward_f32_host_debug(
        weights,
        coords,
        feats,
        n,
        device,
        max_levels,
        NULL,
        NULL,
        debug_options,
        coords_out,
        feats_out,
        n_out,
        channels_out);
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
