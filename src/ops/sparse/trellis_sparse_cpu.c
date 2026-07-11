#include "trellis_sparse_backend.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct trellis_sparse_buffer {
    void * data;
    size_t bytes;
};

struct trellis_sparse_rulebook {
    int32_t * coords;
    int64_t n;
    uint64_t * keys;
    int32_t * values;
    int64_t table_mask;
};

typedef struct trellis_sparse_cpu_backend {
    trellis_sparse_backend base;
} trellis_sparse_cpu_backend;

static uint64_t sparse_pack_coord4(int b, int x, int y, int z) {
    uint64_t raw =
        ((uint64_t) ((uint32_t) b & 0xffffu) << 48) |
        ((uint64_t) ((uint32_t) x & 0xffffu) << 32) |
        ((uint64_t) ((uint32_t) y & 0xffffu) << 16) |
        ((uint64_t) ((uint32_t) z & 0xffffu));
    return raw + 1ull;
}

static uint64_t sparse_hash_u64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

static int64_t sparse_next_power_of_two(int64_t x) {
    int64_t v = 1;
    while (v < x && v < INT64_MAX / 2) {
        v <<= 1;
    }
    return v;
}

static void cpu_destroy(trellis_sparse_backend * backend) {
    free(backend);
}

static trellis_status cpu_alloc_raw(size_t bytes, trellis_sparse_buffer ** out) {
    if (out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    trellis_sparse_buffer * b = (trellis_sparse_buffer *) calloc(1, sizeof(*b));
    if (b == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    b->data = bytes == 0 ? NULL : calloc(1, bytes);
    if (bytes != 0 && b->data == NULL) {
        free(b);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    b->bytes = bytes;
    *out = b;
    return TRELLIS_STATUS_OK;
}

static trellis_status cpu_alloc_f32(trellis_sparse_backend * backend, size_t count, trellis_sparse_buffer ** out) {
    (void) backend;
    if (count > SIZE_MAX / sizeof(float)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    return cpu_alloc_raw(count * sizeof(float), out);
}

static trellis_status cpu_upload_f32(trellis_sparse_backend * backend, const float * src, size_t count, trellis_sparse_buffer ** out) {
    (void) backend;
    if (src == NULL && count != 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_status status = cpu_alloc_f32(backend, count, out);
    if (status == TRELLIS_STATUS_OK && count != 0) {
        memcpy((*out)->data, src, count * sizeof(float));
    }
    return status;
}

static trellis_status cpu_upload_i32(trellis_sparse_backend * backend, const int32_t * src, size_t count, trellis_sparse_buffer ** out) {
    (void) backend;
    if ((src == NULL && count != 0) || count > SIZE_MAX / sizeof(int32_t)) {
        return src == NULL ? TRELLIS_STATUS_INVALID_ARGUMENT : TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    trellis_status status = cpu_alloc_raw(count * sizeof(int32_t), out);
    if (status == TRELLIS_STATUS_OK && count != 0) {
        memcpy((*out)->data, src, count * sizeof(int32_t));
    }
    return status;
}

static trellis_status cpu_download_f32(trellis_sparse_backend * backend, const trellis_sparse_buffer * src, float * dst, size_t count) {
    (void) backend;
    if (src == NULL || dst == NULL || count > SIZE_MAX / sizeof(float) || src->bytes < count * sizeof(float)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memcpy(dst, src->data, count * sizeof(float));
    return TRELLIS_STATUS_OK;
}

static void cpu_free_buffer(trellis_sparse_backend * backend, trellis_sparse_buffer * buffer) {
    (void) backend;
    if (buffer != NULL) {
        free(buffer->data);
        free(buffer);
    }
}

static trellis_status cpu_linear(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const float * weight,
    const float * bias,
    trellis_sparse_buffer * y,
    int64_t n,
    int in_channels,
    int out_channels) {
    (void) backend;
    if (x == NULL || x->data == NULL || weight == NULL || y == NULL || y->data == NULL ||
        n <= 0 || in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const float * xf = (const float *) x->data;
    float * yf = (float *) y->data;
    for (int64_t row = 0; row < n; ++row) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float acc = bias == NULL ? 0.0f : bias[oc];
            const float * xr = xf + row * (int64_t) in_channels;
            const float * wr = weight + (int64_t) oc * in_channels;
            for (int ic = 0; ic < in_channels; ++ic) {
                acc += xr[ic] * wr[ic];
            }
            yf[row * (int64_t) out_channels + oc] = acc;
        }
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status cpu_linear_silu(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const float * weight,
    const float * bias,
    trellis_sparse_buffer * y,
    int64_t n,
    int in_channels,
    int out_channels) {
    trellis_status status = cpu_linear(backend, x, weight, bias, y, n, in_channels, out_channels);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    float * yf = (float *) y->data;
    const size_t count = (size_t) n * (size_t) out_channels;
    for (size_t i = 0; i < count; ++i) {
        yf[i] = yf[i] / (1.0f + expf(-yf[i]));
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status cpu_row_norm(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const float * gamma,
    const float * beta,
    trellis_sparse_buffer * y,
    int64_t n,
    int channels,
    float eps) {
    (void) backend;
    if (x == NULL || x->data == NULL || y == NULL || y->data == NULL ||
        n <= 0 || channels <= 0 || eps <= 0.0f) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const float * xf = (const float *) x->data;
    float * yf = (float *) y->data;
    for (int64_t row = 0; row < n; ++row) {
        const float * xr = xf + row * (int64_t) channels;
        float mean = 0.0f;
        for (int c = 0; c < channels; ++c) {
            mean += xr[c];
        }
        mean /= (float) channels;
        float var = 0.0f;
        for (int c = 0; c < channels; ++c) {
            const float d = xr[c] - mean;
            var += d * d;
        }
        const float inv_std = 1.0f / sqrtf(var / (float) channels + eps);
        float * yr = yf + row * (int64_t) channels;
        for (int c = 0; c < channels; ++c) {
            float v = (xr[c] - mean) * inv_std;
            if (gamma != NULL) v *= gamma[c];
            if (beta != NULL) v += beta[c];
            yr[c] = v;
        }
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status cpu_row_norm_silu(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const float * gamma,
    const float * beta,
    trellis_sparse_buffer * y,
    int64_t n,
    int channels,
    float eps) {
    trellis_status status = cpu_row_norm(backend, x, gamma, beta, y, n, channels, eps);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    float * yf = (float *) y->data;
    const size_t count = (size_t) n * (size_t) channels;
    for (size_t i = 0; i < count; ++i) {
        yf[i] = yf[i] / (1.0f + expf(-yf[i]));
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status cpu_silu_inplace(trellis_sparse_backend * backend, trellis_sparse_buffer * x, size_t count) {
    (void) backend;
    if (x == NULL || x->data == NULL || x->bytes < count * sizeof(float)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    float * xf = (float *) x->data;
    for (size_t i = 0; i < count; ++i) {
        xf[i] = xf[i] / (1.0f + expf(-xf[i]));
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status cpu_add(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * a,
    const trellis_sparse_buffer * b,
    trellis_sparse_buffer * y,
    size_t count) {
    (void) backend;
    if (a == NULL || b == NULL || y == NULL || a->data == NULL || b->data == NULL || y->data == NULL ||
        a->bytes < count * sizeof(float) || b->bytes < count * sizeof(float) || y->bytes < count * sizeof(float)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const float * af = (const float *) a->data;
    const float * bf = (const float *) b->data;
    float * yf = (float *) y->data;
    for (size_t i = 0; i < count; ++i) {
        yf[i] = af[i] + bf[i];
    }
    return TRELLIS_STATUS_OK;
}

static int cpu_rulebook_find(const trellis_sparse_rulebook * r, int b, int x, int y, int z) {
    const uint64_t key = sparse_pack_coord4(b, x, y, z);
    uint64_t slot = sparse_hash_u64(key) & (uint64_t) r->table_mask;
    for (int64_t probe = 0; probe <= r->table_mask; ++probe) {
        const uint64_t found = r->keys[slot];
        if (found == key) {
            return r->values[slot];
        }
        if (found == 0) {
            return -1;
        }
        slot = (slot + 1u) & (uint64_t) r->table_mask;
    }
    return -1;
}

static trellis_status cpu_build_rulebook(
    trellis_sparse_backend * backend,
    const int32_t * coords_bxyz,
    int64_t n,
    trellis_sparse_rulebook ** out) {
    (void) backend;
    if (coords_bxyz == NULL || n <= 0 || out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    trellis_sparse_rulebook * r = (trellis_sparse_rulebook *) calloc(1, sizeof(*r));
    if (r == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    r->coords = (int32_t *) malloc((size_t) n * 4u * sizeof(int32_t));
    const int64_t table_size = sparse_next_power_of_two(n * 2);
    r->keys = (uint64_t *) calloc((size_t) table_size, sizeof(uint64_t));
    r->values = (int32_t *) malloc((size_t) table_size * sizeof(int32_t));
    if (r->coords == NULL || r->keys == NULL || r->values == NULL) {
        free(r->coords);
        free(r->keys);
        free(r->values);
        free(r);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    memcpy(r->coords, coords_bxyz, (size_t) n * 4u * sizeof(int32_t));
    r->n = n;
    r->table_mask = table_size - 1;
    for (int64_t row = 0; row < n; ++row) {
        const int32_t * c = coords_bxyz + row * 4;
        const uint64_t key = sparse_pack_coord4(c[0], c[1], c[2], c[3]);
        uint64_t slot = sparse_hash_u64(key) & (uint64_t) r->table_mask;
        while (r->keys[slot] != 0 && r->keys[slot] != key) {
            slot = (slot + 1u) & (uint64_t) r->table_mask;
        }
        r->keys[slot] = key;
        r->values[slot] = (int32_t) row;
    }
    *out = r;
    return TRELLIS_STATUS_OK;
}

static trellis_status cpu_sparse_conv3d(
    trellis_sparse_backend * backend,
    const trellis_sparse_rulebook * rulebook,
    const trellis_sparse_buffer * feats,
    const float * weight,
    const float * bias,
    trellis_sparse_buffer * out,
    int64_t n,
    int in_channels,
    int out_channels) {
    (void) backend;
    if (rulebook == NULL || feats == NULL || feats->data == NULL || weight == NULL ||
        out == NULL || out->data == NULL || n <= 0 || in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const float * xf = (const float *) feats->data;
    float * yf = (float *) out->data;
    for (int64_t row = 0; row < n; ++row) {
        for (int oc = 0; oc < out_channels; ++oc) {
            yf[row * (int64_t) out_channels + oc] = bias == NULL ? 0.0f : bias[oc];
        }
    }
    for (int64_t row = 0; row < n; ++row) {
        const int32_t * c = rulebook->coords + row * 4;
        for (int kd = 0; kd < 3; ++kd) {
            const int nx = c[1] + kd - 1;
            for (int kh = 0; kh < 3; ++kh) {
                const int ny = c[2] + kh - 1;
                for (int kw = 0; kw < 3; ++kw) {
                    const int nz = c[3] + kw - 1;
                    const int src = cpu_rulebook_find(rulebook, c[0], nx, ny, nz);
                    if (src < 0) {
                        continue;
                    }
                    const int offset = (kd * 3 + kh) * 3 + kw;
                    const float * xr = xf + (int64_t) src * in_channels;
                    for (int oc = 0; oc < out_channels; ++oc) {
                        const float * wr = weight + ((int64_t) oc * 27 + offset) * (int64_t) in_channels;
                        float acc = 0.0f;
                        for (int ic = 0; ic < in_channels; ++ic) {
                            acc += xr[ic] * wr[ic];
                        }
                        yf[row * (int64_t) out_channels + oc] += acc;
                    }
                }
            }
        }
    }
    return TRELLIS_STATUS_OK;
}

static void cpu_free_rulebook(trellis_sparse_backend * backend, trellis_sparse_rulebook * rulebook) {
    (void) backend;
    if (rulebook != NULL) {
        free(rulebook->coords);
        free(rulebook->keys);
        free(rulebook->values);
        free(rulebook);
    }
}

static trellis_status cpu_c2s_gather(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const int32_t * parent,
    const int32_t * subidx,
    trellis_sparse_buffer * y,
    int64_t n_out,
    int out_channels) {
    (void) backend;
    if (x == NULL || y == NULL || x->data == NULL || y->data == NULL ||
        parent == NULL || subidx == NULL || n_out <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const float * xf = (const float *) x->data;
    float * yf = (float *) y->data;
    for (int64_t row = 0; row < n_out; ++row) {
        const int p = parent[row];
        const int s = subidx[row];
        memcpy(
            yf + row * (int64_t) out_channels,
            xf + ((int64_t) p * 8 + s) * (int64_t) out_channels,
            (size_t) out_channels * sizeof(float));
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status cpu_skip_repeat(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const int32_t * parent,
    const int32_t * subidx,
    trellis_sparse_buffer * y,
    int64_t n_out,
    int in_channels,
    int out_channels) {
    (void) backend;
    if (x == NULL || y == NULL || x->data == NULL || y->data == NULL ||
        parent == NULL || subidx == NULL || n_out <= 0 || in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int repeat = (8 * out_channels + in_channels - 1) / in_channels;
    const float * xf = (const float *) x->data;
    float * yf = (float *) y->data;
    for (int64_t row = 0; row < n_out; ++row) {
        const int p = parent[row];
        const int s = subidx[row];
        for (int c = 0; c < out_channels; ++c) {
            int ic = (s * out_channels + c) / repeat;
            if (ic >= in_channels) {
                ic = in_channels - 1;
            }
            yf[row * (int64_t) out_channels + c] = xf[(int64_t) p * in_channels + ic];
        }
    }
    return TRELLIS_STATUS_OK;
}

static const trellis_sparse_backend_ops g_cpu_ops = {
    cpu_destroy,
    cpu_alloc_f32,
    cpu_upload_f32,
    cpu_upload_i32,
    cpu_download_f32,
    cpu_free_buffer,
    cpu_linear,
    cpu_linear_silu,
    cpu_row_norm,
    cpu_row_norm_silu,
    cpu_silu_inplace,
    cpu_add,
    cpu_build_rulebook,
    cpu_sparse_conv3d,
    cpu_free_rulebook,
    cpu_c2s_gather,
    cpu_skip_repeat,
    NULL,
    NULL,
};

trellis_status trellis_sparse_cpu_backend_create(trellis_sparse_backend ** out) {
    if (out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    trellis_sparse_cpu_backend * cpu = (trellis_sparse_cpu_backend *) calloc(1, sizeof(*cpu));
    if (cpu == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    cpu->base.ops = &g_cpu_ops;
    cpu->base.kind = TRELLIS_SPARSE_BACKEND_CPU;
    cpu->base.device = -1;
    *out = &cpu->base;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_sparse_backend_trim(
    trellis_sparse_backend * backend,
    unsigned flags,
    size_t * released_bytes) {
    if (released_bytes != NULL) {
        *released_bytes = 0;
    }
    if (backend == NULL || backend->ops == NULL ||
        (flags & ~(unsigned) TRELLIS_SPARSE_TRIM_ALL) != 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (flags == 0 || backend->ops->trim == NULL) {
        return TRELLIS_STATUS_OK;
    }
    return backend->ops->trim(backend, flags, released_bytes);
}

void trellis_sparse_backend_destroy(trellis_sparse_backend * backend) {
    if (backend != NULL && backend->ops != NULL && backend->ops->destroy != NULL) {
        backend->ops->destroy(backend);
    }
}
