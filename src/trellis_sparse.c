#include "trellis.h"

#include <stdlib.h>
#include <string.h>

static int coord_cmp4(const int32_t a[4], const int32_t b[4]) {
    for (int i = 0; i < 4; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

static int find_coord(const int32_t * coords, int64_t n, const int32_t key[4]) {
    for (int64_t i = 0; i < n; ++i) {
        if (coord_cmp4(&coords[4 * i], key) == 0) {
            return (int) i;
        }
    }
    return -1;
}

static trellis_status alloc_sparse(trellis_sparse_tensor_host * t, int64_t n, int64_t channels) {
    memset(t, 0, sizeof(*t));
    t->n = n;
    t->channels = channels;
    if (n == 0 || channels == 0) {
        return TRELLIS_STATUS_OK;
    }
    t->coords = (int32_t *) calloc((size_t) n * 4, sizeof(int32_t));
    t->feats = (float *) calloc((size_t) n * (size_t) channels, sizeof(float));
    if (t->coords == NULL || t->feats == NULL) {
        trellis_sparse_tensor_free(t);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    return TRELLIS_STATUS_OK;
}

static void sort_sparse(trellis_sparse_tensor_host * t) {
    if (t == NULL || t->n <= 1) {
        return;
    }
    float * tmp_feat = (float *) malloc((size_t) t->channels * sizeof(float));
    if (tmp_feat == NULL) {
        return;
    }
    for (int64_t i = 1; i < t->n; ++i) {
        int32_t key_coord[4];
        memcpy(key_coord, &t->coords[4 * i], sizeof(key_coord));
        memcpy(tmp_feat, &t->feats[i * t->channels], (size_t) t->channels * sizeof(float));

        int64_t j = i - 1;
        while (j >= 0 && coord_cmp4(&t->coords[4 * j], key_coord) > 0) {
            memcpy(&t->coords[4 * (j + 1)], &t->coords[4 * j], 4 * sizeof(int32_t));
            memcpy(&t->feats[(j + 1) * t->channels], &t->feats[j * t->channels], (size_t) t->channels * sizeof(float));
            --j;
        }
        memcpy(&t->coords[4 * (j + 1)], key_coord, sizeof(key_coord));
        memcpy(&t->feats[(j + 1) * t->channels], tmp_feat, (size_t) t->channels * sizeof(float));
    }
    free(tmp_feat);
}

void trellis_sparse_tensor_free(trellis_sparse_tensor_host * tensor) {
    if (tensor == NULL) {
        return;
    }
    free(tensor->coords);
    free(tensor->feats);
    memset(tensor, 0, sizeof(*tensor));
}

trellis_status trellis_sparse_downsample_mean_host(
    const trellis_sparse_tensor_host * input,
    int factor,
    trellis_sparse_tensor_host * output) {
    if (input == NULL || output == NULL || factor <= 0 || input->channels <= 0 || input->n < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_sparse_tensor_host tmp;
    trellis_status status = alloc_sparse(&tmp, input->n, input->channels);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    int64_t * counts = (int64_t *) calloc((size_t) input->n, sizeof(int64_t));
    if (counts == NULL) {
        trellis_sparse_tensor_free(&tmp);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    int64_t groups = 0;
    for (int64_t i = 0; i < input->n; ++i) {
        int32_t key[4] = {
            input->coords[4 * i + 0],
            input->coords[4 * i + 1] / factor,
            input->coords[4 * i + 2] / factor,
            input->coords[4 * i + 3] / factor,
        };
        int idx = find_coord(tmp.coords, groups, key);
        if (idx < 0) {
            idx = (int) groups++;
            memcpy(&tmp.coords[4 * idx], key, sizeof(key));
        }
        counts[idx] += 1;
        for (int64_t c = 0; c < input->channels; ++c) {
            tmp.feats[(int64_t) idx * input->channels + c] += input->feats[i * input->channels + c];
        }
    }

    for (int64_t i = 0; i < groups; ++i) {
        for (int64_t c = 0; c < input->channels; ++c) {
            tmp.feats[i * input->channels + c] /= (float) counts[i];
        }
    }
    free(counts);

    trellis_sparse_tensor_host shrunk;
    status = alloc_sparse(&shrunk, groups, input->channels);
    if (status != TRELLIS_STATUS_OK) {
        trellis_sparse_tensor_free(&tmp);
        return status;
    }
    memcpy(shrunk.coords, tmp.coords, (size_t) groups * 4 * sizeof(int32_t));
    memcpy(shrunk.feats, tmp.feats, (size_t) groups * (size_t) input->channels * sizeof(float));
    trellis_sparse_tensor_free(&tmp);
    sort_sparse(&shrunk);
    *output = shrunk;
    return TRELLIS_STATUS_OK;
}

static int pow_int(int base, int exp) {
    int v = 1;
    for (int i = 0; i < exp; ++i) {
        v *= base;
    }
    return v;
}

trellis_status trellis_sparse_spatial2channel_host(
    const trellis_sparse_tensor_host * input,
    int factor,
    trellis_sparse_tensor_host * output) {
    if (input == NULL || output == NULL || factor <= 0 || input->channels <= 0 || input->n < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int slots = pow_int(factor, 3);

    trellis_sparse_tensor_host tmp;
    trellis_status status = alloc_sparse(&tmp, input->n, input->channels * slots);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    int64_t groups = 0;
    for (int64_t i = 0; i < input->n; ++i) {
        int32_t key[4] = {
            input->coords[4 * i + 0],
            input->coords[4 * i + 1] / factor,
            input->coords[4 * i + 2] / factor,
            input->coords[4 * i + 3] / factor,
        };
        int idx = find_coord(tmp.coords, groups, key);
        if (idx < 0) {
            idx = (int) groups++;
            memcpy(&tmp.coords[4 * idx], key, sizeof(key));
        }
        int sx = input->coords[4 * i + 1] % factor;
        int sy = input->coords[4 * i + 2] % factor;
        int sz = input->coords[4 * i + 3] % factor;
        if (sx < 0) sx += factor;
        if (sy < 0) sy += factor;
        if (sz < 0) sz += factor;
        const int subidx = sx + sy * factor + sz * factor * factor;
        for (int64_t c = 0; c < input->channels; ++c) {
            tmp.feats[(int64_t) idx * tmp.channels + (int64_t) subidx * input->channels + c] =
                input->feats[i * input->channels + c];
        }
    }

    trellis_sparse_tensor_host shrunk;
    status = alloc_sparse(&shrunk, groups, input->channels * slots);
    if (status != TRELLIS_STATUS_OK) {
        trellis_sparse_tensor_free(&tmp);
        return status;
    }
    memcpy(shrunk.coords, tmp.coords, (size_t) groups * 4 * sizeof(int32_t));
    memcpy(shrunk.feats, tmp.feats, (size_t) groups * (size_t) tmp.channels * sizeof(float));
    trellis_sparse_tensor_free(&tmp);
    sort_sparse(&shrunk);
    *output = shrunk;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_sparse_channel2spatial_host(
    const trellis_sparse_tensor_host * input,
    const uint8_t * subdivision,
    int factor,
    trellis_sparse_tensor_host * output) {
    if (input == NULL || output == NULL || subdivision == NULL || factor <= 0 ||
        input->channels <= 0 || input->n < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int slots = pow_int(factor, 3);
    if (input->channels % slots != 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int64_t out_channels = input->channels / slots;
    int64_t n_out = 0;
    for (int64_t i = 0; i < input->n; ++i) {
        for (int s = 0; s < slots; ++s) {
            if (subdivision[i * slots + s]) {
                ++n_out;
            }
        }
    }

    trellis_sparse_tensor_host out;
    trellis_status status = alloc_sparse(&out, n_out, out_channels);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    int64_t row = 0;
    for (int64_t i = 0; i < input->n; ++i) {
        for (int s = 0; s < slots; ++s) {
            if (!subdivision[i * slots + s]) {
                continue;
            }
            const int sx = s % factor;
            const int sy = (s / factor) % factor;
            const int sz = (s / (factor * factor)) % factor;
            out.coords[4 * row + 0] = input->coords[4 * i + 0];
            out.coords[4 * row + 1] = input->coords[4 * i + 1] * factor + sx;
            out.coords[4 * row + 2] = input->coords[4 * i + 2] * factor + sy;
            out.coords[4 * row + 3] = input->coords[4 * i + 3] * factor + sz;
            for (int64_t c = 0; c < out_channels; ++c) {
                out.feats[row * out_channels + c] = input->feats[i * input->channels + (int64_t) s * out_channels + c];
            }
            ++row;
        }
    }
    sort_sparse(&out);
    *output = out;
    return TRELLIS_STATUS_OK;
}
