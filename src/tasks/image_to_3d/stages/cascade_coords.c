#include "image_to_3d_internal.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int cmp_coord4_i32(const void * a, const void * b) {
    const int32_t * ca = (const int32_t *) a;
    const int32_t * cb = (const int32_t *) b;
    for (int i = 0; i < 4; ++i) {
        if (ca[i] < cb[i]) return -1;
        if (ca[i] > cb[i]) return 1;
    }
    return 0;
}

static int same_coord4_i32(const int32_t * a, const int32_t * b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

trellis_status trellis_pipeline_quantize_cascade_coords(
    const int32_t * decoder_coords,
    int64_t decoder_n,
    int lr_resolution,
    int hr_resolution,
    trellis_cascade_coord_quantization quantization,
    int32_t ** coords_out,
    int64_t * n_out) {
    if (coords_out != NULL) {
        *coords_out = NULL;
    }
    if (n_out != NULL) {
        *n_out = 0;
    }
    if (decoder_coords == NULL || decoder_n <= 0 || lr_resolution <= 0 ||
        hr_resolution <= 0 || coords_out == NULL || n_out == NULL ||
        (quantization != TRELLIS_CASCADE_COORD_QUANTIZE_TRELLIS &&
         quantization != TRELLIS_CASCADE_COORD_QUANTIZE_PIXAL) ||
        decoder_n > INT64_MAX / 4) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int hr_slat_edge = hr_resolution / 16;
    if (hr_slat_edge <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    int32_t * tmp = (int32_t *) malloc((size_t) decoder_n * 4u * sizeof(int32_t));
    if (tmp == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < decoder_n; ++i) {
        tmp[(size_t) i * 4u + 0u] = decoder_coords[(size_t) i * 4u + 0u];
        for (int axis = 0; axis < 3; ++axis) {
            const int32_t c = decoder_coords[(size_t) i * 4u + 1u + (size_t) axis];
            int q = (int) floorf((((float) c + 0.5f) / (float) lr_resolution) * (float) hr_slat_edge);
            if (quantization == TRELLIS_CASCADE_COORD_QUANTIZE_PIXAL) {
                q = (int) roundf(
                    (((float) c + 0.5f) / (float) lr_resolution) *
                    (float) (hr_slat_edge - 1));
            }
            if (q < 0) q = 0;
            if (q >= hr_slat_edge) q = hr_slat_edge - 1;
            tmp[(size_t) i * 4u + 1u + (size_t) axis] = (int32_t) q;
        }
    }
    qsort(tmp, (size_t) decoder_n, 4u * sizeof(int32_t), cmp_coord4_i32);
    int64_t unique_n = 0;
    for (int64_t i = 0; i < decoder_n; ++i) {
        if (i == 0 || !same_coord4_i32(&tmp[(size_t) i * 4u], &tmp[(size_t) (i - 1) * 4u])) {
            if (unique_n != i) {
                memcpy(&tmp[(size_t) unique_n * 4u], &tmp[(size_t) i * 4u], 4u * sizeof(int32_t));
            }
            ++unique_n;
        }
    }
    int32_t * shrunk = (int32_t *) realloc(tmp, (size_t) unique_n * 4u * sizeof(int32_t));
    if (shrunk != NULL) {
        tmp = shrunk;
    }
    *coords_out = tmp;
    *n_out = unique_n;
    return TRELLIS_STATUS_OK;
}
