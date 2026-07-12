#include "s2c_host.h"

#include <limits.h>
#include <stdlib.h>

typedef struct trellis_sparse_s2c_sort_entry {
    int32_t b;
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t fine_row;
    int32_t subidx;
} trellis_sparse_s2c_sort_entry;

static int s2c_entry_compare(const void * lhs_ptr, const void * rhs_ptr) {
    const trellis_sparse_s2c_sort_entry * lhs = (const trellis_sparse_s2c_sort_entry *) lhs_ptr;
    const trellis_sparse_s2c_sort_entry * rhs = (const trellis_sparse_s2c_sort_entry *) rhs_ptr;
    if (lhs->b != rhs->b) return lhs->b < rhs->b ? -1 : 1;
    if (lhs->x != rhs->x) return lhs->x < rhs->x ? -1 : 1;
    if (lhs->y != rhs->y) return lhs->y < rhs->y ? -1 : 1;
    if (lhs->z != rhs->z) return lhs->z < rhs->z ? -1 : 1;
    if (lhs->fine_row != rhs->fine_row) return lhs->fine_row < rhs->fine_row ? -1 : 1;
    return 0;
}

static int s2c_same_coarse(
    const trellis_sparse_s2c_sort_entry * lhs,
    const trellis_sparse_s2c_sort_entry * rhs) {
    return lhs->b == rhs->b && lhs->x == rhs->x && lhs->y == rhs->y && lhs->z == rhs->z;
}

trellis_status trellis_sparse_s2c_host_build(
    const int32_t * fine_coords_bxyz,
    int64_t fine_n,
    int32_t ** coarse_coords_bxyz_out,
    int64_t * coarse_n_out,
    int32_t ** parent_out,
    int32_t ** subidx_out) {
    if (fine_coords_bxyz == NULL || coarse_coords_bxyz_out == NULL || coarse_n_out == NULL ||
        parent_out == NULL || subidx_out == NULL || fine_n <= 0 || fine_n > INT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *coarse_coords_bxyz_out = NULL;
    *coarse_n_out = 0;
    *parent_out = NULL;
    *subidx_out = NULL;

    trellis_sparse_s2c_sort_entry * entries = (trellis_sparse_s2c_sort_entry *) malloc(
        (size_t) fine_n * sizeof(trellis_sparse_s2c_sort_entry));
    int32_t * parent = (int32_t *) malloc((size_t) fine_n * sizeof(int32_t));
    int32_t * subidx = (int32_t *) malloc((size_t) fine_n * sizeof(int32_t));
    if (entries == NULL || parent == NULL || subidx == NULL) {
        free(entries);
        free(parent);
        free(subidx);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    for (int64_t row = 0; row < fine_n; ++row) {
        const int32_t * coord = fine_coords_bxyz + row * 4;
        if (coord[0] < 0 || coord[1] < 0 || coord[2] < 0 || coord[3] < 0) {
            free(entries);
            free(parent);
            free(subidx);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        entries[row].b = coord[0];
        entries[row].x = coord[1] / 2;
        entries[row].y = coord[2] / 2;
        entries[row].z = coord[3] / 2;
        entries[row].fine_row = (int32_t) row;
        entries[row].subidx =
            (coord[1] & 1) |
            ((coord[2] & 1) << 1) |
            ((coord[3] & 1) << 2);
    }
    qsort(entries, (size_t) fine_n, sizeof(*entries), s2c_entry_compare);

    int64_t coarse_n = 1;
    for (int64_t i = 1; i < fine_n; ++i) {
        if (s2c_same_coarse(&entries[i - 1], &entries[i]) &&
            entries[i - 1].subidx == entries[i].subidx) {
            free(entries);
            free(parent);
            free(subidx);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        if (!s2c_same_coarse(&entries[i - 1], &entries[i])) {
            ++coarse_n;
        }
    }

    int32_t * coarse = (int32_t *) malloc((size_t) coarse_n * 4u * sizeof(int32_t));
    if (coarse == NULL) {
        free(entries);
        free(parent);
        free(subidx);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    int64_t coarse_row = -1;
    for (int64_t i = 0; i < fine_n; ++i) {
        if (i == 0 || !s2c_same_coarse(&entries[i - 1], &entries[i])) {
            ++coarse_row;
            coarse[coarse_row * 4 + 0] = entries[i].b;
            coarse[coarse_row * 4 + 1] = entries[i].x;
            coarse[coarse_row * 4 + 2] = entries[i].y;
            coarse[coarse_row * 4 + 3] = entries[i].z;
        }
        parent[entries[i].fine_row] = (int32_t) coarse_row;
        subidx[entries[i].fine_row] = entries[i].subidx;
    }
    free(entries);

    *coarse_coords_bxyz_out = coarse;
    *coarse_n_out = coarse_n;
    *parent_out = parent;
    *subidx_out = subidx;
    return TRELLIS_STATUS_OK;
}
