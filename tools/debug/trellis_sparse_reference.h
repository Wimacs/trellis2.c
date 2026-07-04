#ifndef TRELLIS2_C_TOOLS_DEBUG_TRELLIS_SPARSE_REFERENCE_H
#define TRELLIS2_C_TOOLS_DEBUG_TRELLIS_SPARSE_REFERENCE_H

#include "trellis.h"

typedef struct trellis_sparse_tensor_host {
    int32_t * coords; /* [n, 4] = batch, x, y, z */
    float * feats;    /* [n, channels] */
    int64_t n;
    int64_t channels;
} trellis_sparse_tensor_host;

void trellis_sparse_tensor_free(trellis_sparse_tensor_host * tensor);

trellis_status trellis_sparse_downsample_mean_host(
    const trellis_sparse_tensor_host * input,
    int factor,
    trellis_sparse_tensor_host * output);

trellis_status trellis_sparse_spatial2channel_host(
    const trellis_sparse_tensor_host * input,
    int factor,
    trellis_sparse_tensor_host * output);

trellis_status trellis_sparse_channel2spatial_host(
    const trellis_sparse_tensor_host * input,
    const uint8_t * subdivision,
    int factor,
    trellis_sparse_tensor_host * output);

#endif
