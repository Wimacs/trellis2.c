#ifndef TRELLIS2_C_SPARSE_UNET_ENCODER_S2C_HOST_H
#define TRELLIS2_C_SPARSE_UNET_ENCODER_S2C_HOST_H

#include "trellis.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Builds the exact SparseSpatial2Channel(2) coordinate mapping used by
 * TRELLIS.2. All returned arrays are malloc-owned by the caller. Coarse
 * coordinates are sorted in (batch, x, y, z) order; parent and subidx stay
 * aligned with the original fine-coordinate row order. */
trellis_status trellis_sparse_s2c_host_build(
    const int32_t * fine_coords_bxyz,
    int64_t fine_n,
    int32_t ** coarse_coords_bxyz_out,
    int64_t * coarse_n_out,
    int32_t ** parent_out,
    int32_t ** subidx_out);

#ifdef __cplusplus
}
#endif

#endif
