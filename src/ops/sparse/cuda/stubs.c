#include "trellis.h"

#include <stdint.h>

trellis_status trellis_sparse_unet_vae_decoder_forward_f32_host(
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
    (void) weights;
    (void) coords;
    (void) feats;
    (void) n;
    (void) device;
    (void) max_levels;
    (void) guide_subs;
    (void) return_subs;
    if (coords_out != NULL) *coords_out = NULL;
    if (feats_out != NULL) *feats_out = NULL;
    if (n_out != NULL) *n_out = 0;
    if (channels_out != NULL) *channels_out = 0;
    return TRELLIS_STATUS_CUDA_UNAVAILABLE;
}

trellis_status trellis_shape_decoder_forward_f32_host_debug(
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
    (void) weights;
    (void) coords;
    (void) feats;
    (void) n;
    (void) device;
    (void) max_levels;
    (void) debug_options;
    if (coords_out != NULL) *coords_out = NULL;
    if (feats_out != NULL) *feats_out = NULL;
    if (n_out != NULL) *n_out = 0;
    if (channels_out != NULL) *channels_out = 0;
    return TRELLIS_STATUS_CUDA_UNAVAILABLE;
}

trellis_status trellis_shape_decoder_forward_f32_host(
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
