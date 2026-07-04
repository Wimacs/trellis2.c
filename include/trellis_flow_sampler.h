#ifndef TRELLIS2_C_FLOW_SAMPLER_H
#define TRELLIS2_C_FLOW_SAMPLER_H

#include "trellis.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Euler update used by flow samplers to step from t to t_prev. */
void trellis_flow_euler_step_f32(
    const float * x_t,
    const float * pred_v,
    size_t n,
    float sigma_min,
    float t,
    float t_prev,
    float * pred_x_prev,
    float * pred_x0);

/* Classifier-free guidance blend of positive and negative predictions. */
void trellis_flow_cfg_combine_f32(
    const float * pred_pos,
    const float * pred_neg,
    size_t n,
    float guidance_strength,
    float * pred);

/* Classifier-free guidance blend with x0 standard-deviation rescaling. */
void trellis_flow_cfg_rescale_combine_f32(
    const float * x_t,
    const float * pred_pos,
    const float * pred_neg,
    size_t batch,
    size_t sample_stride,
    float sigma_min,
    float t,
    float guidance_strength,
    float guidance_rescale,
    float * pred);

/* Builds flow sampler timestep pairs after TRELLIS rescaling. */
trellis_status trellis_flow_timestep_pairs_f32(
    int steps,
    float rescale_t,
    float * pairs,
    size_t pair_count);

/* Sinusoidal timestep embedding used by debug/reference paths. */
void trellis_timestep_embedding_f32(
    const float * timesteps,
    size_t n_timesteps,
    int dim,
    float max_period,
    float * embedding);

/* Dense 3D rotary phase table for voxel-grid latent tokens. */
trellis_status trellis_rope_3d_phases_f32(
    int resolution,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count);

/* Sparse 3D rotary phase table for active voxel coordinates. */
trellis_status trellis_rope_3d_sparse_phases_f32(
    const int32_t * coords,
    int64_t n_coords,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count);

#ifdef __cplusplus
}
#endif

#endif
