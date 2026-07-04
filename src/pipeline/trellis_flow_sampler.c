#include "trellis.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

void trellis_flow_euler_step_f32(
    const float * x_t,
    const float * pred_v,
    size_t n,
    float sigma_min,
    float t,
    float t_prev,
    float * pred_x_prev,
    float * pred_x0) {
    const float sigma_t = sigma_min + (1.0f - sigma_min) * t;
    for (size_t i = 0; i < n; ++i) {
        pred_x_prev[i] = x_t[i] - (t - t_prev) * pred_v[i];
        pred_x0[i] = (1.0f - sigma_min) * x_t[i] - sigma_t * pred_v[i];
    }
}

void trellis_flow_cfg_combine_f32(
    const float * pred_pos,
    const float * pred_neg,
    size_t n,
    float guidance_strength,
    float * pred) {
    for (size_t i = 0; i < n; ++i) {
        pred[i] = guidance_strength * pred_pos[i] + (1.0f - guidance_strength) * pred_neg[i];
    }
}

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
    float * pred) {
    if (x_t == NULL || pred_pos == NULL || pred_neg == NULL || pred == NULL || sample_stride == 0) {
        return;
    }

    const float sigma_t = sigma_min + (1.0f - sigma_min) * t;
    const float one_minus_sigma_min = 1.0f - sigma_min;
    for (size_t b = 0; b < batch; ++b) {
        const size_t base = b * sample_stride;
        for (size_t i = 0; i < sample_stride; ++i) {
            const size_t j = base + i;
            pred[j] = guidance_strength * pred_pos[j] + (1.0f - guidance_strength) * pred_neg[j];
        }

        if (guidance_rescale <= 0.0f) {
            continue;
        }

        double mean_pos = 0.0;
        double mean_cfg = 0.0;
        for (size_t i = 0; i < sample_stride; ++i) {
            const size_t j = base + i;
            const float x0_pos = one_minus_sigma_min * x_t[j] - sigma_t * pred_pos[j];
            const float x0_cfg = one_minus_sigma_min * x_t[j] - sigma_t * pred[j];
            mean_pos += x0_pos;
            mean_cfg += x0_cfg;
        }
        mean_pos /= (double) sample_stride;
        mean_cfg /= (double) sample_stride;

        double var_pos = 0.0;
        double var_cfg = 0.0;
        for (size_t i = 0; i < sample_stride; ++i) {
            const size_t j = base + i;
            const double x0_pos = (double) (one_minus_sigma_min * x_t[j] - sigma_t * pred_pos[j]) - mean_pos;
            const double x0_cfg = (double) (one_minus_sigma_min * x_t[j] - sigma_t * pred[j]) - mean_cfg;
            var_pos += x0_pos * x0_pos;
            var_cfg += x0_cfg * x0_cfg;
        }

        const double denom = sample_stride > 1 ? (double) (sample_stride - 1) : (double) sample_stride;
        const double std_pos = sqrt(var_pos / denom);
        const double std_cfg = sqrt(var_cfg / denom);
        if (std_cfg <= 0.0) {
            continue;
        }

        const float ratio = (float) (std_pos / std_cfg);
        for (size_t i = 0; i < sample_stride; ++i) {
            const size_t j = base + i;
            const float x0_cfg = one_minus_sigma_min * x_t[j] - sigma_t * pred[j];
            const float x0_rescaled = x0_cfg * ratio;
            const float x0 = guidance_rescale * x0_rescaled + (1.0f - guidance_rescale) * x0_cfg;
            pred[j] = (one_minus_sigma_min * x_t[j] - x0) / sigma_t;
        }
    }
}

trellis_status trellis_flow_timestep_pairs_f32(
    int steps,
    float rescale_t,
    float * pairs,
    size_t pair_count) {
    if (steps <= 0 || rescale_t <= 0.0f || pairs == NULL || pair_count < (size_t) steps * 2u) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int i = 0; i < steps; ++i) {
        const float u0 = 1.0f - (float) i / (float) steps;
        const float u1 = 1.0f - (float) (i + 1) / (float) steps;
        pairs[2 * i + 0] = rescale_t * u0 / (1.0f + (rescale_t - 1.0f) * u0);
        pairs[2 * i + 1] = rescale_t * u1 / (1.0f + (rescale_t - 1.0f) * u1);
    }
    return TRELLIS_STATUS_OK;
}

void trellis_timestep_embedding_f32(
    const float * timesteps,
    size_t n_timesteps,
    int dim,
    float max_period,
    float * embedding) {
    if (timesteps == NULL || embedding == NULL || dim <= 0 || max_period <= 0.0f) {
        return;
    }
    const int half = dim / 2;
    for (size_t n = 0; n < n_timesteps; ++n) {
        float * out = embedding + n * (size_t) dim;
        for (int i = 0; i < half; ++i) {
            const float freq = expf(-logf(max_period) * (float) i / (float) half);
            const float arg = timesteps[n] * freq;
            out[i] = cosf(arg);
            out[half + i] = sinf(arg);
        }
        if ((dim & 1) != 0) {
            out[dim - 1] = 0.0f;
        }
    }
}

trellis_status trellis_rope_3d_phases_f32(
    int resolution,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count) {
    if (resolution <= 0 || head_dim <= 0 || (head_dim & 1) != 0 ||
        freq_scale <= 0.0f || freq_base <= 0.0f || cos_out == NULL || sin_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int half = head_dim / 2;
    const int freq_dim = half / 3;
    const size_t tokens = (size_t) resolution * (size_t) resolution * (size_t) resolution;
    if (phase_count < tokens * (size_t) half) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int x = 0; x < resolution; ++x) {
        for (int y = 0; y < resolution; ++y) {
            for (int z = 0; z < resolution; ++z) {
                const size_t token = ((size_t) x * (size_t) resolution + (size_t) y) * (size_t) resolution + (size_t) z;
                const int indices[3] = {x, y, z};
                float * cos_row = cos_out + token * (size_t) half;
                float * sin_row = sin_out + token * (size_t) half;
                int k = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    for (int f = 0; f < freq_dim; ++f) {
                        const float power = (float) f / (float) freq_dim;
                        const float freq = freq_scale / powf(freq_base, power);
                        const float phase = (float) indices[axis] * freq;
                        cos_row[k] = cosf(phase);
                        sin_row[k] = sinf(phase);
                        ++k;
                    }
                }
                while (k < half) {
                    cos_row[k] = 1.0f;
                    sin_row[k] = 0.0f;
                    ++k;
                }
            }
        }
    }
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_rope_3d_sparse_phases_f32(
    const int32_t * coords,
    int64_t n_coords,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count) {
    if (coords == NULL || n_coords < 0 || head_dim <= 0 || (head_dim & 1) != 0 ||
        freq_scale <= 0.0f || freq_base <= 0.0f || cos_out == NULL || sin_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int half = head_dim / 2;
    const int freq_dim = half / 3;
    if (phase_count < (size_t) n_coords * (size_t) half) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int64_t token = 0; token < n_coords; ++token) {
        const int32_t * row = coords + token * 4;
        const int indices[3] = { row[1], row[2], row[3] };
        float * cos_row = cos_out + (size_t) token * (size_t) half;
        float * sin_row = sin_out + (size_t) token * (size_t) half;
        int k = 0;
        for (int axis = 0; axis < 3; ++axis) {
            for (int f = 0; f < freq_dim; ++f) {
                const float power = (float) f / (float) freq_dim;
                const float freq = freq_scale / powf(freq_base, power);
                const float phase = (float) indices[axis] * freq;
                cos_row[k] = cosf(phase);
                sin_row[k] = sinf(phase);
                ++k;
            }
        }
        while (k < half) {
            cos_row[k] = 1.0f;
            sin_row[k] = 0.0f;
            ++k;
        }
    }
    return TRELLIS_STATUS_OK;
}
