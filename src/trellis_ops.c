#include "trellis.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static struct ggml_tensor * trellis_repeat_or_null(
    struct ggml_context * ctx,
    struct ggml_tensor * src,
    struct ggml_tensor * like) {
    return src == NULL ? NULL : ggml_repeat(ctx, src, like);
}

static int g_trellis_ggml_use_flash_attn = 0;

void trellis_ggml_set_flash_attn_enabled(int enabled) {
    g_trellis_ggml_use_flash_attn = enabled ? 1 : 0;
}

struct ggml_tensor * trellis_ggml_linear(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * weight,
    struct ggml_tensor * bias) {
    struct ggml_tensor * y = ggml_mul_mat(ctx, weight, x);
    if (bias != NULL) {
        y = ggml_add(ctx, y, ggml_repeat(ctx, bias, y));
    }
    return y;
}

struct ggml_tensor * trellis_ggml_layer_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    struct ggml_tensor * beta,
    float eps) {
    struct ggml_tensor * y = ggml_norm(ctx, x, eps);
    if (gamma != NULL) {
        y = ggml_mul(ctx, y, ggml_repeat(ctx, gamma, y));
    }
    if (beta != NULL) {
        y = ggml_add(ctx, y, ggml_repeat(ctx, beta, y));
    }
    return y;
}

struct ggml_tensor * trellis_ggml_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    float eps) {
    struct ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    if (gamma != NULL) {
        y = ggml_mul(ctx, y, ggml_repeat(ctx, gamma, y));
    }
    return y;
}

struct ggml_tensor * trellis_ggml_bf16_roundtrip(
    struct ggml_context * ctx,
    struct ggml_tensor * x) {
    if (ctx == NULL || x == NULL) {
        return NULL;
    }
    return ggml_cast(ctx, ggml_cast(ctx, x, GGML_TYPE_BF16), GGML_TYPE_F32);
}

struct ggml_tensor * trellis_ggml_multihead_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    float eps) {
    struct ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    if (gamma == NULL) {
        return y;
    }
    struct ggml_tensor * g = gamma;
    if (x->ne[2] > 1 && gamma->ne[0] == x->ne[0] && gamma->ne[1] == x->ne[2]) {
        const size_t nb_head = gamma->nb[1];
        g = ggml_view_4d(
            ctx,
            gamma,
            x->ne[0],
            1,
            x->ne[2],
            1,
            nb_head,
            nb_head,
            nb_head * (size_t) gamma->ne[1],
            0);
    }
    return ggml_mul(ctx, y, ggml_repeat(ctx, g, y));
}

struct ggml_tensor * trellis_ggml_feed_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w1,
    struct ggml_tensor * b1,
    struct ggml_tensor * w2,
    struct ggml_tensor * b2) {
    struct ggml_tensor * h = trellis_ggml_linear(ctx, x, w1, b1);
    h = ggml_gelu(ctx, h);
    h = trellis_ggml_linear(ctx, h, w2, b2);
    return h;
}

struct ggml_tensor * trellis_ggml_timestep_mlp(
    struct ggml_context * ctx,
    struct ggml_tensor * timesteps,
    int frequency_dim,
    struct ggml_tensor * w1,
    struct ggml_tensor * b1,
    struct ggml_tensor * w2,
    struct ggml_tensor * b2) {
    struct ggml_tensor * emb = ggml_timestep_embedding(ctx, timesteps, frequency_dim, 10000);
    struct ggml_tensor * h = trellis_ggml_linear(ctx, emb, w1, b1);
    h = ggml_silu(ctx, h);
    h = trellis_ggml_linear(ctx, h, w2, b2);
    return h;
}

struct ggml_tensor * trellis_ggml_sdpa(
    struct ggml_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    float scale) {
    if (g_trellis_ggml_use_flash_attn) {
        struct ggml_tensor * h = ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0f, 0.0f);
        h = ggml_permute(ctx, h, 0, 2, 1, 3);
        return ggml_cont_4d(ctx, h, v->ne[0], q->ne[1], q->ne[2], q->ne[3]);
    }

    /* ggml's CUDA flash-attention path produces NaNs for the stage1 flow at
     * 512+ latent tokens in this build. Use explicit scaled dot-product
     * attention; it is heavier but keeps the pure C/ggml/CUDA path numerically
     * stable while larger-token kernels are tuned.
     */
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_scale(ctx, scores, scale);
    scores = ggml_soft_max(ctx, scores);
    struct ggml_tensor * v_t = ggml_permute(ctx, v, 1, 0, 2, 3);
    v_t = ggml_cont_4d(ctx, v_t, v->ne[1], v->ne[0], v->ne[2], v->ne[3]);
    return ggml_mul_mat(ctx, v_t, scores);
}

static struct ggml_tensor * trellis_qkv_view(
    struct ggml_context * ctx,
    struct ggml_tensor * qkv,
    int64_t channels,
    int64_t tokens,
    int64_t batches,
    int64_t head_dim,
    int64_t heads,
    int which) {
    const size_t elem = ggml_element_size(qkv);
    struct ggml_tensor * t = ggml_view_4d(
        ctx,
        qkv,
        head_dim,
        heads,
        tokens,
        batches,
        head_dim * elem,
        qkv->nb[1],
        qkv->nb[2],
        (size_t) which * (size_t) channels * elem);
    t = ggml_permute(ctx, t, 0, 2, 1, 3);
    t = ggml_cont_4d(ctx, t, head_dim, tokens, heads, batches);
    return t;
}

static struct ggml_tensor * trellis_attention_out_to_tokens(
    struct ggml_context * ctx,
    struct ggml_tensor * h,
    int64_t channels,
    int64_t tokens,
    int64_t batches) {
    h = ggml_permute(ctx, h, 0, 2, 1, 3);
    return ggml_cont_3d(ctx, h, channels, tokens, batches);
}

struct ggml_tensor * trellis_ggml_apply_rope_adjacent(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase) {
    if (ctx == NULL || x == NULL || cos_phase == NULL || sin_phase == NULL) {
        return NULL;
    }
    const int64_t head_dim = x->ne[0];
    const int64_t tokens = x->ne[1];
    const int64_t heads = x->ne[2];
    const int64_t batches = x->ne[3];
    if ((head_dim & 1) != 0 || cos_phase->ne[1] != head_dim / 2 || sin_phase->ne[1] != head_dim / 2 ||
        cos_phase->ne[2] != tokens || sin_phase->ne[2] != tokens) {
        return NULL;
    }

    const size_t elem = ggml_element_size(x);
    const int64_t half = head_dim / 2;
    const int64_t head_batches = heads * batches;
    struct ggml_tensor * x0 = ggml_view_4d(
        ctx,
        x,
        1,
        half,
        tokens,
        head_batches,
        2 * elem,
        x->nb[1],
        x->nb[2],
        0);
    struct ggml_tensor * x1 = ggml_view_4d(
        ctx,
        x,
        1,
        half,
        tokens,
        head_batches,
        2 * elem,
        x->nb[1],
        x->nb[2],
        elem);
    struct ggml_tensor * cos_rep = ggml_repeat(ctx, cos_phase, x0);
    struct ggml_tensor * sin_rep = ggml_repeat(ctx, sin_phase, x0);
    struct ggml_tensor * y0 = ggml_sub(
        ctx,
        ggml_mul(ctx, x0, cos_rep),
        ggml_mul(ctx, x1, sin_rep));
    struct ggml_tensor * y1 = ggml_add(
        ctx,
        ggml_mul(ctx, x0, sin_rep),
        ggml_mul(ctx, x1, cos_rep));
    struct ggml_tensor * pair = ggml_concat(ctx, y0, y1, 0);
    return ggml_cont_4d(ctx, pair, head_dim, tokens, heads, batches);
}

struct ggml_tensor * trellis_ggml_self_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int n_heads,
    struct ggml_tensor * qkv_w,
    struct ggml_tensor * qkv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b) {
    const int64_t channels = x->ne[0];
    const int64_t tokens = x->ne[1];
    const int64_t batches = x->ne[2];
    const int64_t heads = n_heads;
    const int64_t head_dim = channels / heads;

    struct ggml_tensor * qkv = trellis_ggml_linear(ctx, x, qkv_w, qkv_b);
    struct ggml_tensor * q = trellis_qkv_view(ctx, qkv, channels, tokens, batches, head_dim, heads, 0);
    struct ggml_tensor * k = trellis_qkv_view(ctx, qkv, channels, tokens, batches, head_dim, heads, 1);
    struct ggml_tensor * v = trellis_qkv_view(ctx, qkv, channels, tokens, batches, head_dim, heads, 2);

    if (q_rms_gamma != NULL) {
        q = trellis_ggml_multihead_rms_norm(ctx, q, q_rms_gamma, 0.0f);
    }
    if (k_rms_gamma != NULL) {
        k = trellis_ggml_multihead_rms_norm(ctx, k, k_rms_gamma, 0.0f);
    }

    struct ggml_tensor * h = trellis_ggml_sdpa(ctx, q, k, v, 1.0f / sqrtf((float) head_dim));
    h = trellis_attention_out_to_tokens(ctx, h, channels, tokens, batches);
    return trellis_ggml_linear(ctx, h, out_w, out_b);
}

struct ggml_tensor * trellis_ggml_self_attention_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int n_heads,
    struct ggml_tensor * qkv_w,
    struct ggml_tensor * qkv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase) {
    const int64_t channels = x->ne[0];
    const int64_t tokens = x->ne[1];
    const int64_t batches = x->ne[2];
    const int64_t heads = n_heads;
    const int64_t head_dim = channels / heads;

    struct ggml_tensor * qkv = trellis_ggml_linear(ctx, x, qkv_w, qkv_b);
    struct ggml_tensor * q = trellis_qkv_view(ctx, qkv, channels, tokens, batches, head_dim, heads, 0);
    struct ggml_tensor * k = trellis_qkv_view(ctx, qkv, channels, tokens, batches, head_dim, heads, 1);
    struct ggml_tensor * v = trellis_qkv_view(ctx, qkv, channels, tokens, batches, head_dim, heads, 2);

    if (q_rms_gamma != NULL) {
        q = trellis_ggml_multihead_rms_norm(ctx, q, q_rms_gamma, 0.0f);
    }
    if (k_rms_gamma != NULL) {
        k = trellis_ggml_multihead_rms_norm(ctx, k, k_rms_gamma, 0.0f);
    }
    if (cos_phase != NULL && sin_phase != NULL) {
        q = trellis_ggml_apply_rope_adjacent(ctx, q, cos_phase, sin_phase);
        k = trellis_ggml_apply_rope_adjacent(ctx, k, cos_phase, sin_phase);
    }

    struct ggml_tensor * h = trellis_ggml_sdpa(ctx, q, k, v, 1.0f / sqrtf((float) head_dim));
    h = trellis_attention_out_to_tokens(ctx, h, channels, tokens, batches);
    return trellis_ggml_linear(ctx, h, out_w, out_b);
}

static struct ggml_tensor * trellis_split_attention_view(
    struct ggml_context * ctx,
    struct ggml_tensor * t,
    int64_t channels,
    int64_t tokens,
    int64_t batches,
    int64_t head_dim,
    int64_t heads,
    int which,
    int n_parts) {
    const size_t elem = ggml_element_size(t);
    struct ggml_tensor * out = ggml_view_4d(
        ctx,
        t,
        head_dim,
        heads,
        tokens,
        batches,
        head_dim * elem,
        t->nb[1],
        t->nb[2],
        (size_t) which * (size_t) channels * elem);
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    out = ggml_cont_4d(ctx, out, head_dim, tokens, heads, batches);
    (void) n_parts;
    return out;
}

struct ggml_tensor * trellis_ggml_cross_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * context,
    int n_heads,
    struct ggml_tensor * q_w,
    struct ggml_tensor * q_b,
    struct ggml_tensor * kv_w,
    struct ggml_tensor * kv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b) {
    const int64_t channels = x->ne[0];
    const int64_t tokens = x->ne[1];
    const int64_t batches = x->ne[2];
    const int64_t kv_tokens = context->ne[1];
    const int64_t heads = n_heads;
    const int64_t head_dim = channels / heads;

    struct ggml_tensor * q_linear = trellis_ggml_linear(ctx, x, q_w, q_b);
    struct ggml_tensor * kv = trellis_ggml_linear(ctx, context, kv_w, kv_b);

    struct ggml_tensor * q = trellis_split_attention_view(ctx, q_linear, channels, tokens, batches, head_dim, heads, 0, 1);
    struct ggml_tensor * k = trellis_split_attention_view(ctx, kv, channels, kv_tokens, batches, head_dim, heads, 0, 2);
    struct ggml_tensor * v = trellis_split_attention_view(ctx, kv, channels, kv_tokens, batches, head_dim, heads, 1, 2);

    if (q_rms_gamma != NULL) {
        q = trellis_ggml_multihead_rms_norm(ctx, q, q_rms_gamma, 0.0f);
    }
    if (k_rms_gamma != NULL) {
        k = trellis_ggml_multihead_rms_norm(ctx, k, k_rms_gamma, 0.0f);
    }

    struct ggml_tensor * h = trellis_ggml_sdpa(ctx, q, k, v, 1.0f / sqrtf((float) head_dim));
    h = trellis_attention_out_to_tokens(ctx, h, channels, tokens, batches);
    return trellis_ggml_linear(ctx, h, out_w, out_b);
}

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

static struct ggml_tensor * trellis_mod_chunk(
    struct ggml_context * ctx,
    struct ggml_tensor * mod,
    int64_t channels,
    int which) {
    const size_t elem = ggml_element_size(mod);
    const int64_t batches = mod->ne[1];
    return ggml_view_3d(
        ctx,
        mod,
        channels,
        1,
        batches,
        (size_t) channels * elem,
        mod->nb[1],
        (size_t) which * (size_t) channels * elem);
}

static struct ggml_tensor * trellis_ggml_modulated_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * shift,
    struct ggml_tensor * scale) {
    struct ggml_tensor * h = trellis_ggml_layer_norm(ctx, x, NULL, NULL, 1e-6f);
    h = ggml_add(ctx, h, ggml_mul(ctx, h, ggml_repeat(ctx, scale, h)));
    h = ggml_add(ctx, h, ggml_repeat(ctx, shift, h));
    return h;
}

static struct ggml_tensor * trellis_ggml_modulated_cross_block_impl(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase) {
    if (ctx == NULL || x == NULL || mod6 == NULL || context == NULL || params == NULL || n_heads <= 0) {
        return NULL;
    }

    const int64_t channels = x->ne[0];
    struct ggml_tensor * mod = mod6;
    if (params->block_modulation != NULL) {
        mod = ggml_add(ctx, mod6, ggml_repeat(ctx, params->block_modulation, mod6));
    }
    if (params->emulate_bf16) {
        mod = trellis_ggml_bf16_roundtrip(ctx, mod);
    }

    struct ggml_tensor * shift_msa = trellis_mod_chunk(ctx, mod, channels, 0);
    struct ggml_tensor * scale_msa = trellis_mod_chunk(ctx, mod, channels, 1);
    struct ggml_tensor * gate_msa  = trellis_mod_chunk(ctx, mod, channels, 2);
    struct ggml_tensor * shift_mlp = trellis_mod_chunk(ctx, mod, channels, 3);
    struct ggml_tensor * scale_mlp = trellis_mod_chunk(ctx, mod, channels, 4);
    struct ggml_tensor * gate_mlp  = trellis_mod_chunk(ctx, mod, channels, 5);
    const int debug_parts = params->debug_parts < 0 ? 3 : params->debug_parts;
    if (debug_parts <= 0) {
        return x;
    }

    struct ggml_tensor * h = trellis_ggml_modulated_norm(ctx, x, shift_msa, scale_msa);
    if (params->emulate_bf16) {
        h = trellis_ggml_bf16_roundtrip(ctx, h);
    }
    if (cos_phase != NULL && sin_phase != NULL) {
        h = trellis_ggml_self_attention_rope(
            ctx,
            h,
            n_heads,
            params->self_qkv_w,
            params->self_qkv_b,
            params->self_q_rms_gamma,
            params->self_k_rms_gamma,
            params->self_out_w,
            params->self_out_b,
            cos_phase,
            sin_phase);
    } else {
        h = trellis_ggml_self_attention(
            ctx,
            h,
            n_heads,
            params->self_qkv_w,
            params->self_qkv_b,
            params->self_q_rms_gamma,
            params->self_k_rms_gamma,
            params->self_out_w,
            params->self_out_b);
    }
    if (params->emulate_bf16) {
        h = trellis_ggml_bf16_roundtrip(ctx, h);
    }
    h = ggml_mul(ctx, h, ggml_repeat(ctx, gate_msa, h));
    x = ggml_add(ctx, x, h);
    if (params->emulate_bf16) {
        x = trellis_ggml_bf16_roundtrip(ctx, x);
    }
    if (debug_parts <= 1) {
        return x;
    }

    h = trellis_ggml_layer_norm(ctx, x, params->norm2_gamma, params->norm2_beta, 1e-6f);
    if (params->emulate_bf16) {
        h = trellis_ggml_bf16_roundtrip(ctx, h);
    }
    h = trellis_ggml_cross_attention(
        ctx,
        h,
        context,
        n_heads,
        params->cross_q_w,
        params->cross_q_b,
        params->cross_kv_w,
        params->cross_kv_b,
        params->cross_q_rms_gamma,
        params->cross_k_rms_gamma,
        params->cross_out_w,
        params->cross_out_b);
    if (params->emulate_bf16) {
        h = trellis_ggml_bf16_roundtrip(ctx, h);
    }
    x = ggml_add(ctx, x, h);
    if (params->emulate_bf16) {
        x = trellis_ggml_bf16_roundtrip(ctx, x);
    }
    if (debug_parts <= 2) {
        return x;
    }

    h = trellis_ggml_modulated_norm(ctx, x, shift_mlp, scale_mlp);
    if (params->emulate_bf16) {
        h = trellis_ggml_bf16_roundtrip(ctx, h);
    }
    h = trellis_ggml_feed_forward(
        ctx,
        h,
        params->mlp_fc1_w,
        params->mlp_fc1_b,
        params->mlp_fc2_w,
        params->mlp_fc2_b);
    if (params->emulate_bf16) {
        h = trellis_ggml_bf16_roundtrip(ctx, h);
    }
    h = ggml_mul(ctx, h, ggml_repeat(ctx, gate_mlp, h));
    x = ggml_add(ctx, x, h);
    if (params->emulate_bf16) {
        x = trellis_ggml_bf16_roundtrip(ctx, x);
    }
    return x;
}

struct ggml_tensor * trellis_ggml_modulated_cross_block(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params) {
    return trellis_ggml_modulated_cross_block_impl(ctx, x, mod6, context, n_heads, params, NULL, NULL);
}

struct ggml_tensor * trellis_ggml_modulated_cross_block_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase) {
    return trellis_ggml_modulated_cross_block_impl(ctx, x, mod6, context, n_heads, params, cos_phase, sin_phase);
}
