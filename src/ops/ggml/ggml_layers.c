#include "trellis.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int g_trellis_ggml_use_flash_attn = 0;

void trellis_ggml_set_flash_attn_enabled(int enabled) {
    g_trellis_ggml_use_flash_attn = enabled ? 1 : 0;
}

int trellis_ggml_flash_attn_enabled(void) {
    return g_trellis_ggml_use_flash_attn;
}

int trellis_ggml_attention_policy_is_valid(
    const trellis_ggml_attention_policy * policy) {
    return policy != NULL &&
        policy->struct_size >= sizeof(trellis_ggml_attention_policy) &&
        (policy->mode == TRELLIS_GGML_ATTENTION_MODE_EXPLICIT ||
         policy->mode == TRELLIS_GGML_ATTENTION_MODE_FLASH ||
         policy->mode == TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16);
}

static struct ggml_tensor * trellis_ggml_cast_param_like(
    struct ggml_context * ctx,
    struct ggml_tensor * param,
    const struct ggml_tensor * ref) {
    if (ctx == NULL || param == NULL || ref == NULL || param->type == ref->type) {
        return param;
    }
    if (ref->type == GGML_TYPE_F32 || ref->type == GGML_TYPE_F16) {
        return ggml_cast(ctx, param, ref->type);
    }
    return param;
}

static struct ggml_tensor * trellis_ggml_repeat_param(
    struct ggml_context * ctx,
    struct ggml_tensor * param,
    struct ggml_tensor * ref) {
    param = trellis_ggml_cast_param_like(ctx, param, ref);
    return ggml_repeat(ctx, param, ref);
}

struct ggml_tensor * trellis_ggml_linear(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * weight,
    struct ggml_tensor * bias) {
    struct ggml_tensor * y = ggml_mul_mat(ctx, weight, x);
    if (bias != NULL) {
        y = ggml_add(ctx, y, trellis_ggml_repeat_param(ctx, bias, y));
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
        y = ggml_mul(ctx, y, trellis_ggml_repeat_param(ctx, gamma, y));
    }
    if (beta != NULL) {
        y = ggml_add(ctx, y, trellis_ggml_repeat_param(ctx, beta, y));
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
        y = ggml_mul(ctx, y, trellis_ggml_repeat_param(ctx, gamma, y));
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
    return ggml_mul(ctx, y, trellis_ggml_repeat_param(ctx, g, y));
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

struct ggml_tensor * trellis_ggml_sdpa_with_policy(
    struct ggml_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    float scale,
    const trellis_ggml_attention_policy * policy) {
    const int use_flash = policy == NULL ?
        g_trellis_ggml_use_flash_attn :
        (trellis_ggml_attention_policy_is_valid(policy) &&
         (policy->mode == TRELLIS_GGML_ATTENTION_MODE_FLASH ||
          policy->mode == TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16));
    if (policy != NULL && !trellis_ggml_attention_policy_is_valid(policy)) {
        return NULL;
    }
    if (use_flash) {
        const enum ggml_type flash_kv_type =
            policy != NULL &&
            policy->mode == TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16 ?
                GGML_TYPE_BF16 : GGML_TYPE_F16;
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx, k, flash_kv_type);
        }
        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx, v, flash_kv_type);
        }
        struct ggml_tensor * h = ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(h, GGML_PREC_F32);
        h = ggml_permute(ctx, h, 0, 2, 1, 3);
        return ggml_cont_4d(ctx, h, v->ne[0], q->ne[1], q->ne[2], q->ne[3]);
    }

    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_scale(ctx, scores, scale);
    scores = ggml_soft_max(ctx, scores);
    struct ggml_tensor * v_t = ggml_permute(ctx, v, 1, 0, 2, 3);
    v_t = ggml_cont_4d(ctx, v_t, v->ne[1], v->ne[0], v->ne[2], v->ne[3]);
    return ggml_mul_mat(ctx, v_t, scores);
}

struct ggml_tensor * trellis_ggml_sdpa(
    struct ggml_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    float scale) {
    return trellis_ggml_sdpa_with_policy(ctx, q, k, v, scale, NULL);
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

static struct ggml_tensor * trellis_ggml_self_attention_impl(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int n_heads,
    struct ggml_tensor * qkv_w,
    struct ggml_tensor * qkv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b,
    const trellis_ggml_attention_policy * attention_policy) {
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

    struct ggml_tensor * h = trellis_ggml_sdpa_with_policy(
        ctx,
        q,
        k,
        v,
        1.0f / sqrtf((float) head_dim),
        attention_policy);
    if (h == NULL) {
        return NULL;
    }
    h = trellis_attention_out_to_tokens(ctx, h, channels, tokens, batches);
    return trellis_ggml_linear(ctx, h, out_w, out_b);
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
    return trellis_ggml_self_attention_impl(
        ctx,
        x,
        n_heads,
        qkv_w,
        qkv_b,
        q_rms_gamma,
        k_rms_gamma,
        out_w,
        out_b,
        NULL);
}

static struct ggml_tensor * trellis_ggml_self_attention_rope_impl(
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
    struct ggml_tensor * sin_phase,
    const trellis_ggml_attention_policy * attention_policy) {
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

    struct ggml_tensor * h = trellis_ggml_sdpa_with_policy(
        ctx,
        q,
        k,
        v,
        1.0f / sqrtf((float) head_dim),
        attention_policy);
    if (h == NULL) {
        return NULL;
    }
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
    return trellis_ggml_self_attention_rope_impl(
        ctx,
        x,
        n_heads,
        qkv_w,
        qkv_b,
        q_rms_gamma,
        k_rms_gamma,
        out_w,
        out_b,
        cos_phase,
        sin_phase,
        NULL);
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

static struct ggml_tensor * trellis_ggml_cross_attention_impl(
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
    struct ggml_tensor * out_b,
    const trellis_ggml_attention_policy * attention_policy) {
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

    struct ggml_tensor * h = trellis_ggml_sdpa_with_policy(
        ctx,
        q,
        k,
        v,
        1.0f / sqrtf((float) head_dim),
        attention_policy);
    if (h == NULL) {
        return NULL;
    }
    h = trellis_attention_out_to_tokens(ctx, h, channels, tokens, batches);
    return trellis_ggml_linear(ctx, h, out_w, out_b);
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
    return trellis_ggml_cross_attention_impl(
        ctx,
        x,
        context,
        n_heads,
        q_w,
        q_b,
        kv_w,
        kv_b,
        q_rms_gamma,
        k_rms_gamma,
        out_w,
        out_b,
        NULL);
}

static struct ggml_tensor * trellis_ggml_project_attention_impl(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * global_context,
    struct ggml_tensor * projected_context,
    int n_heads,
    struct ggml_tensor * q_w,
    struct ggml_tensor * q_b,
    struct ggml_tensor * kv_w,
    struct ggml_tensor * kv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b,
    struct ggml_tensor * proj_w,
    struct ggml_tensor * proj_b,
    const trellis_ggml_attention_policy * attention_policy) {
    if (ctx == NULL || x == NULL || global_context == NULL || projected_context == NULL ||
        q_w == NULL || kv_w == NULL || out_w == NULL || proj_w == NULL || n_heads <= 0) {
        return NULL;
    }

    const int64_t channels = x->ne[0];
    const int64_t tokens = x->ne[1];
    const int64_t batches = x->ne[2];
    const int64_t proj_in = projected_context->ne[0];
    if (channels <= 0 || channels % n_heads != 0 ||
        global_context->ne[2] != batches ||
        projected_context->ne[1] != tokens || projected_context->ne[2] != batches ||
        q_w->ne[0] != channels || q_w->ne[1] != channels ||
        kv_w->ne[0] != global_context->ne[0] || kv_w->ne[1] != 2 * channels ||
        out_w->ne[0] != channels || out_w->ne[1] != channels ||
        proj_w->ne[0] != proj_in || proj_w->ne[1] != channels ||
        (q_b != NULL && q_b->ne[0] != channels) ||
        (kv_b != NULL && kv_b->ne[0] != 2 * channels) ||
        (out_b != NULL && out_b->ne[0] != channels) ||
        (proj_b != NULL && proj_b->ne[0] != channels)) {
        return NULL;
    }

    struct ggml_tensor * global_out = trellis_ggml_cross_attention_impl(
        ctx,
        x,
        global_context,
        n_heads,
        q_w,
        q_b,
        kv_w,
        kv_b,
        q_rms_gamma,
        k_rms_gamma,
        out_w,
        out_b,
        attention_policy);
    struct ggml_tensor * projected_out = trellis_ggml_linear(ctx, projected_context, proj_w, proj_b);
    if (global_out == NULL || projected_out == NULL) {
        return NULL;
    }
    return ggml_add(ctx, global_out, projected_out);
}

struct ggml_tensor * trellis_ggml_project_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * global_context,
    struct ggml_tensor * projected_context,
    int n_heads,
    struct ggml_tensor * q_w,
    struct ggml_tensor * q_b,
    struct ggml_tensor * kv_w,
    struct ggml_tensor * kv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b,
    struct ggml_tensor * proj_w,
    struct ggml_tensor * proj_b) {
    return trellis_ggml_project_attention_impl(
        ctx,
        x,
        global_context,
        projected_context,
        n_heads,
        q_w,
        q_b,
        kv_w,
        kv_b,
        q_rms_gamma,
        k_rms_gamma,
        out_w,
        out_b,
        proj_w,
        proj_b,
        NULL);
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
    h = ggml_add(ctx, h, ggml_mul(ctx, h, trellis_ggml_repeat_param(ctx, scale, h)));
    h = ggml_add(ctx, h, trellis_ggml_repeat_param(ctx, shift, h));
    return h;
}

static struct ggml_tensor * trellis_ggml_modulated_cross_block_impl(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * global_context,
    struct ggml_tensor * projected_context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * proj_w,
    struct ggml_tensor * proj_b,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_ggml_attention_policy * attention_policy) {
    if (ctx == NULL || x == NULL || mod6 == NULL || global_context == NULL || params == NULL || n_heads <= 0 ||
        (projected_context != NULL && proj_w == NULL)) {
        return NULL;
    }

    const int64_t channels = x->ne[0];
    struct ggml_tensor * mod = mod6;
    if (params->block_modulation != NULL) {
        mod = ggml_add(ctx, mod6, trellis_ggml_repeat_param(ctx, params->block_modulation, mod6));
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
        h = trellis_ggml_self_attention_rope_impl(
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
            sin_phase,
            attention_policy);
    } else {
        h = trellis_ggml_self_attention_impl(
            ctx,
            h,
            n_heads,
            params->self_qkv_w,
            params->self_qkv_b,
            params->self_q_rms_gamma,
            params->self_k_rms_gamma,
            params->self_out_w,
            params->self_out_b,
            attention_policy);
    }
    if (h == NULL) {
        return NULL;
    }
    if (params->emulate_bf16) {
        h = trellis_ggml_bf16_roundtrip(ctx, h);
    }
    h = ggml_mul(ctx, h, trellis_ggml_repeat_param(ctx, gate_msa, h));
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
    if (projected_context != NULL) {
        h = trellis_ggml_project_attention_impl(
            ctx,
            h,
            global_context,
            projected_context,
            n_heads,
            params->cross_q_w,
            params->cross_q_b,
            params->cross_kv_w,
            params->cross_kv_b,
            params->cross_q_rms_gamma,
            params->cross_k_rms_gamma,
            params->cross_out_w,
            params->cross_out_b,
            proj_w,
            proj_b,
            attention_policy);
    } else {
        h = trellis_ggml_cross_attention_impl(
            ctx,
            h,
            global_context,
            n_heads,
            params->cross_q_w,
            params->cross_q_b,
            params->cross_kv_w,
            params->cross_kv_b,
            params->cross_q_rms_gamma,
            params->cross_k_rms_gamma,
            params->cross_out_w,
            params->cross_out_b,
            attention_policy);
    }
    if (h == NULL) {
        return NULL;
    }
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
    h = ggml_mul(ctx, h, trellis_ggml_repeat_param(ctx, gate_mlp, h));
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
    return trellis_ggml_modulated_cross_block_impl(
        ctx, x, mod6, context, NULL, n_heads, params, NULL, NULL, NULL, NULL, NULL);
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
    return trellis_ggml_modulated_cross_block_impl(
        ctx, x, mod6, context, NULL, n_heads, params, NULL, NULL, cos_phase, sin_phase, NULL);
}

struct ggml_tensor * trellis_ggml_modulated_cross_block_rope_with_policy(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_ggml_attention_policy * attention_policy) {
    return trellis_ggml_modulated_cross_block_impl(
        ctx,
        x,
        mod6,
        context,
        NULL,
        n_heads,
        params,
        NULL,
        NULL,
        cos_phase,
        sin_phase,
        attention_policy);
}

struct ggml_tensor * trellis_ggml_modulated_cross_block_projected(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * global_context,
    struct ggml_tensor * projected_context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * proj_w,
    struct ggml_tensor * proj_b) {
    if (projected_context == NULL || proj_w == NULL) {
        return NULL;
    }
    return trellis_ggml_modulated_cross_block_impl(
        ctx,
        x,
        mod6,
        global_context,
        projected_context,
        n_heads,
        params,
        proj_w,
        proj_b,
        NULL,
        NULL,
        NULL);
}

struct ggml_tensor * trellis_ggml_modulated_cross_block_projected_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * global_context,
    struct ggml_tensor * projected_context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * proj_w,
    struct ggml_tensor * proj_b,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase) {
    if (projected_context == NULL || proj_w == NULL) {
        return NULL;
    }
    return trellis_ggml_modulated_cross_block_impl(
        ctx,
        x,
        mod6,
        global_context,
        projected_context,
        n_heads,
        params,
        proj_w,
        proj_b,
        cos_phase,
        sin_phase,
        NULL);
}

struct ggml_tensor * trellis_ggml_modulated_cross_block_projected_rope_with_policy(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * global_context,
    struct ggml_tensor * projected_context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * proj_w,
    struct ggml_tensor * proj_b,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_ggml_attention_policy * attention_policy) {
    if (projected_context == NULL || proj_w == NULL) {
        return NULL;
    }
    return trellis_ggml_modulated_cross_block_impl(
        ctx,
        x,
        mod6,
        global_context,
        projected_context,
        n_heads,
        params,
        proj_w,
        proj_b,
        cos_phase,
        sin_phase,
        attention_policy);
}
