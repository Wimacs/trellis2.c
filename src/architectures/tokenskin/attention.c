#include "tokenskin.h"

#include <math.h>

static struct ggml_tensor * standard_head_view(
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int64_t head_dim,
    int64_t heads) {
    const int64_t tokens = tensor->ne[1];
    const int64_t batches = tensor->ne[2];
    const size_t element = ggml_element_size(tensor);
    return ggml_view_4d(
        ctx,
        tensor,
        head_dim,
        tokens,
        heads,
        batches,
        tensor->nb[1],
        (size_t) head_dim * element,
        tensor->nb[2],
        0);
}

static struct ggml_tensor * interleaved_part_view(
    struct ggml_context * ctx,
    struct ggml_tensor * packed,
    int64_t head_dim,
    int64_t heads,
    int parts,
    int which) {
    const int64_t tokens = packed->ne[1];
    const int64_t batches = packed->ne[2];
    const size_t element = ggml_element_size(packed);
    return ggml_view_4d(
        ctx,
        packed,
        head_dim,
        tokens,
        heads,
        batches,
        packed->nb[1],
        (size_t) parts * (size_t) head_dim * element,
        packed->nb[2],
        (size_t) which * (size_t) head_dim * element);
}

static struct ggml_tensor * attention_to_tokens(
    struct ggml_context * ctx,
    struct ggml_tensor * attention,
    int64_t channels,
    int64_t tokens,
    int64_t batches) {
    attention = ggml_permute(ctx, attention, 0, 2, 1, 3);
    return ggml_cont_3d(ctx, attention, channels, tokens, batches);
}

static struct ggml_tensor * run_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    int64_t channels,
    int64_t query_tokens,
    int64_t batches,
    int64_t head_dim,
    const tokenskin_linear_weights * out_weights,
    const trellis_ggml_attention_policy * attention_policy) {
    struct ggml_tensor * result = trellis_ggml_sdpa_with_policy(
        ctx,
        q,
        k,
        v,
        1.0f / sqrtf((float) head_dim),
        attention_policy);
    if (result == NULL) return NULL;
    result = attention_to_tokens(ctx, result, channels, query_tokens, batches);
    return trellis_ggml_linear(
        ctx, result, out_weights->weight, out_weights->bias);
}

struct ggml_tensor * tokenskin_attention_standard_q_interleaved_kv(
    struct ggml_context * ctx,
    struct ggml_tensor * query,
    struct ggml_tensor * context,
    int heads,
    const tokenskin_linear_weights * q_weights,
    const tokenskin_linear_weights * kv_weights,
    const tokenskin_linear_weights * out_weights,
    const trellis_ggml_attention_policy * attention_policy) {
    if (ctx == NULL || query == NULL || context == NULL || heads <= 0 ||
        q_weights == NULL || kv_weights == NULL || out_weights == NULL) return NULL;
    const int64_t channels = query->ne[0];
    if (channels % heads != 0 || context->ne[0] != channels ||
        query->ne[2] != context->ne[2]) return NULL;
    const int64_t head_dim = channels / heads;
    struct ggml_tensor * q_linear = trellis_ggml_linear(
        ctx, query, q_weights->weight, q_weights->bias);
    struct ggml_tensor * kv = trellis_ggml_linear(
        ctx, context, kv_weights->weight, kv_weights->bias);
    struct ggml_tensor * q = standard_head_view(ctx, q_linear, head_dim, heads);
    struct ggml_tensor * k = interleaved_part_view(ctx, kv, head_dim, heads, 2, 0);
    struct ggml_tensor * v = interleaved_part_view(ctx, kv, head_dim, heads, 2, 1);
    return run_attention(
        ctx, q, k, v, channels, query->ne[1], query->ne[2], head_dim,
        out_weights, attention_policy);
}

struct ggml_tensor * tokenskin_attention_interleaved_qkv(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int heads,
    const tokenskin_linear_weights * qkv_weights,
    const tokenskin_linear_weights * out_weights,
    const trellis_ggml_attention_policy * attention_policy) {
    if (ctx == NULL || x == NULL || heads <= 0 ||
        qkv_weights == NULL || out_weights == NULL || x->ne[0] % heads != 0) return NULL;
    const int64_t channels = x->ne[0];
    const int64_t head_dim = channels / heads;
    struct ggml_tensor * qkv = trellis_ggml_linear(
        ctx, x, qkv_weights->weight, qkv_weights->bias);
    struct ggml_tensor * q = interleaved_part_view(ctx, qkv, head_dim, heads, 3, 0);
    struct ggml_tensor * k = interleaved_part_view(ctx, qkv, head_dim, heads, 3, 1);
    struct ggml_tensor * v = interleaved_part_view(ctx, qkv, head_dim, heads, 3, 2);
    return run_attention(
        ctx, q, k, v, channels, x->ne[1], x->ne[2], head_dim,
        out_weights, attention_policy);
}

struct ggml_tensor * tokenskin_tripo_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * query,
    struct ggml_tensor * context,
    int heads,
    const tokenskin_tripo_block_weights * weights,
    const trellis_ggml_attention_policy * attention_policy) {
    if (ctx == NULL || query == NULL || weights == NULL || heads <= 0 ||
        query->ne[0] % heads != 0) return NULL;
    if (context == NULL) context = query;
    const int64_t channels = query->ne[0];
    const int64_t head_dim = channels / heads;
    struct ggml_tensor * q_linear = trellis_ggml_linear(
        ctx, query, weights->q.weight, weights->q.bias);
    struct ggml_tensor * k_linear = trellis_ggml_linear(
        ctx, context, weights->k.weight, weights->k.bias);
    struct ggml_tensor * v_linear = trellis_ggml_linear(
        ctx, context, weights->v.weight, weights->v.bias);
    struct ggml_tensor * q;
    struct ggml_tensor * k;
    struct ggml_tensor * v;
    if (weights->cross_attention) {
        struct ggml_tensor * kv = ggml_concat(ctx, k_linear, v_linear, 0);
        q = standard_head_view(ctx, q_linear, head_dim, heads);
        k = interleaved_part_view(ctx, kv, head_dim, heads, 2, 0);
        v = interleaved_part_view(ctx, kv, head_dim, heads, 2, 1);
    } else {
        struct ggml_tensor * qk = ggml_concat(ctx, q_linear, k_linear, 0);
        struct ggml_tensor * qkv = ggml_concat(ctx, qk, v_linear, 0);
        q = interleaved_part_view(ctx, qkv, head_dim, heads, 3, 0);
        k = interleaved_part_view(ctx, qkv, head_dim, heads, 3, 1);
        v = interleaved_part_view(ctx, qkv, head_dim, heads, 3, 2);
    }
    return run_attention(
        ctx, q, k, v, channels, query->ne[1], query->ne[2], head_dim,
        &weights->out, attention_policy);
}
