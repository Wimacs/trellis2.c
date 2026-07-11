#include "trellis.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TRELLIS_DINO_HIDDEN = 1024,
    TRELLIS_DINO_INTERMEDIATE = 4096,
    TRELLIS_DINO_HEADS = 16,
    TRELLIS_DINO_HEAD_DIM = 64,
    TRELLIS_DINO_PATCH = 16,
    TRELLIS_DINO_REGISTERS = 4,
};

static void set_issue(char * dst, size_t dst_size, const char * fmt, ...) {
    if (dst == NULL || dst_size == 0 || dst[0] != '\0') {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(dst, dst_size, fmt, args);
    va_end(args);
}

static trellis_status bind_tensor(
    trellis_tensor_store * store,
    const char * name,
    int n_dims,
    const int64_t * ne,
    struct ggml_tensor ** out,
    char * first_issue,
    size_t first_issue_size) {
    if (store == NULL || name == NULL || ne == NULL || out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    struct ggml_tensor * t = trellis_tensor_store_get(store, name);
    if (t == NULL) {
        set_issue(first_issue, first_issue_size, "missing tensor: %s", name);
        return TRELLIS_STATUS_NOT_FOUND;
    }
    for (int i = 0; i < n_dims; ++i) {
        if (t->ne[i] != ne[i]) {
            set_issue(
                first_issue,
                first_issue_size,
                "shape mismatch: %s dim%d got %lld expected %lld",
                name,
                i,
                (long long) t->ne[i],
                (long long) ne[i]);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    for (int i = n_dims; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] != 1) {
            set_issue(
                first_issue,
                first_issue_size,
                "rank mismatch: %s dim%d got %lld expected 1",
                name,
                i,
                (long long) t->ne[i]);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    *out = t;
    return TRELLIS_STATUS_OK;
}

static trellis_status bind_vec(
    trellis_tensor_store * store,
    const char * name,
    int64_t n,
    struct ggml_tensor ** out,
    char * first_issue,
    size_t first_issue_size) {
    int64_t ne[1] = { n };
    return bind_tensor(store, name, 1, ne, out, first_issue, first_issue_size);
}

static trellis_status bind_linear(
    trellis_tensor_store * store,
    const char * w_name,
    const char * b_name,
    int64_t in,
    int64_t out,
    struct ggml_tensor ** w,
    struct ggml_tensor ** b,
    char * first_issue,
    size_t first_issue_size) {
    int64_t ne[2] = { in, out };
    trellis_status status = bind_tensor(store, w_name, 2, ne, w, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (b_name == NULL) {
        *b = NULL;
        return TRELLIS_STATUS_OK;
    }
    return bind_vec(store, b_name, out, b, first_issue, first_issue_size);
}

static trellis_status bind_block(
    trellis_tensor_store * store,
    int index,
    trellis_dino_vit_block_weights * block,
    char * first_issue,
    size_t first_issue_size) {
    char name[160];
    char bias[160];
    trellis_status status;

    snprintf(name, sizeof(name), "layer.%d.norm1.weight", index);
    status = bind_vec(store, name, TRELLIS_DINO_HIDDEN, &block->norm1_gamma, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "layer.%d.norm1.bias", index);
    status = bind_vec(store, name, TRELLIS_DINO_HIDDEN, &block->norm1_beta, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    snprintf(name, sizeof(name), "layer.%d.attention.q_proj.weight", index);
    snprintf(bias, sizeof(bias), "layer.%d.attention.q_proj.bias", index);
    status = bind_linear(store, name, bias, TRELLIS_DINO_HIDDEN, TRELLIS_DINO_HIDDEN, &block->q_w, &block->q_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "layer.%d.attention.k_proj.weight", index);
    struct ggml_tensor * unused_bias = NULL;
    status = bind_linear(store, name, NULL, TRELLIS_DINO_HIDDEN, TRELLIS_DINO_HIDDEN, &block->k_w, &unused_bias, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "layer.%d.attention.v_proj.weight", index);
    snprintf(bias, sizeof(bias), "layer.%d.attention.v_proj.bias", index);
    status = bind_linear(store, name, bias, TRELLIS_DINO_HIDDEN, TRELLIS_DINO_HIDDEN, &block->v_w, &block->v_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "layer.%d.attention.o_proj.weight", index);
    snprintf(bias, sizeof(bias), "layer.%d.attention.o_proj.bias", index);
    status = bind_linear(store, name, bias, TRELLIS_DINO_HIDDEN, TRELLIS_DINO_HIDDEN, &block->o_w, &block->o_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "layer.%d.layer_scale1.lambda1", index);
    status = bind_vec(store, name, TRELLIS_DINO_HIDDEN, &block->layer_scale1, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    snprintf(name, sizeof(name), "layer.%d.norm2.weight", index);
    status = bind_vec(store, name, TRELLIS_DINO_HIDDEN, &block->norm2_gamma, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "layer.%d.norm2.bias", index);
    status = bind_vec(store, name, TRELLIS_DINO_HIDDEN, &block->norm2_beta, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "layer.%d.mlp.up_proj.weight", index);
    snprintf(bias, sizeof(bias), "layer.%d.mlp.up_proj.bias", index);
    status = bind_linear(store, name, bias, TRELLIS_DINO_HIDDEN, TRELLIS_DINO_INTERMEDIATE, &block->mlp_up_w, &block->mlp_up_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "layer.%d.mlp.down_proj.weight", index);
    snprintf(bias, sizeof(bias), "layer.%d.mlp.down_proj.bias", index);
    status = bind_linear(store, name, bias, TRELLIS_DINO_INTERMEDIATE, TRELLIS_DINO_HIDDEN, &block->mlp_down_w, &block->mlp_down_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    snprintf(name, sizeof(name), "layer.%d.layer_scale2.lambda1", index);
    return bind_vec(store, name, TRELLIS_DINO_HIDDEN, &block->layer_scale2, first_issue, first_issue_size);
}

trellis_status trellis_dino_vit_bind_weights(
    trellis_tensor_store * store,
    trellis_dino_vit_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    if (store == NULL || weights == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (first_issue != NULL && first_issue_size != 0) {
        first_issue[0] = '\0';
    }
    memset(weights, 0, sizeof(*weights));
    weights->hidden_size = TRELLIS_DINO_HIDDEN;
    weights->intermediate_size = TRELLIS_DINO_INTERMEDIATE;
    weights->patch_size = TRELLIS_DINO_PATCH;
    weights->heads = TRELLIS_DINO_HEADS;
    weights->head_dim = TRELLIS_DINO_HEAD_DIM;
    weights->register_tokens_count = TRELLIS_DINO_REGISTERS;

    int64_t token_shape[3] = { TRELLIS_DINO_HIDDEN, 1, 1 };
    trellis_status status = bind_tensor(store, "embeddings.cls_token", 3, token_shape, &weights->cls_token, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    status = bind_tensor(store, "embeddings.mask_token", 3, token_shape, &weights->mask_token, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    int64_t reg_shape[3] = { TRELLIS_DINO_HIDDEN, TRELLIS_DINO_REGISTERS, 1 };
    status = bind_tensor(store, "embeddings.register_tokens", 3, reg_shape, &weights->register_tokens, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    int64_t patch_shape[4] = { TRELLIS_DINO_PATCH, TRELLIS_DINO_PATCH, 3, TRELLIS_DINO_HIDDEN };
    status = bind_tensor(store, "embeddings.patch_embeddings.weight", 4, patch_shape, &weights->patch_w, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    status = bind_vec(store, "embeddings.patch_embeddings.bias", TRELLIS_DINO_HIDDEN, &weights->patch_b, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;

    for (int i = 0; i < TRELLIS_DINO_VIT_LAYERS; ++i) {
        status = bind_block(store, i, &weights->blocks[i], first_issue, first_issue_size);
        if (status != TRELLIS_STATUS_OK) {
            return status;
        }
    }
    status = bind_vec(store, "norm.weight", TRELLIS_DINO_HIDDEN, &weights->norm_gamma, first_issue, first_issue_size);
    if (status != TRELLIS_STATUS_OK) return status;
    return bind_vec(store, "norm.bias", TRELLIS_DINO_HIDDEN, &weights->norm_beta, first_issue, first_issue_size);
}

static struct ggml_tensor * dino_attention_view(
    struct ggml_context * ctx,
    struct ggml_tensor * t,
    int64_t channels,
    int64_t tokens,
    int64_t batches,
    int64_t head_dim,
    int64_t heads) {
    t = ggml_cont_3d(ctx, t, channels, tokens, batches);
    t = ggml_reshape_4d(ctx, t, head_dim, heads, tokens, batches);
    t = ggml_permute(ctx, t, 0, 2, 1, 3);
    return ggml_cont_4d(ctx, t, head_dim, tokens, heads, batches);
}

static struct ggml_tensor * dino_attention_out_to_tokens(
    struct ggml_context * ctx,
    struct ggml_tensor * h,
    int64_t channels,
    int64_t tokens,
    int64_t batches) {
    h = ggml_permute(ctx, h, 0, 2, 1, 3);
    return ggml_cont_3d(ctx, h, channels, tokens, batches);
}

static struct ggml_tensor * dino_cast_param_like(
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

static struct ggml_tensor * dino_repeat_param(
    struct ggml_context * ctx,
    struct ggml_tensor * param,
    struct ggml_tensor * ref) {
    param = dino_cast_param_like(ctx, param, ref);
    return ggml_repeat(ctx, param, ref);
}

static struct ggml_tensor * dino_layer_scale(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * scale) {
    return ggml_mul(ctx, x, dino_repeat_param(ctx, scale, x));
}

struct ggml_tensor * trellis_dino_patch_embedding_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * image,
    const trellis_dino_vit_weights * weights) {
    if (ctx == NULL || image == NULL || weights == NULL ||
        image->ne[2] != 3 || image->ne[0] <= 0 || image->ne[1] <= 0 ||
        image->ne[0] % weights->patch_size != 0 ||
        image->ne[1] % weights->patch_size != 0) {
        return NULL;
    }

    const int64_t patches_w = image->ne[0] / weights->patch_size;
    const int64_t patches_h = image->ne[1] / weights->patch_size;
    const int64_t n_patches = patches_w * patches_h;
    const int64_t batch = image->ne[3];

    struct ggml_tensor * patches = ggml_conv_2d_direct(
        ctx,
        weights->patch_w,
        image,
        weights->patch_size,
        weights->patch_size,
        0,
        0,
        1,
        1);
    if (patches == NULL) {
        return NULL;
    }
    if (weights->patch_b != NULL) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx, weights->patch_b, 1, 1, weights->hidden_size, 1);
        patches = ggml_add(ctx, patches, dino_repeat_param(ctx, bias, patches));
    }

    patches = ggml_permute(ctx, patches, 1, 2, 0, 3);
    return ggml_cont_3d(ctx, patches, weights->hidden_size, n_patches, batch);
}

struct ggml_tensor * trellis_dino_image_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * image,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_dino_vit_weights * weights) {
    if (ctx == NULL || image == NULL || weights == NULL) {
        return NULL;
    }
    struct ggml_tensor * patches = trellis_dino_patch_embedding_forward(ctx, image, weights);
    if (patches == NULL) {
        return NULL;
    }

    const int64_t batch = patches->ne[2];
    struct ggml_tensor * special = ggml_concat(ctx, weights->cls_token, weights->register_tokens, 1);
    special = ggml_repeat_4d(ctx, special, weights->hidden_size, 1 + weights->register_tokens_count, batch, 1);
    struct ggml_tensor * tokens = ggml_concat(ctx, special, patches, 1);
    tokens = ggml_reshape_3d(ctx, tokens, weights->hidden_size, tokens->ne[1], batch);
    return trellis_dino_vit_forward(ctx, tokens, cos_phase, sin_phase, weights);
}

static struct ggml_tensor * dino_apply_rope_halves(
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
    if ((head_dim & 1) != 0 ||
        cos_phase->ne[0] != head_dim || cos_phase->ne[1] != tokens ||
        sin_phase->ne[0] != head_dim || sin_phase->ne[1] != tokens) {
        return NULL;
    }

    const size_t elem = ggml_element_size(x);
    const int64_t half = head_dim / 2;
    struct ggml_tensor * x0 = ggml_view_4d(
        ctx,
        x,
        half,
        tokens,
        heads,
        batches,
        x->nb[1],
        x->nb[2],
        x->nb[3],
        0);
    struct ggml_tensor * x1 = ggml_view_4d(
        ctx,
        x,
        half,
        tokens,
        heads,
        batches,
        x->nb[1],
        x->nb[2],
        x->nb[3],
        (size_t) half * elem);
    struct ggml_tensor * cos0 = ggml_view_4d(
        ctx,
        cos_phase,
        half,
        tokens,
        1,
        1,
        cos_phase->nb[1],
        cos_phase->nb[2],
        cos_phase->nb[3],
        0);
    struct ggml_tensor * sin0 = ggml_view_4d(
        ctx,
        sin_phase,
        half,
        tokens,
        1,
        1,
        sin_phase->nb[1],
        sin_phase->nb[2],
        sin_phase->nb[3],
        0);

    struct ggml_tensor * cos_rep = ggml_repeat(ctx, cos0, x0);
    struct ggml_tensor * sin_rep = ggml_repeat(ctx, sin0, x0);
    struct ggml_tensor * y0 = ggml_sub(
        ctx,
        ggml_mul(ctx, x0, cos_rep),
        ggml_mul(ctx, x1, sin_rep));
    struct ggml_tensor * y1 = ggml_add(
        ctx,
        ggml_mul(ctx, x1, cos_rep),
        ggml_mul(ctx, x0, sin_rep));
    struct ggml_tensor * y = ggml_concat(ctx, y0, y1, 0);
    return ggml_cont_4d(ctx, y, head_dim, tokens, heads, batches);
}

static struct ggml_tensor * dino_self_attention_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    const trellis_dino_vit_block_weights * block,
    int heads,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase) {
    const int64_t channels = x->ne[0];
    const int64_t tokens = x->ne[1];
    const int64_t batches = x->ne[2];
    const int64_t head_dim = channels / heads;

    struct ggml_tensor * q_linear = trellis_ggml_linear(ctx, x, block->q_w, block->q_b);
    struct ggml_tensor * k_linear = trellis_ggml_linear(ctx, x, block->k_w, NULL);
    struct ggml_tensor * v_linear = trellis_ggml_linear(ctx, x, block->v_w, block->v_b);

    struct ggml_tensor * q = dino_attention_view(ctx, q_linear, channels, tokens, batches, head_dim, heads);
    struct ggml_tensor * k = dino_attention_view(ctx, k_linear, channels, tokens, batches, head_dim, heads);
    struct ggml_tensor * v = dino_attention_view(ctx, v_linear, channels, tokens, batches, head_dim, heads);

    if (cos_phase != NULL && sin_phase != NULL) {
        q = dino_apply_rope_halves(ctx, q, cos_phase, sin_phase);
        k = dino_apply_rope_halves(ctx, k, cos_phase, sin_phase);
        if (q == NULL || k == NULL) {
            return NULL;
        }
    }

    struct ggml_tensor * h = trellis_ggml_sdpa(ctx, q, k, v, 1.0f / sqrtf((float) head_dim));
    h = dino_attention_out_to_tokens(ctx, h, channels, tokens, batches);
    return trellis_ggml_linear(ctx, h, block->o_w, block->o_b);
}

struct ggml_tensor * trellis_dino_vit_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * tokens,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase,
    const trellis_dino_vit_weights * weights) {
    if (ctx == NULL || tokens == NULL || weights == NULL ||
        tokens->ne[0] != weights->hidden_size || weights->heads <= 0) {
        return NULL;
    }
    if ((cos_phase == NULL) != (sin_phase == NULL)) {
        return NULL;
    }
    if (cos_phase != NULL &&
        (cos_phase->ne[0] != weights->head_dim || cos_phase->ne[1] != tokens->ne[1] ||
         sin_phase->ne[0] != weights->head_dim || sin_phase->ne[1] != tokens->ne[1])) {
        return NULL;
    }

    struct ggml_tensor * h = tokens;
    for (int i = 0; i < TRELLIS_DINO_VIT_LAYERS; ++i) {
        const trellis_dino_vit_block_weights * block = &weights->blocks[i];
        struct ggml_tensor * a = trellis_ggml_layer_norm(ctx, h, block->norm1_gamma, block->norm1_beta, 1e-5f);
        a = dino_self_attention_rope(ctx, a, block, weights->heads, cos_phase, sin_phase);
        if (a == NULL) {
            return NULL;
        }
        a = dino_layer_scale(ctx, a, block->layer_scale1);
        h = ggml_add(ctx, h, a);

        struct ggml_tensor * m = trellis_ggml_layer_norm(ctx, h, block->norm2_gamma, block->norm2_beta, 1e-5f);
        m = trellis_ggml_feed_forward(ctx, m, block->mlp_up_w, block->mlp_up_b, block->mlp_down_w, block->mlp_down_b);
        m = dino_layer_scale(ctx, m, block->layer_scale2);
        h = ggml_add(ctx, h, m);
    }
    return trellis_ggml_layer_norm(ctx, h, NULL, NULL, 1e-5f);
}

trellis_status trellis_dino_image_forward_f32_host(
    const trellis_backend_context * backend,
    const trellis_dino_vit_weights * weights,
    const float * image,
    int batch,
    int image_h,
    int image_w,
    const float * cos_phase_data,
    const float * sin_phase_data,
    float ** tokens_out,
    int * n_tokens_out) {
    if (backend == NULL || backend->backend == NULL || weights == NULL || image == NULL ||
        tokens_out == NULL || n_tokens_out == NULL ||
        batch <= 0 || image_h <= 0 || image_w <= 0 || weights->patch_size <= 0 ||
        image_h % weights->patch_size != 0 || image_w % weights->patch_size != 0 ||
        (cos_phase_data == NULL) != (sin_phase_data == NULL)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *tokens_out = NULL;
    *n_tokens_out = 0;

    const int patches_h = image_h / weights->patch_size;
    const int patches_w = image_w / weights->patch_size;
    const int n_special = 1 + weights->register_tokens_count;
    const int total_tokens = n_special + patches_h * patches_w;
    const size_t phase_count = (size_t) total_tokens * (size_t) weights->head_dim;
    float * owned_cos = NULL;
    float * owned_sin = NULL;
    if (cos_phase_data == NULL) {
        owned_cos = (float *) malloc(phase_count * sizeof(float));
        owned_sin = (float *) malloc(phase_count * sizeof(float));
        if (owned_cos == NULL || owned_sin == NULL) {
            free(owned_cos);
            free(owned_sin);
            return TRELLIS_STATUS_OUT_OF_MEMORY;
        }
        trellis_status phase_status = trellis_dino_rope_2d_phases_f32(
            n_special,
            patches_h,
            patches_w,
            weights->head_dim,
            1.0f,
            100.0f,
            owned_cos,
            owned_sin,
            phase_count);
        if (phase_status != TRELLIS_STATUS_OK) {
            free(owned_cos);
            free(owned_sin);
            return phase_status;
        }
        cos_phase_data = owned_cos;
        sin_phase_data = owned_sin;
    }

    const size_t graph_nodes = 131072;
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false) + 4 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        free(owned_cos);
        free(owned_sin);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    struct ggml_tensor * image_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, image_w, image_h, 3, batch);
    struct ggml_tensor * cos_phase = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, weights->head_dim, total_tokens, 1, 1);
    struct ggml_tensor * sin_phase = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, weights->head_dim, total_tokens, 1, 1);
    struct ggml_tensor * y = trellis_dino_image_forward(ctx, image_tensor, cos_phase, sin_phase, weights);
    if (image_tensor == NULL || cos_phase == NULL || sin_phase == NULL || y == NULL) {
        ggml_free(ctx);
        free(owned_cos);
        free(owned_sin);
        return TRELLIS_STATUS_ERROR;
    }

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    if (graph == NULL) {
        ggml_free(ctx);
        free(owned_cos);
        free(owned_sin);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_build_forward_expand(graph, y);

    ggml_gallocr_t alloc = trellis_backend_new_graph_allocator(backend);
    if (alloc == NULL || !ggml_gallocr_alloc_graph(alloc, graph)) {
        if (alloc != NULL) {
            ggml_gallocr_free(alloc);
        }
        ggml_free(ctx);
        free(owned_cos);
        free(owned_sin);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    ggml_backend_tensor_set(image_tensor, image, 0, ggml_nbytes(image_tensor));
    ggml_backend_tensor_set(cos_phase, cos_phase_data, 0, ggml_nbytes(cos_phase));
    ggml_backend_tensor_set(sin_phase, sin_phase_data, 0, ggml_nbytes(sin_phase));

    trellis_status status = trellis_backend_compute_graph(backend, graph);
    if (status == TRELLIS_STATUS_OK) {
        float * out = (float *) malloc((size_t) ggml_nelements(y) * sizeof(float));
        if (out == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else {
            ggml_backend_tensor_get(y, out, 0, ggml_nbytes(y));
            *tokens_out = out;
            *n_tokens_out = total_tokens;
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    free(owned_cos);
    free(owned_sin);
    return status;
}

trellis_status trellis_dino_rope_2d_phases_f32(
    int n_special_tokens,
    int patches_h,
    int patches_w,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count) {
    if (n_special_tokens < 0 || patches_h <= 0 || patches_w <= 0 ||
        head_dim <= 0 || (head_dim & 1) != 0 ||
        freq_scale <= 0.0f || freq_base <= 0.0f ||
        cos_out == NULL || sin_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int freq_dim = head_dim / 4;
    const int total_tokens = n_special_tokens + patches_h * patches_w;
    if (freq_dim <= 0 || phase_count < (size_t) total_tokens * (size_t) head_dim) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int token = 0; token < total_tokens; ++token) {
        float * cos_row = cos_out + (size_t) token * (size_t) head_dim;
        float * sin_row = sin_out + (size_t) token * (size_t) head_dim;
        if (token < n_special_tokens) {
            for (int k = 0; k < head_dim; ++k) {
                cos_row[k] = 1.0f;
                sin_row[k] = 0.0f;
            }
            continue;
        }

        const int patch = token - n_special_tokens;
        const int py = patch / patches_w;
        const int px = patch - py * patches_w;
        const float coords[2] = {
            2.0f * (((float) py + 0.5f) / (float) patches_h) - 1.0f,
            2.0f * (((float) px + 0.5f) / (float) patches_w) - 1.0f,
        };
        int k = 0;
        for (int axis = 0; axis < 2; ++axis) {
            for (int f = 0; f < freq_dim; ++f) {
                const float power = (float) (4 * f) / (float) head_dim;
                const float freq = freq_scale / powf(freq_base, power);
                const float phase = 6.2831853071795864769f * coords[axis] * freq;
                cos_row[k] = cosf(phase);
                sin_row[k] = sinf(phase);
                ++k;
            }
        }
        for (int i = 0; i < k; ++i) {
            cos_row[k + i] = cos_row[i];
            sin_row[k + i] = sin_row[i];
        }
    }
    return TRELLIS_STATUS_OK;
}
