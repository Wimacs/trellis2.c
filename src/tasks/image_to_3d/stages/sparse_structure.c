#include "trellis.h"
#include "dit_flow_executor.h"
#include "trellis_ggml_layers.h"
#include "image_to_3d_internal.h"
#include "pixal_projection.h"
#include "pixal_naf.h"
#include "projected_attention.h"
#include "sparse/trellis_sparse_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static void trellis_perf_stage_log(const char * name, int64_t elapsed_us) {
    TRELLIS_INFO(
        "perf_stage name=%s ms=%.3f",
        name != NULL ? name : "unknown",
        elapsed_us <= 0 ? 0.0 : (double) elapsed_us / 1000.0);
}

static int choose_path(
    const char * model_dir,
    const char * rel,
    char * dst,
    size_t dst_size) {
    trellis_status status = trellis_make_model_path(model_dir, rel, dst, dst_size);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "path: %s\n", trellis_status_string(status));
        return 0;
    }
    return 1;
}

static int make_join_path(const char * dir, const char * file, char * dst, size_t dst_size) {
    if (dir == NULL || file == NULL || dst == NULL || dst_size == 0) {
        return 0;
    }
    int n = snprintf(dst, dst_size, "%s/%s", dir, file);
    return n >= 0 && (size_t) n < dst_size;
}

static int collect_voxel_coords_xyz(
    const float * logits,
    int resolution,
    float threshold,
    int32_t ** coords_out,
    int64_t * n_out) {
    if (logits == NULL || resolution <= 0 || coords_out == NULL || n_out == NULL) {
        return 0;
    }
    *coords_out = NULL;
    *n_out = 0;
    const int64_t total = (int64_t) resolution * (int64_t) resolution * (int64_t) resolution;
    int64_t count = 0;
    for (int64_t i = 0; i < total; ++i) {
        if (logits[i] > threshold) {
            ++count;
        }
    }
    int32_t * coords = (int32_t *) malloc((size_t) count * 3u * sizeof(int32_t));
    if (coords == NULL && count != 0) {
        return 0;
    }
    int64_t row = 0;
    for (int d = 0; d < resolution; ++d) {
        for (int h = 0; h < resolution; ++h) {
            for (int w = 0; w < resolution; ++w) {
                const int64_t idx = ((int64_t) d * resolution + h) * (int64_t) resolution + w;
                if (logits[idx] <= threshold) {
                    continue;
                }
                coords[(size_t) row * 3u + 0u] = (int32_t) d;
                coords[(size_t) row * 3u + 1u] = (int32_t) h;
                coords[(size_t) row * 3u + 2u] = (int32_t) w;
                ++row;
            }
        }
    }
    *coords_out = coords;
    *n_out = count;
    return 1;
}

static int set_sparse_structure_result(
    trellis_sparse_structure_result * result,
    const int32_t * coords_xyz,
    int64_t n_coords,
    int resolution,
    const float * cond,
    int cond_tokens) {
    if (result == NULL) {
        return 1;
    }
    if (coords_xyz == NULL || n_coords < 0 || cond == NULL || cond_tokens <= 0) {
        return 0;
    }
    trellis_sparse_structure_result_free(result);
    int32_t * coords_bxyz = (int32_t *) malloc((size_t) n_coords * 4u * sizeof(int32_t));
    float * cond_copy = (float *) malloc((size_t) cond_tokens * 1024u * sizeof(float));
    if ((coords_bxyz == NULL && n_coords != 0) || cond_copy == NULL) {
        free(coords_bxyz);
        free(cond_copy);
        return 0;
    }
    for (int64_t i = 0; i < n_coords; ++i) {
        coords_bxyz[(size_t) i * 4u + 0u] = 0;
        coords_bxyz[(size_t) i * 4u + 1u] = coords_xyz[(size_t) i * 3u + 0u];
        coords_bxyz[(size_t) i * 4u + 2u] = coords_xyz[(size_t) i * 3u + 1u];
        coords_bxyz[(size_t) i * 4u + 3u] = coords_xyz[(size_t) i * 3u + 2u];
    }
    memcpy(cond_copy, cond, (size_t) cond_tokens * 1024u * sizeof(float));
    result->coords_bxyz = coords_bxyz;
    result->n_coords = n_coords;
    result->resolution = resolution;
    result->cond = cond_copy;
    result->cond_tokens = cond_tokens;
    return 1;
}

void trellis_sparse_structure_result_free(trellis_sparse_structure_result * result) {
    if (result == NULL) {
        return;
    }
    free(result->coords_bxyz);
    free(result->cond);
    memset(result, 0, sizeof(*result));
}

void trellis_image_condition_result_free(trellis_image_condition_result * result) {
    if (result == NULL) {
        return;
    }
    free(result->cond);
    free(result->projected);
    memset(result, 0, sizeof(*result));
}

static int max_pool_logits_to_resolution(
    const float * logits,
    int input_resolution,
    float threshold,
    int output_resolution,
    float ** pooled_out) {
    if (logits == NULL || input_resolution <= 0 || output_resolution <= 0 || pooled_out == NULL ||
        input_resolution % output_resolution != 0) {
        return 0;
    }
    *pooled_out = NULL;
    if (input_resolution == output_resolution) {
        return 1;
    }
    const int ratio = input_resolution / output_resolution;
    const size_t out_count = (size_t) output_resolution * (size_t) output_resolution * (size_t) output_resolution;
    float * pooled = (float *) malloc(out_count * sizeof(float));
    if (pooled == NULL) {
        return 0;
    }
    for (int d = 0; d < output_resolution; ++d) {
        for (int h = 0; h < output_resolution; ++h) {
            for (int w = 0; w < output_resolution; ++w) {
                int occupied = 0;
                for (int rd = 0; rd < ratio && !occupied; ++rd) {
                    for (int rh = 0; rh < ratio && !occupied; ++rh) {
                        for (int rw = 0; rw < ratio; ++rw) {
                            const int sd = d * ratio + rd;
                            const int sh = h * ratio + rh;
                            const int sw = w * ratio + rw;
                            const int idx = (sd * input_resolution + sh) * input_resolution + sw;
                            if (logits[idx] > threshold) {
                                occupied = 1;
                                break;
                            }
                        }
                    }
                }
                pooled[(size_t) (d * output_resolution + h) * (size_t) output_resolution + (size_t) w] =
                    occupied ? 1.0f : 0.0f;
            }
        }
    }
    *pooled_out = pooled;
    return 1;
}

static void token_channels_to_ncdhw(
    const float * src,
    float * dst,
    int channels,
    int size) {
    for (int d = 0; d < size; ++d) {
        for (int h = 0; h < size; ++h) {
            for (int w = 0; w < size; ++w) {
                const size_t token = ((size_t) d * (size_t) size + (size_t) h) * (size_t) size + (size_t) w;
                for (int c = 0; c < channels; ++c) {
                    const size_t dst_i =
                        (((size_t) c * (size_t) size + (size_t) d) * (size_t) size + (size_t) h) *
                            (size_t) size +
                        (size_t) w;
                    dst[dst_i] = src[token * (size_t) channels + (size_t) c];
                }
            }
        }
    }
}

static int load_sparse_structure_flow(
    const trellis_backend_context * backend,
    const char * model_dir,
    const char * override_path,
    trellis_tensor_store * store,
    trellis_dit_flow_model * model) {
    char path[4096];
    if (override_path != NULL && override_path[0] != '\0') {
        const int length = snprintf(path, sizeof(path), "%s", override_path);
        if (length < 0 || (size_t) length >= sizeof(path)) return 0;
    } else if (!choose_path(model_dir, "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors", path, sizeof(path))) {
        return 0;
    }
    if (!trellis_load_tensor_store(
            backend,
            "sparse-structure flow",
            path,
            true,
            64,
            store,
            NULL)) {
        return 0;
    }
    char issue[256];
    trellis_status status = trellis_ss_flow_model_bind_weights(store, model, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("sparse-structure flow: bind failed: %s%s%s",
            trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
        trellis_tensor_store_free(store);
        return 0;
    }
    TRELLIS_INFO("sparse-structure flow: ready blocks=%d channels=%d",
        model->base.n_blocks, model->base.in_channels);
    if (model->projection.enabled) {
        TRELLIS_INFO(
            "sparse-structure flow: Pixal3D projection channels=%d",
            model->projection.proj_channels);
    }
    return 1;
}

static int load_sparse_structure_decoder(
    const trellis_backend_context * backend,
    const char * model_dir,
    const char * override_path,
    trellis_tensor_store * store,
    trellis_ss_decoder_weights * weights) {
    char path[4096];
    if (override_path != NULL && override_path[0] != '\0') {
        const int length = snprintf(path, sizeof(path), "%s", override_path);
        if (length < 0 || (size_t) length >= sizeof(path)) return 0;
    } else if (!choose_path(model_dir, "ckpts/ss_dec_conv3d_16l8_fp16.safetensors", path, sizeof(path))) {
        return 0;
    }
    if (!trellis_load_tensor_store(
            backend,
            "sparse-structure decoder",
            path,
            false,
            64,
            store,
            NULL)) {
        return 0;
    }
    char issue[256];
    trellis_status status = trellis_ss_decoder_bind_weights(store, weights, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("sparse-structure decoder: bind failed: %s%s%s",
            trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
        trellis_tensor_store_free(store);
        return 0;
    }
    TRELLIS_INFO("sparse-structure decoder: ready");
    return 1;
}

static int load_sparse_structure_dino(
    const trellis_backend_context * backend,
    const char * dino_dir,
    trellis_tensor_store * store,
    trellis_dino_vit_weights * weights);

static int copy_tensor_f32(const struct ggml_tensor * tensor, float * dst, size_t count) {
    if (tensor == NULL || dst == NULL || count > (size_t) ggml_nelements(tensor)) {
        return 0;
    }
    ggml_backend_tensor_get(tensor, dst, 0, count * sizeof(float));
    return 1;
}

static int load_sparse_structure_dino(
    const trellis_backend_context * backend,
    const char * dino_dir,
    trellis_tensor_store * store,
    trellis_dino_vit_weights * weights) {
    char path[4096];
    if (!make_join_path(dino_dir, "model.safetensors", path, sizeof(path))) {
        TRELLIS_ERROR("sparse structure: invalid dino model path");
        return 0;
    }
    if (!trellis_load_tensor_store(backend, "sparse-structure dino image encoder", path, true, 64, store, NULL)) {
        return 0;
    }
    char issue[256];
    trellis_status status = trellis_dino_vit_bind_weights(store, weights, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("sparse-structure dino image encoder: bind failed: %s%s%s",
            trellis_status_string(status), issue[0] == '\0' ? "" : " ", issue);
        trellis_tensor_store_free(store);
        return 0;
    }
    TRELLIS_INFO("sparse-structure dino image encoder: ready");
    return 1;
}

static float lcg_uniform(uint32_t * state) {
    *state = (*state * 1664525u) + 1013904223u;
    return ((float) ((*state >> 8) & 0x00ffffffu) + 0.5f) / 16777216.0f;
}

static void fill_gaussian_latent(float * dst, size_t count, uint32_t seed) {
    if (dst == NULL) {
        return;
    }
    uint32_t state = seed == 0 ? 1u : seed;
    const float two_pi = 6.2831853071795864769f;
    for (size_t i = 0; i < count; i += 2) {
        float u1 = lcg_uniform(&state);
        float u2 = lcg_uniform(&state);
        if (u1 < 1e-7f) {
            u1 = 1e-7f;
        }
        const float r = sqrtf(-2.0f * logf(u1));
        dst[i] = r * cosf(two_pi * u2);
        if (i + 1 < count) {
            dst[i + 1] = r * sinf(two_pi * u2);
        }
    }
}

static float clampf_local(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static float sinc_f32(float x) {
    const float ax = fabsf(x);
    if (ax < 1e-6f) {
        return 1.0f;
    }
    const float pix = 3.14159265358979323846f * x;
    return sinf(pix) / pix;
}

static float lanczos3_f32(float x) {
    const float ax = fabsf(x);
    if (ax >= 3.0f) {
        return 0.0f;
    }
    return sinc_f32(x) * sinc_f32(x / 3.0f);
}

static int preprocess_image_for_dino(
    const char * image_path,
    int edge,
    float foreground_crop_scale,
    float ** image_out,
    float ** guide_out,
    int * src_w_out,
    int * src_h_out) {
    if (image_path == NULL || edge <= 0 || image_out == NULL) {
        return 0;
    }
    *image_out = NULL;
    if (guide_out != NULL) *guide_out = NULL;
    if (src_w_out != NULL) *src_w_out = 0;
    if (src_h_out != NULL) *src_h_out = 0;

    int src_w = 0;
    int src_h = 0;
    int comp = 0;
    unsigned char * rgba = stbi_load(image_path, &src_w, &src_h, &comp, 4);
    if (rgba == NULL || src_w <= 0 || src_h <= 0) {
        fprintf(stderr, "sparse structure: failed to load image %s\n", image_path);
        stbi_image_free(rgba);
        return 0;
    }

    float * image = (float *) malloc(3u * (size_t) edge * (size_t) edge * sizeof(float));
    float * guide = guide_out != NULL ?
        (float *) malloc(3u * (size_t) edge * (size_t) edge * sizeof(float)) : NULL;
    if (image == NULL || (guide_out != NULL && guide == NULL)) {
        free(image);
        free(guide);
        stbi_image_free(rgba);
        return 0;
    }

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};
    const int pixal_preprocess = foreground_crop_scale > 0.0f && foreground_crop_scale < 1.13f;
    int crop_left = 0;
    int crop_top = 0;
    int crop_w = src_w > src_h ? src_w : src_h;
    int crop_h = crop_w;
    int use_alpha_composite = 0;
    int min_x = src_w, min_y = src_h, max_x = -1, max_y = -1;
    for (int yy = 0; yy < src_h; ++yy) {
        for (int xx = 0; xx < src_w; ++xx) {
            const unsigned char * p = rgba + (((size_t) yy * (size_t) src_w + (size_t) xx) * 4u);
            if (p[3] != 255) {
                use_alpha_composite = 1;
            }
            if (p[3] > (pixal_preprocess ? 204 : 8)) {
                if (xx < min_x) min_x = xx;
                if (yy < min_y) min_y = yy;
                if (xx + 1 > max_x) max_x = xx + 1;
                if (yy + 1 > max_y) max_y = yy + 1;
            }
        }
    }
    if (use_alpha_composite && max_x > min_x && max_y > min_y) {
        const float center_x = 0.5f * (float) (min_x + max_x);
        const float center_y = 0.5f * (float) (min_y + max_y);
        int side = max_x - min_x > max_y - min_y ? max_x - min_x : max_y - min_y;
        const float crop_scale = foreground_crop_scale > 0.0f ? foreground_crop_scale : 1.16f;
        side = (int) floorf((float) side * crop_scale + 0.5f);
        if (side < 1) side = 1;
        const int max_side = 2 * (src_w > src_h ? src_w : src_h);
        if (side > max_side) side = max_side;
        crop_left = (int) floorf(center_x - 0.5f * (float) side + 0.5f);
        crop_top = (int) floorf(center_y - 0.5f * (float) side + 0.5f);
        crop_w = side;
        crop_h = side;
    } else {
        crop_left = 0;
        crop_top = 0;
        if (pixal_preprocess) {
            crop_w = src_w;
            crop_h = src_h;
        } else {
            crop_w = src_w > src_h ? src_w : src_h;
            crop_h = crop_w;
        }
    }
    for (int y = 0; y < edge; ++y) {
        for (int x = 0; x < edge; ++x) {
            const float sx = ((float) x + 0.5f) * (float) crop_w / (float) edge - 0.5f;
            const float sy = ((float) y + 0.5f) * (float) crop_h / (float) edge - 0.5f;
            const int x_begin = (int) floorf(sx - 3.0f + 1.0f);
            const int x_end = (int) floorf(sx + 3.0f);
            const int y_begin = (int) floorf(sy - 3.0f + 1.0f);
            const int y_end = (int) floorf(sy + 3.0f);
            float acc[3] = {0.0f, 0.0f, 0.0f};
            float wsum = 0.0f;
            for (int yy = y_begin; yy <= y_end; ++yy) {
                if (yy < 0 || yy >= crop_h) {
                    continue;
                }
                const float wy = lanczos3_f32(sy - (float) yy);
                if (wy == 0.0f) {
                    continue;
                }
                for (int xx = x_begin; xx <= x_end; ++xx) {
                    if (xx < 0 || xx >= crop_w) {
                        continue;
                    }
                    const float wx = lanczos3_f32(sx - (float) xx);
                    const float w = wx * wy;
                    if (w == 0.0f) {
                        continue;
                    }
                    const int src_x = crop_left + xx;
                    const int src_y = crop_top + yy;
                    if (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h) {
                        const unsigned char * p = rgba + (((size_t) src_y * (size_t) src_w + (size_t) src_x) * 4u);
                        const float alpha = use_alpha_composite ? (float) p[3] / 255.0f : 1.0f;
                        for (int c = 0; c < 3; ++c) {
                            acc[c] += w * (float) p[c] * alpha;
                        }
                    }
                    wsum += w;
                }
            }
            for (int c = 0; c < 3; ++c) {
                float v = wsum == 0.0f ? 0.0f : acc[c] / wsum;
                v = clampf_local(v, 0.0f, 255.0f);
                const float rgb = v / 255.0f;
                if (guide != NULL) {
                    guide[((size_t) c * (size_t) edge + (size_t) y) * (size_t) edge + (size_t) x] = rgb;
                }
                image[((size_t) c * (size_t) edge + (size_t) y) * (size_t) edge + (size_t) x] =
                    (rgb - mean[c]) / std[c];
            }
        }
    }

    stbi_image_free(rgba);
    if (src_w_out != NULL) *src_w_out = src_w;
    if (src_h_out != NULL) *src_h_out = src_h;
    *image_out = image;
    if (guide_out != NULL) *guide_out = guide;
    return 1;
}

static int run_dino_condition(
    const trellis_backend_context * backend,
    trellis_pipeline_model_cache * cache,
    const char * dino_dir,
    const char * image_path,
    int cond_resolution,
    int pixal_preprocess,
    float ** guide_out,
    const trellis_dino_vit_weights * preloaded_dino,
    float ** context_out,
    int * cond_tokens_out) {
    if (backend == NULL || dino_dir == NULL || image_path == NULL ||
        cond_resolution <= 0 || context_out == NULL || cond_tokens_out == NULL) {
        return 0;
    }
    *context_out = NULL;
    *cond_tokens_out = 0;
    if (guide_out != NULL) *guide_out = NULL;

    trellis_tensor_store dino_store;
    memset(&dino_store, 0, sizeof(dino_store));
    trellis_dino_vit_weights dino_local;
    const trellis_dino_vit_weights * dino = preloaded_dino;
    int owns_dino_store = 0;
    int rc = 0;
    float * image = NULL;
    float * output_tokens = NULL;

    if (dino == NULL) {
        if (cache != NULL) {
            const trellis_dino_vit_weights * cached_dino = NULL;
            trellis_status status = trellis_pipeline_model_cache_get_dino(cache, dino_dir, &cached_dino);
            if (status != TRELLIS_STATUS_OK) {
                TRELLIS_ERROR("sparse-structure dino image encoder: cache load failed: %s", trellis_status_string(status));
                goto cleanup;
            }
            dino = cached_dino;
        } else {
            if (!load_sparse_structure_dino(backend, dino_dir, &dino_store, &dino_local)) {
                goto cleanup;
            }
            dino = &dino_local;
            owns_dino_store = 1;
        }
    }
    if (cond_resolution % dino->patch_size != 0) {
        fprintf(stderr, "sparse structure: --cond-resolution must be divisible by DINO patch size %d\n", dino->patch_size);
        goto cleanup;
    }

    int src_w = 0;
    int src_h = 0;
    float * guide = NULL;
    if (!preprocess_image_for_dino(
            image_path,
            cond_resolution,
            pixal_preprocess ? 1.1f : 1.16f,
            &image,
            guide_out != NULL ? &guide : NULL,
            &src_w,
            &src_h)) {
        goto cleanup;
    }
    TRELLIS_INFO("sparse structure: preprocessed %s from %dx%d to %dx%d",
        image_path, src_w, src_h, cond_resolution, cond_resolution);

    const int patches_h = cond_resolution / dino->patch_size;
    const int patches_w = cond_resolution / dino->patch_size;
    const int n_patches = patches_h * patches_w;
    const int expected_tokens = 1 + dino->register_tokens_count + n_patches;
    int total_tokens = 0;
    TRELLIS_INFO("sparse structure: running DINO image encoder graph tokens=%d layers=%d",
        expected_tokens, TRELLIS_DINO_VIT_LAYERS);
    trellis_status status = trellis_dino_image_forward_f32_host(
        backend, dino, image, 1, cond_resolution, cond_resolution, NULL, NULL, &output_tokens, &total_tokens);
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "sparse structure: dino image encoder failed: %s\n", trellis_status_string(status));
        goto cleanup;
    }

    TRELLIS_INFO("sparse structure: DINO %s %dx%d -> %dx%d patches=%d tokens=%d hidden=%d",
        image_path, src_w, src_h, cond_resolution, cond_resolution, n_patches, total_tokens, dino->hidden_size);
    *context_out = output_tokens;
    *cond_tokens_out = total_tokens;
    if (guide_out != NULL) {
        *guide_out = guide;
        guide = NULL;
    }
    output_tokens = NULL;
    rc = 1;

cleanup:
    free(image);
    free(guide);
    free(output_tokens);
    if (owns_dino_store) {
        trellis_tensor_store_free(&dino_store);
    }
    return rc;
}

static int run_sparse_structure_image(
    const trellis_backend_context * backend,
    const char * model_dir,
    const char * flow_path,
    const char * decoder_path,
    const char * dino_dir,
    const char * image_path,
    int latent_size,
    int steps,
    int cond_resolution,
    int sparse_resolution,
    uint32_t seed,
    int flow_blocks_override,
    int flow_block_parts_override,
    int flow_no_rope,
    int projected_conditioning,
    int emulate_bf16_blocks,
    const trellis_ggml_attention_policy * requested_attention_policy,
    float threshold,
    float camera_angle_x,
    float camera_distance,
    float mesh_scale,
    trellis_pipeline_model_cache * cache,
    trellis_sparse_structure_result * result) {
    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }
    if (backend == NULL || model_dir == NULL || dino_dir == NULL ||
        image_path == NULL ||
        latent_size <= 0 || steps <= 0 || cond_resolution <= 0 || sparse_resolution <= 0) {
        return 1;
    }

    TRELLIS_INFO(
        "sparse structure: pipeline inference latent=%d^3 steps=%d cond_resolution=%d sparse_resolution=%d seed=%u",
        latent_size,
        steps,
        cond_resolution,
        sparse_resolution,
        (unsigned) seed);

    float * context = NULL;
    float * projected_context = NULL;
    int cond_tokens = 0;
    const int64_t dino_start_us = ggml_time_us();
    if (!run_dino_condition(
            backend,
            cache,
            dino_dir,
            image_path,
            cond_resolution,
            projected_conditioning,
            NULL,
            NULL,
            &context,
            &cond_tokens)) {
        free(context);
        return 1;
    }
    char dino_perf_name[64];
    snprintf(dino_perf_name, sizeof(dino_perf_name), "dino_cond_%d", cond_resolution);
    trellis_perf_stage_log(dino_perf_name, ggml_time_us() - dino_start_us);

    trellis_tensor_store flow_store;
    trellis_tensor_store decoder_store;
    memset(&flow_store, 0, sizeof(flow_store));
    memset(&decoder_store, 0, sizeof(decoder_store));
    trellis_dit_flow_weights flow;
    trellis_dit_flow_model flow_model;
    trellis_ss_decoder_weights decoder;
    memset(&flow, 0, sizeof(flow));
    memset(&flow_model, 0, sizeof(flow_model));
    memset(&decoder, 0, sizeof(decoder));
    int owns_flow_store = 0;
    int owns_decoder_store = 0;
    trellis_dit_flow_executor flow_executor_cfg;
    trellis_dit_flow_executor flow_executor_cond;
    memset(&flow_executor_cfg, 0, sizeof(flow_executor_cfg));
    memset(&flow_executor_cond, 0, sizeof(flow_executor_cond));
    int use_cfg_batch = 0;
    int rc = 1;

    float * neg_context = NULL;
    float * neg_projected_context = NULL;
    float * latent = NULL;
    float * pred_pos = NULL;
    float * pred_neg = NULL;
    float * pred = NULL;
    float * next = NULL;
    float * x0 = NULL;
    float * latent_ncdhw = NULL;
    float * pairs = NULL;
    float * cos_phase = NULL;
    float * sin_phase = NULL;
    int64_t voxel_denoise_us = 0;
    int64_t voxel_decode_us = 0;
    trellis_status status = TRELLIS_STATUS_OK;

    if (cache != NULL) {
        const trellis_dit_flow_model * cached_flow = NULL;
        const trellis_ss_decoder_weights * cached_decoder = NULL;
        status = trellis_pipeline_model_cache_get_sparse_structure_flow_model(
            cache, model_dir, flow_path, &cached_flow);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("sparse-structure flow: cache load failed: %s", trellis_status_string(status));
            goto cleanup;
        }
        status = trellis_pipeline_model_cache_get_sparse_structure_decoder(
            cache, model_dir, decoder_path, &cached_decoder);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("sparse-structure decoder: cache load failed: %s", trellis_status_string(status));
            goto cleanup;
        }
        flow_model = *cached_flow;
        flow = flow_model.base;
        decoder = *cached_decoder;
    } else {
        if (!load_sparse_structure_flow(backend, model_dir, flow_path, &flow_store, &flow_model)) {
            goto cleanup;
        }
        flow = flow_model.base;
        owns_flow_store = 1;
        if (!load_sparse_structure_decoder(backend, model_dir, decoder_path, &decoder_store, &decoder)) {
            goto cleanup;
        }
        owns_decoder_store = 1;
    }
    if (flow_blocks_override >= 0) {
        if (flow_blocks_override > flow.n_blocks) {
            fprintf(stderr, "sparse structure: --flow-blocks %d exceeds checkpoint blocks %d\n",
                flow_blocks_override,
                flow.n_blocks);
            goto cleanup;
        }
        flow.n_blocks = flow_blocks_override;
        TRELLIS_INFO("sparse structure: debug flow blocks override=%d", flow.n_blocks);
    }
    if (flow_block_parts_override >= 0) {
        if (flow_block_parts_override > 3) {
            fprintf(stderr, "sparse structure: --flow-block-parts must be in [0,3]\n");
            goto cleanup;
        }
        flow.debug_block_parts = flow_block_parts_override;
        TRELLIS_INFO("sparse structure: debug flow block parts override=%d", flow.debug_block_parts);
    }
    if (flow_no_rope) {
        flow.debug_disable_rope = 1;
        TRELLIS_INFO("sparse structure: debug flow RoPE disabled");
    }
    if (emulate_bf16_blocks) {
        flow.emulate_bf16_blocks = 1;
        TRELLIS_INFO("sparse structure: bf16 block activation round-trip enabled");
    }
    if (flow_model.projection.enabled != (projected_conditioning ? 1 : 0)) {
        TRELLIS_ERROR(
            "sparse structure: package conditioning mode does not match sparse flow architecture (package=%s checkpoint=%s)",
            projected_conditioning ? "projected" : "global",
            flow_model.projection.enabled ? "projected" : "global");
        goto cleanup;
    }
    if (flow.cond_channels != 1024) {
        fprintf(stderr, "sparse structure: unexpected flow cond channels %d\n", flow.cond_channels);
        goto cleanup;
    }
    const int tokens = latent_size * latent_size * latent_size;
    if (flow_model.projection.enabled) {
        const int special_tokens = 5;
        const int patch_side = cond_resolution / 16;
        const int patch_tokens = patch_side * patch_side;
        if (latent_size != 16 || cond_resolution != 512 || sparse_resolution != 32 ||
            flow_model.projection.proj_channels != 1024 ||
            cond_resolution % 16 != 0 || cond_tokens != special_tokens + patch_tokens) {
            TRELLIS_ERROR(
                "sparse structure: Pixal3D stage mismatch latent=%d cond_resolution=%d sparse_resolution=%d global=%d total=%d patches=%d proj_channels=%d",
                latent_size,
                cond_resolution,
                sparse_resolution,
                special_tokens,
                cond_tokens,
                patch_tokens,
                flow_model.projection.proj_channels);
            goto cleanup;
        }
        const size_t global_count = (size_t) special_tokens * (size_t) flow.cond_channels;
        const size_t projected_count =
            (size_t) tokens * (size_t) flow_model.projection.proj_channels;
        float * global_context = (float *) malloc(global_count * sizeof(float));
        projected_context = (float *) malloc(projected_count * sizeof(float));
        if (global_context == NULL || projected_context == NULL) {
            free(global_context);
            TRELLIS_ERROR("sparse structure: Pixal3D condition allocation failed");
            goto cleanup;
        }
        memcpy(global_context, context, global_count * sizeof(float));
        const float resolved_camera_angle_x =
            camera_angle_x > 0.0f ? camera_angle_x : 0.8575560450553894f;
        const float resolved_mesh_scale = mesh_scale > 0.0f ? mesh_scale : 1.0f;
        float resolved_camera_distance = 0.0f;
        status = trellis_pixal_resolve_camera_distance(
            resolved_camera_angle_x,
            resolved_mesh_scale,
            camera_distance,
            &resolved_camera_distance);
        if (status != TRELLIS_STATUS_OK) {
            free(global_context);
            TRELLIS_ERROR("sparse structure: invalid Pixal3D camera parameters");
            goto cleanup;
        }
        trellis_pixal_camera camera = {
            .camera_angle_x = resolved_camera_angle_x,
            .distance = resolved_camera_distance,
            .mesh_scale = resolved_mesh_scale,
        };
        status = trellis_pixal_project_patch_features_dense_f32(
            context + global_count,
            patch_side,
            patch_side,
            flow_model.projection.proj_channels,
            cond_resolution,
            latent_size,
            &camera,
            projected_context,
            projected_count);
        if (status != TRELLIS_STATUS_OK) {
            free(global_context);
            TRELLIS_ERROR(
                "sparse structure: Pixal3D projection failed: %s",
                trellis_status_string(status));
            goto cleanup;
        }
        free(context);
        context = global_context;
        cond_tokens = special_tokens;
        TRELLIS_INFO(
            "sparse structure: Pixal3D condition global_tokens=%d projected=%d^3x%d",
            cond_tokens,
            latent_size,
            flow_model.projection.proj_channels);
    }
    flow_model.base = flow;
    trellis_ggml_attention_policy attention_policy =
        TRELLIS_GGML_ATTENTION_POLICY_INIT;
    if (trellis_ggml_attention_policy_is_valid(requested_attention_policy)) {
        attention_policy = *requested_attention_policy;
    }

    const size_t latent_count = (size_t) flow.in_channels * (size_t) tokens;
    const size_t context_count = (size_t) flow.cond_channels * (size_t) cond_tokens;
    const size_t projected_count = flow_model.projection.enabled ?
        (size_t) tokens * (size_t) flow_model.projection.proj_channels : 0u;
    const size_t phase_count = (size_t) (flow.head_dim / 2) * (size_t) tokens;
    neg_context = (float *) calloc(context_count, sizeof(float));
    if (projected_count > 0) {
        neg_projected_context = (float *) calloc(projected_count, sizeof(float));
    }
    latent = (float *) malloc(latent_count * sizeof(float));
    pred_pos = (float *) malloc(latent_count * sizeof(float));
    pred_neg = (float *) malloc(latent_count * sizeof(float));
    pred = (float *) malloc(latent_count * sizeof(float));
    next = (float *) malloc(latent_count * sizeof(float));
    x0 = (float *) malloc(latent_count * sizeof(float));
    latent_ncdhw = (float *) malloc(latent_count * sizeof(float));
    pairs = (float *) malloc((size_t) steps * 2u * sizeof(float));
    cos_phase = (float *) malloc(phase_count * sizeof(float));
    sin_phase = (float *) malloc(phase_count * sizeof(float));
    if (neg_context == NULL || (projected_count > 0 && neg_projected_context == NULL) ||
        latent == NULL || pred_pos == NULL || pred_neg == NULL || pred == NULL ||
        next == NULL || x0 == NULL || latent_ncdhw == NULL || pairs == NULL ||
        cos_phase == NULL || sin_phase == NULL) {
        fprintf(stderr, "sparse structure: host allocation failed\n");
        goto cleanup;
    }

    fill_gaussian_latent(latent, latent_count, seed);
    status = trellis_flow_timestep_pairs_f32(steps, 5.0f, pairs, (size_t) steps * 2u);
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_rope_3d_phases_f32(latent_size, flow.head_dim, 1.0f, 10000.0f, cos_phase, sin_phase, phase_count);
    }
    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "sparse structure: schedule/rope failed: %s\n", trellis_status_string(status));
        goto cleanup;
    }
    if (flow_model.projection.enabled) {
        status = trellis_dit_flow_executor_init_cfg_batch_projected_with_policy(
            &flow_executor_cfg,
            backend,
            &flow_model,
            tokens,
            cond_tokens,
            context,
            neg_context,
            projected_context,
            neg_projected_context,
            cos_phase,
            sin_phase,
            &attention_policy);
    } else {
        status = trellis_dit_flow_executor_init_cfg_batch_with_policy(
            &flow_executor_cfg,
            backend,
            &flow,
            tokens,
            cond_tokens,
            context,
            neg_context,
            cos_phase,
            sin_phase,
            &attention_policy);
    }
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_WARN(
            "sparse structure: fused CFG executor init failed (%s); falling back to one reusable batch=1 graph",
            trellis_status_string(status));
        trellis_dit_flow_executor_free(&flow_executor_cfg);
        if (flow_model.projection.enabled) {
            status = trellis_dit_flow_executor_init_single_projected_with_policy(
                &flow_executor_cond,
                backend,
                &flow_model,
                tokens,
                cond_tokens,
                context,
                projected_context,
                cos_phase,
                sin_phase,
                &attention_policy);
        } else {
            status = trellis_dit_flow_executor_init_single_with_policy(
                &flow_executor_cond,
                backend,
                &flow,
                tokens,
                cond_tokens,
                context,
                cos_phase,
                sin_phase,
                &attention_policy);
        }
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "sparse structure: flow executor init failed: %s\n", trellis_status_string(status));
            goto cleanup;
        }
    } else {
        use_cfg_batch = 1;
    }

    for (int step = 0; step < steps; ++step) {
        const int64_t step_start_us = ggml_time_us();
        const int64_t flow_start_us = ggml_time_us();
        const float t = pairs[2 * step + 0];
        const float t_prev = pairs[2 * step + 1];
        if (use_cfg_batch) {
            TRELLIS_DEBUG("sparse structure: step %d/%d running fused cfg flow graph t=%.6g", step + 1, steps, t);
            status = trellis_dit_flow_executor_run_cfg_batch(
                &flow_executor_cfg,
                latent,
                t,
                pred_pos,
                pred_neg);
        } else {
            TRELLIS_DEBUG("sparse structure: step %d/%d reusing one cond/uncond flow graph t=%.6g", step + 1, steps, t);
            status = trellis_dit_flow_executor_set_single_conditioning(
                &flow_executor_cond,
                context,
                projected_context);
            if (status == TRELLIS_STATUS_OK) {
                status = trellis_dit_flow_executor_run_single(
                    &flow_executor_cond,
                    latent,
                    t,
                    pred_pos);
            }
            if (status == TRELLIS_STATUS_OK) {
                status = trellis_dit_flow_executor_set_single_conditioning(
                    &flow_executor_cond,
                    neg_context,
                    neg_projected_context);
            }
            if (status == TRELLIS_STATUS_OK) {
                status = trellis_dit_flow_executor_run_single(
                    &flow_executor_cond,
                    latent,
                    t,
                    pred_neg);
            }
        }
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "sparse structure: flow step %d failed: %s\n", step + 1, trellis_status_string(status));
            goto cleanup;
        }

        if (t >= 0.6f && t <= 1.0f) {
            trellis_flow_cfg_rescale_combine_f32(latent, pred_pos, pred_neg, 1, latent_count, 1e-5f, t, 7.5f, 0.7f, pred);
        } else {
            memcpy(pred, pred_pos, latent_count * sizeof(float));
        }
        trellis_flow_euler_step_f32(latent, pred, latent_count, 1e-5f, t, t_prev, next, x0);
        memcpy(latent, next, latent_count * sizeof(float));
        token_channels_to_ncdhw(latent, latent_ncdhw, flow.in_channels, latent_size);
        voxel_denoise_us += ggml_time_us() - flow_start_us;

        const int need_decode = result != NULL && step + 1 == steps;
        if (!need_decode) {
            char progress_detail[128];
            snprintf(
                progress_detail,
                sizeof(progress_detail),
                "t=%.6g->%.6g flow-only",
                t,
                t_prev);
            trellis_progress_steps(
                "sparse structure",
                step + 1,
                steps,
                ggml_time_us() - step_start_us,
                progress_detail);
            continue;
        }

        const int64_t decode_start_us = ggml_time_us();
        float * logits = NULL;
        int output_size = 0;
        TRELLIS_DEBUG("sparse structure: step %d/%d decoding sparse structure", step + 1, steps);
        status = trellis_ss_decoder_forward_f32_host(&decoder, latent_ncdhw, backend, 1, latent_size, &logits, &output_size);
        if (status != TRELLIS_STATUS_OK) {
            fprintf(stderr, "sparse structure: decoder step %d failed: %s\n", step + 1, trellis_status_string(status));
            free(logits);
            goto cleanup;
        }
        const size_t logits_count = (size_t) output_size * (size_t) output_size * (size_t) output_size;
        float * frame_logits = NULL;
        int frame_resolution = output_size;
        float frame_threshold = threshold;
        if (sparse_resolution != output_size) {
            if (!max_pool_logits_to_resolution(logits, output_size, threshold, sparse_resolution, &frame_logits)) {
                fprintf(stderr, "sparse structure: cannot pool decoder output %d^3 to sparse resolution %d^3\n",
                    output_size,
                    sparse_resolution);
                free(logits);
                goto cleanup;
            }
            frame_resolution = sparse_resolution;
            frame_threshold = 0.5f;
        }
        int32_t * frame_coords = NULL;
        int64_t frame_coord_count = 0;
        if (!collect_voxel_coords_xyz(
                frame_logits != NULL ? frame_logits : logits,
                frame_resolution,
                frame_threshold,
                &frame_coords,
                &frame_coord_count)) {
            fprintf(stderr, "sparse structure: voxel coord extraction failed at step %d\n", step + 1);
            free(frame_logits);
            free(logits);
            goto cleanup;
        }
        int voxel_count = frame_coord_count > INT32_MAX ? INT32_MAX : (int) frame_coord_count;
        if (step + 1 == steps &&
            !set_sparse_structure_result(result, frame_coords, frame_coord_count, frame_resolution, context, cond_tokens)) {
            fprintf(stderr, "sparse structure: failed to retain final coords/condition for structured latent flow\n");
            free(frame_coords);
            free(frame_logits);
            free(logits);
            goto cleanup;
        }
        if (step + 1 == steps && result != NULL) {
            result->projected_conditioning = flow_model.projection.enabled;
        }
        float logit_min = logits_count == 0 ? 0.0f : logits[0];
        float logit_max = logits_count == 0 ? 0.0f : logits[0];
        double logit_sum = 0.0;
        for (size_t i = 0; i < logits_count; ++i) {
            if (logits[i] < logit_min) logit_min = logits[i];
            if (logits[i] > logit_max) logit_max = logits[i];
            logit_sum += logits[i];
        }
        const double logit_mean = logits_count == 0 ? 0.0 : logit_sum / (double) logits_count;
        free(frame_coords);
        free(frame_logits);
        free(logits);
        voxel_decode_us += ggml_time_us() - decode_start_us;
        char progress_detail[256];
        snprintf(
            progress_detail,
            sizeof(progress_detail),
            "t=%.6g->%.6g voxels=%d %d^3 logits[min=%.6g mean=%.6g max=%.6g]",
            t,
            t_prev,
            voxel_count,
            frame_resolution,
            logit_min,
            logit_mean,
            logit_max);
        trellis_progress_steps(
            "sparse structure",
            step + 1,
            steps,
            ggml_time_us() - step_start_us,
            progress_detail);
    }

    TRELLIS_INFO("sparse structure: decoded final voxel frame (threshold=%.6g)", threshold);
    trellis_perf_stage_log("stage1_voxel_denoise", voxel_denoise_us);
    trellis_perf_stage_log("stage1_voxel_decode", voxel_decode_us);
    rc = 0;

cleanup:
    if (rc != 0 && result != NULL) {
        trellis_sparse_structure_result_free(result);
    }
    free(context);
    free(projected_context);
    free(neg_context);
    free(neg_projected_context);
    free(latent);
    free(pred_pos);
    free(pred_neg);
    free(pred);
    free(next);
    free(x0);
    free(latent_ncdhw);
    free(pairs);
    free(cos_phase);
    free(sin_phase);
    trellis_dit_flow_executor_free(&flow_executor_cfg);
    trellis_dit_flow_executor_free(&flow_executor_cond);
    if (owns_decoder_store) {
        trellis_tensor_store_free(&decoder_store);
    }
    if (owns_flow_store) {
        trellis_tensor_store_free(&flow_store);
    }
    return rc;
}

trellis_status trellis_pipeline_run_sparse_structure(
    const trellis_sparse_structure_options * options,
    trellis_sparse_structure_result * result) {
    if (options == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const trellis_backend_context * backend = options->backend != NULL ? options->backend : options->cuda;
    if (backend == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_ggml_attention_policy attention_policy = options->attention_policy;
    if (!trellis_ggml_attention_policy_is_valid(&attention_policy)) {
        attention_policy = (trellis_ggml_attention_policy)
            TRELLIS_GGML_ATTENTION_POLICY_INIT;
        if (options->use_ggml_flash_attn) {
            attention_policy.mode = TRELLIS_GGML_ATTENTION_MODE_FLASH;
        }
    }
    int rc = run_sparse_structure_image(
        backend,
        options->model_dir,
        options->flow_path,
        options->decoder_path,
        options->dino_dir,
        options->image_path,
        options->latent_size,
        options->steps,
        options->cond_resolution,
        options->sparse_resolution,
        options->seed,
        options->flow_blocks_override,
        options->flow_block_parts_override,
        options->flow_no_rope,
        options->projected_conditioning,
        options->emulate_bf16_blocks,
        &attention_policy,
        options->voxel_threshold,
        options->camera_angle_x,
        options->camera_distance,
        options->mesh_scale,
        options->cache,
        result);
    return rc == 0 ? TRELLIS_STATUS_OK : TRELLIS_STATUS_ERROR;
}

static int readable_file(const char * path) {
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    FILE * file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    fclose(file);
    return 1;
}

static int resolve_pixal_naf_path(
    const trellis_image_condition_options * options,
    char * path,
    size_t path_size) {
    if (options == NULL || path == NULL || path_size == 0) {
        return 0;
    }
    const char * explicit_path = options->naf_path;
    if (explicit_path == NULL || explicit_path[0] == '\0') {
        explicit_path = getenv("TRELLIS_NAF_PATH");
    }
    if (explicit_path != NULL && explicit_path[0] != '\0') {
        const int n = snprintf(path, path_size, "%s", explicit_path);
        return n >= 0 && (size_t) n < path_size && readable_file(path);
    }
    if (options->model_dir == NULL || options->model_dir[0] == '\0') {
        return 0;
    }
    const char * candidates[] = {
        "ckpts/naf_release.safetensors",
        "naf_release.safetensors",
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if (trellis_make_model_path(
                options->model_dir,
                candidates[i],
                path,
                path_size) == TRELLIS_STATUS_OK && readable_file(path)) {
            return 1;
        }
    }
    return 0;
}

static int checked_mul_size_local(size_t a, size_t b, size_t * out) {
    if (out == NULL || (a != 0 && b > SIZE_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static trellis_status run_pixal_naf_high_projection(
    const trellis_image_condition_options * options,
    const trellis_backend_context * backend,
    const char * naf_path,
    const float * guide_nchw,
    const float * dino_patch_values,
    int patch_side,
    const trellis_pixal_camera * camera,
    float ** projected_out) {
    if (projected_out != NULL) {
        *projected_out = NULL;
    }
    if (options == NULL || backend == NULL || naf_path == NULL || guide_nchw == NULL ||
        dino_patch_values == NULL || patch_side < TRELLIS_PIXAL_NAF_ATTENTION_KERNEL_SIZE ||
        camera == NULL || projected_out == NULL || options->naf_target_resolution <= 0 ||
        options->projection_n_coords <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    const int query_side = options->naf_target_resolution;
    size_t query_count = 0;
    size_t key_count = 0;
    size_t output_count = 0;
    if (!checked_mul_size_local((size_t) query_side, (size_t) query_side, &query_count) ||
        !checked_mul_size_local(
            query_count,
            TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS,
            &query_count) ||
        !checked_mul_size_local((size_t) patch_side, (size_t) patch_side, &key_count) ||
        !checked_mul_size_local(
            key_count,
            TRELLIS_PIXAL_NAF_ATTENTION_QUERY_CHANNELS,
            &key_count) ||
        (uint64_t) options->projection_n_coords > (uint64_t) SIZE_MAX ||
        !checked_mul_size_local(
            (size_t) options->projection_n_coords,
            TRELLIS_PIXAL_NAF_ATTENTION_VALUE_CHANNELS,
            &output_count) ||
        query_count > SIZE_MAX / sizeof(float) || key_count > SIZE_MAX / sizeof(float) ||
        output_count > SIZE_MAX / sizeof(float)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    float * queries = (float *) malloc(query_count * sizeof(float));
    float * keys = (float *) malloc(key_count * sizeof(float));
    float * high_projected = (float *) malloc(output_count * sizeof(float));
    if (queries == NULL || keys == NULL || high_projected == NULL) {
        free(high_projected);
        free(keys);
        free(queries);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_tensor_store naf_store;
    trellis_pixal_naf_ggml_weights naf_weights;
    memset(&naf_store, 0, sizeof(naf_store));
    memset(&naf_weights, 0, sizeof(naf_weights));
    trellis_status status = TRELLIS_STATUS_ERROR;
    if (!trellis_load_tensor_store_f32(
            backend,
            "Pixal3D NAF image encoder",
            naf_path,
            false,
            16,
            &naf_store,
            NULL)) {
        goto cleanup;
    }
    char issue[256];
    issue[0] = '\0';
    status = trellis_pixal_naf_bind_ggml_weights(
        &naf_store,
        &naf_weights,
        issue,
        sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "Pixal3D NAF: weight bind failed: %s%s%s",
            trellis_status_string(status),
            issue[0] == '\0' ? "" : " ",
            issue);
        goto cleanup;
    }

    status = trellis_pixal_naf_query_key_forward_ggml_host(
        backend,
        &naf_weights,
        guide_nchw,
        1,
        options->cond_resolution,
        options->cond_resolution,
        query_side,
        query_side,
        patch_side,
        patch_side,
        queries,
        query_count,
        keys,
        key_count);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "Pixal3D NAF: image encoder failed: %s",
            trellis_status_string(status));
        goto cleanup;
    }

    const trellis_pixal_naf_attention_desc desc = {
        .batch = 1,
        .query_h = query_side,
        .query_w = query_side,
        .feature_h = patch_side,
        .feature_w = patch_side,
        .image_resolution = options->cond_resolution,
        .grid_resolution = options->projection_grid_resolution,
    };
    if (backend->kind == TRELLIS_BACKEND_CUDA) {
        status = trellis_pixal_naf_attention_project_sparse_cuda_host_f32(
            queries,
            keys,
            dino_patch_values,
            options->projection_coords_bxyz,
            options->projection_n_coords,
            camera,
            &desc,
            backend->device,
            high_projected,
            output_count);
    } else if (backend->kind == TRELLIS_BACKEND_VULKAN && options->sparse_backend != NULL) {
        status = trellis_pixal_naf_attention_project_sparse_vulkan_f32(
            (trellis_sparse_backend *) options->sparse_backend,
            queries,
            keys,
            dino_patch_values,
            options->projection_coords_bxyz,
            options->projection_n_coords,
            camera,
            &desc,
            high_projected,
            output_count);
    } else {
        status = TRELLIS_STATUS_NOT_IMPLEMENTED;
    }
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "Pixal3D NAF: fused neighborhood attention/projection failed: %s",
            trellis_status_string(status));
        goto cleanup;
    }

    *projected_out = high_projected;
    high_projected = NULL;

cleanup:
    trellis_tensor_store_free(&naf_store);
    free(high_projected);
    free(keys);
    free(queries);
    return status;
}

trellis_status trellis_pipeline_run_image_condition(
    const trellis_image_condition_options * options,
    trellis_image_condition_result * result) {
    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }
    if (options == NULL || result == NULL || options->dino_dir == NULL ||
        options->image_path == NULL || options->cond_resolution <= 0 ||
        (options->projection_channels != 0 && options->projection_channels != 1024 &&
         options->projection_channels != 2048)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const trellis_backend_context * backend = options->backend != NULL ? options->backend : options->cuda;
    if (backend == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    float * cond = NULL;
    float * guide = NULL;
    int cond_tokens = 0;
    if (!run_dino_condition(
            backend,
            options->cache,
            options->dino_dir,
            options->image_path,
            options->cond_resolution,
            options->projection_channels > 0,
            options->projection_channels == 2048 ? &guide : NULL,
            NULL,
            &cond,
            &cond_tokens)) {
        free(cond);
        free(guide);
        return TRELLIS_STATUS_ERROR;
    }
    if (options->projection_channels > 0) {
        const int special_tokens = 5;
        const int patch_side = options->cond_resolution / 16;
        const int patch_tokens = patch_side * patch_side;
        if (options->cond_resolution % 16 != 0 ||
            options->projection_grid_resolution <= 0 ||
            options->projection_coords_bxyz == NULL || options->projection_n_coords <= 0 ||
            cond_tokens != special_tokens + patch_tokens ||
            (uint64_t) options->projection_n_coords >
                (uint64_t) SIZE_MAX / (uint64_t) options->projection_channels) {
            free(cond);
            free(guide);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        const float resolved_camera_angle_x = options->camera_angle_x > 0.0f ?
            options->camera_angle_x : 0.8575560450553894f;
        const float resolved_mesh_scale = options->mesh_scale > 0.0f ?
            options->mesh_scale : 1.0f;
        float resolved_camera_distance = 0.0f;
        if (trellis_pixal_resolve_camera_distance(
                resolved_camera_angle_x,
                resolved_mesh_scale,
                options->camera_distance,
                &resolved_camera_distance) != TRELLIS_STATUS_OK) {
            free(cond);
            free(guide);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        trellis_pixal_camera camera = {
            .camera_angle_x = resolved_camera_angle_x,
            .distance = resolved_camera_distance,
            .mesh_scale = resolved_mesh_scale,
        };
        const size_t low_count = (size_t) options->projection_n_coords * 1024u;
        float * global = (float *) malloc((size_t) special_tokens * 1024u * sizeof(float));
        float * low_projected = (float *) malloc(low_count * sizeof(float));
        if (global == NULL || low_projected == NULL) {
            free(global);
            free(low_projected);
            free(cond);
            free(guide);
            return TRELLIS_STATUS_OUT_OF_MEMORY;
        }
        memcpy(global, cond, (size_t) special_tokens * 1024u * sizeof(float));
        const float * dino_patch_values = cond + (size_t) special_tokens * 1024u;
        trellis_status status = trellis_pixal_project_patch_features_sparse_f32(
            dino_patch_values,
            patch_side,
            patch_side,
            1024,
            options->cond_resolution,
            options->projection_grid_resolution,
            &camera,
            options->projection_coords_bxyz,
            options->projection_n_coords,
            low_projected,
            low_count);
        if (status != TRELLIS_STATUS_OK) {
            free(global);
            free(low_projected);
            free(cond);
            free(guide);
            return status;
        }
        if (options->projection_channels == 2048) {
            char naf_path[4096];
            if (!resolve_pixal_naf_path(options, naf_path, sizeof(naf_path))) {
                TRELLIS_ERROR(
                    "Pixal3D requires NAF safetensors; pass --naf FILE or place naf_release.safetensors under the model/ckpts directory (convert with tools/convert_naf_weights.py)");
                free(global);
                free(low_projected);
                free(cond);
                free(guide);
                return TRELLIS_STATUS_NOT_FOUND;
            }
            float * high_projected = NULL;
            const int64_t naf_start_us = ggml_time_us();
            status = run_pixal_naf_high_projection(
                options,
                backend,
                naf_path,
                guide,
                dino_patch_values,
                patch_side,
                &camera,
                &high_projected);
            trellis_perf_stage_log(
                options->naf_target_resolution == 1024 ?
                    "pixal_naf_1024" : "pixal_naf_512",
                ggml_time_us() - naf_start_us);
            if (status != TRELLIS_STATUS_OK) {
                free(high_projected);
                free(global);
                free(low_projected);
                free(cond);
                free(guide);
                return status;
            }
            const size_t projected_count =
                (size_t) options->projection_n_coords * 2048u;
            float * projected = (float *) malloc(projected_count * sizeof(float));
            if (projected == NULL) {
                free(high_projected);
                free(global);
                free(low_projected);
                free(cond);
                free(guide);
                return TRELLIS_STATUS_OUT_OF_MEMORY;
            }
            for (int64_t token = 0; token < options->projection_n_coords; ++token) {
                memcpy(
                    projected + (size_t) token * 2048u,
                    low_projected + (size_t) token * 1024u,
                    1024u * sizeof(float));
                memcpy(
                    projected + (size_t) token * 2048u + 1024u,
                    high_projected + (size_t) token * 1024u,
                    1024u * sizeof(float));
            }
            free(high_projected);
            free(low_projected);
            free(cond);
            free(guide);
            result->cond = global;
            result->cond_tokens = special_tokens;
            result->projected = projected;
            result->projected_tokens = options->projection_n_coords;
            result->projected_channels = 2048;
            result->resolution = options->cond_resolution;
            return TRELLIS_STATUS_OK;
        }
        free(cond);
        free(guide);
        result->cond = global;
        result->cond_tokens = special_tokens;
        result->projected = low_projected;
        result->projected_tokens = options->projection_n_coords;
        result->projected_channels = 1024;
        result->resolution = options->cond_resolution;
        return TRELLIS_STATUS_OK;
    }
    free(guide);
    result->cond = cond;
    result->cond_tokens = cond_tokens;
    result->resolution = options->cond_resolution;
    return TRELLIS_STATUS_OK;
}
