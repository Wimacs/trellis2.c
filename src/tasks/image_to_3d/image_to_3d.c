#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "trellis.h"
#include "adapter.h"
#include "pixal_projection.h"
#include "trellis_platform.h"
#include "sparse/trellis_sparse_backend.h"
#include "trellis_model_package.h"
#include "image_to_3d_internal.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void trellis_perf_stage_log(const char * name, int64_t elapsed_us) {
    TRELLIS_INFO(
        "perf_stage name=%s ms=%.3f",
        name != NULL ? name : "unknown",
        elapsed_us <= 0 ? 0.0 : (double) elapsed_us / 1000.0);
}

static size_t model_cache_budget_bytes_from_options(const trellis_image_to_gltf_options * options) {
    long long mib = options != NULL ? (long long) options->model_cache_budget_mib : 0;
    if (mib <= 0) {
        const char * env = getenv("TRELLIS_MODEL_CACHE_BUDGET_MIB");
        if (env != NULL && env[0] != '\0') {
            char * end = NULL;
            errno = 0;
            long long parsed = strtoll(env, &end, 10);
            if (errno == 0 && end != env && *end == '\0' && parsed > 0) {
                mib = parsed;
            }
        }
    }
    if (mib <= 0) {
        return 0;
    }
    const long long max_mib = (long long) (SIZE_MAX / (1024u * 1024u));
    if (mib > max_mib) {
        mib = max_mib;
    }
    return (size_t) mib * 1024u * 1024u;
}

static int pipeline_component_path(
    const trellis_model_package * package,
    const char * role,
    char path[4096]) {
    const trellis_status status = trellis_model_package_resolve_component_path(
        package, role, path, 4096);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "model package '%s': required component role '%s' is unavailable: %s",
            package != NULL && package->id != NULL ? package->id : "unknown",
            role != NULL ? role : "unknown",
            trellis_status_string(status));
        return 0;
    }
    return 1;
}

static trellis_ggml_attention_policy pipeline_component_attention_policy(
    const trellis_image_to_gltf_options * options,
    const trellis_model_component_instance * component) {
    trellis_ggml_attention_policy policy = TRELLIS_GGML_ATTENTION_POLICY_INIT;
    int emulate_bf16_blocks = 0;
    if (component != NULL) {
        (void) trellis_image_to_3d_component_execution_policy(
            component,
            &policy,
            &emulate_bf16_blocks);
    }
    if (options->no_ggml_flash_attn) {
        policy.mode = TRELLIS_GGML_ATTENTION_MODE_EXPLICIT;
    } else if (options->use_ggml_flash_attn &&
               policy.mode == TRELLIS_GGML_ATTENTION_MODE_EXPLICIT) {
        /* A force-Flash CLI override has no package K/V dtype to preserve. */
        policy.mode = TRELLIS_GGML_ATTENTION_MODE_FLASH;
    }
    return policy;
}

static const char * pipeline_attention_policy_name(
    const trellis_ggml_attention_policy * policy) {
    if (policy != NULL &&
        policy->mode == TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16) {
        return "flash/bf16";
    }
    if (policy != NULL &&
        policy->mode == TRELLIS_GGML_ATTENTION_MODE_FLASH) {
        return "flash/f16";
    }
    return "explicit";
}

static int pipeline_component_emulates_bf16(
    const trellis_image_to_gltf_options * options,
    const trellis_model_component_instance * component) {
    return options->emulate_bf16_blocks ||
        (component != NULL && component->execution.emulate_bf16_blocks);
}

extern unsigned char * stbi_load(char const * filename, int * x, int * y, int * comp, int req_comp);
extern void stbi_image_free(void * retval_from_stbi_load);
extern int stbi_write_png(char const * filename, int w, int h, int comp, const void * data, int stride_in_bytes);

static const trellis_component_status g_status[] = {
    {
        TRELLIS_COMPONENT_SPARSE_STRUCTURE_FLOW,
        "SparseStructureFlowModel",
        true,
        "dense-token DiT operators, RoPE self-attention path, checkpoint binding, real-weight CUDA forward, and CLI sampler wiring are implemented",
    },
    {
        TRELLIS_COMPONENT_SPARSE_STRUCTURE_DECODER,
        "SparseStructureDecoder",
        true,
        "checkpoint binding, Conv3D, ChannelLayerNorm3D, SiLU, residual add, pixel-shuffle, real-weight CUDA decode, and voxel frame export are implemented",
    },
    {
        TRELLIS_COMPONENT_SHAPE_SLAT_FLOW,
        "SLatFlowModel",
        true,
        "sparse coordinate host semantics, dense transformer operators, and ggml FlashAttention SDPA are implemented",
    },
    {
        TRELLIS_COMPONENT_TEX_SLAT_FLOW,
        "Texture SLatFlowModel",
        true,
        "same pure ggml DiT path as shape SLat flow; texture noise plus normalized shape concat conditioning is wired in the CLI pipeline",
    },
    {
        TRELLIS_COMPONENT_SHAPE_SLAT_DECODER,
        "FlexiDualGridVaeDecoder",
        true,
        "shared SparseUnetVaeDecoder CUDA sparse-conv body, predicted subdivision guides, and FlexiDualGrid mesh extraction are implemented",
    },
    {
        TRELLIS_COMPONENT_TEX_SLAT_DECODER,
        "SparseUnetVaeDecoder",
        true,
        "shared SparseUnetVaeDecoder CUDA sparse-conv body is implemented with guide_subs from the shape decoder and 6-channel PBR voxel output",
    },
    {
        TRELLIS_COMPONENT_DINOV3_IMAGE_ENCODER,
        "DINOv3 image encoder",
        true,
        "checkpoint binding, image preprocessing, and full ggml image encoder graph are implemented for CLI conditioning",
    },
    {
        TRELLIS_COMPONENT_BIREFNET_BACKGROUND_REMOVAL,
        "BiRefNet background removal",
        true,
        "GGUF loading, Swin/BiRefNet graph, CLI pre-mask PNG wiring, and ggml deformable convolution backends are implemented",
    },
    {
        TRELLIS_COMPONENT_OVOXEL_POSTPROCESS,
        "O-Voxel postprocess",
        true,
        "vkmesh topology cleanup, narrow-band remesh, simplify, and Vulkan compute UV-space PBR texture bake are implemented",
    },
};

size_t trellis_component_status_count(void) {
    return sizeof(g_status) / sizeof(g_status[0]);
}

const trellis_component_status * trellis_component_status_at(size_t index) {
    if (index >= trellis_component_status_count()) {
        return 0;
    }
    return &g_status[index];
}

static const char * sparse_backend_kind_name(trellis_sparse_backend_kind kind) {
    switch (kind) {
        case TRELLIS_SPARSE_BACKEND_CUDA: return "cuda";
        case TRELLIS_SPARSE_BACKEND_CPU: return "cpu";
        case TRELLIS_SPARSE_BACKEND_VULKAN: return "vulkan";
        default: return "unknown";
    }
}

static trellis_status sparse_backend_kind_from_graph_backend(
    trellis_backend_kind graph_kind,
    trellis_sparse_backend_kind * sparse_kind_out) {
    if (sparse_kind_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    switch (graph_kind) {
        case TRELLIS_BACKEND_CUDA:
            *sparse_kind_out = TRELLIS_SPARSE_BACKEND_CUDA;
            return TRELLIS_STATUS_OK;
        case TRELLIS_BACKEND_VULKAN:
            *sparse_kind_out = TRELLIS_SPARSE_BACKEND_VULKAN;
            return TRELLIS_STATUS_OK;
        default:
            return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
}

static int path_has_ext(const char * path, const char * ext) {
    if (path == NULL || ext == NULL) {
        return 0;
    }
    const char * dot = strrchr(path, '.');
    if (dot == NULL) {
        return 0;
    }
    ++dot;
    while (*dot != '\0' && *ext != '\0') {
        if (tolower((unsigned char) *dot) != tolower((unsigned char) *ext)) {
            return 0;
        }
        ++dot;
        ++ext;
    }
    return *dot == '\0' && *ext == '\0';
}

static trellis_status image_has_transparent_alpha(
    const char * image_path,
    int * has_transparent_alpha_out) {
    if (image_path == NULL || image_path[0] == '\0' || has_transparent_alpha_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *has_transparent_alpha_out = 0;

    int width = 0;
    int height = 0;
    int components = 0;
    unsigned char * rgba = stbi_load(image_path, &width, &height, &components, 4);
    if (rgba == NULL || width <= 0 || height <= 0 ||
        (size_t) width > SIZE_MAX / (size_t) height) {
        TRELLIS_ERROR("image prep: failed to inspect alpha channel: %s", image_path);
        stbi_image_free(rgba);
        return TRELLIS_STATUS_IO_ERROR;
    }

    const size_t pixels = (size_t) width * (size_t) height;
    for (size_t i = 0; i < pixels; ++i) {
        if (rgba[i * 4u + 3u] != 255u) {
            *has_transparent_alpha_out = 1;
            break;
        }
    }
    stbi_image_free(rgba);
    return TRELLIS_STATUS_OK;
}

static int resolve_packaged_birefnet_path(
    const char * model_dir,
    const char * dino_dir,
    char * path,
    size_t path_size) {
    if (model_dir == NULL || model_dir[0] == '\0' || path == NULL || path_size == 0) {
        return 0;
    }
    const char * environment_path = getenv("TRELLIS_BIREFNET_PATH");
    if (environment_path != NULL && environment_path[0] != '\0') {
        const int n = snprintf(path, path_size, "%s", environment_path);
        if (n >= 0 && (size_t) n < path_size && trellis_access_read(path)) {
            return 1;
        }
    }
    static const char * candidates[] = {
        "ckpts/BiRefNet-F16.gguf",
        "BiRefNet-F16.gguf",
        "BiRefNet/BiRefNet-F16.gguf",
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if (trellis_make_model_path(
                model_dir,
                candidates[i],
                path,
                path_size) == TRELLIS_STATUS_OK &&
            trellis_access_read(path)) {
            return 1;
        }
    }
    if (dino_dir != NULL && dino_dir[0] != '\0') {
        static const char * dino_sibling_candidates[] = {
            "../BiRefNet/BiRefNet-F16.gguf",
            "../BiRefNet-F16.gguf",
        };
        for (size_t i = 0;
             i < sizeof(dino_sibling_candidates) / sizeof(dino_sibling_candidates[0]);
             ++i) {
            if (trellis_make_model_path(
                    dino_dir,
                    dino_sibling_candidates[i],
                    path,
                    path_size) == TRELLIS_STATUS_OK &&
                trellis_access_read(path)) {
                return 1;
            }
        }
    }
    path[0] = '\0';
    return 0;
}

static int convert_webp_to_png_ffmpeg(const char * input_path, const char * output_path) {
    char * argv[] = {
        (char *) "ffmpeg",
        (char *) "-y",
        (char *) "-hide_banner",
        (char *) "-loglevel",
        (char *) "error",
        (char *) "-i",
        (char *) input_path,
        (char *) "-frames:v",
        (char *) "1",
        (char *) output_path,
        NULL,
    };
    return trellis_run_process_search_path(argv);
}

static void unlink_if_set(const char * path) {
    if (path != NULL && path[0] != '\0') {
        trellis_unlink(path);
    }
}

static trellis_status apply_birefnet_background_removal(
    const char * input_path,
    const char * gguf_path,
    const char * output_path,
    trellis_backend_kind backend_kind,
    int device) {
    if (input_path == NULL || gguf_path == NULL || output_path == NULL ||
        input_path[0] == '\0' || gguf_path[0] == '\0' || output_path[0] == '\0') {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    int width = 0;
    int height = 0;
    int comp = 0;
    unsigned char * rgba = stbi_load(input_path, &width, &height, &comp, 4);
    if (rgba == NULL || width <= 0 || height <= 0) {
        TRELLIS_ERROR("BiRefNet: failed to load image %s", input_path);
        stbi_image_free(rgba);
        return TRELLIS_STATUS_IO_ERROR;
    }

    TRELLIS_INFO(
        "BiRefNet: loading background-removal GGUF on %s device=%d: %s",
        trellis_backend_kind_name(backend_kind),
        device,
        gguf_path);
    trellis_birefnet_model model;
    memset(&model, 0, sizeof(model));
    trellis_status status = trellis_birefnet_load_gguf_with_backend(&model, gguf_path, backend_kind, device);
    unsigned char * mask = NULL;
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_birefnet_compute_mask_u8(&model, rgba, width, height, &mask);
    }
    if (status == TRELLIS_STATUS_OK) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t i = (size_t) y * (size_t) width + (size_t) x;
                rgba[i * 4u + 3u] = (unsigned char) (((unsigned int) rgba[i * 4u + 3u] * (unsigned int) mask[i] + 127u) / 255u);
            }
        }
        if (!stbi_write_png(output_path, width, height, 4, rgba, width * 4)) {
            status = TRELLIS_STATUS_IO_ERROR;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        TRELLIS_INFO("BiRefNet: wrote masked PNG: %s", output_path);
    } else {
        TRELLIS_ERROR("BiRefNet: background removal failed: %s", trellis_status_string(status));
    }
    free(mask);
    trellis_birefnet_free(&model);
    stbi_image_free(rgba);
    return status;
}

static int mkdir_p(const char * path) {
    return trellis_mkdir_p(path);
}

static int mkdir_parent(const char * path) {
    return trellis_mkdir_parent(path);
}

static int make_shape_decoder_path(
    const char * model_dir,
    const char * override_path,
    char * out,
    size_t out_size) {
    if (override_path != NULL && override_path[0] != '\0') {
        int n = snprintf(out, out_size, "%s", override_path);
        return n >= 0 && (size_t) n < out_size;
    }
    return trellis_make_model_path(
        model_dir,
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        out,
        out_size) == TRELLIS_STATUS_OK;
}

static int make_texture_decoder_path(
    const char * model_dir,
    const char * override_path,
    char * out,
    size_t out_size) {
    if (override_path != NULL && override_path[0] != '\0') {
        int n = snprintf(out, out_size, "%s", override_path);
        return n >= 0 && (size_t) n < out_size;
    }
    return trellis_make_model_path(
        model_dir,
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        out,
        out_size) == TRELLIS_STATUS_OK;
}

static int load_shape_decoder(
    const trellis_cuda_context * cuda,
    const char * path,
    trellis_tensor_store * store,
    trellis_sparse_unet_vae_decoder_weights * decoder) {
    if (!trellis_load_tensor_store_f32(cuda, "FlexiDualGridVaeDecoder", path, true, 64, store, NULL)) {
        return 0;
    }
    char issue[256];
    trellis_status status = trellis_flexi_dual_grid_vae_decoder_bind_weights(store, decoder, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("FlexiDualGridVaeDecoder: bind failed: %s%s%s",
            trellis_status_string(status),
            issue[0] == '\0' ? "" : " ",
            issue);
        return 0;
    }
    TRELLIS_INFO("FlexiDualGridVaeDecoder: ready");
    return 1;
}

static int load_texture_decoder(
    const trellis_cuda_context * cuda,
    const char * path,
    trellis_tensor_store * store,
    trellis_sparse_unet_vae_decoder_weights * decoder) {
    if (!trellis_load_tensor_store_f32(cuda, "Texture SparseUnetVaeDecoder", path, true, 64, store, NULL)) {
        return 0;
    }
    char issue[256];
    trellis_status status = trellis_tex_slat_decoder_bind_weights(store, decoder, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("Texture SparseUnetVaeDecoder: bind failed: %s%s%s",
            trellis_status_string(status),
            issue[0] == '\0' ? "" : " ",
            issue);
        return 0;
    }
    TRELLIS_INFO("Texture SparseUnetVaeDecoder: ready");
    return 1;
}

trellis_status trellis_pipeline_decode_shape_latent_mesh(
    const trellis_pipeline_mesh_options * options,
    trellis_sparse_c2s_guides * subs_out,
    trellis_mesh_host * mesh_out) {
    if (mesh_out != NULL) {
        memset(mesh_out, 0, sizeof(*mesh_out));
    }
    if (options == NULL || mesh_out == NULL || options->model_dir == NULL ||
        options->latent == NULL || options->latent->coords_bxyz == NULL ||
        options->latent->feats == NULL || options->latent->n_coords <= 0 ||
        options->latent->channels != 32 ||
        (options->sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA && options->cuda == NULL)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = TRELLIS_STATUS_ERROR;
    char decoder_path[4096];
    trellis_tensor_store decoder_store;
    trellis_sparse_unet_vae_decoder_weights decoder;
    trellis_backend_context cpu_weights_backend;
    memset(&decoder_store, 0, sizeof(decoder_store));
    memset(&decoder, 0, sizeof(decoder));
    memset(&cpu_weights_backend, 0, sizeof(cpu_weights_backend));
    const trellis_sparse_unet_vae_decoder_weights * decoder_ptr = &decoder;
    int32_t * out_coords = NULL;
    float * out_feats = NULL;

    const trellis_backend_context * weight_backend = options->cuda;
    int owns_weight_backend = 0;
    int owns_decoder_store = 0;
    if (options->cache != NULL) {
        status = trellis_pipeline_model_cache_get_shape_decoder(
            options->cache,
            options->model_dir,
            options->decoder_override_path,
            &decoder_ptr);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("pipeline mesh: cache shape decoder load failed: %s", trellis_status_string(status));
            goto cleanup;
        }
    } else {
        if (!make_shape_decoder_path(options->model_dir, options->decoder_override_path, decoder_path, sizeof(decoder_path))) {
            TRELLIS_ERROR("pipeline mesh: failed to build shape decoder path");
            goto cleanup;
        }
        if (options->sparse_backend_kind != TRELLIS_SPARSE_BACKEND_CUDA) {
            status = trellis_backend_init(&cpu_weights_backend, TRELLIS_BACKEND_CPU, 0);
            if (status != TRELLIS_STATUS_OK) {
                TRELLIS_ERROR("pipeline mesh: CPU decoder weight backend init failed: %s", trellis_status_string(status));
                goto cleanup;
            }
            weight_backend = &cpu_weights_backend;
            owns_weight_backend = 1;
        }
        if (!load_shape_decoder(weight_backend, decoder_path, &decoder_store, &decoder)) {
            goto cleanup;
        }
        owns_decoder_store = 1;
    }

    int64_t n_use = options->latent->n_coords;
    if (options->decode_max_input_tokens > 0 && options->decode_max_input_tokens < n_use) {
        n_use = options->decode_max_input_tokens;
        TRELLIS_WARN(
            "pipeline mesh: truncating decoder input tokens %lld -> %lld",
            (long long) options->latent->n_coords,
            (long long) n_use);
    }

    int64_t n_out = 0;
    int channels_out = 0;
    trellis_sparse_unet_vae_decoder_forward_options forward_options;
    memset(&forward_options, 0, sizeof(forward_options));
    forward_options.backend_kind = options->sparse_backend_kind;
    forward_options.device = options->sparse_device;
    forward_options.max_levels = options->decode_max_levels;
    forward_options.sparse_backend = options->sparse_backend;
    forward_options.return_subs = subs_out;
    status = trellis_sparse_unet_vae_decoder_forward_backend_f32_host(
        decoder_ptr,
        options->latent->coords_bxyz,
        options->latent->feats,
        n_use,
        &forward_options,
        &out_coords,
        &out_feats,
        &n_out,
        &channels_out);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("pipeline mesh: decoder forward failed: %s", trellis_status_string(status));
        goto cleanup;
    }

    const int resolution = options->resolution > 0 ? options->resolution : options->latent->resolution;
    status = trellis_flexible_dual_grid_mesh_from_decoder_logits_host(
        out_coords,
        out_feats,
        n_out,
        channels_out,
        resolution,
        mesh_out);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("pipeline mesh: mesh extraction failed: %s", trellis_status_string(status));
        goto cleanup;
    }

cleanup:
    free(out_coords);
    free(out_feats);
    if (owns_decoder_store) {
        trellis_tensor_store_free(&decoder_store);
    }
    if (owns_weight_backend) {
        trellis_backend_free(&cpu_weights_backend);
    }
    if (status != TRELLIS_STATUS_OK) {
        if (subs_out != NULL) {
            trellis_sparse_c2s_guides_free(subs_out);
        }
        trellis_mesh_free(mesh_out);
    }
    return status;
}

static trellis_status trellis_pipeline_decode_shape_latent_decoder_coords(
    const trellis_pipeline_mesh_options * options,
    int32_t ** coords_out,
    int64_t * n_out) {
    if (coords_out != NULL) {
        *coords_out = NULL;
    }
    if (n_out != NULL) {
        *n_out = 0;
    }
    if (options == NULL || coords_out == NULL || n_out == NULL ||
        options->model_dir == NULL || options->latent == NULL ||
        options->latent->coords_bxyz == NULL || options->latent->feats == NULL ||
        options->latent->n_coords <= 0 || options->latent->channels != 32 ||
        (options->sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA && options->cuda == NULL)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = TRELLIS_STATUS_ERROR;
    char decoder_path[4096];
    trellis_tensor_store decoder_store;
    trellis_sparse_unet_vae_decoder_weights decoder;
    trellis_backend_context cpu_weights_backend;
    memset(&decoder_store, 0, sizeof(decoder_store));
    memset(&decoder, 0, sizeof(decoder));
    memset(&cpu_weights_backend, 0, sizeof(cpu_weights_backend));
    const trellis_sparse_unet_vae_decoder_weights * decoder_ptr = &decoder;
    int owns_weight_backend = 0;
    int owns_decoder_store = 0;
    int32_t * decoder_coords = NULL;
    float * decoder_feats = NULL;

    const trellis_backend_context * weight_backend = options->cuda;
    if (options->cache != NULL) {
        status = trellis_pipeline_model_cache_get_shape_decoder(
            options->cache,
            options->model_dir,
            options->decoder_override_path,
            &decoder_ptr);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("cascade upsample: cache shape decoder load failed: %s", trellis_status_string(status));
            goto cleanup;
        }
    } else {
        if (!make_shape_decoder_path(options->model_dir, options->decoder_override_path, decoder_path, sizeof(decoder_path))) {
            TRELLIS_ERROR("cascade upsample: failed to build shape decoder path");
            goto cleanup;
        }
        if (options->sparse_backend_kind != TRELLIS_SPARSE_BACKEND_CUDA) {
            status = trellis_backend_init(&cpu_weights_backend, TRELLIS_BACKEND_CPU, 0);
            if (status != TRELLIS_STATUS_OK) {
                TRELLIS_ERROR("cascade upsample: CPU decoder weight backend init failed: %s", trellis_status_string(status));
                goto cleanup;
            }
            weight_backend = &cpu_weights_backend;
            owns_weight_backend = 1;
        }
        if (!load_shape_decoder(weight_backend, decoder_path, &decoder_store, &decoder)) {
            goto cleanup;
        }
        owns_decoder_store = 1;
    }

    int64_t n_use = options->latent->n_coords;
    if (options->decode_max_input_tokens > 0 && options->decode_max_input_tokens < n_use) {
        n_use = options->decode_max_input_tokens;
        TRELLIS_WARN(
            "cascade upsample: truncating decoder input tokens %lld -> %lld",
            (long long) options->latent->n_coords,
            (long long) n_use);
    }

    int64_t decoder_n = 0;
    int decoder_channels = 0;
    trellis_sparse_unet_vae_decoder_forward_options forward_options;
    memset(&forward_options, 0, sizeof(forward_options));
    forward_options.backend_kind = options->sparse_backend_kind;
    forward_options.device = options->sparse_device;
    forward_options.max_levels = options->decode_max_levels;
    forward_options.sparse_backend = options->sparse_backend;
    status = trellis_sparse_unet_vae_decoder_forward_backend_f32_host(
        decoder_ptr,
        options->latent->coords_bxyz,
        options->latent->feats,
        n_use,
        &forward_options,
        &decoder_coords,
        &decoder_feats,
        &decoder_n,
        &decoder_channels);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("cascade upsample: decoder forward failed: %s", trellis_status_string(status));
        goto cleanup;
    }
    (void) decoder_channels;
    *coords_out = decoder_coords;
    *n_out = decoder_n;
    decoder_coords = NULL;
    status = TRELLIS_STATUS_OK;

cleanup:
    free(decoder_coords);
    free(decoder_feats);
    if (owns_decoder_store) {
        trellis_tensor_store_free(&decoder_store);
    }
    if (owns_weight_backend) {
        trellis_backend_free(&cpu_weights_backend);
    }
    return status;
}

static int cmp_coord4_i32(const void * a, const void * b) {
    const int32_t * ca = (const int32_t *) a;
    const int32_t * cb = (const int32_t *) b;
    for (int i = 0; i < 4; ++i) {
        if (ca[i] < cb[i]) return -1;
        if (ca[i] > cb[i]) return 1;
    }
    return 0;
}

static int same_coord4_i32(const int32_t * a, const int32_t * b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

trellis_status trellis_pipeline_quantize_cascade_coords(
    const int32_t * decoder_coords,
    int64_t decoder_n,
    int lr_resolution,
    int hr_resolution,
    trellis_cascade_coord_quantization quantization,
    int32_t ** coords_out,
    int64_t * n_out) {
    if (coords_out != NULL) {
        *coords_out = NULL;
    }
    if (n_out != NULL) {
        *n_out = 0;
    }
    if (decoder_coords == NULL || decoder_n <= 0 || lr_resolution <= 0 ||
        hr_resolution <= 0 || coords_out == NULL || n_out == NULL ||
        (quantization != TRELLIS_CASCADE_COORD_QUANTIZE_TRELLIS &&
         quantization != TRELLIS_CASCADE_COORD_QUANTIZE_PIXAL) ||
        decoder_n > INT64_MAX / 4) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int hr_slat_edge = hr_resolution / 16;
    if (hr_slat_edge <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    int32_t * tmp = (int32_t *) malloc((size_t) decoder_n * 4u * sizeof(int32_t));
    if (tmp == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < decoder_n; ++i) {
        tmp[(size_t) i * 4u + 0u] = decoder_coords[(size_t) i * 4u + 0u];
        for (int axis = 0; axis < 3; ++axis) {
            const int32_t c = decoder_coords[(size_t) i * 4u + 1u + (size_t) axis];
            int q = (int) floorf((((float) c + 0.5f) / (float) lr_resolution) * (float) hr_slat_edge);
            if (quantization == TRELLIS_CASCADE_COORD_QUANTIZE_PIXAL) {
                q = (int) roundf(
                    (((float) c + 0.5f) / (float) lr_resolution) *
                    (float) (hr_slat_edge - 1));
            }
            if (q < 0) q = 0;
            if (q >= hr_slat_edge) q = hr_slat_edge - 1;
            tmp[(size_t) i * 4u + 1u + (size_t) axis] = (int32_t) q;
        }
    }
    qsort(tmp, (size_t) decoder_n, 4u * sizeof(int32_t), cmp_coord4_i32);
    int64_t unique_n = 0;
    for (int64_t i = 0; i < decoder_n; ++i) {
        if (i == 0 || !same_coord4_i32(&tmp[(size_t) i * 4u], &tmp[(size_t) (i - 1) * 4u])) {
            if (unique_n != i) {
                memcpy(&tmp[(size_t) unique_n * 4u], &tmp[(size_t) i * 4u], 4u * sizeof(int32_t));
            }
            ++unique_n;
        }
    }
    int32_t * shrunk = (int32_t *) realloc(tmp, (size_t) unique_n * 4u * sizeof(int32_t));
    if (shrunk != NULL) {
        tmp = shrunk;
    }
    *coords_out = tmp;
    *n_out = unique_n;
    return TRELLIS_STATUS_OK;
}

void trellis_pbr_voxels_free(trellis_pbr_voxels * voxels) {
    if (voxels == NULL) {
        return;
    }
    free(voxels->coords_bxyz);
    free(voxels->attrs);
    memset(voxels, 0, sizeof(*voxels));
}

trellis_status trellis_pipeline_decode_texture_latent_voxels(
    const trellis_pipeline_texture_options * options,
    trellis_pbr_voxels * voxels_out) {
    if (voxels_out != NULL) {
        memset(voxels_out, 0, sizeof(*voxels_out));
    }
    if (options == NULL || voxels_out == NULL || options->model_dir == NULL ||
        options->latent == NULL || options->latent->coords_bxyz == NULL ||
        options->latent->feats == NULL || options->latent->n_coords <= 0 ||
        options->latent->channels != 32 || options->guide_subs == NULL ||
        options->guide_subs->n_levels <= 0 ||
        (options->sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA && options->cuda == NULL)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = TRELLIS_STATUS_ERROR;
    char decoder_path[4096];
    trellis_tensor_store decoder_store;
    trellis_sparse_unet_vae_decoder_weights decoder;
    trellis_backend_context cpu_weights_backend;
    memset(&decoder_store, 0, sizeof(decoder_store));
    memset(&decoder, 0, sizeof(decoder));
    memset(&cpu_weights_backend, 0, sizeof(cpu_weights_backend));
    const trellis_sparse_unet_vae_decoder_weights * decoder_ptr = &decoder;

    const trellis_backend_context * weight_backend = options->cuda;
    int owns_weight_backend = 0;
    int owns_decoder_store = 0;
    if (options->cache != NULL) {
        status = trellis_pipeline_model_cache_get_texture_decoder(
            options->cache,
            options->model_dir,
            options->decoder_override_path,
            &decoder_ptr);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("texture decode: cache texture decoder load failed: %s", trellis_status_string(status));
            goto cleanup;
        }
    } else {
        if (!make_texture_decoder_path(
                options->model_dir,
                options->decoder_override_path,
                decoder_path,
                sizeof(decoder_path))) {
            TRELLIS_ERROR("texture decode: failed to build texture decoder path");
            goto cleanup;
        }
        if (options->sparse_backend_kind != TRELLIS_SPARSE_BACKEND_CUDA) {
            status = trellis_backend_init(&cpu_weights_backend, TRELLIS_BACKEND_CPU, 0);
            if (status != TRELLIS_STATUS_OK) {
                TRELLIS_ERROR("texture decode: CPU decoder weight backend init failed: %s", trellis_status_string(status));
                goto cleanup;
            }
            weight_backend = &cpu_weights_backend;
            owns_weight_backend = 1;
        }
        if (!load_texture_decoder(weight_backend, decoder_path, &decoder_store, &decoder)) {
            goto cleanup;
        }
        owns_decoder_store = 1;
    }

    int64_t n_use = options->latent->n_coords;
    if (options->decode_max_input_tokens > 0 && options->decode_max_input_tokens < n_use) {
        n_use = options->decode_max_input_tokens;
        TRELLIS_WARN(
            "texture decode: truncating decoder input tokens %lld -> %lld",
            (long long) options->latent->n_coords,
            (long long) n_use);
    }

    int64_t n_out = 0;
    int channels_out = 0;
    int32_t * out_coords = NULL;
    float * out_feats = NULL;
    trellis_sparse_unet_vae_decoder_forward_options forward_options;
    memset(&forward_options, 0, sizeof(forward_options));
    forward_options.backend_kind = options->sparse_backend_kind;
    forward_options.device = options->sparse_device;
    forward_options.max_levels = options->decode_max_levels;
    forward_options.sparse_backend = options->sparse_backend;
    forward_options.guide_subs = options->guide_subs;
    status = trellis_sparse_unet_vae_decoder_forward_backend_f32_host(
        decoder_ptr,
        options->latent->coords_bxyz,
        options->latent->feats,
        n_use,
        &forward_options,
        &out_coords,
        &out_feats,
        &n_out,
        &channels_out);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("texture decode: decoder forward failed: %s", trellis_status_string(status));
        free(out_coords);
        free(out_feats);
        goto cleanup;
    }
    for (int64_t i = 0; i < n_out; ++i) {
        for (int c = 0; c < channels_out; ++c) {
            float v = out_feats[(size_t) i * (size_t) channels_out + (size_t) c] * 0.5f + 0.5f;
            out_feats[(size_t) i * (size_t) channels_out + (size_t) c] = v;
        }
    }
    voxels_out->coords_bxyz = out_coords;
    voxels_out->attrs = out_feats;
    voxels_out->n_coords = n_out;
    voxels_out->channels = channels_out;
    voxels_out->resolution = options->latent->resolution;
    out_coords = NULL;
    out_feats = NULL;
    status = TRELLIS_STATUS_OK;

cleanup:
    if (owns_decoder_store) {
        trellis_tensor_store_free(&decoder_store);
    }
    if (owns_weight_backend) {
        trellis_backend_free(&cpu_weights_backend);
    }
    if (status != TRELLIS_STATUS_OK) {
        trellis_pbr_voxels_free(voxels_out);
    }
    return status;
}

static trellis_status trellis_pipeline_write_meshbin(const char * path, const trellis_mesh_host * mesh) {
    if (path == NULL || mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!mkdir_parent(path)) {
        TRELLIS_ERROR("pipeline mesh: failed to create output directory for %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        TRELLIS_ERROR("pipeline mesh: failed to open %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    const char magic[8] = { 'T', 'R', 'L', 'M', 'E', 'S', 'H', '1' };
    const uint64_t n_vertices = (uint64_t) mesh->n_vertices;
    const uint64_t n_faces = (uint64_t) mesh->n_faces;
    const uint32_t flags = 0;
    const uint32_t reserved = 0;
    const size_t vertex_count = (size_t) mesh->n_vertices * 3u;
    const size_t face_count = (size_t) mesh->n_faces * 3u;
    trellis_status status = TRELLIS_STATUS_OK;
    if (fwrite(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fwrite(&n_vertices, sizeof(n_vertices), 1, f) != 1 ||
        fwrite(&n_faces, sizeof(n_faces), 1, f) != 1 ||
        fwrite(&flags, sizeof(flags), 1, f) != 1 ||
        fwrite(&reserved, sizeof(reserved), 1, f) != 1 ||
        fwrite(mesh->vertices, sizeof(float), vertex_count, f) != vertex_count ||
        fwrite(mesh->faces, sizeof(int32_t), face_count, f) != face_count) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    if (fclose(f) != 0 && status == TRELLIS_STATUS_OK) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    return status;
}

static trellis_status trellis_pipeline_load_meshbin(const char * path, trellis_mesh_host * mesh_out) {
    if (path == NULL || mesh_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(mesh_out, 0, sizeof(*mesh_out));
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        TRELLIS_ERROR("pipeline mesh: failed to open meshbin %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }

    char magic[8];
    uint64_t n_vertices = 0;
    uint64_t n_faces = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
    trellis_status status = TRELLIS_STATUS_OK;
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fread(&n_vertices, sizeof(n_vertices), 1, f) != 1 ||
        fread(&n_faces, sizeof(n_faces), 1, f) != 1 ||
        fread(&flags, sizeof(flags), 1, f) != 1 ||
        fread(&reserved, sizeof(reserved), 1, f) != 1 ||
        memcmp(magic, "TRLMESH1", 8) != 0 ||
        n_vertices == 0 || n_faces == 0 ||
        n_vertices > (uint64_t) INT64_MAX ||
        n_faces > (uint64_t) INT64_MAX ||
        n_vertices > (uint64_t) SIZE_MAX / (3u * sizeof(float)) ||
        n_faces > (uint64_t) SIZE_MAX / (3u * sizeof(int32_t))) {
        status = TRELLIS_STATUS_PARSE_ERROR;
    }

    trellis_mesh_host mesh;
    memset(&mesh, 0, sizeof(mesh));
    if (status == TRELLIS_STATUS_OK) {
        const size_t vertex_count = (size_t) n_vertices * 3u;
        const size_t face_count = (size_t) n_faces * 3u;
        mesh.vertices = (float *) malloc(vertex_count * sizeof(float));
        mesh.faces = (int32_t *) malloc(face_count * sizeof(int32_t));
        mesh.n_vertices = (int64_t) n_vertices;
        mesh.n_faces = (int64_t) n_faces;
        if (mesh.vertices == NULL || mesh.faces == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else if (fread(mesh.vertices, sizeof(float), vertex_count, f) != vertex_count ||
                   fread(mesh.faces, sizeof(int32_t), face_count, f) != face_count) {
            status = TRELLIS_STATUS_IO_ERROR;
        }
        if (status == TRELLIS_STATUS_OK && (flags & 1u) != 0) {
            const uint64_t uv_bytes = n_vertices * 2u * (uint64_t) sizeof(float);
            if (uv_bytes > (uint64_t) LONG_MAX || fseek(f, (long) uv_bytes, SEEK_CUR) != 0) {
                status = TRELLIS_STATUS_IO_ERROR;
            }
        }
    }
    if (fclose(f) != 0 && status == TRELLIS_STATUS_OK) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_free(&mesh);
        return status;
    }
    *mesh_out = mesh;
    return TRELLIS_STATUS_OK;
}

static trellis_status trellis_pipeline_write_pbr_voxels_debug(
    const char * path,
    const trellis_pbr_voxels * voxels) {
    if (path == NULL || voxels == NULL || voxels->coords_bxyz == NULL || voxels->attrs == NULL ||
        voxels->n_coords <= 0 || voxels->channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!mkdir_parent(path)) {
        TRELLIS_ERROR("pipeline: failed to create PBR voxel dump directory for %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        TRELLIS_ERROR("pipeline: failed to open PBR voxel dump %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    const char magic[8] = { 'T', 'R', 'L', 'P', 'B', 'R', '1', '\0' };
    const size_t coords_count = (size_t) voxels->n_coords * 4u;
    const size_t attrs_count = (size_t) voxels->n_coords * (size_t) voxels->channels;
    trellis_status status = TRELLIS_STATUS_OK;
    if (fwrite(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fwrite(&voxels->n_coords, sizeof(voxels->n_coords), 1, f) != 1 ||
        fwrite(&voxels->channels, sizeof(voxels->channels), 1, f) != 1 ||
        fwrite(&voxels->resolution, sizeof(voxels->resolution), 1, f) != 1 ||
        fwrite(voxels->coords_bxyz, sizeof(int32_t), coords_count, f) != coords_count ||
        fwrite(voxels->attrs, sizeof(float), attrs_count, f) != attrs_count) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    if (fclose(f) != 0 && status == TRELLIS_STATUS_OK) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("pipeline: failed to write PBR voxel dump %s", path);
    }
    return status;
}

static int make_material_dump_path(char * out, size_t out_size, const char * dir, const char * name) {
    int n = snprintf(out, out_size, "%s/%s", dir, name);
    return n >= 0 && (size_t) n < out_size;
}

static trellis_status trellis_pipeline_dump_material_inputs_if_requested(
    const char * dump_dir,
    const trellis_mesh_host * mesh,
    const trellis_mesh_host * sample_mesh,
    const trellis_pbr_voxels * voxels) {
    if (dump_dir == NULL || dump_dir[0] == '\0') {
        return TRELLIS_STATUS_OK;
    }
    if (!mkdir_p(dump_dir)) {
        TRELLIS_ERROR("pipeline: failed to create material dump directory %s", dump_dir);
        return TRELLIS_STATUS_IO_ERROR;
    }

    char path[4096];
    if (!make_material_dump_path(path, sizeof(path), dump_dir, "processed.meshbin")) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_status status = trellis_pipeline_write_meshbin(path, mesh);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    if (sample_mesh != NULL && sample_mesh->vertices != NULL && sample_mesh->faces != NULL &&
        sample_mesh->n_vertices > 0 && sample_mesh->n_faces > 0) {
        if (!make_material_dump_path(path, sizeof(path), dump_dir, "projection_source.meshbin")) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        status = trellis_pipeline_write_meshbin(path, sample_mesh);
        if (status != TRELLIS_STATUS_OK) {
            return status;
        }
    }

    if (!make_material_dump_path(path, sizeof(path), dump_dir, "pbr_voxels.bin")) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    status = trellis_pipeline_write_pbr_voxels_debug(path, voxels);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    if (!make_material_dump_path(path, sizeof(path), dump_dir, "manifest.txt")) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    FILE * f = fopen(path, "w");
    if (f == NULL) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    fprintf(f, "processed_vertices=%lld\n", (long long) mesh->n_vertices);
    fprintf(f, "processed_faces=%lld\n", (long long) mesh->n_faces);
    if (sample_mesh != NULL && sample_mesh->vertices != NULL && sample_mesh->faces != NULL) {
        fprintf(f, "projection_vertices=%lld\n", (long long) sample_mesh->n_vertices);
        fprintf(f, "projection_faces=%lld\n", (long long) sample_mesh->n_faces);
    } else {
        fprintf(f, "projection_vertices=0\n");
        fprintf(f, "projection_faces=0\n");
    }
    fprintf(f, "pbr_voxels=%lld\n", (long long) voxels->n_coords);
    fprintf(f, "pbr_channels=%d\n", voxels->channels);
    fprintf(f, "pbr_resolution=%d\n", voxels->resolution);
    if (fclose(f) != 0) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    TRELLIS_INFO("pipeline: dumped material bake inputs to %s", dump_dir);
    return TRELLIS_STATUS_OK;
}

static int run_vkmesh_postprocess_command(
    const char * vkmesh_path,
    const char * input_mesh,
    const char * output_mesh,
    const char * projection_mesh,
    int decimation_target,
    int no_simplify,
    int remesh,
    int remesh_resolution,
    float remesh_band,
    float remesh_project,
    int device,
    int gpu_workspace_budget_mib) {
    char target[64];
    char remesh_resolution_buf[64];
    char remesh_band_buf[64];
    char remesh_project_buf[64];
    char device_buf[64];
    char workspace_budget_buf[64];
    snprintf(target, sizeof(target), "%d", decimation_target > 0 ? decimation_target : 1000000);
    snprintf(remesh_resolution_buf, sizeof(remesh_resolution_buf), "%d", remesh_resolution > 0 ? remesh_resolution : 1024);
    snprintf(remesh_band_buf, sizeof(remesh_band_buf), "%.9g", remesh_band > 0.0f ? remesh_band : 1.0f);
    snprintf(remesh_project_buf, sizeof(remesh_project_buf), "%.9g", remesh_project > 0.0f ? remesh_project : 0.0f);
    snprintf(device_buf, sizeof(device_buf), "%d", device >= 0 ? device : 0);
    snprintf(workspace_budget_buf, sizeof(workspace_budget_buf), "%d", gpu_workspace_budget_mib > 0 ? gpu_workspace_budget_mib : 0);
    char sibling_vkmesh[PATH_MAX];
    const char * exe = vkmesh_path != NULL && vkmesh_path[0] != '\0' ? vkmesh_path : NULL;
    if (exe == NULL) {
        if (trellis_current_executable_path(sibling_vkmesh, sizeof(sibling_vkmesh))) {
            char * slash = trellis_path_last_sep(sibling_vkmesh);
            if (slash != NULL) {
                slash[1] = '\0';
                size_t dir_len = strlen(sibling_vkmesh);
                const char * vkmesh_name = "vkmesh" TRELLIS_EXE_SUFFIX;
                size_t name_len = strlen(vkmesh_name);
                if (dir_len + name_len + 1u < sizeof(sibling_vkmesh)) {
                    memcpy(sibling_vkmesh + dir_len, vkmesh_name, name_len + 1u);
                    if (trellis_access_executable(sibling_vkmesh)) {
                        exe = sibling_vkmesh;
                    }
                }
            }
        }
    }
    if (exe == NULL) {
        exe = "vkmesh";
    }
    TRELLIS_INFO("mesh postprocess: using vkmesh executable %s", exe);
    char * argv[32];
    int argc = 0;
    argv[argc++] = (char *) exe;
    argv[argc++] = (char *) "--input";
    argv[argc++] = (char *) input_mesh;
    argv[argc++] = (char *) "--output";
    argv[argc++] = (char *) output_mesh;
    if (projection_mesh != NULL && projection_mesh[0] != '\0') {
        argv[argc++] = (char *) "--projection-mesh-output";
        argv[argc++] = (char *) projection_mesh;
    }
    argv[argc++] = (char *) "--postprocess";
    argv[argc++] = (char *) "--device";
    argv[argc++] = device_buf;
    argv[argc++] = (char *) "--gpu-workspace-budget-mib";
    argv[argc++] = workspace_budget_buf;
    if (remesh) {
        argv[argc++] = (char *) "--remesh";
        argv[argc++] = (char *) "--remesh-resolution";
        argv[argc++] = remesh_resolution_buf;
        argv[argc++] = (char *) "--remesh-band";
        argv[argc++] = remesh_band_buf;
        argv[argc++] = (char *) "--remesh-project";
        argv[argc++] = remesh_project_buf;
    } else {
        argv[argc++] = (char *) "--no-remesh";
    }
    argv[argc++] = (char *) "--decimation-target";
    argv[argc++] = target;
    if (no_simplify) {
        argv[argc++] = (char *) "--no-simplify";
    }
    argv[argc++] = (char *) "--no-uv-unwrap";
    argv[argc] = NULL;
    return trellis_path_has_sep(exe) ?
        trellis_run_process_exact(argv) :
        trellis_run_process_search_path(argv);
}

static trellis_status trellis_pipeline_postprocess_mesh_with_vkmesh(
    trellis_mesh_host * mesh,
    trellis_mesh_host * projection_mesh_out,
    const char * vkmesh_path,
    int decimation_target,
    int no_simplify,
    int remesh,
    int remesh_resolution,
    float remesh_band,
    float remesh_project,
    int device,
    int gpu_workspace_budget_mib) {
    if (mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (projection_mesh_out != NULL) {
        memset(projection_mesh_out, 0, sizeof(*projection_mesh_out));
    }

#if TRELLIS_HAS_VKMESH_C_API
    (void) vkmesh_path;
    trellis_vkmesh_postprocess_options vkmesh_options;
    memset(&vkmesh_options, 0, sizeof(vkmesh_options));
    vkmesh_options.decimation_target = decimation_target > 0 ? decimation_target : 1000000;
    vkmesh_options.no_simplify = no_simplify ? 1 : 0;
    vkmesh_options.remesh = remesh ? 1 : 0;
    vkmesh_options.remesh_resolution = remesh_resolution > 0 ? remesh_resolution : 1024;
    vkmesh_options.remesh_band = remesh_band > 0.0f ? remesh_band : 1.0f;
    vkmesh_options.remesh_project = remesh_project > 0.0f ? remesh_project : 0.0f;
    vkmesh_options.device = device >= 0 ? device : 0;
    vkmesh_options.gpu_workspace_budget_mib = gpu_workspace_budget_mib > 0 ? gpu_workspace_budget_mib : 0;

    trellis_mesh_host processed;
    trellis_mesh_host projection;
    memset(&processed, 0, sizeof(processed));
    memset(&projection, 0, sizeof(projection));

    TRELLIS_INFO(
        "mesh postprocess: running vkmesh C API device=%d workspace_budget_mib=%d target=%d no_simplify=%d remesh=%d resolution=%d band=%.9g project=%.9g input_faces=%lld",
        vkmesh_options.device,
        vkmesh_options.gpu_workspace_budget_mib,
        vkmesh_options.decimation_target,
        no_simplify ? 1 : 0,
        vkmesh_options.remesh,
        vkmesh_options.remesh_resolution,
        vkmesh_options.remesh_band,
        vkmesh_options.remesh_project,
        (long long) mesh->n_faces);
    trellis_status status = trellis_vkmesh_postprocess(
        mesh,
        &processed,
        projection_mesh_out != NULL ? &projection : NULL,
        &vkmesh_options);
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_free(&processed);
        trellis_mesh_free(&projection);
        TRELLIS_ERROR("mesh postprocess: vkmesh C API failed: %s", trellis_status_string(status));
        return status;
    }
    if (projection_mesh_out != NULL) {
        *projection_mesh_out = projection;
        memset(&projection, 0, sizeof(projection));
        TRELLIS_INFO(
            "mesh postprocess: loaded projection source vertices=%lld faces=%lld",
            (long long) projection_mesh_out->n_vertices,
            (long long) projection_mesh_out->n_faces);
    }

    int64_t old_vertices = mesh->n_vertices;
    int64_t old_faces = mesh->n_faces;
    trellis_mesh_free(mesh);
    *mesh = processed;
    TRELLIS_INFO(
        "mesh postprocess: done vertices=%lld->%lld faces=%lld->%lld",
        (long long) old_vertices,
        (long long) mesh->n_vertices,
        (long long) old_faces,
        (long long) mesh->n_faces);
    trellis_mesh_free(&projection);
    return TRELLIS_STATUS_OK;
#else
    char input_mesh[4096];
    char output_mesh[4096];
    char projection_mesh[4096];
    if (!trellis_make_temp_path(input_mesh, sizeof(input_mesh), "trellis2_vkmesh_in", ".meshbin") ||
        !trellis_make_temp_path(output_mesh, sizeof(output_mesh), "trellis2_vkmesh_out", ".meshbin") ||
        !trellis_make_temp_path(projection_mesh, sizeof(projection_mesh), "trellis2_vkmesh_projection", ".meshbin")) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_status status = trellis_pipeline_write_meshbin(input_mesh, mesh);
    if (status != TRELLIS_STATUS_OK) {
        unlink_if_set(input_mesh);
        unlink_if_set(output_mesh);
        unlink_if_set(projection_mesh);
        return status;
    }

    TRELLIS_INFO(
        "mesh postprocess: running vkmesh device=%d workspace_budget_mib=%d target=%d no_simplify=%d remesh=%d resolution=%d band=%.9g project=%.9g input_faces=%lld",
        device >= 0 ? device : 0,
        gpu_workspace_budget_mib > 0 ? gpu_workspace_budget_mib : 0,
        decimation_target > 0 ? decimation_target : 1000000,
        no_simplify ? 1 : 0,
        remesh ? 1 : 0,
        remesh_resolution > 0 ? remesh_resolution : 1024,
        remesh_band > 0.0f ? remesh_band : 1.0f,
        remesh_project > 0.0f ? remesh_project : 0.0f,
        (long long) mesh->n_faces);
    if (!run_vkmesh_postprocess_command(
            vkmesh_path,
            input_mesh,
            output_mesh,
            projection_mesh_out != NULL ? projection_mesh : NULL,
            decimation_target,
            no_simplify,
            remesh,
            remesh_resolution,
            remesh_band,
            remesh_project,
            device,
            gpu_workspace_budget_mib)) {
        TRELLIS_ERROR("mesh postprocess: vkmesh failed; pass --vkmesh /path/to/vkmesh if it is not in PATH");
        unlink_if_set(input_mesh);
        unlink_if_set(output_mesh);
        unlink_if_set(projection_mesh);
        return TRELLIS_STATUS_ERROR;
    }

    trellis_mesh_host processed;
    status = trellis_pipeline_load_meshbin(output_mesh, &processed);
    if (status == TRELLIS_STATUS_OK && projection_mesh_out != NULL) {
        status = trellis_pipeline_load_meshbin(projection_mesh, projection_mesh_out);
        if (status == TRELLIS_STATUS_OK) {
            TRELLIS_INFO(
                "mesh postprocess: loaded projection source vertices=%lld faces=%lld",
                (long long) projection_mesh_out->n_vertices,
                (long long) projection_mesh_out->n_faces);
        }
    }
    unlink_if_set(input_mesh);
    unlink_if_set(output_mesh);
    unlink_if_set(projection_mesh);
    if (status != TRELLIS_STATUS_OK) {
        if (projection_mesh_out != NULL) {
            trellis_mesh_free(projection_mesh_out);
        }
        return status;
    }

    int64_t old_vertices = mesh->n_vertices;
    int64_t old_faces = mesh->n_faces;
    trellis_mesh_free(mesh);
    *mesh = processed;
    TRELLIS_INFO(
        "mesh postprocess: done vertices=%lld->%lld faces=%lld->%lld",
        (long long) old_vertices,
        (long long) mesh->n_vertices,
        (long long) old_faces,
        (long long) mesh->n_faces);
    return TRELLIS_STATUS_OK;
#endif
}

trellis_status trellis_pipeline_image_to_gltf(const trellis_image_to_gltf_options * options) {
    trellis_pixal3d_options pixal_options = TRELLIS_PIXAL3D_OPTIONS_INIT;
    return trellis_pipeline_image_to_gltf_ex(options, &pixal_options);
}

trellis_status trellis_pipeline_image_to_gltf_ex(
    const trellis_image_to_gltf_options * options,
    const trellis_pixal3d_options * pixal_options) {
    if (pixal_options == NULL ||
        pixal_options->struct_size < TRELLIS_PIXAL3D_OPTIONS_V1_SIZE ||
        !isfinite(pixal_options->camera_angle_x) ||
        !isfinite(pixal_options->camera_distance) ||
        !isfinite(pixal_options->mesh_scale) ||
        pixal_options->camera_angle_x >= 3.14159265358979323846f ||
        options == NULL || options->model_dir == NULL || options->dino_dir == NULL ||
        options->image_path == NULL || options->device < 0 ||
        options->vkmesh_gpu_workspace_budget_mib < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const char * pixal_naf_override = pixal_options->naf_path;
    if (pixal_naf_override == NULL || pixal_naf_override[0] == '\0') {
        pixal_naf_override = getenv("TRELLIS_NAF_PATH");
    }
    float pixal_camera_angle_x = pixal_options->camera_angle_x;
    float pixal_camera_distance = pixal_options->camera_distance;
    float pixal_mesh_scale = pixal_options->mesh_scale;
    const int pixal_camera_distance_is_automatic = pixal_camera_distance <= 0.0f;
    const char * output_gltf_path =
        options->gltf_path != NULL && options->gltf_path[0] != '\0' ?
            options->gltf_path :
            "output.glb";

    char temp_image[4096];
    char temp_birefnet_image[4096];
    temp_image[0] = '\0';
    temp_birefnet_image[0] = '\0';
    trellis_status status = TRELLIS_STATUS_OK;
    trellis_model_package model_package = TRELLIS_MODEL_PACKAGE_INIT;
    const trellis_image_to_3d_adapter_descriptor * task_adapter = NULL;
    int projected_conditioning = 0;
    const char * backend_name =
        options->backend != NULL && options->backend[0] != '\0' ?
            options->backend :
            TRELLIS_DEFAULT_BACKEND;
    trellis_backend_kind graph_backend_kind = TRELLIS_BACKEND_CUDA;
    status = trellis_backend_kind_from_name(backend_name, &graph_backend_kind);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("backend: invalid backend '%s'", backend_name);
        return status;
    }
    if (strcmp(trellis_backend_kind_name(graph_backend_kind), TRELLIS_DEFAULT_BACKEND) != 0) {
        TRELLIS_ERROR(
            "backend: this binary was compiled for %s; rebuild with -DTRELLIS2_C_BACKEND=%s for that backend",
            TRELLIS_DEFAULT_BACKEND,
            trellis_backend_kind_name(graph_backend_kind));
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    status = trellis_model_package_load(options->model_dir, &model_package);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "model package: failed to load '%s': %s",
            options->model_dir,
            trellis_status_string(status));
        return status;
    }
    status = trellis_image_to_3d_adapter_resolve_package(
        &model_package, &task_adapter);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "model package '%s': no compatible image_to_3d adapter for family '%s': %s",
            model_package.id != NULL ? model_package.id : "unknown",
            model_package.family != NULL ? model_package.family : "unknown",
            trellis_status_string(status));
        trellis_model_package_free(&model_package);
        return status;
    }
    projected_conditioning = task_adapter->uses_projected_conditioning;
    if (pixal_camera_angle_x <= 0.0f) {
        pixal_camera_angle_x = task_adapter->default_camera_angle_x;
    }
    if (pixal_mesh_scale <= 0.0f) {
        pixal_mesh_scale = task_adapter->default_mesh_scale;
    }
    if (projected_conditioning) {
        status = trellis_pixal_resolve_camera_distance(
            pixal_camera_angle_x,
            pixal_mesh_scale,
            pixal_camera_distance,
            &pixal_camera_distance);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "pipeline: invalid Pixal3D camera fov=%.9g mesh_scale=%.9g distance=%.9g",
                (double) pixal_camera_angle_x,
                (double) pixal_mesh_scale,
                (double) pixal_options->camera_distance);
            trellis_model_package_free(&model_package);
            return status;
        }
    } else if (pixal_camera_distance_is_automatic) {
        pixal_camera_distance = task_adapter->default_camera_distance;
    }
    TRELLIS_INFO(
        "model package: id=%s family=%s task=%s profile=%s adapter=%s source=%s",
        model_package.id,
        model_package.family,
        model_package.task,
        model_package.profile,
        task_adapter->id,
        model_package.source == TRELLIS_MODEL_PACKAGE_SOURCE_MANIFEST ? "manifest" : "legacy");
    if (projected_conditioning) {
        TRELLIS_INFO(
            "Pixal3D camera: fov=%.9g rad mesh_scale=%.9g distance=%.9g (%s)",
            (double) pixal_camera_angle_x,
            (double) pixal_mesh_scale,
            (double) pixal_camera_distance,
            pixal_camera_distance_is_automatic ?
                "derived from FOV and mesh scale" : "explicit");
    }
    if (projected_conditioning && options->model_cache &&
        model_cache_budget_bytes_from_options(options) == 0) {
        TRELLIS_WARN(
            "pipeline: Pixal3D with an unlimited model cache can have a large NAF/1024 workspace peak; use --no-model-cache or --model-cache-budget-mib N on memory-constrained devices");
    }

    const char * sparse_structure_image_path = options->image_path;
    const char * effective_birefnet_path =
        options->birefnet_path != NULL && options->birefnet_path[0] != '\0' ?
            options->birefnet_path : NULL;
    char packaged_birefnet_path[4096];
    packaged_birefnet_path[0] = '\0';
    if (path_has_ext(options->image_path, "webp")) {
        if (!trellis_make_temp_path(
            temp_image,
            sizeof(temp_image),
            "trellis2_image_to_gltf",
            ".png")) {
            trellis_model_package_free(&model_package);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        TRELLIS_INFO("image prep: converting WebP -> PNG via ffmpeg: %s", temp_image);
        if (!convert_webp_to_png_ffmpeg(options->image_path, temp_image)) {
            TRELLIS_ERROR("image prep: WebP conversion failed; install ffmpeg or convert the image to PNG/JPEG first");
            unlink_if_set(temp_image);
            trellis_model_package_free(&model_package);
            return TRELLIS_STATUS_IO_ERROR;
        }
        sparse_structure_image_path = temp_image;
    }
    if (effective_birefnet_path == NULL) {
        int has_transparent_alpha = 0;
        status = image_has_transparent_alpha(
            sparse_structure_image_path,
            &has_transparent_alpha);
        if (status != TRELLIS_STATUS_OK) {
            unlink_if_set(temp_image);
            trellis_model_package_free(&model_package);
            return status;
        }
        if (!has_transparent_alpha) {
            if (resolve_packaged_birefnet_path(
                    options->model_dir,
                    options->dino_dir,
                    packaged_birefnet_path,
                    sizeof(packaged_birefnet_path))) {
                effective_birefnet_path = packaged_birefnet_path;
                TRELLIS_INFO("image prep: auto-discovered BiRefNet %s", effective_birefnet_path);
            } else if (task_adapter->requires_transparent_foreground) {
                TRELLIS_ERROR(
                    "pipeline: opaque Pixal3D input requires BiRefNet; place it under model/ckpts or beside DINO, set TRELLIS_BIREFNET_PATH, or pass --birefnet FILE");
                unlink_if_set(temp_image);
                trellis_model_package_free(&model_package);
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    if (effective_birefnet_path != NULL) {
        if (!trellis_make_temp_path(
            temp_birefnet_image,
            sizeof(temp_birefnet_image),
            "trellis2_birefnet",
            ".png")) {
            unlink_if_set(temp_image);
            trellis_model_package_free(&model_package);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        TRELLIS_INFO("image prep: running BiRefNet background removal -> %s", temp_birefnet_image);
        status = apply_birefnet_background_removal(
            sparse_structure_image_path,
            effective_birefnet_path,
            temp_birefnet_image,
            graph_backend_kind,
            options->device);
        if (status != TRELLIS_STATUS_OK) {
            unlink_if_set(temp_birefnet_image);
            unlink_if_set(temp_image);
            trellis_model_package_free(&model_package);
            return status;
        }
        sparse_structure_image_path = temp_birefnet_image;
    }

    trellis_backend_context graph_backend;
    trellis_cuda_context cuda;
    trellis_sparse_backend * shared_sparse_backend = NULL;
    memset(&graph_backend, 0, sizeof(graph_backend));
    memset(&cuda, 0, sizeof(cuda));
    trellis_sparse_backend_kind sparse_backend_kind = TRELLIS_SPARSE_BACKEND_CUDA;
    status = sparse_backend_kind_from_graph_backend(graph_backend_kind, &sparse_backend_kind);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("backend: '%s' is not a full pipeline backend; use cuda or vulkan", backend_name);
        unlink_if_set(temp_birefnet_image);
        unlink_if_set(temp_image);
        trellis_model_package_free(&model_package);
        return status;
    }
    const int graph_device = options->device;
    status = trellis_backend_init(&graph_backend, graph_backend_kind, graph_device);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "ggml backend: init %s device=%d failed: %s",
            trellis_backend_kind_name(graph_backend_kind),
            graph_device,
            trellis_status_string(status));
        unlink_if_set(temp_birefnet_image);
        unlink_if_set(temp_image);
        trellis_model_package_free(&model_package);
        return status;
    }
    TRELLIS_INFO(
        "backend: %s device=%d",
        trellis_backend_kind_name(graph_backend.kind),
        graph_backend.device);

    const int sparse_device = options->device;
    if (sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA) {
        status = trellis_cuda_init(&cuda, options->device);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("SparseUnet CUDA decoder: init device=%d failed: %s", options->device, trellis_status_string(status));
            trellis_backend_free(&graph_backend);
            unlink_if_set(temp_birefnet_image);
            unlink_if_set(temp_image);
            trellis_model_package_free(&model_package);
            return status;
        }
    } else if (sparse_backend_kind == TRELLIS_SPARSE_BACKEND_VULKAN) {
        status = trellis_sparse_vulkan_backend_create(sparse_device, &shared_sparse_backend);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "SparseUnet Vulkan decoder backend: init device=%d failed: %s",
                sparse_device,
                trellis_status_string(status));
            trellis_backend_free(&graph_backend);
            unlink_if_set(temp_birefnet_image);
            unlink_if_set(temp_image);
            trellis_model_package_free(&model_package);
            return status;
        }
    }
    TRELLIS_INFO("SparseUnet backend: %s device=%d", sparse_backend_kind_name(sparse_backend_kind), sparse_device);

    trellis_sparse_structure_result sparse_structure_result;
    trellis_image_condition_result cond_shape_512;
    trellis_image_condition_result cond_1024;
    trellis_structured_latent shape_latent;
    trellis_structured_latent shape_latent_lr;
    trellis_structured_latent texture_latent;
    trellis_sparse_c2s_guides shape_subs;
    trellis_pbr_voxels pbr_voxels;
    trellis_mesh_host mesh;
    trellis_mesh_host gltf_projection_mesh;
    trellis_pipeline_model_cache model_cache;
    memset(&sparse_structure_result, 0, sizeof(sparse_structure_result));
    memset(&cond_shape_512, 0, sizeof(cond_shape_512));
    memset(&cond_1024, 0, sizeof(cond_1024));
    memset(&shape_latent, 0, sizeof(shape_latent));
    memset(&shape_latent_lr, 0, sizeof(shape_latent_lr));
    memset(&texture_latent, 0, sizeof(texture_latent));
    memset(&shape_subs, 0, sizeof(shape_subs));
    memset(&pbr_voxels, 0, sizeof(pbr_voxels));
    memset(&mesh, 0, sizeof(mesh));
    memset(&gltf_projection_mesh, 0, sizeof(gltf_projection_mesh));
    memset(&model_cache, 0, sizeof(model_cache));
    int model_cache_initialized = 0;
    trellis_pipeline_model_cache * model_cache_ptr = NULL;
    const char * material_dump_dir = getenv("TRELLIS_MATERIAL_DUMP_DIR");
    int32_t * cascade_decoder_coords = NULL;
    int64_t cascade_decoder_n = 0;
    int32_t * cascade_coords = NULL;
    int64_t cascade_n = 0;
    char sparse_flow_path[4096];
    char sparse_decoder_path[4096];
    char shape_flow_lr_path[4096];
    char shape_flow_hr_path[4096];
    char texture_flow_path[4096];
    char shape_decoder_path[4096];
    char texture_decoder_path[4096];
    char naf_component_path[4096];
    shape_flow_hr_path[0] = '\0';
    naf_component_path[0] = '\0';
    const char * pixal_naf_path = pixal_naf_override;
    const trellis_model_component_instance * sparse_flow_component = NULL;
    const trellis_model_component_instance * shape_flow_lr_component = NULL;
    const trellis_model_component_instance * shape_flow_hr_component = NULL;
    const trellis_model_component_instance * texture_flow_component = NULL;
    trellis_ggml_attention_policy sparse_flow_attention =
        TRELLIS_GGML_ATTENTION_POLICY_INIT;
    trellis_ggml_attention_policy shape_flow_lr_attention =
        TRELLIS_GGML_ATTENTION_POLICY_INIT;
    trellis_ggml_attention_policy shape_flow_hr_attention =
        TRELLIS_GGML_ATTENTION_POLICY_INIT;
    trellis_ggml_attention_policy texture_flow_attention =
        TRELLIS_GGML_ATTENTION_POLICY_INIT;

    const char * pipeline_type =
        options->pipeline_type != NULL && options->pipeline_type[0] != '\0' ?
            options->pipeline_type :
            model_package.profile;
    int use_1024_cascade = 0;
    int final_resolution = options->resolution > 0 ? options->resolution : 512;
    int sparse_cond_resolution = options->cond_resolution > 0 ? options->cond_resolution : 512;
    int sparse_output_resolution = options->sparse_resolution > 0 ? options->sparse_resolution : 32;
    if (pipeline_type[0] != '\0') {
        if (strcmp(pipeline_type, "512") == 0) {
            final_resolution = 512;
            sparse_cond_resolution = 512;
            sparse_output_resolution = 32;
        } else if (strcmp(pipeline_type, "1024") == 0) {
            final_resolution = 1024;
            sparse_cond_resolution = 1024;
            sparse_output_resolution = 64;
        } else if (strcmp(pipeline_type, "1024_cascade") == 0) {
            use_1024_cascade = 1;
            final_resolution = 1024;
            sparse_cond_resolution = 512;
            sparse_output_resolution = 32;
        } else if (strcmp(pipeline_type, "1536_cascade") == 0) {
            use_1024_cascade = 1;
            final_resolution = 1536;
            sparse_cond_resolution = 512;
            sparse_output_resolution = 32;
        } else {
            TRELLIS_ERROR("pipeline: unsupported --pipeline '%s'", pipeline_type);
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
    }
    TRELLIS_INFO(
        "pipeline: type=%s final_resolution=%d sparse_cond=%d sparse_resolution=%d",
        use_1024_cascade ?
            (final_resolution == 1536 ? "1536_cascade" : "1024_cascade") :
            (final_resolution == 1024 ? "1024" : "512"),
        final_resolution,
        sparse_cond_resolution,
        sparse_output_resolution);
    if (projected_conditioning && !use_1024_cascade) {
        TRELLIS_ERROR(
            "pipeline: Pixal3D requires a cascade profile (use --pipeline 1024_cascade or 1536_cascade)");
        status = TRELLIS_STATUS_INVALID_ARGUMENT;
        goto cleanup;
    }

    const char * shape_flow_lr_role =
        use_1024_cascade || final_resolution < 1024 ?
            "shape_flow_512" : "shape_flow_1024";
    const char * texture_flow_role =
        final_resolution >= 1024 ? "texture_flow_1024" : "texture_flow_512";
    sparse_flow_component = trellis_model_package_find_component(
        &model_package, "sparse_structure_flow");
    shape_flow_lr_component = trellis_model_package_find_component(
        &model_package, shape_flow_lr_role);
    texture_flow_component = trellis_model_package_find_component(
        &model_package, texture_flow_role);
    if (use_1024_cascade) {
        shape_flow_hr_component = trellis_model_package_find_component(
            &model_package, "shape_flow_1024");
    }
    if (sparse_flow_component == NULL || shape_flow_lr_component == NULL ||
        texture_flow_component == NULL ||
        (use_1024_cascade && shape_flow_hr_component == NULL) ||
        !pipeline_component_path(&model_package, "sparse_structure_flow", sparse_flow_path) ||
        !pipeline_component_path(&model_package, "sparse_structure_decoder", sparse_decoder_path) ||
        !pipeline_component_path(&model_package, shape_flow_lr_role, shape_flow_lr_path) ||
        (use_1024_cascade &&
         !pipeline_component_path(&model_package, "shape_flow_1024", shape_flow_hr_path)) ||
        !pipeline_component_path(&model_package, texture_flow_role, texture_flow_path) ||
        !pipeline_component_path(&model_package, "shape_decoder", shape_decoder_path) ||
        !pipeline_component_path(&model_package, "texture_decoder", texture_decoder_path)) {
        status = TRELLIS_STATUS_PARSE_ERROR;
        goto cleanup;
    }
    if (projected_conditioning &&
        (pixal_naf_path == NULL || pixal_naf_path[0] == '\0')) {
        if (!pipeline_component_path(
                &model_package, "naf_encoder", naf_component_path)) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
        pixal_naf_path = naf_component_path;
    }

    sparse_flow_attention = pipeline_component_attention_policy(
        options, sparse_flow_component);
    shape_flow_lr_attention = pipeline_component_attention_policy(
        options, shape_flow_lr_component);
    shape_flow_hr_attention = use_1024_cascade ?
        pipeline_component_attention_policy(options, shape_flow_hr_component) :
        shape_flow_lr_attention;
    texture_flow_attention = pipeline_component_attention_policy(
        options, texture_flow_component);
    TRELLIS_INFO(
        "attention policy: sparse=%s shape_lr=%s shape_hr=%s texture=%s%s",
        pipeline_attention_policy_name(&sparse_flow_attention),
        pipeline_attention_policy_name(&shape_flow_lr_attention),
        pipeline_attention_policy_name(&shape_flow_hr_attention),
        pipeline_attention_policy_name(&texture_flow_attention),
        projected_conditioning && !options->use_ggml_flash_attn ?
            " (Pixal3D strict BF16 Flash package defaults)" : "");

    if (options->model_cache) {
        const size_t model_cache_budget_bytes = model_cache_budget_bytes_from_options(options);
        status = trellis_pipeline_model_cache_init(
            &model_cache,
            &graph_backend,
            sparse_backend_kind != TRELLIS_SPARSE_BACKEND_CUDA,
            model_cache_budget_bytes);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        model_cache_initialized = 1;
        model_cache_ptr = &model_cache;
    }

    TRELLIS_INFO("[1/5] SparseStructureFlowModel image -> sparse structure");
    trellis_sparse_structure_options sparse_structure;
    memset(&sparse_structure, 0, sizeof(sparse_structure));
    sparse_structure.model_dir = options->model_dir;
    sparse_structure.flow_path = sparse_flow_path;
    sparse_structure.decoder_path = sparse_decoder_path;
    sparse_structure.dino_dir = options->dino_dir;
    sparse_structure.image_path = sparse_structure_image_path;
    sparse_structure.latent_size = options->latent_size > 0 ? options->latent_size : 16;
    sparse_structure.steps = options->sparse_structure_steps > 0 ? options->sparse_structure_steps : 12;
    sparse_structure.cond_resolution = sparse_cond_resolution;
    sparse_structure.sparse_resolution = sparse_output_resolution;
    sparse_structure.seed = options->seed == 0 ? 1u : options->seed;
    sparse_structure.flow_blocks_override = options->flow_blocks_override;
    sparse_structure.flow_block_parts_override = options->flow_block_parts_override;
    sparse_structure.flow_no_rope = options->flow_no_rope;
    sparse_structure.projected_conditioning = projected_conditioning;
    sparse_structure.emulate_bf16_blocks = pipeline_component_emulates_bf16(
        options, sparse_flow_component);
    sparse_structure.attention_policy = sparse_flow_attention;
    sparse_structure.use_ggml_flash_attn =
        sparse_flow_attention.mode != TRELLIS_GGML_ATTENTION_MODE_EXPLICIT;
    sparse_structure.voxel_threshold = 0.0f;
    sparse_structure.camera_angle_x = pixal_camera_angle_x;
    sparse_structure.camera_distance = pixal_camera_distance;
    sparse_structure.mesh_scale = pixal_mesh_scale;
    sparse_structure.backend = &graph_backend;
    sparse_structure.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    sparse_structure.cache = model_cache_ptr;

    int64_t perf_start_us = ggml_time_us();
    status = trellis_pipeline_run_sparse_structure(&sparse_structure, &sparse_structure_result);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    trellis_pipeline_model_cache_unpin_all(model_cache_ptr);
    trellis_perf_stage_log("stage1_total", ggml_time_us() - perf_start_us);
    if (sparse_structure_result.n_coords <= 0 ||
        sparse_structure_result.coords_bxyz == NULL ||
        sparse_structure_result.cond == NULL) {
        TRELLIS_ERROR("sparse structure: produced no coords for structured latent flow");
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    const float * shape_lr_cond = sparse_structure_result.cond;
    int shape_lr_cond_tokens = sparse_structure_result.cond_tokens;
    if (sparse_structure_result.projected_conditioning) {
        TRELLIS_INFO("[2a/5] Pixal3D DINOv3 + NAF -> shape-512 projected condition");
        trellis_image_condition_options cond_options;
        memset(&cond_options, 0, sizeof(cond_options));
        cond_options.model_dir = options->model_dir;
        cond_options.dino_dir = options->dino_dir;
        cond_options.image_path = sparse_structure_image_path;
        cond_options.naf_path = pixal_naf_path;
        cond_options.cond_resolution = 512;
        cond_options.projection_grid_resolution = 32;
        cond_options.projection_channels = 2048;
        cond_options.naf_target_resolution = 512;
        cond_options.projection_coords_bxyz = sparse_structure_result.coords_bxyz;
        cond_options.projection_n_coords = sparse_structure_result.n_coords;
        cond_options.camera_angle_x = sparse_structure.camera_angle_x;
        cond_options.camera_distance = sparse_structure.camera_distance;
        cond_options.mesh_scale = sparse_structure.mesh_scale;
        cond_options.backend = &graph_backend;
        cond_options.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
        cond_options.sparse_backend = shared_sparse_backend;
        cond_options.cache = model_cache_ptr;
        perf_start_us = ggml_time_us();
        status = trellis_pipeline_run_image_condition(&cond_options, &cond_shape_512);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        trellis_pipeline_model_cache_unpin_all(model_cache_ptr);
        trellis_perf_stage_log("pixal_cond_shape_512", ggml_time_us() - perf_start_us);
        shape_lr_cond = cond_shape_512.cond;
        shape_lr_cond_tokens = cond_shape_512.cond_tokens;
    }

    TRELLIS_INFO(
        "[2/5] SLatFlowModel image -> shape SLat tokens=%lld cond_tokens=%d resolution=%d",
        (long long) sparse_structure_result.n_coords,
        sparse_structure_result.cond_tokens,
        use_1024_cascade ? 512 : final_resolution);
    trellis_structured_latent_options structured_latent;
    memset(&structured_latent, 0, sizeof(structured_latent));
    structured_latent.model_dir = options->model_dir;
    structured_latent.flow_override_path =
        options->flow_override_path != NULL && options->flow_override_path[0] != '\0' ?
            options->flow_override_path : shape_flow_lr_path;
    structured_latent.flow_component = TRELLIS_COMPONENT_SHAPE_SLAT_FLOW;
    structured_latent.label = "shape";
    structured_latent.normalization_key = "shape_slat_normalization";
    structured_latent.coords_bxyz = sparse_structure_result.coords_bxyz;
    structured_latent.n_coords = sparse_structure_result.n_coords;
    structured_latent.cond = shape_lr_cond;
    structured_latent.cond_tokens = shape_lr_cond_tokens;
    structured_latent.projected_cond = cond_shape_512.projected;
    structured_latent.projected_tokens = cond_shape_512.projected_tokens;
    structured_latent.projected_channels = cond_shape_512.projected_channels;
    structured_latent.noise_seed = options->noise_seed == 0 ? 18u : options->noise_seed;
    structured_latent.resolution = use_1024_cascade ? 512 : final_resolution;
    structured_latent.steps = options->structured_latent_steps > 0 ? options->structured_latent_steps : 12;
    structured_latent.rescale_t = options->rescale_t > 0.0f ? options->rescale_t : 3.0f;
    structured_latent.guidance_strength = options->guidance_strength;
    structured_latent.guidance_rescale = options->guidance_rescale;
    structured_latent.guidance_min = options->guidance_min;
    structured_latent.guidance_max = options->guidance_max;
    if (structured_latent.guidance_strength == 0.0f &&
        structured_latent.guidance_rescale == 0.0f &&
        structured_latent.guidance_min == 0.0f &&
        structured_latent.guidance_max == 0.0f) {
        structured_latent.guidance_strength = 7.5f;
        structured_latent.guidance_rescale = 0.5f;
        structured_latent.guidance_min = 0.6f;
        structured_latent.guidance_max = 1.0f;
    }
    structured_latent.flow_blocks_override = options->flow_blocks_override;
    structured_latent.flow_block_parts_override = options->flow_block_parts_override;
    structured_latent.flow_no_rope = options->flow_no_rope;
    structured_latent.emulate_bf16_blocks = pipeline_component_emulates_bf16(
        options, shape_flow_lr_component);
    structured_latent.attention_policy = shape_flow_lr_attention;
    structured_latent.use_ggml_flash_attn =
        shape_flow_lr_attention.mode != TRELLIS_GGML_ATTENTION_MODE_EXPLICIT;
    structured_latent.backend = &graph_backend;
    structured_latent.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    structured_latent.cache = model_cache_ptr;

    perf_start_us = ggml_time_us();
    status = trellis_pipeline_run_structured_latent(&structured_latent, &shape_latent);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    trellis_pipeline_model_cache_unpin_all(model_cache_ptr);
    trellis_perf_stage_log(
        use_1024_cascade ? "shape_slat_denoise_lr512" : "shape_slat_denoise",
        ggml_time_us() - perf_start_us);
    trellis_image_condition_result_free(&cond_shape_512);

    if (use_1024_cascade) {
        shape_latent_lr = shape_latent;
        memset(&shape_latent, 0, sizeof(shape_latent));

        TRELLIS_INFO("[2b/5] FlexiDualGridVaeDecoder 512 shape SLat -> cascade HR coords");
        trellis_pipeline_mesh_options cascade_upsample_options;
        memset(&cascade_upsample_options, 0, sizeof(cascade_upsample_options));
        cascade_upsample_options.model_dir = options->model_dir;
        cascade_upsample_options.decoder_override_path =
            options->decoder_override_path != NULL && options->decoder_override_path[0] != '\0' ?
                options->decoder_override_path : shape_decoder_path;
        cascade_upsample_options.latent = &shape_latent_lr;
        cascade_upsample_options.resolution = 512;
        cascade_upsample_options.decode_max_levels = 0;
        cascade_upsample_options.decode_max_input_tokens = options->decode_max_input_tokens;
        cascade_upsample_options.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
        cascade_upsample_options.sparse_backend_kind = sparse_backend_kind;
        cascade_upsample_options.sparse_device = sparse_device;
        cascade_upsample_options.sparse_backend = shared_sparse_backend;
        cascade_upsample_options.cache = model_cache_ptr;
        perf_start_us = ggml_time_us();
        status = trellis_pipeline_decode_shape_latent_decoder_coords(
            &cascade_upsample_options,
            &cascade_decoder_coords,
            &cascade_decoder_n);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        trellis_pipeline_model_cache_unpin_all(model_cache_ptr);
        trellis_perf_stage_log("shape_cascade_decode_coords", ggml_time_us() - perf_start_us);
        const int64_t decoded_coord_count = cascade_decoder_n;
        int actual_hr_resolution = final_resolution;
        for (;;) {
            free(cascade_coords);
            cascade_coords = NULL;
            cascade_n = 0;
            status = trellis_pipeline_quantize_cascade_coords(
                cascade_decoder_coords,
                cascade_decoder_n,
                512,
                actual_hr_resolution,
                task_adapter->cascade_quantization ==
                        TRELLIS_IMAGE_TO_3D_CASCADE_QUANTIZATION_ROUND ?
                    TRELLIS_CASCADE_COORD_QUANTIZE_PIXAL :
                    TRELLIS_CASCADE_COORD_QUANTIZE_TRELLIS,
                &cascade_coords,
                &cascade_n);
            if (status != TRELLIS_STATUS_OK || options->max_num_tokens <= 0 ||
                cascade_n < options->max_num_tokens || actual_hr_resolution == 1024) {
                break;
            }
            actual_hr_resolution -= 128;
            if (actual_hr_resolution < 1024) {
                actual_hr_resolution = 1024;
            }
        }
        free(cascade_decoder_coords);
        cascade_decoder_coords = NULL;
        cascade_decoder_n = 0;
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        TRELLIS_INFO(
            "pipeline cascade: decoder_coords=%lld quantized_shape_tokens=%lld resolution=%d",
            (long long) decoded_coord_count,
            (long long) cascade_n,
            actual_hr_resolution);
        if (options->max_num_tokens > 0 && cascade_n > options->max_num_tokens) {
            TRELLIS_WARN(
                "pipeline cascade: quantized tokens %lld exceed max_num_tokens=%d at minimum resolution 1024",
                (long long) cascade_n,
                options->max_num_tokens);
        }
        trellis_structured_latent_free(&shape_latent_lr);

        TRELLIS_INFO(
            sparse_structure_result.projected_conditioning ?
                "[2c/5] Pixal3D DINOv3 + NAF -> shape-1024 projected condition" :
                "[2c/5] DINOv3 image encoder -> 1024 condition");
        trellis_image_condition_options cond_options;
        memset(&cond_options, 0, sizeof(cond_options));
        cond_options.model_dir = options->model_dir;
        cond_options.dino_dir = options->dino_dir;
        cond_options.image_path = sparse_structure_image_path;
        cond_options.naf_path = pixal_naf_path;
        cond_options.cond_resolution = 1024;
        if (sparse_structure_result.projected_conditioning) {
            cond_options.projection_grid_resolution = actual_hr_resolution / 16;
            cond_options.projection_channels = 2048;
            cond_options.naf_target_resolution = 512;
            cond_options.projection_coords_bxyz = cascade_coords;
            cond_options.projection_n_coords = cascade_n;
            cond_options.camera_angle_x = sparse_structure.camera_angle_x;
            cond_options.camera_distance = sparse_structure.camera_distance;
            cond_options.mesh_scale = sparse_structure.mesh_scale;
        }
        cond_options.backend = &graph_backend;
        cond_options.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
        cond_options.sparse_backend = shared_sparse_backend;
        cond_options.cache = model_cache_ptr;
        perf_start_us = ggml_time_us();
        status = trellis_pipeline_run_image_condition(&cond_options, &cond_1024);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        trellis_pipeline_model_cache_unpin_all(model_cache_ptr);
        trellis_perf_stage_log(
            sparse_structure_result.projected_conditioning ?
                "pixal_cond_shape_1024" : "dino_cond_1024",
            ggml_time_us() - perf_start_us);

        TRELLIS_INFO(
            "[2c/5] SLatFlowModel 1024 image -> shape SLat tokens=%lld cond_tokens=%d resolution=%d",
            (long long) cascade_n,
            cond_1024.cond_tokens,
            actual_hr_resolution);
        trellis_structured_latent_options hr_shape_options = structured_latent;
        hr_shape_options.label = actual_hr_resolution == 1024 ? "shape1024" : "shape_hr";
        hr_shape_options.flow_override_path = shape_flow_hr_path;
        hr_shape_options.coords_bxyz = cascade_coords;
        hr_shape_options.n_coords = cascade_n;
        hr_shape_options.cond = cond_1024.cond;
        hr_shape_options.cond_tokens = cond_1024.cond_tokens;
        hr_shape_options.projected_cond = cond_1024.projected;
        hr_shape_options.projected_tokens = cond_1024.projected_tokens;
        hr_shape_options.projected_channels = cond_1024.projected_channels;
        hr_shape_options.noise_seed = options->noise_seed == 0 ? 19u : options->noise_seed + 1u;
        hr_shape_options.resolution = actual_hr_resolution;
        hr_shape_options.emulate_bf16_blocks = pipeline_component_emulates_bf16(
            options, shape_flow_hr_component);
        hr_shape_options.attention_policy = shape_flow_hr_attention;
        hr_shape_options.use_ggml_flash_attn =
            shape_flow_hr_attention.mode != TRELLIS_GGML_ATTENTION_MODE_EXPLICIT;
        perf_start_us = ggml_time_us();
        status = trellis_pipeline_run_structured_latent(&hr_shape_options, &shape_latent);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        trellis_pipeline_model_cache_unpin_all(model_cache_ptr);
        trellis_perf_stage_log(
            actual_hr_resolution == 1024 ?
                "shape_slat_denoise_hr1024" : "shape_slat_denoise_hr_dynamic",
            ggml_time_us() - perf_start_us);
    }

    TRELLIS_INFO("[3/5] FlexiDualGridVaeDecoder shape SLat -> mesh/subdivision guides");
    trellis_pipeline_mesh_options mesh_options;
    memset(&mesh_options, 0, sizeof(mesh_options));
    mesh_options.model_dir = options->model_dir;
    mesh_options.decoder_override_path =
        options->decoder_override_path != NULL && options->decoder_override_path[0] != '\0' ?
            options->decoder_override_path : shape_decoder_path;
    mesh_options.latent = &shape_latent;
    mesh_options.resolution = shape_latent.resolution;
    mesh_options.decode_max_levels = options->decode_max_levels;
    mesh_options.decode_max_input_tokens = options->decode_max_input_tokens;
    mesh_options.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    mesh_options.sparse_backend_kind = sparse_backend_kind;
    mesh_options.sparse_device = sparse_device;
    mesh_options.sparse_backend = shared_sparse_backend;
    mesh_options.cache = model_cache_ptr;

    perf_start_us = ggml_time_us();
    status = trellis_pipeline_decode_shape_latent_mesh(&mesh_options, &shape_subs, &mesh);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    trellis_pipeline_model_cache_unpin_all(model_cache_ptr);
    trellis_perf_stage_log("shape_slat_decode", ggml_time_us() - perf_start_us);
    if (shared_sparse_backend != NULL) {
        size_t released_bytes = 0;
        status = trellis_sparse_backend_trim(
            shared_sparse_backend,
            TRELLIS_SPARSE_TRIM_FREE_BUFFERS | TRELLIS_SPARSE_TRIM_WEIGHTS,
            &released_bytes);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "SparseUnet Vulkan decoder backend: post-shape trim failed: %s",
                trellis_status_string(status));
            goto cleanup;
        }
        TRELLIS_INFO(
            "SparseUnet Vulkan decoder backend: released %.1f MiB of inactive shape buffers; subdivision guides retained",
            (double) released_bytes / (1024.0 * 1024.0));
    }
    if (material_dump_dir != NULL && material_dump_dir[0] != '\0') {
        char raw_mesh_path[4096];
        if (!make_material_dump_path(raw_mesh_path, sizeof(raw_mesh_path), material_dump_dir, "raw.meshbin")) {
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
        status = trellis_pipeline_write_meshbin(raw_mesh_path, &mesh);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        TRELLIS_INFO("pipeline: dumped raw shape mesh to %s", raw_mesh_path);
    }
    if (options->mesh_postprocess) {
        status = trellis_pipeline_postprocess_mesh_with_vkmesh(
            &mesh,
            &gltf_projection_mesh,
            options->vkmesh_path,
            options->mesh_postprocess_decimation_target > 0 ? options->mesh_postprocess_decimation_target : 1000000,
            options->mesh_postprocess_no_simplify,
            options->mesh_remesh,
            options->mesh_remesh_resolution > 0 ? options->mesh_remesh_resolution : shape_latent.resolution,
            options->mesh_remesh_band > 0.0f ? options->mesh_remesh_band : 1.0f,
            options->mesh_remesh_project > 0.0f ? options->mesh_remesh_project : 0.0f,
            options->device,
            options->vkmesh_gpu_workspace_budget_mib);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
    }

    const float * texture_cond = use_1024_cascade ?
        cond_1024.cond : sparse_structure_result.cond;
    int texture_cond_tokens = use_1024_cascade ?
        cond_1024.cond_tokens : sparse_structure_result.cond_tokens;
    if (sparse_structure_result.projected_conditioning) {
        trellis_image_condition_result_free(&cond_1024);
        TRELLIS_INFO("[4a/5] Pixal3D DINOv3 + NAF -> texture projected condition");
        trellis_image_condition_options cond_options;
        memset(&cond_options, 0, sizeof(cond_options));
        cond_options.model_dir = options->model_dir;
        cond_options.dino_dir = options->dino_dir;
        cond_options.image_path = sparse_structure_image_path;
        cond_options.naf_path = pixal_naf_path;
        cond_options.cond_resolution = 1024;
        cond_options.projection_grid_resolution = shape_latent.resolution / 16;
        cond_options.projection_channels = 2048;
        cond_options.naf_target_resolution = 1024;
        cond_options.projection_coords_bxyz = shape_latent.coords_bxyz;
        cond_options.projection_n_coords = shape_latent.n_coords;
        cond_options.camera_angle_x = sparse_structure.camera_angle_x;
        cond_options.camera_distance = sparse_structure.camera_distance;
        cond_options.mesh_scale = sparse_structure.mesh_scale;
        cond_options.backend = &graph_backend;
        cond_options.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
        cond_options.sparse_backend = shared_sparse_backend;
        cond_options.cache = model_cache_ptr;
        perf_start_us = ggml_time_us();
        status = trellis_pipeline_run_image_condition(&cond_options, &cond_1024);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        trellis_pipeline_model_cache_unpin_all(model_cache_ptr);
        trellis_perf_stage_log("pixal_cond_texture", ggml_time_us() - perf_start_us);
        texture_cond = cond_1024.cond;
        texture_cond_tokens = cond_1024.cond_tokens;
    }

    TRELLIS_INFO(
        "[4/5] SLatFlowModel image+shape -> texture SLat tokens=%lld cond_tokens=%d resolution=%d",
        (long long) shape_latent.n_coords,
        texture_cond_tokens,
        shape_latent.resolution);
    trellis_structured_latent_options texture_options;
    memset(&texture_options, 0, sizeof(texture_options));
    texture_options.model_dir = options->model_dir;
    texture_options.flow_override_path = texture_flow_path;
    texture_options.flow_component = TRELLIS_COMPONENT_TEX_SLAT_FLOW;
    texture_options.label = "texture";
    texture_options.normalization_key = "tex_slat_normalization";
    texture_options.coords_bxyz = shape_latent.coords_bxyz;
    texture_options.n_coords = shape_latent.n_coords;
    texture_options.cond = texture_cond;
    texture_options.cond_tokens = texture_cond_tokens;
    texture_options.projected_cond = cond_1024.projected;
    texture_options.projected_tokens = cond_1024.projected_tokens;
    texture_options.projected_channels = cond_1024.projected_channels;
    texture_options.concat_cond = shape_latent.feats;
    texture_options.concat_channels = shape_latent.channels;
    texture_options.noise_seed = options->noise_seed == 0 ? (use_1024_cascade ? 20u : 19u) : options->noise_seed + (use_1024_cascade ? 2u : 1u);
    texture_options.resolution = shape_latent.resolution;
    texture_options.steps = options->structured_latent_steps > 0 ? options->structured_latent_steps : 12;
    texture_options.rescale_t = 3.0f;
    texture_options.guidance_strength = 1.0f;
    texture_options.guidance_rescale = 0.0f;
    texture_options.guidance_min = 0.6f;
    texture_options.guidance_max = 0.9f;
    texture_options.flow_blocks_override = options->flow_blocks_override;
    texture_options.flow_block_parts_override = options->flow_block_parts_override;
    texture_options.flow_no_rope = options->flow_no_rope;
    texture_options.emulate_bf16_blocks = pipeline_component_emulates_bf16(
        options, texture_flow_component);
    texture_options.attention_policy = texture_flow_attention;
    texture_options.use_ggml_flash_attn =
        texture_flow_attention.mode != TRELLIS_GGML_ATTENTION_MODE_EXPLICIT;
    texture_options.backend = &graph_backend;
    texture_options.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    texture_options.cache = model_cache_ptr;

    perf_start_us = ggml_time_us();
    status = trellis_pipeline_run_structured_latent(&texture_options, &texture_latent);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    trellis_pipeline_model_cache_unpin_all(model_cache_ptr);
    trellis_perf_stage_log("tex_slat_denoise", ggml_time_us() - perf_start_us);
    trellis_sparse_structure_result_free(&sparse_structure_result);

    TRELLIS_INFO("[5/5] SparseUnetVaeDecoder texture SLat -> PBR voxels");
    trellis_pipeline_texture_options tex_decode;
    memset(&tex_decode, 0, sizeof(tex_decode));
    tex_decode.model_dir = options->model_dir;
    tex_decode.decoder_override_path = texture_decoder_path;
    tex_decode.latent = &texture_latent;
    tex_decode.guide_subs = &shape_subs;
    tex_decode.decode_max_levels = options->decode_max_levels;
    tex_decode.decode_max_input_tokens = options->decode_max_input_tokens;
    tex_decode.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    tex_decode.sparse_backend_kind = sparse_backend_kind;
    tex_decode.sparse_device = sparse_device;
    tex_decode.sparse_backend = shared_sparse_backend;
    tex_decode.cache = model_cache_ptr;

    perf_start_us = ggml_time_us();
    status = trellis_pipeline_decode_texture_latent_voxels(&tex_decode, &pbr_voxels);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    trellis_pipeline_model_cache_unpin_all(model_cache_ptr);
    trellis_perf_stage_log("tex_slat_decode", ggml_time_us() - perf_start_us);
    TRELLIS_INFO(
        "pipeline: decoded PBR voxels=%lld channels=%d resolution=%d",
        (long long) pbr_voxels.n_coords,
        pbr_voxels.channels,
        pbr_voxels.resolution);
    if (shared_sparse_backend != NULL) {
        trellis_sparse_c2s_guides_free(&shape_subs);
        trellis_sparse_backend_destroy(shared_sparse_backend);
        shared_sparse_backend = NULL;
        TRELLIS_INFO("SparseUnet Vulkan decoder backend: released after texture decode");
    }

    if (material_dump_dir != NULL && material_dump_dir[0] != '\0') {
        const trellis_mesh_host * sample_mesh =
            gltf_projection_mesh.vertices != NULL && gltf_projection_mesh.faces != NULL ?
                &gltf_projection_mesh :
                NULL;
        status = trellis_pipeline_dump_material_inputs_if_requested(
            material_dump_dir,
            &mesh,
            sample_mesh,
            &pbr_voxels);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
    }

    const int texture_size = options->texture_size > 0 ? options->texture_size : 1024;
    const trellis_mesh_host * sample_mesh =
        gltf_projection_mesh.vertices != NULL && gltf_projection_mesh.faces != NULL ?
            &gltf_projection_mesh :
            NULL;
    status = trellis_pipeline_write_gltf_ex(
        output_gltf_path,
        &mesh,
        sample_mesh,
        &pbr_voxels,
        texture_size,
        options->device,
        task_adapter->gltf_coordinate_transform);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("pipeline: glTF export failed: %s", trellis_status_string(status));
        goto cleanup;
    }

cleanup:
    free(cascade_decoder_coords);
    free(cascade_coords);
    trellis_mesh_free(&gltf_projection_mesh);
    trellis_mesh_free(&mesh);
    trellis_pbr_voxels_free(&pbr_voxels);
    trellis_sparse_c2s_guides_free(&shape_subs);
    trellis_structured_latent_free(&texture_latent);
    trellis_structured_latent_free(&shape_latent_lr);
    trellis_structured_latent_free(&shape_latent);
    trellis_image_condition_result_free(&cond_shape_512);
    trellis_image_condition_result_free(&cond_1024);
    trellis_sparse_structure_result_free(&sparse_structure_result);
    if (model_cache_initialized) {
        trellis_pipeline_model_cache_free(&model_cache);
    }
    if (sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA) {
        trellis_cuda_free(&cuda);
    }
    trellis_sparse_backend_destroy(shared_sparse_backend);
    trellis_backend_free(&graph_backend);
    unlink_if_set(temp_birefnet_image);
    unlink_if_set(temp_image);
    trellis_model_package_free(&model_package);
    return status;
}
