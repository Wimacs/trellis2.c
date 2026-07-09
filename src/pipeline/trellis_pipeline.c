#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "trellis.h"
#include "trellis_platform.h"
#include "sparse/trellis_sparse_backend.h"
#include "trellis_pipeline_internal.h"

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
        false,
        "vkmesh topology cleanup and Vulkan compute UV-space PBR texture bake are implemented; optional remesh_narrow_band_dc is pending",
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
    char * out,
    size_t out_size) {
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

static trellis_status quantize_cascade_coords(
    const int32_t * decoder_coords,
    int64_t decoder_n,
    int lr_resolution,
    int hr_resolution,
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
            &decoder_ptr);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("texture decode: cache texture decoder load failed: %s", trellis_status_string(status));
            goto cleanup;
        }
    } else {
        if (!make_texture_decoder_path(options->model_dir, decoder_path, sizeof(decoder_path))) {
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
    int no_simplify) {
    char target[64];
    snprintf(target, sizeof(target), "%d", decimation_target > 0 ? decimation_target : 1000000);
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
    char * argv[16];
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
    int no_simplify) {
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

    trellis_mesh_host processed;
    trellis_mesh_host projection;
    memset(&processed, 0, sizeof(processed));
    memset(&projection, 0, sizeof(projection));

    TRELLIS_INFO(
        "mesh postprocess: running vkmesh C API target=%d no_simplify=%d input_faces=%lld",
        vkmesh_options.decimation_target,
        no_simplify ? 1 : 0,
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
        "mesh postprocess: running vkmesh target=%d no_simplify=%d input_faces=%lld",
        decimation_target > 0 ? decimation_target : 1000000,
        no_simplify ? 1 : 0,
        (long long) mesh->n_faces);
    if (!run_vkmesh_postprocess_command(
            vkmesh_path,
            input_mesh,
            output_mesh,
            projection_mesh_out != NULL ? projection_mesh : NULL,
            decimation_target,
            no_simplify)) {
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
    if (options == NULL || options->model_dir == NULL || options->dino_dir == NULL ||
        options->image_path == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const char * output_gltf_path =
        options->gltf_path != NULL && options->gltf_path[0] != '\0' ?
            options->gltf_path :
            "output.glb";

    char temp_image[4096];
    char temp_birefnet_image[4096];
    temp_image[0] = '\0';
    temp_birefnet_image[0] = '\0';
    trellis_status status = TRELLIS_STATUS_OK;
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

    const char * sparse_structure_image_path = options->image_path;
    if (path_has_ext(options->image_path, "webp")) {
        if (!trellis_make_temp_path(
            temp_image,
            sizeof(temp_image),
            "trellis2_image_to_gltf",
            ".png")) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        TRELLIS_INFO("image prep: converting WebP -> PNG via ffmpeg: %s", temp_image);
        if (!convert_webp_to_png_ffmpeg(options->image_path, temp_image)) {
            TRELLIS_ERROR("image prep: WebP conversion failed; install ffmpeg or convert the image to PNG/JPEG first");
            unlink_if_set(temp_image);
            return TRELLIS_STATUS_IO_ERROR;
        }
        sparse_structure_image_path = temp_image;
    }
    if (options->birefnet_path != NULL && options->birefnet_path[0] != '\0') {
        if (!trellis_make_temp_path(
            temp_birefnet_image,
            sizeof(temp_birefnet_image),
            "trellis2_birefnet",
            ".png")) {
            unlink_if_set(temp_image);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        TRELLIS_INFO("image prep: running BiRefNet background removal -> %s", temp_birefnet_image);
        status = apply_birefnet_background_removal(
            sparse_structure_image_path,
            options->birefnet_path,
            temp_birefnet_image,
            graph_backend_kind,
            options->device);
        if (status != TRELLIS_STATUS_OK) {
            unlink_if_set(temp_birefnet_image);
            unlink_if_set(temp_image);
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
            return status;
        }
    }
    TRELLIS_INFO("SparseUnet backend: %s device=%d", sparse_backend_kind_name(sparse_backend_kind), sparse_device);

    trellis_sparse_structure_result sparse_structure_result;
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
    int use_ggml_flash_attn = options->no_ggml_flash_attn ? 0 : 1;
    if (options->use_ggml_flash_attn) {
        use_ggml_flash_attn = 1;
    }
    TRELLIS_INFO("ggml flash attention: %s", use_ggml_flash_attn ? "enabled" : "disabled");

    const char * pipeline_type =
        options->pipeline_type != NULL && options->pipeline_type[0] != '\0' ?
            options->pipeline_type :
            "";
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
        } else {
            TRELLIS_ERROR("pipeline: unsupported --pipeline '%s'", pipeline_type);
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
    }
    TRELLIS_INFO(
        "pipeline: type=%s final_resolution=%d sparse_cond=%d sparse_resolution=%d",
        use_1024_cascade ? "1024_cascade" : (final_resolution == 1024 ? "1024" : "512"),
        final_resolution,
        sparse_cond_resolution,
        sparse_output_resolution);

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
    sparse_structure.use_ggml_flash_attn = use_ggml_flash_attn;
    sparse_structure.voxel_threshold = 0.0f;
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

    TRELLIS_INFO(
        "[2/5] SLatFlowModel image -> shape SLat tokens=%lld cond_tokens=%d resolution=%d",
        (long long) sparse_structure_result.n_coords,
        sparse_structure_result.cond_tokens,
        use_1024_cascade ? 512 : final_resolution);
    trellis_structured_latent_options structured_latent;
    memset(&structured_latent, 0, sizeof(structured_latent));
    structured_latent.model_dir = options->model_dir;
    structured_latent.flow_override_path = options->flow_override_path;
    structured_latent.flow_component = TRELLIS_COMPONENT_SHAPE_SLAT_FLOW;
    structured_latent.label = "shape";
    structured_latent.normalization_key = "shape_slat_normalization";
    structured_latent.coords_bxyz = sparse_structure_result.coords_bxyz;
    structured_latent.n_coords = sparse_structure_result.n_coords;
    structured_latent.cond = sparse_structure_result.cond;
    structured_latent.cond_tokens = sparse_structure_result.cond_tokens;
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
    structured_latent.emulate_bf16_blocks = options->emulate_bf16_blocks;
    structured_latent.use_ggml_flash_attn = use_ggml_flash_attn;
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

    if (use_1024_cascade) {
        shape_latent_lr = shape_latent;
        memset(&shape_latent, 0, sizeof(shape_latent));

        TRELLIS_INFO("[2a/5] DINOv3 image encoder -> 1024 condition");
        trellis_image_condition_options cond_options;
        memset(&cond_options, 0, sizeof(cond_options));
        cond_options.dino_dir = options->dino_dir;
        cond_options.image_path = sparse_structure_image_path;
        cond_options.cond_resolution = 1024;
        cond_options.backend = &graph_backend;
        cond_options.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
        cond_options.cache = model_cache_ptr;
        perf_start_us = ggml_time_us();
        status = trellis_pipeline_run_image_condition(&cond_options, &cond_1024);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        trellis_pipeline_model_cache_unpin_all(model_cache_ptr);
        trellis_perf_stage_log("dino_cond_1024", ggml_time_us() - perf_start_us);

        TRELLIS_INFO("[2b/5] FlexiDualGridVaeDecoder 512 shape SLat -> cascade HR coords");
        trellis_pipeline_mesh_options cascade_upsample_options;
        memset(&cascade_upsample_options, 0, sizeof(cascade_upsample_options));
        cascade_upsample_options.model_dir = options->model_dir;
        cascade_upsample_options.decoder_override_path = options->decoder_override_path;
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
        status = quantize_cascade_coords(
            cascade_decoder_coords,
            cascade_decoder_n,
            512,
            1024,
            &cascade_coords,
            &cascade_n);
        free(cascade_decoder_coords);
        cascade_decoder_coords = NULL;
        cascade_decoder_n = 0;
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        TRELLIS_INFO(
            "pipeline cascade: decoder_coords=%lld quantized_shape_tokens=%lld",
            (long long) decoded_coord_count,
            (long long) cascade_n);
        if (options->max_num_tokens > 0 && cascade_n > options->max_num_tokens) {
            TRELLIS_WARN(
                "pipeline cascade: quantized tokens %lld exceed max_num_tokens=%d; matching PyTorch 1024_cascade keeps resolution 1024",
                (long long) cascade_n,
                options->max_num_tokens);
        }
        trellis_structured_latent_free(&shape_latent_lr);

        TRELLIS_INFO(
            "[2c/5] SLatFlowModel 1024 image -> shape SLat tokens=%lld cond_tokens=%d",
            (long long) cascade_n,
            cond_1024.cond_tokens);
        trellis_structured_latent_options hr_shape_options = structured_latent;
        hr_shape_options.label = "shape1024";
        hr_shape_options.coords_bxyz = cascade_coords;
        hr_shape_options.n_coords = cascade_n;
        hr_shape_options.cond = cond_1024.cond;
        hr_shape_options.cond_tokens = cond_1024.cond_tokens;
        hr_shape_options.noise_seed = options->noise_seed == 0 ? 19u : options->noise_seed + 1u;
        hr_shape_options.resolution = 1024;
        perf_start_us = ggml_time_us();
        status = trellis_pipeline_run_structured_latent(&hr_shape_options, &shape_latent);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        trellis_pipeline_model_cache_unpin_all(model_cache_ptr);
        trellis_perf_stage_log("shape_slat_denoise_hr1024", ggml_time_us() - perf_start_us);
    }

    TRELLIS_INFO("[3/5] FlexiDualGridVaeDecoder shape SLat -> mesh/subdivision guides");
    trellis_pipeline_mesh_options mesh_options;
    memset(&mesh_options, 0, sizeof(mesh_options));
    mesh_options.model_dir = options->model_dir;
    mesh_options.decoder_override_path = options->decoder_override_path;
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
            options->mesh_postprocess_no_simplify);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
    }

    TRELLIS_INFO(
        "[4/5] SLatFlowModel image+shape -> texture SLat tokens=%lld cond_tokens=%d resolution=%d",
        (long long) shape_latent.n_coords,
        use_1024_cascade ? cond_1024.cond_tokens : sparse_structure_result.cond_tokens,
        shape_latent.resolution);
    trellis_structured_latent_options texture_options;
    memset(&texture_options, 0, sizeof(texture_options));
    texture_options.model_dir = options->model_dir;
    texture_options.flow_component = TRELLIS_COMPONENT_TEX_SLAT_FLOW;
    texture_options.label = "texture";
    texture_options.normalization_key = "tex_slat_normalization";
    texture_options.coords_bxyz = shape_latent.coords_bxyz;
    texture_options.n_coords = shape_latent.n_coords;
    texture_options.cond = use_1024_cascade ? cond_1024.cond : sparse_structure_result.cond;
    texture_options.cond_tokens = use_1024_cascade ? cond_1024.cond_tokens : sparse_structure_result.cond_tokens;
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
    texture_options.emulate_bf16_blocks = options->emulate_bf16_blocks;
    texture_options.use_ggml_flash_attn = use_ggml_flash_attn;
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
    status = trellis_pipeline_write_gltf(output_gltf_path, &mesh, sample_mesh, &pbr_voxels, texture_size);
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
    return status;
}
