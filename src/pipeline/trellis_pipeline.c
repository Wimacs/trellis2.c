#include "trellis.h"
#include "trellis_pipeline_internal.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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
        "sparse coordinate host semantics and dense transformer operators are implemented; varlen FlashAttention CUDA packing is pending",
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
        "mesh extraction and vertex-color OBJ sampling are implemented; full UV unwrap/raster/inpaint PBR texture bake is pending",
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
    pid_t pid = fork();
    if (pid < 0) {
        return 0;
    }
    if (pid == 0) {
        execlp(
            "ffmpeg",
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_path,
            "-frames:v",
            "1",
            output_path,
            (char *) NULL);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return 0;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

static void unlink_if_set(const char * path) {
    if (path != NULL && path[0] != '\0') {
        unlink(path);
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
    if (path == NULL || path[0] == '\0') {
        return 1;
    }
    char tmp[4096];
    int n = snprintf(tmp, sizeof(tmp), "%s", path);
    if (n < 0 || (size_t) n >= sizeof(tmp)) {
        return 0;
    }
    for (char * p = tmp + 1; *p != '\0'; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
                return 0;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
        return 0;
    }
    return 1;
}

static int mkdir_parent(const char * path) {
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    char dir[4096];
    int n = snprintf(dir, sizeof(dir), "%s", path);
    if (n < 0 || (size_t) n >= sizeof(dir)) {
        return 0;
    }
    char * slash = strrchr(dir, '/');
    if (slash == NULL || slash == dir) {
        return 1;
    }
    *slash = '\0';
    return mkdir_p(dir);
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
    int32_t * out_coords = NULL;
    float * out_feats = NULL;

    if (!make_shape_decoder_path(options->model_dir, options->decoder_override_path, decoder_path, sizeof(decoder_path))) {
        TRELLIS_ERROR("pipeline mesh: failed to build shape decoder path");
        goto cleanup;
    }
    const trellis_backend_context * weight_backend = options->cuda;
    int owns_weight_backend = 0;
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
    forward_options.return_subs = subs_out;
    status = trellis_sparse_unet_vae_decoder_forward_backend_f32_host(
        &decoder,
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
    trellis_tensor_store_free(&decoder_store);
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

    if (!make_texture_decoder_path(options->model_dir, decoder_path, sizeof(decoder_path))) {
        TRELLIS_ERROR("texture decode: failed to build texture decoder path");
        goto cleanup;
    }
    const trellis_backend_context * weight_backend = options->cuda;
    int owns_weight_backend = 0;
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
    forward_options.guide_subs = options->guide_subs;
    status = trellis_sparse_unet_vae_decoder_forward_backend_f32_host(
        &decoder,
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
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
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
    trellis_tensor_store_free(&decoder_store);
    if (owns_weight_backend) {
        trellis_backend_free(&cpu_weights_backend);
    }
    if (status != TRELLIS_STATUS_OK) {
        trellis_pbr_voxels_free(voxels_out);
    }
    return status;
}

static float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static uint64_t pbr_coord_key(int32_t x, int32_t y, int32_t z) {
    return (((uint64_t) (uint32_t) x) << 42) ^
        (((uint64_t) (uint32_t) y) << 21) ^
        ((uint64_t) (uint32_t) z);
}

static uint64_t hash_u64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static int next_pow2_size(size_t n, size_t * out) {
    size_t v = 1;
    while (v < n) {
        if (v > SIZE_MAX / 2u) {
            return 0;
        }
        v *= 2u;
    }
    *out = v;
    return 1;
}

static int32_t pbr_hash_lookup(
    const uint64_t * keys,
    const int32_t * values,
    size_t table_size,
    int32_t x,
    int32_t y,
    int32_t z) {
    uint64_t key = pbr_coord_key(x, y, z);
    if (key == 0) {
        key = 1;
    }
    size_t slot = (size_t) hash_u64(key) & (table_size - 1u);
    while (values[slot] >= 0) {
        if (keys[slot] == key) {
            return values[slot];
        }
        slot = (slot + 1u) & (table_size - 1u);
    }
    return -1;
}

static int pbr_sample_color_trilinear(
    const trellis_pbr_voxels * voxels,
    const uint64_t * keys,
    const int32_t * values,
    size_t table_size,
    const float vertex[3],
    float color[3]) {
    const int resolution = voxels->resolution > 0 ? voxels->resolution : 512;
    float gx = (vertex[0] + 0.5f) * (float) resolution;
    float gy = (vertex[1] + 0.5f) * (float) resolution;
    float gz = (vertex[2] + 0.5f) * (float) resolution;
    if (gx < 0.0f) gx = 0.0f;
    if (gy < 0.0f) gy = 0.0f;
    if (gz < 0.0f) gz = 0.0f;
    if (gx > (float) (resolution - 1)) gx = (float) (resolution - 1);
    if (gy > (float) (resolution - 1)) gy = (float) (resolution - 1);
    if (gz > (float) (resolution - 1)) gz = (float) (resolution - 1);

    const int32_t x0 = (int32_t) floorf(gx);
    const int32_t y0 = (int32_t) floorf(gy);
    const int32_t z0 = (int32_t) floorf(gz);
    const int32_t x1 = x0 + 1 < resolution ? x0 + 1 : x0;
    const int32_t y1 = y0 + 1 < resolution ? y0 + 1 : y0;
    const int32_t z1 = z0 + 1 < resolution ? z0 + 1 : z0;
    const float tx = gx - (float) x0;
    const float ty = gy - (float) y0;
    const float tz = gz - (float) z0;

    float acc[3] = {0.0f, 0.0f, 0.0f};
    float weight_sum = 0.0f;
    for (int dz = 0; dz < 2; ++dz) {
        const int32_t z = dz == 0 ? z0 : z1;
        const float wz = dz == 0 ? 1.0f - tz : tz;
        for (int dy = 0; dy < 2; ++dy) {
            const int32_t y = dy == 0 ? y0 : y1;
            const float wy = dy == 0 ? 1.0f - ty : ty;
            for (int dx = 0; dx < 2; ++dx) {
                const int32_t x = dx == 0 ? x0 : x1;
                const float wx = dx == 0 ? 1.0f - tx : tx;
                const float w = wx * wy * wz;
                if (w <= 0.0f) {
                    continue;
                }
                const int32_t idx = pbr_hash_lookup(keys, values, table_size, x, y, z);
                if (idx < 0) {
                    continue;
                }
                const float * src = voxels->attrs + (size_t) idx * (size_t) voxels->channels;
                acc[0] += w * clamp01(src[0]);
                acc[1] += w * clamp01(src[1]);
                acc[2] += w * clamp01(src[2]);
                weight_sum += w;
            }
        }
    }
    if (weight_sum > 1e-8f) {
        color[0] = acc[0] / weight_sum;
        color[1] = acc[1] / weight_sum;
        color[2] = acc[2] / weight_sum;
        return 1;
    }

    const int32_t cx = (int32_t) floorf(gx + 0.5f);
    const int32_t cy = (int32_t) floorf(gy + 0.5f);
    const int32_t cz = (int32_t) floorf(gz + 0.5f);
    int32_t best = -1;
    for (int radius = 0; best < 0 && radius <= 2; ++radius) {
        for (int dz = -radius; best < 0 && dz <= radius; ++dz) {
            for (int dy = -radius; best < 0 && dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (radius > 0 &&
                        abs(dx) != radius && abs(dy) != radius && abs(dz) != radius) {
                        continue;
                    }
                    const int32_t x = cx + dx;
                    const int32_t y = cy + dy;
                    const int32_t z = cz + dz;
                    if (x < 0 || y < 0 || z < 0 ||
                        x >= resolution || y >= resolution || z >= resolution) {
                        continue;
                    }
                    best = pbr_hash_lookup(keys, values, table_size, x, y, z);
                }
            }
        }
    }
    if (best >= 0) {
        const float * src = voxels->attrs + (size_t) best * (size_t) voxels->channels;
        color[0] = clamp01(src[0]);
        color[1] = clamp01(src[1]);
        color[2] = clamp01(src[2]);
        return 1;
    }
    color[0] = 0.8f;
    color[1] = 0.8f;
    color[2] = 0.8f;
    return 0;
}

trellis_status trellis_pipeline_apply_pbr_voxels_to_mesh(
    const trellis_pbr_voxels * voxels,
    trellis_mesh_host * mesh) {
    if (voxels == NULL || mesh == NULL || voxels->coords_bxyz == NULL ||
        voxels->attrs == NULL || voxels->n_coords <= 0 || voxels->channels < 3 ||
        mesh->vertices == NULL || mesh->n_vertices <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    size_t table_size = 0;
    if (!next_pow2_size((size_t) voxels->n_coords * 4u, &table_size)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    uint64_t * keys = (uint64_t *) calloc(table_size, sizeof(uint64_t));
    int32_t * values = (int32_t *) malloc(table_size * sizeof(int32_t));
    if (keys == NULL || values == NULL) {
        free(keys);
        free(values);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < table_size; ++i) {
        values[i] = -1;
    }
    for (int64_t i = 0; i < voxels->n_coords; ++i) {
        const int32_t * c = voxels->coords_bxyz + (size_t) i * 4u;
        uint64_t key = pbr_coord_key(c[1], c[2], c[3]);
        if (key == 0) {
            key = 1;
        }
        size_t slot = (size_t) hash_u64(key) & (table_size - 1u);
        while (values[slot] >= 0 && keys[slot] != key) {
            slot = (slot + 1u) & (table_size - 1u);
        }
        keys[slot] = key;
        values[slot] = (int32_t) i;
    }

    float * colors = (float *) malloc((size_t) mesh->n_vertices * 3u * sizeof(float));
    if (colors == NULL) {
        free(keys);
        free(values);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    const int progress_steps = mesh->n_vertices >= 100000 ? 20 : 0;
    int progress_step = 1;
    int64_t chunk_start_us = ggml_time_us();
    int64_t misses = 0;
    for (int64_t vi = 0; vi < mesh->n_vertices; ++vi) {
        const float * v = mesh->vertices + (size_t) vi * 3u;
        float * dst = colors + (size_t) vi * 3u;
        if (!pbr_sample_color_trilinear(voxels, keys, values, table_size, v, dst)) {
            ++misses;
        }
        if (progress_steps > 0 &&
            vi + 1 >= (mesh->n_vertices * (int64_t) progress_step) / progress_steps) {
            char detail[128];
            snprintf(
                detail,
                sizeof(detail),
                "vertices=%lld/%lld misses=%lld",
                (long long) (vi + 1),
                (long long) mesh->n_vertices,
                (long long) misses);
            trellis_progress_steps(
                "OBJ vertex colors",
                progress_step,
                progress_steps,
                ggml_time_us() - chunk_start_us,
                detail);
            chunk_start_us = ggml_time_us();
            ++progress_step;
        }
    }

    free(mesh->vertex_colors);
    mesh->vertex_colors = colors;
    free(keys);
    free(values);
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pipeline_write_obj(const char * path, const trellis_mesh_host * mesh) {
    if (path == NULL || mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!mkdir_parent(path)) {
        TRELLIS_ERROR("pipeline mesh: failed to create output directory for %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    FILE * f = fopen(path, "w");
    if (f == NULL) {
        TRELLIS_ERROR("pipeline mesh: failed to open %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    fprintf(f, "# trellis2.c mesh\n");
    for (int64_t i = 0; i < mesh->n_vertices; ++i) {
        const float * v = mesh->vertices + (size_t) i * 3u;
        const float * c = mesh->vertex_colors != NULL ? mesh->vertex_colors + (size_t) i * 3u : NULL;
        /* Match TRELLIS.2 GLB export convention: internal (x,y,z) -> viewer (x,z,-y). */
        int rc = c != NULL ?
            fprintf(f, "v %.9g %.9g %.9g %.9g %.9g %.9g\n", v[0], v[2], -v[1], c[0], c[1], c[2]) :
            fprintf(f, "v %.9g %.9g %.9g\n", v[0], v[2], -v[1]);
        if (rc < 0) {
            fclose(f);
            return TRELLIS_STATUS_IO_ERROR;
        }
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * face = mesh->faces + (size_t) i * 3u;
        if (fprintf(f, "f %d %d %d\n", face[0] + 1, face[1] + 1, face[2] + 1) < 0) {
            fclose(f);
            return TRELLIS_STATUS_IO_ERROR;
        }
    }
    return fclose(f) == 0 ? TRELLIS_STATUS_OK : TRELLIS_STATUS_IO_ERROR;
}

static int postprocess_mesh_reserve_vertices(trellis_mesh_host * mesh, int64_t need, int64_t * vertex_capacity) {
    if (need <= *vertex_capacity) {
        return 1;
    }
    int64_t cap = *vertex_capacity > 0 ? *vertex_capacity : 1024;
    while (cap < need) {
        if (cap > INT64_MAX / 2) return 0;
        cap *= 2;
    }
    float * vertices = (float *) realloc(mesh->vertices, (size_t) cap * 3u * sizeof(float));
    if (vertices == NULL) return 0;
    mesh->vertices = vertices;
    *vertex_capacity = cap;
    return 1;
}

static int postprocess_mesh_reserve_faces(trellis_mesh_host * mesh, int64_t need, int64_t * face_capacity) {
    if (need <= *face_capacity) {
        return 1;
    }
    int64_t cap = *face_capacity > 0 ? *face_capacity : 1024;
    while (cap < need) {
        if (cap > INT64_MAX / 2) return 0;
        cap *= 2;
    }
    int32_t * faces = (int32_t *) realloc(mesh->faces, (size_t) cap * 3u * sizeof(int32_t));
    if (faces == NULL) return 0;
    mesh->faces = faces;
    *face_capacity = cap;
    return 1;
}

static int postprocess_mesh_add_vertex(
    trellis_mesh_host * mesh,
    int64_t * vertex_capacity,
    float obj_x,
    float obj_y,
    float obj_z) {
    if (!postprocess_mesh_reserve_vertices(mesh, mesh->n_vertices + 1, vertex_capacity)) {
        return 0;
    }
    float * v = mesh->vertices + (size_t) mesh->n_vertices * 3u;
    v[0] = obj_x;
    v[1] = -obj_z;
    v[2] = obj_y;
    ++mesh->n_vertices;
    return 1;
}

static int postprocess_mesh_add_face(trellis_mesh_host * mesh, int64_t * face_capacity, int32_t a, int32_t b, int32_t c) {
    if (a < 0 || b < 0 || c < 0 ||
        a >= mesh->n_vertices || b >= mesh->n_vertices || c >= mesh->n_vertices) {
        return 0;
    }
    if (!postprocess_mesh_reserve_faces(mesh, mesh->n_faces + 1, face_capacity)) {
        return 0;
    }
    int32_t * f = mesh->faces + (size_t) mesh->n_faces * 3u;
    f[0] = a;
    f[1] = b;
    f[2] = c;
    ++mesh->n_faces;
    return 1;
}

static int postprocess_parse_obj_index(const char * token, int64_t n_vertices, int32_t * out) {
    char * end = NULL;
    long idx = strtol(token, &end, 10);
    if (end == token || idx == 0) {
        return 0;
    }
    int64_t resolved = idx > 0 ? (int64_t) idx - 1 : n_vertices + (int64_t) idx;
    if (resolved < 0 || resolved >= n_vertices || resolved > INT32_MAX) {
        return 0;
    }
    *out = (int32_t) resolved;
    return 1;
}

static trellis_status load_postprocessed_obj_mesh(const char * path, trellis_mesh_host * mesh_out) {
    if (path == NULL || mesh_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(mesh_out, 0, sizeof(*mesh_out));
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        TRELLIS_ERROR("mesh postprocess: failed to open processed OBJ %s", path);
        return TRELLIS_STATUS_IO_ERROR;
    }

    trellis_mesh_host mesh;
    memset(&mesh, 0, sizeof(mesh));
    int64_t vertex_capacity = 0;
    int64_t face_capacity = 0;
    char line[8192];
    int64_t line_no = 0;
    trellis_status status = TRELLIS_STATUS_OK;
    while (fgets(line, sizeof(line), f) != NULL) {
        ++line_no;
        char * p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (p[0] == 'v' && isspace((unsigned char) p[1])) {
            float x = 0.0f, y = 0.0f, z = 0.0f;
            if (sscanf(p + 1, "%f %f %f", &x, &y, &z) != 3 ||
                !postprocess_mesh_add_vertex(&mesh, &vertex_capacity, x, y, z)) {
                TRELLIS_ERROR("mesh postprocess: bad vertex at %s:%lld", path, (long long) line_no);
                status = TRELLIS_STATUS_IO_ERROR;
                break;
            }
        } else if (p[0] == 'f' && isspace((unsigned char) p[1])) {
            int32_t ids[256];
            int n = 0;
            char * q = p + 1;
            while (*q != '\0') {
                while (*q != '\0' && isspace((unsigned char) *q)) ++q;
                if (*q == '\0' || *q == '#') break;
                if (n >= (int) (sizeof(ids) / sizeof(ids[0])) ||
                    !postprocess_parse_obj_index(q, mesh.n_vertices, &ids[n])) {
                    TRELLIS_ERROR("mesh postprocess: bad face index at %s:%lld", path, (long long) line_no);
                    status = TRELLIS_STATUS_IO_ERROR;
                    break;
                }
                ++n;
                while (*q != '\0' && !isspace((unsigned char) *q)) ++q;
            }
            if (status != TRELLIS_STATUS_OK) break;
            if (n < 3) {
                TRELLIS_ERROR("mesh postprocess: face has fewer than 3 vertices at %s:%lld", path, (long long) line_no);
                status = TRELLIS_STATUS_IO_ERROR;
                break;
            }
            for (int i = 1; i + 1 < n; ++i) {
                if (!postprocess_mesh_add_face(&mesh, &face_capacity, ids[0], ids[i], ids[i + 1])) {
                    TRELLIS_ERROR("mesh postprocess: failed to append face at %s:%lld", path, (long long) line_no);
                    status = TRELLIS_STATUS_IO_ERROR;
                    break;
                }
            }
            if (status != TRELLIS_STATUS_OK) break;
        }
    }
    fclose(f);
    if (status == TRELLIS_STATUS_OK && (mesh.n_vertices <= 0 || mesh.n_faces <= 0)) {
        TRELLIS_ERROR("mesh postprocess: processed OBJ has no usable triangle mesh: %s", path);
        status = TRELLIS_STATUS_IO_ERROR;
    }
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_free(&mesh);
        return status;
    }
    *mesh_out = mesh;
    return TRELLIS_STATUS_OK;
}

static int run_vkmesh_postprocess_command(
    const char * vkmesh_path,
    const char * input_obj,
    const char * output_obj,
    int decimation_target,
    int no_simplify) {
    char target[64];
    snprintf(target, sizeof(target), "%d", decimation_target > 0 ? decimation_target : 1000000);
    const char * exe = vkmesh_path != NULL && vkmesh_path[0] != '\0' ? vkmesh_path : "vkmesh";
    pid_t pid = fork();
    if (pid < 0) {
        return 0;
    }
    if (pid == 0) {
        char * argv[12];
        int argc = 0;
        argv[argc++] = (char *) exe;
        argv[argc++] = (char *) "--input";
        argv[argc++] = (char *) input_obj;
        argv[argc++] = (char *) "--output";
        argv[argc++] = (char *) output_obj;
        argv[argc++] = (char *) "--postprocess";
        argv[argc++] = (char *) "--decimation-target";
        argv[argc++] = target;
        if (no_simplify) {
            argv[argc++] = (char *) "--no-simplify";
        }
        argv[argc++] = (char *) "--no-uv-unwrap";
        argv[argc] = NULL;
        if (strchr(exe, '/') != NULL) {
            execv(exe, argv);
        } else {
            execvp(exe, argv);
        }
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return 0;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

static trellis_status trellis_pipeline_postprocess_mesh_with_vkmesh(
    trellis_mesh_host * mesh,
    const char * vkmesh_path,
    int decimation_target,
    int no_simplify) {
    if (mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    char input_obj[4096];
    char output_obj[4096];
    int n0 = snprintf(input_obj, sizeof(input_obj), "/tmp/trellis2_vkmesh_in_%ld.obj", (long) getpid());
    int n1 = snprintf(output_obj, sizeof(output_obj), "/tmp/trellis2_vkmesh_out_%ld.obj", (long) getpid());
    if (n0 < 0 || (size_t) n0 >= sizeof(input_obj) || n1 < 0 || (size_t) n1 >= sizeof(output_obj)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = trellis_pipeline_write_obj(input_obj, mesh);
    if (status != TRELLIS_STATUS_OK) {
        unlink_if_set(input_obj);
        unlink_if_set(output_obj);
        return status;
    }

    TRELLIS_INFO(
        "mesh postprocess: running vkmesh target=%d no_simplify=%d input_faces=%lld",
        decimation_target > 0 ? decimation_target : 1000000,
        no_simplify ? 1 : 0,
        (long long) mesh->n_faces);
    if (!run_vkmesh_postprocess_command(vkmesh_path, input_obj, output_obj, decimation_target, no_simplify)) {
        TRELLIS_ERROR("mesh postprocess: vkmesh failed; pass --vkmesh /path/to/vkmesh if it is not in PATH");
        unlink_if_set(input_obj);
        unlink_if_set(output_obj);
        return TRELLIS_STATUS_ERROR;
    }

    trellis_mesh_host processed;
    status = load_postprocessed_obj_mesh(output_obj, &processed);
    unlink_if_set(input_obj);
    unlink_if_set(output_obj);
    if (status != TRELLIS_STATUS_OK) {
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
}

trellis_status trellis_pipeline_image_to_obj(const trellis_image_to_obj_options * options) {
    if (options == NULL || options->model_dir == NULL || options->dino_dir == NULL ||
        options->image_path == NULL ||
        ((options->obj_path == NULL || options->obj_path[0] == '\0') &&
         (options->gltf_path == NULL || options->gltf_path[0] == '\0'))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

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
        int n = snprintf(
            temp_image,
            sizeof(temp_image),
            "/tmp/trellis2_image_to_obj_%ld.png",
            (long) getpid());
        if (n < 0 || (size_t) n >= sizeof(temp_image)) {
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
        int n = snprintf(
            temp_birefnet_image,
            sizeof(temp_birefnet_image),
            "/tmp/trellis2_birefnet_%ld.png",
            (long) getpid());
        if (n < 0 || (size_t) n >= sizeof(temp_birefnet_image)) {
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
    }
    TRELLIS_INFO("SparseUnet backend: %s device=%d", sparse_backend_kind_name(sparse_backend_kind), sparse_device);

    trellis_sparse_structure_result sparse_structure_result;
    trellis_structured_latent shape_latent;
    trellis_structured_latent texture_latent;
    trellis_sparse_c2s_guides shape_subs;
    trellis_pbr_voxels pbr_voxels;
    trellis_mesh_host mesh;
    memset(&sparse_structure_result, 0, sizeof(sparse_structure_result));
    memset(&shape_latent, 0, sizeof(shape_latent));
    memset(&texture_latent, 0, sizeof(texture_latent));
    memset(&shape_subs, 0, sizeof(shape_subs));
    memset(&pbr_voxels, 0, sizeof(pbr_voxels));
    memset(&mesh, 0, sizeof(mesh));
    const int want_obj = options->obj_path != NULL && options->obj_path[0] != '\0';
    const int want_gltf = options->gltf_path != NULL && options->gltf_path[0] != '\0';

    TRELLIS_INFO("[1/5] SparseStructureFlowModel image -> sparse structure");
    trellis_sparse_structure_options sparse_structure;
    memset(&sparse_structure, 0, sizeof(sparse_structure));
    sparse_structure.model_dir = options->model_dir;
    sparse_structure.dino_dir = options->dino_dir;
    sparse_structure.image_path = sparse_structure_image_path;
    sparse_structure.latent_size = options->latent_size > 0 ? options->latent_size : 16;
    sparse_structure.steps = options->sparse_structure_steps > 0 ? options->sparse_structure_steps : 12;
    sparse_structure.cond_resolution = options->cond_resolution > 0 ? options->cond_resolution : 512;
    sparse_structure.sparse_resolution = options->sparse_resolution > 0 ? options->sparse_resolution : 32;
    sparse_structure.seed = options->seed == 0 ? 1u : options->seed;
    sparse_structure.flow_blocks_override = options->flow_blocks_override;
    sparse_structure.flow_block_parts_override = options->flow_block_parts_override;
    sparse_structure.flow_no_rope = options->flow_no_rope;
    sparse_structure.voxel_threshold = 0.0f;
    sparse_structure.backend = &graph_backend;
    sparse_structure.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;

    status = trellis_pipeline_run_sparse_structure(&sparse_structure, &sparse_structure_result);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    if (sparse_structure_result.n_coords <= 0 ||
        sparse_structure_result.coords_bxyz == NULL ||
        sparse_structure_result.cond == NULL) {
        TRELLIS_ERROR("sparse structure: produced no coords for structured latent flow");
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }

    TRELLIS_INFO(
        "[2/5] SLatFlowModel image -> shape SLat tokens=%lld cond_tokens=%d",
        (long long) sparse_structure_result.n_coords,
        sparse_structure_result.cond_tokens);
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
    structured_latent.resolution = options->resolution > 0 ? options->resolution : 512;
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
    structured_latent.use_ggml_flash_attn = options->use_ggml_flash_attn;
    structured_latent.backend = &graph_backend;
    structured_latent.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;

    status = trellis_pipeline_run_structured_latent(&structured_latent, &shape_latent);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }

    TRELLIS_INFO("[3/5] FlexiDualGridVaeDecoder shape SLat -> mesh/subdivision guides");
    trellis_pipeline_mesh_options mesh_options;
    memset(&mesh_options, 0, sizeof(mesh_options));
    mesh_options.model_dir = options->model_dir;
    mesh_options.decoder_override_path = options->decoder_override_path;
    mesh_options.latent = &shape_latent;
    mesh_options.resolution = structured_latent.resolution;
    mesh_options.decode_max_levels = options->decode_max_levels;
    mesh_options.decode_max_input_tokens = options->decode_max_input_tokens;
    mesh_options.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    mesh_options.sparse_backend_kind = sparse_backend_kind;
    mesh_options.sparse_device = sparse_device;

    status = trellis_pipeline_decode_shape_latent_mesh(&mesh_options, &shape_subs, &mesh);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    if (options->mesh_postprocess) {
        status = trellis_pipeline_postprocess_mesh_with_vkmesh(
            &mesh,
            options->vkmesh_path,
            options->mesh_postprocess_decimation_target > 0 ? options->mesh_postprocess_decimation_target : 1000000,
            options->mesh_postprocess_no_simplify);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
    }

    TRELLIS_INFO(
        "[4/5] SLatFlowModel image+shape -> texture SLat tokens=%lld cond_tokens=%d",
        (long long) sparse_structure_result.n_coords,
        sparse_structure_result.cond_tokens);
    trellis_structured_latent_options texture_options;
    memset(&texture_options, 0, sizeof(texture_options));
    texture_options.model_dir = options->model_dir;
    texture_options.flow_component = TRELLIS_COMPONENT_TEX_SLAT_FLOW;
    texture_options.label = "texture";
    texture_options.normalization_key = "tex_slat_normalization";
    texture_options.coords_bxyz = sparse_structure_result.coords_bxyz;
    texture_options.n_coords = sparse_structure_result.n_coords;
    texture_options.cond = sparse_structure_result.cond;
    texture_options.cond_tokens = sparse_structure_result.cond_tokens;
    texture_options.concat_cond = shape_latent.feats;
    texture_options.concat_channels = shape_latent.channels;
    texture_options.noise_seed = options->noise_seed == 0 ? 19u : options->noise_seed + 1u;
    texture_options.resolution = structured_latent.resolution;
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
    texture_options.use_ggml_flash_attn = options->use_ggml_flash_attn;
    texture_options.backend = &graph_backend;
    texture_options.cuda = sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;

    status = trellis_pipeline_run_structured_latent(&texture_options, &texture_latent);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
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

    status = trellis_pipeline_decode_texture_latent_voxels(&tex_decode, &pbr_voxels);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    TRELLIS_INFO(
        "pipeline: decoded PBR voxels=%lld channels=%d resolution=%d",
        (long long) pbr_voxels.n_coords,
        pbr_voxels.channels,
        pbr_voxels.resolution);

    if (want_obj) {
        TRELLIS_INFO(
            "pipeline: applying PBR voxels to OBJ vertex colors vertices=%lld voxels=%lld",
            (long long) mesh.n_vertices,
            (long long) pbr_voxels.n_coords);
        status = trellis_pipeline_apply_pbr_voxels_to_mesh(&pbr_voxels, &mesh);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        status = trellis_pipeline_write_obj(options->obj_path, &mesh);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
        TRELLIS_INFO(
            "pipeline: wrote %s (%lld vertices, %lld faces)",
            options->obj_path,
            (long long) mesh.n_vertices,
            (long long) mesh.n_faces);
    }
    if (want_gltf) {
        const int texture_size = options->texture_size > 0 ? options->texture_size : 1024;
        status = trellis_pipeline_write_gltf(options->gltf_path, &mesh, &pbr_voxels, texture_size);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR("pipeline: glTF export failed: %s", trellis_status_string(status));
            goto cleanup;
        }
    }

cleanup:
    trellis_mesh_free(&mesh);
    trellis_pbr_voxels_free(&pbr_voxels);
    trellis_sparse_c2s_guides_free(&shape_subs);
    trellis_structured_latent_free(&texture_latent);
    trellis_structured_latent_free(&shape_latent);
    trellis_sparse_structure_result_free(&sparse_structure_result);
    if (sparse_backend_kind == TRELLIS_SPARSE_BACKEND_CUDA) {
        trellis_cuda_free(&cuda);
    }
    trellis_backend_free(&graph_backend);
    unlink_if_set(temp_birefnet_image);
    unlink_if_set(temp_image);
    return status;
}
