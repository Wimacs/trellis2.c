#include "trellis.h"
#include "trellis_pipeline_internal.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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
        options->latent->channels != 32 || options->cuda == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = TRELLIS_STATUS_ERROR;
    char decoder_path[4096];
    trellis_tensor_store decoder_store;
    trellis_sparse_unet_vae_decoder_weights decoder;
    memset(&decoder_store, 0, sizeof(decoder_store));
    memset(&decoder, 0, sizeof(decoder));
    int32_t * out_coords = NULL;
    float * out_feats = NULL;

    if (!make_shape_decoder_path(options->model_dir, options->decoder_override_path, decoder_path, sizeof(decoder_path))) {
        TRELLIS_ERROR("pipeline mesh: failed to build shape decoder path");
        goto cleanup;
    }
    if (!load_shape_decoder(options->cuda, decoder_path, &decoder_store, &decoder)) {
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
    status = trellis_sparse_unet_vae_decoder_forward_f32_host(
        &decoder,
        options->latent->coords_bxyz,
        options->latent->feats,
        n_use,
        options->cuda->device,
        options->decode_max_levels,
        NULL,
        subs_out,
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
        options->guide_subs->n_levels <= 0 || options->cuda == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = TRELLIS_STATUS_ERROR;
    char decoder_path[4096];
    trellis_tensor_store decoder_store;
    trellis_sparse_unet_vae_decoder_weights decoder;
    memset(&decoder_store, 0, sizeof(decoder_store));
    memset(&decoder, 0, sizeof(decoder));

    if (!make_texture_decoder_path(options->model_dir, decoder_path, sizeof(decoder_path))) {
        TRELLIS_ERROR("texture decode: failed to build texture decoder path");
        goto cleanup;
    }
    if (!load_texture_decoder(options->cuda, decoder_path, &decoder_store, &decoder)) {
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
    status = trellis_sparse_unet_vae_decoder_forward_f32_host(
        &decoder,
        options->latent->coords_bxyz,
        options->latent->feats,
        n_use,
        options->cuda->device,
        options->decode_max_levels,
        options->guide_subs,
        NULL,
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

    const int resolution = voxels->resolution > 0 ? voxels->resolution : 512;
    for (int64_t vi = 0; vi < mesh->n_vertices; ++vi) {
        const float * v = mesh->vertices + (size_t) vi * 3u;
        int32_t gx = (int32_t) floorf((v[0] + 0.5f) * (float) resolution + 0.5f);
        int32_t gy = (int32_t) floorf((v[1] + 0.5f) * (float) resolution + 0.5f);
        int32_t gz = (int32_t) floorf((v[2] + 0.5f) * (float) resolution + 0.5f);
        if (gx < 0) gx = 0;
        if (gy < 0) gy = 0;
        if (gz < 0) gz = 0;
        if (gx >= resolution) gx = resolution - 1;
        if (gy >= resolution) gy = resolution - 1;
        if (gz >= resolution) gz = resolution - 1;

        int32_t best = -1;
        for (int radius = 0; best < 0 && radius <= 3; ++radius) {
            for (int dz = -radius; best < 0 && dz <= radius; ++dz) {
                for (int dy = -radius; best < 0 && dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        if (radius > 0 &&
                            abs(dx) != radius && abs(dy) != radius && abs(dz) != radius) {
                            continue;
                        }
                        int32_t x = gx + dx;
                        int32_t y = gy + dy;
                        int32_t z = gz + dz;
                        if (x < 0 || y < 0 || z < 0 ||
                            x >= resolution || y >= resolution || z >= resolution) {
                            continue;
                        }
                        uint64_t key = pbr_coord_key(x, y, z);
                        if (key == 0) {
                            key = 1;
                        }
                        size_t slot = (size_t) hash_u64(key) & (table_size - 1u);
                        while (values[slot] >= 0) {
                            if (keys[slot] == key) {
                                best = values[slot];
                                break;
                            }
                            slot = (slot + 1u) & (table_size - 1u);
                        }
                        if (best >= 0) {
                            break;
                        }
                    }
                }
            }
        }
        float * dst = colors + (size_t) vi * 3u;
        if (best >= 0) {
            const float * src = voxels->attrs + (size_t) best * (size_t) voxels->channels;
            dst[0] = clamp01(src[0]);
            dst[1] = clamp01(src[1]);
            dst[2] = clamp01(src[2]);
        } else {
            dst[0] = 0.8f;
            dst[1] = 0.8f;
            dst[2] = 0.8f;
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

trellis_status trellis_pipeline_image_to_obj(const trellis_image_to_obj_options * options) {
    if (options == NULL || options->model_dir == NULL || options->dino_dir == NULL ||
        options->image_path == NULL || options->obj_path == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    char temp_image[4096];
    temp_image[0] = '\0';
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
            unlink(temp_image);
            return TRELLIS_STATUS_IO_ERROR;
        }
        sparse_structure_image_path = temp_image;
    }

    trellis_cuda_context cuda;
    memset(&cuda, 0, sizeof(cuda));
    trellis_status status = trellis_cuda_init(&cuda, options->device);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("cuda: init failed: %s", trellis_status_string(status));
        if (temp_image[0] != '\0') {
            unlink(temp_image);
        }
        return status;
    }

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
    sparse_structure.cuda = &cuda;

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
    structured_latent.cuda = &cuda;

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
    mesh_options.cuda = &cuda;

    status = trellis_pipeline_decode_shape_latent_mesh(&mesh_options, &shape_subs, &mesh);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
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
    texture_options.cuda = &cuda;

    status = trellis_pipeline_run_structured_latent(&texture_options, &texture_latent);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    trellis_sparse_structure_result_free(&sparse_structure_result);

    TRELLIS_INFO("[5/5] SparseUnetVaeDecoder texture SLat -> PBR vertex colors");
    trellis_pipeline_texture_options tex_decode;
    memset(&tex_decode, 0, sizeof(tex_decode));
    tex_decode.model_dir = options->model_dir;
    tex_decode.latent = &texture_latent;
    tex_decode.guide_subs = &shape_subs;
    tex_decode.decode_max_levels = options->decode_max_levels;
    tex_decode.decode_max_input_tokens = options->decode_max_input_tokens;
    tex_decode.cuda = &cuda;

    status = trellis_pipeline_decode_texture_latent_voxels(&tex_decode, &pbr_voxels);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
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

cleanup:
    trellis_mesh_free(&mesh);
    trellis_pbr_voxels_free(&pbr_voxels);
    trellis_sparse_c2s_guides_free(&shape_subs);
    trellis_structured_latent_free(&texture_latent);
    trellis_structured_latent_free(&shape_latent);
    trellis_sparse_structure_result_free(&sparse_structure_result);
    trellis_cuda_free(&cuda);
    if (temp_image[0] != '\0') {
        unlink(temp_image);
    }
    return status;
}
