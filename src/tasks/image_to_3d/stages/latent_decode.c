#include "image_to_3d_internal.h"

#include <stdlib.h>
#include <string.h>

typedef struct trellis_transient_decoder_cache {
    trellis_pipeline_model_cache models;
    trellis_backend_context cpu_backend;
    int models_initialized;
    int cpu_backend_initialized;
} trellis_transient_decoder_cache;

static trellis_status transient_decoder_cache_acquire(
    trellis_pipeline_model_cache * provided,
    trellis_sparse_backend_kind sparse_backend_kind,
    const trellis_cuda_context * cuda,
    trellis_transient_decoder_cache * transient,
    trellis_pipeline_model_cache ** cache_out) {
    if (transient == NULL || cache_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(transient, 0, sizeof(*transient));
    if (provided != NULL) {
        *cache_out = provided;
        return TRELLIS_STATUS_OK;
    }
    const trellis_backend_context * weight_backend = cuda;
    if (sparse_backend_kind != TRELLIS_SPARSE_BACKEND_CUDA) {
        trellis_status status = trellis_backend_init(
            &transient->cpu_backend,
            TRELLIS_BACKEND_CPU,
            0);
        if (status != TRELLIS_STATUS_OK) {
            return status;
        }
        transient->cpu_backend_initialized = 1;
        weight_backend = &transient->cpu_backend;
    }
    trellis_status status = trellis_pipeline_model_cache_init(
        &transient->models,
        weight_backend,
        0,
        0);
    if (status != TRELLIS_STATUS_OK) {
        if (transient->cpu_backend_initialized) {
            trellis_backend_free(&transient->cpu_backend);
            transient->cpu_backend_initialized = 0;
        }
        return status;
    }
    transient->models_initialized = 1;
    *cache_out = &transient->models;
    return TRELLIS_STATUS_OK;
}

static void transient_decoder_cache_release(
    trellis_transient_decoder_cache * transient) {
    if (transient == NULL) return;
    if (transient->models_initialized) {
        trellis_pipeline_model_cache_free(&transient->models);
    }
    if (transient->cpu_backend_initialized) {
        trellis_backend_free(&transient->cpu_backend);
    }
    memset(transient, 0, sizeof(*transient));
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
    trellis_transient_decoder_cache transient_cache;
    trellis_pipeline_model_cache * cache = NULL;
    status = transient_decoder_cache_acquire(
        options->cache,
        options->sparse_backend_kind,
        options->cuda,
        &transient_cache,
        &cache);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "pipeline mesh: decoder cache init failed: %s",
            trellis_status_string(status));
        return status;
    }
    const trellis_sparse_unet_vae_decoder_weights * decoder_ptr = NULL;
    int32_t * out_coords = NULL;
    float * out_feats = NULL;

    status = trellis_pipeline_model_cache_get_shape_decoder(
        cache,
        options->model_dir,
        options->decoder_override_path,
        &decoder_ptr);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("pipeline mesh: shape decoder load failed: %s", trellis_status_string(status));
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
    transient_decoder_cache_release(&transient_cache);
    if (status != TRELLIS_STATUS_OK) {
        if (subs_out != NULL) {
            trellis_sparse_c2s_guides_free(subs_out);
        }
        trellis_mesh_free(mesh_out);
    }
    return status;
}
trellis_status trellis_pipeline_decode_shape_latent_decoder_coords(
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
    trellis_transient_decoder_cache transient_cache;
    trellis_pipeline_model_cache * cache = NULL;
    status = transient_decoder_cache_acquire(
        options->cache,
        options->sparse_backend_kind,
        options->cuda,
        &transient_cache,
        &cache);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "cascade upsample: decoder cache init failed: %s",
            trellis_status_string(status));
        return status;
    }
    const trellis_sparse_unet_vae_decoder_weights * decoder_ptr = NULL;
    int32_t * decoder_coords = NULL;
    float * decoder_feats = NULL;

    status = trellis_pipeline_model_cache_get_shape_decoder(
        cache,
        options->model_dir,
        options->decoder_override_path,
        &decoder_ptr);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("cascade upsample: shape decoder load failed: %s", trellis_status_string(status));
        goto cleanup;
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
    transient_decoder_cache_release(&transient_cache);
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
    trellis_transient_decoder_cache transient_cache;
    trellis_pipeline_model_cache * cache = NULL;
    status = transient_decoder_cache_acquire(
        options->cache,
        options->sparse_backend_kind,
        options->cuda,
        &transient_cache,
        &cache);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "texture decode: decoder cache init failed: %s",
            trellis_status_string(status));
        return status;
    }
    const trellis_sparse_unet_vae_decoder_weights * decoder_ptr = NULL;

    status = trellis_pipeline_model_cache_get_texture_decoder(
        cache,
        options->model_dir,
        options->decoder_override_path,
        &decoder_ptr);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("texture decode: texture decoder load failed: %s", trellis_status_string(status));
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
    transient_decoder_cache_release(&transient_cache);
    if (status != TRELLIS_STATUS_OK) {
        trellis_pbr_voxels_free(voxels_out);
    }
    return status;
}
