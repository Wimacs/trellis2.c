#define _POSIX_C_SOURCE 200809L

#include "trellis.h"
#include "trellis_model_package.h"
#include "trellis_sparse_backend.h"
#include "image_to_3d_internal.h"
#include "adapter.h"
#include "gltf_io.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char * texturing_sparse_backend_name(trellis_sparse_backend_kind kind) {
    switch (kind) {
        case TRELLIS_SPARSE_BACKEND_CUDA: return "cuda";
        case TRELLIS_SPARSE_BACKEND_VULKAN: return "vulkan";
        case TRELLIS_SPARSE_BACKEND_CPU: return "cpu";
        default: return "unknown";
    }
}

static trellis_status texturing_sparse_backend_kind(
    trellis_backend_kind graph_kind,
    trellis_sparse_backend_kind * sparse_kind_out) {
    if (sparse_kind_out == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;
    if (graph_kind == TRELLIS_BACKEND_CUDA) {
        *sparse_kind_out = TRELLIS_SPARSE_BACKEND_CUDA;
        return TRELLIS_STATUS_OK;
    }
    if (graph_kind == TRELLIS_BACKEND_VULKAN) {
        *sparse_kind_out = TRELLIS_SPARSE_BACKEND_VULKAN;
        return TRELLIS_STATUS_OK;
    }
    return TRELLIS_STATUS_INVALID_ARGUMENT;
}

static int texturing_copy_path(
    char * output,
    size_t output_size,
    const char * path) {
    if (output == NULL || output_size == 0 || path == NULL || path[0] == '\0') {
        return 0;
    }
    const int n = snprintf(output, output_size, "%s", path);
    return n >= 0 && (size_t) n < output_size;
}

static int texturing_component_path(
    const trellis_model_package * package,
    const char * role,
    const char * override_path,
    char output[4096]) {
    if (override_path != NULL && override_path[0] != '\0') {
        return texturing_copy_path(output, 4096, override_path);
    }
    const trellis_status status = trellis_model_package_resolve_component_path(
        package,
        role,
        output,
        4096);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh texturing: component role '%s' is unavailable: %s",
            role,
            trellis_status_string(status));
        return 0;
    }
    return 1;
}

static trellis_status texturing_load_and_normalize_mesh(
    const char * path,
    trellis_mesh_host * mesh_out) {
    if (path == NULL || path[0] == '\0' || mesh_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(mesh_out, 0, sizeof(*mesh_out));
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    char error[512];
    error[0] = '\0';
    trellis_status status = trellis_mesh_rigging_gltf_load(
        path,
        &asset,
        error,
        sizeof(error));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh texturing: failed to load mesh: %s%s%s",
            trellis_status_string(status),
            error[0] == '\0' ? "" : " ",
            error);
        return status;
    }
    if (asset.vertex_count > (size_t) INT32_MAX ||
        asset.vertex_count > (size_t) INT64_MAX ||
        asset.triangle_count > (size_t) INT64_MAX) {
        status = TRELLIS_STATUS_NOT_IMPLEMENTED;
        goto cleanup;
    }
    for (size_t face = 0; face < asset.triangle_count * 3u; ++face) {
        if (asset.triangles[face] > (uint32_t) INT32_MAX) {
            status = TRELLIS_STATUS_NOT_IMPLEMENTED;
            goto cleanup;
        }
    }

    double center[3];
    double max_extent = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        center[axis] =
            ((double) asset.aabb_min[axis] + (double) asset.aabb_max[axis]) * 0.5;
        const double extent =
            (double) asset.aabb_max[axis] - (double) asset.aabb_min[axis];
        if (extent > max_extent) max_extent = extent;
    }
    if (!(max_extent > 0.0) || !isfinite(max_extent)) {
        status = TRELLIS_STATUS_PARSE_ERROR;
        goto cleanup;
    }
    const double scale = 0.99999 / max_extent;
    for (size_t vertex = 0; vertex < asset.vertex_count; ++vertex) {
        float * position = asset.positions + vertex * 3u;
        const double x = ((double) position[0] - center[0]) * scale;
        const double y = ((double) position[1] - center[1]) * scale;
        const double z = ((double) position[2] - center[2]) * scale;
        /* TRELLIS.2 texturing preprocessing: (x,y,z) -> (x,-z,y). */
        position[0] = (float) x;
        position[1] = (float) -z;
        position[2] = (float) y;
        if (!isfinite(position[0]) || !isfinite(position[1]) ||
            !isfinite(position[2]) || position[0] < -0.50001f ||
            position[0] > 0.50001f || position[1] < -0.50001f ||
            position[1] > 0.50001f || position[2] < -0.50001f ||
            position[2] > 0.50001f) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            goto cleanup;
        }
    }

    mesh_out->vertices = asset.positions;
    mesh_out->faces = (int32_t *) asset.triangles;
    mesh_out->n_vertices = (int64_t) asset.vertex_count;
    mesh_out->n_faces = (int64_t) asset.triangle_count;
    asset.positions = NULL;
    asset.triangles = NULL;
    TRELLIS_INFO(
        "mesh texturing: normalized input vertices=%lld triangles=%lld scale=%.9g",
        (long long) mesh_out->n_vertices,
        (long long) mesh_out->n_faces,
        scale);

cleanup:
    trellis_mesh_rigging_asset_free(&asset);
    if (status != TRELLIS_STATUS_OK) trellis_mesh_free(mesh_out);
    return status;
}

static trellis_status texturing_make_encoder_features(
    const trellis_flexible_dual_grid * grid,
    const trellis_flexible_dual_grid_options * grid_options,
    float ** features_out) {
    if (grid == NULL || grid_options == NULL || features_out == NULL ||
        grid->n <= 0 || grid->coords == NULL || grid->dual_vertices == NULL ||
        grid->intersected == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *features_out = NULL;
    if ((uint64_t) grid->n > SIZE_MAX / (6u * sizeof(float))) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    float voxel_size[3];
    for (int axis = 0; axis < 3; ++axis) {
        voxel_size[axis] =
            (grid_options->aabb_max[axis] - grid_options->aabb_min[axis]) /
            (float) grid_options->grid_size[axis];
        if (!(voxel_size[axis] > 0.0f) || !isfinite(voxel_size[axis])) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }
    float * features = (float *) malloc((size_t) grid->n * 6u * sizeof(float));
    if (features == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    for (int64_t row = 0; row < grid->n; ++row) {
        for (int axis = 0; axis < 3; ++axis) {
            const float local =
                grid->dual_vertices[(size_t) row * 3u + (size_t) axis] /
                    voxel_size[axis] -
                (float) grid->coords[(size_t) row * 4u + 1u + (size_t) axis];
            features[(size_t) row * 6u + (size_t) axis] = local - 0.5f;
            features[(size_t) row * 6u + 3u + (size_t) axis] =
                grid->intersected[(size_t) row * 3u + (size_t) axis] ? 0.5f : -0.5f;
            if (!isfinite(local) || local < -1e-4f || local > 1.0001f) {
                free(features);
                return TRELLIS_STATUS_ERROR;
            }
        }
    }
    *features_out = features;
    return TRELLIS_STATUS_OK;
}

static trellis_status texturing_load_encoder(
    const trellis_backend_context * weight_backend,
    const char * path,
    trellis_tensor_store * store,
    trellis_sparse_unet_vae_encoder_weights * encoder) {
    if (!trellis_load_tensor_store_f32(
            weight_backend,
            "FlexiDualGridVaeEncoder",
            path,
            true,
            64,
            store,
            NULL)) {
        return TRELLIS_STATUS_ERROR;
    }
    char issue[512];
    issue[0] = '\0';
    const trellis_status status = trellis_sparse_unet_vae_encoder_bind_weights(
        store,
        encoder,
        issue,
        sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "FlexiDualGridVaeEncoder: bind failed: %s%s%s",
            trellis_status_string(status),
            issue[0] == '\0' ? "" : " ",
            issue);
    }
    return status;
}

trellis_status trellis_pipeline_trellis2_texture_mesh(
    const trellis_mesh_texturing_options * options) {
    if (options == NULL ||
        options->struct_size < TRELLIS_MESH_TEXTURING_OPTIONS_V1_SIZE ||
        options->model_dir == NULL || options->dino_dir == NULL ||
        options->input_path == NULL || options->image_path == NULL ||
        options->output_path == NULL || options->device < 0 ||
        (options->resolution != 512 && options->resolution != 1024) ||
        options->texture_size < 64 || options->texture_size > 8192 ||
        options->steps <= 0 || options->steps > 1000 ||
        options->flow_blocks_override < -1 ||
        options->flow_block_parts_override < -1 ||
        options->flow_block_parts_override > 3) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_status status = TRELLIS_STATUS_ERROR;
    trellis_model_package package = TRELLIS_MODEL_PACKAGE_INIT;
    trellis_backend_context graph_backend;
    trellis_cuda_context cuda;
    trellis_backend_context cpu_weight_backend;
    trellis_sparse_backend * sparse_backend = NULL;
    trellis_prepared_condition_image prepared_image;
    trellis_mesh_host mesh;
    trellis_flexible_dual_grid fdg;
    trellis_tensor_store encoder_store;
    trellis_sparse_unet_vae_encoder_weights encoder;
    trellis_sparse_c2s_guides guides;
    trellis_structured_latent shape_latent;
    trellis_image_condition_result condition;
    trellis_structured_latent texture_latent;
    trellis_pbr_voxels pbr_voxels;
    float * encoder_input = NULL;
    int graph_backend_initialized = 0;
    int cuda_initialized = 0;
    int cpu_weight_backend_initialized = 0;
    int encoder_store_initialized = 0;
    memset(&graph_backend, 0, sizeof(graph_backend));
    memset(&cuda, 0, sizeof(cuda));
    memset(&cpu_weight_backend, 0, sizeof(cpu_weight_backend));
    memset(&prepared_image, 0, sizeof(prepared_image));
    memset(&mesh, 0, sizeof(mesh));
    memset(&fdg, 0, sizeof(fdg));
    memset(&encoder_store, 0, sizeof(encoder_store));
    memset(&encoder, 0, sizeof(encoder));
    memset(&guides, 0, sizeof(guides));
    memset(&shape_latent, 0, sizeof(shape_latent));
    memset(&condition, 0, sizeof(condition));
    memset(&texture_latent, 0, sizeof(texture_latent));
    memset(&pbr_voxels, 0, sizeof(pbr_voxels));

    const char * backend_name =
        options->backend != NULL && options->backend[0] != '\0' ?
            options->backend : TRELLIS_DEFAULT_BACKEND;
    trellis_backend_kind graph_kind = TRELLIS_BACKEND_CUDA;
    status = trellis_backend_kind_from_name(backend_name, &graph_kind);
    if (status != TRELLIS_STATUS_OK ||
        strcmp(trellis_backend_kind_name(graph_kind), TRELLIS_DEFAULT_BACKEND) != 0) {
        TRELLIS_ERROR(
            "mesh texturing: this binary was compiled for %s, requested %s",
            TRELLIS_DEFAULT_BACKEND,
            backend_name);
        status = TRELLIS_STATUS_INVALID_ARGUMENT;
        goto cleanup;
    }
    trellis_sparse_backend_kind sparse_kind = TRELLIS_SPARSE_BACKEND_CUDA;
    status = texturing_sparse_backend_kind(graph_kind, &sparse_kind);
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    /* Family rejection deliberately precedes image/mesh parsing and GPU init. */
    status = trellis_model_package_load(options->model_dir, &package);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh texturing: failed to load model package '%s': %s",
            options->model_dir,
            trellis_status_string(status));
        goto cleanup;
    }
    if (package.family == NULL || strcmp(package.family, "trellis2") != 0) {
        TRELLIS_ERROR(
            "mesh texturing: model package family '%s' is not trellis2",
            package.family != NULL ? package.family : "unknown");
        status = TRELLIS_STATUS_PARSE_ERROR;
        goto cleanup;
    }

    const char * texture_flow_role =
        options->resolution == 1024 ? "texture_flow_1024" : "texture_flow_512";
    const trellis_model_component_instance * encoder_component =
        trellis_model_package_find_component(&package, "shape_encoder");
    const trellis_model_component_instance * texture_flow_component =
        trellis_model_package_find_component(&package, texture_flow_role);
    const trellis_model_component_instance * texture_decoder_component =
        trellis_model_package_find_component(&package, "texture_decoder");
    if (encoder_component == NULL || texture_flow_component == NULL ||
        texture_decoder_component == NULL ||
        strcmp(encoder_component->architecture, "sparse_unet_vae_encoder") != 0 ||
        strcmp(texture_flow_component->architecture, "trellis_dit_flow") != 0 ||
        strcmp(texture_decoder_component->architecture, "sparse_unet_vae_decoder") != 0) {
        TRELLIS_ERROR("mesh texturing: incompatible Trellis2 component contract");
        status = TRELLIS_STATUS_PARSE_ERROR;
        goto cleanup;
    }

    char encoder_path[4096];
    char texture_flow_path[4096];
    char texture_decoder_path[4096];
    if (!texturing_component_path(
            &package, "shape_encoder", options->encoder_path, encoder_path) ||
        !texturing_component_path(
            &package, texture_flow_role, options->texture_flow_path, texture_flow_path) ||
        !texturing_component_path(
            &package, "texture_decoder", options->texture_decoder_path, texture_decoder_path)) {
        status = TRELLIS_STATUS_NOT_FOUND;
        goto cleanup;
    }

    status = trellis_pipeline_prepare_condition_image(
        options->model_dir,
        options->dino_dir,
        options->image_path,
        options->birefnet_path,
        graph_kind,
        options->device,
        1,
        &prepared_image);
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    status = texturing_load_and_normalize_mesh(options->input_path, &mesh);
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    trellis_flexible_dual_grid_options fdg_options;
    trellis_flexible_dual_grid_options_default(&fdg_options);
    for (int axis = 0; axis < 3; ++axis) {
        fdg_options.grid_size[axis] = options->resolution;
    }
    int64_t stage_start_us = ggml_time_us();
    status = trellis_mesh_to_flexible_dual_grid_host(
        mesh.vertices,
        mesh.n_vertices,
        mesh.faces,
        mesh.n_faces,
        &fdg_options,
        &fdg);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh texturing: mesh -> Flexible Dual Grid failed: %s",
            trellis_status_string(status));
        goto cleanup;
    }
    TRELLIS_INFO(
        "perf_stage name=mesh_to_flexible_dual_grid ms=%.3f tokens=%lld",
        (double) (ggml_time_us() - stage_start_us) / 1000.0,
        (long long) fdg.n);
    status = texturing_make_encoder_features(&fdg, &fdg_options, &encoder_input);
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    status = trellis_backend_init(&graph_backend, graph_kind, options->device);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh texturing: graph backend init failed: %s",
            trellis_status_string(status));
        goto cleanup;
    }
    graph_backend_initialized = 1;
    if (sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA) {
        status = trellis_cuda_init(&cuda, options->device);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        cuda_initialized = 1;
    } else {
        status = trellis_sparse_vulkan_backend_create(options->device, &sparse_backend);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        status = trellis_backend_init(&cpu_weight_backend, TRELLIS_BACKEND_CPU, 0);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        cpu_weight_backend_initialized = 1;
    }
    TRELLIS_INFO(
        "mesh texturing: graph=%s sparse=%s device=%d",
        trellis_backend_kind_name(graph_kind),
        texturing_sparse_backend_name(sparse_kind),
        options->device);

    const trellis_backend_context * encoder_weight_backend =
        sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : &cpu_weight_backend;
    /* The loader may have populated part of the store before reporting an
     * error, so make cleanup own it before loading starts. */
    encoder_store_initialized = 1;
    status = texturing_load_encoder(
        encoder_weight_backend,
        encoder_path,
        &encoder_store,
        &encoder);
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    trellis_sparse_unet_vae_encoder_forward_options encoder_options;
    memset(&encoder_options, 0, sizeof(encoder_options));
    encoder_options.backend_kind = sparse_kind;
    encoder_options.device = options->device;
    encoder_options.sparse_backend = sparse_backend;
    encoder_options.return_subs = &guides;
    stage_start_us = ggml_time_us();
    status = trellis_sparse_unet_vae_encoder_forward_backend_f32_host(
        &encoder,
        fdg.coords,
        encoder_input,
        fdg.n,
        &encoder_options,
        &shape_latent.coords_bxyz,
        &shape_latent.feats,
        &shape_latent.n_coords,
        &shape_latent.channels);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "mesh texturing: shape encoder failed: %s",
            trellis_status_string(status));
        goto cleanup;
    }
    shape_latent.resolution = options->resolution;
    if (shape_latent.n_coords <= 0 || shape_latent.coords_bxyz == NULL ||
        shape_latent.feats == NULL || shape_latent.channels != 32 ||
        guides.n_levels != 4) {
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    const int64_t encoded_values =
        shape_latent.n_coords * (int64_t) shape_latent.channels;
    float encoded_min = shape_latent.feats[0];
    float encoded_max = shape_latent.feats[0];
    double encoded_sum = 0.0;
    double encoded_square_sum = 0.0;
    for (int64_t index = 0; index < encoded_values; ++index) {
        const float value = shape_latent.feats[index];
        if (!isfinite(value)) {
            TRELLIS_ERROR(
                "mesh texturing: shape encoder produced a non-finite value at %lld",
                (long long) index);
            status = TRELLIS_STATUS_ERROR;
            goto cleanup;
        }
        if (value < encoded_min) encoded_min = value;
        if (value > encoded_max) encoded_max = value;
        encoded_sum += (double) value;
        encoded_square_sum += (double) value * (double) value;
    }
    const double encoded_mean = encoded_sum / (double) encoded_values;
    double encoded_variance =
        encoded_square_sum / (double) encoded_values - encoded_mean * encoded_mean;
    if (encoded_variance < 0.0) encoded_variance = 0.0;
    TRELLIS_INFO(
        "perf_stage name=shape_slat_encode ms=%.3f tokens=%lld channels=%d guides=%d "
        "min=%.6g max=%.6g mean=%.6g std=%.6g",
        (double) (ggml_time_us() - stage_start_us) / 1000.0,
        (long long) shape_latent.n_coords,
        shape_latent.channels,
        guides.n_levels,
        encoded_min,
        encoded_max,
        encoded_mean,
        sqrt(encoded_variance));

    if (sparse_backend != NULL) {
        size_t released = 0;
        status = trellis_sparse_backend_trim(
            sparse_backend,
            TRELLIS_SPARSE_TRIM_ALL,
            &released);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        TRELLIS_INFO(
            "mesh texturing: released %.1f MiB of encoder sparse cache",
            (double) released / (1024.0 * 1024.0));
    }
    trellis_tensor_store_free(&encoder_store);
    encoder_store_initialized = 0;
    if (cpu_weight_backend_initialized) {
        trellis_backend_free(&cpu_weight_backend);
        cpu_weight_backend_initialized = 0;
    }
    free(encoder_input);
    encoder_input = NULL;
    trellis_flexible_dual_grid_free(&fdg);

    trellis_image_condition_options condition_options;
    memset(&condition_options, 0, sizeof(condition_options));
    condition_options.model_dir = options->model_dir;
    condition_options.dino_dir = options->dino_dir;
    condition_options.image_path = prepared_image.source_path;
    condition_options.cond_resolution = options->resolution;
    condition_options.foreground_alpha_threshold = 204;
    condition_options.foreground_crop_scale = 1.0f;
    condition_options.backend = &graph_backend;
    condition_options.cuda =
        sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    stage_start_us = ggml_time_us();
    status = trellis_pipeline_run_image_condition(&condition_options, &condition);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    TRELLIS_INFO(
        "perf_stage name=texture_image_condition ms=%.3f tokens=%d",
        (double) (ggml_time_us() - stage_start_us) / 1000.0,
        condition.cond_tokens);

    trellis_ggml_attention_policy texture_attention =
        TRELLIS_GGML_ATTENTION_POLICY_INIT;
    int package_emulates_bf16 = 0;
    status = trellis_image_to_3d_component_execution_policy(
        texture_flow_component,
        &texture_attention,
        &package_emulates_bf16);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    if (options->no_flash_attn) {
        texture_attention.mode = TRELLIS_GGML_ATTENTION_MODE_EXPLICIT;
    }

    trellis_structured_latent_options texture_options;
    memset(&texture_options, 0, sizeof(texture_options));
    texture_options.model_dir = options->model_dir;
    texture_options.flow_override_path = texture_flow_path;
    texture_options.flow_component = TRELLIS_COMPONENT_TEX_SLAT_FLOW;
    texture_options.label = "texture";
    texture_options.normalization_key = "tex_slat_normalization";
    texture_options.coords_bxyz = shape_latent.coords_bxyz;
    texture_options.n_coords = shape_latent.n_coords;
    texture_options.cond = condition.cond;
    texture_options.cond_tokens = condition.cond_tokens;
    texture_options.concat_cond = shape_latent.feats;
    texture_options.concat_channels = shape_latent.channels;
    texture_options.noise_seed = options->seed == 0 ? 1u : options->seed;
    texture_options.resolution = options->resolution;
    texture_options.steps = options->steps;
    texture_options.rescale_t = 3.0f;
    texture_options.guidance_strength = 1.0f;
    texture_options.guidance_rescale = 0.0f;
    texture_options.guidance_min = 0.6f;
    texture_options.guidance_max = 0.9f;
    texture_options.flow_blocks_override = options->flow_blocks_override;
    texture_options.flow_block_parts_override = options->flow_block_parts_override;
    texture_options.flow_no_rope = options->flow_no_rope;
    texture_options.emulate_bf16_blocks =
        options->emulate_bf16_blocks || package_emulates_bf16;
    texture_options.attention_policy = texture_attention;
    texture_options.use_ggml_flash_attn =
        texture_attention.mode != TRELLIS_GGML_ATTENTION_MODE_EXPLICIT;
    texture_options.backend = &graph_backend;
    texture_options.cuda =
        sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    stage_start_us = ggml_time_us();
    status = trellis_pipeline_run_structured_latent(
        &texture_options,
        &texture_latent);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    TRELLIS_INFO(
        "perf_stage name=texture_slat_denoise ms=%.3f tokens=%lld",
        (double) (ggml_time_us() - stage_start_us) / 1000.0,
        (long long) texture_latent.n_coords);
    trellis_image_condition_result_free(&condition);
    trellis_structured_latent_free(&shape_latent);

    trellis_pipeline_texture_options decoder_options;
    memset(&decoder_options, 0, sizeof(decoder_options));
    decoder_options.model_dir = options->model_dir;
    decoder_options.decoder_override_path = texture_decoder_path;
    decoder_options.latent = &texture_latent;
    decoder_options.guide_subs = &guides;
    decoder_options.cuda =
        sparse_kind == TRELLIS_SPARSE_BACKEND_CUDA ? &cuda : NULL;
    decoder_options.sparse_backend_kind = sparse_kind;
    decoder_options.sparse_device = options->device;
    decoder_options.sparse_backend = sparse_backend;
    stage_start_us = ggml_time_us();
    status = trellis_pipeline_decode_texture_latent_voxels(
        &decoder_options,
        &pbr_voxels);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    if (pbr_voxels.n_coords <= 0 || pbr_voxels.coords_bxyz == NULL ||
        pbr_voxels.attrs == NULL || pbr_voxels.channels != 6) {
        TRELLIS_ERROR(
            "mesh texturing: texture decoder returned an invalid PBR tensor "
            "tokens=%lld channels=%d",
            (long long) pbr_voxels.n_coords,
            pbr_voxels.channels);
        status = TRELLIS_STATUS_ERROR;
        goto cleanup;
    }
    TRELLIS_INFO(
        "perf_stage name=texture_slat_decode ms=%.3f voxels=%lld channels=%d",
        (double) (ggml_time_us() - stage_start_us) / 1000.0,
        (long long) pbr_voxels.n_coords,
        pbr_voxels.channels);
    trellis_structured_latent_free(&texture_latent);

    status = trellis_pipeline_write_gltf_ex(
        options->output_path,
        &mesh,
        NULL,
        &pbr_voxels,
        options->texture_size,
        options->device,
        TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_TRELLIS);
    if (status == TRELLIS_STATUS_OK) {
        TRELLIS_INFO("mesh texturing: wrote %s", options->output_path);
    }

cleanup:
    free(encoder_input);
    trellis_flexible_dual_grid_free(&fdg);
    if (encoder_store_initialized) trellis_tensor_store_free(&encoder_store);
    trellis_pbr_voxels_free(&pbr_voxels);
    trellis_structured_latent_free(&texture_latent);
    trellis_image_condition_result_free(&condition);
    trellis_structured_latent_free(&shape_latent);
    trellis_sparse_c2s_guides_free(&guides);
    trellis_mesh_free(&mesh);
    if (cpu_weight_backend_initialized) trellis_backend_free(&cpu_weight_backend);
    if (cuda_initialized) trellis_cuda_free(&cuda);
    trellis_sparse_backend_destroy(sparse_backend);
    if (graph_backend_initialized) trellis_backend_free(&graph_backend);
    trellis_pipeline_prepared_condition_image_free(&prepared_image);
    trellis_model_package_free(&package);
    return status;
}
