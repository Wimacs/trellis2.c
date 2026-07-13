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

static trellis_status trellis_pipeline_image_to_gltf_impl(
    const trellis_image_to_gltf_options * options,
    const trellis_pixal3d_options * pixal_options,
    const trellis_image_to_gltf_feature_options * feature_options,
    const char * required_family);

trellis_status trellis_pipeline_image_to_gltf(const trellis_image_to_gltf_options * options) {
    trellis_pixal3d_options pixal_options = TRELLIS_PIXAL3D_OPTIONS_INIT;
    return trellis_pipeline_image_to_gltf_impl(options, &pixal_options, NULL, NULL);
}

trellis_status trellis_pipeline_image_to_gltf_ex(
    const trellis_image_to_gltf_options * options,
    const trellis_pixal3d_options * pixal_options) {
    return trellis_pipeline_image_to_gltf_impl(options, pixal_options, NULL, NULL);
}

trellis_status trellis_pipeline_trellis2_image_to_gltf(
    const trellis_image_to_gltf_options * options) {
    trellis_pixal3d_options pixal_options = TRELLIS_PIXAL3D_OPTIONS_INIT;
    return trellis_pipeline_image_to_gltf_impl(
        options,
        &pixal_options,
        NULL,
        "trellis2");
}

trellis_status trellis_pipeline_trellis2_image_to_gltf_ex(
    const trellis_image_to_gltf_options * options,
    const trellis_image_to_gltf_feature_options * feature_options) {
    trellis_pixal3d_options pixal_options = TRELLIS_PIXAL3D_OPTIONS_INIT;
    return trellis_pipeline_image_to_gltf_impl(
        options,
        &pixal_options,
        feature_options,
        "trellis2");
}

trellis_status trellis_pipeline_pixal3d_image_to_gltf(
    const trellis_image_to_gltf_options * options,
    const trellis_pixal3d_options * pixal_options) {
    return trellis_pipeline_image_to_gltf_impl(
        options,
        pixal_options,
        NULL,
        "pixal3d");
}

trellis_status trellis_pipeline_pixal3d_image_to_gltf_ex(
    const trellis_image_to_gltf_options * options,
    const trellis_pixal3d_options * pixal_options,
    const trellis_image_to_gltf_feature_options * feature_options) {
    return trellis_pipeline_image_to_gltf_impl(
        options,
        pixal_options,
        feature_options,
        "pixal3d");
}

static trellis_status trellis_pipeline_image_to_gltf_impl(
    const trellis_image_to_gltf_options * options,
    const trellis_pixal3d_options * pixal_options,
    const trellis_image_to_gltf_feature_options * feature_options,
    const char * required_family) {
    if ((feature_options != NULL &&
         (feature_options->struct_size < TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_V1_SIZE ||
          feature_options->version != TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_VERSION)) ||
        pixal_options == NULL ||
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
    const int shape_only =
        feature_options != NULL && feature_options->shape_only != 0;
    const char * prepared_image_output_path = feature_options != NULL ?
        feature_options->prepared_image_output_path : NULL;
    const char * shape_latent_output_path = feature_options != NULL ?
        feature_options->shape_latent_output_path : NULL;
    if (shape_latent_output_path != NULL &&
        shape_latent_output_path[0] != '\0' &&
        required_family != NULL && strcmp(required_family, "trellis2") != 0) {
        TRELLIS_ERROR(
            "pipeline: reusable shape SLat caches are only supported by the trellis2 adapter");
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

    trellis_prepared_condition_image prepared_image;
    memset(&prepared_image, 0, sizeof(prepared_image));
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
    if (required_family != NULL &&
        (model_package.family == NULL ||
         strcmp(model_package.family, required_family) != 0)) {
        TRELLIS_ERROR(
            "model package '%s' has family '%s', but this entry point requires family '%s'",
            model_package.id != NULL ? model_package.id : "unknown",
            model_package.family != NULL ? model_package.family : "unknown",
            required_family);
        trellis_model_package_free(&model_package);
        return TRELLIS_STATUS_PARSE_ERROR;
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
    if (shape_latent_output_path != NULL &&
        shape_latent_output_path[0] != '\0' &&
        strcmp(task_adapter->family, "trellis2") != 0) {
        TRELLIS_ERROR(
            "pipeline: reusable shape SLat caches are only supported by the trellis2 adapter");
        trellis_model_package_free(&model_package);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
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
            "pipeline: Pixal3D with an unlimited model cache keeps reused weights, including NAF, resident until cleanup; use --no-model-cache or --model-cache-budget-mib N on memory-constrained devices");
    }

    status = trellis_pipeline_prepare_condition_image_for_image_to_gltf(
        options->model_dir,
        options->dino_dir,
        options->image_path,
        options->birefnet_path,
        graph_backend_kind,
        options->device,
        task_adapter->requires_transparent_foreground,
        &prepared_image);
    if (status != TRELLIS_STATUS_OK) {
        trellis_model_package_free(&model_package);
        return status;
    }
    const char * sparse_structure_image_path = prepared_image.source_path;
    if (prepared_image_output_path != NULL &&
        prepared_image_output_path[0] != '\0') {
        status = trellis_pipeline_write_prepared_condition_image_png(
            &prepared_image,
            prepared_image_output_path);
        if (status != TRELLIS_STATUS_OK) {
            trellis_pipeline_prepared_condition_image_free(&prepared_image);
            trellis_model_package_free(&model_package);
            return status;
        }
        sparse_structure_image_path = prepared_image_output_path;
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
        trellis_pipeline_prepared_condition_image_free(&prepared_image);
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
        trellis_pipeline_prepared_condition_image_free(&prepared_image);
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
            trellis_pipeline_prepared_condition_image_free(&prepared_image);
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
            trellis_pipeline_prepared_condition_image_free(&prepared_image);
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
    const int pipeline_stage_count = shape_only ? 4 : 5;

    const char * shape_flow_lr_role =
        use_1024_cascade || final_resolution < 1024 ?
            "shape_flow_512" : "shape_flow_1024";
    const char * texture_flow_role =
        final_resolution >= 1024 ? "texture_flow_1024" : "texture_flow_512";
    sparse_flow_component = trellis_model_package_find_component(
        &model_package, "sparse_structure_flow");
    shape_flow_lr_component = trellis_model_package_find_component(
        &model_package, shape_flow_lr_role);
    if (!shape_only) {
        texture_flow_component = trellis_model_package_find_component(
            &model_package, texture_flow_role);
    }
    if (use_1024_cascade) {
        shape_flow_hr_component = trellis_model_package_find_component(
            &model_package, "shape_flow_1024");
    }
    if (sparse_flow_component == NULL || shape_flow_lr_component == NULL ||
        (use_1024_cascade && shape_flow_hr_component == NULL) ||
        !pipeline_component_path(&model_package, "sparse_structure_flow", sparse_flow_path) ||
        !pipeline_component_path(&model_package, "sparse_structure_decoder", sparse_decoder_path) ||
        !pipeline_component_path(&model_package, shape_flow_lr_role, shape_flow_lr_path) ||
        (use_1024_cascade &&
         !pipeline_component_path(&model_package, "shape_flow_1024", shape_flow_hr_path)) ||
        !pipeline_component_path(&model_package, "shape_decoder", shape_decoder_path) ||
        (!shape_only &&
         (texture_flow_component == NULL ||
          !pipeline_component_path(&model_package, texture_flow_role, texture_flow_path) ||
          !pipeline_component_path(&model_package, "texture_decoder", texture_decoder_path)))) {
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
    if (shape_only) {
        TRELLIS_INFO(
            "attention policy: sparse=%s shape_lr=%s shape_hr=%s texture=disabled (shape-only)%s",
            pipeline_attention_policy_name(&sparse_flow_attention),
            pipeline_attention_policy_name(&shape_flow_lr_attention),
            pipeline_attention_policy_name(&shape_flow_hr_attention),
            projected_conditioning && !options->use_ggml_flash_attn ?
                " (Pixal3D strict BF16 Flash package defaults)" : "");
    } else {
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
    }

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

    TRELLIS_INFO(
        "[1/%d] SparseStructureFlowModel image -> sparse structure",
        pipeline_stage_count);
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
        TRELLIS_INFO(
            "[2a/%d] Pixal3D DINOv3 + NAF -> shape-512 projected condition",
            pipeline_stage_count);
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
        "[2/%d] SLatFlowModel image -> shape SLat tokens=%lld cond_tokens=%d resolution=%d",
        pipeline_stage_count,
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

        TRELLIS_INFO(
            "[2b/%d] FlexiDualGridVaeDecoder 512 shape SLat -> cascade HR coords",
            pipeline_stage_count);
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
                "[2c/%d] Pixal3D DINOv3 + NAF -> shape-1024 projected condition" :
                "[2c/%d] DINOv3 image encoder -> 1024 condition",
            pipeline_stage_count);
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
            "[2c/%d] SLatFlowModel 1024 image -> shape SLat tokens=%lld cond_tokens=%d resolution=%d",
            pipeline_stage_count,
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

    TRELLIS_INFO(
        shape_only ?
            "[3/%d] FlexiDualGridVaeDecoder shape SLat -> mesh" :
            "[3/%d] FlexiDualGridVaeDecoder shape SLat -> mesh/subdivision guides",
        pipeline_stage_count);
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
    status = trellis_pipeline_decode_shape_latent_mesh(
        &mesh_options,
        shape_only ? NULL : &shape_subs,
        &mesh);
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
        if (shape_only) {
            TRELLIS_INFO(
                "SparseUnet Vulkan decoder backend: released %.1f MiB of inactive shape buffers",
                (double) released_bytes / (1024.0 * 1024.0));
        } else {
            TRELLIS_INFO(
                "SparseUnet Vulkan decoder backend: released %.1f MiB of inactive shape buffers; subdivision guides retained",
                (double) released_bytes / (1024.0 * 1024.0));
        }
    }
    if (material_dump_dir != NULL && material_dump_dir[0] != '\0') {
        status = trellis_pipeline_dump_raw_mesh_if_requested(
            material_dump_dir,
            &mesh);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
    }
    /* The anchor belongs to the shape SLat/decoder coordinate frame.  Save it
     * before vkmesh can introduce a small scale or center drift; a later
     * retopologized mesh is mapped back to this frame on cache reuse. */
    if (shape_latent_output_path != NULL &&
        shape_latent_output_path[0] != '\0') {
        status = trellis_shape_latent_cache_write(
            shape_latent_output_path,
            &shape_latent,
            &mesh,
            NULL);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "pipeline: shape latent cache export failed: %s",
                trellis_status_string(status));
            goto cleanup;
        }
    }
    if (options->mesh_postprocess) {
        status = trellis_pipeline_postprocess_mesh_with_vkmesh(
            &mesh,
            shape_only ? NULL : &gltf_projection_mesh,
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

    if (shape_only) {
        TRELLIS_INFO("[4/4] Shape-only mesh -> glTF");
        status = trellis_pipeline_write_shape_gltf_ex(
            output_gltf_path,
            &mesh,
            task_adapter->gltf_coordinate_transform);
        if (status != TRELLIS_STATUS_OK) {
            TRELLIS_ERROR(
                "pipeline: shape-only glTF export failed: %s",
                trellis_status_string(status));
        }
        goto cleanup;
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
    trellis_pipeline_prepared_condition_image_free(&prepared_image);
    trellis_model_package_free(&model_package);
    return status;
}
