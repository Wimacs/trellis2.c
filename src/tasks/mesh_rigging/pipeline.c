#include "mesh_rigging_internal.h"

#include "generation.h"
#include "gltf_io.h"
#include "preprocess.h"
#include "rigged_gltf.h"
#include "tokenizer.h"
#include "tokenskin_features.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void perf_stage(const char * name, int64_t started_us) {
    TRELLIS_INFO(
        "perf_stage name=tokenskin_%s ms=%.3f",
        name,
        (double) (ggml_time_us() - started_us) / 1000.0);
}

static int component_matches(
    const trellis_model_package * package,
    const char * role,
    const char * architecture) {
    const trellis_model_component_instance * component =
        trellis_model_package_find_component(package, role);
    return component != NULL && component->architecture != NULL &&
        strcmp(component->architecture, architecture) == 0 &&
        component->execution.compute_dtype == TRELLIS_DTYPE_BF16 &&
        component->execution.attention == TRELLIS_ATTENTION_FLASH &&
        component->execution.flash_kv_dtype == TRELLIS_DTYPE_BF16 &&
        component->execution.emulate_bf16_blocks == 0;
}

static trellis_status validate_tokenskin_package(
    const trellis_model_package * package) {
    if (package == NULL || package->family == NULL || package->task == NULL ||
        strcmp(package->family, "tokenskin") != 0 ||
        strcmp(package->task, "mesh_rigging") != 0) {
        TRELLIS_ERROR(
            "tokenskin-rig requires family=tokenskin task=mesh_rigging; got family=%s task=%s",
            package != NULL && package->family != NULL ? package->family : "(null)",
            package != NULL && package->task != NULL ? package->task : "(null)");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!component_matches(
            package, "mesh_encoder", "tokenskin_michelangelo_encoder") ||
        !component_matches(
            package, "rig_language_model", "tokenskin_qwen3") ||
        !component_matches(
            package, "skin_decoder", "tokenskin_fsq_cvae")) {
        TRELLIS_ERROR(
            "TokenSkin package components must match TokenRig architectures and "
            "declare compute_dtype=bf16 attention=flash flash_kv_dtype=bf16");
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status validate_options(
    const trellis_tokenskin_rig_options * options) {
    if (options == NULL ||
        options->struct_size < TRELLIS_TOKENSKIN_RIG_OPTIONS_V1_SIZE ||
        options->model_dir == NULL || options->model_dir[0] == '\0' ||
        options->input_path == NULL || options->input_path[0] == '\0' ||
        options->output_path == NULL || options->output_path[0] == '\0' ||
        options->device < 0 || options->sample_count < 2048 ||
        options->max_length <= TOKENSKIN_MESH_TOKENS + 2 ||
        options->max_length > TOKENSKIN_QWEN_MAX_POSITIONS ||
        options->top_k < 0 || !isfinite(options->top_p) ||
        options->top_p <= 0.0f || options->top_p > 1.0f ||
        !isfinite(options->temperature) || options->temperature < 0.0f ||
        !isfinite(options->repetition_penalty) ||
        options->repetition_penalty <= 0.0f || options->num_beams <= 0 ||
        options->num_beams > TRELLIS_TOKENSKIN_MAX_BEAMS) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status decode_sampled_skin(
    const tokenskin_runtime_model * model,
    const trellis_tokenskin_rig_options * options,
    const int32_t * sequence,
    size_t sequence_count,
    const float * condition_384x512,
    const float * query_features,
    size_t query_count,
    trellis_tokenskin_skeleton * skeleton_out,
    float ** sampled_skin_out) {
    if (query_count > INT64_MAX) return TRELLIS_STATUS_INVALID_ARGUMENT;
    trellis_status status = trellis_tokenskin_tokenizer_decode_skeleton(
        sequence, sequence_count, skeleton_out);
    if (status != TRELLIS_STATUS_OK) return status;
    const size_t joints = skeleton_out->joint_count;
    if (joints == 0 || joints > SIZE_MAX / TRELLIS_TOKENSKIN_TOKENS_PER_SKIN ||
        query_count > SIZE_MAX / joints ||
        query_count * joints > SIZE_MAX / sizeof(float)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    const size_t index_count = joints * TRELLIS_TOKENSKIN_TOKENS_PER_SKIN;
    int32_t * indices = (int32_t *) malloc(index_count * sizeof(int32_t));
    float * sampled_skin = (float *) malloc(query_count * joints * sizeof(float));
    float * probabilities = (float *) malloc(query_count * sizeof(float));
    if (indices == NULL || sampled_skin == NULL || probabilities == NULL) {
        free(probabilities);
        free(sampled_skin);
        free(indices);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    const trellis_tokenskin_eos_mode eos_mode = options->official_eos_compat ?
        TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT :
        TRELLIS_TOKENSKIN_EOS_CORRECTED;
    size_t extracted = 0;
    status = trellis_tokenskin_tokenizer_extract_skin_indices(
        sequence,
        sequence_count,
        eos_mode,
        indices,
        index_count,
        &extracted);
    if (status != TRELLIS_STATUS_OK || extracted != index_count) {
        status = status == TRELLIS_STATUS_OK ? TRELLIS_STATUS_PARSE_ERROR : status;
        goto cleanup;
    }
    for (size_t joint = 0; joint < joints; ++joint) {
        float codes[TRELLIS_TOKENSKIN_TOKENS_PER_SKIN *
            TRELLIS_TOKENSKIN_FSQ_CODE_DIM];
        for (size_t token = 0; token < TRELLIS_TOKENSKIN_TOKENS_PER_SKIN; ++token) {
            status = trellis_tokenskin_tokenizer_fsq_code(
                indices[joint * TRELLIS_TOKENSKIN_TOKENS_PER_SKIN + token],
                eos_mode,
                codes + token * TRELLIS_TOKENSKIN_FSQ_CODE_DIM);
            if (status != TRELLIS_STATUS_OK) goto cleanup;
        }
        const int64_t started_us = ggml_time_us();
        status = tokenskin_fsq_skin_decode_compute(
            &model->backend,
            &model->skin,
            &model->attention_policy,
            codes,
            condition_384x512,
            query_features,
            (int64_t) query_count,
            probabilities);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        for (size_t point = 0; point < query_count; ++point) {
            sampled_skin[point * joints + joint] = probabilities[point];
        }
        TRELLIS_INFO(
            "TokenSkin skin decode: joint=%zu/%zu ms=%.3f",
            joint + 1u,
            joints,
            (double) (ggml_time_us() - started_us) / 1000.0);
    }
    *sampled_skin_out = sampled_skin;
    sampled_skin = NULL;
    status = TRELLIS_STATUS_OK;

cleanup:
    free(probabilities);
    free(sampled_skin);
    free(indices);
    return status;
}

trellis_status trellis_pipeline_tokenskin_rig(
    const trellis_tokenskin_rig_options * options) {
    trellis_status status = validate_options(options);
    if (status != TRELLIS_STATUS_OK) return status;

    trellis_model_package package = TRELLIS_MODEL_PACKAGE_INIT;
    trellis_mesh_rigging_asset asset = TRELLIS_MESH_RIGGING_ASSET_INIT;
    trellis_mesh_rigging_preprocessed preprocessed =
        TRELLIS_MESH_RIGGING_PREPROCESSED_INIT;
    tokenskin_runtime_model model;
    memset(&model, 0, sizeof(model));
    trellis_tokenskin_skeleton skeleton;
    memset(&skeleton, 0, sizeof(skeleton));
    float * mesh_features = NULL;
    float * mesh_queries = NULL;
    float * vae_features = NULL;
    float * vae_queries = NULL;
    float * mesh_embeddings = NULL;
    float * condition = NULL;
    float * sampled_skin = NULL;
    int32_t * sequence = NULL;
    size_t sequence_count = 0;
    char error[1024] = {0};

    /* Family/task preflight deliberately precedes input parsing and backend
     * initialization so per-model executables cannot accidentally cross-load. */
    status = trellis_model_package_load(options->model_dir, &package);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    status = validate_tokenskin_package(&package);
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    int64_t started_us = ggml_time_us();
    status = trellis_mesh_rigging_gltf_load(
        options->input_path, &asset, error, sizeof(error));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("TokenSkin input mesh: %s", error);
        goto cleanup;
    }
    perf_stage("mesh_load", started_us);
    TRELLIS_INFO(
        "TokenSkin input: vertices=%zu triangles=%zu primitives=%zu",
        asset.vertex_count,
        asset.triangle_count,
        asset.primitive_count);

    trellis_mesh_rigging_preprocess_options preprocess_options =
        TRELLIS_MESH_RIGGING_PREPROCESS_OPTIONS_INIT;
    preprocess_options.surface_sample_count = (size_t) options->sample_count;
    preprocess_options.seed = options->seed;
    started_us = ggml_time_us();
    status = trellis_mesh_rigging_preprocess(
        &asset,
        &preprocess_options,
        &preprocessed,
        error,
        sizeof(error));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("TokenSkin preprocessing: %s", error);
        goto cleanup;
    }
    perf_stage("preprocess", started_us);

    status = trellis_tokenskin_build_features(
        preprocessed.sample_positions,
        preprocessed.sample_normals,
        preprocessed.sample_count,
        TRELLIS_TOKENSKIN_FEATURE_MICHELANGELO,
        &mesh_features);
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_tokenskin_gather_features(
            mesh_features,
            preprocessed.sample_count,
            preprocessed.mesh_fps_indices,
            preprocessed.mesh_fps_count,
            &mesh_queries);
    }
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    trellis_backend_kind backend_kind;
    status = trellis_backend_kind_from_name(
        options->backend != NULL && options->backend[0] != '\0' ?
            options->backend : TRELLIS_DEFAULT_BACKEND,
        &backend_kind);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    started_us = ggml_time_us();
    status = tokenskin_runtime_model_load(
        &package,
        backend_kind,
        options->device,
        options->no_flash_attn,
        &model);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    perf_stage("model_load", started_us);

    mesh_embeddings = (float *) malloc(
        (size_t) TOKENSKIN_MESH_TOKENS * TOKENSKIN_QWEN_HIDDEN * sizeof(float));
    condition = (float *) malloc(
        (size_t) TOKENSKIN_VAE_COND_TOKENS *
        TOKENSKIN_VAE_LATENT_CHANNELS * sizeof(float));
    if (mesh_embeddings == NULL || condition == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    started_us = ggml_time_us();
    status = tokenskin_mesh_encoder_compute(
        &model.backend,
        &model.mesh,
        &model.attention_policy,
        mesh_features,
        (int64_t) preprocessed.sample_count,
        mesh_queries,
        mesh_embeddings);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    perf_stage("mesh_encode", started_us);
    free(mesh_queries);
    mesh_queries = NULL;
    free(mesh_features);
    mesh_features = NULL;

    status = trellis_tokenskin_build_features(
        preprocessed.sample_positions,
        preprocessed.sample_normals,
        preprocessed.sample_count,
        TRELLIS_TOKENSKIN_FEATURE_FSQ_CVAE,
        &vae_features);
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_tokenskin_gather_features(
            vae_features,
            preprocessed.sample_count,
            preprocessed.skin_fps_indices,
            preprocessed.skin_fps_count,
            &vae_queries);
    }
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    started_us = ggml_time_us();
    status = tokenskin_fsq_condition_compute(
        &model.backend,
        &model.skin,
        &model.attention_policy,
        vae_features,
        (int64_t) preprocessed.sample_count,
        vae_queries,
        condition);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    perf_stage("skin_condition_encode", started_us);
    free(vae_queries);
    vae_queries = NULL;

    started_us = ggml_time_us();
    status = tokenskin_generate_rig_tokens(
        &model,
        mesh_embeddings,
        options,
        &sequence,
        &sequence_count);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    perf_stage("qwen_generate", started_us);
    free(mesh_embeddings);
    mesh_embeddings = NULL;

    started_us = ggml_time_us();
    status = decode_sampled_skin(
        &model,
        options,
        sequence,
        sequence_count,
        condition,
        vae_features,
        preprocessed.sample_count,
        &skeleton,
        &sampled_skin);
    if (status != TRELLIS_STATUS_OK) goto cleanup;
    perf_stage("skin_decode", started_us);
    TRELLIS_INFO("TokenSkin skeleton: joints=%zu", skeleton.joint_count);

    trellis_mesh_rigging_dense_skin_weights dense_weights =
        TRELLIS_MESH_RIGGING_DENSE_SKIN_WEIGHTS_INIT;
    dense_weights.values = sampled_skin;
    dense_weights.sample_positions = preprocessed.sample_positions;
    dense_weights.point_count = preprocessed.sample_count;
    dense_weights.joint_count = skeleton.joint_count;
    dense_weights.interpolation_neighbors = 8;
    started_us = ggml_time_us();
    status = trellis_mesh_rigging_write_rigged_glb(
        options->output_path,
        &asset,
        &preprocessed.normalization,
        &skeleton,
        &dense_weights,
        error,
        sizeof(error));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("TokenSkin rigged GLB export: %s", error);
        goto cleanup;
    }
    if (error[0] != '\0') {
        TRELLIS_WARN("TokenSkin rigged GLB export: %s", error);
    } else {
        TRELLIS_INFO(
            "TokenSkin rig export preserved the available source appearance "
            "and texture coordinates");
    }
    perf_stage("export", started_us);
    TRELLIS_INFO("TokenSkin rigged GLB: %s", options->output_path);

cleanup:
    free(sequence);
    free(sampled_skin);
    free(condition);
    free(mesh_embeddings);
    free(vae_queries);
    free(vae_features);
    free(mesh_queries);
    free(mesh_features);
    trellis_tokenskin_skeleton_free(&skeleton);
    tokenskin_runtime_model_free(&model);
    trellis_mesh_rigging_preprocessed_free(&preprocessed);
    trellis_mesh_rigging_asset_free(&asset);
    trellis_model_package_free(&package);
    return status;
}
