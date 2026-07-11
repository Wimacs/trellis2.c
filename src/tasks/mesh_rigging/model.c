#include "mesh_rigging_internal.h"

#include <string.h>

static trellis_status load_component_store(
    const trellis_model_package * package,
    const char * role,
    const char * label,
    const trellis_backend_context * backend,
    trellis_tensor_store * store) {
    char path[4096];
    trellis_status status = trellis_model_package_resolve_component_path(
        package, role, path, sizeof(path));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("TokenSkin: cannot resolve component '%s'", role);
        return status;
    }
    trellis_model_load_result result;
    memset(&result, 0, sizeof(result));
    if (!trellis_load_tensor_store(
            backend,
            label,
            path,
            true,
            64,
            store,
            &result)) {
        TRELLIS_ERROR("TokenSkin: failed to load %s from %s", label, path);
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    TRELLIS_INFO(
        "TokenSkin: loaded %s tensors=%zu bytes=%llu seconds=%.3f",
        label,
        result.tensors,
        (unsigned long long) result.bytes,
        result.seconds);
    return TRELLIS_STATUS_OK;
}

void tokenskin_runtime_model_free(tokenskin_runtime_model * model) {
    if (model == NULL) return;
    trellis_tensor_store_free(&model->skin_store);
    trellis_tensor_store_free(&model->qwen_store);
    trellis_tensor_store_free(&model->mesh_store);
    trellis_backend_free(&model->backend);
    memset(model, 0, sizeof(*model));
}

trellis_status tokenskin_runtime_model_load(
    const trellis_model_package * package,
    trellis_backend_kind backend_kind,
    int device,
    int disable_flash,
    tokenskin_runtime_model * model) {
    if (package == NULL || model == NULL || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(model, 0, sizeof(*model));
    trellis_status status = trellis_backend_init(&model->backend, backend_kind, device);
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR(
            "TokenSkin: cannot initialize %s backend device %d",
            trellis_backend_kind_name(backend_kind),
            device);
        return status;
    }
    model->attention_policy = (trellis_ggml_attention_policy) TRELLIS_GGML_ATTENTION_POLICY_INIT;
    model->attention_policy.mode = disable_flash ?
        TRELLIS_GGML_ATTENTION_MODE_EXPLICIT :
        TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16;

    status = load_component_store(
        package, "mesh_encoder", "TokenSkin mesh encoder", &model->backend, &model->mesh_store);
    if (status != TRELLIS_STATUS_OK) goto fail;
    char issue[512] = {0};
    status = tokenskin_bind_mesh_encoder_weights(
        &model->mesh_store, &model->mesh, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("TokenSkin mesh encoder contract: %s", issue);
        goto fail;
    }

    status = load_component_store(
        package, "rig_language_model", "TokenSkin Qwen3", &model->backend, &model->qwen_store);
    if (status != TRELLIS_STATUS_OK) goto fail;
    issue[0] = '\0';
    status = tokenskin_bind_qwen_weights(
        &model->qwen_store, &model->qwen, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("TokenSkin Qwen3 contract: %s", issue);
        goto fail;
    }

    status = load_component_store(
        package, "skin_decoder", "TokenSkin FSQ-CVAE", &model->backend, &model->skin_store);
    if (status != TRELLIS_STATUS_OK) goto fail;
    issue[0] = '\0';
    status = tokenskin_bind_fsq_cvae_weights(
        &model->skin_store, &model->skin, issue, sizeof(issue));
    if (status != TRELLIS_STATUS_OK) {
        TRELLIS_ERROR("TokenSkin FSQ-CVAE contract: %s", issue);
        goto fail;
    }
    return TRELLIS_STATUS_OK;

fail:
    tokenskin_runtime_model_free(model);
    return status;
}
