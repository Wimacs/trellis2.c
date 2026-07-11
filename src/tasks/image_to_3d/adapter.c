#include "adapter.h"

#include <math.h>
#include <string.h>

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))

typedef const trellis_image_to_3d_adapter_descriptor * (*adapter_accessor)(void);

static const adapter_accessor g_builtin_adapters[] = {
    trellis_image_to_3d_trellis2_adapter,
    trellis_image_to_3d_pixal3d_adapter,
};

static int descriptor_is_valid(
    const trellis_image_to_3d_adapter_descriptor * adapter) {
    if (adapter == NULL ||
        adapter->abi_version != TRELLIS_IMAGE_TO_3D_ADAPTER_ABI_VERSION ||
        adapter->struct_size < sizeof(*adapter) ||
        adapter->id == NULL || adapter->id[0] == '\0' ||
        adapter->family == NULL || adapter->family[0] == '\0' ||
        (adapter->uses_projected_conditioning != 0 &&
         adapter->uses_projected_conditioning != 1) ||
        (adapter->requires_transparent_foreground != 0 &&
         adapter->requires_transparent_foreground != 1) ||
        (adapter->cascade_quantization !=
             TRELLIS_IMAGE_TO_3D_CASCADE_QUANTIZATION_FLOOR &&
         adapter->cascade_quantization !=
             TRELLIS_IMAGE_TO_3D_CASCADE_QUANTIZATION_ROUND) ||
        (adapter->gltf_coordinate_transform !=
             TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_TRELLIS &&
         adapter->gltf_coordinate_transform !=
             TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_PIXAL3D) ||
        adapter->required_components == NULL ||
        adapter->required_component_count == 0) {
        return 0;
    }

    if (adapter->uses_projected_conditioning) {
        const float pi = 3.14159265358979323846f;
        if (!isfinite(adapter->default_camera_angle_x) ||
            !isfinite(adapter->default_camera_distance) ||
            !isfinite(adapter->default_mesh_scale) ||
            adapter->default_camera_angle_x <= 0.0f ||
            adapter->default_camera_angle_x >= pi ||
            adapter->default_camera_distance <= 0.0f ||
            adapter->default_mesh_scale <= 0.0f) {
            return 0;
        }
    }

    for (size_t i = 0; i < adapter->required_component_count; ++i) {
        const trellis_image_to_3d_component_requirement * requirement =
            &adapter->required_components[i];
        if (requirement->role == NULL || requirement->role[0] == '\0' ||
            requirement->architecture == NULL ||
            requirement->architecture[0] == '\0' ||
            requirement->attention < TRELLIS_ATTENTION_NONE ||
            requirement->attention > TRELLIS_ATTENTION_FLASH ||
            (requirement->emulate_bf16_blocks != 0 &&
             requirement->emulate_bf16_blocks != 1) ||
            (requirement->attention == TRELLIS_ATTENTION_FLASH &&
             requirement->flash_kv_dtype != TRELLIS_DTYPE_F16 &&
             requirement->flash_kv_dtype != TRELLIS_DTYPE_BF16) ||
            (requirement->attention != TRELLIS_ATTENTION_FLASH &&
             requirement->flash_kv_dtype != TRELLIS_DTYPE_UNKNOWN)) {
            return 0;
        }
        for (size_t j = 0; j < i; ++j) {
            if (strcmp(requirement->role,
                       adapter->required_components[j].role) == 0) {
                return 0;
            }
        }
    }
    return 1;
}

size_t trellis_image_to_3d_adapter_count(void) {
    return ARRAY_COUNT(g_builtin_adapters);
}

const trellis_image_to_3d_adapter_descriptor *
trellis_image_to_3d_adapter_at(size_t index) {
    if (index >= ARRAY_COUNT(g_builtin_adapters)) {
        return NULL;
    }
    const trellis_image_to_3d_adapter_descriptor * adapter =
        g_builtin_adapters[index]();
    return descriptor_is_valid(adapter) ? adapter : NULL;
}

const trellis_image_to_3d_adapter_descriptor *
trellis_image_to_3d_adapter_find(const char * family) {
    if (family == NULL || family[0] == '\0') {
        return NULL;
    }
    for (size_t i = 0; i < trellis_image_to_3d_adapter_count(); ++i) {
        const trellis_image_to_3d_adapter_descriptor * adapter =
            trellis_image_to_3d_adapter_at(i);
        if (adapter != NULL && strcmp(adapter->family, family) == 0) {
            return adapter;
        }
    }
    return NULL;
}

const trellis_image_to_3d_adapter_descriptor *
trellis_image_to_3d_adapter_find_for_package(
    const trellis_model_package * package) {
    if (package == NULL || package->struct_size < sizeof(*package)) {
        return NULL;
    }
    return trellis_image_to_3d_adapter_find(package->family);
}

trellis_status trellis_image_to_3d_component_execution_policy(
    const trellis_model_component_instance * component,
    trellis_ggml_attention_policy * attention_out,
    int * emulate_bf16_blocks_out) {
    if (component == NULL || attention_out == NULL ||
        emulate_bf16_blocks_out == NULL ||
        component->execution.struct_size < sizeof(component->execution) ||
        (component->execution.emulate_bf16_blocks != 0 &&
         component->execution.emulate_bf16_blocks != 1)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_ggml_attention_policy attention =
        TRELLIS_GGML_ATTENTION_POLICY_INIT;
    switch (component->execution.attention) {
        case TRELLIS_ATTENTION_NONE:
        case TRELLIS_ATTENTION_SDPA:
            if (component->execution.flash_kv_dtype != TRELLIS_DTYPE_UNKNOWN) {
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
            attention.mode = TRELLIS_GGML_ATTENTION_MODE_EXPLICIT;
            break;
        case TRELLIS_ATTENTION_FLASH:
            if (component->execution.flash_kv_dtype != TRELLIS_DTYPE_F16 &&
                component->execution.flash_kv_dtype != TRELLIS_DTYPE_BF16) {
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
            attention.mode =
                component->execution.flash_kv_dtype == TRELLIS_DTYPE_BF16 ?
                    TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16 :
                    TRELLIS_GGML_ATTENTION_MODE_FLASH;
            break;
        default:
            return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    *attention_out = attention;
    *emulate_bf16_blocks_out = component->execution.emulate_bf16_blocks;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_image_to_3d_adapter_validate_package(
    const trellis_image_to_3d_adapter_descriptor * adapter,
    const trellis_model_package * package) {
    if (!descriptor_is_valid(adapter) || package == NULL ||
        package->struct_size < sizeof(*package) ||
        package->family == NULL || package->task == NULL ||
        package->components == NULL || package->component_count == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (strcmp(package->task, TRELLIS_IMAGE_TO_3D_TASK_ID) != 0 ||
        strcmp(package->family, adapter->family) != 0) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    for (size_t i = 0; i < adapter->required_component_count; ++i) {
        const trellis_image_to_3d_component_requirement * requirement =
            &adapter->required_components[i];
        const trellis_model_component_instance * component =
            trellis_model_package_find_component(package, requirement->role);
        if (component == NULL || component->architecture == NULL ||
            strcmp(component->architecture, requirement->architecture) != 0 ||
            component->execution.attention != requirement->attention ||
            component->execution.flash_kv_dtype != requirement->flash_kv_dtype ||
            component->execution.emulate_bf16_blocks !=
                requirement->emulate_bf16_blocks) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }

        trellis_ggml_attention_policy attention =
            TRELLIS_GGML_ATTENTION_POLICY_INIT;
        int emulate_bf16_blocks = 0;
        if (trellis_image_to_3d_component_execution_policy(
                component,
                &attention,
                &emulate_bf16_blocks) != TRELLIS_STATUS_OK) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        trellis_ggml_attention_mode expected_mode =
            TRELLIS_GGML_ATTENTION_MODE_EXPLICIT;
        if (requirement->attention == TRELLIS_ATTENTION_FLASH) {
            expected_mode =
                requirement->flash_kv_dtype == TRELLIS_DTYPE_BF16 ?
                    TRELLIS_GGML_ATTENTION_MODE_FLASH_BF16 :
                    TRELLIS_GGML_ATTENTION_MODE_FLASH;
        }
        if (attention.mode != expected_mode ||
            emulate_bf16_blocks != requirement->emulate_bf16_blocks) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_image_to_3d_adapter_resolve_package(
    const trellis_model_package * package,
    const trellis_image_to_3d_adapter_descriptor ** adapter_out) {
    if (adapter_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *adapter_out = NULL;
    if (package == NULL || package->struct_size < sizeof(*package)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const trellis_image_to_3d_adapter_descriptor * adapter =
        trellis_image_to_3d_adapter_find_for_package(package);
    if (adapter == NULL) {
        return TRELLIS_STATUS_NOT_FOUND;
    }
    trellis_status status =
        trellis_image_to_3d_adapter_validate_package(adapter, package);
    if (status == TRELLIS_STATUS_OK) {
        *adapter_out = adapter;
    }
    return status;
}
