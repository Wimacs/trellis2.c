#include "trellis_registry.h"

#include "trellis_model_package.h"

#include <string.h>

static const trellis_model_family_descriptor g_families[] = {
    {
        TRELLIS_REGISTRY_ABI_VERSION,
        sizeof(trellis_model_family_descriptor),
        "trellis2",
        "image_to_3d",
        "512",
    },
    {
        TRELLIS_REGISTRY_ABI_VERSION,
        sizeof(trellis_model_family_descriptor),
        "pixal3d",
        "image_to_3d",
        "1024_cascade",
    },
    {
        TRELLIS_REGISTRY_ABI_VERSION,
        sizeof(trellis_model_family_descriptor),
        "tokenskin",
        "mesh_rigging",
        "qwen3-0.6b-fsq",
    },
    {
        TRELLIS_REGISTRY_ABI_VERSION,
        sizeof(trellis_model_family_descriptor),
        "test_fixture",
        "test_weight_binding",
        "single_tensor",
    },
};

static const trellis_architecture_descriptor g_architectures[] = {
    { TRELLIS_REGISTRY_ABI_VERSION, sizeof(trellis_architecture_descriptor), "trellis_dit_flow", "trellis2" },
    { TRELLIS_REGISTRY_ABI_VERSION, sizeof(trellis_architecture_descriptor), "pixal_dit_flow", "pixal3d" },
    { TRELLIS_REGISTRY_ABI_VERSION, sizeof(trellis_architecture_descriptor), "sparse_structure_decoder", NULL },
    { TRELLIS_REGISTRY_ABI_VERSION, sizeof(trellis_architecture_descriptor), "sparse_unet_vae_decoder", NULL },
    { TRELLIS_REGISTRY_ABI_VERSION, sizeof(trellis_architecture_descriptor), "pixal_naf", "pixal3d" },
    { TRELLIS_REGISTRY_ABI_VERSION, sizeof(trellis_architecture_descriptor), "dinov3", NULL },
    { TRELLIS_REGISTRY_ABI_VERSION, sizeof(trellis_architecture_descriptor), "birefnet", NULL },
    { TRELLIS_REGISTRY_ABI_VERSION, sizeof(trellis_architecture_descriptor), "tokenskin_michelangelo_encoder", "tokenskin" },
    { TRELLIS_REGISTRY_ABI_VERSION, sizeof(trellis_architecture_descriptor), "tokenskin_qwen3", "tokenskin" },
    { TRELLIS_REGISTRY_ABI_VERSION, sizeof(trellis_architecture_descriptor), "tokenskin_fsq_cvae", "tokenskin" },
    { TRELLIS_REGISTRY_ABI_VERSION, sizeof(trellis_architecture_descriptor), "test_linear", "test_fixture" },
};

static const trellis_task_descriptor g_tasks[] = {
    {
        TRELLIS_REGISTRY_ABI_VERSION,
        sizeof(trellis_task_descriptor),
        "image_to_3d",
        "image",
        "gltf",
        0,
    },
    {
        TRELLIS_REGISTRY_ABI_VERSION,
        sizeof(trellis_task_descriptor),
        "mesh_rigging",
        "mesh",
        "rigged_mesh",
        0,
    },
    {
        TRELLIS_REGISTRY_ABI_VERSION,
        sizeof(trellis_task_descriptor),
        "test_weight_binding",
        "tensor",
        "binding_report",
        TRELLIS_TASK_FLAG_TEST_FIXTURE,
    },
};

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))

size_t trellis_registry_family_count(void) {
    return ARRAY_COUNT(g_families);
}

const trellis_model_family_descriptor * trellis_registry_family_at(size_t index) {
    return index < ARRAY_COUNT(g_families) ? &g_families[index] : NULL;
}

const trellis_model_family_descriptor * trellis_registry_find_family(const char * id) {
    if (id == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < ARRAY_COUNT(g_families); ++i) {
        if (strcmp(g_families[i].id, id) == 0) {
            return &g_families[i];
        }
    }
    return NULL;
}

size_t trellis_registry_architecture_count(void) {
    return ARRAY_COUNT(g_architectures);
}

const trellis_architecture_descriptor * trellis_registry_architecture_at(size_t index) {
    return index < ARRAY_COUNT(g_architectures) ? &g_architectures[index] : NULL;
}

const trellis_architecture_descriptor * trellis_registry_find_architecture(const char * id) {
    if (id == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < ARRAY_COUNT(g_architectures); ++i) {
        if (strcmp(g_architectures[i].id, id) == 0) {
            return &g_architectures[i];
        }
    }
    return NULL;
}

size_t trellis_registry_task_count(void) {
    return ARRAY_COUNT(g_tasks);
}

const trellis_task_descriptor * trellis_registry_task_at(size_t index) {
    return index < ARRAY_COUNT(g_tasks) ? &g_tasks[index] : NULL;
}

const trellis_task_descriptor * trellis_registry_find_task(const char * id) {
    if (id == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < ARRAY_COUNT(g_tasks); ++i) {
        if (strcmp(g_tasks[i].id, id) == 0) {
            return &g_tasks[i];
        }
    }
    return NULL;
}

static int descriptor_headers_are_valid(void) {
    for (size_t i = 0; i < ARRAY_COUNT(g_families); ++i) {
        if (g_families[i].abi_version != TRELLIS_REGISTRY_ABI_VERSION ||
            g_families[i].struct_size < sizeof(trellis_model_family_descriptor)) {
            return 0;
        }
    }
    for (size_t i = 0; i < ARRAY_COUNT(g_architectures); ++i) {
        if (g_architectures[i].abi_version != TRELLIS_REGISTRY_ABI_VERSION ||
            g_architectures[i].struct_size < sizeof(trellis_architecture_descriptor)) {
            return 0;
        }
    }
    for (size_t i = 0; i < ARRAY_COUNT(g_tasks); ++i) {
        if (g_tasks[i].abi_version != TRELLIS_REGISTRY_ABI_VERSION ||
            g_tasks[i].struct_size < sizeof(trellis_task_descriptor)) {
            return 0;
        }
    }
    return 1;
}

trellis_status trellis_registry_validate_package(const struct trellis_model_package * package) {
    if (package == NULL ||
        package->struct_size < sizeof(trellis_model_package) ||
        package->family == NULL || package->task == NULL ||
        package->components == NULL || package->component_count == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!descriptor_headers_are_valid()) {
        return TRELLIS_STATUS_ERROR;
    }

    const trellis_model_family_descriptor * family = trellis_registry_find_family(package->family);
    if (family == NULL || trellis_registry_find_task(package->task) == NULL) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    for (size_t i = 0; i < package->component_count; ++i) {
        const trellis_model_component_instance * component = &package->components[i];
        if (component->role == NULL || component->role[0] == '\0' ||
            component->architecture == NULL || component->weights == NULL ||
            !trellis_model_weights_path_is_safe(component->weights) ||
            component->execution.struct_size < sizeof(trellis_execution_policy) ||
            (component->execution.compute_dtype != TRELLIS_DTYPE_F32 &&
             component->execution.compute_dtype != TRELLIS_DTYPE_F16 &&
             component->execution.compute_dtype != TRELLIS_DTYPE_BF16) ||
            component->execution.attention < TRELLIS_ATTENTION_NONE ||
            component->execution.attention > TRELLIS_ATTENTION_FLASH ||
            (component->execution.attention == TRELLIS_ATTENTION_FLASH &&
             component->execution.flash_kv_dtype != TRELLIS_DTYPE_F16 &&
             component->execution.flash_kv_dtype != TRELLIS_DTYPE_BF16) ||
            (component->execution.attention != TRELLIS_ATTENTION_FLASH &&
             component->execution.flash_kv_dtype != TRELLIS_DTYPE_UNKNOWN)) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        for (size_t j = 0; j < i; ++j) {
            if (strcmp(component->role, package->components[j].role) == 0) {
                return TRELLIS_STATUS_PARSE_ERROR;
            }
        }
        const trellis_architecture_descriptor * architecture =
            trellis_registry_find_architecture(component->architecture);
        if (architecture == NULL ||
            (architecture->family != NULL && strcmp(architecture->family, family->id) != 0)) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    return TRELLIS_STATUS_OK;
}
