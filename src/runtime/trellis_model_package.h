#ifndef TRELLIS2_C_MODEL_PACKAGE_H
#define TRELLIS2_C_MODEL_PACKAGE_H

#include "trellis.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRELLIS_MODEL_PACKAGE_SCHEMA_V1 1u

typedef enum trellis_model_package_source {
    TRELLIS_MODEL_PACKAGE_SOURCE_MANIFEST = 0,
    TRELLIS_MODEL_PACKAGE_SOURCE_LEGACY = 1,
} trellis_model_package_source;

typedef enum trellis_attention_mode {
    TRELLIS_ATTENTION_NONE = 0,
    TRELLIS_ATTENTION_SDPA = 1,
    TRELLIS_ATTENTION_FLASH = 2,
} trellis_attention_mode;

/* Execution policy belongs to a component instance.  Architectures describe
 * graph and weight contracts; a package decides how each instance is run. */
typedef struct trellis_execution_policy {
    size_t struct_size;
    trellis_dtype compute_dtype;
    trellis_attention_mode attention;
    trellis_dtype flash_kv_dtype;
    int emulate_bf16_blocks;
} trellis_execution_policy;

#define TRELLIS_EXECUTION_POLICY_INIT \
    { sizeof(trellis_execution_policy), TRELLIS_DTYPE_F32, TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 }

typedef struct trellis_model_component_instance {
    char * role;
    char * architecture;
    char * weights;
    trellis_execution_policy execution;
} trellis_model_component_instance;

typedef struct trellis_model_package {
    size_t struct_size;
    uint32_t schema_version;
    trellis_model_package_source source;
    char * root;
    char * id;
    char * family;
    char * task;
    char * profile;
    trellis_model_component_instance * components;
    size_t component_count;

    /* Diagnostic evidence for the legacy compatibility path.  It is zero for
     * manifest packages and exactly one after a successful legacy load. */
    uint32_t legacy_pixal_marker_probes;
} trellis_model_package;

#define TRELLIS_MODEL_PACKAGE_INIT \
    { sizeof(trellis_model_package), 0, TRELLIS_MODEL_PACKAGE_SOURCE_MANIFEST, \
      NULL, NULL, NULL, NULL, NULL, NULL, 0, 0 }

/* Load <root>/model.json when present.  A present but invalid manifest is an
 * error and never falls back to legacy probing. */
trellis_status trellis_model_package_load(
    const char * root,
    trellis_model_package * package_out);

void trellis_model_package_free(trellis_model_package * package);

const trellis_model_component_instance * trellis_model_package_find_component(
    const trellis_model_package * package,
    const char * role);

trellis_status trellis_model_package_resolve_component_path(
    const trellis_model_package * package,
    const char * role,
    char * path_out,
    size_t path_size);

int trellis_model_weights_path_is_safe(const char * relative_path);

const char * trellis_attention_mode_name(trellis_attention_mode mode);

#ifdef __cplusplus
}
#endif

#endif
