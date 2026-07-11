#ifndef TRELLIS2_C_IMAGE_TO_3D_ADAPTER_H
#define TRELLIS2_C_IMAGE_TO_3D_ADAPTER_H

#include "trellis_ggml_layers.h"
#include "trellis_model_package.h"
#include "gltf_axes.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRELLIS_IMAGE_TO_3D_ADAPTER_ABI_VERSION 2u
#define TRELLIS_IMAGE_TO_3D_TASK_ID "image_to_3d"

typedef enum trellis_image_to_3d_cascade_quantization {
    TRELLIS_IMAGE_TO_3D_CASCADE_QUANTIZATION_INVALID = 0,
    TRELLIS_IMAGE_TO_3D_CASCADE_QUANTIZATION_FLOOR = 1,
    TRELLIS_IMAGE_TO_3D_CASCADE_QUANTIZATION_ROUND = 2,
} trellis_image_to_3d_cascade_quantization;

/* A requirement is intentionally expressed in package terms.  The task owns
 * roles while an architecture owns tensor binding and graph construction. */
typedef struct trellis_image_to_3d_component_requirement {
    const char * role;
    const char * architecture;
    trellis_attention_mode attention;
    trellis_dtype flash_kv_dtype;
    int emulate_bf16_blocks;
} trellis_image_to_3d_component_requirement;

typedef struct trellis_image_to_3d_adapter_descriptor {
    uint32_t abi_version;
    size_t struct_size;
    const char * id;
    const char * family;

    int uses_projected_conditioning;
    int requires_transparent_foreground;
    trellis_image_to_3d_cascade_quantization cascade_quantization;

    /* Zero means that the family has no camera-conditioned defaults. */
    float default_camera_angle_x;
    float default_camera_distance;
    float default_mesh_scale;

    /* Final coordinate transform selected after texture baking. */
    trellis_pipeline_gltf_coordinate_transform gltf_coordinate_transform;

    const trellis_image_to_3d_component_requirement * required_components;
    size_t required_component_count;
} trellis_image_to_3d_adapter_descriptor;

size_t trellis_image_to_3d_adapter_count(void);

const trellis_image_to_3d_adapter_descriptor *
trellis_image_to_3d_adapter_at(size_t index);

/* Family lookup deliberately does not validate the package task.  This keeps
 * dispatch and validation separate, so a known family carrying a non-3D task
 * is still rejected by validate_package rather than silently mis-dispatched. */
const trellis_image_to_3d_adapter_descriptor *
trellis_image_to_3d_adapter_find(const char * family);

const trellis_image_to_3d_adapter_descriptor *
trellis_image_to_3d_adapter_find_for_package(
    const trellis_model_package * package);

trellis_status trellis_image_to_3d_adapter_validate_package(
    const trellis_image_to_3d_adapter_descriptor * adapter,
    const trellis_model_package * package);

trellis_status trellis_image_to_3d_adapter_resolve_package(
    const trellis_model_package * package,
    const trellis_image_to_3d_adapter_descriptor ** adapter_out);

/* Converts package execution metadata into the policy consumed by ggml flow
 * executors.  TRELLIS_ATTENTION_NONE and SDPA both select the explicit graph;
 * FLASH selects the instance-scoped flash graph. */
trellis_status trellis_image_to_3d_component_execution_policy(
    const trellis_model_component_instance * component,
    trellis_ggml_attention_policy * attention_out,
    int * emulate_bf16_blocks_out);

/* Built-in descriptors are exposed as accessors so static registries and
 * focused tests do not depend on descriptor object linkage details. */
const trellis_image_to_3d_adapter_descriptor *
trellis_image_to_3d_trellis2_adapter(void);

const trellis_image_to_3d_adapter_descriptor *
trellis_image_to_3d_pixal3d_adapter(void);

#ifdef __cplusplus
}
#endif

#endif
