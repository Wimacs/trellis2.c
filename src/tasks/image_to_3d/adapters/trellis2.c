#include "../adapter.h"

static const trellis_image_to_3d_component_requirement g_trellis2_components[] = {
    { "sparse_structure_flow", "trellis_dit_flow", TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_F16, 0 },
    { "sparse_structure_decoder", "sparse_structure_decoder", TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
    { "shape_flow_512", "trellis_dit_flow", TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_F16, 0 },
    { "shape_flow_1024", "trellis_dit_flow", TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_BF16, 0 },
    { "texture_flow_512", "trellis_dit_flow", TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_F16, 0 },
    { "texture_flow_1024", "trellis_dit_flow", TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_BF16, 0 },
    { "shape_decoder", "sparse_unet_vae_decoder", TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
    { "texture_decoder", "sparse_unet_vae_decoder", TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
};

static const trellis_image_to_3d_adapter_descriptor g_trellis2_adapter = {
    TRELLIS_IMAGE_TO_3D_ADAPTER_ABI_VERSION,
    sizeof(trellis_image_to_3d_adapter_descriptor),
    "trellis2.image_to_3d",
    "trellis2",
    0,
    0,
    TRELLIS_IMAGE_TO_3D_CASCADE_QUANTIZATION_FLOOR,
    0.0f,
    0.0f,
    0.0f,
    TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_TRELLIS,
    g_trellis2_components,
    sizeof(g_trellis2_components) / sizeof(g_trellis2_components[0]),
};

const trellis_image_to_3d_adapter_descriptor *
trellis_image_to_3d_trellis2_adapter(void) {
    return &g_trellis2_adapter;
}
