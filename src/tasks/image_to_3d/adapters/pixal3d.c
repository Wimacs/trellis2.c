#include "../adapter.h"

static const trellis_image_to_3d_component_requirement g_pixal3d_components[] = {
    { "sparse_structure_flow", "pixal_dit_flow", TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_BF16, 1 },
    { "sparse_structure_decoder", "sparse_structure_decoder", TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
    { "shape_flow_512", "pixal_dit_flow", TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_BF16, 1 },
    { "shape_flow_1024", "pixal_dit_flow", TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_BF16, 1 },
    { "texture_flow_1024", "pixal_dit_flow", TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_BF16, 1 },
    { "shape_decoder", "sparse_unet_vae_decoder", TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
    { "texture_decoder", "sparse_unet_vae_decoder", TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
    { "naf_encoder", "pixal_naf", TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
};

static const trellis_image_to_3d_adapter_descriptor g_pixal3d_adapter = {
    TRELLIS_IMAGE_TO_3D_ADAPTER_ABI_VERSION,
    sizeof(trellis_image_to_3d_adapter_descriptor),
    "pixal3d.image_to_3d",
    "pixal3d",
    1,
    1,
    TRELLIS_IMAGE_TO_3D_CASCADE_QUANTIZATION_ROUND,
    0.8575560450553894f,
    2.0f,
    1.0f,
    g_pixal3d_components,
    sizeof(g_pixal3d_components) / sizeof(g_pixal3d_components[0]),
};

const trellis_image_to_3d_adapter_descriptor *
trellis_image_to_3d_pixal3d_adapter(void) {
    return &g_pixal3d_adapter;
}
