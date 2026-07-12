#include "trellis.h"

static const trellis_component_status g_status[] = {
    {
        TRELLIS_COMPONENT_SPARSE_STRUCTURE_FLOW,
        "SparseStructureFlowModel",
        true,
        "dense-token DiT operators, RoPE self-attention path, checkpoint binding, real-weight CUDA forward, and CLI sampler wiring are implemented",
    },
    {
        TRELLIS_COMPONENT_SPARSE_STRUCTURE_DECODER,
        "SparseStructureDecoder",
        true,
        "checkpoint binding, Conv3D, ChannelLayerNorm3D, SiLU, residual add, pixel-shuffle, real-weight CUDA decode, and voxel frame export are implemented",
    },
    {
        TRELLIS_COMPONENT_SHAPE_SLAT_FLOW,
        "SLatFlowModel",
        true,
        "sparse coordinate host semantics, dense transformer operators, and ggml FlashAttention SDPA are implemented",
    },
    {
        TRELLIS_COMPONENT_TEX_SLAT_FLOW,
        "Texture SLatFlowModel",
        true,
        "same pure ggml DiT path as shape SLat flow; texture noise plus normalized shape concat conditioning is wired in the CLI pipeline",
    },
    {
        TRELLIS_COMPONENT_SHAPE_SLAT_DECODER,
        "FlexiDualGridVaeDecoder",
        true,
        "shared SparseUnetVaeDecoder CUDA sparse-conv body, predicted subdivision guides, and FlexiDualGrid mesh extraction are implemented",
    },
    {
        TRELLIS_COMPONENT_TEX_SLAT_DECODER,
        "SparseUnetVaeDecoder",
        true,
        "shared SparseUnetVaeDecoder CUDA sparse-conv body is implemented with guide_subs from the shape decoder and 6-channel PBR voxel output",
    },
    {
        TRELLIS_COMPONENT_DINOV3_IMAGE_ENCODER,
        "DINOv3 image encoder",
        true,
        "checkpoint binding, image preprocessing, and full ggml image encoder graph are implemented for CLI conditioning",
    },
    {
        TRELLIS_COMPONENT_BIREFNET_BACKGROUND_REMOVAL,
        "BiRefNet background removal",
        true,
        "GGUF loading, Swin/BiRefNet graph, CLI pre-mask PNG wiring, and ggml deformable convolution backends are implemented",
    },
    {
        TRELLIS_COMPONENT_OVOXEL_POSTPROCESS,
        "O-Voxel postprocess",
        true,
        "vkmesh topology cleanup, narrow-band remesh, simplify, and Vulkan compute UV-space PBR texture bake are implemented",
    },
};

size_t trellis_component_status_count(void) {
    return sizeof(g_status) / sizeof(g_status[0]);
}

const trellis_component_status * trellis_component_status_at(size_t index) {
    if (index >= trellis_component_status_count()) {
        return 0;
    }
    return &g_status[index];
}
