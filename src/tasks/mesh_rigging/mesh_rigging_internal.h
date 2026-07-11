#ifndef TRELLIS2_C_MESH_RIGGING_INTERNAL_H
#define TRELLIS2_C_MESH_RIGGING_INTERNAL_H

#include "trellis_model_package.h"
#include "tokenskin.h"

typedef struct tokenskin_runtime_model {
    trellis_backend_context backend;
    trellis_tensor_store mesh_store;
    trellis_tensor_store qwen_store;
    trellis_tensor_store skin_store;
    tokenskin_mesh_encoder_weights mesh;
    tokenskin_qwen_weights qwen;
    tokenskin_fsq_cvae_weights skin;
    trellis_ggml_attention_policy attention_policy;
} tokenskin_runtime_model;

trellis_status tokenskin_runtime_model_load(
    const trellis_model_package * package,
    trellis_backend_kind backend_kind,
    int device,
    int disable_flash,
    tokenskin_runtime_model * model);

void tokenskin_runtime_model_free(tokenskin_runtime_model * model);

#endif
