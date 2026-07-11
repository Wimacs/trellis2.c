#ifndef TRELLIS2_C_MESH_RIGGING_GENERATION_H
#define TRELLIS2_C_MESH_RIGGING_GENERATION_H

#include "mesh_rigging_internal.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Generates one reference-compatible TokenRig sequence.  The returned array
 * includes manually prepended BOS/class IDs and the final global EOS. */
trellis_status tokenskin_generate_rig_tokens(
    const tokenskin_runtime_model * model,
    const float * mesh_embeddings_512x896,
    const trellis_tokenskin_rig_options * options,
    int32_t ** sequence_out,
    size_t * sequence_count_out);

#ifdef __cplusplus
}
#endif

#endif
