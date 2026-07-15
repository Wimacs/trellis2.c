#ifndef TRELLIS2_C_MESH_RIGGING_SOURCE_APPEARANCE_H
#define TRELLIS2_C_MESH_RIGGING_SOURCE_APPEARANCE_H

#include "cgltf.h"
#include "gltf_io.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trellis_gltf_source_image {
    const unsigned char * bytes;
    unsigned char * owned_bytes;
    size_t size;
    const char * embedded_mime_type;
    char * owned_mime_type;
    char * original_uri;
    cgltf_buffer_view * original_buffer_view;
    char * original_mime_type;
} trellis_gltf_source_image;

typedef struct trellis_gltf_source_appearance {
    cgltf_data * data;
    trellis_gltf_source_image * images;
} trellis_gltf_source_appearance;

/* Reopens the flattened asset's source document and makes every referenced
 * image payload directly readable. Required extensions are rejected because
 * cgltf_write cannot preserve their required status. The caller may borrow
 * the source materials/textures/samplers/images until free is called. */
trellis_status trellis_gltf_source_appearance_load(
    const trellis_mesh_rigging_asset * asset,
    trellis_gltf_source_appearance * appearance,
    char * error_out,
    size_t error_size);

void trellis_gltf_source_appearance_free(
    trellis_gltf_source_appearance * appearance);

const char * trellis_gltf_cgltf_result_name(cgltf_result result);

#ifdef __cplusplus
}
#endif

#endif
