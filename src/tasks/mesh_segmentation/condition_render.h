#ifndef TRELLIS2_C_MESH_SEGMENTATION_CONDITION_RENDER_H
#define TRELLIS2_C_MESH_SEGMENTATION_CONDITION_RENDER_H

#include "../image_to_3d/image_to_3d_internal.h"
#include "../mesh_rigging/gltf_io.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Task-private RGBA8 image returned by the deterministic CPU condition
 * renderer. Pixels are tightly packed, top row first. */
typedef struct trellis_mesh_segmentation_condition_image {
    int width;
    int height;
    uint8_t * rgba;
} trellis_mesh_segmentation_condition_image;

#define TRELLIS_MESH_SEGMENTATION_CONDITION_IMAGE_INIT { 0, 0, NULL }

/* Renders flattened world-space geometry with SegviGen's fixed 40-degree
 * Blender camera. Geometry receives the glTF importer's render-only Y-up to
 * Z-up mapping, then is centered and uniformly scaled to maximum extent 1.
 * When asset->source_path is present, glTF base-color textures/factors and
 * alpha modes are sampled per fragment. A deterministic 2x supersampled
 * raster is resolved to the requested size. Voxelization and SLat encoding
 * remain in direct glTF world coordinates. image_size must be at least 2.
 * The caller owns image_out->rgba and releases it with the matching free
 * function. */
trellis_status trellis_mesh_segmentation_render_condition_rgba(
    const trellis_mesh_rigging_asset * asset,
    int image_size,
    trellis_mesh_segmentation_condition_image * image_out);

void trellis_mesh_segmentation_condition_image_free(
    trellis_mesh_segmentation_condition_image * image);

/* Renders a PNG and makes it directly consumable by the existing DINO image
 * conditioning stage. When optional_png_path is NULL or empty, a temporary
 * PNG is created and recorded in converted_path, so the existing prepared
 * image cleanup function removes it. A supplied path is caller-owned and is
 * never removed by prepared-image cleanup. */
trellis_status trellis_mesh_segmentation_render_condition(
    const trellis_mesh_rigging_asset * asset,
    int image_size,
    trellis_prepared_condition_image * prepared_out,
    const char * optional_png_path);

#ifdef __cplusplus
}
#endif

#endif
