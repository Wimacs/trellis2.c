#ifndef TRELLIS2_C_MESH_RIGGING_GLTF_IO_H
#define TRELLIS2_C_MESH_RIGGING_GLTF_IO_H

#include "trellis.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A flattened primitive instance.  A glTF mesh referenced by two nodes creates
 * two ranges because the positions are transformed into world space.  The
 * source indices let the rigged-glTF writer map generated weights back to the
 * original primitive without exposing cgltf outside this task. */
typedef struct trellis_mesh_rigging_primitive_range {
    size_t first_vertex;
    size_t vertex_count;
    size_t first_triangle;
    size_t triangle_count;
    size_t source_node_index;
    size_t source_mesh_index;
    size_t source_primitive_index;
    size_t source_position_accessor_index;
    size_t source_material_index; /* SIZE_MAX when the primitive has no material. */
    /* Bit N is set when the source primitive contains TEXCOORD_N.  The
     * corresponding flattened values live in asset->texcoords[N]. */
    uint64_t texcoord_mask;
} trellis_mesh_rigging_primitive_range;

/* Task-private, inference-oriented representation of a glTF asset.  Positions
 * are flattened to world space and primitives are deliberately not welded:
 * keeping their vertex order intact is required when JOINTS_0/WEIGHTS_0 are
 * appended to the source document later. */
typedef struct trellis_mesh_rigging_asset {
    char * source_path;
    float * positions;     /* [vertex_count, 3], world space */
    float * normals;       /* [vertex_count, 3], area-weighted unit normals */
    /* One optional [vertex_count, 2] float array per source TEXCOORD set.
     * Gaps are NULL.  Flattening keeps accessor element order, so these UVs
     * can be emitted alongside the generated JOINTS_0/WEIGHTS_0 attributes. */
    float ** texcoords;
    size_t texcoord_set_count;
    float * face_normals;  /* [triangle_count, 3], unit normals */
    uint32_t * triangles;  /* [triangle_count, 3], indices into positions */
    trellis_mesh_rigging_primitive_range * primitives;
    size_t vertex_count;
    size_t triangle_count;
    size_t primitive_count;
    float aabb_min[3];
    float aabb_max[3];
} trellis_mesh_rigging_asset;

#define TRELLIS_MESH_RIGGING_ASSET_INIT \
    { NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL, 0, 0, 0, \
      { 0, 0, 0 }, { 0, 0, 0 } }

/* Loads .glb or .gltf (including external/data-URI buffers), expands indexed or
 * non-indexed triangle/strip/fan primitives, applies every mesh node's world
 * transform in the selected scene, and computes robust geometric normals and
 * the world-space AABB. Lines and point primitives are ignored. */
trellis_status trellis_mesh_rigging_gltf_load(
    const char * path,
    trellis_mesh_rigging_asset * asset_out,
    char * error_out,
    size_t error_size);

void trellis_mesh_rigging_asset_free(trellis_mesh_rigging_asset * asset);

#ifdef __cplusplus
}
#endif

#endif
