#ifndef TRELLIS2_C_IMAGE_TO_3D_GLTF_AXES_H
#define TRELLIS2_C_IMAGE_TO_3D_GLTF_AXES_H

typedef enum trellis_pipeline_gltf_coordinate_transform {
    /* o-voxel's generic GLB conversion: raw (x, y, z) -> (x, z, -y). */
    TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_TRELLIS = 0,
    /* Pixal3D's final GLB conversion: raw (x, y, z) -> (-x, y, -z). */
    TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_PIXAL3D = 1,
} trellis_pipeline_gltf_coordinate_transform;

#endif
