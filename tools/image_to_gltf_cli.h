#ifndef TRELLIS2_C_IMAGE_TO_GLTF_CLI_H
#define TRELLIS2_C_IMAGE_TO_GLTF_CLI_H

typedef enum trellis_image_to_gltf_cli_model {
    TRELLIS_IMAGE_TO_GLTF_CLI_TRELLIS2 = 0,
    TRELLIS_IMAGE_TO_GLTF_CLI_PIXAL3D = 1,
} trellis_image_to_gltf_cli_model;

int trellis_image_to_gltf_cli_main(
    trellis_image_to_gltf_cli_model model,
    int argc,
    char ** argv);

#endif
