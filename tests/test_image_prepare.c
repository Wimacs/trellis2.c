#include "image_to_3d_internal.h"
#include "trellis_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static int write_opaque_ppm(const char * path) {
    static const unsigned char pixel[3] = { 255, 0, 0 };
    FILE * file = fopen(path, "wb");
    if (file == NULL) return 0;
    int ok = fputs("P6\n1 1\n255\n", file) >= 0 &&
        fwrite(pixel, 1, sizeof(pixel), file) == sizeof(pixel);
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int write_transparent_tga(const char * path) {
    unsigned char header[18] = {0};
    header[2] = 2;   /* uncompressed true-color */
    header[12] = 1;  /* width */
    header[14] = 1;  /* height */
    header[16] = 32;
    header[17] = 8;  /* eight alpha bits */
    static const unsigned char pixel_bgra[4] = { 0, 0, 255, 0 };
    FILE * file = fopen(path, "wb");
    if (file == NULL) return 0;
    int ok = fwrite(header, 1, sizeof(header), file) == sizeof(header) &&
        fwrite(pixel_bgra, 1, sizeof(pixel_bgra), file) == sizeof(pixel_bgra);
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static void restore_environment(const char * saved) {
    if (saved != NULL) {
        CHECK_TRUE(trellis_setenv("TRELLIS_BIREFNET_PATH", saved, 1) == 0);
    } else {
        CHECK_TRUE(trellis_unsetenv("TRELLIS_BIREFNET_PATH") == 0);
    }
}

static void test_prepare_profiles(void) {
    char model_dir[PATH_MAX];
    char opaque_path[PATH_MAX];
    char transparent_path[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(
        model_dir, sizeof(model_dir), "trellis_prepare_model", NULL));
    CHECK_TRUE(trellis_make_temp_path(
        opaque_path, sizeof(opaque_path), "trellis_prepare_opaque", ".ppm"));
    CHECK_TRUE(trellis_make_temp_path(
        transparent_path, sizeof(transparent_path), "trellis_prepare_alpha", ".tga"));
    CHECK_TRUE(trellis_mkdir_p(model_dir));
    CHECK_TRUE(write_opaque_ppm(opaque_path));
    CHECK_TRUE(write_transparent_tga(transparent_path));

    const char * environment = getenv("TRELLIS_BIREFNET_PATH");
    char * saved = NULL;
    if (environment != NULL) {
        saved = (char *) malloc(strlen(environment) + 1u);
        if (saved != NULL) strcpy(saved, environment);
    }
    CHECK_TRUE(environment == NULL || saved != NULL);
    CHECK_TRUE(trellis_unsetenv("TRELLIS_BIREFNET_PATH") == 0);

    trellis_prepared_condition_image prepared;
    memset(&prepared, 0, sizeof(prepared));
    CHECK_TRUE(trellis_pipeline_prepare_condition_image(
        model_dir,
        model_dir,
        opaque_path,
        NULL,
        TRELLIS_BACKEND_CPU,
        0,
        0,
        &prepared) == TRELLIS_STATUS_OK);
    CHECK_TRUE(strcmp(prepared.source_path, opaque_path) == 0);
    CHECK_TRUE(prepared.converted_path[0] == '\0');
    CHECK_TRUE(prepared.foreground_path[0] == '\0');
    trellis_pipeline_prepared_condition_image_free(&prepared);

    CHECK_TRUE(trellis_pipeline_prepare_condition_image(
        model_dir,
        model_dir,
        opaque_path,
        NULL,
        TRELLIS_BACKEND_CPU,
        0,
        1,
        &prepared) == TRELLIS_STATUS_NOT_FOUND);
    CHECK_TRUE(trellis_pipeline_prepare_condition_image_for_image_to_gltf(
        model_dir,
        model_dir,
        opaque_path,
        NULL,
        TRELLIS_BACKEND_CPU,
        0,
        1,
        &prepared) == TRELLIS_STATUS_INVALID_ARGUMENT);

    CHECK_TRUE(trellis_pipeline_prepare_condition_image(
        model_dir,
        model_dir,
        transparent_path,
        NULL,
        TRELLIS_BACKEND_CPU,
        0,
        1,
        &prepared) == TRELLIS_STATUS_OK);
    CHECK_TRUE(strcmp(prepared.source_path, transparent_path) == 0);
    trellis_pipeline_prepared_condition_image_free(&prepared);

    restore_environment(saved);
    free(saved);
    trellis_unlink(transparent_path);
    trellis_unlink(opaque_path);
    remove(model_dir);
}

int main(void) {
    test_prepare_profiles();
    if (failures != 0) {
        fprintf(stderr, "%d image prepare test(s) failed\n", failures);
        return 1;
    }
    printf("image prepare tests passed\n");
    return 0;
}
