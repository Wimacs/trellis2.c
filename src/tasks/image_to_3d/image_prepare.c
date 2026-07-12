#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "image_to_3d_internal.h"

#include "trellis_platform.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern unsigned char * stbi_load(
    char const * filename,
    int * x,
    int * y,
    int * comp,
    int req_comp);
extern void stbi_image_free(void * retval_from_stbi_load);
extern int stbi_write_png(
    char const * filename,
    int w,
    int h,
    int comp,
    const void * data,
    int stride_in_bytes);

typedef struct trellis_condition_image_policy {
    const char * temp_prefix;
    int inspect_alpha_with_explicit_birefnet;
    trellis_status missing_foreground_status;
    const char * missing_foreground_message;
} trellis_condition_image_policy;

static int path_has_ext(const char * path, const char * ext) {
    if (path == NULL || ext == NULL) {
        return 0;
    }
    const char * dot = strrchr(path, '.');
    if (dot == NULL) {
        return 0;
    }
    ++dot;
    while (*dot != '\0' && *ext != '\0') {
        if (tolower((unsigned char) *dot) != tolower((unsigned char) *ext)) {
            return 0;
        }
        ++dot;
        ++ext;
    }
    return *dot == '\0' && *ext == '\0';
}

static trellis_status image_has_transparent_alpha(
    const char * image_path,
    int * has_transparent_alpha_out) {
    if (image_path == NULL || image_path[0] == '\0' ||
        has_transparent_alpha_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *has_transparent_alpha_out = 0;

    int width = 0;
    int height = 0;
    int components = 0;
    unsigned char * rgba = stbi_load(
        image_path,
        &width,
        &height,
        &components,
        4);
    if (rgba == NULL || width <= 0 || height <= 0 ||
        (size_t) width > SIZE_MAX / (size_t) height) {
        TRELLIS_ERROR("image prep: failed to inspect alpha channel: %s", image_path);
        stbi_image_free(rgba);
        return TRELLIS_STATUS_IO_ERROR;
    }

    const size_t pixels = (size_t) width * (size_t) height;
    for (size_t i = 0; i < pixels; ++i) {
        if (rgba[i * 4u + 3u] != 255u) {
            *has_transparent_alpha_out = 1;
            break;
        }
    }
    stbi_image_free(rgba);
    return TRELLIS_STATUS_OK;
}

static int resolve_packaged_birefnet_path(
    const char * model_dir,
    const char * dino_dir,
    char * path,
    size_t path_size) {
    if (model_dir == NULL || model_dir[0] == '\0' || path == NULL ||
        path_size == 0) {
        return 0;
    }
    const char * environment_path = getenv("TRELLIS_BIREFNET_PATH");
    if (environment_path != NULL && environment_path[0] != '\0') {
        const int n = snprintf(path, path_size, "%s", environment_path);
        if (n >= 0 && (size_t) n < path_size && trellis_access_read(path)) {
            return 1;
        }
    }
    static const char * candidates[] = {
        "ckpts/BiRefNet-F16.gguf",
        "BiRefNet-F16.gguf",
        "BiRefNet/BiRefNet-F16.gguf",
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if (trellis_make_model_path(
                model_dir,
                candidates[i],
                path,
                path_size) == TRELLIS_STATUS_OK &&
            trellis_access_read(path)) {
            return 1;
        }
    }
    if (dino_dir != NULL && dino_dir[0] != '\0') {
        static const char * dino_sibling_candidates[] = {
            "../BiRefNet/BiRefNet-F16.gguf",
            "../BiRefNet-F16.gguf",
        };
        for (size_t i = 0;
             i < sizeof(dino_sibling_candidates) /
                 sizeof(dino_sibling_candidates[0]);
             ++i) {
            if (trellis_make_model_path(
                    dino_dir,
                    dino_sibling_candidates[i],
                    path,
                    path_size) == TRELLIS_STATUS_OK &&
                trellis_access_read(path)) {
                return 1;
            }
        }
    }
    path[0] = '\0';
    return 0;
}

static int convert_webp_to_png_ffmpeg(
    const char * input_path,
    const char * output_path) {
    char * argv[] = {
        (char *) "ffmpeg",
        (char *) "-y",
        (char *) "-hide_banner",
        (char *) "-loglevel",
        (char *) "error",
        (char *) "-i",
        (char *) input_path,
        (char *) "-frames:v",
        (char *) "1",
        (char *) output_path,
        NULL,
    };
    return trellis_run_process_search_path(argv);
}

static void unlink_if_set(const char * path) {
    if (path != NULL && path[0] != '\0') {
        trellis_unlink(path);
    }
}

static trellis_status apply_birefnet_background_removal(
    const char * input_path,
    const char * gguf_path,
    const char * output_path,
    trellis_backend_kind backend_kind,
    int device) {
    if (input_path == NULL || gguf_path == NULL || output_path == NULL ||
        input_path[0] == '\0' || gguf_path[0] == '\0' ||
        output_path[0] == '\0') {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    int width = 0;
    int height = 0;
    int comp = 0;
    unsigned char * rgba = stbi_load(
        input_path,
        &width,
        &height,
        &comp,
        4);
    if (rgba == NULL || width <= 0 || height <= 0) {
        TRELLIS_ERROR("BiRefNet: failed to load image %s", input_path);
        stbi_image_free(rgba);
        return TRELLIS_STATUS_IO_ERROR;
    }

    TRELLIS_INFO(
        "BiRefNet: loading background-removal GGUF on %s device=%d: %s",
        trellis_backend_kind_name(backend_kind),
        device,
        gguf_path);
    trellis_birefnet_model model;
    memset(&model, 0, sizeof(model));
    trellis_status status = trellis_birefnet_load_gguf_with_backend(
        &model,
        gguf_path,
        backend_kind,
        device);
    unsigned char * mask = NULL;
    if (status == TRELLIS_STATUS_OK) {
        status = trellis_birefnet_compute_mask_u8(
            &model,
            rgba,
            width,
            height,
            &mask);
    }
    if (status == TRELLIS_STATUS_OK) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t i = (size_t) y * (size_t) width + (size_t) x;
                rgba[i * 4u + 3u] = (unsigned char) (
                    ((unsigned int) rgba[i * 4u + 3u] *
                         (unsigned int) mask[i] +
                     127u) /
                    255u);
            }
        }
        if (!stbi_write_png(
                output_path,
                width,
                height,
                4,
                rgba,
                width * 4)) {
            status = TRELLIS_STATUS_IO_ERROR;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        TRELLIS_INFO("BiRefNet: wrote masked PNG: %s", output_path);
    } else {
        TRELLIS_ERROR(
            "BiRefNet: background removal failed: %s",
            trellis_status_string(status));
    }
    free(mask);
    trellis_birefnet_free(&model);
    stbi_image_free(rgba);
    return status;
}

void trellis_pipeline_prepared_condition_image_free(
    trellis_prepared_condition_image * prepared) {
    if (prepared == NULL) {
        return;
    }
    unlink_if_set(prepared->foreground_path);
    unlink_if_set(prepared->converted_path);
    memset(prepared, 0, sizeof(*prepared));
}

static trellis_status prepare_condition_image_impl(
    const char * model_dir,
    const char * dino_dir,
    const char * image_path,
    const char * birefnet_path,
    trellis_backend_kind backend_kind,
    int device,
    int require_foreground,
    const trellis_condition_image_policy * policy,
    trellis_prepared_condition_image * prepared_out) {
    if (model_dir == NULL || model_dir[0] == '\0' || image_path == NULL ||
        image_path[0] == '\0' || device < 0 || policy == NULL ||
        policy->temp_prefix == NULL || prepared_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_prepared_condition_image prepared;
    memset(&prepared, 0, sizeof(prepared));
    const char * current_path = image_path;
    trellis_status status = TRELLIS_STATUS_OK;

    if (path_has_ext(image_path, "webp")) {
        if (!trellis_make_temp_path(
                prepared.converted_path,
                sizeof(prepared.converted_path),
                policy->temp_prefix,
                ".png")) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        TRELLIS_INFO(
            "image prep: converting WebP -> PNG via ffmpeg: %s",
            prepared.converted_path);
        if (!convert_webp_to_png_ffmpeg(
                image_path,
                prepared.converted_path)) {
            TRELLIS_ERROR(
                "image prep: WebP conversion failed; install ffmpeg or convert the image to PNG/JPEG first");
            trellis_pipeline_prepared_condition_image_free(&prepared);
            return TRELLIS_STATUS_IO_ERROR;
        }
        current_path = prepared.converted_path;
    }

    char discovered_birefnet[4096];
    discovered_birefnet[0] = '\0';
    const char * effective_birefnet =
        birefnet_path != NULL && birefnet_path[0] != '\0' ?
            birefnet_path :
            NULL;
    int has_transparent_alpha = 0;
    if (effective_birefnet == NULL ||
        policy->inspect_alpha_with_explicit_birefnet) {
        status = image_has_transparent_alpha(
            current_path,
            &has_transparent_alpha);
        if (status != TRELLIS_STATUS_OK) {
            trellis_pipeline_prepared_condition_image_free(&prepared);
            return status;
        }
    }

    if (!has_transparent_alpha && effective_birefnet == NULL &&
        resolve_packaged_birefnet_path(
            model_dir,
            dino_dir,
            discovered_birefnet,
            sizeof(discovered_birefnet))) {
        effective_birefnet = discovered_birefnet;
        TRELLIS_INFO(
            "image prep: auto-discovered BiRefNet %s",
            effective_birefnet);
    }
    if (!has_transparent_alpha && effective_birefnet == NULL &&
        require_foreground) {
        TRELLIS_ERROR("%s", policy->missing_foreground_message);
        trellis_pipeline_prepared_condition_image_free(&prepared);
        return policy->missing_foreground_status;
    }

    if (effective_birefnet != NULL) {
        if (!trellis_make_temp_path(
                prepared.foreground_path,
                sizeof(prepared.foreground_path),
                "trellis2_birefnet",
                ".png")) {
            trellis_pipeline_prepared_condition_image_free(&prepared);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        TRELLIS_INFO(
            "image prep: running BiRefNet background removal -> %s",
            prepared.foreground_path);
        status = apply_birefnet_background_removal(
            current_path,
            effective_birefnet,
            prepared.foreground_path,
            backend_kind,
            device);
        if (status != TRELLIS_STATUS_OK) {
            trellis_pipeline_prepared_condition_image_free(&prepared);
            return status;
        }
        current_path = prepared.foreground_path;
    }

    const int n = snprintf(
        prepared.source_path,
        sizeof(prepared.source_path),
        "%s",
        current_path);
    if (n < 0 || (size_t) n >= sizeof(prepared.source_path)) {
        trellis_pipeline_prepared_condition_image_free(&prepared);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *prepared_out = prepared;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pipeline_prepare_condition_image(
    const char * model_dir,
    const char * dino_dir,
    const char * image_path,
    const char * birefnet_path,
    trellis_backend_kind backend_kind,
    int device,
    int require_foreground,
    trellis_prepared_condition_image * prepared_out) {
    static const trellis_condition_image_policy policy = {
        "trellis2_condition",
        1,
        TRELLIS_STATUS_NOT_FOUND,
        "image prep: opaque input requires BiRefNet; pass --birefnet FILE or set TRELLIS_BIREFNET_PATH",
    };
    return prepare_condition_image_impl(
        model_dir,
        dino_dir,
        image_path,
        birefnet_path,
        backend_kind,
        device,
        require_foreground,
        &policy,
        prepared_out);
}

trellis_status trellis_pipeline_prepare_condition_image_for_image_to_gltf(
    const char * model_dir,
    const char * dino_dir,
    const char * image_path,
    const char * birefnet_path,
    trellis_backend_kind backend_kind,
    int device,
    int require_foreground,
    trellis_prepared_condition_image * prepared_out) {
    static const trellis_condition_image_policy policy = {
        "trellis2_image_to_gltf",
        0,
        TRELLIS_STATUS_INVALID_ARGUMENT,
        "pipeline: opaque Pixal3D input requires BiRefNet; place it under model/ckpts or beside DINO, set TRELLIS_BIREFNET_PATH, or pass --birefnet FILE",
    };
    return prepare_condition_image_impl(
        model_dir,
        dino_dir,
        image_path,
        birefnet_path,
        backend_kind,
        device,
        require_foreground,
        &policy,
        prepared_out);
}
