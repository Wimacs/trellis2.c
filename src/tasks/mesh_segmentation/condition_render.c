#include "condition_render.h"
#include "material_features.h"

#include "trellis_platform.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int stbi_write_png(
    char const * filename,
    int w,
    int h,
    int comp,
    const void * data,
    int stride_in_bytes);

typedef struct segmentation_render_vertex {
    float x;
    float y;
    float inverse_depth;
    float position[3];
    float normal[3];
    int visible;
} segmentation_render_vertex;

#define SEGMENTATION_CONDITION_SSAA 2

/* Camera-to-world transform from SegviGen/data_toolkit/transforms.json.
 * Blender cameras look down local -Z, so projection uses -camera_z. */
static const float k_camera_right[3] = {
    0.8819212913513184f,
    -0.4713967740535736f,
    -0.0000000488115717f,
};
static const float k_camera_up[3] = {
    0.06494797021150589f,
    0.12150891125202179f,
    0.9904633164405823f,
};
static const float k_camera_back[3] = {
    -0.46690112352371216f,
    -0.8735106587409973f,
    0.13777753710746765f,
};
static const float k_camera_position[3] = {
    -0.9338021874427795f,
    -1.7470210790634155f,
    0.2755555510520935f,
};

static float dot3(const float left[3], const float right[3]) {
    return left[0] * right[0] + left[1] * right[1] +
           left[2] * right[2];
}

static void cross3(
    const float left[3],
    const float right[3],
    float result[3]) {
    result[0] = left[1] * right[2] - left[2] * right[1];
    result[1] = left[2] * right[0] - left[0] * right[2];
    result[2] = left[0] * right[1] - left[1] * right[0];
}

static int normalize3(float value[3]) {
    const float length_squared = dot3(value, value);
    if (!(length_squared > 1.0e-20f) || !isfinite(length_squared)) {
        return 0;
    }
    const float inverse_length = 1.0f / sqrtf(length_squared);
    value[0] *= inverse_length;
    value[1] *= inverse_length;
    value[2] *= inverse_length;
    return 1;
}

/* Blender's glTF importer maps glTF's Y-up frame into Blender's Z-up frame
 * before the released camera matrix is assigned. Voxelization/encoders stay
 * in direct glTF world coordinates; this conversion is render-only. */
static void gltf_world_to_blender(
    const float source[3],
    float destination[3]) {
    destination[0] = source[0];
    destination[1] = -source[2];
    destination[2] = source[1];
}

static float edge_function(
    float ax,
    float ay,
    float bx,
    float by,
    float px,
    float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

static uint8_t linear_color_to_u8(float value) {
    if (value <= 0.0f) return 0u;
    if (value >= 1.0f) return 255u;
    /* The condition PNG is consumed as sRGB. */
    const float srgb = value <= 0.0031308f ?
        12.92f * value :
        1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
    int quantized = (int) floorf(srgb * 255.0f + 0.5f);
    if (quantized < 0) quantized = 0;
    if (quantized > 255) quantized = 255;
    return (uint8_t) quantized;
}

static uint8_t unit_to_u8(float value) {
    if (!isfinite(value) || value <= 0.0f) return 0u;
    if (value >= 1.0f) return 255u;
    return (uint8_t) floorf(value * 255.0f + 0.5f);
}

static void shade_pixel(
    float normal[3],
    const float position[3],
    const trellis_mesh_segmentation_surface_sample * surface,
    uint8_t rgba[4]) {
    static const float key_direction[3] = {
        -0.467109f, -0.873900f, 0.137876f,
    };
    static const float fill_direction[3] = {
        0.642824f, -0.247240f, 0.724850f,
    };
    float view_direction[3] = {
        k_camera_position[0] - position[0],
        k_camera_position[1] - position[1],
        k_camera_position[2] - position[2],
    };
    (void) normalize3(view_direction);
    if (dot3(normal, view_direction) < 0.0f) {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
        normal[2] = -normal[2];
    }
    float key = dot3(normal, key_direction);
    float fill = dot3(normal, fill_direction);
    float top = normal[2];
    float bottom = -normal[2];
    if (key < 0.0f) key = 0.0f;
    if (fill < 0.0f) fill = 0.0f;
    if (top < 0.0f) top = 0.0f;
    if (bottom < 0.0f) bottom = 0.0f;
    float intensity = surface->unlit ? 1.0f :
        0.28f + 0.45f * key + 0.07f * fill +
        0.35f * top + 0.03f * bottom;
    if (intensity > 1.0f) intensity = 1.0f;

    rgba[0] = linear_color_to_u8(surface->base_color[0] * intensity);
    rgba[1] = linear_color_to_u8(surface->base_color[1] * intensity);
    rgba[2] = linear_color_to_u8(surface->base_color[2] * intensity);
    rgba[3] = unit_to_u8(surface->alpha);
}

static int compute_bounds(
    const trellis_mesh_rigging_asset * asset,
    float center[3],
    float * scale_out) {
    float minimum[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float maximum[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (size_t vertex = 0; vertex < asset->vertex_count; ++vertex) {
        float position[3];
        gltf_world_to_blender(asset->positions + vertex * 3u, position);
        for (int axis = 0; axis < 3; ++axis) {
            if (!isfinite(position[axis])) return 0;
            if (position[axis] < minimum[axis]) minimum[axis] = position[axis];
            if (position[axis] > maximum[axis]) maximum[axis] = position[axis];
        }
    }
    float maximum_extent = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        center[axis] = 0.5f * (minimum[axis] + maximum[axis]);
        const float extent = maximum[axis] - minimum[axis];
        if (extent > maximum_extent) maximum_extent = extent;
    }
    if (!(maximum_extent > 1.0e-12f) || !isfinite(maximum_extent)) return 0;
    *scale_out = 1.0f / maximum_extent;
    return 1;
}

static void project_vertex(
    const float source_position[3],
    const float * source_normal,
    const float center[3],
    float scale,
    int image_size,
    segmentation_render_vertex * vertex) {
    float blender_position[3];
    gltf_world_to_blender(source_position, blender_position);
    float blender_normal[3] = {0.0f, 0.0f, 0.0f};
    if (source_normal != NULL) {
        gltf_world_to_blender(source_normal, blender_normal);
    }
    for (int axis = 0; axis < 3; ++axis) {
        vertex->position[axis] =
            (blender_position[axis] - center[axis]) * scale;
        vertex->normal[axis] = source_normal != NULL ? blender_normal[axis] : 0.0f;
    }
    if (!normalize3(vertex->normal)) {
        vertex->normal[0] = 0.0f;
        vertex->normal[1] = 0.0f;
        vertex->normal[2] = 0.0f;
    }

    const float relative[3] = {
        vertex->position[0] - k_camera_position[0],
        vertex->position[1] - k_camera_position[1],
        vertex->position[2] - k_camera_position[2],
    };
    const float camera_x = dot3(relative, k_camera_right);
    const float camera_y = dot3(relative, k_camera_up);
    const float depth = -dot3(relative, k_camera_back);
    if (!(depth > 1.0e-5f) || !isfinite(depth)) {
        vertex->visible = 0;
        return;
    }

    /* tan(40 degrees / 2); the source camera uses a horizontal FOV and a
     * square target, so horizontal and vertical focal lengths are equal. */
    const float focal = 1.0f / 0.36397023426620234f;
    const float ndc_x = camera_x * focal / depth;
    const float ndc_y = camera_y * focal / depth;
    const float image_extent = (float) (image_size - 1);
    vertex->x = (0.5f * ndc_x + 0.5f) * image_extent;
    vertex->y = (0.5f - 0.5f * ndc_y) * image_extent;
    vertex->inverse_depth = 1.0f / depth;
    vertex->visible = isfinite(vertex->x) && isfinite(vertex->y);
}

static trellis_status rasterize_triangle(
    const segmentation_render_vertex * vertices,
    const uint32_t indices[3],
    size_t face,
    int image_size,
    float * depth_buffer,
    uint8_t * rgba,
    trellis_mesh_segmentation_material_sampler * material_sampler) {
    const segmentation_render_vertex * first = vertices + indices[0];
    const segmentation_render_vertex * second = vertices + indices[1];
    const segmentation_render_vertex * third = vertices + indices[2];
    if (!first->visible || !second->visible || !third->visible) {
        return TRELLIS_STATUS_OK;
    }

    const float signed_area = edge_function(
        first->x, first->y, second->x, second->y, third->x, third->y);
    if (!isfinite(signed_area) || fabsf(signed_area) <= 1.0e-8f) {
        return TRELLIS_STATUS_OK;
    }
    const float orientation = signed_area > 0.0f ? 1.0f : -1.0f;
    const float inverse_area = 1.0f / fabsf(signed_area);

    float edge_a[3] = {
        second->position[0] - first->position[0],
        second->position[1] - first->position[1],
        second->position[2] - first->position[2],
    };
    float edge_b[3] = {
        third->position[0] - first->position[0],
        third->position[1] - first->position[1],
        third->position[2] - first->position[2],
    };
    float face_normal[3];
    cross3(edge_a, edge_b, face_normal);
    if (!normalize3(face_normal)) return TRELLIS_STATUS_OK;

    float minimum_x = fminf(first->x, fminf(second->x, third->x));
    float maximum_x = fmaxf(first->x, fmaxf(second->x, third->x));
    float minimum_y = fminf(first->y, fminf(second->y, third->y));
    float maximum_y = fmaxf(first->y, fmaxf(second->y, third->y));
    if (maximum_x < 0.0f || maximum_y < 0.0f ||
        minimum_x > (float) (image_size - 1) ||
        minimum_y > (float) (image_size - 1)) {
        return TRELLIS_STATUS_OK;
    }
    if (minimum_x < 0.0f) minimum_x = 0.0f;
    if (minimum_y < 0.0f) minimum_y = 0.0f;
    if (maximum_x > (float) (image_size - 1)) maximum_x = (float) (image_size - 1);
    if (maximum_y > (float) (image_size - 1)) maximum_y = (float) (image_size - 1);

    const int x_begin = (int) floorf(minimum_x);
    const int x_end = (int) ceilf(maximum_x);
    const int y_begin = (int) floorf(minimum_y);
    const int y_end = (int) ceilf(maximum_y);
    for (int y = y_begin; y <= y_end; ++y) {
        for (int x = x_begin; x <= x_end; ++x) {
            const float sample_x = (float) x + 0.5f;
            const float sample_y = (float) y + 0.5f;
            const float lambda0 = orientation * edge_function(
                second->x, second->y, third->x, third->y, sample_x, sample_y) *
                inverse_area;
            const float lambda1 = orientation * edge_function(
                third->x, third->y, first->x, first->y, sample_x, sample_y) *
                inverse_area;
            const float lambda2 = orientation * edge_function(
                first->x, first->y, second->x, second->y, sample_x, sample_y) *
                inverse_area;
            if (lambda0 < 0.0f || lambda1 < 0.0f || lambda2 < 0.0f) continue;

            const float weight0 = lambda0 * first->inverse_depth;
            const float weight1 = lambda1 * second->inverse_depth;
            const float weight2 = lambda2 * third->inverse_depth;
            const float inverse_depth = weight0 + weight1 + weight2;
            const size_t pixel = (size_t) y * (size_t) image_size + (size_t) x;
            if (!(inverse_depth > depth_buffer[pixel])) continue;

            const float reciprocal_weight = 1.0f / inverse_depth;
            float position[3];
            float normal[3];
            for (int axis = 0; axis < 3; ++axis) {
                position[axis] =
                    (weight0 * first->position[axis] +
                     weight1 * second->position[axis] +
                     weight2 * third->position[axis]) * reciprocal_weight;
                normal[axis] =
                    (weight0 * first->normal[axis] +
                     weight1 * second->normal[axis] +
                     weight2 * third->normal[axis]) * reciprocal_weight;
            }
            if (!normalize3(normal)) {
                memcpy(normal, face_normal, sizeof(normal));
            }
            trellis_mesh_segmentation_surface_sample surface = {
                { 0.72f, 0.76f, 0.82f },
                1.0f,
                0,
            };
            if (material_sampler != NULL &&
                material_sampler->implementation != NULL) {
                const float material_barycentric[3] = {
                    weight0 * reciprocal_weight,
                    weight1 * reciprocal_weight,
                    weight2 * reciprocal_weight,
                };
                trellis_status status =
                    trellis_mesh_segmentation_material_sampler_sample(
                        material_sampler,
                        face,
                        material_barycentric,
                        &surface);
                if (status != TRELLIS_STATUS_OK) return status;
            }
            if (!(surface.alpha > 0.0f)) continue;
            shade_pixel(normal, position, &surface, rgba + pixel * 4u);
            depth_buffer[pixel] = inverse_depth;
        }
    }
    return TRELLIS_STATUS_OK;
}

static void resolve_supersampled_rgba(
    const uint8_t * source,
    int source_size,
    uint8_t * destination,
    int destination_size) {
    for (int y = 0; y < destination_size; ++y) {
        for (int x = 0; x < destination_size; ++x) {
            float alpha_sum = 0.0f;
            float premultiplied[3] = { 0.0f, 0.0f, 0.0f };
            for (int sample_y = 0; sample_y < SEGMENTATION_CONDITION_SSAA; ++sample_y) {
                for (int sample_x = 0; sample_x < SEGMENTATION_CONDITION_SSAA; ++sample_x) {
                    const int source_x = x * SEGMENTATION_CONDITION_SSAA + sample_x;
                    const int source_y = y * SEGMENTATION_CONDITION_SSAA + sample_y;
                    const uint8_t * pixel = source +
                        ((size_t) source_y * (size_t) source_size +
                         (size_t) source_x) * 4u;
                    const float alpha = (float) pixel[3] / 255.0f;
                    alpha_sum += alpha;
                    for (int channel = 0; channel < 3; ++channel) {
                        premultiplied[channel] += (float) pixel[channel] * alpha;
                    }
                }
            }
            uint8_t * output = destination +
                ((size_t) y * (size_t) destination_size + (size_t) x) * 4u;
            if (alpha_sum > 0.0f) {
                for (int channel = 0; channel < 3; ++channel) {
                    int value = (int) floorf(
                        premultiplied[channel] / alpha_sum + 0.5f);
                    if (value < 0) value = 0;
                    if (value > 255) value = 255;
                    output[channel] = (uint8_t) value;
                }
            }
            output[3] = unit_to_u8(
                alpha_sum /
                (float) (SEGMENTATION_CONDITION_SSAA *
                         SEGMENTATION_CONDITION_SSAA));
        }
    }
}

trellis_status trellis_mesh_segmentation_render_condition_rgba(
    const trellis_mesh_rigging_asset * asset,
    int image_size,
    trellis_mesh_segmentation_condition_image * image_out) {
    if (asset == NULL || image_out == NULL || asset->positions == NULL ||
        asset->triangles == NULL || asset->vertex_count == 0 ||
        asset->triangle_count == 0 || image_size < 2 ||
        image_size > INT_MAX / SEGMENTATION_CONDITION_SSAA ||
        asset->vertex_count > SIZE_MAX / (3u * sizeof(float)) ||
        asset->triangle_count > SIZE_MAX / (3u * sizeof(uint32_t))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    image_out->width = 0;
    image_out->height = 0;
    image_out->rgba = NULL;
    const size_t side = (size_t) image_size;
    if (side > SIZE_MAX / side) return TRELLIS_STATUS_OUT_OF_MEMORY;
    const size_t pixel_count = side * side;
    const size_t raster_side = side * SEGMENTATION_CONDITION_SSAA;
    if (raster_side > SIZE_MAX / raster_side ||
        raster_side > (size_t) INT_MAX) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    const size_t raster_pixel_count = raster_side * raster_side;
    if (pixel_count > SIZE_MAX / 4u ||
        raster_pixel_count > SIZE_MAX / 4u ||
        raster_pixel_count > SIZE_MAX / sizeof(float) ||
        asset->vertex_count > SIZE_MAX / sizeof(segmentation_render_vertex)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    float center[3];
    float scale = 0.0f;
    if (!compute_bounds(asset, center, &scale)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (size_t triangle = 0; triangle < asset->triangle_count; ++triangle) {
        for (int corner = 0; corner < 3; ++corner) {
            if (asset->triangles[triangle * 3u + (size_t) corner] >=
                asset->vertex_count) {
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    uint8_t * rgba = (uint8_t *) calloc(pixel_count, 4u);
    uint8_t * raster_rgba = (uint8_t *) calloc(raster_pixel_count, 4u);
    float * depth_buffer = (float *) calloc(raster_pixel_count, sizeof(float));
    segmentation_render_vertex * vertices =
        (segmentation_render_vertex *) calloc(
            asset->vertex_count,
            sizeof(*vertices));
    if (rgba == NULL || raster_rgba == NULL || depth_buffer == NULL ||
        vertices == NULL) {
        free(rgba);
        free(raster_rgba);
        free(depth_buffer);
        free(vertices);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_mesh_segmentation_material_sampler material_sampler =
        TRELLIS_MESH_SEGMENTATION_MATERIAL_SAMPLER_INIT;
    if (asset->source_path != NULL && asset->source_path[0] != '\0' &&
        asset->primitives != NULL && asset->primitive_count != 0u) {
        trellis_status status =
            trellis_mesh_segmentation_material_sampler_open(
                asset->source_path,
                asset,
                &material_sampler);
        if (status != TRELLIS_STATUS_OK) {
            free(rgba);
            free(raster_rgba);
            free(depth_buffer);
            free(vertices);
            return status;
        }
    }

    const int projection_size =
        (image_size - 1) * SEGMENTATION_CONDITION_SSAA + 1;
    for (size_t vertex = 0; vertex < asset->vertex_count; ++vertex) {
        project_vertex(
            asset->positions + vertex * 3u,
            asset->normals != NULL ? asset->normals + vertex * 3u : NULL,
            center,
            scale,
            projection_size,
            vertices + vertex);
    }
    trellis_status status = TRELLIS_STATUS_OK;
    for (size_t triangle = 0; triangle < asset->triangle_count; ++triangle) {
        status = rasterize_triangle(
            vertices,
            asset->triangles + triangle * 3u,
            triangle,
            (int) raster_side,
            depth_buffer,
            raster_rgba,
            &material_sampler);
        if (status != TRELLIS_STATUS_OK) break;
    }

    trellis_mesh_segmentation_material_sampler_close(&material_sampler);
    free(vertices);
    free(depth_buffer);
    if (status != TRELLIS_STATUS_OK) {
        free(raster_rgba);
        free(rgba);
        return status;
    }
    resolve_supersampled_rgba(
        raster_rgba,
        (int) raster_side,
        rgba,
        image_size);
    free(raster_rgba);
    trellis_mesh_segmentation_condition_image image = {
        image_size,
        image_size,
        rgba,
    };
    *image_out = image;
    return TRELLIS_STATUS_OK;
}

void trellis_mesh_segmentation_condition_image_free(
    trellis_mesh_segmentation_condition_image * image) {
    if (image == NULL) return;
    free(image->rgba);
    memset(image, 0, sizeof(*image));
}

trellis_status trellis_mesh_segmentation_render_condition(
    const trellis_mesh_rigging_asset * asset,
    int image_size,
    trellis_prepared_condition_image * prepared_out,
    const char * optional_png_path) {
    if (prepared_out == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;

    trellis_mesh_segmentation_condition_image image =
        TRELLIS_MESH_SEGMENTATION_CONDITION_IMAGE_INIT;
    trellis_status status = trellis_mesh_segmentation_render_condition_rgba(
        asset,
        image_size,
        &image);
    if (status != TRELLIS_STATUS_OK) return status;

    trellis_prepared_condition_image prepared;
    memset(&prepared, 0, sizeof(prepared));
    const int temporary = optional_png_path == NULL || optional_png_path[0] == '\0';
    char path[4096];
    if (temporary) {
        if (!trellis_make_temp_path(
                path,
                sizeof(path),
                "trellis2_segvigen_condition",
                ".png")) {
            trellis_mesh_segmentation_condition_image_free(&image);
            return TRELLIS_STATUS_IO_ERROR;
        }
    } else {
        const int copied = snprintf(path, sizeof(path), "%s", optional_png_path);
        if (copied < 0 || (size_t) copied >= sizeof(path)) {
            trellis_mesh_segmentation_condition_image_free(&image);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }

    const int wrote = stbi_write_png(
        path,
        image.width,
        image.height,
        4,
        image.rgba,
        image.width * 4);
    trellis_mesh_segmentation_condition_image_free(&image);
    if (!wrote) {
        if (temporary) trellis_unlink(path);
        return TRELLIS_STATUS_IO_ERROR;
    }

    const int source_length = snprintf(
        prepared.source_path,
        sizeof(prepared.source_path),
        "%s",
        path);
    int converted_length = 0;
    if (temporary) {
        converted_length = snprintf(
            prepared.converted_path,
            sizeof(prepared.converted_path),
            "%s",
            path);
    }
    if (source_length < 0 ||
        (size_t) source_length >= sizeof(prepared.source_path) ||
        converted_length < 0 ||
        (temporary &&
         (size_t) converted_length >= sizeof(prepared.converted_path))) {
        if (temporary) trellis_unlink(path);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *prepared_out = prepared;
    TRELLIS_INFO(
        "mesh segmentation: rendered fixed-view condition PNG: %s",
        prepared.source_path);
    return TRELLIS_STATUS_OK;
}
