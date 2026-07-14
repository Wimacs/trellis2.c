#include "material_features.h"

#include "cgltf.h"

#include <float.h>
#include <limits.h>
#include <math.h>
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
extern unsigned char * stbi_load_from_memory(
    unsigned char const * buffer,
    int len,
    int * x,
    int * y,
    int * channels_in_file,
    int desired_channels);
extern void stbi_image_free(void * retval_from_stbi_load);

#define SEGMENTATION_FDG_AABB_MIN (-0.5f)
#define MATERIAL_BVH_LEAF_SIZE 8u

typedef struct material_image {
    unsigned char * rgba;
    int width;
    int height;
    int attempted;
} material_image;

typedef struct material_source {
    cgltf_data * data;
    material_image * images;
    const char * source_path;
} material_source;

typedef struct material_sampler_implementation {
    material_source source;
    const trellis_mesh_rigging_asset * asset;
} material_sampler_implementation;

typedef struct material_triangle_ref {
    size_t face;
    float centroid[3];
    float sort_key;
} material_triangle_ref;

typedef struct material_bvh_node {
    float minimum[3];
    float maximum[3];
    size_t start;
    size_t count;
    size_t left;
    size_t right;
} material_bvh_node;

typedef struct material_bvh {
    const trellis_mesh_host * mesh;
    material_triangle_ref * refs;
    material_bvh_node * nodes;
    size_t node_count;
} material_bvh;

typedef struct material_nearest {
    size_t face;
    float distance_squared;
    float barycentric[3];
} material_nearest;

static float clamp_unit(float value) {
    if (!isfinite(value) || value <= 0.0f) return 0.0f;
    if (value >= 1.0f) return 1.0f;
    return value;
}

static float srgb_to_linear(float value) {
    value = clamp_unit(value);
    return value <= 0.04045f ?
        value / 12.92f :
        powf((value + 0.055f) / 1.055f, 2.4f);
}

static trellis_status status_from_cgltf(cgltf_result result) {
    switch (result) {
        case cgltf_result_success: return TRELLIS_STATUS_OK;
        case cgltf_result_file_not_found: return TRELLIS_STATUS_NOT_FOUND;
        case cgltf_result_io_error: return TRELLIS_STATUS_IO_ERROR;
        case cgltf_result_out_of_memory: return TRELLIS_STATUS_OUT_OF_MEMORY;
        default: return TRELLIS_STATUS_PARSE_ERROR;
    }
}

static int add_size(size_t a, size_t b, size_t * result) {
    if (result == NULL || a > SIZE_MAX - b) return 0;
    *result = a + b;
    return 1;
}

static int multiply_size(size_t a, size_t b, size_t * result) {
    if (result == NULL || (a != 0u && b > SIZE_MAX / a)) return 0;
    *result = a * b;
    return 1;
}

static int is_absolute_path(const char * path) {
    if (path == NULL || path[0] == '\0') return 0;
    if (path[0] == '/' || path[0] == '\\') return 1;
    return path[1] == ':' &&
        ((path[0] >= 'A' && path[0] <= 'Z') ||
         (path[0] >= 'a' && path[0] <= 'z'));
}

static char * external_image_path(const char * source_path, const char * uri) {
    if (source_path == NULL || uri == NULL || uri[0] == '\0' ||
        strstr(uri, "://") != NULL) {
        return NULL;
    }
    const size_t uri_length = strlen(uri);
    size_t prefix_length = 0u;
    if (!is_absolute_path(uri)) {
        const char * slash = strrchr(source_path, '/');
        const char * backslash = strrchr(source_path, '\\');
        const char * separator = slash;
        if (backslash != NULL && (separator == NULL || backslash > separator)) {
            separator = backslash;
        }
        prefix_length = separator != NULL ? (size_t) (separator - source_path + 1) : 0u;
    }
    size_t total = 0u;
    if (!add_size(prefix_length, uri_length, &total) ||
        !add_size(total, 1u, &total)) {
        return NULL;
    }
    char * path = (char *) malloc(total);
    if (path == NULL) return NULL;
    if (prefix_length != 0u) memcpy(path, source_path, prefix_length);
    memcpy(path + prefix_length, uri, uri_length + 1u);
    cgltf_decode_uri(path + prefix_length);
    return path;
}

static int base64_value(unsigned char value) {
    if (value >= 'A' && value <= 'Z') return (int) value - 'A';
    if (value >= 'a' && value <= 'z') return (int) value - 'a' + 26;
    if (value >= '0' && value <= '9') return (int) value - '0' + 52;
    if (value == '+') return 62;
    if (value == '/') return 63;
    return -1;
}

static unsigned char * decode_base64(
    const char * text,
    size_t * byte_count_out) {
    if (text == NULL || byte_count_out == NULL) return NULL;
    *byte_count_out = 0u;
    const size_t length = strlen(text);
    if (length > SIZE_MAX - 3u) return NULL;
    const size_t groups = (length + 3u) / 4u;
    if (groups > SIZE_MAX / 3u) return NULL;
    const size_t capacity = groups * 3u;
    unsigned char * output = (unsigned char *) malloc(capacity == 0u ? 1u : capacity);
    if (output == NULL) return NULL;
    uint32_t accumulator = 0u;
    unsigned bits = 0u;
    size_t written = 0u;
    for (size_t index = 0u; index < length; ++index) {
        const unsigned char character = (unsigned char) text[index];
        if (character == '=') break;
        if (character == ' ' || character == '\t' || character == '\r' || character == '\n') {
            continue;
        }
        const int value = base64_value(character);
        if (value < 0) {
            free(output);
            return NULL;
        }
        accumulator = (accumulator << 6u) | (uint32_t) value;
        bits += 6u;
        if (bits >= 8u) {
            bits -= 8u;
            if (written >= capacity) {
                free(output);
                return NULL;
            }
            output[written++] = (unsigned char) (accumulator >> bits);
            if (bits == 0u) {
                accumulator = 0u;
            } else {
                accumulator &= (UINT32_C(1) << bits) - UINT32_C(1);
            }
        }
    }
    *byte_count_out = written;
    return output;
}

static unsigned char * load_material_image(
    material_source * source,
    const cgltf_image * image,
    int * width_out,
    int * height_out) {
    if (source == NULL || source->data == NULL || source->images == NULL ||
        image == NULL || width_out == NULL || height_out == NULL ||
        image < source->data->images ||
        image >= source->data->images + source->data->images_count) {
        return NULL;
    }
    const size_t image_index = (size_t) (image - source->data->images);
    material_image * cached = &source->images[image_index];
    if (cached->attempted) {
        *width_out = cached->width;
        *height_out = cached->height;
        return cached->rgba;
    }
    cached->attempted = 1;
    int source_channels = 0;
    if (image->buffer_view != NULL) {
        const cgltf_buffer_view * view = image->buffer_view;
        const unsigned char * bytes = NULL;
        if (view->data != NULL) {
            bytes = (const unsigned char *) view->data;
        } else if (view->buffer != NULL && view->buffer->data != NULL &&
                   view->offset <= view->buffer->size &&
                   view->size <= view->buffer->size - view->offset) {
            bytes = (const unsigned char *) view->buffer->data + view->offset;
        }
        if (bytes != NULL && view->size <= (cgltf_size) INT_MAX) {
            cached->rgba = stbi_load_from_memory(
                bytes,
                (int) view->size,
                &cached->width,
                &cached->height,
                &source_channels,
                4);
        }
    } else if (image->uri != NULL && strncmp(image->uri, "data:", 5u) == 0) {
        const char * comma = strchr(image->uri, ',');
        if (comma != NULL && comma - image->uri >= 7 &&
            strncmp(comma - 7, ";base64", 7u) == 0) {
            size_t byte_count = 0u;
            unsigned char * bytes = decode_base64(comma + 1, &byte_count);
            if (bytes != NULL && byte_count <= (size_t) INT_MAX) {
                cached->rgba = stbi_load_from_memory(
                    bytes,
                    (int) byte_count,
                    &cached->width,
                    &cached->height,
                    &source_channels,
                    4);
            }
            free(bytes);
        }
    } else if (image->uri != NULL) {
        char * path = external_image_path(source->source_path, image->uri);
        if (path != NULL) {
            cached->rgba = stbi_load(
                path,
                &cached->width,
                &cached->height,
                &source_channels,
                4);
        }
        free(path);
    }
    if (cached->rgba == NULL || cached->width <= 0 || cached->height <= 0) {
        stbi_image_free(cached->rgba);
        cached->rgba = NULL;
        cached->width = 0;
        cached->height = 0;
        TRELLIS_ERROR("mesh segmentation: failed to decode material image %zu", image_index);
        return NULL;
    }
    *width_out = cached->width;
    *height_out = cached->height;
    return cached->rgba;
}

static float wrap_coordinate(float value, cgltf_wrap_mode mode) {
    if (mode == cgltf_wrap_mode_clamp_to_edge) {
        return clamp_unit(value);
    }
    if (mode == cgltf_wrap_mode_mirrored_repeat) {
        float wrapped = fmodf(value, 2.0f);
        if (wrapped < 0.0f) wrapped += 2.0f;
        return wrapped <= 1.0f ? wrapped : 2.0f - wrapped;
    }
    return value - floorf(value);
}

static int wrap_texel(int value, int extent, cgltf_wrap_mode mode) {
    if (extent <= 1) return 0;
    if (mode == cgltf_wrap_mode_clamp_to_edge) {
        if (value < 0) return 0;
        if (value >= extent) return extent - 1;
        return value;
    }
    if (mode == cgltf_wrap_mode_mirrored_repeat) {
        const int period = extent * 2;
        int wrapped = value % period;
        if (wrapped < 0) wrapped += period;
        return wrapped < extent ? wrapped : period - 1 - wrapped;
    }
    int wrapped = value % extent;
    if (wrapped < 0) wrapped += extent;
    return wrapped;
}

static void read_texel(
    const unsigned char * rgba,
    int width,
    int x,
    int y,
    float output[4]) {
    const unsigned char * pixel = rgba + ((size_t) y * (size_t) width + (size_t) x) * 4u;
    for (int channel = 0; channel < 4; ++channel) {
        output[channel] = (float) pixel[channel] / 255.0f;
    }
}

static void sample_image(
    const unsigned char * rgba,
    int width,
    int height,
    float u,
    float v,
    cgltf_wrap_mode wrap_s,
    cgltf_wrap_mode wrap_t,
    int nearest,
    float output[4]) {
    u = wrap_coordinate(u, wrap_s);
    v = wrap_coordinate(v, wrap_t);
    if (nearest) {
        int x = (int) floorf(u * (float) width);
        int y = (int) floorf(v * (float) height);
        x = wrap_texel(x, width, wrap_s);
        y = wrap_texel(y, height, wrap_t);
        read_texel(rgba, width, x, y, output);
        return;
    }
    const float x = u * (float) width - 0.5f;
    const float y = v * (float) height - 0.5f;
    const int x0_raw = (int) floorf(x);
    const int y0_raw = (int) floorf(y);
    const float tx = x - floorf(x);
    const float ty = y - floorf(y);
    const int x0 = wrap_texel(x0_raw, width, wrap_s);
    const int x1 = wrap_texel(x0_raw + 1, width, wrap_s);
    const int y0 = wrap_texel(y0_raw, height, wrap_t);
    const int y1 = wrap_texel(y0_raw + 1, height, wrap_t);
    float p00[4], p10[4], p01[4], p11[4];
    read_texel(rgba, width, x0, y0, p00);
    read_texel(rgba, width, x1, y0, p10);
    read_texel(rgba, width, x0, y1, p01);
    read_texel(rgba, width, x1, y1, p11);
    for (int channel = 0; channel < 4; ++channel) {
        const float top = p00[channel] + (p10[channel] - p00[channel]) * tx;
        const float bottom = p01[channel] + (p11[channel] - p01[channel]) * tx;
        output[channel] = top + (bottom - top) * ty;
    }
}

static int sample_texture_view(
    material_source * source,
    const cgltf_texture_view * view,
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_rigging_primitive_range * range,
    const int32_t indices[3],
    const float barycentric[3],
    float output[4]) {
    if (view == NULL || view->texture == NULL || source == NULL ||
        asset == NULL || range == NULL || indices == NULL ||
        barycentric == NULL || output == NULL) {
        return 0;
    }
    cgltf_int texcoord = view->texcoord;
    if (view->has_transform && view->transform.has_texcoord) {
        texcoord = view->transform.texcoord;
    }
    if (texcoord < 0 || texcoord >= 64 ||
        (size_t) texcoord >= asset->texcoord_set_count ||
        asset->texcoords == NULL || asset->texcoords[texcoord] == NULL ||
        (range->texcoord_mask & (UINT64_C(1) << (unsigned) texcoord)) == 0u) {
        TRELLIS_ERROR("mesh segmentation: material texture requires missing TEXCOORD_%d", texcoord);
        return 0;
    }
    float uv[2] = { 0.0f, 0.0f };
    for (int corner = 0; corner < 3; ++corner) {
        if (indices[corner] < 0 || (size_t) indices[corner] >= asset->vertex_count) return 0;
        const float * source_uv = asset->texcoords[texcoord] + (size_t) indices[corner] * 2u;
        uv[0] += barycentric[corner] * source_uv[0];
        uv[1] += barycentric[corner] * source_uv[1];
    }
    if (view->has_transform) {
        const float scaled_u = uv[0] * view->transform.scale[0];
        const float scaled_v = uv[1] * view->transform.scale[1];
        const float cosine = cosf(view->transform.rotation);
        const float sine = sinf(view->transform.rotation);
        uv[0] = view->transform.offset[0] + cosine * scaled_u - sine * scaled_v;
        uv[1] = view->transform.offset[1] + sine * scaled_u + cosine * scaled_v;
    }
    const cgltf_texture * texture = view->texture;
    const cgltf_image * image = texture->image;
    if (image == NULL && texture->has_webp) image = texture->webp_image;
    if (image == NULL) {
        TRELLIS_ERROR("mesh segmentation: unsupported or missing material texture image");
        return 0;
    }
    int width = 0;
    int height = 0;
    const unsigned char * rgba = load_material_image(source, image, &width, &height);
    if (rgba == NULL) return 0;
    cgltf_wrap_mode wrap_s = cgltf_wrap_mode_repeat;
    cgltf_wrap_mode wrap_t = cgltf_wrap_mode_repeat;
    int nearest = 0;
    if (texture->sampler != NULL) {
        wrap_s = texture->sampler->wrap_s;
        wrap_t = texture->sampler->wrap_t;
        nearest = texture->sampler->mag_filter == cgltf_filter_type_nearest ||
            texture->sampler->min_filter == cgltf_filter_type_nearest ||
            texture->sampler->min_filter == cgltf_filter_type_nearest_mipmap_nearest ||
            texture->sampler->min_filter == cgltf_filter_type_nearest_mipmap_linear;
    }
    sample_image(rgba, width, height, uv[0], uv[1], wrap_s, wrap_t, nearest, output);
    return 1;
}

static void closest_degenerate_triangle(
    const float point[3],
    const float * a,
    const float * b,
    const float * c,
    float barycentric[3],
    float * distance_squared_out) {
    const float * vertices[3] = { a, b, c };
    float best = FLT_MAX;
    float best_barycentric[3] = { 1.0f, 0.0f, 0.0f };
    for (int start = 0; start < 3; ++start) {
        const int end = (start + 1) % 3;
        float direction[3];
        float length_squared = 0.0f;
        float projection = 0.0f;
        for (int axis = 0; axis < 3; ++axis) {
            direction[axis] = vertices[end][axis] - vertices[start][axis];
            length_squared += direction[axis] * direction[axis];
            projection += (point[axis] - vertices[start][axis]) * direction[axis];
        }
        float t = length_squared > 1e-30f ? projection / length_squared : 0.0f;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        float distance_squared = 0.0f;
        for (int axis = 0; axis < 3; ++axis) {
            const float closest = vertices[start][axis] + direction[axis] * t;
            const float delta = point[axis] - closest;
            distance_squared += delta * delta;
        }
        if (distance_squared < best) {
            best = distance_squared;
            best_barycentric[0] = 0.0f;
            best_barycentric[1] = 0.0f;
            best_barycentric[2] = 0.0f;
            best_barycentric[start] = 1.0f - t;
            best_barycentric[end] = t;
        }
    }
    memcpy(barycentric, best_barycentric, sizeof(best_barycentric));
    *distance_squared_out = best;
}

static void closest_triangle(
    const float point[3],
    const float * a,
    const float * b,
    const float * c,
    float barycentric[3],
    float * distance_squared_out) {
    const float ab[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
    const float ac[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
    const float normal[3] = {
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    };
    const float normal_squared =
        normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if (!(normal_squared > 1e-30f) || !isfinite(normal_squared)) {
        closest_degenerate_triangle(point, a, b, c, barycentric, distance_squared_out);
        return;
    }
    const float ap[3] = { point[0] - a[0], point[1] - a[1], point[2] - a[2] };
    const float d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2];
    const float d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2];
    if (d1 <= 0.0f && d2 <= 0.0f) {
        barycentric[0] = 1.0f; barycentric[1] = 0.0f; barycentric[2] = 0.0f;
    } else {
        const float bp[3] = { point[0] - b[0], point[1] - b[1], point[2] - b[2] };
        const float d3 = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2];
        const float d4 = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2];
        if (d3 >= 0.0f && d4 <= d3) {
            barycentric[0] = 0.0f; barycentric[1] = 1.0f; barycentric[2] = 0.0f;
        } else {
            const float vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                const float v = d1 / (d1 - d3);
                barycentric[0] = 1.0f - v; barycentric[1] = v; barycentric[2] = 0.0f;
            } else {
                const float cp[3] = { point[0] - c[0], point[1] - c[1], point[2] - c[2] };
                const float d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2];
                const float d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2];
                if (d6 >= 0.0f && d5 <= d6) {
                    barycentric[0] = 0.0f; barycentric[1] = 0.0f; barycentric[2] = 1.0f;
                } else {
                    const float vb = d5 * d2 - d1 * d6;
                    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                        const float w = d2 / (d2 - d6);
                        barycentric[0] = 1.0f - w; barycentric[1] = 0.0f; barycentric[2] = w;
                    } else {
                        const float va = d3 * d6 - d5 * d4;
                        if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f) {
                            const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            barycentric[0] = 0.0f; barycentric[1] = 1.0f - w; barycentric[2] = w;
                        } else {
                            const float denominator = 1.0f / (va + vb + vc);
                            const float v = vb * denominator;
                            const float w = vc * denominator;
                            barycentric[0] = 1.0f - v - w;
                            barycentric[1] = v;
                            barycentric[2] = w;
                        }
                    }
                }
            }
        }
    }
    float distance_squared = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        const float closest = barycentric[0] * a[axis] +
            barycentric[1] * b[axis] + barycentric[2] * c[axis];
        const float delta = point[axis] - closest;
        distance_squared += delta * delta;
    }
    *distance_squared_out = distance_squared;
}

static int compare_triangle_ref(const void * left, const void * right) {
    const material_triangle_ref * a = (const material_triangle_ref *) left;
    const material_triangle_ref * b = (const material_triangle_ref *) right;
    if (a->sort_key < b->sort_key) return -1;
    if (a->sort_key > b->sort_key) return 1;
    if (a->face < b->face) return -1;
    if (a->face > b->face) return 1;
    return 0;
}

static size_t bvh_build_node(
    material_bvh * bvh,
    size_t start,
    size_t count) {
    const size_t node_index = bvh->node_count++;
    material_bvh_node * node = &bvh->nodes[node_index];
    node->start = start;
    node->count = count;
    node->left = SIZE_MAX;
    node->right = SIZE_MAX;
    for (int axis = 0; axis < 3; ++axis) {
        node->minimum[axis] = FLT_MAX;
        node->maximum[axis] = -FLT_MAX;
    }
    for (size_t offset = 0u; offset < count; ++offset) {
        const size_t face = bvh->refs[start + offset].face;
        const int32_t * indices = bvh->mesh->faces + face * 3u;
        for (int corner = 0; corner < 3; ++corner) {
            const float * vertex = bvh->mesh->vertices + (size_t) indices[corner] * 3u;
            for (int axis = 0; axis < 3; ++axis) {
                if (vertex[axis] < node->minimum[axis]) node->minimum[axis] = vertex[axis];
                if (vertex[axis] > node->maximum[axis]) node->maximum[axis] = vertex[axis];
            }
        }
    }
    if (count <= MATERIAL_BVH_LEAF_SIZE) return node_index;
    int split_axis = 0;
    float widest = -1.0f;
    for (int axis = 0; axis < 3; ++axis) {
        float minimum = FLT_MAX;
        float maximum = -FLT_MAX;
        for (size_t offset = 0u; offset < count; ++offset) {
            const float value = bvh->refs[start + offset].centroid[axis];
            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;
        }
        if (maximum - minimum > widest) {
            widest = maximum - minimum;
            split_axis = axis;
        }
    }
    for (size_t offset = 0u; offset < count; ++offset) {
        bvh->refs[start + offset].sort_key =
            bvh->refs[start + offset].centroid[split_axis];
    }
    qsort(bvh->refs + start, count, sizeof(*bvh->refs), compare_triangle_ref);
    const size_t left_count = count / 2u;
    node->left = bvh_build_node(bvh, start, left_count);
    node = &bvh->nodes[node_index];
    node->right = bvh_build_node(bvh, start + left_count, count - left_count);
    return node_index;
}

static void bvh_free(material_bvh * bvh) {
    if (bvh == NULL) return;
    free(bvh->refs);
    free(bvh->nodes);
    memset(bvh, 0, sizeof(*bvh));
}

static trellis_status bvh_build(
    const trellis_mesh_host * mesh,
    material_bvh * bvh) {
    if (mesh == NULL || bvh == NULL || mesh->n_faces <= 0 ||
        mesh->faces == NULL || mesh->vertices == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(bvh, 0, sizeof(*bvh));
    const size_t face_count = (size_t) mesh->n_faces;
    size_t node_capacity = 0u;
    size_t ref_bytes = 0u;
    size_t node_bytes = 0u;
    if (!multiply_size(face_count, 2u, &node_capacity) ||
        !multiply_size(face_count, sizeof(*bvh->refs), &ref_bytes) ||
        !multiply_size(node_capacity, sizeof(*bvh->nodes), &node_bytes)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    bvh->refs = (material_triangle_ref *) malloc(ref_bytes);
    bvh->nodes = (material_bvh_node *) malloc(node_bytes);
    bvh->mesh = mesh;
    if (bvh->refs == NULL || bvh->nodes == NULL) {
        bvh_free(bvh);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (size_t face = 0u; face < face_count; ++face) {
        material_triangle_ref * ref = &bvh->refs[face];
        const int32_t * indices = mesh->faces + face * 3u;
        ref->face = face;
        ref->sort_key = 0.0f;
        for (int axis = 0; axis < 3; ++axis) {
            ref->centroid[axis] = (
                mesh->vertices[(size_t) indices[0] * 3u + (size_t) axis] +
                mesh->vertices[(size_t) indices[1] * 3u + (size_t) axis] +
                mesh->vertices[(size_t) indices[2] * 3u + (size_t) axis]) / 3.0f;
        }
    }
    (void) bvh_build_node(bvh, 0u, face_count);
    return TRELLIS_STATUS_OK;
}

static float point_box_distance_squared(
    const float point[3],
    const material_bvh_node * node) {
    float result = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        float delta = 0.0f;
        if (point[axis] < node->minimum[axis]) delta = node->minimum[axis] - point[axis];
        else if (point[axis] > node->maximum[axis]) delta = point[axis] - node->maximum[axis];
        result += delta * delta;
    }
    return result;
}

static void bvh_query_node(
    const material_bvh * bvh,
    size_t node_index,
    const float point[3],
    material_nearest * nearest) {
    const material_bvh_node * node = &bvh->nodes[node_index];
    if (point_box_distance_squared(point, node) > nearest->distance_squared) return;
    if (node->left == SIZE_MAX) {
        for (size_t offset = 0u; offset < node->count; ++offset) {
            const size_t face = bvh->refs[node->start + offset].face;
            const int32_t * indices = bvh->mesh->faces + face * 3u;
            const float * a = bvh->mesh->vertices + (size_t) indices[0] * 3u;
            const float * b = bvh->mesh->vertices + (size_t) indices[1] * 3u;
            const float * c = bvh->mesh->vertices + (size_t) indices[2] * 3u;
            float barycentric[3];
            float distance_squared = 0.0f;
            closest_triangle(point, a, b, c, barycentric, &distance_squared);
            if (distance_squared < nearest->distance_squared ||
                (distance_squared == nearest->distance_squared && face < nearest->face)) {
                nearest->face = face;
                nearest->distance_squared = distance_squared;
                memcpy(nearest->barycentric, barycentric, sizeof(barycentric));
            }
        }
        return;
    }
    const float left_distance = point_box_distance_squared(point, &bvh->nodes[node->left]);
    const float right_distance = point_box_distance_squared(point, &bvh->nodes[node->right]);
    if (left_distance <= right_distance) {
        bvh_query_node(bvh, node->left, point, nearest);
        bvh_query_node(bvh, node->right, point, nearest);
    } else {
        bvh_query_node(bvh, node->right, point, nearest);
        bvh_query_node(bvh, node->left, point, nearest);
    }
}

static const trellis_mesh_rigging_primitive_range * range_for_face(
    const trellis_mesh_rigging_asset * asset,
    size_t face) {
    size_t low = 0u;
    size_t high = asset->primitive_count;
    while (low < high) {
        const size_t middle = low + (high - low) / 2u;
        const trellis_mesh_rigging_primitive_range * range = &asset->primitives[middle];
        if (face < range->first_triangle) {
            high = middle;
        } else if (face >= range->first_triangle + range->triangle_count) {
            low = middle + 1u;
        } else {
            return range;
        }
    }
    return NULL;
}

static void material_source_release(material_source * source) {
    if (source == NULL) return;
    if (source->images != NULL && source->data != NULL) {
        for (size_t image = 0u;
             image < (size_t) source->data->images_count;
             ++image) {
            stbi_image_free(source->images[image].rgba);
        }
    }
    free(source->images);
    cgltf_free(source->data);
    memset(source, 0, sizeof(*source));
}

static trellis_status validate_material_asset_ranges(
    const trellis_mesh_rigging_asset * asset,
    const cgltf_data * data) {
    if (asset == NULL || data == NULL || asset->positions == NULL ||
        asset->triangles == NULL || asset->primitives == NULL ||
        asset->vertex_count == 0u || asset->vertex_count > (size_t) INT32_MAX ||
        asset->triangle_count == 0u || asset->primitive_count == 0u) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    size_t triangle_cursor = 0u;
    for (size_t primitive = 0u; primitive < asset->primitive_count; ++primitive) {
        const trellis_mesh_rigging_primitive_range * range =
            &asset->primitives[primitive];
        if (range->first_triangle != triangle_cursor ||
            range->triangle_count == 0u ||
            !add_size(triangle_cursor, range->triangle_count, &triangle_cursor) ||
            triangle_cursor > asset->triangle_count ||
            range->source_node_index >= (size_t) data->nodes_count ||
            range->source_mesh_index >= (size_t) data->meshes_count ||
            (range->source_material_index != SIZE_MAX &&
             range->source_material_index >= (size_t) data->materials_count)) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        const cgltf_node * node = &data->nodes[range->source_node_index];
        const cgltf_mesh * mesh = &data->meshes[range->source_mesh_index];
        if (node->mesh != mesh ||
            range->source_primitive_index >= (size_t) mesh->primitives_count) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        const cgltf_primitive * source_primitive =
            &mesh->primitives[range->source_primitive_index];
        const size_t source_material = source_primitive->material != NULL ?
            (size_t) (source_primitive->material - data->materials) : SIZE_MAX;
        if (source_material != range->source_material_index) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    if (triangle_cursor != asset->triangle_count) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    for (size_t face = 0u; face < asset->triangle_count; ++face) {
        for (int corner = 0; corner < 3; ++corner) {
            if (asset->triangles[face * 3u + (size_t) corner] >=
                asset->vertex_count) {
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_mesh_segmentation_material_sampler_open(
    const char * source_path,
    const trellis_mesh_rigging_asset * asset,
    trellis_mesh_segmentation_material_sampler * sampler_out) {
    if (sampler_out == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;
    sampler_out->implementation = NULL;
    if (source_path == NULL || source_path[0] == '\0' || asset == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    cgltf_options options;
    memset(&options, 0, sizeof(options));
    cgltf_data * data = NULL;
    cgltf_result result = cgltf_parse_file(&options, source_path, &data);
    if (result != cgltf_result_success || data == NULL) {
        cgltf_free(data);
        return status_from_cgltf(result);
    }
    result = cgltf_load_buffers(&options, data, source_path);
    if (result == cgltf_result_success) result = cgltf_validate(data);
    if (result != cgltf_result_success) {
        trellis_status status = status_from_cgltf(result);
        cgltf_free(data);
        return status;
    }
    trellis_status status = validate_material_asset_ranges(asset, data);
    if (status != TRELLIS_STATUS_OK) {
        cgltf_free(data);
        return status;
    }

    material_sampler_implementation * implementation =
        (material_sampler_implementation *) calloc(1u, sizeof(*implementation));
    if (implementation == NULL) {
        cgltf_free(data);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    implementation->source.data = data;
    implementation->source.source_path = source_path;
    implementation->asset = asset;
    if (data->images_count != 0u) {
        implementation->source.images = (material_image *) calloc(
            (size_t) data->images_count,
            sizeof(*implementation->source.images));
        if (implementation->source.images == NULL) {
            material_source_release(&implementation->source);
            free(implementation);
            return TRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    sampler_out->implementation = implementation;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_mesh_segmentation_material_sampler_sample(
    trellis_mesh_segmentation_material_sampler * sampler,
    size_t face,
    const float barycentric[3],
    trellis_mesh_segmentation_surface_sample * sample_out) {
    if (sampler == NULL || sampler->implementation == NULL ||
        barycentric == NULL || sample_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    material_sampler_implementation * implementation =
        (material_sampler_implementation *) sampler->implementation;
    const trellis_mesh_rigging_asset * asset = implementation->asset;
    if (face >= asset->triangle_count) return TRELLIS_STATUS_INVALID_ARGUMENT;
    float barycentric_sum = 0.0f;
    for (int corner = 0; corner < 3; ++corner) {
        if (!isfinite(barycentric[corner]) || barycentric[corner] < -1.0e-4f ||
            barycentric[corner] > 1.0001f) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        barycentric_sum += barycentric[corner];
    }
    if (!isfinite(barycentric_sum) || fabsf(barycentric_sum - 1.0f) > 1.0e-3f) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const trellis_mesh_rigging_primitive_range * range =
        range_for_face(asset, face);
    if (range == NULL) return TRELLIS_STATUS_PARSE_ERROR;

    const cgltf_material * material = NULL;
    if (range->source_material_index != SIZE_MAX) {
        material = &implementation->source.data->materials[
            range->source_material_index];
    }
    float linear_rgb[3] = { 1.0f, 1.0f, 1.0f };
    float alpha = 1.0f;
    int unlit = 0;
    if (material != NULL) {
        const cgltf_pbr_metallic_roughness * pbr =
            &material->pbr_metallic_roughness;
        for (int channel = 0; channel < 3; ++channel) {
            linear_rgb[channel] = clamp_unit(pbr->base_color_factor[channel]);
        }
        alpha = clamp_unit(pbr->base_color_factor[3]);
        unlit = material->unlit != 0;
        if (pbr->base_color_texture.texture != NULL) {
            int32_t indices[3];
            for (int corner = 0; corner < 3; ++corner) {
                indices[corner] = (int32_t) asset->triangles[
                    face * 3u + (size_t) corner];
            }
            float texture_sample[4];
            if (!sample_texture_view(
                    &implementation->source,
                    &pbr->base_color_texture,
                    asset,
                    range,
                    indices,
                    barycentric,
                    texture_sample)) {
                return TRELLIS_STATUS_PARSE_ERROR;
            }
            for (int channel = 0; channel < 3; ++channel) {
                linear_rgb[channel] *= srgb_to_linear(texture_sample[channel]);
            }
            alpha *= clamp_unit(texture_sample[3]);
        }
        alpha = clamp_unit(alpha);
        if (material->alpha_mode == cgltf_alpha_mode_opaque) {
            alpha = 1.0f;
        } else if (material->alpha_mode == cgltf_alpha_mode_mask) {
            alpha = alpha >= material->alpha_cutoff ? 1.0f : 0.0f;
        }
    }
    for (int channel = 0; channel < 3; ++channel) {
        sample_out->base_color[channel] = clamp_unit(linear_rgb[channel]);
    }
    sample_out->alpha = clamp_unit(alpha);
    sample_out->unlit = unlit;
    return TRELLIS_STATUS_OK;
}

void trellis_mesh_segmentation_material_sampler_close(
    trellis_mesh_segmentation_material_sampler * sampler) {
    if (sampler == NULL) return;
    material_sampler_implementation * implementation =
        (material_sampler_implementation *) sampler->implementation;
    if (implementation != NULL) {
        material_source_release(&implementation->source);
        free(implementation);
    }
    sampler->implementation = NULL;
}

static trellis_status validate_inputs(
    const char * source_path,
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_host * mesh,
    const trellis_flexible_dual_grid * grid,
    float ** features_out) {
    if (features_out == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;
    *features_out = NULL;
    if (source_path == NULL || source_path[0] == '\0' || asset == NULL ||
        mesh == NULL || grid == NULL || asset->positions == NULL ||
        asset->triangles == NULL || asset->primitives == NULL ||
        mesh->vertices == NULL || mesh->faces == NULL ||
        grid->coords == NULL || grid->dual_vertices == NULL ||
        asset->vertex_count == 0u || asset->triangle_count == 0u ||
        asset->primitive_count == 0u || mesh->n_vertices <= 0 ||
        mesh->n_faces <= 0 || grid->n <= 0 ||
        (uint64_t) mesh->n_vertices != (uint64_t) asset->vertex_count ||
        (uint64_t) mesh->n_faces != (uint64_t) asset->triangle_count ||
        asset->vertex_count > SIZE_MAX / 3u ||
        asset->triangle_count > SIZE_MAX / 3u ||
        (uint64_t) grid->n > SIZE_MAX / 4u ||
        (uint64_t) grid->n > SIZE_MAX / 3u) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    size_t triangle_cursor = 0u;
    for (size_t primitive = 0u; primitive < asset->primitive_count; ++primitive) {
        const trellis_mesh_rigging_primitive_range * range = &asset->primitives[primitive];
        if (range->first_triangle != triangle_cursor || range->triangle_count == 0u ||
            !add_size(triangle_cursor, range->triangle_count, &triangle_cursor) ||
            triangle_cursor > asset->triangle_count) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }
    if (triangle_cursor != asset->triangle_count) return TRELLIS_STATUS_INVALID_ARGUMENT;
    for (size_t face = 0u; face < asset->triangle_count; ++face) {
        for (int corner = 0; corner < 3; ++corner) {
            const int32_t index = mesh->faces[face * 3u + (size_t) corner];
            if (index < 0 || (size_t) index >= asset->vertex_count ||
                asset->triangles[face * 3u + (size_t) corner] != (uint32_t) index) {
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    size_t position_values = 0u;
    if (!multiply_size(asset->vertex_count, 3u, &position_values)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (size_t vertex = 0u; vertex < position_values; ++vertex) {
        if (!isfinite(mesh->vertices[vertex])) return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int64_t row = 0; row < grid->n; ++row) {
        if (grid->coords[(size_t) row * 4u] != 0) return TRELLIS_STATUS_INVALID_ARGUMENT;
        for (int axis = 0; axis < 3; ++axis) {
            if (!isfinite(grid->dual_vertices[(size_t) row * 3u + (size_t) axis])) {
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_mesh_segmentation_material_features(
    const char * source_path,
    const trellis_mesh_rigging_asset * asset,
    const trellis_mesh_host * normalized_mesh,
    const trellis_flexible_dual_grid * grid,
    float ** features_out) {
    trellis_status status = validate_inputs(
        source_path, asset, normalized_mesh, grid, features_out);
    if (status != TRELLIS_STATUS_OK) return status;

    cgltf_options options;
    memset(&options, 0, sizeof(options));
    cgltf_data * data = NULL;
    cgltf_result result = cgltf_parse_file(&options, source_path, &data);
    if (result != cgltf_result_success || data == NULL) {
        cgltf_free(data);
        return status_from_cgltf(result);
    }
    result = cgltf_load_buffers(&options, data, source_path);
    if (result == cgltf_result_success) result = cgltf_validate(data);
    if (result != cgltf_result_success) {
        status = status_from_cgltf(result);
        cgltf_free(data);
        return status;
    }
    for (size_t primitive = 0u; primitive < asset->primitive_count; ++primitive) {
        const trellis_mesh_rigging_primitive_range * range = &asset->primitives[primitive];
        const size_t material = range->source_material_index;
        if (range->source_node_index >= (size_t) data->nodes_count ||
            range->source_mesh_index >= (size_t) data->meshes_count ||
            (material != SIZE_MAX && material >= (size_t) data->materials_count)) {
            cgltf_free(data);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        const cgltf_node * node = &data->nodes[range->source_node_index];
        const cgltf_mesh * mesh = &data->meshes[range->source_mesh_index];
        if (node->mesh != mesh ||
            range->source_primitive_index >= (size_t) mesh->primitives_count) {
            cgltf_free(data);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        const cgltf_primitive * source_primitive =
            &mesh->primitives[range->source_primitive_index];
        const size_t source_material = source_primitive->material != NULL ?
            (size_t) (source_primitive->material - data->materials) : SIZE_MAX;
        if (source_material != material) {
            cgltf_free(data);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }

    material_source source;
    memset(&source, 0, sizeof(source));
    source.data = data;
    source.source_path = source_path;
    if (data->images_count != 0u) {
        source.images = (material_image *) calloc(
            (size_t) data->images_count, sizeof(*source.images));
        if (source.images == NULL) {
            cgltf_free(data);
            return TRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    material_bvh bvh;
    status = bvh_build(normalized_mesh, &bvh);
    if (status != TRELLIS_STATUS_OK) goto cleanup;

    size_t feature_values = 0u;
    if ((uint64_t) grid->n > SIZE_MAX ||
        !multiply_size((size_t) grid->n, 6u, &feature_values) ||
        feature_values > SIZE_MAX / sizeof(float)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup_bvh;
    }
    float * features = (float *) malloc(feature_values * sizeof(float));
    if (features == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup_bvh;
    }

    for (int64_t row = 0; row < grid->n; ++row) {
        float point[3];
        for (int axis = 0; axis < 3; ++axis) {
            point[axis] = grid->dual_vertices[(size_t) row * 3u + (size_t) axis] +
                SEGMENTATION_FDG_AABB_MIN;
        }
        material_nearest nearest;
        nearest.face = SIZE_MAX;
        nearest.distance_squared = FLT_MAX;
        nearest.barycentric[0] = 1.0f;
        nearest.barycentric[1] = 0.0f;
        nearest.barycentric[2] = 0.0f;
        bvh_query_node(&bvh, 0u, point, &nearest);
        const trellis_mesh_rigging_primitive_range * range =
            range_for_face(asset, nearest.face);
        if (nearest.face == SIZE_MAX || range == NULL) {
            free(features);
            status = TRELLIS_STATUS_ERROR;
            goto cleanup_bvh;
        }

        const cgltf_material * material = NULL;
        if (range->source_material_index != SIZE_MAX) {
            material = &data->materials[range->source_material_index];
        }
        float base_color[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        float metallic = 1.0f;
        float roughness = 1.0f;
        if (material != NULL) {
            const cgltf_pbr_metallic_roughness * pbr =
                &material->pbr_metallic_roughness;
            memcpy(base_color, pbr->base_color_factor, sizeof(base_color));
            metallic = pbr->metallic_factor;
            roughness = pbr->roughness_factor;
            const int32_t * indices = normalized_mesh->faces + nearest.face * 3u;
            if (pbr->base_color_texture.texture != NULL) {
                float sample[4];
                if (!sample_texture_view(
                        &source,
                        &pbr->base_color_texture,
                        asset,
                        range,
                        indices,
                        nearest.barycentric,
                        sample)) {
                    free(features);
                    status = TRELLIS_STATUS_PARSE_ERROR;
                    goto cleanup_bvh;
                }
                for (int channel = 0; channel < 4; ++channel) {
                    base_color[channel] *= sample[channel];
                }
            }
            if (pbr->metallic_roughness_texture.texture != NULL) {
                float sample[4];
                if (!sample_texture_view(
                        &source,
                        &pbr->metallic_roughness_texture,
                        asset,
                        range,
                        indices,
                        nearest.barycentric,
                        sample)) {
                    free(features);
                    status = TRELLIS_STATUS_PARSE_ERROR;
                    goto cleanup_bvh;
                }
                roughness *= sample[1];
                metallic *= sample[2];
            }
        }
        float channels[6] = {
            base_color[0], base_color[1], base_color[2],
            metallic, roughness, base_color[3],
        };
        for (int channel = 0; channel < 6; ++channel) {
            features[(size_t) row * 6u + (size_t) channel] =
                clamp_unit(channels[channel]) * 2.0f - 1.0f;
        }
    }
    *features_out = features;
    status = TRELLIS_STATUS_OK;

cleanup_bvh:
    bvh_free(&bvh);
cleanup:
    if (source.images != NULL) {
        for (size_t image = 0u; image < (size_t) data->images_count; ++image) {
            stbi_image_free(source.images[image].rgba);
        }
    }
    free(source.images);
    cgltf_free(data);
    return status;
}
