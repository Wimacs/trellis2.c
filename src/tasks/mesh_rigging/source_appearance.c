#include "source_appearance.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void appearance_set_error(
    char * output,
    size_t output_size,
    const char * format,
    ...) {
    if (output == NULL || output_size == 0) return;
    va_list args;
    va_start(args, format);
    vsnprintf(output, output_size, format, args);
    va_end(args);
    output[output_size - 1u] = '\0';
}

static int appearance_add_size(size_t a, size_t b, size_t * result) {
    if (result == NULL || a > SIZE_MAX - b) return 0;
    *result = a + b;
    return 1;
}

static char * appearance_duplicate_string(const char * value) {
    if (value == NULL) return NULL;
    const size_t length = strlen(value) + 1u;
    char * result = (char *) malloc(length);
    if (result != NULL) memcpy(result, value, length);
    return result;
}

static int appearance_ascii_equal_nocase(
    const char * left,
    const char * right,
    size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (tolower((unsigned char) left[i]) !=
            tolower((unsigned char) right[i])) return 0;
    }
    return 1;
}

static const char * appearance_infer_image_mime_type(const char * uri) {
    if (uri == NULL) return NULL;
    const char * end = uri + strlen(uri);
    while (end > uri && (end[-1] == ' ' || end[-1] == '\t')) --end;
    const char * query = memchr(uri, '?', (size_t) (end - uri));
    if (query != NULL) end = query;
    const char * dot = end;
    while (dot > uri && dot[-1] != '.' && dot[-1] != '/' && dot[-1] != '\\') --dot;
    if (dot == uri || dot[-1] != '.') return NULL;
    const size_t length = (size_t) (end - dot);
    if (length == 3u && appearance_ascii_equal_nocase(dot, "png", 3u)) return "image/png";
    if ((length == 3u && appearance_ascii_equal_nocase(dot, "jpg", 3u)) ||
        (length == 4u && appearance_ascii_equal_nocase(dot, "jpeg", 4u))) return "image/jpeg";
    if (length == 4u && appearance_ascii_equal_nocase(dot, "webp", 4u)) return "image/webp";
    if (length == 4u && appearance_ascii_equal_nocase(dot, "ktx2", 4u)) return "image/ktx2";
    return NULL;
}

static trellis_status appearance_read_file_bytes(
    const char * path,
    unsigned char ** bytes_out,
    size_t * size_out,
    char * error,
    size_t error_size) {
    FILE * file = fopen(path, "rb");
    if (file == NULL) {
        appearance_set_error(error, error_size, "could not open source image '%s'", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        appearance_set_error(error, error_size, "could not size source image '%s'", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    const long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        appearance_set_error(error, error_size, "source image '%s' is empty or unreadable", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    unsigned char * bytes = (unsigned char *) malloc((size_t) length);
    if (bytes == NULL) {
        fclose(file);
        appearance_set_error(error, error_size, "out of memory loading source image '%s'", path);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    const int ok = fread(bytes, 1, (size_t) length, file) == (size_t) length;
    fclose(file);
    if (!ok) {
        free(bytes);
        appearance_set_error(error, error_size, "could not read source image '%s'", path);
        return TRELLIS_STATUS_IO_ERROR;
    }
    *bytes_out = bytes;
    *size_out = (size_t) length;
    return TRELLIS_STATUS_OK;
}

static trellis_status appearance_load_uri_image(
    const char * source_path,
    const char * uri,
    trellis_gltf_source_image * image,
    char * error,
    size_t error_size) {
    if (strncmp(uri, "data:", 5u) == 0) {
        const char * comma = strchr(uri + 5u, ',');
        if (comma == NULL) {
            appearance_set_error(error, error_size, "source image has an invalid data URI");
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        const char * semicolon = memchr(uri + 5u, ';', (size_t) (comma - (uri + 5u)));
        const char * mime_end = semicolon != NULL ? semicolon : comma;
        if (mime_end > uri + 5u) {
            const size_t mime_length = (size_t) (mime_end - (uri + 5u));
            image->owned_mime_type = (char *) malloc(mime_length + 1u);
            if (image->owned_mime_type == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
            memcpy(image->owned_mime_type, uri + 5u, mime_length);
            image->owned_mime_type[mime_length] = '\0';
            image->embedded_mime_type = image->owned_mime_type;
        }
        const int is_base64 = semicolon != NULL &&
            (size_t) (comma - semicolon) >= 7u &&
            strncmp(comma - 7u, ";base64", 7u) == 0;
        if (is_base64) {
            const char * encoded = comma + 1u;
            const size_t encoded_length = strlen(encoded);
            if (encoded_length == 0 || encoded_length % 4u != 0) {
                appearance_set_error(error, error_size, "source image has malformed base64 data");
                return TRELLIS_STATUS_PARSE_ERROR;
            }
            size_t decoded_size = encoded_length / 4u * 3u;
            if (encoded[encoded_length - 1u] == '=') --decoded_size;
            if (encoded_length > 1u && encoded[encoded_length - 2u] == '=') --decoded_size;
            cgltf_options options;
            memset(&options, 0, sizeof(options));
            void * decoded = NULL;
            const cgltf_result result = cgltf_load_buffer_base64(
                &options, (cgltf_size) decoded_size, encoded, &decoded);
            if (result != cgltf_result_success || decoded == NULL) {
                appearance_set_error(error, error_size, "could not decode source image data URI");
                return result == cgltf_result_out_of_memory ?
                    TRELLIS_STATUS_OUT_OF_MEMORY : TRELLIS_STATUS_PARSE_ERROR;
            }
            image->owned_bytes = (unsigned char *) decoded;
            image->bytes = image->owned_bytes;
            image->size = decoded_size;
            return TRELLIS_STATUS_OK;
        }
        char * decoded = appearance_duplicate_string(comma + 1u);
        if (decoded == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
        const size_t decoded_size = (size_t) cgltf_decode_uri(decoded);
        image->owned_bytes = (unsigned char *) decoded;
        image->bytes = image->owned_bytes;
        image->size = decoded_size;
        if (decoded_size == 0) {
            appearance_set_error(error, error_size, "source image data URI decodes to an empty payload");
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        return TRELLIS_STATUS_OK;
    }

    char * decoded_uri = appearance_duplicate_string(uri);
    if (decoded_uri == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    (void) cgltf_decode_uri(decoded_uri);
    const char * slash = strrchr(source_path, '/');
    const char * backslash = strrchr(source_path, '\\');
    if (backslash != NULL && (slash == NULL || backslash > slash)) slash = backslash;
    const int absolute = decoded_uri[0] == '/' || decoded_uri[0] == '\\' ||
        (isalpha((unsigned char) decoded_uri[0]) && decoded_uri[1] == ':');
    const size_t directory_length = !absolute && slash != NULL ?
        (size_t) (slash - source_path + 1u) : 0u;
    size_t path_length = 0;
    if (!appearance_add_size(directory_length, strlen(decoded_uri), &path_length) ||
        !appearance_add_size(path_length, 1u, &path_length)) {
        free(decoded_uri);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    char * path = (char *) malloc(path_length);
    if (path == NULL) {
        free(decoded_uri);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    if (directory_length != 0) memcpy(path, source_path, directory_length);
    memcpy(path + directory_length, decoded_uri, strlen(decoded_uri) + 1u);
    const trellis_status status = appearance_read_file_bytes(
        path, &image->owned_bytes, &image->size, error, error_size);
    image->bytes = image->owned_bytes;
    if (image->embedded_mime_type == NULL) {
        image->embedded_mime_type = appearance_infer_image_mime_type(decoded_uri);
    }
    free(path);
    free(decoded_uri);
    return status;
}

const char * trellis_gltf_cgltf_result_name(cgltf_result result) {
    switch (result) {
        case cgltf_result_success: return "success";
        case cgltf_result_data_too_short: return "data too short";
        case cgltf_result_unknown_format: return "unknown format";
        case cgltf_result_invalid_json: return "invalid JSON";
        case cgltf_result_invalid_gltf: return "invalid glTF";
        case cgltf_result_invalid_options: return "invalid options";
        case cgltf_result_file_not_found: return "file not found";
        case cgltf_result_io_error: return "I/O error";
        case cgltf_result_out_of_memory: return "out of memory";
        case cgltf_result_legacy_gltf: return "legacy glTF";
        default: return "unknown error";
    }
}

static trellis_status appearance_status_from_cgltf(cgltf_result result) {
    if (result == cgltf_result_out_of_memory) return TRELLIS_STATUS_OUT_OF_MEMORY;
    if (result == cgltf_result_file_not_found || result == cgltf_result_io_error) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    return TRELLIS_STATUS_PARSE_ERROR;
}

void trellis_gltf_source_appearance_free(
    trellis_gltf_source_appearance * appearance) {
    if (appearance == NULL) return;
    if (appearance->data != NULL && appearance->images != NULL) {
        for (size_t i = 0; i < (size_t) appearance->data->images_count; ++i) {
            appearance->data->images[i].uri = appearance->images[i].original_uri;
            appearance->data->images[i].buffer_view =
                appearance->images[i].original_buffer_view;
            appearance->data->images[i].mime_type =
                appearance->images[i].original_mime_type;
            free(appearance->images[i].owned_mime_type);
            free(appearance->images[i].owned_bytes);
        }
    }
    free(appearance->images);
    cgltf_free(appearance->data);
    memset(appearance, 0, sizeof(*appearance));
}

trellis_status trellis_gltf_source_appearance_load(
    const trellis_mesh_rigging_asset * asset,
    trellis_gltf_source_appearance * appearance,
    char * error,
    size_t error_size) {
    if (appearance == NULL || asset == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;
    memset(appearance, 0, sizeof(*appearance));
    if (asset->source_path == NULL || asset->source_path[0] == '\0') {
        return TRELLIS_STATUS_OK;
    }
    cgltf_options options;
    memset(&options, 0, sizeof(options));
    cgltf_result result = cgltf_parse_file(
        &options, asset->source_path, &appearance->data);
    if (result == cgltf_result_success) {
        result = cgltf_load_buffers(&options, appearance->data, asset->source_path);
    }
    if (result == cgltf_result_success) result = cgltf_validate(appearance->data);
    if (result != cgltf_result_success || appearance->data == NULL) {
        appearance_set_error(
            error, error_size, "could not reopen source appearance '%s': %s",
            asset->source_path, trellis_gltf_cgltf_result_name(result));
        trellis_gltf_source_appearance_free(appearance);
        return appearance_status_from_cgltf(result);
    }
    if (appearance->data->extensions_required_count != 0) {
        appearance_set_error(
            error, error_size,
            "source appearance requires glTF extension '%s'; required-extension preservation is unsupported",
            appearance->data->extensions_required[0]);
        trellis_gltf_source_appearance_free(appearance);
        return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }
    if (appearance->data->data_extensions_count != 0) {
        appearance_set_error(
            error,
            error_size,
            "source appearance contains unsupported raw glTF extension '%s'",
            appearance->data->data_extensions[0].name);
        trellis_gltf_source_appearance_free(appearance);
        return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }
#define REJECT_RAW_APPEARANCE_EXTENSIONS(array_name, count_name, owner_name) do { \
        for (size_t i = 0; i < (size_t) appearance->data->count_name; ++i) { \
            if (appearance->data->array_name[i].extensions_count != 0) { \
                appearance_set_error( \
                    error, error_size, \
                    "source %s %zu contains unsupported raw glTF extension '%s'", \
                    owner_name, i, appearance->data->array_name[i].extensions[0].name); \
                trellis_gltf_source_appearance_free(appearance); \
                return TRELLIS_STATUS_NOT_IMPLEMENTED; \
            } \
        } \
    } while (0)
    REJECT_RAW_APPEARANCE_EXTENSIONS(materials, materials_count, "material");
    REJECT_RAW_APPEARANCE_EXTENSIONS(textures, textures_count, "texture");
    REJECT_RAW_APPEARANCE_EXTENSIONS(images, images_count, "image");
    REJECT_RAW_APPEARANCE_EXTENSIONS(samplers, samplers_count, "sampler");
#undef REJECT_RAW_APPEARANCE_EXTENSIONS
    for (size_t primitive = 0; primitive < asset->primitive_count; ++primitive) {
        const trellis_mesh_rigging_primitive_range * range = &asset->primitives[primitive];
        if (range->source_mesh_index >= (size_t) appearance->data->meshes_count ||
            range->source_node_index >= (size_t) appearance->data->nodes_count ||
            range->source_primitive_index >= (size_t)
                appearance->data->meshes[range->source_mesh_index].primitives_count ||
            (range->source_material_index != SIZE_MAX &&
             range->source_material_index >= (size_t) appearance->data->materials_count)) {
            appearance_set_error(
                error, error_size,
                "source appearance mapping is inconsistent at primitive %zu", primitive);
            trellis_gltf_source_appearance_free(appearance);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        const cgltf_node * source_node =
            &appearance->data->nodes[range->source_node_index];
        const cgltf_mesh * source_mesh =
            &appearance->data->meshes[range->source_mesh_index];
        const cgltf_primitive * source_primitive =
            &source_mesh->primitives[range->source_primitive_index];
        const cgltf_accessor * source_position = cgltf_find_accessor(
            source_primitive, cgltf_attribute_type_position, 0);
        const size_t source_material_index = source_primitive->material != NULL ?
            (size_t) (source_primitive->material - appearance->data->materials) : SIZE_MAX;
        const cgltf_accessor * source_normal = cgltf_find_accessor(
            source_primitive, cgltf_attribute_type_normal, 0);
        const cgltf_accessor * source_tangent = cgltf_find_accessor(
            source_primitive, cgltf_attribute_type_tangent, 0);
        uint64_t texcoord_mask = 0;
        uint64_t color_mask = 0;
        uint64_t color_vec4_mask = 0;
        int attribute_mapping_valid = 1;
        for (size_t attribute = 0;
             attribute < (size_t) source_primitive->attributes_count;
             ++attribute) {
            const cgltf_attribute * source_attribute =
                &source_primitive->attributes[attribute];
            if (source_attribute->type != cgltf_attribute_type_texcoord &&
                source_attribute->type != cgltf_attribute_type_color) continue;
            if (source_attribute->index < 0 || source_attribute->index >= 64) {
                attribute_mapping_valid = 0;
                break;
            }
            const uint64_t bit = UINT64_C(1) << (unsigned) source_attribute->index;
            if (source_attribute->type == cgltf_attribute_type_texcoord) {
                texcoord_mask |= bit;
            } else {
                color_mask |= bit;
                if (source_attribute->data != NULL &&
                    source_attribute->data->type == cgltf_type_vec4) {
                    color_vec4_mask |= bit;
                }
            }
        }
        if (source_node->mesh != source_mesh || source_position == NULL ||
            (size_t) (source_position - appearance->data->accessors) !=
                range->source_position_accessor_index ||
            (size_t) source_position->count != range->vertex_count ||
            source_material_index != range->source_material_index ||
            !attribute_mapping_valid ||
            (source_normal != NULL) != range->has_source_normal ||
            (source_tangent != NULL) != range->has_source_tangent ||
            texcoord_mask != range->texcoord_mask ||
            color_mask != range->color_mask ||
            color_vec4_mask != range->color_vec4_mask) {
            appearance_set_error(
                error,
                error_size,
                "source document changed after flattening at primitive %zu",
                primitive);
            trellis_gltf_source_appearance_free(appearance);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    if (appearance->data->images_count == 0) return TRELLIS_STATUS_OK;

    appearance->images = (trellis_gltf_source_image *) calloc(
        (size_t) appearance->data->images_count, sizeof(*appearance->images));
    if (appearance->images == NULL) {
        trellis_gltf_source_appearance_free(appearance);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < (size_t) appearance->data->images_count; ++i) {
        cgltf_image * source_image = &appearance->data->images[i];
        trellis_gltf_source_image * image = &appearance->images[i];
        image->original_uri = source_image->uri;
        image->original_buffer_view = source_image->buffer_view;
        image->original_mime_type = source_image->mime_type;
        image->embedded_mime_type = source_image->mime_type;
    }
    for (size_t i = 0; i < (size_t) appearance->data->images_count; ++i) {
        cgltf_image * source_image = &appearance->data->images[i];
        trellis_gltf_source_image * image = &appearance->images[i];
        if (source_image->buffer_view != NULL) {
            image->bytes = cgltf_buffer_view_data(source_image->buffer_view);
            image->size = (size_t) source_image->buffer_view->size;
            if (image->bytes == NULL || image->size == 0) {
                appearance_set_error(
                    error, error_size,
                    "source image %zu has no readable buffer-view payload", i);
                trellis_gltf_source_appearance_free(appearance);
                return TRELLIS_STATUS_PARSE_ERROR;
            }
        } else if (source_image->uri != NULL) {
            const trellis_status status = appearance_load_uri_image(
                asset->source_path, source_image->uri, image, error, error_size);
            if (status != TRELLIS_STATUS_OK) {
                trellis_gltf_source_appearance_free(appearance);
                return status;
            }
        } else {
            appearance_set_error(
                error, error_size,
                "source image %zu has neither bufferView nor URI", i);
            trellis_gltf_source_appearance_free(appearance);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        if (image->embedded_mime_type == NULL) {
            appearance_set_error(
                error, error_size,
                "source image %zu has no MIME type that can be embedded", i);
            trellis_gltf_source_appearance_free(appearance);
            return TRELLIS_STATUS_NOT_IMPLEMENTED;
        }
    }
    return TRELLIS_STATUS_OK;
}
