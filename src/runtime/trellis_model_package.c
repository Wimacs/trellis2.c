#include "trellis_model_package.h"

#include "trellis_registry.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRELLIS_MODEL_MANIFEST_MAX_BYTES (4u * 1024u * 1024u)
#define TRELLIS_JSON_MAX_DEPTH 64

typedef enum json_type {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT,
} json_type;

typedef struct json_value json_value;

typedef struct json_member {
    char * key;
    json_value * value;
} json_member;

struct json_value {
    json_type type;
    union {
        int boolean;
        double number;
        char * string;
        struct {
            json_value ** items;
            size_t count;
        } array;
        struct {
            json_member * members;
            size_t count;
        } object;
    } as;
};

typedef struct json_parser {
    const char * cursor;
    const char * end;
    int depth;
} json_parser;

static char * string_duplicate(const char * value) {
    if (value == NULL) {
        return NULL;
    }
    const size_t size = strlen(value) + 1;
    char * copy = (char *) malloc(size);
    if (copy != NULL) {
        memcpy(copy, value, size);
    }
    return copy;
}

static void json_value_free(json_value * value) {
    if (value == NULL) {
        return;
    }
    if (value->type == JSON_STRING) {
        free(value->as.string);
    } else if (value->type == JSON_ARRAY) {
        for (size_t i = 0; i < value->as.array.count; ++i) {
            json_value_free(value->as.array.items[i]);
        }
        free(value->as.array.items);
    } else if (value->type == JSON_OBJECT) {
        for (size_t i = 0; i < value->as.object.count; ++i) {
            free(value->as.object.members[i].key);
            json_value_free(value->as.object.members[i].value);
        }
        free(value->as.object.members);
    }
    free(value);
}

static void json_skip_ws(json_parser * parser) {
    while (parser->cursor < parser->end &&
           (*parser->cursor == ' ' || *parser->cursor == '\t' ||
            *parser->cursor == '\r' || *parser->cursor == '\n')) {
        ++parser->cursor;
    }
}

static int json_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int json_parse_hex4(json_parser * parser, uint32_t * value_out) {
    if ((size_t) (parser->end - parser->cursor) < 4) {
        return 0;
    }
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        int digit = json_hex_digit(parser->cursor[i]);
        if (digit < 0) {
            return 0;
        }
        value = value * 16u + (uint32_t) digit;
    }
    parser->cursor += 4;
    *value_out = value;
    return 1;
}

static int json_append_utf8(char * output, size_t capacity, size_t * length, uint32_t codepoint) {
    if (codepoint == 0 || codepoint > 0x10ffffu ||
        (codepoint >= 0xd800u && codepoint <= 0xdfffu)) {
        return 0;
    }
    size_t needed = 1;
    if (codepoint >= 0x80u) needed = codepoint < 0x800u ? 2 : (codepoint < 0x10000u ? 3 : 4);
    if (*length + needed >= capacity) {
        return 0;
    }
    if (needed == 1) {
        output[(*length)++] = (char) codepoint;
    } else if (needed == 2) {
        output[(*length)++] = (char) (0xc0u | (codepoint >> 6));
        output[(*length)++] = (char) (0x80u | (codepoint & 0x3fu));
    } else if (needed == 3) {
        output[(*length)++] = (char) (0xe0u | (codepoint >> 12));
        output[(*length)++] = (char) (0x80u | ((codepoint >> 6) & 0x3fu));
        output[(*length)++] = (char) (0x80u | (codepoint & 0x3fu));
    } else {
        output[(*length)++] = (char) (0xf0u | (codepoint >> 18));
        output[(*length)++] = (char) (0x80u | ((codepoint >> 12) & 0x3fu));
        output[(*length)++] = (char) (0x80u | ((codepoint >> 6) & 0x3fu));
        output[(*length)++] = (char) (0x80u | (codepoint & 0x3fu));
    }
    return 1;
}

static char * json_parse_string(json_parser * parser) {
    if (parser->cursor >= parser->end || *parser->cursor != '"') {
        return NULL;
    }
    const char * raw_begin = ++parser->cursor;
    const char * scan = raw_begin;
    while (scan < parser->end && *scan != '"') {
        if ((unsigned char) *scan < 0x20u) {
            return NULL;
        }
        if (*scan == '\\') {
            ++scan;
            if (scan >= parser->end) return NULL;
        }
        ++scan;
    }
    if (scan >= parser->end) {
        return NULL;
    }

    const size_t capacity = (size_t) (scan - raw_begin) + 1;
    char * output = (char *) malloc(capacity);
    if (output == NULL) {
        return NULL;
    }
    size_t length = 0;
    while (parser->cursor < scan) {
        unsigned char c = (unsigned char) *parser->cursor++;
        if (c != '\\') {
            output[length++] = (char) c;
            continue;
        }
        if (parser->cursor >= scan) {
            free(output);
            return NULL;
        }
        char escaped = *parser->cursor++;
        switch (escaped) {
            case '"': output[length++] = '"'; break;
            case '\\': output[length++] = '\\'; break;
            case '/': output[length++] = '/'; break;
            case 'b': output[length++] = '\b'; break;
            case 'f': output[length++] = '\f'; break;
            case 'n': output[length++] = '\n'; break;
            case 'r': output[length++] = '\r'; break;
            case 't': output[length++] = '\t'; break;
            case 'u': {
                uint32_t codepoint = 0;
                if (!json_parse_hex4(parser, &codepoint) || parser->cursor > scan) {
                    free(output);
                    return NULL;
                }
                if (codepoint >= 0xd800u && codepoint <= 0xdbffu) {
                    if (scan - parser->cursor < 6 || parser->cursor[0] != '\\' || parser->cursor[1] != 'u') {
                        free(output);
                        return NULL;
                    }
                    parser->cursor += 2;
                    uint32_t low = 0;
                    if (!json_parse_hex4(parser, &low) || low < 0xdc00u || low > 0xdfffu) {
                        free(output);
                        return NULL;
                    }
                    codepoint = 0x10000u + ((codepoint - 0xd800u) << 10) + (low - 0xdc00u);
                }
                if (!json_append_utf8(output, capacity, &length, codepoint)) {
                    free(output);
                    return NULL;
                }
                break;
            }
            default:
                free(output);
                return NULL;
        }
    }
    parser->cursor = scan + 1;
    output[length] = '\0';
    return output;
}

static json_value * json_parse_value(json_parser * parser);

static json_value * json_new(json_type type) {
    json_value * value = (json_value *) calloc(1, sizeof(*value));
    if (value != NULL) {
        value->type = type;
    }
    return value;
}

static json_value * json_parse_array(json_parser * parser) {
    if (parser->depth >= TRELLIS_JSON_MAX_DEPTH) {
        return NULL;
    }
    ++parser->depth;
    ++parser->cursor;
    json_value * array = json_new(JSON_ARRAY);
    if (array == NULL) goto fail;
    json_skip_ws(parser);
    if (parser->cursor < parser->end && *parser->cursor == ']') {
        ++parser->cursor;
        --parser->depth;
        return array;
    }
    for (;;) {
        json_skip_ws(parser);
        json_value * item = json_parse_value(parser);
        if (item == NULL) goto fail_value;
        json_value ** grown = (json_value **) realloc(
            array->as.array.items,
            (array->as.array.count + 1) * sizeof(*grown));
        if (grown == NULL) {
            json_value_free(item);
            goto fail_value;
        }
        array->as.array.items = grown;
        array->as.array.items[array->as.array.count++] = item;
        json_skip_ws(parser);
        if (parser->cursor >= parser->end) goto fail_value;
        if (*parser->cursor == ']') {
            ++parser->cursor;
            --parser->depth;
            return array;
        }
        if (*parser->cursor++ != ',') goto fail_value;
    }

fail_value:
    json_value_free(array);
fail:
    --parser->depth;
    return NULL;
}

static json_value * json_parse_object(json_parser * parser) {
    if (parser->depth >= TRELLIS_JSON_MAX_DEPTH) {
        return NULL;
    }
    ++parser->depth;
    ++parser->cursor;
    json_value * object = json_new(JSON_OBJECT);
    if (object == NULL) goto fail;
    json_skip_ws(parser);
    if (parser->cursor < parser->end && *parser->cursor == '}') {
        ++parser->cursor;
        --parser->depth;
        return object;
    }
    for (;;) {
        json_skip_ws(parser);
        char * key = json_parse_string(parser);
        if (key == NULL) goto fail_value;
        json_skip_ws(parser);
        if (parser->cursor >= parser->end || *parser->cursor++ != ':') {
            free(key);
            goto fail_value;
        }
        json_skip_ws(parser);
        json_value * member_value = json_parse_value(parser);
        if (member_value == NULL) {
            free(key);
            goto fail_value;
        }
        json_member * grown = (json_member *) realloc(
            object->as.object.members,
            (object->as.object.count + 1) * sizeof(*grown));
        if (grown == NULL) {
            free(key);
            json_value_free(member_value);
            goto fail_value;
        }
        object->as.object.members = grown;
        object->as.object.members[object->as.object.count].key = key;
        object->as.object.members[object->as.object.count].value = member_value;
        ++object->as.object.count;
        json_skip_ws(parser);
        if (parser->cursor >= parser->end) goto fail_value;
        if (*parser->cursor == '}') {
            ++parser->cursor;
            --parser->depth;
            return object;
        }
        if (*parser->cursor++ != ',') goto fail_value;
    }

fail_value:
    json_value_free(object);
fail:
    --parser->depth;
    return NULL;
}

static int json_match_literal(json_parser * parser, const char * literal) {
    const size_t length = strlen(literal);
    if ((size_t) (parser->end - parser->cursor) < length ||
        memcmp(parser->cursor, literal, length) != 0) {
        return 0;
    }
    parser->cursor += length;
    return 1;
}

static json_value * json_parse_number(json_parser * parser) {
    const char * begin = parser->cursor;
    if (*parser->cursor == '-') ++parser->cursor;
    if (parser->cursor >= parser->end) return NULL;
    if (*parser->cursor == '0') {
        ++parser->cursor;
        if (parser->cursor < parser->end && isdigit((unsigned char) *parser->cursor)) return NULL;
    } else {
        if (!isdigit((unsigned char) *parser->cursor)) return NULL;
        while (parser->cursor < parser->end && isdigit((unsigned char) *parser->cursor)) ++parser->cursor;
    }
    if (parser->cursor < parser->end && *parser->cursor == '.') {
        ++parser->cursor;
        if (parser->cursor >= parser->end || !isdigit((unsigned char) *parser->cursor)) return NULL;
        while (parser->cursor < parser->end && isdigit((unsigned char) *parser->cursor)) ++parser->cursor;
    }
    if (parser->cursor < parser->end && (*parser->cursor == 'e' || *parser->cursor == 'E')) {
        ++parser->cursor;
        if (parser->cursor < parser->end && (*parser->cursor == '+' || *parser->cursor == '-')) ++parser->cursor;
        if (parser->cursor >= parser->end || !isdigit((unsigned char) *parser->cursor)) return NULL;
        while (parser->cursor < parser->end && isdigit((unsigned char) *parser->cursor)) ++parser->cursor;
    }

    const size_t length = (size_t) (parser->cursor - begin);
    char local[64];
    if (length == 0 || length >= sizeof(local)) return NULL;
    memcpy(local, begin, length);
    local[length] = '\0';
    char * end = NULL;
    errno = 0;
    double number = strtod(local, &end);
    if (errno == ERANGE || end == NULL || *end != '\0' || !isfinite(number)) return NULL;
    json_value * value = json_new(JSON_NUMBER);
    if (value != NULL) value->as.number = number;
    return value;
}

static json_value * json_parse_value(json_parser * parser) {
    json_skip_ws(parser);
    if (parser->cursor >= parser->end) return NULL;
    if (*parser->cursor == '{') return json_parse_object(parser);
    if (*parser->cursor == '[') return json_parse_array(parser);
    if (*parser->cursor == '"') {
        char * string = json_parse_string(parser);
        if (string == NULL) return NULL;
        json_value * value = json_new(JSON_STRING);
        if (value == NULL) {
            free(string);
            return NULL;
        }
        value->as.string = string;
        return value;
    }
    if (*parser->cursor == '-' || isdigit((unsigned char) *parser->cursor)) {
        return json_parse_number(parser);
    }
    if (json_match_literal(parser, "true")) {
        json_value * value = json_new(JSON_BOOL);
        if (value != NULL) value->as.boolean = 1;
        return value;
    }
    if (json_match_literal(parser, "false")) return json_new(JSON_BOOL);
    if (json_match_literal(parser, "null")) return json_new(JSON_NULL);
    return NULL;
}

static json_value * json_parse_document(const char * text, size_t length) {
    json_parser parser = { text, text + length, 0 };
    json_value * root = json_parse_value(&parser);
    if (root == NULL) return NULL;
    json_skip_ws(&parser);
    if (parser.cursor != parser.end) {
        json_value_free(root);
        return NULL;
    }
    return root;
}

static int json_object_has_duplicate_keys(const json_value * object) {
    if (object == NULL || object->type != JSON_OBJECT) return 1;
    for (size_t i = 0; i < object->as.object.count; ++i) {
        for (size_t j = i + 1; j < object->as.object.count; ++j) {
            if (strcmp(object->as.object.members[i].key, object->as.object.members[j].key) == 0) {
                return 1;
            }
        }
    }
    return 0;
}

static const json_value * json_object_get(const json_value * object, const char * key) {
    if (object == NULL || object->type != JSON_OBJECT || key == NULL) return NULL;
    for (size_t i = 0; i < object->as.object.count; ++i) {
        if (strcmp(object->as.object.members[i].key, key) == 0) {
            return object->as.object.members[i].value;
        }
    }
    return NULL;
}

static int model_identifier_is_valid(const char * value) {
    if (value == NULL || value[0] == '\0') return 0;
    for (const unsigned char * p = (const unsigned char *) value; *p != '\0'; ++p) {
        if (!(isalnum(*p) || *p == '_' || *p == '-' || *p == '.')) return 0;
    }
    return 1;
}

int trellis_model_weights_path_is_safe(const char * relative_path) {
    if (relative_path == NULL || relative_path[0] == '\0' ||
        relative_path[0] == '/' || relative_path[0] == '\\') {
        return 0;
    }
    const char * segment = relative_path;
    for (const unsigned char * p = (const unsigned char *) relative_path;; ++p) {
        const unsigned char c = *p;
        if (c != '\0' && (c == '\\' || c == ':' || c < 0x20u || c == 0x7fu)) return 0;
        if (c == '/' || c == '\0') {
            const size_t length = (size_t) ((const char *) p - segment);
            if (length == 0 ||
                (length == 1 && segment[0] == '.') ||
                (length == 2 && segment[0] == '.' && segment[1] == '.')) {
                return 0;
            }
            if (c == '\0') break;
            segment = (const char *) p + 1;
        }
    }
    return 1;
}

const char * trellis_attention_mode_name(trellis_attention_mode mode) {
    switch (mode) {
        case TRELLIS_ATTENTION_NONE: return "none";
        case TRELLIS_ATTENTION_SDPA: return "sdpa";
        case TRELLIS_ATTENTION_FLASH: return "flash";
        default: return "unknown";
    }
}

static trellis_status parse_dtype(const json_value * value, trellis_dtype * dtype_out) {
    if (value == NULL || value->type != JSON_STRING || dtype_out == NULL) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    if (strcmp(value->as.string, "f32") == 0) *dtype_out = TRELLIS_DTYPE_F32;
    else if (strcmp(value->as.string, "f16") == 0) *dtype_out = TRELLIS_DTYPE_F16;
    else if (strcmp(value->as.string, "bf16") == 0) *dtype_out = TRELLIS_DTYPE_BF16;
    else return TRELLIS_STATUS_PARSE_ERROR;
    return TRELLIS_STATUS_OK;
}

static trellis_status parse_attention(
    const json_value * value,
    trellis_attention_mode * attention_out) {
    if (value == NULL || value->type != JSON_STRING || attention_out == NULL) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    if (strcmp(value->as.string, "none") == 0) *attention_out = TRELLIS_ATTENTION_NONE;
    else if (strcmp(value->as.string, "sdpa") == 0) *attention_out = TRELLIS_ATTENTION_SDPA;
    else if (strcmp(value->as.string, "flash") == 0) *attention_out = TRELLIS_ATTENTION_FLASH;
    else return TRELLIS_STATUS_PARSE_ERROR;
    return TRELLIS_STATUS_OK;
}

static trellis_status parse_optional_bool(
    const json_value * object,
    const char * key,
    int default_value,
    int * value_out) {
    const json_value * value = json_object_get(object, key);
    if (value == NULL) {
        *value_out = default_value;
        return TRELLIS_STATUS_OK;
    }
    if (value->type != JSON_BOOL) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    *value_out = value->as.boolean ? 1 : 0;
    return TRELLIS_STATUS_OK;
}

static trellis_status parse_execution_policy(
    const json_value * value,
    trellis_execution_policy * policy_out) {
    if (value == NULL || value->type != JSON_OBJECT || policy_out == NULL ||
        json_object_has_duplicate_keys(value)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    trellis_execution_policy policy = TRELLIS_EXECUTION_POLICY_INIT;
    trellis_status status = parse_dtype(json_object_get(value, "compute_dtype"), &policy.compute_dtype);
    if (status != TRELLIS_STATUS_OK) return status;
    status = parse_attention(json_object_get(value, "attention"), &policy.attention);
    if (status != TRELLIS_STATUS_OK) return status;

    const json_value * flash_kv = json_object_get(value, "flash_kv_dtype");
    if (policy.attention == TRELLIS_ATTENTION_FLASH) {
        status = parse_dtype(flash_kv, &policy.flash_kv_dtype);
        if (status != TRELLIS_STATUS_OK ||
            (policy.flash_kv_dtype != TRELLIS_DTYPE_F16 &&
             policy.flash_kv_dtype != TRELLIS_DTYPE_BF16)) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    } else if (flash_kv != NULL) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    status = parse_optional_bool(
        value,
        "emulate_bf16_blocks",
        0,
        &policy.emulate_bf16_blocks);
    if (status != TRELLIS_STATUS_OK) return status;
    *policy_out = policy;
    return TRELLIS_STATUS_OK;
}

static trellis_status duplicate_required_identifier(
    const json_value * object,
    const char * key,
    char ** output) {
    const json_value * value = json_object_get(object, key);
    if (value == NULL || value->type != JSON_STRING ||
        !model_identifier_is_valid(value->as.string)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    *output = string_duplicate(value->as.string);
    return *output == NULL ? TRELLIS_STATUS_OUT_OF_MEMORY : TRELLIS_STATUS_OK;
}

static trellis_status parse_manifest_component(
    const json_value * value,
    trellis_model_component_instance * component) {
    if (value == NULL || value->type != JSON_OBJECT || component == NULL ||
        json_object_has_duplicate_keys(value)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    memset(component, 0, sizeof(*component));
    trellis_status status = duplicate_required_identifier(value, "role", &component->role);
    if (status != TRELLIS_STATUS_OK) return status;
    status = duplicate_required_identifier(value, "architecture", &component->architecture);
    if (status != TRELLIS_STATUS_OK) return status;

    const json_value * weights = json_object_get(value, "weights");
    if (weights == NULL || weights->type != JSON_STRING ||
        !trellis_model_weights_path_is_safe(weights->as.string)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    component->weights = string_duplicate(weights->as.string);
    if (component->weights == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    return parse_execution_policy(json_object_get(value, "execution"), &component->execution);
}

static trellis_status parse_manifest(
    const char * root_path,
    const char * json,
    size_t json_size,
    trellis_model_package * package) {
    json_value * document = json_parse_document(json, json_size);
    if (document == NULL) return TRELLIS_STATUS_PARSE_ERROR;
    trellis_status status = TRELLIS_STATUS_PARSE_ERROR;
    if (document->type != JSON_OBJECT || json_object_has_duplicate_keys(document)) goto cleanup;

    const json_value * schema = json_object_get(document, "schema_version");
    if (schema == NULL || schema->type != JSON_NUMBER ||
        schema->as.number != (double) TRELLIS_MODEL_PACKAGE_SCHEMA_V1) {
        goto cleanup;
    }
    package->schema_version = TRELLIS_MODEL_PACKAGE_SCHEMA_V1;
    package->source = TRELLIS_MODEL_PACKAGE_SOURCE_MANIFEST;
    package->root = string_duplicate(root_path);
    if (package->root == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    if ((status = duplicate_required_identifier(document, "id", &package->id)) != TRELLIS_STATUS_OK ||
        (status = duplicate_required_identifier(document, "family", &package->family)) != TRELLIS_STATUS_OK ||
        (status = duplicate_required_identifier(document, "task", &package->task)) != TRELLIS_STATUS_OK ||
        (status = duplicate_required_identifier(document, "profile", &package->profile)) != TRELLIS_STATUS_OK) {
        goto cleanup;
    }

    const json_value * components = json_object_get(document, "components");
    if (components == NULL || components->type != JSON_ARRAY || components->as.array.count == 0) {
        status = TRELLIS_STATUS_PARSE_ERROR;
        goto cleanup;
    }
    if (components->as.array.count > SIZE_MAX / sizeof(*package->components)) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    package->components = (trellis_model_component_instance *) calloc(
        components->as.array.count,
        sizeof(*package->components));
    if (package->components == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    package->component_count = components->as.array.count;
    for (size_t i = 0; i < package->component_count; ++i) {
        status = parse_manifest_component(components->as.array.items[i], &package->components[i]);
        if (status != TRELLIS_STATUS_OK) goto cleanup;
        for (size_t j = 0; j < i; ++j) {
            if (strcmp(package->components[i].role, package->components[j].role) == 0) {
                status = TRELLIS_STATUS_PARSE_ERROR;
                goto cleanup;
            }
        }
    }
    status = trellis_registry_validate_package(package);

cleanup:
    json_value_free(document);
    return status;
}

static trellis_status join_path(
    const char * root,
    const char * relative,
    char * output,
    size_t output_size) {
    if (root == NULL || relative == NULL || output == NULL || output_size == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    int length = snprintf(output, output_size, "%s/%s", root, relative);
    return length >= 0 && (size_t) length < output_size ?
        TRELLIS_STATUS_OK : TRELLIS_STATUS_INVALID_ARGUMENT;
}

static trellis_status read_manifest_if_present(
    const char * root,
    char ** contents_out,
    size_t * size_out,
    int * present_out) {
    char path[4096];
    trellis_status status = join_path(root, "model.json", path, sizeof(path));
    if (status != TRELLIS_STATUS_OK) return status;
    errno = 0;
    FILE * file = fopen(path, "rb");
    if (file == NULL) {
        if (errno == ENOENT) {
            *present_out = 0;
            return TRELLIS_STATUS_OK;
        }
        return TRELLIS_STATUS_IO_ERROR;
    }
    *present_out = 1;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return TRELLIS_STATUS_IO_ERROR;
    }
    long length = ftell(file);
    if (length < 0 || (unsigned long) length > TRELLIS_MODEL_MANIFEST_MAX_BYTES ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return length < 0 ? TRELLIS_STATUS_IO_ERROR : TRELLIS_STATUS_PARSE_ERROR;
    }
    char * contents = (char *) malloc((size_t) length + 1);
    if (contents == NULL) {
        fclose(file);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    const size_t bytes_read = fread(contents, 1, (size_t) length, file);
    const int close_result = fclose(file);
    if ((size_t) length != bytes_read || close_result != 0) {
        free(contents);
        return TRELLIS_STATUS_IO_ERROR;
    }
    contents[length] = '\0';
    *contents_out = contents;
    *size_out = (size_t) length;
    return TRELLIS_STATUS_OK;
}

typedef struct legacy_component_template {
    const char * role;
    const char * architecture;
    const char * weights;
    trellis_dtype compute_dtype;
    trellis_attention_mode attention;
    trellis_dtype flash_kv_dtype;
    int emulate_bf16_blocks;
} legacy_component_template;

static const legacy_component_template g_trellis2_legacy_components[] = {
    { "sparse_structure_flow", "trellis_dit_flow", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors", TRELLIS_DTYPE_BF16, TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_F16, 0 },
    { "sparse_structure_decoder", "sparse_structure_decoder", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors", TRELLIS_DTYPE_F16, TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
    { "shape_flow_512", "trellis_dit_flow", "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", TRELLIS_DTYPE_BF16, TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_F16, 0 },
    { "shape_flow_1024", "trellis_dit_flow", "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors", TRELLIS_DTYPE_BF16, TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_F16, 0 },
    { "texture_flow_512", "trellis_dit_flow", "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors", TRELLIS_DTYPE_BF16, TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_F16, 0 },
    { "texture_flow_1024", "trellis_dit_flow", "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors", TRELLIS_DTYPE_BF16, TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_F16, 0 },
    { "shape_decoder", "sparse_unet_vae_decoder", "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors", TRELLIS_DTYPE_F16, TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
    { "texture_decoder", "sparse_unet_vae_decoder", "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors", TRELLIS_DTYPE_F16, TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
};

static const legacy_component_template g_pixal3d_legacy_components[] = {
    { "sparse_structure_flow", "pixal_dit_flow", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors", TRELLIS_DTYPE_BF16, TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_BF16, 1 },
    { "sparse_structure_decoder", "sparse_structure_decoder", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors", TRELLIS_DTYPE_F16, TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
    { "shape_flow_512", "pixal_dit_flow", "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", TRELLIS_DTYPE_BF16, TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_BF16, 1 },
    { "shape_flow_1024", "pixal_dit_flow", "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors", TRELLIS_DTYPE_BF16, TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_BF16, 1 },
    { "texture_flow_1024", "pixal_dit_flow", "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors", TRELLIS_DTYPE_BF16, TRELLIS_ATTENTION_FLASH, TRELLIS_DTYPE_BF16, 1 },
    { "shape_decoder", "sparse_unet_vae_decoder", "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors", TRELLIS_DTYPE_F16, TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
    { "texture_decoder", "sparse_unet_vae_decoder", "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors", TRELLIS_DTYPE_F16, TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
    { "naf_encoder", "pixal_naf", "ckpts/naf_release.safetensors", TRELLIS_DTYPE_F32, TRELLIS_ATTENTION_NONE, TRELLIS_DTYPE_UNKNOWN, 0 },
};

static int file_exists(const char * path) {
    FILE * file = fopen(path, "rb");
    if (file == NULL) return 0;
    fclose(file);
    return 1;
}

static trellis_status resolve_legacy_marker_path(
    const char * root,
    char * output,
    size_t output_size) {
    static const char * candidates[] = {
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "ss_flow_img_dit_1_3B_64_bf16.safetensors",
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        trellis_status status = join_path(root, candidates[i], output, output_size);
        if (status != TRELLIS_STATUS_OK) return status;
        if (file_exists(output)) return TRELLIS_STATUS_OK;
    }
    return TRELLIS_STATUS_NOT_FOUND;
}

static trellis_status populate_legacy_package(
    const char * root,
    int is_pixal,
    trellis_model_package * package) {
    const legacy_component_template * templates = is_pixal ?
        g_pixal3d_legacy_components : g_trellis2_legacy_components;
    const size_t count = is_pixal ?
        sizeof(g_pixal3d_legacy_components) / sizeof(g_pixal3d_legacy_components[0]) :
        sizeof(g_trellis2_legacy_components) / sizeof(g_trellis2_legacy_components[0]);
    package->schema_version = TRELLIS_MODEL_PACKAGE_SCHEMA_V1;
    package->source = TRELLIS_MODEL_PACKAGE_SOURCE_LEGACY;
    package->root = string_duplicate(root);
    package->id = string_duplicate(is_pixal ? "pixal3d-legacy" : "trellis2-legacy");
    package->family = string_duplicate(is_pixal ? "pixal3d" : "trellis2");
    package->task = string_duplicate("image_to_3d");
    package->profile = string_duplicate("1024_cascade");
    if (package->root == NULL || package->id == NULL || package->family == NULL ||
        package->task == NULL || package->profile == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    package->components = (trellis_model_component_instance *) calloc(count, sizeof(*package->components));
    if (package->components == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    package->component_count = count;
    for (size_t i = 0; i < count; ++i) {
        trellis_model_component_instance * component = &package->components[i];
        component->role = string_duplicate(templates[i].role);
        component->architecture = string_duplicate(templates[i].architecture);
        component->weights = string_duplicate(templates[i].weights);
        component->execution.struct_size = sizeof(component->execution);
        component->execution.compute_dtype = templates[i].compute_dtype;
        component->execution.attention = templates[i].attention;
        component->execution.flash_kv_dtype = templates[i].flash_kv_dtype;
        component->execution.emulate_bf16_blocks = templates[i].emulate_bf16_blocks;
        if (component->role == NULL || component->architecture == NULL || component->weights == NULL) {
            return TRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    return trellis_registry_validate_package(package);
}

static trellis_status load_legacy_package(
    const char * root,
    trellis_model_package * package) {
    char marker_path[4096];
    trellis_status status = resolve_legacy_marker_path(root, marker_path, sizeof(marker_path));
    if (status != TRELLIS_STATUS_OK) return status;

    trellis_safetensors checkpoint;
    memset(&checkpoint, 0, sizeof(checkpoint));
    ++package->legacy_pixal_marker_probes;
    status = trellis_safetensors_open(marker_path, &checkpoint);
    if (status != TRELLIS_STATUS_OK) return status;
    const int is_pixal = trellis_safetensors_find(
        &checkpoint,
        "blocks.0.cross_attn.proj_linear.weight") != NULL;
    trellis_safetensors_close(&checkpoint);
    return populate_legacy_package(root, is_pixal, package);
}

void trellis_model_package_free(trellis_model_package * package) {
    if (package == NULL) return;
    free(package->root);
    free(package->id);
    free(package->family);
    free(package->task);
    free(package->profile);
    for (size_t i = 0; i < package->component_count; ++i) {
        free(package->components[i].role);
        free(package->components[i].architecture);
        free(package->components[i].weights);
    }
    free(package->components);
    *package = (trellis_model_package) TRELLIS_MODEL_PACKAGE_INIT;
}

const trellis_model_component_instance * trellis_model_package_find_component(
    const trellis_model_package * package,
    const char * role) {
    if (package == NULL || role == NULL) return NULL;
    for (size_t i = 0; i < package->component_count; ++i) {
        if (strcmp(package->components[i].role, role) == 0) return &package->components[i];
    }
    return NULL;
}

trellis_status trellis_model_package_resolve_component_path(
    const trellis_model_package * package,
    const char * role,
    char * path_out,
    size_t path_size) {
    if (package == NULL || package->root == NULL || role == NULL ||
        path_out == NULL || path_size == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const trellis_model_component_instance * component =
        trellis_model_package_find_component(package, role);
    if (component == NULL) {
        return TRELLIS_STATUS_NOT_FOUND;
    }
    return join_path(package->root, component->weights, path_out, path_size);
}

trellis_status trellis_model_package_load(
    const char * root,
    trellis_model_package * package_out) {
    if (root == NULL || root[0] == '\0' || package_out == NULL ||
        package_out->struct_size < sizeof(trellis_model_package)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    char * manifest = NULL;
    size_t manifest_size = 0;
    int manifest_present = 0;
    trellis_status status = read_manifest_if_present(
        root,
        &manifest,
        &manifest_size,
        &manifest_present);
    if (status != TRELLIS_STATUS_OK) return status;

    trellis_model_package loaded = TRELLIS_MODEL_PACKAGE_INIT;
    if (manifest_present) {
        status = parse_manifest(root, manifest, manifest_size, &loaded);
    } else {
        status = load_legacy_package(root, &loaded);
    }
    free(manifest);
    if (status != TRELLIS_STATUS_OK) {
        trellis_model_package_free(&loaded);
        return status;
    }
    trellis_model_package_free(package_out);
    *package_out = loaded;
    return TRELLIS_STATUS_OK;
}
