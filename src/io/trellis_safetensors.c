#include "trellis.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t read_le_u64(const unsigned char b[8]) {
    uint64_t x = 0;
    for (int i = 7; i >= 0; --i) {
        x = (x << 8) | b[i];
    }
    return x;
}

const char * trellis_dtype_name(trellis_dtype dtype) {
    switch (dtype) {
        case TRELLIS_DTYPE_F32: return "F32";
        case TRELLIS_DTYPE_F16: return "F16";
        case TRELLIS_DTYPE_BF16: return "BF16";
        case TRELLIS_DTYPE_I64: return "I64";
        case TRELLIS_DTYPE_I32: return "I32";
        case TRELLIS_DTYPE_U8: return "U8";
        case TRELLIS_DTYPE_BOOL: return "BOOL";
        default: return "UNKNOWN";
    }
}

size_t trellis_dtype_size(trellis_dtype dtype) {
    switch (dtype) {
        case TRELLIS_DTYPE_F32: return 4;
        case TRELLIS_DTYPE_F16: return 2;
        case TRELLIS_DTYPE_BF16: return 2;
        case TRELLIS_DTYPE_I64: return 8;
        case TRELLIS_DTYPE_I32: return 4;
        case TRELLIS_DTYPE_U8: return 1;
        case TRELLIS_DTYPE_BOOL: return 1;
        default: return 0;
    }
}

static trellis_dtype parse_dtype(const char * s) {
    if (strcmp(s, "F32") == 0) return TRELLIS_DTYPE_F32;
    if (strcmp(s, "F16") == 0) return TRELLIS_DTYPE_F16;
    if (strcmp(s, "BF16") == 0) return TRELLIS_DTYPE_BF16;
    if (strcmp(s, "I64") == 0) return TRELLIS_DTYPE_I64;
    if (strcmp(s, "I32") == 0) return TRELLIS_DTYPE_I32;
    if (strcmp(s, "U8") == 0) return TRELLIS_DTYPE_U8;
    if (strcmp(s, "BOOL") == 0) return TRELLIS_DTYPE_BOOL;
    return TRELLIS_DTYPE_UNKNOWN;
}

static char * dup_range(const char * begin, const char * end) {
    if (end < begin) {
        return NULL;
    }
    size_t n = (size_t) (end - begin);
    char * out = (char *) malloc(n + 1);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, begin, n);
    out[n] = '\0';
    return out;
}

static void skip_ws(const char ** p) {
    while (**p != '\0' && isspace((unsigned char) **p)) {
        ++(*p);
    }
}

static char * parse_string(const char ** p) {
    skip_ws(p);
    if (**p != '"') {
        return NULL;
    }
    ++(*p);
    char * buf = NULL;
    size_t len = 0;
    size_t cap = 0;
    while (**p != '\0' && **p != '"') {
        unsigned char ch = (unsigned char) **p;
        if (ch == '\\') {
            ++(*p);
            ch = (unsigned char) **p;
            if (ch == '\0') {
                free(buf);
                return NULL;
            }
            switch (ch) {
                case '"': ch = '"'; break;
                case '\\': ch = '\\'; break;
                case '/': ch = '/'; break;
                case 'b': ch = '\b'; break;
                case 'f': ch = '\f'; break;
                case 'n': ch = '\n'; break;
                case 'r': ch = '\r'; break;
                case 't': ch = '\t'; break;
                case 'u':
                    ch = '?';
                    for (int i = 0; i < 4 && (*p)[1] != '\0'; ++i) {
                        ++(*p);
                    }
                    break;
                default:
                    break;
            }
        }
        if (len + 1 >= cap) {
            size_t new_cap = cap == 0 ? 32 : cap * 2;
            char * nb = (char *) realloc(buf, new_cap);
            if (nb == NULL) {
                free(buf);
                return NULL;
            }
            buf = nb;
            cap = new_cap;
        }
        buf[len++] = (char) ch;
        ++(*p);
    }
    if (**p != '"') {
        free(buf);
        return NULL;
    }
    ++(*p);
    if (len + 1 >= cap) {
        char * nb = (char *) realloc(buf, len + 1);
        if (nb == NULL) {
            free(buf);
            return NULL;
        }
        buf = nb;
    }
    buf[len] = '\0';
    return buf;
}

static bool parse_u64(const char ** p, uint64_t * out) {
    skip_ws(p);
    if (!isdigit((unsigned char) **p)) {
        return false;
    }
    uint64_t value = 0;
    while (isdigit((unsigned char) **p)) {
        value = value * 10 + (uint64_t) (**p - '0');
        ++(*p);
    }
    *out = value;
    return true;
}

static bool parse_i64(const char ** p, int64_t * out) {
    skip_ws(p);
    int sign = 1;
    if (**p == '-') {
        sign = -1;
        ++(*p);
    }
    uint64_t v = 0;
    if (!parse_u64(p, &v)) {
        return false;
    }
    *out = sign < 0 ? -(int64_t) v : (int64_t) v;
    return true;
}

static bool skip_json_value(const char ** p);

static bool skip_compound_raw(const char ** p, char open, char close) {
    if (**p != open) {
        return false;
    }
    int depth = 0;
    while (**p != '\0') {
        if (**p == '"') {
            char * s = parse_string(p);
            if (s == NULL) {
                return false;
            }
            free(s);
            continue;
        }
        if (**p == open) {
            ++depth;
        } else if (**p == close) {
            --depth;
            ++(*p);
            return depth == 0;
        }
        ++(*p);
    }
    return false;
}

static bool skip_number_or_literal(const char ** p) {
    const char * start = *p;
    while (**p != '\0' && !isspace((unsigned char) **p) &&
           **p != ',' && **p != ']' && **p != '}') {
        ++(*p);
    }
    return *p > start;
}

static bool skip_json_value(const char ** p) {
    skip_ws(p);
    if (**p == '"') {
        char * s = parse_string(p);
        if (s == NULL) {
            return false;
        }
        free(s);
        return true;
    }
    if (**p == '{') {
        return skip_compound_raw(p, '{', '}');
    }
    if (**p == '[') {
        return skip_compound_raw(p, '[', ']');
    }
    return skip_number_or_literal(p);
}

static bool parse_shape(const char ** p, trellis_safetensor_meta * meta) {
    skip_ws(p);
    if (**p != '[') {
        return false;
    }
    ++(*p);
    meta->n_dims = 0;
    skip_ws(p);
    while (**p != '\0' && **p != ']') {
        if (meta->n_dims >= TRELLIS_MAX_DIMS) {
            return false;
        }
        int64_t dim = 0;
        if (!parse_i64(p, &dim) || dim < 0) {
            return false;
        }
        meta->shape[meta->n_dims++] = dim;
        skip_ws(p);
        if (**p == ',') {
            ++(*p);
            skip_ws(p);
        } else if (**p != ']') {
            return false;
        }
    }
    if (**p != ']') {
        return false;
    }
    ++(*p);
    return true;
}

static bool parse_offsets(const char ** p, trellis_safetensor_meta * meta) {
    skip_ws(p);
    if (**p != '[') {
        return false;
    }
    ++(*p);
    if (!parse_u64(p, &meta->data_begin)) {
        return false;
    }
    skip_ws(p);
    if (**p != ',') {
        return false;
    }
    ++(*p);
    if (!parse_u64(p, &meta->data_end)) {
        return false;
    }
    skip_ws(p);
    if (**p != ']') {
        return false;
    }
    ++(*p);
    return true;
}

static bool parse_tensor_object(const char ** p, const char * name, trellis_safetensor_meta * meta) {
    memset(meta, 0, sizeof(*meta));
    meta->name = dup_range(name, name + strlen(name));
    if (meta->name == NULL) {
        return false;
    }

    skip_ws(p);
    if (**p != '{') {
        return false;
    }
    ++(*p);
    skip_ws(p);
    while (**p != '\0' && **p != '}') {
        char * key = parse_string(p);
        if (key == NULL) {
            return false;
        }
        skip_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        ++(*p);
        bool ok = true;
        if (strcmp(key, "dtype") == 0) {
            char * dtype = parse_string(p);
            if (dtype == NULL) {
                ok = false;
            } else {
                meta->dtype = parse_dtype(dtype);
                free(dtype);
            }
        } else if (strcmp(key, "shape") == 0) {
            ok = parse_shape(p, meta);
        } else if (strcmp(key, "data_offsets") == 0) {
            ok = parse_offsets(p, meta);
        } else {
            ok = skip_json_value(p);
        }
        free(key);
        if (!ok) {
            return false;
        }
        skip_ws(p);
        if (**p == ',') {
            ++(*p);
            skip_ws(p);
        } else if (**p != '}') {
            return false;
        }
    }
    if (**p != '}') {
        return false;
    }
    ++(*p);
    return meta->dtype != TRELLIS_DTYPE_UNKNOWN && meta->data_end >= meta->data_begin;
}

static bool append_meta(trellis_safetensors * st, trellis_safetensor_meta * meta) {
    trellis_safetensor_meta * next = (trellis_safetensor_meta *) realloc(
        st->tensors,
        (st->n_tensors + 1) * sizeof(st->tensors[0]));
    if (next == NULL) {
        return false;
    }
    st->tensors = next;
    st->tensors[st->n_tensors++] = *meta;
    memset(meta, 0, sizeof(*meta));
    return true;
}

static trellis_status parse_header(trellis_safetensors * st) {
    const char * p = st->header_json;
    skip_ws(&p);
    if (*p != '{') {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    ++p;
    skip_ws(&p);
    while (*p != '\0' && *p != '}') {
        char * key = parse_string(&p);
        if (key == NULL) {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        skip_ws(&p);
        if (*p != ':') {
            free(key);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        ++p;
        if (strcmp(key, "__metadata__") == 0) {
            if (!skip_json_value(&p)) {
                free(key);
                return TRELLIS_STATUS_PARSE_ERROR;
            }
        } else {
            trellis_safetensor_meta meta;
            if (!parse_tensor_object(&p, key, &meta)) {
                free(key);
                return TRELLIS_STATUS_PARSE_ERROR;
            }
            if (!append_meta(st, &meta)) {
                free(key);
                free(meta.name);
                return TRELLIS_STATUS_OUT_OF_MEMORY;
            }
        }
        free(key);
        skip_ws(&p);
        if (*p == ',') {
            ++p;
            skip_ws(&p);
        } else if (*p != '}') {
            return TRELLIS_STATUS_PARSE_ERROR;
        }
    }
    if (*p != '}') {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_safetensors_open(const char * path, trellis_safetensors * out) {
    if (path == NULL || out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));

    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return TRELLIS_STATUS_IO_ERROR;
    }

    unsigned char nbuf[8];
    if (fread(nbuf, 1, 8, f) != 8) {
        fclose(f);
        return TRELLIS_STATUS_IO_ERROR;
    }
    out->header_size = read_le_u64(nbuf);
    out->data_base_offset = 8 + out->header_size;
    out->header_json = (char *) malloc((size_t) out->header_size + 1);
    if (out->header_json == NULL) {
        fclose(f);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    if (fread(out->header_json, 1, (size_t) out->header_size, f) != (size_t) out->header_size) {
        fclose(f);
        trellis_safetensors_close(out);
        return TRELLIS_STATUS_IO_ERROR;
    }
    fclose(f);
    out->header_json[out->header_size] = '\0';
    out->path = dup_range(path, path + strlen(path));
    if (out->path == NULL) {
        trellis_safetensors_close(out);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_status status = parse_header(out);
    if (status != TRELLIS_STATUS_OK) {
        trellis_safetensors_close(out);
    }
    return status;
}

void trellis_safetensors_close(trellis_safetensors * st) {
    if (st == NULL) {
        return;
    }
    free(st->path);
    free(st->header_json);
    for (size_t i = 0; i < st->n_tensors; ++i) {
        free(st->tensors[i].name);
    }
    free(st->tensors);
    memset(st, 0, sizeof(*st));
}

const trellis_safetensor_meta * trellis_safetensors_find(const trellis_safetensors * st, const char * name) {
    if (st == NULL || name == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < st->n_tensors; ++i) {
        if (strcmp(st->tensors[i].name, name) == 0) {
            return &st->tensors[i];
        }
    }
    return NULL;
}

uint64_t trellis_safetensor_nelements(const trellis_safetensor_meta * meta) {
    if (meta == NULL) {
        return 0;
    }
    uint64_t n = 1;
    for (int i = 0; i < meta->n_dims; ++i) {
        n *= (uint64_t) meta->shape[i];
    }
    return n;
}

trellis_status trellis_safetensors_read_f32(
    const trellis_safetensors * st,
    const trellis_safetensor_meta * meta,
    float * dst,
    size_t dst_count) {
    if (st == NULL || meta == NULL || dst == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    uint64_t n = trellis_safetensor_nelements(meta);
    if ((uint64_t) dst_count < n) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (n > (uint64_t) SIZE_MAX) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    size_t count = (size_t) n;
    size_t elem_size = trellis_dtype_size(meta->dtype);
    if (elem_size == 0 || n > UINT64_MAX / (uint64_t) elem_size ||
        meta->data_end < meta->data_begin ||
        meta->data_end - meta->data_begin != n * (uint64_t) elem_size) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    FILE * f = fopen(st->path, "rb");
    if (f == NULL) {
        return TRELLIS_STATUS_IO_ERROR;
    }
    if (fseek(f, (long) (st->data_base_offset + meta->data_begin), SEEK_SET) != 0) {
        fclose(f);
        return TRELLIS_STATUS_IO_ERROR;
    }

    trellis_status status = TRELLIS_STATUS_OK;
    if (meta->dtype == TRELLIS_DTYPE_F32) {
        if (fread(dst, sizeof(float), count, f) != count) {
            status = TRELLIS_STATUS_IO_ERROR;
        }
    } else if (count > SIZE_MAX / elem_size) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
    } else {
        const size_t bytes = count * elem_size;
        unsigned char * raw = (unsigned char *) malloc(bytes == 0 ? 1 : bytes);
        if (raw == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else if (fread(raw, elem_size, count, f) != count) {
            status = TRELLIS_STATUS_IO_ERROR;
        } else if (meta->dtype == TRELLIS_DTYPE_F16) {
            const ggml_fp16_t * src = (const ggml_fp16_t *) raw;
            for (size_t i = 0; i < count; ++i) {
                dst[i] = ggml_fp16_to_fp32(src[i]);
            }
        } else if (meta->dtype == TRELLIS_DTYPE_BF16) {
            const ggml_bf16_t * src = (const ggml_bf16_t *) raw;
            for (size_t i = 0; i < count; ++i) {
                dst[i] = ggml_bf16_to_fp32(src[i]);
            }
        } else if (meta->dtype == TRELLIS_DTYPE_I64) {
            const int64_t * src = (const int64_t *) raw;
            for (size_t i = 0; i < count; ++i) {
                dst[i] = (float) src[i];
            }
        } else if (meta->dtype == TRELLIS_DTYPE_I32) {
            const int32_t * src = (const int32_t *) raw;
            for (size_t i = 0; i < count; ++i) {
                dst[i] = (float) src[i];
            }
        } else if (meta->dtype == TRELLIS_DTYPE_U8 || meta->dtype == TRELLIS_DTYPE_BOOL) {
            for (size_t i = 0; i < count; ++i) {
                dst[i] = (float) raw[i];
            }
        } else {
            status = TRELLIS_STATUS_PARSE_ERROR;
        }
        free(raw);
    }

    fclose(f);
    return status;
}
