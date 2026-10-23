/* trellis2-server — headless HTTP server for TRELLIS.2 image-to-3D (GLB output).
 *
 * Wraps the trellis2.c image-to-3D pipeline so long-lived hosts (e.g.
 * quartermaster) can spawn it once and reverse-proxy requests, instead of
 * paying the checkpoint load on every invocation. One process, one model, one
 * GPU: requests serialize, and the pipeline's model cache means the second and
 * later requests skip the weight load entirely.
 *
 * Usage:
 *   trellis2-server --model DIR [--dino DIR] [--birefnet FILE] [--vkmesh FILE]
 *                   [--host 127.0.0.1] [--port 8080] [--device N]
 *                   [--pipeline 512|1024|1024_cascade|1536_cascade]
 *                   [--texture-size N] [--steps N] [--model-cache-budget-mib N]
 *                   [--out-dir DIR] [--max-body-mib N] [--verbose]
 *
 * Endpoints:
 *   GET  /health              -> 200 while the process is alive; the body says
 *                                whether the weight paths are there
 *   GET  /ready               -> 200 only when every required path exists
 *   POST /generate            -> image bytes in, GLB bytes out
 *   POST /v1/3d/generations   -> the same handler (quartermaster's route shape)
 *   GET  /output/NAME         -> a retained GLB from --out-dir
 *
 * Request: the image IS the body (image/png, image/jpeg or image/webp; the
 * loader sniffs content, so the extension is cosmetic), and every knob is an
 * optional query parameter:
 *
 *   seed=<u32>            noise seed for the flow samplers
 *   noise_seed=<u32>      seed for the initial latent noise
 *   pipeline=<name>       512 (default), 1024, 1024_cascade, 1536_cascade
 *   steps=<n>             sampler steps, stages 1 and 2
 *   ss_steps=<n>          stage 1 only
 *   slat_steps=<n>        stage 2 only
 *   texture_size=<n>      texture resolution (default 1024)
 *   shape_only=1          skip texturing, export an untextured mesh
 *   keep=1                leave the GLB on disk and answer with JSON
 *   format=json           same as keep=1 (implies it)
 *
 * Response: model/gltf-binary by default, with X-Trellis-Seconds and
 * X-Trellis-File headers. With keep/format=json the body is
 * {"file":"...","path":"...","bytes":N,"seconds":X}. Errors are JSON
 * {"error":"..."} with a 4xx/5xx status.
 *
 * There is ONE generation at a time on purpose: the pipeline owns the GPU and
 * its model cache, so a second request waits in the listen backlog (and one
 * that arrives while a generation runs gets 409 rather than a queue slot).
 *
 * The server never downloads anything. Weights are the caller's business:
 * tools/download_weights.py builds the layout --model/--dino/--birefnet want.
 */

#define _POSIX_C_SOURCE 200809L

#include "trellis.h"
#include "trellis_platform.h"
#include "trellis_tool_cli.h"

#include "ggml.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <sys/stat.h>
#  include <winsock2.h>
#  include <ws2tcpip.h>
typedef SOCKET trellis_socket_t;
#  define TRELLIS_SOCKET_INVALID INVALID_SOCKET
#  define trellis_socket_close closesocket
#else
#  include <arpa/inet.h>
#  include <dirent.h>
#  include <netinet/in.h>
#  include <signal.h>
#  include <sys/socket.h>
#  include <sys/stat.h>
#  include <unistd.h>
typedef int trellis_socket_t;
#  define TRELLIS_SOCKET_INVALID (-1)
#  define trellis_socket_close close
#endif

#ifndef TRELLIS_DEFAULT_BACKEND
#  define TRELLIS_DEFAULT_BACKEND "unknown"
#endif

#define TRELLIS_SERVER_MAX_HEADER_BYTES (64u * 1024u)
#define TRELLIS_SERVER_MAX_PATH_BYTES 1024u
#define TRELLIS_SERVER_MAX_QUERY_BYTES 2048u
#define TRELLIS_SERVER_DEFAULT_MAX_BODY_MIB 64u
#define TRELLIS_SERVER_IO_CHUNK (64u * 1024u)

typedef struct trellis_server_options {
    const char * host;
    int port;
    const char * model_dir;
    const char * dino_dir;
    const char * birefnet_path;
    const char * vkmesh_path;
    const char * out_dir;
    int device;
    const char * pipeline;
    int texture_size;
    int steps;
    int model_cache_budget_mib;
    int mesh_no_simplify;
    int mesh_decimation_target;
    size_t max_body_bytes;
} trellis_server_options;

typedef struct trellis_http_request {
    char method[16];
    char path[TRELLIS_SERVER_MAX_PATH_BYTES];
    char query[TRELLIS_SERVER_MAX_QUERY_BYTES];
    char content_type[128];
    size_t content_length;
    int has_content_length;
    int expect_continue;
    int chunked;
} trellis_http_request;

typedef struct trellis_generate_params {
    char pipeline[64];
    uint32_t seed;
    uint32_t noise_seed;
    int sparse_steps;
    int latent_steps;
    int texture_size;
    int shape_only;
    int keep;
} trellis_generate_params;

static uint64_t g_request_counter = 0;

/* ------------------------------------------------------------------------- */
/* small helpers
 * ------------------------------------------------------------------------- */

static int string_ieq_n(const char * a, const char * b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        char ca = a[i];
        char cb = b[i];
        if (ca >= 'A' && ca <= 'Z') ca = (char) (ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (char) (cb - 'A' + 'a');
        if (ca != cb) {
            return 0;
        }
        if (ca == '\0') {
            return 1;
        }
    }
    return 1;
}

static int parse_int_arg(const char * text, int * out) {
    if (text == NULL || text[0] == '\0') {
        return 0;
    }
    char * end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < INT_MIN || value > INT_MAX) {
        return 0;
    }
    *out = (int) value;
    return 1;
}

static int parse_u32_arg(const char * text, uint32_t * out) {
    if (text == NULL || text[0] == '\0') {
        return 0;
    }
    char * end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > 0xfffffffful) {
        return 0;
    }
    *out = (uint32_t) value;
    return 1;
}

static const char * basename_of(const char * path) {
    if (path == NULL) {
        return "";
    }
    const char * last = path;
    for (const char * p = path; *p != '\0'; ++p) {
        if (*p == '/' || *p == '\\') {
            last = p + 1;
        }
    }
    return last;
}

/* MSVC's struct stat carries a 32-bit st_size unless the 64-bit variant is
 * asked for, and its fseek cannot seek past 2 GiB. Both matter here: TRELLIS.2
 * checkpoints are 2.5 GB, so this is the one place that must use the 64-bit
 * calls or a complete download looks truncated. */
static int stat_path_ex(const char * path, int * is_dir, long long * size) {
#if defined(_WIN32)
    struct _stati64 info;
    if (path == NULL || path[0] == '\0' || _stati64(path, &info) != 0) {
        return 0;
    }
    if (is_dir != NULL) {
        /* MSVC's sys/stat.h has no S_ISDIR. */
        *is_dir = (info.st_mode & _S_IFDIR) != 0;
    }
#else
    struct stat info;
    if (path == NULL || path[0] == '\0' || stat(path, &info) != 0) {
        return 0;
    }
    if (is_dir != NULL) {
        *is_dir = S_ISDIR(info.st_mode) != 0;
    }
#endif
    if (size != NULL) {
        *size = (long long) info.st_size;
    }
    return 1;
}

static int stat_path(const char * path, int * is_dir) {
    return stat_path_ex(path, is_dir, NULL);
}

/* Separators follow the host: stat() on Windows accepts '/' as well, and every
 * path we build keeps whatever the caller passed. */
static int join_path(char * out, size_t out_size, const char * dir, const char * name) {
    const int written = snprintf(out, out_size, "%s%c%s", dir, '/', name);
    return written > 0 && (size_t) written < out_size;
}

/* A weight directory is only usable once its manifest and checkpoint folder are
 * both on disk; a half-finished download must not read as ready. */
static int dir_has(const char * dir, const char * name, int want_directory) {
    char path[1200];
    int is_dir = 0;
    if (dir == NULL || dir[0] == '\0' || !join_path(path, sizeof(path), dir, name)) {
        return 0;
    }
    if (!stat_path(path, &is_dir)) {
        return 0;
    }
    return want_directory ? is_dir : !is_dir;
}

static int path_exists(const char * path) {
    return stat_path(path, NULL);
}

/* Directory enumeration, so the readiness check can look at what is actually on
 * disk instead of guessing from a filename. */
#if defined(_WIN32)
typedef struct dir_iter {
    HANDLE handle;
    WIN32_FIND_DATAA data;
    int first;
} dir_iter;

static int dir_iter_open(dir_iter * it, const char * path) {
    char pattern[1200];
    if (!join_path(pattern, sizeof(pattern), path, "*")) {
        return 0;
    }
    it->handle = FindFirstFileA(pattern, &it->data);
    it->first = 1;
    return it->handle != INVALID_HANDLE_VALUE;
}

static const char * dir_iter_next(dir_iter * it) {
    if (it->first) {
        it->first = 0;
        return it->data.cFileName;
    }
    return FindNextFileA(it->handle, &it->data) ? it->data.cFileName : NULL;
}

static void dir_iter_close(dir_iter * it) {
    if (it->handle != INVALID_HANDLE_VALUE) {
        FindClose(it->handle);
    }
}
#else
typedef struct dir_iter {
    DIR * dir;
    struct dirent * entry;
} dir_iter;

static int dir_iter_open(dir_iter * it, const char * path) {
    it->dir = opendir(path);
    it->entry = NULL;
    return it->dir != NULL;
}

static const char * dir_iter_next(dir_iter * it) {
    it->entry = readdir(it->dir);
    return it->entry == NULL ? NULL : it->entry->d_name;
}

static void dir_iter_close(dir_iter * it) {
    if (it->dir != NULL) {
        closedir(it->dir);
    }
}
#endif

static long long file_size_bytes(const char * path);
static int ckpt_weights_complete(const char * model_dir, char * missing, size_t missing_size);

/* One predicate for "this request can actually run", shared by /ready and
 * /generate so a doomed request is refused in microseconds instead of spending
 * minutes loading a half-finished weight set. */
static int weights_ready(const trellis_server_options * options, char * reason, size_t reason_size) {
    const char * model_dir = options->model_dir == NULL ? "" : options->model_dir;
    const char * dino_dir = options->dino_dir == NULL ? "" : options->dino_dir;

    if (!dir_has(model_dir, "pipeline.json", 0) || !dir_has(model_dir, "ckpts", 1)) {
        snprintf(reason, reason_size, "no TRELLIS.2 package at '%s' (need pipeline.json and ckpts/)", model_dir);
        return 0;
    }

    char missing[1200];
    if (!ckpt_weights_complete(model_dir, missing, sizeof(missing))) {
        snprintf(reason, reason_size, "'%s' is absent or empty; the download is unfinished", missing);
        return 0;
    }

    if (!dir_has(dino_dir, "model.safetensors", 0) || !dir_has(dino_dir, "config.json", 0)) {
        snprintf(reason, reason_size, "DINOv3 weights are missing or incomplete at '%s'", dino_dir);
        return 0;
    }

    return 1;
}

static int has_suffix(const char * text, const char * suffix) {
    const size_t text_len = strlen(text);
    const size_t suffix_len = strlen(suffix);
    return text_len > suffix_len && strcmp(text + text_len - suffix_len, suffix) == 0;
}

/* The loader wants one .safetensors per ckpts/*.json manifest. A snapshot
 * download writes the manifests first and the weights one file at a time, so
 * "every manifest has a weight file" is exactly the test that separates a
 * usable package from one that is still arriving. Fills `missing` with the
 * first absent weight file for the caller's error message. */
static int ckpt_weights_complete(const char * model_dir, char * missing, size_t missing_size) {
    char ckpts_dir[1200];
    if (missing != NULL && missing_size > 0) {
        missing[0] = '\0';
    }
    if (model_dir == NULL || model_dir[0] == '\0' ||
        !join_path(ckpts_dir, sizeof(ckpts_dir), model_dir, "ckpts")) {
        return 0;
    }

    dir_iter it;
    if (!dir_iter_open(&it, ckpts_dir)) {
        return 0;
    }

    int manifests = 0;
    int complete = 1;
    for (const char * name = dir_iter_next(&it); name != NULL; name = dir_iter_next(&it)) {
        if (!has_suffix(name, ".json")) {
            continue;
        }
        ++manifests;

        char stem[1024];
        const size_t stem_len = strlen(name) - strlen(".json");
        if (stem_len >= sizeof(stem)) {
            continue;
        }
        memcpy(stem, name, stem_len);
        stem[stem_len] = '\0';

        char weights_name[1100];
        char weights[1200];
        snprintf(weights_name, sizeof(weights_name), "%s.safetensors", stem);
        if (!join_path(weights, sizeof(weights), ckpts_dir, weights_name) ||
            !path_exists(weights) || file_size_bytes(weights) <= 0) {
            complete = 0;
            if (missing != NULL && missing_size > 0) {
                snprintf(missing, missing_size, "ckpts/%s", weights_name);
            }
            break;
        }
    }
    dir_iter_close(&it);

    return complete && manifests > 0;
}

static long long file_size_bytes(const char * path) {
    long long size = -1;
    if (!stat_path_ex(path, NULL, &size)) {
        return -1;
    }
    return size;
}

static const char * out_dir_of(const trellis_server_options * options) {
    if (options->out_dir != NULL && options->out_dir[0] != '\0') {
        return options->out_dir;
    }
    const char * env = getenv("TRELLIS2_OUT_DIR");
    if (env != NULL && env[0] != '\0') {
        return env;
    }
    env = getenv("TMPDIR");
    if (env != NULL && env[0] != '\0') {
        return env;
    }
#if defined(_WIN32)
    env = getenv("TEMP");
    if (env != NULL && env[0] != '\0') {
        return env;
    }
    env = getenv("TMP");
    if (env != NULL && env[0] != '\0') {
        return env;
    }
#endif
    return ".";
}

static void make_scratch_path(
    const trellis_server_options * options,
    uint64_t request_id,
    const char * suffix,
    char * out,
    size_t out_size) {
    snprintf(
        out,
        out_size,
        "%s%ctrellis2-%d-%llu%s",
        out_dir_of(options),
        '/',
        options->port,
        (unsigned long long) request_id,
        suffix);
}

/* ------------------------------------------------------------------------- */
/* HTTP plumbing (single-threaded, Connection: close, no keep-alive)
 * ------------------------------------------------------------------------- */

static int send_all(trellis_socket_t sock, const void * data, size_t len) {
    const char * cursor = (const char *) data;
    while (len > 0) {
        int chunk = len > (size_t) INT_MAX ? INT_MAX : (int) len;
        int sent = send(sock, cursor, chunk, 0);
        if (sent <= 0) {
            return 0;
        }
        cursor += sent;
        len -= (size_t) sent;
    }
    return 1;
}

static int send_text(trellis_socket_t sock, const char * text) {
    return send_all(sock, text, strlen(text));
}

static void json_escape_into(const char * text, char * out, size_t out_size) {
    size_t pos = 0;
    for (const char * p = text; p != NULL && *p != '\0' && pos + 7 < out_size; ++p) {
        const char * escape = NULL;
        char small[8];
        switch (*p) {
            case '"': escape = "\\\""; break;
            case '\\': escape = "\\\\"; break;
            case '\n': escape = "\\n"; break;
            case '\r': escape = "\\r"; break;
            case '\t': escape = "\\t"; break;
            default:
                if ((unsigned char) *p < 0x20) {
                    snprintf(small, sizeof(small), "\\u%04x", (unsigned char) *p);
                    escape = small;
                }
                break;
        }
        if (escape != NULL) {
            size_t n = strlen(escape);
            memcpy(out + pos, escape, n);
            pos += n;
        } else {
            out[pos++] = *p;
        }
    }
    out[pos] = '\0';
}

static const char * status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 409: return "Conflict";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 415: return "Unsupported Media Type";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default: return "Error";
    }
}

static int send_json_error(trellis_socket_t sock, int status, const char * fmt, ...) {
    char message[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);

    char escaped[2048];
    json_escape_into(message, escaped, sizeof(escaped));

    char body[2304];
    int body_len = snprintf(body, sizeof(body), "{\"error\":\"%s\"}\n", escaped);
    if (body_len < 0 || (size_t) body_len >= sizeof(body)) {
        body_len = (int) strlen(body);
    }

    char head[512];
    int head_len = snprintf(
        head,
        sizeof(head),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n"
        "\r\n",
        status,
        status_text(status),
        body_len);

    TRELLIS_TOOL_ERROR("http %d: %s", status, message);
    return send_all(sock, head, (size_t) head_len) && send_all(sock, body, (size_t) body_len);
}

static int send_json_ok(trellis_socket_t sock, const char * body, size_t body_len) {
    char head[512];
    int head_len = snprintf(
        head,
        sizeof(head),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n"
        "\r\n",
        body_len);
    return send_all(sock, head, (size_t) head_len) && send_all(sock, body, body_len);
}

/* Sends a file as the response body, then deletes it unless keep_file is set. */
static int send_file_response(
    trellis_socket_t sock,
    const char * path,
    const char * content_type,
    double seconds,
    int keep_file) {
    long long size = file_size_bytes(path);
    if (size < 0) {
        return send_json_error(sock, 500, "output file '%s' is missing", path);
    }

    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return send_json_error(sock, 500, "cannot open output file '%s'", path);
    }

    char head[640];
    int head_len = snprintf(
        head,
        sizeof(head),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %lld\r\n"
        "X-Trellis-Seconds: %.2f\r\n"
        "X-Trellis-File: %s\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n"
        "\r\n",
        content_type,
        size,
        seconds,
        basename_of(path));

    int ok = send_all(sock, head, (size_t) head_len);
    if (ok) {
        char chunk[TRELLIS_SERVER_IO_CHUNK];
        size_t read_bytes = 0;
        while ((read_bytes = fread(chunk, 1, sizeof(chunk), f)) > 0) {
            if (!send_all(sock, chunk, read_bytes)) {
                ok = 0;
                break;
            }
        }
        if (ferror(f)) {
            ok = 0;
        }
    }
    fclose(f);

    if (!keep_file) {
        remove(path);
    }
    return ok;
}

static size_t header_terminator(const char * buffer, size_t len) {
    for (size_t i = 0; i + 3 < len; ++i) {
        if (buffer[i] == '\r' && buffer[i + 1] == '\n' && buffer[i + 2] == '\r' && buffer[i + 3] == '\n') {
            return i + 4;
        }
    }
    return 0;
}

/* Reads until the end of the header block. Returns the total number of bytes
 * read on success, which is usually more than the header block: the peer can
 * send body bytes in the same segment, and those must be counted or the body
 * read below waits for bytes that already arrived. Sets *header_len to the
 * offset just past the terminator, returns 0 when the peer closed first, and
 * (size_t) -1 when the block does not fit in `cap`. */
static size_t read_header_block(trellis_socket_t sock, char * buffer, size_t cap, size_t * header_len) {
    size_t used = 0;
    while (used < cap) {
        int received = recv(sock, buffer + used, (int) (cap - used), 0);
        if (received <= 0) {
            return 0;
        }
        used += (size_t) received;
        const size_t end = header_terminator(buffer, used);
        if (end > 0) {
            *header_len = end;
            return used;
        }
    }
    return (size_t) -1;
}

static void parse_query(const char * query, trellis_generate_params * params) {
    const char * cursor = query;
    while (cursor != NULL && *cursor != '\0') {
        const char * amp = strchr(cursor, '&');
        size_t pair_len = amp == NULL ? strlen(cursor) : (size_t) (amp - cursor);

        char pair[256];
        if (pair_len < sizeof(pair)) {
            memcpy(pair, cursor, pair_len);
            pair[pair_len] = '\0';

            char * eq = strchr(pair, '=');
            if (eq != NULL) {
                *eq = '\0';
                const char * key = pair;
                const char * value = eq + 1;
                int int_value = 0;

                if (strcmp(key, "seed") == 0 && parse_u32_arg(value, &params->seed)) {
                    /* accepted */
                } else if (strcmp(key, "noise_seed") == 0 && parse_u32_arg(value, &params->noise_seed)) {
                    /* accepted */
                } else if (strcmp(key, "steps") == 0 && parse_int_arg(value, &int_value) && int_value > 0) {
                    params->sparse_steps = int_value;
                    params->latent_steps = int_value;
                } else if (strcmp(key, "ss_steps") == 0 && parse_int_arg(value, &int_value) && int_value > 0) {
                    params->sparse_steps = int_value;
                } else if (strcmp(key, "slat_steps") == 0 && parse_int_arg(value, &int_value) && int_value > 0) {
                    params->latent_steps = int_value;
                } else if (strcmp(key, "texture_size") == 0 && parse_int_arg(value, &int_value) && int_value > 0) {
                    params->texture_size = int_value;
                } else if (strcmp(key, "pipeline") == 0 && value[0] != '\0') {
                    snprintf(params->pipeline, sizeof(params->pipeline), "%s", value); /* validated by the caller */
                } else if (strcmp(key, "shape_only") == 0) {
                    params->shape_only = strcmp(value, "0") != 0 && strcmp(value, "false") != 0;
                } else if (strcmp(key, "keep") == 0) {
                    params->keep = strcmp(value, "0") != 0 && strcmp(value, "false") != 0;
                } else if (strcmp(key, "format") == 0) {
                    if (strcmp(value, "json") == 0) {
                        params->keep = 1;
                    }
                } else {
                    TRELLIS_TOOL_WARN("ignoring unknown query parameter '%s'", key);
                }
            }
        }

        if (amp == NULL) {
            break;
        }
        cursor = amp + 1;
    }
}

static int pipeline_resolution(const char * pipeline, int * resolution) {
    if (strcmp(pipeline, "512") == 0) {
        *resolution = 512;
        return 1;
    }
    if (strcmp(pipeline, "1024") == 0 || strcmp(pipeline, "1024_cascade") == 0) {
        *resolution = 1024;
        return 1;
    }
    if (strcmp(pipeline, "1536_cascade") == 0) {
        *resolution = 1536;
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* endpoints
 * ------------------------------------------------------------------------- */

static int handle_health(trellis_socket_t sock, const trellis_server_options * options, int require_ready) {
    /* Presence means "usable", not "the path is there": a directory that is
     * still downloading must not report as ready. */
    const int model_ok = dir_has(options->model_dir, "pipeline.json", 0) &&
                         dir_has(options->model_dir, "ckpts", 1);
    char missing[1200];
    const int model_complete = model_ok && ckpt_weights_complete(options->model_dir, missing, sizeof(missing));
    const int dino_ok = dir_has(options->dino_dir, "model.safetensors", 0) &&
                        dir_has(options->dino_dir, "config.json", 0);
    const int birefnet_ok = options->birefnet_path == NULL || options->birefnet_path[0] == '\0'
        ? 1
        : path_exists(options->birefnet_path);

    if (require_ready) {
        char reason[2048];
        if (!weights_ready(options, reason, sizeof(reason))) {
            return send_json_error(sock, 503, "not ready: %s", reason);
        }
    }

    char model_escaped[1024];
    char dino_escaped[1024];
    json_escape_into(options->model_dir == NULL ? "" : options->model_dir, model_escaped, sizeof(model_escaped));
    json_escape_into(options->dino_dir == NULL ? "" : options->dino_dir, dino_escaped, sizeof(dino_escaped));

    char body[2560];
    int body_len = snprintf(
        body,
        sizeof(body),
        "{\"status\":\"ok\",\"engine\":\"trellis2.c\",\"backend\":\"%s\","
        "\"model\":\"%s\",\"model_dir\":\"%s\",\"model_present\":%s,\"model_complete\":%s,"
        "\"dino_dir\":\"%s\",\"dino_present\":%s,\"birefnet_present\":%s,"
        "\"pipeline\":\"%s\",\"texture_size\":%d,\"model_cache_budget_mib\":%d,"
        "\"requests\":%llu}\n",
        TRELLIS_DEFAULT_BACKEND,
        basename_of(options->model_dir == NULL ? "" : options->model_dir),
        model_escaped,
        model_ok ? "true" : "false",
        model_complete ? "true" : "false",
        dino_escaped,
        dino_ok ? "true" : "false",
        birefnet_ok ? "true" : "false",
        options->pipeline,
        options->texture_size,
        options->model_cache_budget_mib,
        (unsigned long long) g_request_counter);

    if (body_len < 0 || (size_t) body_len >= sizeof(body)) {
        return send_json_error(sock, 500, "health response overflow");
    }
    return send_json_ok(sock, body, (size_t) body_len);
}

static int handle_output(trellis_socket_t sock, const trellis_server_options * options, const char * name) {
    if (name[0] == '\0' || strstr(name, "..") != NULL) {
        return send_json_error(sock, 400, "invalid output name");
    }
    for (const char * p = name; *p != '\0'; ++p) {
        const int ok = (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                       (*p >= '0' && *p <= '9') || *p == '.' || *p == '_' || *p == '-';
        if (!ok) {
            return send_json_error(sock, 400, "invalid output name");
        }
    }

    char path[1200];
    snprintf(path, sizeof(path), "%s/%s", out_dir_of(options), name);
    const long long size = file_size_bytes(path);
    if (size < 0) {
        return send_json_error(sock, 404, "no output named '%s'", name);
    }

    char head[512];
    int head_len = snprintf(
        head,
        sizeof(head),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: model/gltf-binary\r\n"
        "Content-Length: %lld\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n"
        "\r\n",
        size);
    if (!send_all(sock, head, (size_t) head_len)) {
        return 0;
    }

    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return 0;
    }
    char chunk[TRELLIS_SERVER_IO_CHUNK];
    size_t read_bytes = 0;
    int ok = 1;
    while ((read_bytes = fread(chunk, 1, sizeof(chunk), f)) > 0) {
        if (!send_all(sock, chunk, read_bytes)) {
            ok = 0;
            break;
        }
    }
    fclose(f);
    return ok;
}

static int handle_generate(
    trellis_socket_t sock,
    const trellis_server_options * options,
    const trellis_http_request * request,
    const char * body,
    size_t body_len) {
    trellis_generate_params params;
    memset(&params, 0, sizeof(params));
    snprintf(params.pipeline, sizeof(params.pipeline), "%s", options->pipeline);
    params.seed = 1u;
    params.noise_seed = 18u;
    params.sparse_steps = options->steps;
    params.latent_steps = options->steps;
    params.texture_size = options->texture_size;
    params.shape_only = 0;
    params.keep = 0;
    parse_query(request->query, &params);

    int resolution = 0;
    if (!pipeline_resolution(params.pipeline, &resolution)) {
        return send_json_error(
            sock,
            400,
            "unknown pipeline '%s' (expected 512, 1024, 1024_cascade or 1536_cascade)",
            params.pipeline);
    }
    if (body_len == 0) {
        return send_json_error(sock, 400, "request body is empty; send the image bytes");
    }

    char reason[2048];
    if (!weights_ready(options, reason, sizeof(reason))) {
        return send_json_error(sock, 503, "cannot generate: %s", reason);
    }

    const uint64_t request_id = ++g_request_counter;
    char image_path[1200];
    char glb_path[1200];
    if (!trellis_mkdir_p(out_dir_of(options))) {
        return send_json_error(sock, 500, "cannot create the output directory '%s'", out_dir_of(options));
    }
    make_scratch_path(options, request_id, ".png", image_path, sizeof(image_path));
    make_scratch_path(options, request_id, ".glb", glb_path, sizeof(glb_path));

    FILE * image_file = fopen(image_path, "wb");
    if (image_file == NULL || fwrite(body, 1, body_len, image_file) != body_len) {
        if (image_file != NULL) {
            fclose(image_file);
        }
        remove(image_path);
        return send_json_error(sock, 500, "cannot stage the request image");
    }
    fclose(image_file);

    trellis_image_to_gltf_options pipeline_options;
    memset(&pipeline_options, 0, sizeof(pipeline_options));
    pipeline_options.model_dir = options->model_dir;
    pipeline_options.dino_dir = options->dino_dir;
    pipeline_options.birefnet_path = options->birefnet_path;
    pipeline_options.vkmesh_path = options->vkmesh_path;
    pipeline_options.image_path = image_path;
    pipeline_options.gltf_path = glb_path;
    pipeline_options.device = options->device;
    pipeline_options.pipeline_type = params.pipeline;
    pipeline_options.resolution = resolution;
    pipeline_options.cond_resolution = 512;
    pipeline_options.sparse_resolution = 32;
    pipeline_options.latent_size = 16;
    pipeline_options.sparse_structure_steps = params.sparse_steps;
    pipeline_options.structured_latent_steps = params.latent_steps;
    pipeline_options.seed = params.seed;
    pipeline_options.noise_seed = params.noise_seed;
    pipeline_options.rescale_t = 3.0f;
    pipeline_options.guidance_strength = 7.5f;
    pipeline_options.guidance_rescale = 0.5f;
    pipeline_options.guidance_min = 0.6f;
    pipeline_options.guidance_max = 1.0f;
    pipeline_options.flow_blocks_override = -1;
    pipeline_options.flow_block_parts_override = -1;
    pipeline_options.mesh_postprocess = 1;
    pipeline_options.mesh_postprocess_no_simplify = options->mesh_no_simplify;
    pipeline_options.mesh_postprocess_decimation_target = options->mesh_decimation_target;
    pipeline_options.mesh_remesh = 1;
    pipeline_options.mesh_remesh_resolution = 0;
    pipeline_options.mesh_remesh_band = 1.0f;
    pipeline_options.mesh_remesh_project = 0.0f;
    pipeline_options.max_num_tokens = 49152;
    pipeline_options.texture_size = params.texture_size;
    pipeline_options.model_cache = 1;
    pipeline_options.model_cache_budget_mib = options->model_cache_budget_mib;
    pipeline_options.use_ggml_flash_attn = 0;

    trellis_image_to_gltf_feature_options feature_options = TRELLIS_IMAGE_TO_GLTF_FEATURE_OPTIONS_INIT;
    feature_options.shape_only = params.shape_only;

    TRELLIS_TOOL_INFO(
        "request %llu: pipeline=%s resolution=%d steps=%d/%d texture=%d seed=%u noise_seed=%u shape_only=%d image=%.1f MiB",
        (unsigned long long) request_id,
        params.pipeline,
        resolution,
        params.sparse_steps,
        params.latent_steps,
        params.texture_size,
        params.seed,
        params.noise_seed,
        params.shape_only,
        (double) body_len / (1024.0 * 1024.0));

    const int64_t started_us = ggml_time_us();
    const trellis_status status = trellis_pipeline_trellis2_image_to_gltf_ex(&pipeline_options, &feature_options);
    const double elapsed = (double) (ggml_time_us() - started_us) / 1000000.0;

    remove(image_path);

    if (status != TRELLIS_STATUS_OK) {
        remove(glb_path);
        return send_json_error(
            sock,
            500,
            "generation failed after %.1fs: %s",
            elapsed,
            trellis_status_string(status));
    }

    const long long glb_size = file_size_bytes(glb_path);
    if (glb_size < 0) {
        return send_json_error(sock, 500, "generation produced no mesh");
    }

    TRELLIS_TOOL_INFO(
        "request %llu: done in %.2fs, glb %.1f MiB%s",
        (unsigned long long) request_id,
        elapsed,
        (double) glb_size / (1024.0 * 1024.0),
        params.keep ? " (kept)" : "");

    if (params.keep) {
        char escaped[1200];
        json_escape_into(glb_path, escaped, sizeof(escaped));
        char json[1600];
        int json_len = snprintf(
            json,
            sizeof(json),
            "{\"file\":\"%s\",\"path\":\"%s\",\"bytes\":%lld,\"seconds\":%.2f}\n",
            basename_of(glb_path),
            escaped,
            glb_size,
            elapsed);
        if (json_len < 0 || (size_t) json_len >= sizeof(json)) {
            return send_json_error(sock, 500, "response overflow");
        }
        return send_json_ok(sock, json, (size_t) json_len);
    }

    return send_file_response(sock, glb_path, "model/gltf-binary", elapsed, 0);
}

/* ------------------------------------------------------------------------- */
/* request handling
 * ------------------------------------------------------------------------- */

static int parse_request_head(char * head, size_t head_len, trellis_http_request * out) {
    memset(out, 0, sizeof(*out));

    /* head ends with CRLF CRLF; terminate the last line for the loop below. */
    head[head_len] = '\0';
    char * cursor = head;
    char * line_end = strstr(cursor, "\r\n");
    if (line_end == NULL) {
        return 0;
    }
    *line_end = '\0';

    char version[16];
    if (sscanf(cursor, "%15s %1023s %15s", out->method, out->path, version) != 3) {
        return 0;
    }

    char * query = strchr(out->path, '?');
    if (query != NULL) {
        snprintf(out->query, sizeof(out->query), "%s", query + 1);
        *query = '\0';
    }

    cursor = line_end + 2;
    while (*cursor != '\0') {
        line_end = strstr(cursor, "\r\n");
        const int has_crlf = line_end != NULL;
        if (!has_crlf) {
            line_end = cursor + strlen(cursor);
        }
        if (line_end == cursor) {
            break; /* the blank line that ends the header block */
        }
        *line_end = '\0';

        char * colon = strchr(cursor, ':');
        if (colon != NULL) {
            *colon = '\0';
            const char * name = cursor;
            const char * value = colon + 1;
            while (*value == ' ' || *value == '\t') {
                ++value;
            }

            const size_t name_len = strlen(name);
            if (name_len == strlen("Content-Length") && string_ieq_n(name, "Content-Length", name_len)) {
                char * end = NULL;
                const unsigned long long parsed = strtoull(value, &end, 10);
                if (end != value && *end == '\0') {
                    out->content_length = (size_t) parsed;
                    out->has_content_length = 1;
                }
            } else if (name_len == strlen("Content-Type") && string_ieq_n(name, "Content-Type", name_len)) {
                snprintf(out->content_type, sizeof(out->content_type), "%s", value);
            } else if (name_len == strlen("Expect") && string_ieq_n(name, "Expect", name_len)) {
                if (string_ieq_n(value, "100-continue", strlen("100-continue"))) {
                    out->expect_continue = 1;
                }
            } else if (name_len == strlen("Transfer-Encoding") && string_ieq_n(name, "Transfer-Encoding", name_len)) {
                if (string_ieq_n(value, "chunked", strlen("chunked"))) {
                    out->chunked = 1;
                }
            }
        }

        if (!has_crlf) {
            break;
        }
        cursor = line_end + 2;
    }
    return 1;
}

static int handle_client(
    const trellis_server_options * options,
    trellis_socket_t sock) {
    char head[TRELLIS_SERVER_MAX_HEADER_BYTES + 1];
    size_t header_len = 0;
    const size_t head_bytes = read_header_block(sock, head, TRELLIS_SERVER_MAX_HEADER_BYTES, &header_len);
    if (head_bytes == 0) {
        return 0;
    }
    if (head_bytes == (size_t) -1) {
        return send_json_error(sock, 413, "request header block too large");
    }

    /* The body starts right after the header block, so any bytes already read
     * past it belong to the body. Parse a COPY: parse_request_head scatters
     * '\0' over the buffer it is handed. */
    if (header_len == 0 || header_len > head_bytes) {
        return send_json_error(sock, 400, "malformed HTTP request");
    }
    char scratch[TRELLIS_SERVER_MAX_HEADER_BYTES + 1];
    memcpy(scratch, head, header_len);
    scratch[header_len] = '\0';

    trellis_http_request request;
    if (!parse_request_head(scratch, header_len, &request)) {
        return send_json_error(sock, 400, "malformed HTTP request");
    }

    if (request.chunked) {
        return send_json_error(sock, 501, "Transfer-Encoding: chunked is not supported; send Content-Length");
    }

    const int is_get = strcmp(request.method, "GET") == 0 || strcmp(request.method, "HEAD") == 0;
    const int is_post = strcmp(request.method, "POST") == 0;
    const int wants_generate = strcmp(request.path, "/generate") == 0 ||
                               strcmp(request.path, "/v1/3d/generations") == 0 ||
                               strcmp(request.path, "/v1/images/generations") == 0;
    const int wants_status = strcmp(request.path, "/health") == 0 ||
                             strcmp(request.path, "/") == 0 ||
                             strcmp(request.path, "/ready") == 0;
    const int wants_output = strncmp(request.path, "/output/", strlen("/output/")) == 0;

    if (wants_generate) {
        if (!is_post) {
            return send_json_error(
                sock,
                405,
                "method '%s' is not allowed on %s; POST the image bytes",
                request.method,
                request.path);
        }
        if (!request.has_content_length) {
            return send_json_error(sock, 411, "Content-Length is required");
        }
        if (request.content_length > options->max_body_bytes) {
            return send_json_error(
                sock,
                413,
                "request body is %zu bytes, limit is %zu (raise --max-body-mib)",
                request.content_length,
                options->max_body_bytes);
        }

        if (request.expect_continue) {
            if (!send_text(sock, "HTTP/1.1 100 Continue\r\n\r\n")) {
                return 0;
            }
        }

        char * body = (char *) malloc(request.content_length + 1);
        if (body == NULL) {
            return send_json_error(sock, 500, "out of memory staging a %zu byte body", request.content_length);
        }

        size_t staged = 0;
        if (head_bytes > header_len) {
            staged = head_bytes - header_len;
            if (staged > request.content_length) {
                staged = request.content_length;
            }
            memcpy(body, head + header_len, staged);
        }
        while (staged < request.content_length) {
            int received = recv(sock, body + staged, (int) (request.content_length - staged), 0);
            if (received <= 0) {
                free(body);
                return send_json_error(sock, 400, "request body ended after %zu of %zu bytes", staged, request.content_length);
            }
            staged += (size_t) received;
        }
        body[request.content_length] = '\0';

        const int result = handle_generate(sock, options, &request, body, staged);
        free(body);
        return result;
    }

    if (wants_status || wants_output) {
        if (!is_get) {
            return send_json_error(
                sock,
                405,
                "method '%s' is not allowed on %s; use GET",
                request.method,
                request.path);
        }
        if (wants_status) {
            return handle_health(sock, options, strcmp(request.path, "/ready") == 0);
        }
        return handle_output(sock, options, request.path + strlen("/output/"));
    }

    return send_json_error(sock, 404, "unknown path '%s'", request.path);
}

/* ------------------------------------------------------------------------- */
/* startup
 * ------------------------------------------------------------------------- */

static const char * arg_value(int argc, char ** argv, int * index) {
    if (*index + 1 >= argc) {
        return NULL;
    }
    *index += 1;
    return argv[*index];
}

static void usage(FILE * out, const char * exe) {
    fprintf(
        out,
        "trellis2-server — HTTP server for TRELLIS.2 image-to-3D\n"
        "\n"
        "Usage: %s --model DIR --dino DIR [options]\n"
        "\n"
        "Model:\n"
        "  --model DIR             TRELLIS.2-4B package directory (required)\n"
        "  --dino DIR              DINOv3 conditioning directory (required to generate)\n"
        "  --birefnet FILE         BiRefNet GGUF for opaque input; transparent PNG skips it\n"
        "  --vkmesh FILE           vkmesh executable; default searches sibling binary then PATH\n"
        "  --device N              GPU device index (default 0)\n"
        "  --pipeline NAME         512 (default), 1024, 1024_cascade or 1536_cascade\n"
        "  --steps N               sampler steps for both flow stages (default 12)\n"
        "  --texture-size N        texture resolution (default 1024)\n"
        "  --model-cache-budget-mib N  GPU-resident weight cache cap; 0 is unlimited (default 0)\n"
        "  --mesh-postprocess-simplify     Run the simplify pass (default: off, matching\n"
        "                                  the CLI; the mesh keeps ~5x the triangles)\n"
        "  --mesh-postprocess-no-simplify  Skip the simplify pass (default)\n"
        "  --mesh-decimation-target N      Simplify target triangle count (default 1000000)\n"
        "\n"
        "Server:\n"
        "  --host ADDR             bind address (default 127.0.0.1)\n"
        "  --port N                bind port (default 8080)\n"
        "  --out-dir DIR           scratch/output directory (default $TMPDIR or $TEMP)\n"
        "  --max-body-mib N        request body cap in MiB (default %u)\n"
        "  --verbose               debug logging\n"
        "  --help                  this text\n"
        "\n"
        "Endpoints: GET /health, GET /ready, GET /output/NAME, POST /generate,\n"
        "           POST /v1/3d/generations\n"
        "Query:     seed, noise_seed, pipeline, steps, ss_steps, slat_steps,\n"
        "           texture_size, shape_only, keep, format=json\n",
        exe,
        (unsigned) TRELLIS_SERVER_DEFAULT_MAX_BODY_MIB);
}

int main(int argc, char ** argv) {
    trellis_server_options options;
    memset(&options, 0, sizeof(options));
    options.host = "127.0.0.1";
    options.port = 8080;
    options.device = 0;
    options.pipeline = "512";
    options.steps = 12;
    options.texture_size = 1024;
    options.model_cache_budget_mib = 0;
    options.mesh_no_simplify = 1;
    options.mesh_decimation_target = 1000000;
    options.max_body_bytes = (size_t) TRELLIS_SERVER_DEFAULT_MAX_BODY_MIB * 1024u * 1024u;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout, argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--model") == 0) {
            options.model_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--dino") == 0) {
            options.dino_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--birefnet") == 0) {
            options.birefnet_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--vkmesh") == 0) {
            options.vkmesh_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--device") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.device)) {
                fprintf(stderr, "invalid --device\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--pipeline") == 0) {
            options.pipeline = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--steps") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.steps) || options.steps <= 0) {
                fprintf(stderr, "invalid --steps\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--texture-size") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.texture_size) || options.texture_size <= 0) {
                fprintf(stderr, "invalid --texture-size\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--mesh-postprocess-simplify") == 0) {
            options.mesh_no_simplify = 0;
        } else if (strcmp(argv[i], "--mesh-postprocess-no-simplify") == 0) {
            options.mesh_no_simplify = 1;
        } else if (strcmp(argv[i], "--mesh-decimation-target") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.mesh_decimation_target) ||
                options.mesh_decimation_target <= 0) {
                fprintf(stderr, "invalid --mesh-decimation-target\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--model-cache-budget-mib") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.model_cache_budget_mib) ||
                options.model_cache_budget_mib < 0) {
                fprintf(stderr, "invalid --model-cache-budget-mib\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--host") == 0) {
            options.host = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--port") == 0) {
            if (!parse_int_arg(arg_value(argc, argv, &i), &options.port) || options.port <= 0 || options.port > 65535) {
                fprintf(stderr, "invalid --port\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--out-dir") == 0) {
            options.out_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--max-body-mib") == 0) {
            int mib = 0;
            if (!parse_int_arg(arg_value(argc, argv, &i), &mib) || mib <= 0) {
                fprintf(stderr, "invalid --max-body-mib\n");
                return 2;
            }
            options.max_body_bytes = (size_t) mib * 1024u * 1024u;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            trellis_tool_set_verbose(1);
        } else {
            fprintf(stderr, "unknown argument '%s'\n\n", argv[i]);
            usage(stderr, argv[0]);
            return 2;
        }
    }

    if (options.model_dir == NULL || options.model_dir[0] == '\0') {
        fprintf(stderr, "--model is required\n\n");
        usage(stderr, argv[0]);
        return 2;
    }

    int resolution = 0;
    if (!pipeline_resolution(options.pipeline, &resolution)) {
        fprintf(stderr, "unknown --pipeline '%s'\n", options.pipeline);
        return 2;
    }

    trellis_runtime_init();

    /* The write path is a socket send; a peer that hangs up mid-response must
     * not take the process down with it. */
#if !defined(_WIN32)
    signal(SIGPIPE, SIG_IGN);
#else
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return 1;
    }
#endif

    trellis_socket_t listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener == TRELLIS_SOCKET_INVALID) {
        fprintf(stderr, "cannot create a socket\n");
        return 1;
    }

    int reuse = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, (const char *) &reuse, sizeof(reuse));

    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons((unsigned short) options.port);
    if (inet_pton(AF_INET, options.host, &address.sin_addr) != 1) {
        fprintf(stderr, "invalid --host '%s'\n", options.host);
        trellis_socket_close(listener);
        return 2;
    }
    if (bind(listener, (struct sockaddr *) &address, sizeof(address)) != 0) {
        fprintf(stderr, "cannot bind %s:%d\n", options.host, options.port);
        trellis_socket_close(listener);
        return 1;
    }
    if (listen(listener, 8) != 0) {
        fprintf(stderr, "cannot listen on %s:%d\n", options.host, options.port);
        trellis_socket_close(listener);
        return 1;
    }

    if (!trellis_mkdir_p(out_dir_of(&options))) {
        fprintf(stderr, "cannot create --out-dir '%s'\n", out_dir_of(&options));
        trellis_socket_close(listener);
        return 1;
    }

    // Establish ggml's timer base before the first request. ggml_time_us() is
    // (QueryPerformanceCounter - timer_start), and timer_start is only set by
    // ggml_time_init(), which the runtime runs once, lazily, from inside the
    // first pipeline call. Timing a request across that first init measures the
    // end against a zero base and reports a negative duration (observed:
    // "generation failed after -239.8s").
    trellis_runtime_init();

    TRELLIS_TOOL_INFO("trellis2-server listening on %s:%d", options.host, options.port);
    TRELLIS_TOOL_INFO(
        "backend=%s model=%s dino=%s pipeline=%s texture=%d cache_budget_mib=%d",
        TRELLIS_DEFAULT_BACKEND,
        options.model_dir,
        options.dino_dir == NULL ? "(unset)" : options.dino_dir,
        options.pipeline,
        options.texture_size,
        options.model_cache_budget_mib);
    if (!dir_has(options.model_dir, "pipeline.json", 0)) {
        TRELLIS_TOOL_WARN("model directory '%s' has no pipeline.json yet; /generate will fail", options.model_dir);
    } else {
        char missing[1200];
        if (!ckpt_weights_complete(options.model_dir, missing, sizeof(missing))) {
            TRELLIS_TOOL_WARN("model weights are incomplete ('%s' absent or empty); /ready stays 503 until the download finishes", missing);
        }
    }
    if (!dir_has(options.dino_dir, "model.safetensors", 0)) {
        TRELLIS_TOOL_WARN("DINOv3 weights are missing or incomplete; /generate will fail");
    }
    TRELLIS_TOOL_INFO("endpoints: GET /health, POST /generate, POST /v1/3d/generations");

    for (;;) {
        trellis_socket_t client = accept(listener, NULL, NULL);
        if (client == TRELLIS_SOCKET_INVALID) {
            continue;
        }
        handle_client(&options, client);
        trellis_socket_close(client);
    }

    trellis_socket_close(listener);
    return 0;
}
