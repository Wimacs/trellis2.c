#define _POSIX_C_SOURCE 200809L

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "external/glad.h"

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct mesh_frame_info {
    int step;
    int resolution;
    int vertices;
    int faces;
    char stage[64];
    char source[32];
    char status[32];
    char ply[512];
    char vertices_f32[512];
    char faces_i32[512];
} mesh_frame_info;

typedef struct frame_list {
    mesh_frame_info * frames;
    int count;
    int capacity;
} frame_list;

typedef struct loaded_mesh {
    Mesh * meshes;
    int count;
    int capacity;
    int ready;
    int gpu_ready;
    unsigned int vao_id;
    unsigned int vbo_id;
    unsigned int ebo_id;
    int gpu_vertex_count;
    int gpu_index_count;
    int source_vertices;
    int source_faces;
    int drawn_faces;
} loaded_mesh;

typedef struct mesh_shader {
    Shader shader;
    int loc_mvp;
    int loc_base_color;
    int loc_wire_color;
    int loc_light_dir;
    int loc_wire_mode;
} mesh_shader;

typedef struct face3 {
    int v[3];
} face3;

static const float MESH_BASE_COLOR[4] = {0.72f, 0.76f, 0.82f, 1.0f};
static const float MESH_AMBIENT_LIGHT = 0.36f;
static const float MESH_DIFFUSE_LIGHT = 0.64f;
static const Vector3 MESH_LIGHT_DIR = {-0.449609f, 0.719374f, 0.519548f};

static void usage(const char * argv0) {
    fprintf(stderr,
        "Usage: %s --snapshot-dir DIR [--source pred_x0|x_t|final_x_t|all]\n"
        "Options:\n"
        "  --snapshot-dir DIR   Directory containing mesh_frames.tsv and mesh arrays/PLYs\n"
        "  --source NAME        Frame source to display, default all\n"
        "  --stage NAME         Optional stage filter, e.g. shape_512 or shape_hr\n"
        "  --display DISPLAY    X11 display, e.g. :1. Auto-detected when unset\n"
        "  --xauthority PATH    Optional Xauthority file for the desktop session\n"
        "  --width N            Window width, default 1280\n"
        "  --height N           Window height, default 800\n"
        "  --max-faces N        Draw at most N sampled faces per frame; 0 draws all, default 0\n"
        "  --mesh-chunk-faces N Faces per legacy GPU mesh chunk, default 21000\n"
        "  --mesh-upload-mode M gpu_indexed, expanded, or indexed; default gpu_indexed\n"
        "  --mesh-style STYLE   solid, wire, or solid_wire, default solid\n"
        "  --hold SECONDS       Seconds per frame while playing, default 0.35\n"
        "  --dry-run            Load frame table and print a summary without opening a window\n",
        argv0);
}

static const char * arg_value(int argc, char ** argv, int * i) {
    if (*i + 1 >= argc) {
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

static int append_frame(frame_list * list, const mesh_frame_info * frame) {
    if (list->count == list->capacity) {
        int next_capacity = list->capacity == 0 ? 16 : list->capacity * 2;
        mesh_frame_info * next = (mesh_frame_info *) realloc(list->frames, (size_t) next_capacity * sizeof(mesh_frame_info));
        if (next == NULL) {
            return 0;
        }
        list->frames = next;
        list->capacity = next_capacity;
    }
    list->frames[list->count++] = *frame;
    return 1;
}

static void free_frames(frame_list * list) {
    free(list->frames);
    list->frames = NULL;
    list->count = 0;
    list->capacity = 0;
}

static int join_path(char * dst, size_t dst_size, const char * dir, const char * file) {
    int n = snprintf(dst, dst_size, "%s/%s", dir, file);
    return n >= 0 && (size_t) n < dst_size;
}

static int path_exists(const char * path) {
    struct stat st;
    return path != NULL && stat(path, &st) == 0;
}

static int env_is_empty(const char * name) {
    const char * value = getenv(name);
    return value == NULL || value[0] == '\0';
}

static void detect_local_x11_display(char * dst, size_t dst_size) {
    int best = -1;
    DIR * dir = opendir("/tmp/.X11-unix");
    if (dir != NULL) {
        struct dirent * ent;
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] != 'X') {
                continue;
            }
            char * end = NULL;
            long display = strtol(ent->d_name + 1, &end, 10);
            if (end != NULL && *end == '\0' && display >= 0 && display <= 9999) {
                if (best < 0 || display < best) {
                    best = (int) display;
                }
            }
        }
        closedir(dir);
    }
    snprintf(dst, dst_size, ":%d", best >= 0 ? best : 0);
}

static int detect_xauthority(char * dst, size_t dst_size) {
    const char * home = getenv("HOME");
    if (home != NULL && home[0] != '\0') {
        char home_auth[PATH_MAX];
        int n = snprintf(home_auth, sizeof(home_auth), "%s/.Xauthority", home);
        if (n >= 0 && (size_t) n < sizeof(home_auth) && path_exists(home_auth)) {
            snprintf(dst, dst_size, "%s", home_auth);
            return 1;
        }
    }

    char gdm_auth[PATH_MAX];
    int n = snprintf(gdm_auth, sizeof(gdm_auth), "/run/user/%ld/gdm/Xauthority", (long) getuid());
    if (n >= 0 && (size_t) n < sizeof(gdm_auth) && path_exists(gdm_auth)) {
        snprintf(dst, dst_size, "%s", gdm_auth);
        return 1;
    }
    return 0;
}

static void configure_display_env(const char * display, const char * xauthority) {
    if (display != NULL && display[0] != '\0') {
        setenv("DISPLAY", display, 1);
    } else if (env_is_empty("DISPLAY")) {
        char detected[32];
        detect_local_x11_display(detected, sizeof(detected));
        setenv("DISPLAY", detected, 1);
    }

    if (xauthority != NULL && xauthority[0] != '\0') {
        setenv("XAUTHORITY", xauthority, 1);
    } else if (env_is_empty("XAUTHORITY")) {
        char detected[PATH_MAX];
        if (detect_xauthority(detected, sizeof(detected))) {
            setenv("XAUTHORITY", detected, 1);
        }
    }
}

static int parse_frames_tsv(const char * snapshot_dir, const char * source_filter, const char * stage_filter, frame_list * out) {
    char path[1024];
    if (!join_path(path, sizeof(path), snapshot_dir, "mesh_frames.tsv")) {
        return 0;
    }
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return 0;
    }

    char line[4096];
    int line_no = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        ++line_no;
        if (line_no == 1 && strncmp(line, "stage\t", 6) == 0) {
            continue;
        }
        mesh_frame_info frame;
        memset(&frame, 0, sizeof(frame));
        char npz[512] = {0};
        int matched = sscanf(line, "%63[^\t]\t%d\t%31[^\t]\t%d\t%d\t%d\t%31[^\t]\t%511[^\t]\t%511[^\t\n]\t%511[^\t\n]\t%511[^\n]",
            frame.stage,
            &frame.step,
            frame.source,
            &frame.resolution,
            &frame.vertices,
            &frame.faces,
            frame.status,
            npz,
            frame.ply,
            frame.vertices_f32,
            frame.faces_i32);
        if (matched < 9 || frame.ply[0] == '\0') {
            continue;
        }
        if (strcmp(frame.status, "ok") != 0) {
            continue;
        }
        if (strcmp(source_filter, "all") != 0 && strcmp(source_filter, frame.source) != 0) {
            continue;
        }
        if (stage_filter != NULL && stage_filter[0] != '\0' && strcmp(stage_filter, frame.stage) != 0) {
            continue;
        }
        if (!append_frame(out, &frame)) {
            fclose(f);
            return 0;
        }
    }
    fclose(f);
    return out->count > 0;
}

static int sampled_index(int i, int count, int max_count) {
    if (max_count <= 0 || count <= max_count) {
        return i;
    }
    if (max_count <= 1) {
        return 0;
    }
    return (int) llround((double) i * (double) (count - 1) / (double) (max_count - 1));
}

static Vector3 transformed_vertex(const float * vertices, int index) {
    const float * src = &vertices[(size_t) index * 3u];
    Vector3 out = {src[0] * 2.0f, src[2] * 2.0f, src[1] * 2.0f};
    return out;
}

static Vector3 normal_for_triangle(Vector3 a, Vector3 b, Vector3 c) {
    Vector3 ab = Vector3Subtract(b, a);
    Vector3 ac = Vector3Subtract(c, a);
    return Vector3Normalize(Vector3CrossProduct(ab, ac));
}

static unsigned char color_byte(float value) {
    if (value < 0.0f) value = 0.0f;
    if (value > 1.0f) value = 1.0f;
    return (unsigned char) (value * 255.0f + 0.5f);
}

static void write_shaded_color(unsigned char * dst, float shade) {
    if (!isfinite(shade)) {
        shade = MESH_AMBIENT_LIGHT;
    }
    dst[0] = color_byte(MESH_BASE_COLOR[0] * shade);
    dst[1] = color_byte(MESH_BASE_COLOR[1] * shade);
    dst[2] = color_byte(MESH_BASE_COLOR[2] * shade);
    dst[3] = color_byte(MESH_BASE_COLOR[3]);
}

static const char * MESH_GPU_VS =
    "#version 330\n"
    "in vec3 vertexPosition;\n"
    "uniform mat4 mvp;\n"
    "out vec3 fragPos;\n"
    "void main() {\n"
    "    vec3 p = vec3(vertexPosition.x * 2.0, vertexPosition.z * 2.0, vertexPosition.y * 2.0);\n"
    "    fragPos = p;\n"
    "    gl_Position = mvp * vec4(p, 1.0);\n"
    "}\n";

static const char * MESH_GPU_FS =
    "#version 330\n"
    "in vec3 fragPos;\n"
    "out vec4 finalColor;\n"
    "uniform vec4 baseColor;\n"
    "uniform vec4 wireColor;\n"
    "uniform vec3 lightDir;\n"
    "uniform int wireMode;\n"
    "void main() {\n"
    "    if (wireMode != 0) {\n"
    "        finalColor = wireColor;\n"
    "        return;\n"
    "    }\n"
    "    vec3 dx = dFdx(fragPos);\n"
    "    vec3 dy = dFdy(fragPos);\n"
    "    vec3 n = normalize(cross(dx, dy));\n"
    "    float ndl = abs(dot(n, normalize(lightDir)));\n"
    "    float shade = 0.36 + 0.64 * ndl;\n"
    "    finalColor = vec4(baseColor.rgb * shade, baseColor.a);\n"
    "}\n";

static int load_mesh_shader(mesh_shader * out) {
    if (out == NULL) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    out->shader = LoadShaderFromMemory(MESH_GPU_VS, MESH_GPU_FS);
    if (out->shader.id == 0) {
        return 0;
    }
    out->loc_mvp = out->shader.locs[SHADER_LOC_MATRIX_MVP];
    out->loc_base_color = GetShaderLocation(out->shader, "baseColor");
    out->loc_wire_color = GetShaderLocation(out->shader, "wireColor");
    out->loc_light_dir = GetShaderLocation(out->shader, "lightDir");
    out->loc_wire_mode = GetShaderLocation(out->shader, "wireMode");
    return out->loc_mvp >= 0 && out->loc_base_color >= 0 &&
        out->loc_wire_color >= 0 && out->loc_light_dir >= 0 &&
        out->loc_wire_mode >= 0;
}

static void unload_mesh_shader(mesh_shader * shader) {
    if (shader != NULL && shader->shader.id != 0) {
        UnloadShader(shader->shader);
        memset(shader, 0, sizeof(*shader));
    }
}

static void set_mesh_shader_common(const mesh_shader * shader, int wire_mode) {
    float base_color[4] = {MESH_BASE_COLOR[0], MESH_BASE_COLOR[1], MESH_BASE_COLOR[2], 1.0f};
    float wire_color[4] = {74.0f / 255.0f, 210.0f / 255.0f, 178.0f / 255.0f, 1.0f};
    float light_dir[3] = {MESH_LIGHT_DIR.x, MESH_LIGHT_DIR.y, MESH_LIGHT_DIR.z};
    Matrix mvp = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection());
    rlSetUniformMatrix(shader->loc_mvp, mvp);
    rlSetUniform(shader->loc_base_color, base_color, RL_SHADER_UNIFORM_VEC4, 1);
    rlSetUniform(shader->loc_wire_color, wire_color, RL_SHADER_UNIFORM_VEC4, 1);
    rlSetUniform(shader->loc_light_dir, light_dir, RL_SHADER_UNIFORM_VEC3, 1);
    rlSetUniform(shader->loc_wire_mode, &wire_mode, RL_SHADER_UNIFORM_INT, 1);
}

static int append_uploaded_mesh(loaded_mesh * out, Mesh mesh, int faces) {
    if (out->count == out->capacity) {
        int next_capacity = out->capacity == 0 ? 8 : out->capacity * 2;
        Mesh * next = (Mesh *) realloc(out->meshes, (size_t) next_capacity * sizeof(Mesh));
        if (next == NULL) {
            return 0;
        }
        out->meshes = next;
        out->capacity = next_capacity;
    }
    out->meshes[out->count++] = mesh;
    out->drawn_faces += faces;
    out->ready = 1;
    return 1;
}

static int upload_expanded_chunk(
    const float * vertices,
    const face3 * faces,
    int face_count,
    loaded_mesh * out) {
    if (vertices == NULL || faces == NULL || face_count <= 0 || out == NULL) {
        return 0;
    }
    const int vertex_count = face_count * 3;
    float * vertex_data = (float *) malloc((size_t) vertex_count * 3u * sizeof(float));
    unsigned char * color_data = (unsigned char *) malloc((size_t) vertex_count * 4u);
    if (vertex_data == NULL || color_data == NULL) {
        free(vertex_data);
        free(color_data);
        return 0;
    }

    for (int i = 0; i < face_count; ++i) {
        Vector3 tri[3];
        for (int k = 0; k < 3; ++k) {
            tri[k] = transformed_vertex(vertices, faces[i].v[k]);
            float * dst = &vertex_data[(size_t) (i * 3 + k) * 3u];
            dst[0] = tri[k].x;
            dst[1] = tri[k].y;
            dst[2] = tri[k].z;
        }
        Vector3 normal = normal_for_triangle(tri[0], tri[1], tri[2]);
        float intensity = fabsf(Vector3DotProduct(normal, MESH_LIGHT_DIR));
        float shade = MESH_AMBIENT_LIGHT + MESH_DIFFUSE_LIGHT * intensity;
        for (int k = 0; k < 3; ++k) {
            write_shaded_color(&color_data[(size_t) (i * 3 + k) * 4u], shade);
        }
    }

    Mesh mesh = {0};
    mesh.vertexCount = vertex_count;
    mesh.triangleCount = face_count;
    mesh.vertices = vertex_data;
    mesh.colors = color_data;
    UploadMesh(&mesh, false);
    if (!append_uploaded_mesh(out, mesh, face_count)) {
        UnloadMesh(mesh);
        return 0;
    }
    return 1;
}

static int upload_indexed_chunk(
    const float * vertices,
    int vertex_source_count,
    const face3 * faces,
    int face_count,
    int * local_map,
    loaded_mesh * out) {
    if (vertices == NULL || faces == NULL || face_count <= 0 || local_map == NULL || out == NULL) {
        return 0;
    }
    const int max_unique = face_count * 3;
    int * unique = (int *) malloc((size_t) max_unique * sizeof(int));
    unsigned short * indices = (unsigned short *) malloc((size_t) face_count * 3u * sizeof(unsigned short));
    if (unique == NULL || indices == NULL) {
        free(unique);
        free(indices);
        return 0;
    }

    int unique_count = 0;
    for (int i = 0; i < face_count; ++i) {
        for (int k = 0; k < 3; ++k) {
            int id = faces[i].v[k];
            if (id < 0 || id >= vertex_source_count) {
                free(unique);
                free(indices);
                return 0;
            }
            int local = local_map[id];
            if (local < 0) {
                if (unique_count >= 65535) {
                    free(unique);
                    free(indices);
                    return 0;
                }
                local = unique_count;
                local_map[id] = local;
                unique[unique_count++] = id;
            }
            indices[(size_t) i * 3u + (size_t) k] = (unsigned short) local;
        }
    }

    float * vertex_data = (float *) malloc((size_t) unique_count * 3u * sizeof(float));
    unsigned char * color_data = (unsigned char *) malloc((size_t) unique_count * 4u);
    Vector3 * normals = (Vector3 *) calloc((size_t) unique_count, sizeof(Vector3));
    if (vertex_data == NULL || color_data == NULL || normals == NULL) {
        free(vertex_data);
        free(color_data);
        free(normals);
        for (int i = 0; i < unique_count; ++i) local_map[unique[i]] = -1;
        free(unique);
        free(indices);
        return 0;
    }

    for (int i = 0; i < unique_count; ++i) {
        Vector3 v = transformed_vertex(vertices, unique[i]);
        vertex_data[(size_t) i * 3u + 0u] = v.x;
        vertex_data[(size_t) i * 3u + 1u] = v.y;
        vertex_data[(size_t) i * 3u + 2u] = v.z;
    }
    for (int i = 0; i < face_count; ++i) {
        int ia = indices[(size_t) i * 3u + 0u];
        int ib = indices[(size_t) i * 3u + 1u];
        int ic = indices[(size_t) i * 3u + 2u];
        Vector3 a = {vertex_data[(size_t) ia * 3u + 0u], vertex_data[(size_t) ia * 3u + 1u], vertex_data[(size_t) ia * 3u + 2u]};
        Vector3 b = {vertex_data[(size_t) ib * 3u + 0u], vertex_data[(size_t) ib * 3u + 1u], vertex_data[(size_t) ib * 3u + 2u]};
        Vector3 c = {vertex_data[(size_t) ic * 3u + 0u], vertex_data[(size_t) ic * 3u + 1u], vertex_data[(size_t) ic * 3u + 2u]};
        Vector3 normal = normal_for_triangle(a, b, c);
        normals[ia] = Vector3Add(normals[ia], normal);
        normals[ib] = Vector3Add(normals[ib], normal);
        normals[ic] = Vector3Add(normals[ic], normal);
    }
    for (int i = 0; i < unique_count; ++i) {
        Vector3 normal = Vector3Normalize(normals[i]);
        float intensity = fabsf(Vector3DotProduct(normal, MESH_LIGHT_DIR));
        float shade = MESH_AMBIENT_LIGHT + MESH_DIFFUSE_LIGHT * intensity;
        write_shaded_color(&color_data[(size_t) i * 4u], shade);
        local_map[unique[i]] = -1;
    }

    Mesh mesh = {0};
    mesh.vertexCount = unique_count;
    mesh.triangleCount = face_count;
    mesh.vertices = vertex_data;
    mesh.indices = indices;
    mesh.colors = color_data;
    UploadMesh(&mesh, false);
    int ok = append_uploaded_mesh(out, mesh, face_count);
    if (!ok) {
        UnloadMesh(mesh);
    }
    free(normals);
    free(unique);
    return ok;
}

static int read_header_count(const char * line, const char * prefix, int * out) {
    size_t n = strlen(prefix);
    if (strncmp(line, prefix, n) != 0) {
        return 0;
    }
    *out = atoi(line + n);
    return 1;
}

static int upload_gpu_indexed_selected_faces(
    const float * vertices,
    int vertex_count,
    int source_face_count,
    const face3 * selected_faces,
    int selected_face_count,
    loaded_mesh * out) {
    if (vertices == NULL || selected_faces == NULL ||
        vertex_count <= 0 || source_face_count <= 0 || selected_face_count <= 0 || out == NULL) {
        return 0;
    }
    const int64_t vertex_bytes64 = (int64_t) vertex_count * 3ll * (int64_t) sizeof(float);
    const int64_t index_count64 = (int64_t) selected_face_count * 3ll;
    const int64_t index_bytes64 = index_count64 * (int64_t) sizeof(uint32_t);
    if (vertex_bytes64 <= 0 || vertex_bytes64 > INT_MAX ||
        index_count64 <= 0 || index_count64 > INT_MAX ||
        index_bytes64 <= 0 || index_bytes64 > INT_MAX) {
        return 0;
    }

    const void * index_src = selected_faces;
    uint32_t * index_copy = NULL;
    if (sizeof(int) != sizeof(uint32_t)) {
        index_copy = (uint32_t *) malloc((size_t) index_count64 * sizeof(uint32_t));
        if (index_copy == NULL) {
            return 0;
        }
        for (int i = 0; i < selected_face_count; ++i) {
            index_copy[(size_t) i * 3u + 0u] = (uint32_t) selected_faces[i].v[0];
            index_copy[(size_t) i * 3u + 1u] = (uint32_t) selected_faces[i].v[1];
            index_copy[(size_t) i * 3u + 2u] = (uint32_t) selected_faces[i].v[2];
        }
        index_src = index_copy;
    }

    out->vao_id = rlLoadVertexArray();
    if (out->vao_id == 0 || !rlEnableVertexArray(out->vao_id)) {
        free(index_copy);
        memset(out, 0, sizeof(*out));
        return 0;
    }
    out->vbo_id = rlLoadVertexBuffer(vertices, (int) vertex_bytes64, false);
    rlEnableVertexBuffer(out->vbo_id);
    rlSetVertexAttribute(0, 3, RL_FLOAT, false, 3 * (int) sizeof(float), 0);
    rlEnableVertexAttribute(0);
    out->ebo_id = rlLoadVertexBufferElement(index_src, selected_face_count * 3 * (int) sizeof(uint32_t), false);
    rlDisableVertexArray();
    rlDisableVertexBuffer();
    rlDisableVertexBufferElement();
    free(index_copy);

    if (out->vbo_id == 0 || out->ebo_id == 0) {
        if (out->vao_id != 0) rlUnloadVertexArray(out->vao_id);
        if (out->vbo_id != 0) rlUnloadVertexBuffer(out->vbo_id);
        if (out->ebo_id != 0) rlUnloadVertexBuffer(out->ebo_id);
        memset(out, 0, sizeof(*out));
        return 0;
    }
    out->ready = 1;
    out->gpu_ready = 1;
    out->gpu_vertex_count = vertex_count;
    out->gpu_index_count = selected_face_count * 3;
    out->source_vertices = vertex_count;
    out->source_faces = source_face_count;
    out->drawn_faces = selected_face_count;
    out->count = 1;
    return 1;
}

static int upload_selected_faces(
    const float * vertices,
    int vertex_count,
    int source_face_count,
    const face3 * selected_faces,
    int selected_face_count,
    int mesh_chunk_faces,
    const char * upload_mode,
    loaded_mesh * out) {
    if (vertices == NULL || selected_faces == NULL || selected_face_count <= 0 || upload_mode == NULL || out == NULL) {
        return 0;
    }
    if (strcmp(upload_mode, "gpu_indexed") == 0) {
        return upload_gpu_indexed_selected_faces(
            vertices,
            vertex_count,
            source_face_count,
            selected_faces,
            selected_face_count,
            out);
    }
    int chunk_size = mesh_chunk_faces <= 0 ? 21000 : mesh_chunk_faces;
    if (chunk_size > 21000) {
        chunk_size = 21000;
    }
    out->source_vertices = vertex_count;
    out->source_faces = source_face_count;
    int ok = 1;
    int * local_map = NULL;
    if (strcmp(upload_mode, "indexed") == 0) {
        local_map = (int *) malloc((size_t) vertex_count * sizeof(int));
        if (local_map == NULL) {
            ok = 0;
        } else {
            for (int i = 0; i < vertex_count; ++i) {
                local_map[i] = -1;
            }
        }
    }
    for (int start = 0; ok && start < selected_face_count; start += chunk_size) {
        int count = selected_face_count - start;
        if (count > chunk_size) {
            count = chunk_size;
        }
        if (strcmp(upload_mode, "indexed") == 0) {
            ok = upload_indexed_chunk(vertices, vertex_count, selected_faces + start, count, local_map, out);
        } else {
            ok = upload_expanded_chunk(vertices, selected_faces + start, count, out);
        }
    }
    free(local_map);
    if (!ok) {
        for (int i = 0; i < out->count; ++i) {
            UnloadMesh(out->meshes[i]);
        }
        free(out->meshes);
        memset(out, 0, sizeof(*out));
    }
    return ok;
}

static int face_indices_valid(const int32_t idx[3], int vertex_count) {
    return idx[0] >= 0 && idx[1] >= 0 && idx[2] >= 0 &&
        idx[0] < vertex_count && idx[1] < vertex_count && idx[2] < vertex_count;
}

static int load_binary_mesh(
    const char * vertices_path,
    const char * faces_path,
    int vertex_count,
    int face_count,
    int max_faces,
    int mesh_chunk_faces,
    const char * upload_mode,
    loaded_mesh * out) {
    memset(out, 0, sizeof(*out));
    if (vertices_path == NULL || faces_path == NULL || vertex_count <= 0 || face_count <= 0) {
        return 0;
    }
    float * vertices = (float *) malloc((size_t) vertex_count * 3u * sizeof(float));
    if (vertices == NULL) {
        return 0;
    }
    FILE * vf = fopen(vertices_path, "rb");
    if (vf == NULL) {
        free(vertices);
        fprintf(stderr, "failed to open %s: %s\n", vertices_path, strerror(errno));
        return 0;
    }
    size_t got_vertices = fread(vertices, 3u * sizeof(float), (size_t) vertex_count, vf);
    fclose(vf);
    if (got_vertices != (size_t) vertex_count) {
        free(vertices);
        fprintf(stderr, "failed to read %s\n", vertices_path);
        return 0;
    }

    int draw_faces = face_count;
    if (max_faces > 0 && draw_faces > max_faces) {
        draw_faces = max_faces;
    }
    face3 * selected_faces = (face3 *) malloc((size_t) draw_faces * sizeof(face3));
    if (selected_faces == NULL && draw_faces != 0) {
        free(vertices);
        return 0;
    }

    FILE * ff = fopen(faces_path, "rb");
    if (ff == NULL) {
        free(selected_faces);
        free(vertices);
        fprintf(stderr, "failed to open %s: %s\n", faces_path, strerror(errno));
        return 0;
    }
    int written = 0;
    if (draw_faces == face_count) {
        int32_t tmp[3];
        for (int i = 0; i < draw_faces; ++i) {
            if (fread(tmp, sizeof(int32_t), 3u, ff) != 3u) {
                break;
            }
            if (!face_indices_valid(tmp, vertex_count)) {
                continue;
            }
            selected_faces[written].v[0] = (int) tmp[0];
            selected_faces[written].v[1] = (int) tmp[1];
            selected_faces[written].v[2] = (int) tmp[2];
            ++written;
        }
    } else {
        for (int i = 0; i < draw_faces; ++i) {
            int idx = sampled_index(i, face_count, draw_faces);
            long offset = (long) ((int64_t) idx * 3ll * (int64_t) sizeof(int32_t));
            if (fseek(ff, offset, SEEK_SET) != 0) {
                break;
            }
            int32_t tmp[3];
            if (fread(tmp, sizeof(int32_t), 3u, ff) != 3u) {
                break;
            }
            if (!face_indices_valid(tmp, vertex_count)) {
                continue;
            }
            selected_faces[written].v[0] = (int) tmp[0];
            selected_faces[written].v[1] = (int) tmp[1];
            selected_faces[written].v[2] = (int) tmp[2];
            ++written;
        }
    }
    fclose(ff);
    int ok = written > 0 && upload_selected_faces(
        vertices,
        vertex_count,
        face_count,
        selected_faces,
        written,
        mesh_chunk_faces,
        upload_mode,
        out);
    free(selected_faces);
    free(vertices);
    return ok;
}

static int load_ascii_ply_mesh(
    const char * path,
    int max_faces,
    int mesh_chunk_faces,
    const char * upload_mode,
    loaded_mesh * out) {
    memset(out, 0, sizeof(*out));
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return 0;
    }

    char line[4096];
    int vertex_count = 0;
    int face_count = 0;
    int ascii = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strncmp(line, "format ascii", 12) == 0) {
            ascii = 1;
        }
        (void) read_header_count(line, "element vertex ", &vertex_count);
        (void) read_header_count(line, "element face ", &face_count);
        if (strncmp(line, "end_header", 10) == 0) {
            break;
        }
    }
    if (!ascii || vertex_count <= 0 || face_count <= 0) {
        fclose(f);
        fprintf(stderr, "unsupported PLY header in %s\n", path);
        return 0;
    }

    float * vertices = (float *) malloc((size_t) vertex_count * 3u * sizeof(float));
    if (vertices == NULL) {
        fclose(f);
        return 0;
    }
    for (int i = 0; i < vertex_count; ++i) {
        if (fgets(line, sizeof(line), f) == NULL ||
            sscanf(line, "%f %f %f", &vertices[3 * i + 0], &vertices[3 * i + 1], &vertices[3 * i + 2]) != 3) {
            free(vertices);
            fclose(f);
            return 0;
        }
    }

    int draw_faces = face_count;
    if (max_faces > 0 && draw_faces > max_faces) {
        draw_faces = max_faces;
    }
    face3 * selected_faces = (face3 *) malloc((size_t) draw_faces * sizeof(face3));
    if (selected_faces == NULL && draw_faces != 0) {
        free(vertices);
        fclose(f);
        return 0;
    }

    int written = 0;
    int next_sample = sampled_index(0, face_count, draw_faces);
    for (int face_idx = 0; face_idx < face_count; ++face_idx) {
        if (fgets(line, sizeof(line), f) == NULL) {
            break;
        }
        if (written >= draw_faces || face_idx != next_sample) {
            continue;
        }
        int n = 0;
        int a = 0;
        int b = 0;
        int c = 0;
        if (sscanf(line, "%d %d %d %d", &n, &a, &b, &c) != 4 || n != 3 ||
            a < 0 || b < 0 || c < 0 || a >= vertex_count || b >= vertex_count || c >= vertex_count) {
            continue;
        }
        selected_faces[written].v[0] = a;
        selected_faces[written].v[1] = b;
        selected_faces[written].v[2] = c;
        ++written;
        if (written < draw_faces) {
            next_sample = sampled_index(written, face_count, draw_faces);
        }
    }
    fclose(f);

    if (written == 0) {
        free(selected_faces);
        return 0;
    }

    int ok = upload_selected_faces(
        vertices,
        vertex_count,
        face_count,
        selected_faces,
        written,
        mesh_chunk_faces,
        upload_mode,
        out);
    free(selected_faces);
    free(vertices);
    return ok;
}

static void unload_loaded_mesh(loaded_mesh * mesh) {
    if (mesh != NULL && mesh->gpu_ready) {
        if (mesh->vao_id != 0) rlUnloadVertexArray(mesh->vao_id);
        if (mesh->vbo_id != 0) rlUnloadVertexBuffer(mesh->vbo_id);
        if (mesh->ebo_id != 0) rlUnloadVertexBuffer(mesh->ebo_id);
        memset(mesh, 0, sizeof(*mesh));
    } else if (mesh != NULL && mesh->ready) {
        for (int i = 0; i < mesh->count; ++i) {
            UnloadMesh(mesh->meshes[i]);
        }
        free(mesh->meshes);
        memset(mesh, 0, sizeof(*mesh));
    }
}

static void draw_loaded_meshes(
    const loaded_mesh * loaded,
    Material material,
    const mesh_shader * shader,
    const char * style) {
    if (loaded == NULL || !loaded->ready || loaded->count <= 0) {
        return;
    }
    if (loaded->gpu_ready) {
        if (shader == NULL || shader->shader.id == 0 || loaded->vao_id == 0 || loaded->gpu_index_count <= 0) {
            return;
        }
        rlDrawRenderBatchActive();
        rlDisableBackfaceCulling();
        rlEnableShader(shader->shader.id);
        rlEnableVertexArray(loaded->vao_id);
        if (strcmp(style, "wire") != 0) {
            set_mesh_shader_common(shader, 0);
            glDrawElements(GL_TRIANGLES, loaded->gpu_index_count, GL_UNSIGNED_INT, 0);
        }
        if (strcmp(style, "solid") != 0) {
            set_mesh_shader_common(shader, 1);
            rlEnableWireMode();
            glDrawElements(GL_TRIANGLES, loaded->gpu_index_count, GL_UNSIGNED_INT, 0);
            rlDisableWireMode();
        }
        rlDisableVertexArray();
        rlDisableShader();
        rlEnableBackfaceCulling();
        return;
    }
    rlDisableBackfaceCulling();
    if (strcmp(style, "wire") != 0) {
        material.maps[MATERIAL_MAP_DIFFUSE].color = WHITE;
        for (int i = 0; i < loaded->count; ++i) {
            DrawMesh(loaded->meshes[i], material, MatrixIdentity());
        }
    }
    if (strcmp(style, "solid") != 0) {
        material.maps[MATERIAL_MAP_DIFFUSE].color = (Color) {74, 210, 178, 255};
        rlEnableWireMode();
        for (int i = 0; i < loaded->count; ++i) {
            DrawMesh(loaded->meshes[i], material, MatrixIdentity());
        }
        rlDisableWireMode();
    }
    rlEnableBackfaceCulling();
}

int main(int argc, char ** argv) {
    const char * snapshot_dir = NULL;
    const char * source = "all";
    const char * stage = NULL;
    const char * display = NULL;
    const char * xauthority = NULL;
    int width = 1280;
    int height = 800;
    int max_faces = 0;
    int mesh_chunk_faces = 21000;
    const char * mesh_upload_mode = "gpu_indexed";
    const char * mesh_style = "solid";
    float hold = 0.35f;
    int dry_run = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--snapshot-dir") == 0) {
            snapshot_dir = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--source") == 0) {
            source = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--stage") == 0) {
            stage = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--display") == 0) {
            display = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--xauthority") == 0) {
            xauthority = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--width") == 0) {
            const char * v = arg_value(argc, argv, &i);
            width = v == NULL ? width : atoi(v);
        } else if (strcmp(argv[i], "--height") == 0) {
            const char * v = arg_value(argc, argv, &i);
            height = v == NULL ? height : atoi(v);
        } else if (strcmp(argv[i], "--max-faces") == 0) {
            const char * v = arg_value(argc, argv, &i);
            max_faces = v == NULL ? max_faces : atoi(v);
        } else if (strcmp(argv[i], "--mesh-chunk-faces") == 0) {
            const char * v = arg_value(argc, argv, &i);
            mesh_chunk_faces = v == NULL ? mesh_chunk_faces : atoi(v);
        } else if (strcmp(argv[i], "--mesh-upload-mode") == 0) {
            mesh_upload_mode = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--mesh-style") == 0) {
            mesh_style = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--hold") == 0) {
            const char * v = arg_value(argc, argv, &i);
            hold = v == NULL ? hold : (float) atof(v);
        } else if (strcmp(argv[i], "--dry-run") == 0) {
            dry_run = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (snapshot_dir == NULL || source == NULL) {
        usage(argv[0]);
        return 2;
    }
    if (mesh_upload_mode == NULL ||
        (strcmp(mesh_upload_mode, "gpu_indexed") != 0 && strcmp(mesh_upload_mode, "expanded") != 0 && strcmp(mesh_upload_mode, "indexed") != 0)) {
        fprintf(stderr, "--mesh-upload-mode must be gpu_indexed, expanded, or indexed\n");
        return 2;
    }
    if (mesh_style == NULL ||
        (strcmp(mesh_style, "solid") != 0 && strcmp(mesh_style, "wire") != 0 && strcmp(mesh_style, "solid_wire") != 0)) {
        fprintf(stderr, "--mesh-style must be solid, wire, or solid_wire\n");
        return 2;
    }

    frame_list frames;
    memset(&frames, 0, sizeof(frames));
    if (!parse_frames_tsv(snapshot_dir, source, stage, &frames)) {
        fprintf(stderr, "no mesh frames loaded from %s\n", snapshot_dir);
        free_frames(&frames);
        return 1;
    }

    if (dry_run) {
        int max_frame_faces = 0;
        for (int i = 0; i < frames.count; ++i) {
            if (frames.frames[i].faces > max_frame_faces) {
                max_frame_faces = frames.frames[i].faces;
            }
        }
        printf("loaded %d mesh frame records from %s (source=%s, max_frame_faces=%d, upload=%s, style=%s)\n",
            frames.count, snapshot_dir, source, max_frame_faces, mesh_upload_mode, mesh_style);
        free_frames(&frames);
        return 0;
    }

    configure_display_env(display, xauthority);
    SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(width, height, "TRELLIS.2 stage2 mesh viewer");
    SetTargetFPS(60);

    Material material = LoadMaterialDefault();
    material.maps[MATERIAL_MAP_DIFFUSE].color = WHITE;
    mesh_shader shader;
    memset(&shader, 0, sizeof(shader));
    if (strcmp(mesh_upload_mode, "gpu_indexed") == 0 && !load_mesh_shader(&shader)) {
        fprintf(stderr, "mesh viewer: failed to load gpu indexed mesh shader\n");
        UnloadMaterial(material);
        CloseWindow();
        free_frames(&frames);
        return 1;
    }

    loaded_mesh loaded;
    memset(&loaded, 0, sizeof(loaded));
    int loaded_index = -1;
    int current = 0;
    int paused = 0;
    double next_time = GetTime() + hold;
    float yaw = 0.75f;
    float pitch = 0.35f;
    float distance = 3.2f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_RIGHT)) {
            current = (current + 1) % frames.count;
            next_time = GetTime() + hold;
        }
        if (IsKeyPressed(KEY_LEFT)) {
            current = (current + frames.count - 1) % frames.count;
            next_time = GetTime() + hold;
        }
        if (IsKeyPressed(KEY_R)) {
            current = 0;
            next_time = GetTime() + hold;
        }
        if (!paused && GetTime() >= next_time) {
            current = (current + 1) % frames.count;
            next_time = GetTime() + hold;
        }

        if (loaded_index != current) {
            unload_loaded_mesh(&loaded);
            mesh_frame_info * load_frame = &frames.frames[current];
            if (load_frame->vertices_f32[0] != '\0' && load_frame->faces_i32[0] != '\0') {
                char vertices_path[1024];
                char faces_path[1024];
                if (join_path(vertices_path, sizeof(vertices_path), snapshot_dir, load_frame->vertices_f32) &&
                    join_path(faces_path, sizeof(faces_path), snapshot_dir, load_frame->faces_i32)) {
                    (void) load_binary_mesh(
                        vertices_path,
                        faces_path,
                        load_frame->vertices,
                        load_frame->faces,
                        max_faces,
                        mesh_chunk_faces,
                        mesh_upload_mode,
                        &loaded);
                }
            }
            if (!loaded.ready) {
                char ply_path[1024];
                if (join_path(ply_path, sizeof(ply_path), snapshot_dir, load_frame->ply)) {
                    (void) load_ascii_ply_mesh(ply_path, max_faces, mesh_chunk_faces, mesh_upload_mode, &loaded);
                }
            }
            loaded_index = current;
        }

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            Vector2 delta = GetMouseDelta();
            yaw -= delta.x * 0.008f;
            pitch -= delta.y * 0.008f;
            if (pitch < -1.35f) pitch = -1.35f;
            if (pitch > 1.35f) pitch = 1.35f;
        }
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            distance *= powf(0.9f, wheel);
            if (distance < 1.2f) distance = 1.2f;
            if (distance > 8.0f) distance = 8.0f;
        }

        float radius_xz = cosf(pitch) * distance;
        Camera3D camera = {
            .position = {sinf(yaw) * radius_xz, sinf(pitch) * distance, cosf(yaw) * radius_xz},
            .target = {0.0f, 0.0f, 0.0f},
            .up = {0.0f, 1.0f, 0.0f},
            .fovy = 45.0f,
            .projection = CAMERA_PERSPECTIVE,
        };

        mesh_frame_info * frame = &frames.frames[current];
        BeginDrawing();
        ClearBackground((Color) {18, 20, 24, 255});
        BeginMode3D(camera);
        draw_loaded_meshes(&loaded, material, &shader, mesh_style);
        EndMode3D();

        DrawRectangle(0, 0, GetScreenWidth(), 90, (Color) {0, 0, 0, 175});
        char title[256];
        snprintf(title, sizeof(title), "%s %s step %d", frame->stage, frame->source, frame->step);
        DrawText(title, 18, 16, 22, RAYWHITE);
        char detail[320];
        snprintf(detail, sizeof(detail), "%s: %d/%d faces drawn in %d chunks; %d vertices; frame %d/%d%s",
            loaded.drawn_faces < frame->faces ? "sampled" : "all",
            loaded.drawn_faces,
            frame->faces,
            loaded.count,
            frame->vertices,
            current + 1,
            frames.count,
            paused ? "; paused" : "");
        DrawText(detail, 18, 50, 16, (Color) {195, 204, 216, 255});
        DrawText("drag rotate; wheel zoom; space pause; left/right step; R reset", 18, GetScreenHeight() - 30, 16, (Color) {180, 188, 200, 255});
        DrawFPS(GetScreenWidth() - 96, 14);
        EndDrawing();
    }

    unload_loaded_mesh(&loaded);
    unload_mesh_shader(&shader);
    UnloadMaterial(material);
    CloseWindow();
    free_frames(&frames);
    return 0;
}
