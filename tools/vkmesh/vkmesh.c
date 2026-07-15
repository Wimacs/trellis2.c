#include <vulkan/vulkan.h>

#ifdef TRELLIS_VKMESH_LIBRARY
#include "trellis.h"
#endif

#include "vkmesh_accumulate_hole_components_spv.h"
#include "vkmesh_add_block_offsets_u32_spv.h"
#include "vkmesh_apply_orientation_flips_spv.h"
#include "vkmesh_assign_vertex_map_spv.h"
#include "vkmesh_assign_corner_vertices_spv.h"
#include "vkmesh_boundary_degree_owner_spv.h"
#include "vkmesh_compact_boundary_edges_spv.h"
#include "vkmesh_compact_face_pairs_spv.h"
#include "vkmesh_compact_faces_spv.h"
#include "vkmesh_compact_vertices_spv.h"
#include "vkmesh_compact_unique_simplify_edges_spv.h"
#include "vkmesh_copy_u32_spv.h"
#include "vkmesh_degenerate_faces_spv.h"
#include "vkmesh_expand_edges_spv.h"
#include "vkmesh_face_keys_spv.h"
#include "vkmesh_fill_u32_spv.h"
#include "vkmesh_component_area_spv.h"
#include "vkmesh_compress_parents_spv.h"
#include "vkmesh_init_u32_sequence_spv.h"
#include "vkmesh_init_orientation_state_spv.h"
#include "vkmesh_mark_boundary_edges_spv.h"
#include "vkmesh_mark_component_keep_spv.h"
#include "vkmesh_mark_duplicate_faces_spv.h"
#include "vkmesh_mark_hole_faces_spv.h"
#include "vkmesh_mark_hole_roots_spv.h"
#include "vkmesh_merge_edges_spv.h"
#include "vkmesh_merge_face_keys_spv.h"
#include "vkmesh_remap_faces_spv.h"
#include "vkmesh_block_scan_u32_spv.h"
#include "vkmesh_seed_vertex_offsets_spv.h"
#include "vkmesh_sort_edges_spv.h"
#include "vkmesh_sort_face_keys_spv.h"
#include "vkmesh_simplify_best_edge_spv.h"
#include "vkmesh_simplify_collapse_edges_spv.h"
#include "vkmesh_simplify_edge_cost_spv.h"
#include "vkmesh_simplify_propagate_cost_spv.h"
#include "vkmesh_union_face_edges_spv.h"
#include "vkmesh_union_boundary_edges_spv.h"
#include "vkmesh_union_corner_edges_spv.h"
#include "vkmesh_union_orientation_edges_spv.h"
#include "vkmesh_unsigned_distance_spv.h"
#include "vkmesh_vertex_face_adjacency_spv.h"
#include "vkmesh_vertex_face_degree_spv.h"
#include "vkmesh_vertex_qem_spv.h"
#include "vkmesh_compress_orientation_state_spv.h"
#include "vkmesh_write_hole_faces_spv.h"
#include "vkmesh_write_hole_vertices_spv.h"
#include "vkmesh_write_repaired_mesh_spv.h"
#include "xatlas_c.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct vkmesh_mesh {
    float * vertices;
    float * uvs;
    int32_t * faces;
    int64_t n_vertices;
    int64_t n_faces;
    int64_t vertex_capacity;
    int64_t face_capacity;
    int has_uvs;
} vkmesh_mesh;

typedef struct vkmesh_edge {
    uint32_t min_v;
    uint32_t max_v;
    uint32_t a;
    uint32_t b;
    uint32_t face;
} vkmesh_edge;

typedef struct vkmesh_boundary_edge {
    int32_t a;
    int32_t b;
    uint32_t min_v;
    uint32_t max_v;
} vkmesh_boundary_edge;

typedef struct vkmesh_face_key {
    int32_t v0;
    int32_t v1;
    int32_t v2;
    int32_t face;
} vkmesh_face_key;

typedef struct vkmesh_face_pair {
    int32_t f0;
    int32_t f1;
    uint32_t v0;
    uint32_t v1;
} vkmesh_face_pair;

typedef struct vkmesh_qem {
    double m[10];
} vkmesh_qem;

typedef struct vkmesh_simplify_edge {
    int32_t v0;
    int32_t v1;
    double cost;
    float pos[3];
} vkmesh_simplify_edge;

typedef struct vkmesh_bvh_tri {
    float bmin[3];
    float bmax[3];
    float centroid[3];
    uint32_t face;
} vkmesh_bvh_tri;

typedef struct vkmesh_bvh_node {
    float bmin[3];
    float bmax[3];
    uint32_t left;
    uint32_t meta;
} vkmesh_bvh_node;

typedef struct vkmesh_remesh_coord {
    int32_t x;
    int32_t y;
    int32_t z;
} vkmesh_remesh_coord;

typedef struct vkmesh_u64_u32_hash {
    uint64_t * keys;
    uint32_t * vals;
    size_t capacity;
    size_t count;
} vkmesh_u64_u32_hash;

typedef struct vkmesh_vk_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void * mapped;
    size_t bytes;
    size_t allocation_bytes;
} vkmesh_vk_buffer;

typedef enum vkmesh_pipeline_kind {
    VKMESH_PIPE_EXPAND_EDGES = 0,
    VKMESH_PIPE_DEGENERATE_FACES,
    VKMESH_PIPE_FACE_KEYS,
    VKMESH_PIPE_SORT_EDGES,
    VKMESH_PIPE_SORT_FACE_KEYS,
    VKMESH_PIPE_MERGE_EDGES,
    VKMESH_PIPE_MERGE_FACE_KEYS,
    VKMESH_PIPE_MARK_DUPLICATE_FACES,
    VKMESH_PIPE_MARK_BOUNDARY_EDGES,
    VKMESH_PIPE_BOUNDARY_DEGREE_OWNER,
    VKMESH_PIPE_COMPACT_BOUNDARY_EDGES,
    VKMESH_PIPE_COMPACT_FACE_PAIRS,
    VKMESH_PIPE_COMPACT_FACES,
    VKMESH_PIPE_COMPACT_UNIQUE_SIMPLIFY_EDGES,
    VKMESH_PIPE_COPY_U32,
    VKMESH_PIPE_FILL_U32,
    VKMESH_PIPE_ACCUMULATE_HOLE_COMPONENTS,
    VKMESH_PIPE_APPLY_ORIENTATION_FLIPS,
    VKMESH_PIPE_ASSIGN_VERTEX_MAP,
    VKMESH_PIPE_REMAP_FACES,
    VKMESH_PIPE_COMPACT_VERTICES,
    VKMESH_PIPE_INIT_U32_SEQUENCE,
    VKMESH_PIPE_INIT_ORIENTATION_STATE,
    VKMESH_PIPE_SEED_VERTEX_OFFSETS,
    VKMESH_PIPE_BLOCK_SCAN_U32,
    VKMESH_PIPE_ADD_BLOCK_OFFSETS_U32,
    VKMESH_PIPE_UNION_FACE_EDGES,
    VKMESH_PIPE_UNION_BOUNDARY_EDGES,
    VKMESH_PIPE_UNION_CORNER_EDGES,
    VKMESH_PIPE_UNION_ORIENTATION_EDGES,
    VKMESH_PIPE_COMPRESS_PARENTS,
    VKMESH_PIPE_COMPRESS_ORIENTATION_STATE,
    VKMESH_PIPE_VERTEX_FACE_DEGREE,
    VKMESH_PIPE_VERTEX_FACE_ADJACENCY,
    VKMESH_PIPE_VERTEX_QEM,
    VKMESH_PIPE_SIMPLIFY_EDGE_COST,
    VKMESH_PIPE_SIMPLIFY_PROPAGATE_COST,
    VKMESH_PIPE_SIMPLIFY_BEST_EDGE,
    VKMESH_PIPE_SIMPLIFY_COLLAPSE_EDGES,
    VKMESH_PIPE_ASSIGN_CORNER_VERTICES,
    VKMESH_PIPE_WRITE_REPAIRED_MESH,
    VKMESH_PIPE_COMPONENT_AREA,
    VKMESH_PIPE_MARK_COMPONENT_KEEP,
    VKMESH_PIPE_MARK_HOLE_ROOTS,
    VKMESH_PIPE_MARK_HOLE_FACES,
    VKMESH_PIPE_WRITE_HOLE_VERTICES,
    VKMESH_PIPE_WRITE_HOLE_FACES,
    VKMESH_PIPE_UNSIGNED_DISTANCE,
    VKMESH_PIPE_COUNT,
} vkmesh_pipeline_kind;

typedef struct vkmesh_vk {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkCommandPool command_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipelines[VKMESH_PIPE_COUNT];
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    VkCommandBuffer command_buffer;
    VkFence completion_fence;
    size_t workspace_budget_bytes;
    size_t workspace_current_bytes;
    size_t workspace_peak_bytes;
    uint32_t workspace_memory_type;
    int workspace_memory_type_reported;
} vkmesh_vk;

typedef struct vkmesh_device_mesh {
    vkmesh_vk * vk;
    vkmesh_vk_buffer vertices;
    vkmesh_vk_buffer faces;
    uint32_t n_vertices;
    uint32_t n_faces;
} vkmesh_device_mesh;

typedef struct vkmesh_distance_query {
    vkmesh_vk * vk;
    vkmesh_vk_buffer buffers[4];
    uint32_t face_count;
    uint32_t node_count;
    uint32_t point_capacity;
    uint32_t output_word_offset;
    uint32_t batch_count;
} vkmesh_distance_query;

typedef struct vkmesh_push {
    uint32_t n;
    uint32_t aux0;
    uint32_t aux1;
    uint32_t aux2;
    float eps;
    float rel_eps;
} vkmesh_push;

typedef struct vkmesh_shader_blob {
    const unsigned char * data;
    unsigned int len;
    const char * name;
} vkmesh_shader_blob;

static const vkmesh_shader_blob vkmesh_shaders[VKMESH_PIPE_COUNT] = {
    { vkmesh_expand_edges_spv, sizeof(vkmesh_expand_edges_spv), "expand_edges" },
    { vkmesh_degenerate_faces_spv, sizeof(vkmesh_degenerate_faces_spv), "degenerate_faces" },
    { vkmesh_face_keys_spv, sizeof(vkmesh_face_keys_spv), "face_keys" },
    { vkmesh_sort_edges_spv, sizeof(vkmesh_sort_edges_spv), "sort_edges" },
    { vkmesh_sort_face_keys_spv, sizeof(vkmesh_sort_face_keys_spv), "sort_face_keys" },
    { vkmesh_merge_edges_spv, sizeof(vkmesh_merge_edges_spv), "merge_edges" },
    { vkmesh_merge_face_keys_spv, sizeof(vkmesh_merge_face_keys_spv), "merge_face_keys" },
    { vkmesh_mark_duplicate_faces_spv, sizeof(vkmesh_mark_duplicate_faces_spv), "mark_duplicate_faces" },
    { vkmesh_mark_boundary_edges_spv, sizeof(vkmesh_mark_boundary_edges_spv), "mark_boundary_edges" },
    { vkmesh_boundary_degree_owner_spv, sizeof(vkmesh_boundary_degree_owner_spv), "boundary_degree_owner" },
    { vkmesh_compact_boundary_edges_spv, sizeof(vkmesh_compact_boundary_edges_spv), "compact_boundary_edges" },
    { vkmesh_compact_face_pairs_spv, sizeof(vkmesh_compact_face_pairs_spv), "compact_face_pairs" },
    { vkmesh_compact_faces_spv, sizeof(vkmesh_compact_faces_spv), "compact_faces" },
    { vkmesh_compact_unique_simplify_edges_spv, sizeof(vkmesh_compact_unique_simplify_edges_spv), "compact_unique_simplify_edges" },
    { vkmesh_copy_u32_spv, sizeof(vkmesh_copy_u32_spv), "copy_u32" },
    { vkmesh_fill_u32_spv, sizeof(vkmesh_fill_u32_spv), "fill_u32" },
    { vkmesh_accumulate_hole_components_spv, sizeof(vkmesh_accumulate_hole_components_spv), "accumulate_hole_components" },
    { vkmesh_apply_orientation_flips_spv, sizeof(vkmesh_apply_orientation_flips_spv), "apply_orientation_flips" },
    { vkmesh_assign_vertex_map_spv, sizeof(vkmesh_assign_vertex_map_spv), "assign_vertex_map" },
    { vkmesh_remap_faces_spv, sizeof(vkmesh_remap_faces_spv), "remap_faces" },
    { vkmesh_compact_vertices_spv, sizeof(vkmesh_compact_vertices_spv), "compact_vertices" },
    { vkmesh_init_u32_sequence_spv, sizeof(vkmesh_init_u32_sequence_spv), "init_u32_sequence" },
    { vkmesh_init_orientation_state_spv, sizeof(vkmesh_init_orientation_state_spv), "init_orientation_state" },
    { vkmesh_seed_vertex_offsets_spv, sizeof(vkmesh_seed_vertex_offsets_spv), "seed_vertex_offsets" },
    { vkmesh_block_scan_u32_spv, sizeof(vkmesh_block_scan_u32_spv), "block_scan_u32" },
    { vkmesh_add_block_offsets_u32_spv, sizeof(vkmesh_add_block_offsets_u32_spv), "add_block_offsets_u32" },
    { vkmesh_union_face_edges_spv, sizeof(vkmesh_union_face_edges_spv), "union_face_edges" },
    { vkmesh_union_boundary_edges_spv, sizeof(vkmesh_union_boundary_edges_spv), "union_boundary_edges" },
    { vkmesh_union_corner_edges_spv, sizeof(vkmesh_union_corner_edges_spv), "union_corner_edges" },
    { vkmesh_union_orientation_edges_spv, sizeof(vkmesh_union_orientation_edges_spv), "union_orientation_edges" },
    { vkmesh_compress_parents_spv, sizeof(vkmesh_compress_parents_spv), "compress_parents" },
    { vkmesh_compress_orientation_state_spv, sizeof(vkmesh_compress_orientation_state_spv), "compress_orientation_state" },
    { vkmesh_vertex_face_degree_spv, sizeof(vkmesh_vertex_face_degree_spv), "vertex_face_degree" },
    { vkmesh_vertex_face_adjacency_spv, sizeof(vkmesh_vertex_face_adjacency_spv), "vertex_face_adjacency" },
    { vkmesh_vertex_qem_spv, sizeof(vkmesh_vertex_qem_spv), "vertex_qem" },
    { vkmesh_simplify_edge_cost_spv, sizeof(vkmesh_simplify_edge_cost_spv), "simplify_edge_cost" },
    { vkmesh_simplify_propagate_cost_spv, sizeof(vkmesh_simplify_propagate_cost_spv), "simplify_propagate_cost" },
    { vkmesh_simplify_best_edge_spv, sizeof(vkmesh_simplify_best_edge_spv), "simplify_best_edge" },
    { vkmesh_simplify_collapse_edges_spv, sizeof(vkmesh_simplify_collapse_edges_spv), "simplify_collapse_edges" },
    { vkmesh_assign_corner_vertices_spv, sizeof(vkmesh_assign_corner_vertices_spv), "assign_corner_vertices" },
    { vkmesh_write_repaired_mesh_spv, sizeof(vkmesh_write_repaired_mesh_spv), "write_repaired_mesh" },
    { vkmesh_component_area_spv, sizeof(vkmesh_component_area_spv), "component_area" },
    { vkmesh_mark_component_keep_spv, sizeof(vkmesh_mark_component_keep_spv), "mark_component_keep" },
    { vkmesh_mark_hole_roots_spv, sizeof(vkmesh_mark_hole_roots_spv), "mark_hole_roots" },
    { vkmesh_mark_hole_faces_spv, sizeof(vkmesh_mark_hole_faces_spv), "mark_hole_faces" },
    { vkmesh_write_hole_vertices_spv, sizeof(vkmesh_write_hole_vertices_spv), "write_hole_vertices" },
    { vkmesh_write_hole_faces_spv, sizeof(vkmesh_write_hole_faces_spv), "write_hole_faces" },
    { vkmesh_unsigned_distance_spv, sizeof(vkmesh_unsigned_distance_spv), "unsigned_distance" },
};

#if defined(_MSC_VER)
#define VKMESH_THREAD_LOCAL __declspec(thread)
#else
#define VKMESH_THREAD_LOCAL _Thread_local
#endif

/* Nested helpers reuse one context per call without sharing Vulkan objects across threads. */
static VKMESH_THREAD_LOCAL vkmesh_vk * g_active_vkmesh_vk = NULL;
static VKMESH_THREAD_LOCAL int g_vkmesh_device_index = 0;
static VKMESH_THREAD_LOCAL int g_vkmesh_workspace_budget_mib = 0;

typedef enum vkmesh_error_kind {
    VKMESH_ERROR_NONE = 0,
    VKMESH_ERROR_INVALID_ARGUMENT,
    VKMESH_ERROR_OUT_OF_MEMORY,
    VKMESH_ERROR_VULKAN_UNAVAILABLE,
    VKMESH_ERROR_VULKAN,
} vkmesh_error_kind;

static VKMESH_THREAD_LOCAL vkmesh_error_kind g_vkmesh_last_error = VKMESH_ERROR_NONE;

static void vkmesh_set_error(vkmesh_error_kind error) {
    g_vkmesh_last_error = error;
}

static int vkmesh_check_vk_result(const char * operation, VkResult result) {
    if (result == VK_SUCCESS) return 1;
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        vkmesh_set_error(VKMESH_ERROR_OUT_OF_MEMORY);
    } else {
        vkmesh_set_error(VKMESH_ERROR_VULKAN);
    }
    fprintf(stderr, "vkmesh: %s failed: VkResult=%d\n", operation, (int) result);
    return 0;
}

static void mesh_free(vkmesh_mesh * mesh) {
    if (mesh == NULL) return;
    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    memset(mesh, 0, sizeof(*mesh));
}

static int mesh_clone(const vkmesh_mesh * src, vkmesh_mesh * dst) {
    if (src == NULL || dst == NULL || src->vertices == NULL || src->faces == NULL ||
        src->n_vertices <= 0 || src->n_faces <= 0 ||
        src->n_vertices > INT64_MAX / 3 || src->n_faces > INT64_MAX / 3 ||
        (uint64_t) src->n_vertices > (uint64_t) SIZE_MAX / (3u * sizeof(float)) ||
        (uint64_t) src->n_faces > (uint64_t) SIZE_MAX / (3u * sizeof(int32_t))) {
        return 0;
    }
    vkmesh_mesh out;
    memset(&out, 0, sizeof(out));
    const size_t vertex_count = (size_t) src->n_vertices * 3u;
    const size_t face_count = (size_t) src->n_faces * 3u;
    out.vertices = (float *) malloc(vertex_count * sizeof(float));
    out.faces = (int32_t *) malloc(face_count * sizeof(int32_t));
    if (src->has_uvs && src->uvs != NULL) {
        if ((uint64_t) src->n_vertices > (uint64_t) SIZE_MAX / (2u * sizeof(float))) {
            mesh_free(&out);
            return 0;
        }
        out.uvs = (float *) malloc((size_t) src->n_vertices * 2u * sizeof(float));
    }
    if (out.vertices == NULL || out.faces == NULL || (src->has_uvs && src->uvs != NULL && out.uvs == NULL)) {
        mesh_free(&out);
        return 0;
    }
    memcpy(out.vertices, src->vertices, vertex_count * sizeof(float));
    memcpy(out.faces, src->faces, face_count * sizeof(int32_t));
    if (out.uvs != NULL) {
        memcpy(out.uvs, src->uvs, (size_t) src->n_vertices * 2u * sizeof(float));
        out.has_uvs = 1;
    }
    out.n_vertices = src->n_vertices;
    out.n_faces = src->n_faces;
    out.vertex_capacity = src->n_vertices;
    out.face_capacity = src->n_faces;
    *dst = out;
    return 1;
}

static int mesh_reserve_vertices(vkmesh_mesh * mesh, int64_t need) {
    if (need <= mesh->vertex_capacity) return 1;
    int64_t cap = mesh->vertex_capacity > 0 ? mesh->vertex_capacity : 1024;
    while (cap < need) {
        if (cap > INT64_MAX / 2) return 0;
        cap *= 2;
    }
    float * next = (float *) realloc(mesh->vertices, (size_t) cap * 3u * sizeof(float));
    if (next == NULL) return 0;
    mesh->vertices = next;
    if (mesh->uvs != NULL) {
        float * next_uvs = (float *) realloc(mesh->uvs, (size_t) cap * 2u * sizeof(float));
        if (next_uvs == NULL) return 0;
        mesh->uvs = next_uvs;
    }
    mesh->vertex_capacity = cap;
    return 1;
}

static int mesh_reserve_faces(vkmesh_mesh * mesh, int64_t need) {
    if (need <= mesh->face_capacity) return 1;
    int64_t cap = mesh->face_capacity > 0 ? mesh->face_capacity : 1024;
    while (cap < need) {
        if (cap > INT64_MAX / 2) return 0;
        cap *= 2;
    }
    int32_t * next = (int32_t *) realloc(mesh->faces, (size_t) cap * 3u * sizeof(int32_t));
    if (next == NULL) return 0;
    mesh->faces = next;
    mesh->face_capacity = cap;
    return 1;
}

static int mesh_add_vertex(vkmesh_mesh * mesh, float x, float y, float z) {
    if (!mesh_reserve_vertices(mesh, mesh->n_vertices + 1)) return 0;
    float * v = mesh->vertices + (size_t) mesh->n_vertices * 3u;
    v[0] = x;
    v[1] = y;
    v[2] = z;
    if (mesh->uvs != NULL) {
        float * uv = mesh->uvs + (size_t) mesh->n_vertices * 2u;
        uv[0] = 0.0f;
        uv[1] = 0.0f;
    }
    ++mesh->n_vertices;
    return 1;
}

static int mesh_add_face(vkmesh_mesh * mesh, int32_t a, int32_t b, int32_t c) {
    if (a < 0 || b < 0 || c < 0 ||
        a >= mesh->n_vertices || b >= mesh->n_vertices || c >= mesh->n_vertices) {
        return 0;
    }
    if (!mesh_reserve_faces(mesh, mesh->n_faces + 1)) return 0;
    int32_t * f = mesh->faces + (size_t) mesh->n_faces * 3u;
    f[0] = a;
    f[1] = b;
    f[2] = c;
    ++mesh->n_faces;
    return 1;
}

static int meshbin_checked_count(uint64_t n, uint64_t channels, size_t * out) {
    if (out == NULL || channels == 0 || n > (uint64_t) SIZE_MAX / channels) {
        return 0;
    }
    *out = (size_t) n * (size_t) channels;
    return 1;
}

static int parse_int_arg(const char * text, int * out) {
    if (text == NULL || text[0] == '\0' || out == NULL) return 0;
    errno = 0;
    char * end = NULL;
    long value = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < INT_MIN || value > INT_MAX) return 0;
    *out = (int) value;
    return 1;
}

static int parse_float_arg(const char * text, float * out) {
    if (text == NULL || text[0] == '\0' || out == NULL) return 0;
    errno = 0;
    char * end = NULL;
    float value = strtof(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !isfinite(value)) return 0;
    *out = value;
    return 1;
}

static int load_meshbin(const char * path, vkmesh_mesh * mesh) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "vkmesh: failed to open meshbin %s: %s\n", path, strerror(errno));
        return 0;
    }

    char magic[8];
    uint64_t n_vertices = 0;
    uint64_t n_faces = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
    int ok = 1;
    size_t vertex_count = 0;
    size_t face_count = 0;
    size_t uv_count = 0;
    vkmesh_mesh out;
    memset(&out, 0, sizeof(out));
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fread(&n_vertices, sizeof(n_vertices), 1, f) != 1 ||
        fread(&n_faces, sizeof(n_faces), 1, f) != 1 ||
        fread(&flags, sizeof(flags), 1, f) != 1 ||
        fread(&reserved, sizeof(reserved), 1, f) != 1 ||
        memcmp(magic, "TRLMESH1", 8) != 0 ||
        n_vertices == 0 || n_faces == 0 ||
        n_vertices > (uint64_t) INT64_MAX ||
        n_faces > (uint64_t) INT64_MAX ||
        !meshbin_checked_count(n_vertices, 3u, &vertex_count) ||
        !meshbin_checked_count(n_faces, 3u, &face_count) ||
        vertex_count > SIZE_MAX / sizeof(float) ||
        face_count > SIZE_MAX / sizeof(int32_t) ||
        ((flags & 1u) != 0 && (!meshbin_checked_count(n_vertices, 2u, &uv_count) ||
                               uv_count > SIZE_MAX / sizeof(float)))) {
        ok = 0;
    }
    (void) reserved;

    if (ok) {
        out.vertices = (float *) malloc(vertex_count * sizeof(float));
        out.faces = (int32_t *) malloc(face_count * sizeof(int32_t));
        if ((flags & 1u) != 0) {
            out.uvs = (float *) malloc(uv_count * sizeof(float));
        }
        out.n_vertices = (int64_t) n_vertices;
        out.n_faces = (int64_t) n_faces;
        out.vertex_capacity = out.n_vertices;
        out.face_capacity = out.n_faces;
        out.has_uvs = ((flags & 1u) != 0 && out.uvs != NULL) ? 1 : 0;
        if (out.vertices == NULL || out.faces == NULL || (((flags & 1u) != 0) && out.uvs == NULL)) {
            ok = 0;
        } else if (fread(out.vertices, sizeof(float), vertex_count, f) != vertex_count ||
                   fread(out.faces, sizeof(int32_t), face_count, f) != face_count ||
                   (((flags & 1u) != 0) && fread(out.uvs, sizeof(float), uv_count, f) != uv_count)) {
            ok = 0;
        }
    }
    if (ok) {
        for (size_t i = 0; i < face_count; ++i) {
            if (out.faces[i] < 0 || (uint64_t) out.faces[i] >= n_vertices) {
                ok = 0;
                break;
            }
        }
    }
    if (fclose(f) != 0) {
        ok = 0;
    }
    if (!ok) {
        fprintf(stderr, "vkmesh: failed to read meshbin %s\n", path);
        mesh_free(&out);
        return 0;
    }
    *mesh = out;
    return 1;
}

static int write_meshbin(const char * path, const vkmesh_mesh * mesh) {
    if (path == NULL || path[0] == '\0' ||
        mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0) {
        fprintf(stderr, "vkmesh: cannot write empty meshbin %s\n", path != NULL ? path : "(null)");
        return 0;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "vkmesh: failed to open %s for writing: %s\n", path, strerror(errno));
        return 0;
    }

    const char magic[8] = { 'T', 'R', 'L', 'M', 'E', 'S', 'H', '1' };
    const uint64_t n_vertices = (uint64_t) mesh->n_vertices;
    const uint64_t n_faces = (uint64_t) mesh->n_faces;
    const uint32_t flags = (mesh->has_uvs && mesh->uvs != NULL) ? 1u : 0u;
    const uint32_t reserved = 0;
    size_t vertex_count = 0;
    size_t face_count = 0;
    size_t uv_count = 0;
    int ok =
        meshbin_checked_count(n_vertices, 3u, &vertex_count) &&
        meshbin_checked_count(n_faces, 3u, &face_count) &&
        vertex_count <= SIZE_MAX / sizeof(float) &&
        face_count <= SIZE_MAX / sizeof(int32_t) &&
        (flags == 0u || (meshbin_checked_count(n_vertices, 2u, &uv_count) &&
                         uv_count <= SIZE_MAX / sizeof(float)));
    if (ok) {
        ok = fwrite(magic, 1, sizeof(magic), f) == sizeof(magic) &&
             fwrite(&n_vertices, sizeof(n_vertices), 1, f) == 1 &&
             fwrite(&n_faces, sizeof(n_faces), 1, f) == 1 &&
             fwrite(&flags, sizeof(flags), 1, f) == 1 &&
             fwrite(&reserved, sizeof(reserved), 1, f) == 1 &&
             fwrite(mesh->vertices, sizeof(float), vertex_count, f) == vertex_count &&
             fwrite(mesh->faces, sizeof(int32_t), face_count, f) == face_count &&
             (flags == 0u || fwrite(mesh->uvs, sizeof(float), uv_count, f) == uv_count);
    }
    if (fclose(f) != 0) {
        ok = 0;
    }
    if (!ok) {
        fprintf(stderr, "vkmesh: failed to write meshbin %s\n", path);
        return 0;
    }
    return 1;
}

static uint32_t find_memory_type(
    VkPhysicalDevice physical,
    uint32_t type_bits,
    VkMemoryPropertyFlags required_flags,
    VkMemoryPropertyFlags preferred_flags) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    uint32_t fallback = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) == 0 ||
            (props.memoryTypes[i].propertyFlags & required_flags) != required_flags) continue;
        if (fallback == UINT32_MAX) fallback = i;
        if ((props.memoryTypes[i].propertyFlags & preferred_flags) == preferred_flags) return i;
    }
    return fallback;
}

static int vkmesh_device_supports_memory_budget(VkPhysicalDevice physical) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical, NULL, &count, NULL) != VK_SUCCESS || count == 0) {
        return 0;
    }
    VkExtensionProperties * extensions = (VkExtensionProperties *) malloc((size_t) count * sizeof(*extensions));
    if (extensions == NULL) return 0;
    int supported = 0;
    if (vkEnumerateDeviceExtensionProperties(physical, NULL, &count, extensions) == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; ++i) {
            if (strcmp(extensions[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
                supported = 1;
                break;
            }
        }
    }
    free(extensions);
    return supported;
}

static size_t vkmesh_resolve_workspace_budget(VkPhysicalDevice physical) {
    const size_t mib = 1024u * 1024u;
    int requested_mib = g_vkmesh_workspace_budget_mib;
    if (requested_mib <= 0) {
        const char * env = getenv("TRELLIS_VKMESH_GPU_WORKSPACE_BUDGET_MIB");
        int parsed = 0;
        if (env != NULL && env[0] != '\0') {
            if (parse_int_arg(env, &parsed) && parsed > 0) {
                requested_mib = parsed;
            } else {
                fprintf(stderr,
                    "vkmesh: ignoring invalid TRELLIS_VKMESH_GPU_WORKSPACE_BUDGET_MIB='%s'\n",
                    env);
            }
        }
    }
    if (requested_mib > 0) {
        if ((size_t) requested_mib > SIZE_MAX / mib) return SIZE_MAX;
        return (size_t) requested_mib * mib;
    }

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_props;
    memset(&budget_props, 0, sizeof(budget_props));
    budget_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 props2;
    memset(&props2, 0, sizeof(props2));
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    const int has_budget = vkmesh_device_supports_memory_budget(physical);
    if (has_budget) props2.pNext = &budget_props;
    vkGetPhysicalDeviceMemoryProperties2(physical, &props2);

    const VkMemoryPropertyFlags required =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags preferred = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    int has_preferred_type = 0;
    for (uint32_t i = 0; i < props2.memoryProperties.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = props2.memoryProperties.memoryTypes[i].propertyFlags;
        if ((flags & required) == required && (flags & preferred) == preferred) {
            has_preferred_type = 1;
            break;
        }
    }

    VkDeviceSize best_heap_budget = 0;
    VkDeviceSize best_heap_available = 0;
    for (uint32_t i = 0; i < props2.memoryProperties.memoryTypeCount; ++i) {
        const VkMemoryType * type = &props2.memoryProperties.memoryTypes[i];
        if ((type->propertyFlags & required) != required) continue;
        if (has_preferred_type && (type->propertyFlags & preferred) != preferred) continue;
        const uint32_t heap_index = type->heapIndex;
        VkDeviceSize heap_budget = has_budget ?
            budget_props.heapBudget[heap_index] : props2.memoryProperties.memoryHeaps[heap_index].size;
        VkDeviceSize heap_usage = has_budget ? budget_props.heapUsage[heap_index] : 0;
        VkDeviceSize available = heap_budget > heap_usage ? heap_budget - heap_usage : 0;
        if (available > best_heap_available) {
            best_heap_available = available;
            best_heap_budget = heap_budget;
        }
    }

    const VkDeviceSize cap = (VkDeviceSize) 2048u * mib;
    const VkDeviceSize minimum = (VkDeviceSize) 64u * mib;
    VkDeviceSize headroom = best_heap_budget / 8u;
    if (headroom < (VkDeviceSize) 512u * mib) headroom = (VkDeviceSize) 512u * mib;
    VkDeviceSize automatic = best_heap_budget / 4u;
    if (automatic > cap) automatic = cap;
    VkDeviceSize pressure_limit = best_heap_available > headroom ? best_heap_available - headroom : best_heap_available / 2u;
    if (automatic > pressure_limit) automatic = pressure_limit;
    if (automatic < minimum) automatic = minimum;
    if (automatic > (VkDeviceSize) SIZE_MAX) return SIZE_MAX;
    return (size_t) automatic;
}

static void vk_buffer_destroy(vkmesh_vk * vk, vkmesh_vk_buffer * b);
static int vkmesh_vk_init(vkmesh_vk * vk);

static int vk_buffer_create(vkmesh_vk * vk, size_t bytes, vkmesh_vk_buffer * out) {
    if (vk == NULL || out == NULL || bytes == 0) return 0;
    memset(out, 0, sizeof(*out));
    out->bytes = bytes;
    VkBufferCreateInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = (VkDeviceSize) bytes;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(vk->device, &buffer_info, NULL, &out->buffer);
    if (!vkmesh_check_vk_result("vkCreateBuffer", result)) return 0;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk->device, out->buffer, &req);
    if (req.size > (VkDeviceSize) SIZE_MAX ||
        (vk->workspace_budget_bytes > 0 &&
         ((size_t) req.size > vk->workspace_budget_bytes ||
          vk->workspace_current_bytes > vk->workspace_budget_bytes - (size_t) req.size))) {
        vkmesh_set_error(VKMESH_ERROR_OUT_OF_MEMORY);
        fprintf(stderr,
            "vkmesh: GPU workspace budget exceeded request=%.1f MiB current=%.1f MiB budget=%.1f MiB\n",
            (double) req.size / (1024.0 * 1024.0),
            (double) vk->workspace_current_bytes / (1024.0 * 1024.0),
            (double) vk->workspace_budget_bytes / (1024.0 * 1024.0));
        vkDestroyBuffer(vk->device, out->buffer, NULL);
        memset(out, 0, sizeof(*out));
        return 0;
    }
    uint32_t memory_type = find_memory_type(
        vk->physical_device,
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memory_type == UINT32_MAX) {
        vkmesh_set_error(VKMESH_ERROR_VULKAN_UNAVAILABLE);
        fprintf(stderr, "vkmesh: selected device has no host-visible coherent storage-buffer memory type\n");
        vkDestroyBuffer(vk->device, out->buffer, NULL);
        memset(out, 0, sizeof(*out));
        return 0;
    }
    if (!vk->workspace_memory_type_reported) {
        VkPhysicalDeviceMemoryProperties memory_props;
        vkGetPhysicalDeviceMemoryProperties(vk->physical_device, &memory_props);
        const VkMemoryType * type = &memory_props.memoryTypes[memory_type];
        vk->workspace_memory_type = memory_type;
        vk->workspace_memory_type_reported = 1;
        fprintf(stderr,
            "vkmesh: workspace memory type=%u heap=%u device_local=%d host_visible=%d coherent=%d\n",
            memory_type,
            type->heapIndex,
            (type->propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
            (type->propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0,
            (type->propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0);
    }

    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = memory_type;
    result = vkAllocateMemory(vk->device, &alloc_info, NULL, &out->memory);
    if (result == VK_SUCCESS) result = vkBindBufferMemory(vk->device, out->buffer, out->memory, 0);
    if (result == VK_SUCCESS) result = vkMapMemory(vk->device, out->memory, 0, req.size, 0, &out->mapped);
    if (result != VK_SUCCESS) {
        (void) vkmesh_check_vk_result("Vulkan buffer allocation/map", result);
        vk_buffer_destroy(vk, out);
        return 0;
    }
    out->allocation_bytes = (size_t) req.size;
    vk->workspace_current_bytes += out->allocation_bytes;
    if (vk->workspace_current_bytes > vk->workspace_peak_bytes) {
        vk->workspace_peak_bytes = vk->workspace_current_bytes;
    }
    memset(out->mapped, 0, bytes);
    return 1;
}

static void vk_buffer_destroy(vkmesh_vk * vk, vkmesh_vk_buffer * b) {
    if (b == NULL) return;
    if (b->mapped != NULL) vkUnmapMemory(vk->device, b->memory);
    if (b->buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk->device, b->buffer, NULL);
    if (b->memory != VK_NULL_HANDLE) vkFreeMemory(vk->device, b->memory, NULL);
    if (vk != NULL && b->allocation_bytes <= vk->workspace_current_bytes) {
        vk->workspace_current_bytes -= b->allocation_bytes;
    }
    memset(b, 0, sizeof(*b));
}

static void vkmesh_vk_destroy(vkmesh_vk * vk) {
    if (vk == NULL) return;
    if (vk->device != VK_NULL_HANDLE) vkDeviceWaitIdle(vk->device);
    if (vk->workspace_peak_bytes > 0) {
        fprintf(stderr,
            "vkmesh: GPU workspace peak %.1f MiB (budget %.1f MiB, live at destroy %.1f MiB)\n",
            (double) vk->workspace_peak_bytes / (1024.0 * 1024.0),
            (double) vk->workspace_budget_bytes / (1024.0 * 1024.0),
            (double) vk->workspace_current_bytes / (1024.0 * 1024.0));
    }
    if (vk->completion_fence != VK_NULL_HANDLE) vkDestroyFence(vk->device, vk->completion_fence, NULL);
    if (vk->descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(vk->device, vk->descriptor_pool, NULL);
    for (uint32_t i = 0; i < VKMESH_PIPE_COUNT; ++i) {
        if (vk->pipelines[i] != VK_NULL_HANDLE) vkDestroyPipeline(vk->device, vk->pipelines[i], NULL);
    }
    if (vk->pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(vk->device, vk->pipeline_layout, NULL);
    if (vk->descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk->device, vk->descriptor_set_layout, NULL);
    if (vk->command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(vk->device, vk->command_pool, NULL);
    if (vk->device != VK_NULL_HANDLE) vkDestroyDevice(vk->device, NULL);
    if (vk->instance != VK_NULL_HANDLE) vkDestroyInstance(vk->instance, NULL);
    memset(vk, 0, sizeof(*vk));
}

static int vkmesh_acquire_vk(vkmesh_vk ** vk_out, vkmesh_vk * local_vk, int * owns_vk) {
    if (vk_out == NULL || local_vk == NULL || owns_vk == NULL) return 0;
    if (g_active_vkmesh_vk != NULL) {
        *vk_out = g_active_vkmesh_vk;
        *owns_vk = 0;
        return 1;
    }
    if (!vkmesh_vk_init(local_vk)) {
        vkmesh_vk_destroy(local_vk);
        return 0;
    }
    *vk_out = local_vk;
    *owns_vk = 1;
    return 1;
}

static int vkmesh_create_compute_pipeline(
    vkmesh_vk * vk,
    const vkmesh_shader_blob * blob,
    VkPipeline * pipeline) {
    VkShaderModuleCreateInfo shader_info;
    memset(&shader_info, 0, sizeof(shader_info));
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = blob->len;
    shader_info.pCode = (const uint32_t *) blob->data;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkResult result = vkCreateShaderModule(vk->device, &shader_info, NULL, &shader);
    if (!vkmesh_check_vk_result("vkCreateShaderModule", result)) {
        fprintf(stderr, "vkmesh: failed to create shader module for %s\n", blob->name);
        return 0;
    }

    VkPipelineShaderStageCreateInfo stage;
    memset(&stage, 0, sizeof(stage));
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = vk->pipeline_layout;
    result = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, pipeline);
    vkDestroyShaderModule(vk->device, shader, NULL);
    if (!vkmesh_check_vk_result("vkCreateComputePipelines", result)) {
        fprintf(stderr, "vkmesh: failed to create compute pipeline for %s\n", blob->name);
        return 0;
    }
    return 1;
}

static int vkmesh_vk_init(vkmesh_vk * vk) {
    memset(vk, 0, sizeof(*vk));
    VkApplicationInfo app;
    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "vkmesh";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    VkResult result = vkCreateInstance(&instance_info, NULL, &vk->instance);
    if (!vkmesh_check_vk_result("vkCreateInstance", result)) {
        vkmesh_set_error(VKMESH_ERROR_VULKAN_UNAVAILABLE);
        return 0;
    }

    uint32_t physical_count = 0;
    result = vkEnumeratePhysicalDevices(vk->instance, &physical_count, NULL);
    if (result != VK_SUCCESS || physical_count == 0) {
        vkmesh_set_error(VKMESH_ERROR_VULKAN_UNAVAILABLE);
        if (result != VK_SUCCESS) {
            (void) vkmesh_check_vk_result("vkEnumeratePhysicalDevices(count)", result);
            vkmesh_set_error(VKMESH_ERROR_VULKAN_UNAVAILABLE);
        } else {
            fprintf(stderr, "vkmesh: no Vulkan physical devices\n");
        }
        return 0;
    }
    VkPhysicalDevice * physical = (VkPhysicalDevice *) malloc((size_t) physical_count * sizeof(*physical));
    if (physical == NULL) {
        vkmesh_set_error(VKMESH_ERROR_OUT_OF_MEMORY);
        return 0;
    }
    result = vkEnumeratePhysicalDevices(vk->instance, &physical_count, physical);
    if (!vkmesh_check_vk_result("vkEnumeratePhysicalDevices(list)", result)) {
        free(physical);
        return 0;
    }
    int found = 0;
    if (g_vkmesh_device_index < 0 || (uint32_t) g_vkmesh_device_index >= physical_count) {
        vkmesh_set_error(VKMESH_ERROR_INVALID_ARGUMENT);
        fprintf(stderr,
            "vkmesh: Vulkan device index %d is out of range (available devices: %u)\n",
            g_vkmesh_device_index,
            physical_count);
        free(physical);
        return 0;
    }
    VkPhysicalDevice selected = physical[(uint32_t) g_vkmesh_device_index];
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(selected, &family_count, NULL);
    VkQueueFamilyProperties * families =
        (VkQueueFamilyProperties *) malloc((size_t) family_count * sizeof(*families));
    if (families == NULL) {
        vkmesh_set_error(VKMESH_ERROR_OUT_OF_MEMORY);
        free(physical);
        return 0;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(selected, &family_count, families);
    for (uint32_t q = 0; q < family_count; ++q) {
        if ((families[q].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            vk->physical_device = selected;
            vk->queue_family = q;
            found = 1;
            break;
        }
    }
    free(families);
    free(physical);
    if (!found) {
        vkmesh_set_error(VKMESH_ERROR_VULKAN_UNAVAILABLE);
        fprintf(stderr, "vkmesh: Vulkan device %d has no compute queue\n", g_vkmesh_device_index);
        return 0;
    }

    VkPhysicalDeviceProperties device_props;
    vkGetPhysicalDeviceProperties(vk->physical_device, &device_props);
    vk->workspace_budget_bytes = vkmesh_resolve_workspace_budget(vk->physical_device);
    fprintf(stderr,
        "vkmesh: using Vulkan device %d: %s (GPU workspace budget %.1f MiB)\n",
        g_vkmesh_device_index,
        device_props.deviceName,
        (double) vk->workspace_budget_bytes / (1024.0 * 1024.0));

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = vk->queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkDeviceCreateInfo device_info;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    result = vkCreateDevice(vk->physical_device, &device_info, NULL, &vk->device);
    if (!vkmesh_check_vk_result("vkCreateDevice", result)) {
        vkmesh_set_error(VKMESH_ERROR_VULKAN_UNAVAILABLE);
        return 0;
    }
    vkGetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);

    VkCommandPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = vk->queue_family;
    result = vkCreateCommandPool(vk->device, &pool_info, NULL, &vk->command_pool);
    if (!vkmesh_check_vk_result("vkCreateCommandPool", result)) return 0;

    VkDescriptorSetLayoutBinding bindings[4];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 4; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 4;
    layout_info.pBindings = bindings;
    result = vkCreateDescriptorSetLayout(vk->device, &layout_info, NULL, &vk->descriptor_set_layout);
    if (!vkmesh_check_vk_result("vkCreateDescriptorSetLayout", result)) return 0;

    VkPushConstantRange push_range;
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(vkmesh_push);
    VkPipelineLayoutCreateInfo pipeline_layout_info;
    memset(&pipeline_layout_info, 0, sizeof(pipeline_layout_info));
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &vk->descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    result = vkCreatePipelineLayout(vk->device, &pipeline_layout_info, NULL, &vk->pipeline_layout);
    if (!vkmesh_check_vk_result("vkCreatePipelineLayout", result)) return 0;

    for (uint32_t i = 0; i < VKMESH_PIPE_COUNT; ++i) {
        if (!vkmesh_create_compute_pipeline(vk, &vkmesh_shaders[i], &vk->pipelines[i])) return 0;
    }

    VkDescriptorPoolSize pool_size;
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 12;
    VkDescriptorPoolCreateInfo descriptor_pool_info;
    memset(&descriptor_pool_info, 0, sizeof(descriptor_pool_info));
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptor_pool_info.maxSets = 3;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &pool_size;
    result = vkCreateDescriptorPool(vk->device, &descriptor_pool_info, NULL, &vk->descriptor_pool);
    if (!vkmesh_check_vk_result("vkCreateDescriptorPool", result)) return 0;

    VkDescriptorSetAllocateInfo set_alloc;
    memset(&set_alloc, 0, sizeof(set_alloc));
    set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc.descriptorPool = vk->descriptor_pool;
    set_alloc.descriptorSetCount = 1;
    set_alloc.pSetLayouts = &vk->descriptor_set_layout;
    result = vkAllocateDescriptorSets(vk->device, &set_alloc, &vk->descriptor_set);
    if (!vkmesh_check_vk_result("vkAllocateDescriptorSets", result)) return 0;

    VkCommandBufferAllocateInfo cmd_alloc;
    memset(&cmd_alloc, 0, sizeof(cmd_alloc));
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = vk->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(vk->device, &cmd_alloc, &vk->command_buffer);
    if (!vkmesh_check_vk_result("vkAllocateCommandBuffers", result)) return 0;

    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    result = vkCreateFence(vk->device, &fence_info, NULL, &vk->completion_fence);
    if (!vkmesh_check_vk_result("vkCreateFence", result)) return 0;
    return 1;
}

static int vkmesh_submit_and_wait(vkmesh_vk * vk, const VkSubmitInfo * submit) {
    VkResult result = vkResetFences(vk->device, 1, &vk->completion_fence);
    if (result == VK_SUCCESS) {
        result = vkQueueSubmit(vk->queue, 1, submit, vk->completion_fence);
    }
    if (result == VK_SUCCESS) {
        result = vkWaitForFences(vk->device, 1, &vk->completion_fence, VK_TRUE, UINT64_MAX);
    }
    if (result != VK_SUCCESS) {
        vkmesh_set_error(VKMESH_ERROR_VULKAN);
        fprintf(stderr, "vkmesh: Vulkan submit/wait failed: VkResult=%d\n", (int) result);
        return 0;
    }
    return 1;
}

static void vkmesh_update_descriptor_set_handle(
    vkmesh_vk * vk,
    VkDescriptorSet descriptor_set,
    vkmesh_vk_buffer buffers[4]) {
    VkDescriptorBufferInfo infos[4];
    VkWriteDescriptorSet writes[4];
    memset(infos, 0, sizeof(infos));
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < 4; ++i) {
        infos[i].buffer = buffers[i].buffer;
        infos[i].offset = 0;
        infos[i].range = (VkDeviceSize) buffers[i].bytes;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(vk->device, 4, writes, 0, NULL);
}

static void vkmesh_update_descriptor_set(vkmesh_vk * vk, vkmesh_vk_buffer buffers[4]) {
    vkmesh_update_descriptor_set_handle(vk, vk->descriptor_set, buffers);
}

static int vkmesh_dispatch(
    vkmesh_vk * vk,
    vkmesh_pipeline_kind pipeline_kind,
    vkmesh_vk_buffer buffers[4],
    const vkmesh_push * push,
    uint32_t groups_x) {
    vkmesh_update_descriptor_set(vk, buffers);
    if (vkResetCommandBuffer(vk->command_buffer, 0) != VK_SUCCESS) return 0;

    VkCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(vk->command_buffer, &begin) != VK_SUCCESS) return 0;
    vkCmdBindPipeline(vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipelines[pipeline_kind]);
    vkCmdBindDescriptorSets(vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_layout, 0, 1, &vk->descriptor_set, 0, NULL);
    vkCmdPushConstants(vk->command_buffer, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*push), push);
    vkCmdDispatch(vk->command_buffer, groups_x, 1, 1);
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(
        vk->command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
    if (vkEndCommandBuffer(vk->command_buffer) != VK_SUCCESS) return 0;

    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &vk->command_buffer;
    return vkmesh_submit_and_wait(vk, &submit);
}

static int vkmesh_dispatch_pair(
    vkmesh_vk * vk,
    vkmesh_pipeline_kind first_kind,
    vkmesh_vk_buffer first_buffers[4],
    const vkmesh_push * first_push,
    uint32_t first_groups,
    vkmesh_pipeline_kind second_kind,
    vkmesh_vk_buffer second_buffers[4],
    uint32_t second_n,
    uint32_t second_groups) {
    VkDescriptorSetLayout layouts[2] = { vk->descriptor_set_layout, vk->descriptor_set_layout };
    VkDescriptorSetAllocateInfo alloc;
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = vk->descriptor_pool;
    alloc.descriptorSetCount = 2;
    alloc.pSetLayouts = layouts;
    VkDescriptorSet sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkResult result = vkAllocateDescriptorSets(vk->device, &alloc, sets);
    if (!vkmesh_check_vk_result("vkAllocateDescriptorSets(batch)", result)) return 0;
    vkmesh_update_descriptor_set_handle(vk, sets[0], first_buffers);
    vkmesh_update_descriptor_set_handle(vk, sets[1], second_buffers);

    int ok = 0;
    if (vkResetCommandBuffer(vk->command_buffer, 0) == VK_SUCCESS) {
        VkCommandBufferBeginInfo begin;
        memset(&begin, 0, sizeof(begin));
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(vk->command_buffer, &begin) == VK_SUCCESS) {
            vkCmdBindPipeline(vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipelines[first_kind]);
            vkCmdBindDescriptorSets(
                vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_layout, 0, 1, &sets[0], 0, NULL);
            vkCmdPushConstants(
                vk->command_buffer, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*first_push), first_push);
            vkCmdDispatch(vk->command_buffer, first_groups, 1, 1);

            VkMemoryBarrier barrier;
            memset(&barrier, 0, sizeof(barrier));
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(
                vk->command_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &barrier, 0, NULL, 0, NULL);

            vkmesh_push second_push;
            memset(&second_push, 0, sizeof(second_push));
            second_push.n = second_n;
            vkCmdBindPipeline(vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipelines[second_kind]);
            vkCmdBindDescriptorSets(
                vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_layout, 0, 1, &sets[1], 0, NULL);
            vkCmdPushConstants(
                vk->command_buffer, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(second_push), &second_push);
            vkCmdDispatch(vk->command_buffer, second_groups, 1, 1);

            barrier.dstAccessMask =
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(
                vk->command_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                0, 1, &barrier, 0, NULL, 0, NULL);
            if (vkEndCommandBuffer(vk->command_buffer) == VK_SUCCESS) {
                VkSubmitInfo submit;
                memset(&submit, 0, sizeof(submit));
                submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit.commandBufferCount = 1;
                submit.pCommandBuffers = &vk->command_buffer;
                ok = vkmesh_submit_and_wait(vk, &submit);
            }
        }
    }
    result = vkFreeDescriptorSets(vk->device, vk->descriptor_pool, 2, sets);
    if (!vkmesh_check_vk_result("vkFreeDescriptorSets(batch)", result)) ok = 0;
    return ok;
}

#define VKMESH_SORT_TILE_SIZE 256u

static int vkmesh_sort_records_vulkan(
    vkmesh_vk * vk,
    vkmesh_vk_buffer * records,
    vkmesh_vk_buffer * dummy,
    size_t record_count,
    vkmesh_pipeline_kind pipeline_kind) {
    if (record_count <= 1) return 1;
    if (record_count > UINT32_MAX) return 0;

    size_t record_size = 0;
    vkmesh_pipeline_kind merge_pipeline = VKMESH_PIPE_COUNT;
    if (pipeline_kind == VKMESH_PIPE_SORT_EDGES) {
        record_size = sizeof(vkmesh_edge);
        merge_pipeline = VKMESH_PIPE_MERGE_EDGES;
    } else if (pipeline_kind == VKMESH_PIPE_SORT_FACE_KEYS) {
        record_size = sizeof(vkmesh_face_key);
        merge_pipeline = VKMESH_PIPE_MERGE_FACE_KEYS;
    } else {
        return 0;
    }
    if (record_count > SIZE_MAX / record_size || records->bytes < record_count * record_size) return 0;

    uint32_t merge_passes = 0u;
    for (size_t width = VKMESH_SORT_TILE_SIZE; width < record_count; width <<= 1u) {
        ++merge_passes;
    }

    vkmesh_vk_buffer temporary;
    memset(&temporary, 0, sizeof(temporary));
    int ok = 0;
    if (merge_passes > 0u && !vk_buffer_create(vk, record_count * record_size, &temporary)) goto cleanup;

    /* Both sort pipelines keep the same descriptors for the entire batch.
       Push parity chooses the source/destination without descriptor updates. */
    vkmesh_vk_buffer buffers[4];
    buffers[0] = *records;
    buffers[1] = merge_passes > 0u ? temporary : *records;
    buffers[2] = *dummy;
    buffers[3] = *dummy;
    vkmesh_update_descriptor_set(vk, buffers);
    if (vkResetCommandBuffer(vk->command_buffer, 0) != VK_SUCCESS) goto cleanup;

    VkCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(vk->command_buffer, &begin) != VK_SUCCESS) goto cleanup;

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) record_count;
    push.aux1 = merge_passes & 1u;
    uint32_t groups = (uint32_t) ((record_count + VKMESH_SORT_TILE_SIZE - 1u) / VKMESH_SORT_TILE_SIZE);
    vkCmdBindDescriptorSets(
        vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_layout,
        0, 1, &vk->descriptor_set, 0, NULL);
    vkCmdBindPipeline(vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipelines[pipeline_kind]);
    vkCmdPushConstants(
        vk->command_buffer, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(push), &push);
    vkCmdDispatch(vk->command_buffer, groups, 1, 1);

    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    if (merge_passes > 0u) {
        vkCmdPipelineBarrier(
            vk->command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, NULL, 0, NULL);
        vkCmdBindPipeline(vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipelines[merge_pipeline]);
    }

    uint32_t source_is_temporary = merge_passes & 1u;
    uint32_t pass_index = 0u;
    for (size_t width = VKMESH_SORT_TILE_SIZE; width < record_count; width <<= 1u) {
        memset(&push, 0, sizeof(push));
        push.n = (uint32_t) record_count;
        push.aux0 = (uint32_t) width;
        push.aux1 = source_is_temporary;
        vkCmdPushConstants(
            vk->command_buffer, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(push), &push);
        vkCmdDispatch(vk->command_buffer, groups, 1, 1);
        source_is_temporary ^= 1u;
        ++pass_index;
        if (pass_index < merge_passes) {
            vkCmdPipelineBarrier(
                vk->command_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &barrier, 0, NULL, 0, NULL);
        }
    }
    if (source_is_temporary != 0u || pass_index != merge_passes) {
        (void) vkEndCommandBuffer(vk->command_buffer);
        goto cleanup;
    }

    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(
        vk->command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
    if (vkEndCommandBuffer(vk->command_buffer) != VK_SUCCESS) goto cleanup;

    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &vk->command_buffer;
    ok = vkmesh_submit_and_wait(vk, &submit);

cleanup:
    vk_buffer_destroy(vk, &temporary);
    return ok;
}

#define VKMESH_SCAN_BLOCK_SIZE 256u
#define VKMESH_SCAN_MAX_LEVELS 8u

/* Inclusive scan with a fixed output buffer. Each upward pass scans blocks
   locally and emits block sums; downward passes add the scanned parent sum. */
static int vkmesh_scan_u32_inclusive(
    vkmesh_vk * vk,
    vkmesh_vk_buffer * input,
    vkmesh_vk_buffer * output,
    uint32_t count,
    vkmesh_vk_buffer * dummy) {
    if (vk == NULL || input == NULL || output == NULL || dummy == NULL || count == 0u ||
        input->bytes < (size_t) count * sizeof(uint32_t) ||
        output->bytes < (size_t) count * sizeof(uint32_t) ||
        dummy->bytes < sizeof(uint32_t)) {
        return 0;
    }

    vkmesh_vk_buffer block_sums[VKMESH_SCAN_MAX_LEVELS];
    uint32_t block_sum_counts[VKMESH_SCAN_MAX_LEVELS];
    memset(block_sums, 0, sizeof(block_sums));
    memset(block_sum_counts, 0, sizeof(block_sum_counts));
    uint32_t level_count = 0u;
    int ok = 0;

    vkmesh_vk_buffer * level_input = input;
    vkmesh_vk_buffer * level_output = output;
    uint32_t level_element_count = count;
    for (;;) {
        uint32_t groups = (level_element_count - 1u) / VKMESH_SCAN_BLOCK_SIZE + 1u;
        vkmesh_vk_buffer * level_block_sums = dummy;
        if (groups > 1u) {
            if (level_count >= VKMESH_SCAN_MAX_LEVELS ||
                !vk_buffer_create(vk, (size_t) groups * sizeof(uint32_t), &block_sums[level_count])) {
                goto cleanup;
            }
            block_sum_counts[level_count] = groups;
            level_block_sums = &block_sums[level_count];
        }

        vkmesh_vk_buffer buffers[4];
        buffers[0] = *level_input;
        buffers[1] = *level_output;
        buffers[2] = *level_block_sums;
        buffers[3] = *dummy;
        vkmesh_push push;
        memset(&push, 0, sizeof(push));
        push.n = level_element_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_BLOCK_SCAN_U32, buffers, &push, groups)) goto cleanup;
        if (groups == 1u) break;

        level_element_count = groups;
        level_input = &block_sums[level_count];
        level_output = &block_sums[level_count];
        ++level_count;
    }

    for (uint32_t depth = level_count; depth > 0u; --depth) {
        vkmesh_vk_buffer * data = depth == 1u ? output : &block_sums[depth - 2u];
        uint32_t data_count = depth == 1u ? count : block_sum_counts[depth - 2u];
        vkmesh_vk_buffer buffers[4];
        buffers[0] = *data;
        buffers[1] = block_sums[depth - 1u];
        buffers[2] = *dummy;
        buffers[3] = *dummy;
        vkmesh_push push;
        memset(&push, 0, sizeof(push));
        push.n = data_count;
        uint32_t groups = (data_count - 1u) / VKMESH_SCAN_BLOCK_SIZE + 1u;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_ADD_BLOCK_OFFSETS_U32, buffers, &push, groups)) goto cleanup;
    }
    ok = 1;

cleanup:
    for (uint32_t i = 0u; i < VKMESH_SCAN_MAX_LEVELS; ++i) {
        vk_buffer_destroy(vk, &block_sums[i]);
    }
    return ok;
}

static int expand_edges_vulkan(const vkmesh_mesh * mesh, vkmesh_edge ** edges_out, int64_t * edge_count_out) {
    *edges_out = NULL;
    *edge_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX) return 0;

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer buffers[4];
    memset(buffers, 0, sizeof(buffers));
    int ok = 0;
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t edge_count = (size_t) mesh->n_faces * 3u;
    if (edge_count > UINT32_MAX) goto cleanup;
    const size_t edges_bytes = edge_count * sizeof(vkmesh_edge);
    if (!vk_buffer_create(vk, faces_bytes, &buffers[0]) ||
        !vk_buffer_create(vk, vertices_bytes, &buffers[1]) ||
        !vk_buffer_create(vk, edges_bytes, &buffers[2]) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &buffers[3])) {
        goto cleanup;
    }
    memcpy(buffers[0].mapped, mesh->faces, faces_bytes);
    memcpy(buffers[1].mapped, mesh->vertices, vertices_bytes);
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_faces;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_EXPAND_EDGES, buffers, &push, groups)) goto cleanup;
    if (!vkmesh_sort_records_vulkan(vk, &buffers[2], &buffers[3], edge_count, VKMESH_PIPE_SORT_EDGES)) goto cleanup;

    vkmesh_edge * edges = (vkmesh_edge *) malloc(edge_count * sizeof(vkmesh_edge));
    if (edges == NULL) goto cleanup;
    memcpy(edges, buffers[2].mapped, edge_count * sizeof(vkmesh_edge));
    *edges_out = edges;
    *edge_count_out = (int64_t) edge_count;
    ok = 1;

cleanup:
    for (uint32_t i = 0; i < 4; ++i) vk_buffer_destroy(vk, &buffers[i]);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int expand_edges_device(
    vkmesh_vk * vk,
    const vkmesh_mesh * mesh,
    vkmesh_vk_buffer * faces_buffer,
    vkmesh_vk_buffer * vertices_buffer,
    vkmesh_vk_buffer * edges_buffer_out,
    vkmesh_vk_buffer * dummy_buffer_out,
    size_t * edge_count_out,
    size_t * edge_sort_count_out) {
    memset(edges_buffer_out, 0, sizeof(*edges_buffer_out));
    memset(dummy_buffer_out, 0, sizeof(*dummy_buffer_out));
    *edge_count_out = 0;
    *edge_sort_count_out = 0;
    if (vk == NULL || mesh == NULL || faces_buffer == NULL || vertices_buffer == NULL ||
        mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX) {
        return 0;
    }

    const size_t edge_count = (size_t) mesh->n_faces * 3u;
    if (edge_count > UINT32_MAX) return 0;
    const size_t edges_bytes = edge_count * sizeof(vkmesh_edge);
    if (!vk_buffer_create(vk, edges_bytes, edges_buffer_out) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), dummy_buffer_out)) {
        vk_buffer_destroy(vk, edges_buffer_out);
        vk_buffer_destroy(vk, dummy_buffer_out);
        return 0;
    }

    vkmesh_vk_buffer buffers[4];
    buffers[0] = *faces_buffer;
    buffers[1] = *vertices_buffer;
    buffers[2] = *edges_buffer_out;
    buffers[3] = *dummy_buffer_out;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_faces;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_EXPAND_EDGES, buffers, &push, groups) ||
        !vkmesh_sort_records_vulkan(vk, edges_buffer_out, dummy_buffer_out, edge_count, VKMESH_PIPE_SORT_EDGES)) {
        vk_buffer_destroy(vk, edges_buffer_out);
        vk_buffer_destroy(vk, dummy_buffer_out);
        return 0;
    }
    *edge_count_out = edge_count;
    *edge_sort_count_out = edge_count;
    return 1;
}

static int uf_find(int * parent, int x) {
    int r = x;
    while (parent[r] != r) r = parent[r];
    while (parent[x] != x) {
        int p = parent[x];
        parent[x] = r;
        x = p;
    }
    return r;
}

static void uf_union(int * parent, int a, int b) {
    int ra = uf_find(parent, a);
    int rb = uf_find(parent, b);
    if (ra == rb) return;
    if (ra < rb) parent[rb] = ra;
    else parent[ra] = rb;
}

static double edge_length(const vkmesh_mesh * mesh, int32_t a, int32_t b) {
    const float * va = mesh->vertices + (size_t) a * 3u;
    const float * vb = mesh->vertices + (size_t) b * 3u;
    double dx = (double) va[0] - (double) vb[0];
    double dy = (double) va[1] - (double) vb[1];
    double dz = (double) va[2] - (double) vb[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

static float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static double face_area(const vkmesh_mesh * mesh, int64_t face_id) {
    const int32_t * f = mesh->faces + (size_t) face_id * 3u;
    const float * a = mesh->vertices + (size_t) f[0] * 3u;
    const float * b = mesh->vertices + (size_t) f[1] * 3u;
    const float * c = mesh->vertices + (size_t) f[2] * 3u;
    double abx = (double) b[0] - (double) a[0];
    double aby = (double) b[1] - (double) a[1];
    double abz = (double) b[2] - (double) a[2];
    double acx = (double) c[0] - (double) a[0];
    double acy = (double) c[1] - (double) a[1];
    double acz = (double) c[2] - (double) a[2];
    double cx = aby * acz - abz * acy;
    double cy = abz * acx - abx * acz;
    double cz = abx * acy - aby * acx;
    return 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
}

static int mesh_remove_unreferenced_vertices(vkmesh_mesh * mesh) {
    if (mesh->n_vertices <= 0) return 1;
    uint8_t * referenced = (uint8_t *) calloc((size_t) mesh->n_vertices, sizeof(uint8_t));
    int32_t * map = (int32_t *) malloc((size_t) mesh->n_vertices * sizeof(int32_t));
    if (referenced == NULL || map == NULL) {
        free(referenced);
        free(map);
        return 0;
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        referenced[f[0]] = 1u;
        referenced[f[1]] = 1u;
        referenced[f[2]] = 1u;
    }
    int64_t new_count = 0;
    for (int64_t i = 0; i < mesh->n_vertices; ++i) {
        if (referenced[i]) map[i] = (int32_t) new_count++;
        else map[i] = -1;
    }

    float * new_vertices = NULL;
    float * new_uvs = NULL;
    if (new_count > 0) {
        new_vertices = (float *) malloc((size_t) new_count * 3u * sizeof(float));
        if (new_vertices == NULL) {
            free(referenced);
            free(map);
            return 0;
        }
        if (mesh->has_uvs && mesh->uvs != NULL) {
            new_uvs = (float *) malloc((size_t) new_count * 2u * sizeof(float));
            if (new_uvs == NULL) {
                free(new_vertices);
                free(referenced);
                free(map);
                return 0;
            }
        }
    }
    for (int64_t i = 0; i < mesh->n_vertices; ++i) {
        int32_t dst = map[i];
        if (dst < 0) continue;
        memcpy(new_vertices + (size_t) dst * 3u, mesh->vertices + (size_t) i * 3u, 3u * sizeof(float));
        if (new_uvs != NULL) {
            memcpy(new_uvs + (size_t) dst * 2u, mesh->uvs + (size_t) i * 2u, 2u * sizeof(float));
        }
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        int32_t * f = mesh->faces + (size_t) i * 3u;
        f[0] = map[f[0]];
        f[1] = map[f[1]];
        f[2] = map[f[2]];
    }
    free(mesh->vertices);
    free(mesh->uvs);
    mesh->vertices = new_vertices;
    mesh->uvs = new_uvs;
    mesh->n_vertices = new_count;
    mesh->vertex_capacity = new_count;
    if (new_uvs == NULL) mesh->has_uvs = 0;
    free(referenced);
    free(map);
    return 1;
}

static int mesh_remove_faces_by_mask(vkmesh_mesh * mesh, const uint8_t * keep_mask, int * removed_faces) {
    int64_t keep_count = 0;
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        if (keep_mask[i]) ++keep_count;
    }
    int32_t * new_faces = NULL;
    if (keep_count > 0) {
        new_faces = (int32_t *) malloc((size_t) keep_count * 3u * sizeof(int32_t));
        if (new_faces == NULL) return 0;
    }
    int64_t dst = 0;
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        if (!keep_mask[i]) continue;
        memcpy(new_faces + (size_t) dst * 3u, mesh->faces + (size_t) i * 3u, 3u * sizeof(int32_t));
        ++dst;
    }
    if (removed_faces != NULL) *removed_faces = (int) (mesh->n_faces - keep_count);
    free(mesh->faces);
    mesh->faces = new_faces;
    mesh->n_faces = keep_count;
    mesh->face_capacity = keep_count;
    return mesh_remove_unreferenced_vertices(mesh);
}

static int mesh_remove_faces_by_mask_vulkan(vkmesh_mesh * mesh, const uint8_t * keep_mask, int * removed_faces);

static void vkmesh_device_mesh_destroy(vkmesh_device_mesh * dm) {
    if (dm == NULL || dm->vk == NULL) return;
    vk_buffer_destroy(dm->vk, &dm->vertices);
    vk_buffer_destroy(dm->vk, &dm->faces);
    memset(dm, 0, sizeof(*dm));
}

static int vkmesh_device_mesh_upload(vkmesh_vk * vk, const vkmesh_mesh * mesh, vkmesh_device_mesh * dm) {
    if (vk == NULL || mesh == NULL || dm == NULL ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }
    memset(dm, 0, sizeof(*dm));
    dm->vk = vk;
    dm->n_vertices = (uint32_t) mesh->n_vertices;
    dm->n_faces = (uint32_t) mesh->n_faces;
    const size_t vertices_bytes = (size_t) dm->n_vertices * 3u * sizeof(float);
    const size_t faces_bytes = (size_t) dm->n_faces * 3u * sizeof(int32_t);
    if (!vk_buffer_create(vk, vertices_bytes, &dm->vertices) ||
        !vk_buffer_create(vk, faces_bytes, &dm->faces)) {
        vkmesh_device_mesh_destroy(dm);
        return 0;
    }
    memcpy(dm->vertices.mapped, mesh->vertices, vertices_bytes);
    memcpy(dm->faces.mapped, mesh->faces, faces_bytes);
    return 1;
}

static int vkmesh_device_mesh_download(vkmesh_device_mesh * dm, vkmesh_mesh * mesh) {
    if (dm == NULL || mesh == NULL || dm->vk == NULL ||
        dm->n_vertices == 0u || dm->n_faces == 0u) {
        return 0;
    }
    float * vertices = (float *) malloc((size_t) dm->n_vertices * 3u * sizeof(float));
    int32_t * faces = (int32_t *) malloc((size_t) dm->n_faces * 3u * sizeof(int32_t));
    if (vertices == NULL || faces == NULL) {
        free(vertices);
        free(faces);
        return 0;
    }
    memcpy(vertices, dm->vertices.mapped, (size_t) dm->n_vertices * 3u * sizeof(float));
    memcpy(faces, dm->faces.mapped, (size_t) dm->n_faces * 3u * sizeof(int32_t));
    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = vertices;
    mesh->uvs = NULL;
    mesh->faces = faces;
    mesh->n_vertices = (int64_t) dm->n_vertices;
    mesh->n_faces = (int64_t) dm->n_faces;
    mesh->vertex_capacity = mesh->n_vertices;
    mesh->face_capacity = mesh->n_faces;
    mesh->has_uvs = 0;
    return 1;
}

static int compact_device_mesh_from_flags(
    vkmesh_device_mesh * dm,
    vkmesh_vk_buffer * flags_buffer,
    uint32_t flags_offset,
    uint32_t * removed_faces) {
    if (removed_faces != NULL) *removed_faces = 0u;
    if (dm == NULL || dm->vk == NULL || flags_buffer == NULL ||
        dm->n_faces == 0u || dm->n_vertices == 0u) {
        return 0;
    }
    vkmesh_vk * vk = dm->vk;
    vkmesh_vk_buffer out_faces;
    vkmesh_vk_buffer counter;
    vkmesh_vk_buffer vertex_map;
    vkmesh_vk_buffer out_vertices;
    memset(&out_faces, 0, sizeof(out_faces));
    memset(&counter, 0, sizeof(counter));
    memset(&vertex_map, 0, sizeof(vertex_map));
    memset(&out_vertices, 0, sizeof(out_vertices));

    const size_t faces_bytes = (size_t) dm->n_faces * 3u * sizeof(int32_t);
    const size_t vertex_map_bytes = (size_t) dm->n_vertices * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) dm->n_vertices * 3u * sizeof(float);
    int ok = 0;
    if (!vk_buffer_create(vk, faces_bytes, &out_faces) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter) ||
        !vk_buffer_create(vk, vertex_map_bytes, &vertex_map) ||
        !vk_buffer_create(vk, vertices_bytes, &out_vertices)) {
        goto cleanup;
    }
    memset(vertex_map.mapped, 0xff, vertex_map_bytes);

    vkmesh_vk_buffer compact_buffers[4];
    compact_buffers[0] = dm->faces;
    compact_buffers[1] = *flags_buffer;
    compact_buffers[2] = out_faces;
    compact_buffers[3] = counter;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_faces;
    push.aux0 = flags_offset;
    uint32_t groups = (dm->n_faces + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_FACES, compact_buffers, &push, groups)) goto cleanup;

    uint32_t keep_count = ((const uint32_t *) counter.mapped)[0];
    if (keep_count == 0u || keep_count > dm->n_faces) goto cleanup;
    if (removed_faces != NULL) *removed_faces = dm->n_faces - keep_count;

    ((uint32_t *) counter.mapped)[0] = 0u;
    vkmesh_vk_buffer assign_buffers[4];
    assign_buffers[0] = out_faces;
    assign_buffers[1] = vertex_map;
    assign_buffers[2] = counter;
    assign_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = keep_count * 3u;
    groups = (push.n + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ASSIGN_VERTEX_MAP, assign_buffers, &push, groups)) goto cleanup;
    uint32_t vertex_count = ((const uint32_t *) counter.mapped)[0];
    if (vertex_count == 0u || vertex_count > dm->n_vertices) goto cleanup;

    vkmesh_vk_buffer remap_buffers[4];
    remap_buffers[0] = out_faces;
    remap_buffers[1] = vertex_map;
    remap_buffers[2] = counter;
    remap_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = keep_count * 3u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_REMAP_FACES, remap_buffers, &push, groups)) goto cleanup;

    vkmesh_vk_buffer vertex_buffers[4];
    vertex_buffers[0] = dm->vertices;
    vertex_buffers[1] = vertex_map;
    vertex_buffers[2] = out_vertices;
    vertex_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_vertices;
    groups = (push.n + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_VERTICES, vertex_buffers, &push, groups)) goto cleanup;

    vk_buffer_destroy(vk, &dm->faces);
    vk_buffer_destroy(vk, &dm->vertices);
    dm->faces = out_faces;
    dm->vertices = out_vertices;
    memset(&out_faces, 0, sizeof(out_faces));
    memset(&out_vertices, 0, sizeof(out_vertices));
    dm->n_faces = keep_count;
    dm->n_vertices = vertex_count;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &out_faces);
    vk_buffer_destroy(vk, &counter);
    vk_buffer_destroy(vk, &vertex_map);
    vk_buffer_destroy(vk, &out_vertices);
    return ok;
}

static int compact_mesh_from_device_flags(
    vkmesh_vk * vk,
    const vkmesh_mesh * mesh,
    vkmesh_vk_buffer * faces_buffer,
    vkmesh_vk_buffer * vertices_buffer,
    vkmesh_vk_buffer * flags_buffer,
    uint32_t flags_offset,
    float ** vertices_out,
    int64_t * vertex_count_out,
    int32_t ** faces_out,
    int64_t * face_count_out) {
    *vertices_out = NULL;
    *vertex_count_out = 0;
    *faces_out = NULL;
    *face_count_out = 0;
    if (vk == NULL || mesh == NULL || faces_buffer == NULL || vertices_buffer == NULL || flags_buffer == NULL ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX || mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk_buffer out_faces;
    vkmesh_vk_buffer counter;
    vkmesh_vk_buffer vertex_map;
    vkmesh_vk_buffer out_vertices;
    memset(&out_faces, 0, sizeof(out_faces));
    memset(&counter, 0, sizeof(counter));
    memset(&vertex_map, 0, sizeof(vertex_map));
    memset(&out_vertices, 0, sizeof(out_vertices));
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertex_map_bytes = (size_t) mesh->n_vertices * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    int ok = 0;
    if (!vk_buffer_create(vk, faces_bytes, &out_faces) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter) ||
        !vk_buffer_create(vk, vertex_map_bytes, &vertex_map) ||
        !vk_buffer_create(vk, vertices_bytes, &out_vertices)) {
        goto cleanup;
    }
    memset(vertex_map.mapped, 0xff, vertex_map_bytes);

    vkmesh_vk_buffer compact_buffers[4];
    compact_buffers[0] = *faces_buffer;
    compact_buffers[1] = *flags_buffer;
    compact_buffers[2] = out_faces;
    compact_buffers[3] = counter;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_faces;
    push.aux0 = flags_offset;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_FACES, compact_buffers, &push, groups)) goto cleanup;

    uint32_t keep_count_u32 = ((const uint32_t *) counter.mapped)[0];
    if (keep_count_u32 == 0) {
        ok = 1;
        goto cleanup;
    }

    ((uint32_t *) counter.mapped)[0] = 0u;
    vkmesh_vk_buffer assign_buffers[4];
    assign_buffers[0] = out_faces;
    assign_buffers[1] = vertex_map;
    assign_buffers[2] = counter;
    assign_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = keep_count_u32 * 3u;
    groups = (push.n + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ASSIGN_VERTEX_MAP, assign_buffers, &push, groups)) goto cleanup;

    uint32_t vertex_count_u32 = ((const uint32_t *) counter.mapped)[0];

    vkmesh_vk_buffer remap_buffers[4];
    remap_buffers[0] = out_faces;
    remap_buffers[1] = vertex_map;
    remap_buffers[2] = counter;
    remap_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = keep_count_u32 * 3u;
    groups = (push.n + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_REMAP_FACES, remap_buffers, &push, groups)) goto cleanup;

    vkmesh_vk_buffer vertex_buffers[4];
    vertex_buffers[0] = *vertices_buffer;
    vertex_buffers[1] = vertex_map;
    vertex_buffers[2] = out_vertices;
    vertex_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_vertices;
    groups = (push.n + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_VERTICES, vertex_buffers, &push, groups)) goto cleanup;

    int32_t * compacted_faces = (int32_t *) malloc((size_t) keep_count_u32 * 3u * sizeof(int32_t));
    float * compacted_vertices = (float *) malloc((size_t) vertex_count_u32 * 3u * sizeof(float));
    if (compacted_faces == NULL || compacted_vertices == NULL) {
        free(compacted_faces);
        free(compacted_vertices);
        goto cleanup;
    }
    memcpy(compacted_faces, out_faces.mapped, (size_t) keep_count_u32 * 3u * sizeof(int32_t));
    memcpy(compacted_vertices, out_vertices.mapped, (size_t) vertex_count_u32 * 3u * sizeof(float));
    *vertices_out = compacted_vertices;
    *vertex_count_out = (int64_t) vertex_count_u32;
    *faces_out = compacted_faces;
    *face_count_out = (int64_t) keep_count_u32;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &out_faces);
    vk_buffer_destroy(vk, &counter);
    vk_buffer_destroy(vk, &vertex_map);
    vk_buffer_destroy(vk, &out_vertices);
    return ok;
}

static int mesh_remove_faces_by_mask_vulkan(vkmesh_mesh * mesh, const uint8_t * keep_mask, int * removed_faces) {
    if (removed_faces != NULL) *removed_faces = 0;
    if (mesh == NULL || keep_mask == NULL ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer flags_buffer;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&flags_buffer, 0, sizeof(flags_buffer));

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t flags_bytes = (size_t) mesh->n_faces * sizeof(uint32_t);
    int ok = 0;
    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, flags_bytes, &flags_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);
    uint32_t * flags = (uint32_t *) flags_buffer.mapped;
    for (int64_t i = 0; i < mesh->n_faces; ++i) flags[i] = keep_mask[i] ? 1u : 0u;

    float * new_vertices = NULL;
    int64_t new_vertex_count = 0;
    int32_t * new_faces = NULL;
    int64_t new_face_count = 0;
    if (!compact_mesh_from_device_flags(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &flags_buffer,
            0u,
            &new_vertices,
            &new_vertex_count,
            &new_faces,
            &new_face_count)) {
        goto cleanup;
    }

    if (removed_faces != NULL) *removed_faces = (int) (mesh->n_faces - new_face_count);
    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = new_vertices;
    mesh->uvs = NULL;
    mesh->faces = new_faces;
    mesh->n_vertices = new_vertex_count;
    mesh->n_faces = new_face_count;
    mesh->vertex_capacity = new_vertex_count;
    mesh->face_capacity = new_face_count;
    mesh->has_uvs = 0;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &flags_buffer);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_device_remove_duplicate_faces(vkmesh_device_mesh * dm, uint32_t * removed_faces) {
    if (removed_faces != NULL) *removed_faces = 0u;
    if (dm == NULL || dm->vk == NULL || dm->n_faces <= 1u) return 1;
    vkmesh_vk * vk = dm->vk;
    vkmesh_vk_buffer keys;
    vkmesh_vk_buffer flags;
    vkmesh_vk_buffer dummy;
    memset(&keys, 0, sizeof(keys));
    memset(&flags, 0, sizeof(flags));
    memset(&dummy, 0, sizeof(dummy));
    int ok = 0;
    size_t sort_count = (size_t) dm->n_faces;
    if (!vk_buffer_create(vk, sort_count * sizeof(vkmesh_face_key), &keys) ||
        !vk_buffer_create(vk, (size_t) dm->n_faces * sizeof(uint32_t), &flags) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &dummy)) {
        goto cleanup;
    }
    vkmesh_vk_buffer key_buffers[4];
    key_buffers[0] = dm->faces;
    key_buffers[1] = dummy;
    key_buffers[2] = keys;
    key_buffers[3] = flags;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_faces;
    uint32_t groups = (dm->n_faces + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FACE_KEYS, key_buffers, &push, groups)) goto cleanup;
    if (!vkmesh_sort_records_vulkan(vk, &keys, &dummy, sort_count, VKMESH_PIPE_SORT_FACE_KEYS)) goto cleanup;

    vkmesh_vk_buffer mark_buffers[4];
    mark_buffers[0] = keys;
    mark_buffers[1] = dummy;
    mark_buffers[2] = flags;
    mark_buffers[3] = dummy;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) sort_count;
    push.aux0 = dm->n_faces;
    groups = ((uint32_t) sort_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_DUPLICATE_FACES, mark_buffers, &push, groups)) goto cleanup;
    if (!compact_device_mesh_from_flags(dm, &flags, 0u, removed_faces)) goto cleanup;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &keys);
    vk_buffer_destroy(vk, &flags);
    vk_buffer_destroy(vk, &dummy);
    return ok;
}

static int vkmesh_device_remove_degenerate_faces(
    vkmesh_device_mesh * dm,
    float abs_thresh,
    float rel_thresh,
    uint32_t * removed_faces) {
    if (removed_faces != NULL) *removed_faces = 0u;
    if (dm == NULL || dm->vk == NULL || dm->n_faces == 0u) return 0;
    vkmesh_vk * vk = dm->vk;
    vkmesh_vk_buffer flags;
    vkmesh_vk_buffer dummy;
    memset(&flags, 0, sizeof(flags));
    memset(&dummy, 0, sizeof(dummy));
    int ok = 0;
    if (!vk_buffer_create(vk, (size_t) dm->n_faces * sizeof(uint32_t), &flags) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &dummy)) {
        goto cleanup;
    }
    vkmesh_vk_buffer buffers[4];
    buffers[0] = dm->faces;
    buffers[1] = dm->vertices;
    buffers[2] = flags;
    buffers[3] = dummy;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_faces;
    push.eps = abs_thresh;
    push.rel_eps = rel_thresh;
    uint32_t groups = (dm->n_faces + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_DEGENERATE_FACES, buffers, &push, groups)) goto cleanup;
    if (!compact_device_mesh_from_flags(dm, &flags, 0u, removed_faces)) goto cleanup;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &flags);
    vk_buffer_destroy(vk, &dummy);
    return ok;
}

static int vkmesh_log_duplicate_degenerate_device_cluster(
    vkmesh_mesh * mesh,
    int run_degenerate_cleanup,
    float degenerate_abs,
    float degenerate_rel,
    const char * label) {
    if (mesh == NULL || label == NULL ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }
    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_device_mesh dm;
    memset(&dm, 0, sizeof(dm));
    uint32_t duplicate_removed = 0u;
    uint32_t degenerate_removed = 0u;
    int ok = 0;
    if (!vkmesh_device_mesh_upload(vk, mesh, &dm)) goto cleanup;
    if (!vkmesh_device_remove_duplicate_faces(&dm, &duplicate_removed)) goto cleanup;
    fprintf(stderr, "vkmesh: %s remove_duplicate_faces removed=%u faces=%u\n", label, duplicate_removed, dm.n_faces);
    if (run_degenerate_cleanup) {
        if (!vkmesh_device_remove_degenerate_faces(&dm, degenerate_abs, degenerate_rel, &degenerate_removed)) goto cleanup;
        fprintf(stderr, "vkmesh: %s remove_degenerate_faces removed=%u faces=%u\n", label, degenerate_removed, dm.n_faces);
    }
    if (!vkmesh_device_mesh_download(&dm, mesh)) goto cleanup;
    ok = 1;

cleanup:
    vkmesh_device_mesh_destroy(&dm);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_device_remove_small_connected_components(
    vkmesh_device_mesh * dm,
    float min_area,
    uint32_t * removed_faces) {
    if (removed_faces != NULL) *removed_faces = 0u;
    if (dm == NULL || dm->vk == NULL || dm->n_faces == 0u || dm->n_vertices == 0u) return 0;
    vkmesh_vk * vk = dm->vk;
    vkmesh_mesh view;
    memset(&view, 0, sizeof(view));
    view.n_vertices = dm->n_vertices;
    view.n_faces = dm->n_faces;

    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer edge_dummy;
    vkmesh_vk_buffer parents;
    vkmesh_vk_buffer changed;
    vkmesh_vk_buffer area_bits;
    vkmesh_vk_buffer keep_flags;
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&edge_dummy, 0, sizeof(edge_dummy));
    memset(&parents, 0, sizeof(parents));
    memset(&changed, 0, sizeof(changed));
    memset(&area_bits, 0, sizeof(area_bits));
    memset(&keep_flags, 0, sizeof(keep_flags));

    const size_t parents_bytes = (size_t) dm->n_faces * sizeof(uint32_t);
    size_t edge_count = 0;
    size_t edge_sort_count = 0;
    int ok = 0;
    if (!vk_buffer_create(vk, parents_bytes, &parents) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &changed) ||
        !vk_buffer_create(vk, parents_bytes, &area_bits) ||
        !vk_buffer_create(vk, parents_bytes, &keep_flags)) {
        goto cleanup;
    }
    if (!expand_edges_device(vk, &view, &dm->faces, &dm->vertices, &edges_buffer, &edge_dummy, &edge_count, &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = parents;
    init_buffers[1] = changed;
    init_buffers[2] = changed;
    init_buffers[3] = changed;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_faces;
    uint32_t face_groups = (dm->n_faces + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, face_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = edges_buffer;
    union_buffers[1] = parents;
    union_buffers[2] = changed;
    union_buffers[3] = changed;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = parents;
    compress_buffers[1] = changed;
    compress_buffers[2] = changed;
    compress_buffers[3] = changed;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
    int converged = 0;
    for (int iter = 0; iter < 32; ++iter) {
        ((uint32_t *) changed.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = (uint32_t) edge_count;
        push.aux0 = dm->n_faces;
        if (!vkmesh_dispatch_pair(
                vk, VKMESH_PIPE_UNION_FACE_EDGES, union_buffers, &push, edge_groups,
                VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, dm->n_faces, face_groups)) goto cleanup;
        if (((const uint32_t *) changed.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    memset(area_bits.mapped, 0, parents_bytes);
    vkmesh_vk_buffer area_buffers[4];
    area_buffers[0] = dm->faces;
    area_buffers[1] = dm->vertices;
    area_buffers[2] = parents;
    area_buffers[3] = area_bits;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_faces;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPONENT_AREA, area_buffers, &push, face_groups)) goto cleanup;

    vkmesh_vk_buffer mark_buffers[4];
    mark_buffers[0] = parents;
    mark_buffers[1] = area_bits;
    mark_buffers[2] = keep_flags;
    mark_buffers[3] = changed;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_faces;
    push.eps = min_area;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_COMPONENT_KEEP, mark_buffers, &push, face_groups)) goto cleanup;
    if (!compact_device_mesh_from_flags(dm, &keep_flags, 0u, removed_faces)) goto cleanup;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &edge_dummy);
    vk_buffer_destroy(vk, &parents);
    vk_buffer_destroy(vk, &changed);
    vk_buffer_destroy(vk, &area_bits);
    vk_buffer_destroy(vk, &keep_flags);
    return ok;
}

static int vkmesh_device_repair_non_manifold_edges(
    vkmesh_device_mesh * dm,
    uint32_t * old_vertices,
    uint32_t * new_vertices) {
    if (old_vertices != NULL) *old_vertices = dm != NULL ? dm->n_vertices : 0u;
    if (new_vertices != NULL) *new_vertices = dm != NULL ? dm->n_vertices : 0u;
    if (dm == NULL || dm->vk == NULL || dm->n_faces == 0u || dm->n_vertices == 0u ||
        dm->n_faces > UINT32_MAX / 3u) {
        return 0;
    }

    vkmesh_vk * vk = dm->vk;
    vkmesh_mesh view;
    memset(&view, 0, sizeof(view));
    view.n_vertices = dm->n_vertices;
    view.n_faces = dm->n_faces;

    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer parents;
    vkmesh_vk_buffer changed;
    vkmesh_vk_buffer repair_data;
    vkmesh_vk_buffer out_vertices;
    vkmesh_vk_buffer out_faces;
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&parents, 0, sizeof(parents));
    memset(&changed, 0, sizeof(changed));
    memset(&repair_data, 0, sizeof(repair_data));
    memset(&out_vertices, 0, sizeof(out_vertices));
    memset(&out_faces, 0, sizeof(out_faces));

    const uint32_t face_count = dm->n_faces;
    const uint32_t corner_count = face_count * 3u;
    const uint32_t root_map_words = corner_count;
    const uint32_t counter_word = root_map_words;
    const uint32_t out_vertices_word = counter_word + 1u;
    const uint32_t out_faces_word = out_vertices_word + corner_count * 3u;
    const size_t repair_words = (size_t) out_faces_word + (size_t) corner_count;
    if (repair_words > UINT32_MAX) return 0;

    const size_t parents_bytes = (size_t) corner_count * sizeof(uint32_t);
    const size_t repair_bytes = repair_words * sizeof(uint32_t);
    size_t edge_count = 0;
    size_t edge_sort_count = 0;
    int ok = 0;

    if (!vk_buffer_create(vk, parents_bytes, &parents) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &changed) ||
        !vk_buffer_create(vk, repair_bytes, &repair_data)) {
        goto cleanup;
    }

    if (!expand_edges_device(
            vk,
            &view,
            &dm->faces,
            &dm->vertices,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = parents;
    init_buffers[1] = changed;
    init_buffers[2] = changed;
    init_buffers[3] = changed;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = corner_count;
    uint32_t corner_groups = (corner_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, corner_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = edges_buffer;
    union_buffers[1] = dm->faces;
    union_buffers[2] = parents;
    union_buffers[3] = changed;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = parents;
    compress_buffers[1] = changed;
    compress_buffers[2] = changed;
    compress_buffers[3] = changed;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
    int converged = 0;
    for (int iter = 0; iter < 128; ++iter) {
        ((uint32_t *) changed.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = (uint32_t) edge_count;
        push.aux0 = face_count;
        if (!vkmesh_dispatch_pair(
                vk, VKMESH_PIPE_UNION_CORNER_EDGES, union_buffers, &push, edge_groups,
                VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, corner_count, corner_groups)) goto cleanup;
        if (((const uint32_t *) changed.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    uint32_t * repair_words_u32 = (uint32_t *) repair_data.mapped;
    memset(repair_words_u32, 0xff, (size_t) root_map_words * sizeof(uint32_t));
    repair_words_u32[counter_word] = 0u;

    vkmesh_vk_buffer assign_buffers[4];
    assign_buffers[0] = parents;
    assign_buffers[1] = repair_data;
    assign_buffers[2] = changed;
    assign_buffers[3] = changed;
    memset(&push, 0, sizeof(push));
    push.n = corner_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ASSIGN_CORNER_VERTICES, assign_buffers, &push, corner_groups)) goto cleanup;

    uint32_t vertex_count = repair_words_u32[counter_word];
    if (vertex_count == 0u || vertex_count > corner_count) goto cleanup;
    if (!vk_buffer_create(vk, (size_t) vertex_count * 3u * sizeof(float), &out_vertices) ||
        !vk_buffer_create(vk, (size_t) face_count * 3u * sizeof(int32_t), &out_faces)) {
        goto cleanup;
    }

    vkmesh_vk_buffer write_buffers[4];
    write_buffers[0] = dm->faces;
    write_buffers[1] = dm->vertices;
    write_buffers[2] = parents;
    write_buffers[3] = repair_data;
    memset(&push, 0, sizeof(push));
    push.n = corner_count;
    push.aux1 = out_vertices_word;
    push.aux2 = out_faces_word;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_WRITE_REPAIRED_MESH, write_buffers, &push, corner_groups)) goto cleanup;

    vkmesh_vk_buffer copy_buffers[4];
    copy_buffers[0] = repair_data;
    copy_buffers[1] = out_vertices;
    copy_buffers[2] = changed;
    copy_buffers[3] = changed;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count * 3u;
    push.aux0 = out_vertices_word;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    copy_buffers[1] = out_faces;
    memset(&push, 0, sizeof(push));
    push.n = face_count * 3u;
    push.aux0 = out_faces_word;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    vk_buffer_destroy(vk, &dm->vertices);
    vk_buffer_destroy(vk, &dm->faces);
    dm->vertices = out_vertices;
    dm->faces = out_faces;
    memset(&out_vertices, 0, sizeof(out_vertices));
    memset(&out_faces, 0, sizeof(out_faces));
    dm->n_vertices = vertex_count;
    dm->n_faces = face_count;
    if (new_vertices != NULL) *new_vertices = vertex_count;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &parents);
    vk_buffer_destroy(vk, &changed);
    vk_buffer_destroy(vk, &repair_data);
    vk_buffer_destroy(vk, &out_vertices);
    vk_buffer_destroy(vk, &out_faces);
    return ok;
}

static int vkmesh_device_unify_face_orientations(vkmesh_device_mesh * dm, uint32_t * flipped_faces) {
    if (flipped_faces != NULL) *flipped_faces = 0u;
    if (dm == NULL || dm->vk == NULL || dm->n_faces == 0u || dm->n_vertices == 0u ||
        dm->n_faces > UINT32_MAX / 3u || dm->n_faces > UINT32_MAX / 2u) {
        return 0;
    }

    vkmesh_vk * vk = dm->vk;
    vkmesh_mesh view;
    memset(&view, 0, sizeof(view));
    view.n_vertices = dm->n_vertices;
    view.n_faces = dm->n_faces;

    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer state;
    vkmesh_vk_buffer changed;
    vkmesh_vk_buffer counter;
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&state, 0, sizeof(state));
    memset(&changed, 0, sizeof(changed));
    memset(&counter, 0, sizeof(counter));

    const uint32_t face_count = dm->n_faces;
    const size_t state_bytes = (size_t) face_count * sizeof(uint32_t);
    size_t edge_count = 0;
    size_t edge_sort_count = 0;
    int ok = 0;

    if (!vk_buffer_create(vk, state_bytes, &state) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &changed) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter)) {
        goto cleanup;
    }

    if (!expand_edges_device(
            vk,
            &view,
            &dm->faces,
            &dm->vertices,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    uint32_t face_groups = (face_count + 127u) / 128u;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = state;
    init_buffers[1] = changed;
    init_buffers[2] = changed;
    init_buffers[3] = changed;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_ORIENTATION_STATE, init_buffers, &push, face_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = edges_buffer;
    union_buffers[1] = dm->faces;
    union_buffers[2] = state;
    union_buffers[3] = changed;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = state;
    compress_buffers[1] = changed;
    compress_buffers[2] = changed;
    compress_buffers[3] = changed;
    int converged = 0;
    for (int iter = 0; iter < 512; ++iter) {
        ((uint32_t *) changed.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = (uint32_t) edge_count;
        push.aux0 = face_count;
        if (!vkmesh_dispatch_pair(
                vk, VKMESH_PIPE_UNION_ORIENTATION_EDGES, union_buffers, &push, edge_groups,
                VKMESH_PIPE_COMPRESS_ORIENTATION_STATE, compress_buffers, face_count, face_groups)) goto cleanup;

        if (((const uint32_t *) changed.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    ((uint32_t *) counter.mapped)[0] = 0u;
    vkmesh_vk_buffer apply_buffers[4];
    apply_buffers[0] = dm->faces;
    apply_buffers[1] = state;
    apply_buffers[2] = counter;
    apply_buffers[3] = changed;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_APPLY_ORIENTATION_FLIPS, apply_buffers, &push, face_groups)) goto cleanup;

    if (flipped_faces != NULL) *flipped_faces = ((const uint32_t *) counter.mapped)[0];
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &state);
    vk_buffer_destroy(vk, &changed);
    vk_buffer_destroy(vk, &counter);
    return ok;
}

static int compute_duplicate_compacted_mesh_vulkan(
    const vkmesh_mesh * mesh,
    float ** vertices_out,
    int64_t * vertex_count_out,
    int32_t ** faces_out,
    int64_t * face_count_out) {
    *vertices_out = NULL;
    *vertex_count_out = 0;
    *faces_out = NULL;
    *face_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX) return 0;
    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }
    vkmesh_vk_buffer buffers[4];
    memset(buffers, 0, sizeof(buffers));
    int ok = 0;
    size_t sort_count = (size_t) mesh->n_faces;
    size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    size_t keys_bytes = sort_count * sizeof(vkmesh_face_key);
    size_t flags_bytes = (size_t) mesh->n_faces * sizeof(uint32_t);
    if (!vk_buffer_create(vk, faces_bytes, &buffers[0]) ||
        !vk_buffer_create(vk, vertices_bytes, &buffers[1]) ||
        !vk_buffer_create(vk, keys_bytes, &buffers[2]) ||
        !vk_buffer_create(vk, flags_bytes, &buffers[3])) {
        goto cleanup;
    }
    memcpy(buffers[0].mapped, mesh->faces, faces_bytes);
    memcpy(buffers[1].mapped, mesh->vertices, vertices_bytes);
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_faces;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FACE_KEYS, buffers, &push, groups)) goto cleanup;
    if (!vkmesh_sort_records_vulkan(vk, &buffers[2], &buffers[1], sort_count, VKMESH_PIPE_SORT_FACE_KEYS)) goto cleanup;

    vkmesh_vk_buffer mark_buffers[4];
    mark_buffers[0] = buffers[2];
    mark_buffers[1] = buffers[1];
    mark_buffers[2] = buffers[3];
    mark_buffers[3] = buffers[1];
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) sort_count;
    push.aux0 = (uint32_t) mesh->n_faces;
    groups = (uint32_t) ((sort_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_DUPLICATE_FACES, mark_buffers, &push, groups)) goto cleanup;

    if (!compact_mesh_from_device_flags(
            vk, mesh, &buffers[0], &buffers[1], &buffers[3], 0u,
            vertices_out, vertex_count_out, faces_out, face_count_out)) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    for (uint32_t i = 0; i < 4; ++i) vk_buffer_destroy(vk, &buffers[i]);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_remove_duplicate_faces(vkmesh_mesh * mesh, int * removed_faces) {
    *removed_faces = 0;
    if (mesh->n_faces <= 1) return 1;
    float * new_vertices = NULL;
    int32_t * new_faces = NULL;
    int64_t new_vertex_count = 0;
    int64_t keep_count = 0;
    if (!compute_duplicate_compacted_mesh_vulkan(mesh, &new_vertices, &new_vertex_count, &new_faces, &keep_count)) return 0;
    *removed_faces = (int) (mesh->n_faces - keep_count);
    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = new_vertices;
    mesh->uvs = NULL;
    mesh->faces = new_faces;
    mesh->n_vertices = new_vertex_count;
    mesh->n_faces = keep_count;
    mesh->vertex_capacity = new_vertex_count;
    mesh->face_capacity = keep_count;
    mesh->has_uvs = 0;
    return 1;
}

static int compute_degenerate_compacted_mesh_vulkan(
    const vkmesh_mesh * mesh,
    float abs_thresh,
    float rel_thresh,
    float ** vertices_out,
    int64_t * vertex_count_out,
    int32_t ** faces_out,
    int64_t * face_count_out) {
    *vertices_out = NULL;
    *vertex_count_out = 0;
    *faces_out = NULL;
    *face_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX) return 0;

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer buffers[4];
    memset(buffers, 0, sizeof(buffers));
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t flags_bytes = (size_t) mesh->n_faces * sizeof(uint32_t);
    int ok = 0;
    if (!vk_buffer_create(vk, faces_bytes, &buffers[0]) ||
        !vk_buffer_create(vk, vertices_bytes, &buffers[1]) ||
        !vk_buffer_create(vk, flags_bytes, &buffers[2]) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &buffers[3])) {
        goto cleanup;
    }
    memcpy(buffers[0].mapped, mesh->faces, faces_bytes);
    memcpy(buffers[1].mapped, mesh->vertices, vertices_bytes);

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_faces;
    push.eps = abs_thresh;
    push.rel_eps = rel_thresh;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_DEGENERATE_FACES, buffers, &push, groups)) goto cleanup;

    if (!compact_mesh_from_device_flags(
            vk, mesh, &buffers[0], &buffers[1], &buffers[2], 0u,
            vertices_out, vertex_count_out, faces_out, face_count_out)) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    for (uint32_t i = 0; i < 4; ++i) vk_buffer_destroy(vk, &buffers[i]);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_remove_degenerate_faces(vkmesh_mesh * mesh, float abs_thresh, float rel_thresh, int * removed_faces) {
    *removed_faces = 0;
    float * new_vertices = NULL;
    int32_t * new_faces = NULL;
    int64_t new_vertex_count = 0;
    int64_t keep_count = 0;
    if (!compute_degenerate_compacted_mesh_vulkan(mesh, abs_thresh, rel_thresh, &new_vertices, &new_vertex_count, &new_faces, &keep_count)) {
        fprintf(stderr, "vkmesh: Vulkan degenerate-face pass failed\n");
        return 0;
    }
    *removed_faces = (int) (mesh->n_faces - keep_count);
    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = new_vertices;
    mesh->uvs = NULL;
    mesh->faces = new_faces;
    mesh->n_vertices = new_vertex_count;
    mesh->n_faces = keep_count;
    mesh->vertex_capacity = new_vertex_count;
    mesh->face_capacity = keep_count;
    mesh->has_uvs = 0;
    return 1;
}

static int get_sorted_edges(const vkmesh_mesh * mesh, vkmesh_edge ** edges_out, int64_t * edge_count_out) {
    if (!expand_edges_vulkan(mesh, edges_out, edge_count_out)) return 0;
    return 1;
}

static int mark_boundary_edges_vulkan(const vkmesh_edge * edges, int64_t edge_count, uint32_t ** flags_out) {
    *flags_out = NULL;
    if (edge_count <= 0 || edge_count > UINT32_MAX) return 0;
    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }
    vkmesh_vk_buffer buffers[4];
    memset(buffers, 0, sizeof(buffers));
    const size_t edges_bytes = (size_t) edge_count * sizeof(vkmesh_edge);
    const size_t flags_bytes = (size_t) edge_count * sizeof(uint32_t);
    int ok = 0;
    if (!vk_buffer_create(vk, edges_bytes, &buffers[0]) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &buffers[1]) ||
        !vk_buffer_create(vk, flags_bytes, &buffers[2]) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &buffers[3])) {
        goto cleanup;
    }
    memcpy(buffers[0].mapped, edges, edges_bytes);
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) edge_count;
    uint32_t groups = (uint32_t) ((edge_count + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_BOUNDARY_EDGES, buffers, &push, groups)) goto cleanup;
    uint32_t * flags = (uint32_t *) malloc(flags_bytes);
    if (flags == NULL) goto cleanup;
    memcpy(flags, buffers[2].mapped, flags_bytes);
    *flags_out = flags;
    ok = 1;

cleanup:
    for (uint32_t i = 0; i < 4; ++i) vk_buffer_destroy(vk, &buffers[i]);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int get_boundary_edges_vulkan(
    const vkmesh_mesh * mesh,
    vkmesh_boundary_edge ** boundary_out,
    int64_t * boundary_count_out) {
    *boundary_out = NULL;
    *boundary_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer boundary_buffer;
    vkmesh_vk_buffer counter_buffer;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&boundary_buffer, 0, sizeof(boundary_buffer));
    memset(&counter_buffer, 0, sizeof(counter_buffer));

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t max_boundary_count = (size_t) mesh->n_faces * 3u;
    int ok = 0;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, max_boundary_count * sizeof(vkmesh_boundary_edge), &boundary_buffer) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer buffers[4];
    buffers[0] = edges_buffer;
    buffers[1] = boundary_buffer;
    buffers[2] = counter_buffer;
    buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) edge_count;
    uint32_t groups = (uint32_t) ((edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_BOUNDARY_EDGES, buffers, &push, groups)) goto cleanup;

    uint32_t count_u32 = ((const uint32_t *) counter_buffer.mapped)[0];
    if ((size_t) count_u32 > max_boundary_count) goto cleanup;
    if (count_u32 > 0u) {
        vkmesh_boundary_edge * boundary = (vkmesh_boundary_edge *) malloc((size_t) count_u32 * sizeof(*boundary));
        if (boundary == NULL) goto cleanup;
        memcpy(boundary, boundary_buffer.mapped, (size_t) count_u32 * sizeof(*boundary));
        *boundary_out = boundary;
    }
    *boundary_count_out = (int64_t) count_u32;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &boundary_buffer);
    vk_buffer_destroy(vk, &counter_buffer);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int get_boundary_edges_fallback(
    const vkmesh_mesh * mesh,
    vkmesh_boundary_edge ** boundary_out,
    int64_t * boundary_count_out) {
    *boundary_out = NULL;
    *boundary_count_out = 0;
    vkmesh_edge * edges = NULL;
    int64_t edge_count = 0;
    if (!expand_edges_vulkan(mesh, &edges, &edge_count)) {
        fprintf(stderr, "vkmesh: Vulkan edge expansion failed\n");
        return 0;
    }
    uint32_t * boundary_flags = NULL;
    if (!mark_boundary_edges_vulkan(edges, edge_count, &boundary_flags)) {
        fprintf(stderr, "vkmesh: Vulkan boundary-edge marking failed\n");
        free(edges);
        return 0;
    }

    vkmesh_boundary_edge * boundary = NULL;
    int64_t boundary_count = 0;
    int64_t boundary_capacity = 0;
    for (int64_t i = 0; i < edge_count; ++i) {
        if (!boundary_flags[i]) continue;
        if (boundary_count == boundary_capacity) {
            int64_t next_cap = boundary_capacity > 0 ? boundary_capacity * 2 : 1024;
            vkmesh_boundary_edge * next =
                (vkmesh_boundary_edge *) realloc(boundary, (size_t) next_cap * sizeof(*boundary));
            if (next == NULL) {
                free(edges);
                free(boundary_flags);
                free(boundary);
                return 0;
            }
            boundary = next;
            boundary_capacity = next_cap;
        }
        boundary[boundary_count].a = (int32_t) edges[i].a;
        boundary[boundary_count].b = (int32_t) edges[i].b;
        boundary[boundary_count].min_v = edges[i].min_v;
        boundary[boundary_count].max_v = edges[i].max_v;
        ++boundary_count;
    }

    free(edges);
    free(boundary_flags);
    *boundary_out = boundary;
    *boundary_count_out = boundary_count;
    return 1;
}

static int get_boundary_edges(
    const vkmesh_mesh * mesh,
    vkmesh_boundary_edge ** boundary_out,
    int64_t * boundary_count_out) {
    if (get_boundary_edges_vulkan(mesh, boundary_out, boundary_count_out)) return 1;
    return get_boundary_edges_fallback(mesh, boundary_out, boundary_count_out);
}

static int get_manifold_face_pairs_vulkan(
    const vkmesh_mesh * mesh,
    vkmesh_face_pair ** pairs_out,
    int64_t * pair_count_out) {
    *pairs_out = NULL;
    *pair_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer pairs_buffer;
    vkmesh_vk_buffer counter_buffer;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&pairs_buffer, 0, sizeof(pairs_buffer));
    memset(&counter_buffer, 0, sizeof(counter_buffer));

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t max_pair_count = ((size_t) mesh->n_faces * 3u) / 2u;
    int ok = 0;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, max_pair_count * sizeof(vkmesh_face_pair), &pairs_buffer) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer buffers[4];
    buffers[0] = edges_buffer;
    buffers[1] = pairs_buffer;
    buffers[2] = counter_buffer;
    buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) edge_count;
    push.aux0 = (uint32_t) mesh->n_faces;
    uint32_t groups = (uint32_t) ((edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_FACE_PAIRS, buffers, &push, groups)) goto cleanup;

    uint32_t count_u32 = ((const uint32_t *) counter_buffer.mapped)[0];
    if ((size_t) count_u32 > max_pair_count) goto cleanup;
    if (count_u32 > 0u) {
        vkmesh_face_pair * pairs = (vkmesh_face_pair *) malloc((size_t) count_u32 * sizeof(*pairs));
        if (pairs == NULL) goto cleanup;
        memcpy(pairs, pairs_buffer.mapped, (size_t) count_u32 * sizeof(*pairs));
        *pairs_out = pairs;
    }
    *pair_count_out = (int64_t) count_u32;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &pairs_buffer);
    vk_buffer_destroy(vk, &counter_buffer);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int get_manifold_face_pairs_cpu(
    const vkmesh_mesh * mesh,
    vkmesh_face_pair ** pairs_out,
    int64_t * pair_count_out) {
    *pairs_out = NULL;
    *pair_count_out = 0;
    vkmesh_edge * edges = NULL;
    int64_t edge_count = 0;
    if (!get_sorted_edges(mesh, &edges, &edge_count)) return 0;
    vkmesh_face_pair * pairs = NULL;
    int64_t count = 0;
    int64_t capacity = 0;
    for (int64_t i = 0; i < edge_count;) {
        int64_t j = i + 1;
        while (j < edge_count && edges[j].min_v == edges[i].min_v && edges[j].max_v == edges[i].max_v) ++j;
        if (j - i == 2) {
            if (count == capacity) {
                int64_t next_cap = capacity > 0 ? capacity * 2 : 1024;
                vkmesh_face_pair * next = (vkmesh_face_pair *) realloc(pairs, (size_t) next_cap * sizeof(*pairs));
                if (next == NULL) {
                    free(edges);
                    free(pairs);
                    return 0;
                }
                pairs = next;
                capacity = next_cap;
            }
            pairs[count].f0 = (int32_t) edges[i].face;
            pairs[count].f1 = (int32_t) edges[i + 1].face;
            pairs[count].v0 = edges[i].min_v;
            pairs[count].v1 = edges[i].max_v;
            ++count;
        }
        i = j;
    }
    free(edges);
    *pairs_out = pairs;
    *pair_count_out = count;
    return 1;
}

static int get_manifold_face_pairs(
    const vkmesh_mesh * mesh,
    vkmesh_face_pair ** pairs_out,
    int64_t * pair_count_out) {
    if (get_manifold_face_pairs_vulkan(mesh, pairs_out, pair_count_out)) return 1;
    return get_manifold_face_pairs_cpu(mesh, pairs_out, pair_count_out);
}

static int face_local_index(const vkmesh_mesh * mesh, int32_t face_id, int32_t vertex_id) {
    const int32_t * f = mesh->faces + (size_t) face_id * 3u;
    if (f[0] == vertex_id) return 0;
    if (f[1] == vertex_id) return 1;
    if (f[2] == vertex_id) return 2;
    return -1;
}

static int vkmesh_repair_non_manifold_edges_vulkan(vkmesh_mesh * mesh, int * old_vertices, int * new_vertices) {
    *old_vertices = (int) mesh->n_vertices;
    *new_vertices = (int) mesh->n_vertices;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer parents;
    vkmesh_vk_buffer changed;
    vkmesh_vk_buffer repair_data;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&parents, 0, sizeof(parents));
    memset(&changed, 0, sizeof(changed));
    memset(&repair_data, 0, sizeof(repair_data));

    const uint32_t face_count = (uint32_t) mesh->n_faces;
    const uint32_t corner_count = face_count * 3u;
    const uint32_t root_map_words = corner_count;
    const uint32_t counter_word = root_map_words;
    const uint32_t out_vertices_word = counter_word + 1u;
    const uint32_t out_faces_word = out_vertices_word + corner_count * 3u;
    const size_t repair_words = (size_t) out_faces_word + (size_t) corner_count;
    if (repair_words > UINT32_MAX) {
        if (owns_vk) vkmesh_vk_destroy(vk);
        return 0;
    }

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t parents_bytes = (size_t) corner_count * sizeof(uint32_t);
    const size_t repair_bytes = repair_words * sizeof(uint32_t);
    int ok = 0;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, parents_bytes, &parents) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &changed) ||
        !vk_buffer_create(vk, repair_bytes, &repair_data)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = parents;
    init_buffers[1] = changed;
    init_buffers[2] = changed;
    init_buffers[3] = changed;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = corner_count;
    uint32_t corner_groups = (corner_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, corner_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = edges_buffer;
    union_buffers[1] = faces_buffer;
    union_buffers[2] = parents;
    union_buffers[3] = changed;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = parents;
    compress_buffers[1] = changed;
    compress_buffers[2] = changed;
    compress_buffers[3] = changed;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
    int converged = 0;
    for (int iter = 0; iter < 128; ++iter) {
        ((uint32_t *) changed.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = (uint32_t) edge_count;
        push.aux0 = face_count;
        if (!vkmesh_dispatch_pair(
                vk, VKMESH_PIPE_UNION_CORNER_EDGES, union_buffers, &push, edge_groups,
                VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, corner_count, corner_groups)) goto cleanup;
        if (((const uint32_t *) changed.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    uint32_t * repair_words_u32 = (uint32_t *) repair_data.mapped;
    memset(repair_words_u32, 0xff, (size_t) root_map_words * sizeof(uint32_t));
    repair_words_u32[counter_word] = 0u;

    vkmesh_vk_buffer assign_buffers[4];
    assign_buffers[0] = parents;
    assign_buffers[1] = repair_data;
    assign_buffers[2] = changed;
    assign_buffers[3] = changed;
    memset(&push, 0, sizeof(push));
    push.n = corner_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ASSIGN_CORNER_VERTICES, assign_buffers, &push, corner_groups)) goto cleanup;

    uint32_t vertex_count = repair_words_u32[counter_word];
    if (vertex_count == 0u || vertex_count > corner_count) goto cleanup;

    vkmesh_vk_buffer write_buffers[4];
    write_buffers[0] = faces_buffer;
    write_buffers[1] = vertices_buffer;
    write_buffers[2] = parents;
    write_buffers[3] = repair_data;
    memset(&push, 0, sizeof(push));
    push.n = corner_count;
    push.aux1 = out_vertices_word;
    push.aux2 = out_faces_word;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_WRITE_REPAIRED_MESH, write_buffers, &push, corner_groups)) goto cleanup;

    float * out_vertices = (float *) malloc((size_t) vertex_count * 3u * sizeof(float));
    int32_t * out_faces = (int32_t *) malloc((size_t) face_count * 3u * sizeof(int32_t));
    if (out_vertices == NULL || out_faces == NULL) {
        free(out_vertices);
        free(out_faces);
        goto cleanup;
    }
    memcpy(out_vertices, repair_words_u32 + out_vertices_word, (size_t) vertex_count * 3u * sizeof(float));
    memcpy(out_faces, repair_words_u32 + out_faces_word, (size_t) face_count * 3u * sizeof(int32_t));

    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = out_vertices;
    mesh->uvs = NULL;
    mesh->faces = out_faces;
    mesh->n_vertices = (int64_t) vertex_count;
    mesh->n_faces = (int64_t) face_count;
    mesh->vertex_capacity = mesh->n_vertices;
    mesh->face_capacity = mesh->n_faces;
    mesh->has_uvs = 0;
    *new_vertices = (int) mesh->n_vertices;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &parents);
    vk_buffer_destroy(vk, &changed);
    vk_buffer_destroy(vk, &repair_data);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_repair_non_manifold_edges_cpu(vkmesh_mesh * mesh, int * old_vertices, int * new_vertices) {
    *old_vertices = (int) mesh->n_vertices;
    *new_vertices = (int) mesh->n_vertices;
    if (mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX / 3) return 0;
    vkmesh_face_pair * pairs = NULL;
    int64_t pair_count = 0;
    if (!get_manifold_face_pairs(mesh, &pairs, &pair_count)) return 0;

    int64_t corner_count = mesh->n_faces * 3;
    int * parent = (int *) malloc((size_t) corner_count * sizeof(int));
    int32_t * root_to_vertex = (int32_t *) malloc((size_t) corner_count * sizeof(int32_t));
    int32_t * corner_to_vertex = (int32_t *) malloc((size_t) corner_count * sizeof(int32_t));
    if (parent == NULL || root_to_vertex == NULL || corner_to_vertex == NULL) {
        free(pairs);
        free(parent);
        free(root_to_vertex);
        free(corner_to_vertex);
        return 0;
    }
    for (int64_t i = 0; i < corner_count; ++i) {
        parent[i] = (int) i;
        root_to_vertex[i] = -1;
    }
    for (int64_t i = 0; i < pair_count; ++i) {
        int a0 = face_local_index(mesh, pairs[i].f0, (int32_t) pairs[i].v0);
        int a1 = face_local_index(mesh, pairs[i].f0, (int32_t) pairs[i].v1);
        int b0 = face_local_index(mesh, pairs[i].f1, (int32_t) pairs[i].v0);
        int b1 = face_local_index(mesh, pairs[i].f1, (int32_t) pairs[i].v1);
        if (a0 >= 0 && b0 >= 0) uf_union(parent, pairs[i].f0 * 3 + a0, pairs[i].f1 * 3 + b0);
        if (a1 >= 0 && b1 >= 0) uf_union(parent, pairs[i].f0 * 3 + a1, pairs[i].f1 * 3 + b1);
    }

    int64_t out_v_count = 0;
    for (int64_t corner = 0; corner < corner_count; ++corner) {
        int root = uf_find(parent, (int) corner);
        if (root_to_vertex[root] < 0) root_to_vertex[root] = (int32_t) out_v_count++;
        corner_to_vertex[corner] = root_to_vertex[root];
    }

    float * out_vertices = (float *) malloc((size_t) out_v_count * 3u * sizeof(float));
    float * out_uvs = NULL;
    if (out_vertices == NULL) {
        free(pairs); free(parent); free(root_to_vertex); free(corner_to_vertex);
        return 0;
    }
    if (mesh->has_uvs && mesh->uvs != NULL) {
        out_uvs = (float *) malloc((size_t) out_v_count * 2u * sizeof(float));
        if (out_uvs == NULL) {
            free(out_vertices);
            free(pairs); free(parent); free(root_to_vertex); free(corner_to_vertex);
            return 0;
        }
    }
    memset(root_to_vertex, 0xff, (size_t) corner_count * sizeof(int32_t));
    for (int64_t corner = 0; corner < corner_count; ++corner) {
        int root = uf_find(parent, (int) corner);
        int32_t dst = corner_to_vertex[corner];
        if (root_to_vertex[root] >= 0) continue;
        int64_t face_id = corner / 3;
        int local = (int) (corner % 3);
        int32_t src_v = mesh->faces[(size_t) face_id * 3u + (size_t) local];
        memcpy(out_vertices + (size_t) dst * 3u, mesh->vertices + (size_t) src_v * 3u, 3u * sizeof(float));
        if (out_uvs != NULL) {
            memcpy(out_uvs + (size_t) dst * 2u, mesh->uvs + (size_t) src_v * 2u, 2u * sizeof(float));
        }
        root_to_vertex[root] = dst;
    }
    for (int64_t face_id = 0; face_id < mesh->n_faces; ++face_id) {
        int32_t * f = mesh->faces + (size_t) face_id * 3u;
        f[0] = corner_to_vertex[face_id * 3 + 0];
        f[1] = corner_to_vertex[face_id * 3 + 1];
        f[2] = corner_to_vertex[face_id * 3 + 2];
    }

    free(mesh->vertices);
    free(mesh->uvs);
    mesh->vertices = out_vertices;
    mesh->uvs = out_uvs;
    mesh->n_vertices = out_v_count;
    mesh->vertex_capacity = out_v_count;
    if (out_uvs == NULL) mesh->has_uvs = 0;
    *new_vertices = (int) mesh->n_vertices;
    free(pairs);
    free(parent);
    free(root_to_vertex);
    free(corner_to_vertex);
    return 1;
}

static int vkmesh_repair_non_manifold_edges(vkmesh_mesh * mesh, int * old_vertices, int * new_vertices) {
    if (vkmesh_repair_non_manifold_edges_vulkan(mesh, old_vertices, new_vertices)) return 1;
    return vkmesh_repair_non_manifold_edges_cpu(mesh, old_vertices, new_vertices);
}

static int vkmesh_remove_small_connected_components(vkmesh_mesh * mesh, float min_area, int * removed_faces) {
    *removed_faces = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX) return 0;
    if (mesh->n_vertices > 0 && mesh->n_vertices <= INT32_MAX) {
        vkmesh_vk local_vk;
        vkmesh_vk * vk = NULL;
        int owns_vk = 0;
        if (vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
            vkmesh_device_mesh dm;
            memset(&dm, 0, sizeof(dm));
            uint32_t removed_u32 = 0u;
            int ok = 0;
            if (vkmesh_device_mesh_upload(vk, mesh, &dm) &&
                vkmesh_device_remove_small_connected_components(&dm, min_area, &removed_u32) &&
                vkmesh_device_mesh_download(&dm, mesh)) {
                *removed_faces = (int) removed_u32;
                ok = 1;
            }
            vkmesh_device_mesh_destroy(&dm);
            if (owns_vk) vkmesh_vk_destroy(vk);
            if (ok) return 1;
            fprintf(stderr, "vkmesh: device connected-components pass failed, falling back\n");
        }
    }
    if (mesh->n_vertices > 0 && mesh->n_vertices <= INT32_MAX) {
        vkmesh_vk local_vk;
        vkmesh_vk * vk = NULL;
        int owns_vk = 0;
        if (vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
            vkmesh_vk_buffer faces_buffer;
            vkmesh_vk_buffer vertices_buffer;
            vkmesh_vk_buffer edges_buffer;
            vkmesh_vk_buffer edge_dummy;
            vkmesh_vk_buffer parents;
            vkmesh_vk_buffer changed;
            vkmesh_vk_buffer area_bits;
            vkmesh_vk_buffer keep_flags;
            memset(&faces_buffer, 0, sizeof(faces_buffer));
            memset(&vertices_buffer, 0, sizeof(vertices_buffer));
            memset(&edges_buffer, 0, sizeof(edges_buffer));
            memset(&edge_dummy, 0, sizeof(edge_dummy));
            memset(&parents, 0, sizeof(parents));
            memset(&changed, 0, sizeof(changed));
            memset(&area_bits, 0, sizeof(area_bits));
            memset(&keep_flags, 0, sizeof(keep_flags));

            const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
            const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
            const size_t parents_bytes = (size_t) mesh->n_faces * sizeof(uint32_t);
            const size_t flags_bytes = (size_t) mesh->n_faces * sizeof(uint32_t);
            size_t edge_count = 0;
            size_t edge_sort_count = 0;
            int ok = 0;
            if (vk_buffer_create(vk, faces_bytes, &faces_buffer) &&
                vk_buffer_create(vk, vertices_bytes, &vertices_buffer) &&
                vk_buffer_create(vk, parents_bytes, &parents) &&
                vk_buffer_create(vk, sizeof(uint32_t), &changed) &&
                vk_buffer_create(vk, parents_bytes, &area_bits) &&
                vk_buffer_create(vk, flags_bytes, &keep_flags)) {
                memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
                memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);
                if (expand_edges_device(vk, mesh, &faces_buffer, &vertices_buffer, &edges_buffer, &edge_dummy, &edge_count, &edge_sort_count)) {
                    vkmesh_vk_buffer init_buffers[4];
                    init_buffers[0] = parents;
                    init_buffers[1] = changed;
                    init_buffers[2] = changed;
                    init_buffers[3] = changed;
                    vkmesh_push push;
                    memset(&push, 0, sizeof(push));
                    push.n = (uint32_t) mesh->n_faces;
                    uint32_t groups = (push.n + 127u) / 128u;
                    if (vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, groups)) {
                        vkmesh_vk_buffer union_buffers[4];
                        union_buffers[0] = edges_buffer;
                        union_buffers[1] = parents;
                        union_buffers[2] = changed;
                        union_buffers[3] = changed;
                        vkmesh_vk_buffer compress_buffers[4];
                        compress_buffers[0] = parents;
                        compress_buffers[1] = changed;
                        compress_buffers[2] = changed;
                        compress_buffers[3] = changed;
                        uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
                        uint32_t face_groups = (uint32_t) ((mesh->n_faces + 127) / 128);
                        int converged = 0;
                        for (int iter = 0; iter < 32; ++iter) {
                            ((uint32_t *) changed.mapped)[0] = 0u;
                            memset(&push, 0, sizeof(push));
                            push.n = (uint32_t) edge_count;
                            push.aux0 = (uint32_t) mesh->n_faces;
                            if (!vkmesh_dispatch_pair(
                                    vk, VKMESH_PIPE_UNION_FACE_EDGES, union_buffers, &push, edge_groups,
                                    VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers,
                                    (uint32_t) mesh->n_faces, face_groups)) break;
                            if (((const uint32_t *) changed.mapped)[0] == 0u) {
                                converged = 1;
                                break;
                            }
                        }
                        if (converged) {
                            memset(area_bits.mapped, 0, parents_bytes);
                            vkmesh_vk_buffer area_buffers[4];
                            area_buffers[0] = faces_buffer;
                            area_buffers[1] = vertices_buffer;
                            area_buffers[2] = parents;
                            area_buffers[3] = area_bits;
                            memset(&push, 0, sizeof(push));
                            push.n = (uint32_t) mesh->n_faces;
                            if (vkmesh_dispatch(vk, VKMESH_PIPE_COMPONENT_AREA, area_buffers, &push, face_groups)) {
                                vkmesh_vk_buffer mark_buffers[4];
                                mark_buffers[0] = parents;
                                mark_buffers[1] = area_bits;
                                mark_buffers[2] = keep_flags;
                                mark_buffers[3] = changed;
                                memset(&push, 0, sizeof(push));
                                push.n = (uint32_t) mesh->n_faces;
                                push.eps = min_area;
                                if (vkmesh_dispatch(vk, VKMESH_PIPE_MARK_COMPONENT_KEEP, mark_buffers, &push, face_groups)) {
                                    float * new_vertices = NULL;
                                    int64_t new_vertex_count = 0;
                                    int32_t * new_faces = NULL;
                                    int64_t new_face_count = 0;
                                    if (compact_mesh_from_device_flags(
                                            vk,
                                            mesh,
                                            &faces_buffer,
                                            &vertices_buffer,
                                            &keep_flags,
                                            0u,
                                            &new_vertices,
                                            &new_vertex_count,
                                            &new_faces,
                                            &new_face_count)) {
                                        *removed_faces = (int) (mesh->n_faces - new_face_count);
                                        free(mesh->vertices);
                                        free(mesh->uvs);
                                        free(mesh->faces);
                                        mesh->vertices = new_vertices;
                                        mesh->uvs = NULL;
                                        mesh->faces = new_faces;
                                        mesh->n_vertices = new_vertex_count;
                                        mesh->n_faces = new_face_count;
                                        mesh->vertex_capacity = new_vertex_count;
                                        mesh->face_capacity = new_face_count;
                                        mesh->has_uvs = 0;
                                        ok = 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            vk_buffer_destroy(vk, &faces_buffer);
            vk_buffer_destroy(vk, &vertices_buffer);
            vk_buffer_destroy(vk, &edges_buffer);
            vk_buffer_destroy(vk, &edge_dummy);
            vk_buffer_destroy(vk, &parents);
            vk_buffer_destroy(vk, &changed);
            vk_buffer_destroy(vk, &area_bits);
            vk_buffer_destroy(vk, &keep_flags);
            if (owns_vk) vkmesh_vk_destroy(vk);
            if (ok) return 1;
            fprintf(stderr, "vkmesh: Vulkan connected-components pass failed, falling back to CPU\n");
        }
    }

    vkmesh_face_pair * pairs = NULL;
    int64_t pair_count = 0;
    if (!get_manifold_face_pairs(mesh, &pairs, &pair_count)) return 0;
    int * parent = (int *) malloc((size_t) mesh->n_faces * sizeof(int));
    double * comp_area = (double *) calloc((size_t) mesh->n_faces, sizeof(double));
    uint8_t * keep = (uint8_t *) malloc((size_t) mesh->n_faces);
    if (parent == NULL || comp_area == NULL || keep == NULL) {
        free(pairs); free(parent); free(comp_area); free(keep);
        return 0;
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) parent[i] = (int) i;
    for (int64_t i = 0; i < pair_count; ++i) uf_union(parent, pairs[i].f0, pairs[i].f1);
    for (int64_t i = 0; i < mesh->n_faces; ++i) comp_area[uf_find(parent, (int) i)] += face_area(mesh, i);
    for (int64_t i = 0; i < mesh->n_faces; ++i) keep[i] = comp_area[uf_find(parent, (int) i)] >= (double) min_area ? 1u : 0u;
    int ok = mesh_remove_faces_by_mask(mesh, keep, removed_faces);
    free(pairs);
    free(parent);
    free(comp_area);
    free(keep);
    return ok;
}

static int shared_edge_needs_flip(const vkmesh_mesh * mesh, const vkmesh_face_pair * pair) {
    const int32_t * f0 = mesh->faces + (size_t) pair->f0 * 3u;
    const int32_t * f1 = mesh->faces + (size_t) pair->f1 * 3u;
    int idx0[2];
    int idx1[2];
    int found = 0;
    for (int i = 0; i < 3 && found < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (f0[i] == f1[j]) {
                idx0[found] = i;
                idx1[found] = j;
                ++found;
                break;
            }
        }
    }
    if (found != 2) return 0;
    int dir0 = (idx0[1] - idx0[0] + 3) % 3;
    int dir1 = (idx1[1] - idx1[0] + 3) % 3;
    return dir0 == dir1;
}

static int vkmesh_unify_face_orientations_vulkan(vkmesh_mesh * mesh, int * flipped_faces) {
    *flipped_faces = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_faces > UINT32_MAX / 2u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer state;
    vkmesh_vk_buffer changed;
    vkmesh_vk_buffer counter;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&state, 0, sizeof(state));
    memset(&changed, 0, sizeof(changed));
    memset(&counter, 0, sizeof(counter));

    const uint32_t face_count = (uint32_t) mesh->n_faces;
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t state_bytes = (size_t) mesh->n_faces * sizeof(uint32_t);
    int ok = 0;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, state_bytes, &state) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &changed) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    uint32_t face_groups = (face_count + 127u) / 128u;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = state;
    init_buffers[1] = changed;
    init_buffers[2] = changed;
    init_buffers[3] = changed;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_ORIENTATION_STATE, init_buffers, &push, face_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = edges_buffer;
    union_buffers[1] = faces_buffer;
    union_buffers[2] = state;
    union_buffers[3] = changed;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = state;
    compress_buffers[1] = changed;
    compress_buffers[2] = changed;
    compress_buffers[3] = changed;
    int converged = 0;
    for (int iter = 0; iter < 512; ++iter) {
        ((uint32_t *) changed.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = (uint32_t) edge_count;
        push.aux0 = face_count;
        if (!vkmesh_dispatch_pair(
                vk, VKMESH_PIPE_UNION_ORIENTATION_EDGES, union_buffers, &push, edge_groups,
                VKMESH_PIPE_COMPRESS_ORIENTATION_STATE, compress_buffers, face_count, face_groups)) goto cleanup;

        if (((const uint32_t *) changed.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    ((uint32_t *) counter.mapped)[0] = 0u;
    vkmesh_vk_buffer apply_buffers[4];
    apply_buffers[0] = faces_buffer;
    apply_buffers[1] = state;
    apply_buffers[2] = counter;
    apply_buffers[3] = changed;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_APPLY_ORIENTATION_FLIPS, apply_buffers, &push, face_groups)) goto cleanup;

    memcpy(mesh->faces, faces_buffer.mapped, faces_bytes);
    *flipped_faces = (int) ((const uint32_t *) counter.mapped)[0];
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &state);
    vk_buffer_destroy(vk, &changed);
    vk_buffer_destroy(vk, &counter);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_unify_face_orientations_cpu(vkmesh_mesh * mesh, int * flipped_faces) {
    *flipped_faces = 0;
    vkmesh_face_pair * pairs = NULL;
    int64_t pair_count = 0;
    if (!get_manifold_face_pairs(mesh, &pairs, &pair_count)) return 0;
    int * degree = (int *) calloc((size_t) mesh->n_faces + 1u, sizeof(int));
    int * offset = (int *) calloc((size_t) mesh->n_faces + 1u, sizeof(int));
    int * cursor = (int *) calloc((size_t) mesh->n_faces + 1u, sizeof(int));
    int * adjacency = NULL;
    int8_t * flip = (int8_t *) malloc((size_t) mesh->n_faces);
    if (degree == NULL || offset == NULL || cursor == NULL || flip == NULL) {
        free(pairs); free(degree); free(offset); free(cursor); free(flip);
        return 0;
    }
    memset(flip, -1, (size_t) mesh->n_faces);
    for (int64_t i = 0; i < pair_count; ++i) {
        ++degree[pairs[i].f0];
        ++degree[pairs[i].f1];
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) offset[i + 1] = offset[i] + degree[i];
    adjacency = (int *) malloc((size_t) offset[mesh->n_faces] * sizeof(int));
    if (adjacency == NULL && offset[mesh->n_faces] > 0) {
        free(pairs); free(degree); free(offset); free(cursor); free(flip);
        return 0;
    }
    memcpy(cursor, offset, ((size_t) mesh->n_faces + 1u) * sizeof(int));
    for (int64_t i = 0; i < pair_count; ++i) {
        adjacency[cursor[pairs[i].f0]++] = (int) i;
        adjacency[cursor[pairs[i].f1]++] = (int) i;
    }
    int * queue = (int *) malloc((size_t) mesh->n_faces * sizeof(int));
    if (queue == NULL) {
        free(pairs); free(degree); free(offset); free(cursor); free(adjacency); free(flip);
        return 0;
    }
    for (int64_t seed = 0; seed < mesh->n_faces; ++seed) {
        if (flip[seed] >= 0) continue;
        int begin = 0;
        int end = 0;
        queue[end++] = (int) seed;
        flip[seed] = 0;
        while (begin < end) {
            int f = queue[begin++];
            for (int it = offset[f]; it < offset[f + 1]; ++it) {
                int pair_id = adjacency[it];
                vkmesh_face_pair * pair = &pairs[pair_id];
                int other = pair->f0 == f ? pair->f1 : pair->f0;
                int need = shared_edge_needs_flip(mesh, pair);
                int next_flip = flip[f] ^ need;
                if (flip[other] < 0) {
                    flip[other] = (int8_t) next_flip;
                    queue[end++] = other;
                }
            }
        }
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        if ((flip[i] & 1) == 0) continue;
        int32_t * f = mesh->faces + (size_t) i * 3u;
        int32_t t = f[1];
        f[1] = f[2];
        f[2] = t;
        ++(*flipped_faces);
    }
    free(pairs);
    free(degree);
    free(offset);
    free(cursor);
    free(adjacency);
    free(flip);
    free(queue);
    return 1;
}

static int vkmesh_unify_face_orientations(vkmesh_mesh * mesh, int * flipped_faces) {
    if (vkmesh_unify_face_orientations_vulkan(mesh, flipped_faces)) return 1;
    return vkmesh_unify_face_orientations_cpu(mesh, flipped_faces);
}

static int build_vertex_face_adjacency_vulkan(const vkmesh_mesh * mesh, int ** offset_out, int ** adj_out) {
    *offset_out = NULL;
    *adj_out = NULL;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX ||
        mesh->n_vertices <= 0 || mesh->n_vertices >= UINT32_MAX ||
        (uint64_t) mesh->n_faces * 3ull > INT_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer degree_buffer;
    vkmesh_vk_buffer offset_a_buffer;
    vkmesh_vk_buffer offset_b_buffer;
    vkmesh_vk_buffer adjacency_buffer;
    vkmesh_vk_buffer dummy_buffer;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&degree_buffer, 0, sizeof(degree_buffer));
    memset(&offset_a_buffer, 0, sizeof(offset_a_buffer));
    memset(&offset_b_buffer, 0, sizeof(offset_b_buffer));
    memset(&adjacency_buffer, 0, sizeof(adjacency_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));

    const uint32_t vertex_count = (uint32_t) mesh->n_vertices;
    const uint32_t offset_count = vertex_count + 1u;
    const uint32_t face_count = (uint32_t) mesh->n_faces;
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t degree_bytes = (size_t) mesh->n_vertices * sizeof(uint32_t);
    const size_t offset_bytes = (size_t) offset_count * sizeof(uint32_t);
    int ok = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, degree_bytes, &degree_buffer) ||
        !vk_buffer_create(vk, offset_bytes, &offset_a_buffer) ||
        !vk_buffer_create(vk, offset_bytes, &offset_b_buffer) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &dummy_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);

    vkmesh_vk_buffer degree_buffers[4];
    degree_buffers[0] = faces_buffer;
    degree_buffers[1] = degree_buffer;
    degree_buffers[2] = dummy_buffer;
    degree_buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = vertex_count;
    uint32_t groups = (face_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_FACE_DEGREE, degree_buffers, &push, groups)) goto cleanup;

    vkmesh_vk_buffer seed_buffers[4];
    seed_buffers[0] = degree_buffer;
    seed_buffers[1] = offset_a_buffer;
    seed_buffers[2] = dummy_buffer;
    seed_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    uint32_t offset_groups = (offset_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SEED_VERTEX_OFFSETS, seed_buffers, &push, offset_groups)) goto cleanup;

    if (!vkmesh_scan_u32_inclusive(
            vk, &offset_a_buffer, &offset_b_buffer, offset_count, &dummy_buffer)) goto cleanup;
    vkmesh_vk_buffer * scan_in = &offset_b_buffer;

    int * offset = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(*offset));
    if (offset == NULL) goto cleanup;
    const uint32_t * offset_u32 = (const uint32_t *) scan_in->mapped;
    int64_t total_adj = (int64_t) offset_u32[vertex_count];
    if (total_adj > INT_MAX) {
        free(offset);
        goto cleanup;
    }
    for (int64_t i = 0; i <= mesh->n_vertices; ++i) {
        if (offset_u32[i] > INT_MAX) {
            free(offset);
            goto cleanup;
        }
        offset[i] = (int) offset_u32[i];
    }

    int * adj = NULL;
    if (total_adj > 0) {
        adj = (int *) malloc((size_t) total_adj * sizeof(*adj));
        if (adj == NULL) {
            free(offset);
            goto cleanup;
        }
    }

    const size_t adj_bytes = (size_t) (total_adj > 0 ? total_adj : 1) * sizeof(uint32_t);
    if (!vk_buffer_create(vk, adj_bytes, &adjacency_buffer)) {
        free(offset);
        free(adj);
        goto cleanup;
    }

    vkmesh_vk_buffer adj_buffers[4];
    adj_buffers[0] = faces_buffer;
    adj_buffers[1] = *scan_in;
    adj_buffers[2] = adjacency_buffer;
    adj_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = vertex_count;
    push.aux1 = (uint32_t) total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_FACE_ADJACENCY, adj_buffers, &push, groups)) {
        free(offset);
        free(adj);
        goto cleanup;
    }
    if (total_adj > 0) memcpy(adj, adjacency_buffer.mapped, (size_t) total_adj * sizeof(*adj));

    *offset_out = offset;
    *adj_out = adj;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &degree_buffer);
    vk_buffer_destroy(vk, &offset_a_buffer);
    vk_buffer_destroy(vk, &offset_b_buffer);
    vk_buffer_destroy(vk, &adjacency_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int build_vertex_face_adjacency_cpu(const vkmesh_mesh * mesh, int ** offset_out, int ** adj_out) {
    *offset_out = NULL;
    *adj_out = NULL;
    int * degree = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(int));
    int * offset = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(int));
    int * cursor = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(int));
    if (degree == NULL || offset == NULL || cursor == NULL) {
        free(degree); free(offset); free(cursor);
        return 0;
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        ++degree[f[0]];
        ++degree[f[1]];
        ++degree[f[2]];
    }
    for (int64_t i = 0; i < mesh->n_vertices; ++i) offset[i + 1] = offset[i] + degree[i];
    int * adj = (int *) malloc((size_t) offset[mesh->n_vertices] * sizeof(int));
    if (adj == NULL && offset[mesh->n_vertices] > 0) {
        free(degree); free(offset); free(cursor);
        return 0;
    }
    memcpy(cursor, offset, ((size_t) mesh->n_vertices + 1u) * sizeof(int));
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        adj[cursor[f[0]]++] = (int) i;
        adj[cursor[f[1]]++] = (int) i;
        adj[cursor[f[2]]++] = (int) i;
    }
    free(degree);
    free(cursor);
    *offset_out = offset;
    *adj_out = adj;
    return 1;
}

static int build_vertex_face_adjacency(const vkmesh_mesh * mesh, int ** offset_out, int ** adj_out) {
    if (build_vertex_face_adjacency_vulkan(mesh, offset_out, adj_out)) return 1;
    return build_vertex_face_adjacency_cpu(mesh, offset_out, adj_out);
}

static int get_unique_simplify_edges_vulkan(
    const vkmesh_mesh * mesh,
    vkmesh_simplify_edge ** edges_out,
    int64_t * edge_count_out,
    uint8_t ** vertex_boundary_out) {
    *edges_out = NULL;
    *edge_count_out = 0;
    *vertex_boundary_out = NULL;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer unique_buffer;
    vkmesh_vk_buffer boundary_flags;
    vkmesh_vk_buffer counter;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&unique_buffer, 0, sizeof(unique_buffer));
    memset(&boundary_flags, 0, sizeof(boundary_flags));
    memset(&counter, 0, sizeof(counter));

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t boundary_bytes = (size_t) mesh->n_vertices * sizeof(uint32_t);
    int ok = 0;
    size_t raw_count = 0;
    size_t raw_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, boundary_bytes, &boundary_flags) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &edges_buffer,
            &dummy_buffer,
            &raw_count,
            &raw_sort_count)) {
        goto cleanup;
    }
    if (!vk_buffer_create(vk, raw_count * 2u * sizeof(uint32_t), &unique_buffer)) goto cleanup;

    vkmesh_vk_buffer buffers[4];
    buffers[0] = edges_buffer;
    buffers[1] = unique_buffer;
    buffers[2] = boundary_flags;
    buffers[3] = counter;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) raw_count;
    push.aux0 = (uint32_t) mesh->n_vertices;
    uint32_t groups = (uint32_t) ((raw_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_UNIQUE_SIMPLIFY_EDGES, buffers, &push, groups)) goto cleanup;

    uint32_t unique_count = ((const uint32_t *) counter.mapped)[0];
    if ((size_t) unique_count > raw_count) goto cleanup;
    vkmesh_simplify_edge * edges = (vkmesh_simplify_edge *) malloc((size_t) unique_count * sizeof(*edges));
    uint8_t * boundary = (uint8_t *) calloc((size_t) mesh->n_vertices, sizeof(*boundary));
    if ((edges == NULL && unique_count > 0u) || boundary == NULL) {
        free(edges);
        free(boundary);
        goto cleanup;
    }
    const uint32_t * unique_u32 = (const uint32_t *) unique_buffer.mapped;
    for (uint32_t i = 0; i < unique_count; ++i) {
        edges[i].v0 = (int32_t) unique_u32[(size_t) i * 2u + 0u];
        edges[i].v1 = (int32_t) unique_u32[(size_t) i * 2u + 1u];
        edges[i].cost = INFINITY;
        edges[i].pos[0] = 0.0f;
        edges[i].pos[1] = 0.0f;
        edges[i].pos[2] = 0.0f;
    }
    const uint32_t * boundary_u32 = (const uint32_t *) boundary_flags.mapped;
    for (int64_t i = 0; i < mesh->n_vertices; ++i) boundary[i] = boundary_u32[i] != 0u ? 1u : 0u;

    *edges_out = edges;
    *edge_count_out = (int64_t) unique_count;
    *vertex_boundary_out = boundary;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &unique_buffer);
    vk_buffer_destroy(vk, &boundary_flags);
    vk_buffer_destroy(vk, &counter);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int get_unique_simplify_edges_cpu(
    const vkmesh_mesh * mesh,
    vkmesh_simplify_edge ** edges_out,
    int64_t * edge_count_out,
    uint8_t ** vertex_boundary_out) {
    *edges_out = NULL;
    *edge_count_out = 0;
    *vertex_boundary_out = NULL;
    vkmesh_edge * raw = NULL;
    int64_t raw_count = 0;
    if (!get_sorted_edges(mesh, &raw, &raw_count)) return 0;
    vkmesh_simplify_edge * edges = (vkmesh_simplify_edge *) malloc((size_t) raw_count * sizeof(*edges));
    uint8_t * boundary = (uint8_t *) calloc((size_t) mesh->n_vertices, sizeof(uint8_t));
    if (edges == NULL || boundary == NULL) {
        free(raw); free(edges); free(boundary);
        return 0;
    }
    int64_t count = 0;
    for (int64_t i = 0; i < raw_count;) {
        int64_t j = i + 1;
        while (j < raw_count && raw[j].min_v == raw[i].min_v && raw[j].max_v == raw[i].max_v) ++j;
        edges[count].v0 = (int32_t) raw[i].min_v;
        edges[count].v1 = (int32_t) raw[i].max_v;
        edges[count].cost = INFINITY;
        edges[count].pos[0] = 0.0f;
        edges[count].pos[1] = 0.0f;
        edges[count].pos[2] = 0.0f;
        if (j - i == 1) {
            boundary[raw[i].min_v] = 1u;
            boundary[raw[i].max_v] = 1u;
        }
        ++count;
        i = j;
    }
    free(raw);
    *edges_out = edges;
    *edge_count_out = count;
    *vertex_boundary_out = boundary;
    return 1;
}

static int get_unique_simplify_edges(
    const vkmesh_mesh * mesh,
    vkmesh_simplify_edge ** edges_out,
    int64_t * edge_count_out,
    uint8_t ** vertex_boundary_out) {
    if (get_unique_simplify_edges_vulkan(mesh, edges_out, edge_count_out, vertex_boundary_out)) return 1;
    return get_unique_simplify_edges_cpu(mesh, edges_out, edge_count_out, vertex_boundary_out);
}

static void qem_add_plane(vkmesh_qem * q, double a, double b, double c, double d) {
    q->m[0] += a * a;
    q->m[1] += a * b;
    q->m[2] += a * c;
    q->m[3] += a * d;
    q->m[4] += b * b;
    q->m[5] += b * c;
    q->m[6] += b * d;
    q->m[7] += c * c;
    q->m[8] += c * d;
    q->m[9] += d * d;
}

static double qem_eval(const vkmesh_qem * q, const float p[3]) {
    double x = p[0], y = p[1], z = p[2], w = 1.0;
    return
        q->m[0] * x * x + 2.0 * q->m[1] * x * y + 2.0 * q->m[2] * x * z + 2.0 * q->m[3] * x * w +
        q->m[4] * y * y + 2.0 * q->m[5] * y * z + 2.0 * q->m[6] * y * w +
        q->m[7] * z * z + 2.0 * q->m[8] * z * w +
        q->m[9] * w * w;
}

static int build_vertex_qems_vulkan(const vkmesh_mesh * mesh, vkmesh_qem ** qems_out) {
    *qems_out = NULL;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer qem_buffer;
    vkmesh_vk_buffer dummy_buffer;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&qem_buffer, 0, sizeof(qem_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t qem_bytes = (size_t) mesh->n_vertices * 10u * sizeof(uint32_t);
    int ok = 0;
    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, qem_bytes, &qem_buffer) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &dummy_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    vkmesh_vk_buffer buffers[4];
    buffers[0] = faces_buffer;
    buffers[1] = vertices_buffer;
    buffers[2] = qem_buffer;
    buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_faces;
    push.aux0 = (uint32_t) mesh->n_vertices;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_QEM, buffers, &push, groups)) goto cleanup;

    vkmesh_qem * qems = (vkmesh_qem *) calloc((size_t) mesh->n_vertices, sizeof(*qems));
    if (qems == NULL) goto cleanup;
    const float * qem_f = (const float *) qem_buffer.mapped;
    for (int64_t v = 0; v < mesh->n_vertices; ++v) {
        for (uint32_t k = 0; k < 10u; ++k) {
            qems[v].m[k] = (double) qem_f[(size_t) v * 10u + k];
        }
    }
    *qems_out = qems;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &qem_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int build_vertex_qems_cpu(const vkmesh_mesh * mesh, vkmesh_qem ** qems_out) {
    *qems_out = (vkmesh_qem *) calloc((size_t) mesh->n_vertices, sizeof(vkmesh_qem));
    if (*qems_out == NULL) return 0;
    vkmesh_qem * qems = *qems_out;
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        const float * a = mesh->vertices + (size_t) f[0] * 3u;
        const float * b = mesh->vertices + (size_t) f[1] * 3u;
        const float * c = mesh->vertices + (size_t) f[2] * 3u;
        double abx = (double) b[0] - (double) a[0];
        double aby = (double) b[1] - (double) a[1];
        double abz = (double) b[2] - (double) a[2];
        double acx = (double) c[0] - (double) a[0];
        double acy = (double) c[1] - (double) a[1];
        double acz = (double) c[2] - (double) a[2];
        double nx = aby * acz - abz * acy;
        double ny = abz * acx - abx * acz;
        double nz = abx * acy - aby * acx;
        double len = sqrt(nx * nx + ny * ny + nz * nz);
        if (len <= 1e-30) continue;
        nx /= len; ny /= len; nz /= len;
        double d = -(nx * a[0] + ny * a[1] + nz * a[2]);
        qem_add_plane(&qems[f[0]], nx, ny, nz, d);
        qem_add_plane(&qems[f[1]], nx, ny, nz, d);
        qem_add_plane(&qems[f[2]], nx, ny, nz, d);
    }
    return 1;
}

static int build_vertex_qems(const vkmesh_mesh * mesh, vkmesh_qem ** qems_out) {
    return build_vertex_qems_cpu(mesh, qems_out);
}

static int face_contains_vertex(const int32_t * f, int32_t v) {
    return f[0] == v || f[1] == v || f[2] == v;
}

static int process_incident_for_simplify(
    const vkmesh_mesh * mesh,
    int32_t face_id,
    int32_t keep_v,
    int32_t other_v,
    const float new_pos[3],
    double * skinny_cost,
    int * num_tri) {
    const int32_t * f = mesh->faces + (size_t) face_id * 3u;
    if (face_contains_vertex(f, other_v)) return 1;
    const float * p0 = mesh->vertices + (size_t) f[0] * 3u;
    const float * p1 = mesh->vertices + (size_t) f[1] * 3u;
    const float * p2 = mesh->vertices + (size_t) f[2] * 3u;
    double old_e1[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
    double old_e2[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
    double old_n[3] = {
        old_e1[1] * old_e2[2] - old_e1[2] * old_e2[1],
        old_e1[2] * old_e2[0] - old_e1[0] * old_e2[2],
        old_e1[0] * old_e2[1] - old_e1[1] * old_e2[0],
    };
    double np[3][3];
    const float * src[3] = { p0, p1, p2 };
    for (int i = 0; i < 3; ++i) {
        if (f[i] == keep_v) {
            np[i][0] = new_pos[0];
            np[i][1] = new_pos[1];
            np[i][2] = new_pos[2];
        } else {
            np[i][0] = src[i][0];
            np[i][1] = src[i][1];
            np[i][2] = src[i][2];
        }
    }
    double new_e1[3] = { np[1][0] - np[0][0], np[1][1] - np[0][1], np[1][2] - np[0][2] };
    double new_e2[3] = { np[2][0] - np[0][0], np[2][1] - np[0][1], np[2][2] - np[0][2] };
    double new_n[3] = {
        new_e1[1] * new_e2[2] - new_e1[2] * new_e2[1],
        new_e1[2] * new_e2[0] - new_e1[0] * new_e2[2],
        new_e1[0] * new_e2[1] - new_e1[1] * new_e2[0],
    };
    double dot_n = old_n[0] * new_n[0] + old_n[1] * new_n[1] + old_n[2] * new_n[2];
    if (dot_n < 0.0) return 0;

    double e0[3] = { np[2][0] - np[1][0], np[2][1] - np[1][1], np[2][2] - np[1][2] };
    double area2 = sqrt(new_n[0] * new_n[0] + new_n[1] * new_n[1] + new_n[2] * new_n[2]);
    double len0 = e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2];
    double len1 = new_e1[0] * new_e1[0] + new_e1[1] * new_e1[1] + new_e1[2] * new_e1[2];
    double len2 = new_e2[0] * new_e2[0] + new_e2[1] * new_e2[1] + new_e2[2] * new_e2[2];
    double denom = len0 + len1 + len2;
    if (denom < 1e-24) denom = 1e-24;
    double shape = 2.0 * sqrt(3.0) * area2 / denom;
    if (shape < 0.0) shape = 0.0;
    if (shape > 1.0) shape = 1.0;
    *skinny_cost += 1.0 - shape;
    ++(*num_tri);
    return 1;
}

static double simplify_edge_cost(
    const vkmesh_mesh * mesh,
    const int * v2f_offset,
    const int * v2f,
    const uint8_t * boundary,
    const vkmesh_qem * qems,
    float lambda_edge_length,
    float lambda_skinny,
    vkmesh_simplify_edge * edge) {
    const float * v0 = mesh->vertices + (size_t) edge->v0 * 3u;
    const float * v1 = mesh->vertices + (size_t) edge->v1 * 3u;
    float w0 = 0.5f;
    if (boundary[edge->v0] && !boundary[edge->v1]) w0 = 1.0f;
    else if (!boundary[edge->v0] && boundary[edge->v1]) w0 = 0.0f;
    edge->pos[0] = v0[0] * w0 + v1[0] * (1.0f - w0);
    edge->pos[1] = v0[1] * w0 + v1[1] * (1.0f - w0);
    edge->pos[2] = v0[2] * w0 + v1[2] * (1.0f - w0);

    vkmesh_qem q = qems[edge->v0];
    for (int i = 0; i < 10; ++i) q.m[i] += qems[edge->v1].m[i];
    double cost = qem_eval(&q, edge->pos);
    double dx = (double) v1[0] - (double) v0[0];
    double dy = (double) v1[1] - (double) v0[1];
    double dz = (double) v1[2] - (double) v0[2];
    double len2 = dx * dx + dy * dy + dz * dz;
    cost += (double) lambda_edge_length * len2;

    double skinny = 0.0;
    int num_tri = 0;
    for (int i = v2f_offset[edge->v0]; i < v2f_offset[edge->v0 + 1]; ++i) {
        if (!process_incident_for_simplify(mesh, v2f[i], edge->v0, edge->v1, edge->pos, &skinny, &num_tri)) return INFINITY;
    }
    for (int i = v2f_offset[edge->v1]; i < v2f_offset[edge->v1 + 1]; ++i) {
        if (!process_incident_for_simplify(mesh, v2f[i], edge->v1, edge->v0, edge->pos, &skinny, &num_tri)) return INFINITY;
    }
    if (num_tri > 0) skinny /= (double) num_tri;
    cost += (double) lambda_skinny * skinny * len2;
    return cost;
}

static int propagate_edge_cost_to_faces(
    const vkmesh_mesh * mesh,
    const int * v2f_offset,
    const int * v2f,
    const vkmesh_simplify_edge * edges,
    int64_t edge_count,
    int ** best_edge_out) {
    int * best_edge = (int *) malloc((size_t) mesh->n_faces * sizeof(int));
    double * best_cost = (double *) malloc((size_t) mesh->n_faces * sizeof(double));
    if (best_edge == NULL || best_cost == NULL) {
        free(best_edge); free(best_cost);
        return 0;
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        best_edge[i] = -1;
        best_cost[i] = INFINITY;
    }
    for (int64_t e = 0; e < edge_count; ++e) {
        if (!isfinite(edges[e].cost)) continue;
        int32_t verts[2] = { edges[e].v0, edges[e].v1 };
        for (int side = 0; side < 2; ++side) {
            int32_t v = verts[side];
            for (int i = v2f_offset[v]; i < v2f_offset[v + 1]; ++i) {
                int f = v2f[i];
                if (edges[e].cost < best_cost[f] || (edges[e].cost == best_cost[f] && e < best_edge[f])) {
                    best_cost[f] = edges[e].cost;
                    best_edge[f] = (int) e;
                }
            }
        }
    }
    free(best_cost);
    *best_edge_out = best_edge;
    return 1;
}

static int vkmesh_simplify_step_vulkan(
    vkmesh_mesh * mesh,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    int * collapsed_edges,
    int * removed_faces) {
    *collapsed_edges = 0;
    *removed_faces = 0;
    if (mesh == NULL || mesh->n_faces <= 0) return 1;
    if (mesh->n_faces > UINT32_MAX || mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces > UINT32_MAX / 3u) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer degree_buffer;
    vkmesh_vk_buffer offset_a_buffer;
    vkmesh_vk_buffer offset_b_buffer;
    vkmesh_vk_buffer adjacency_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer edge_dummy;
    vkmesh_vk_buffer unique_edges;
    vkmesh_vk_buffer boundary_flags;
    vkmesh_vk_buffer counter;
    vkmesh_vk_buffer scratch;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&degree_buffer, 0, sizeof(degree_buffer));
    memset(&offset_a_buffer, 0, sizeof(offset_a_buffer));
    memset(&offset_b_buffer, 0, sizeof(offset_b_buffer));
    memset(&adjacency_buffer, 0, sizeof(adjacency_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&edge_dummy, 0, sizeof(edge_dummy));
    memset(&unique_edges, 0, sizeof(unique_edges));
    memset(&boundary_flags, 0, sizeof(boundary_flags));
    memset(&counter, 0, sizeof(counter));
    memset(&scratch, 0, sizeof(scratch));

    const uint32_t vertex_count = (uint32_t) mesh->n_vertices;
    const uint32_t face_count = (uint32_t) mesh->n_faces;
    const uint32_t offset_count = vertex_count + 1u;
    const size_t faces_bytes = (size_t) face_count * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) vertex_count * 3u * sizeof(float);
    const size_t degree_bytes = (size_t) vertex_count * sizeof(uint32_t);
    const size_t offset_bytes = (size_t) offset_count * sizeof(uint32_t);
    int ok = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, degree_bytes, &degree_buffer) ||
        !vk_buffer_create(vk, offset_bytes, &offset_a_buffer) ||
        !vk_buffer_create(vk, offset_bytes, &offset_b_buffer) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &counter)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    uint32_t face_groups = (face_count + 127u) / 128u;
    uint32_t vertex_groups = (vertex_count + 127u) / 128u;

    vkmesh_vk_buffer degree_buffers[4];
    degree_buffers[0] = faces_buffer;
    degree_buffers[1] = degree_buffer;
    degree_buffers[2] = counter;
    degree_buffers[3] = counter;
    push.n = face_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_FACE_DEGREE, degree_buffers, &push, face_groups)) goto cleanup;

    vkmesh_vk_buffer seed_buffers[4];
    seed_buffers[0] = degree_buffer;
    seed_buffers[1] = offset_a_buffer;
    seed_buffers[2] = counter;
    seed_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    uint32_t offset_groups = (offset_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SEED_VERTEX_OFFSETS, seed_buffers, &push, offset_groups)) goto cleanup;

    if (!vkmesh_scan_u32_inclusive(
            vk, &offset_a_buffer, &offset_b_buffer, offset_count, &counter)) goto cleanup;
    vkmesh_vk_buffer * scan_in = &offset_b_buffer;
    vkmesh_vk_buffer * scan_out = &offset_a_buffer;

    const uint32_t total_adj = ((const uint32_t *) scan_in->mapped)[vertex_count];
    if (total_adj != face_count * 3u) goto cleanup;
    if (!vk_buffer_create(vk, (size_t) total_adj * sizeof(uint32_t), &adjacency_buffer)) goto cleanup;

    vkmesh_vk_buffer cursor_copy_buffers[4];
    cursor_copy_buffers[0] = *scan_in;
    cursor_copy_buffers[1] = *scan_out;
    cursor_copy_buffers[2] = counter;
    cursor_copy_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, cursor_copy_buffers, &push, offset_groups)) goto cleanup;

    vkmesh_vk_buffer adj_buffers[4];
    adj_buffers[0] = faces_buffer;
    adj_buffers[1] = *scan_out;
    adj_buffers[2] = adjacency_buffer;
    adj_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = vertex_count;
    push.aux1 = total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_FACE_ADJACENCY, adj_buffers, &push, face_groups)) goto cleanup;

    size_t raw_edge_count = 0;
    size_t raw_edge_sort_count = 0;
    if (!expand_edges_device(vk, mesh, &faces_buffer, &vertices_buffer, &edges_buffer, &edge_dummy, &raw_edge_count, &raw_edge_sort_count)) {
        goto cleanup;
    }
    if (raw_edge_count > UINT32_MAX) goto cleanup;
    if (!vk_buffer_create(vk, raw_edge_count * 2u * sizeof(uint32_t), &unique_edges) ||
        !vk_buffer_create(vk, (size_t) vertex_count * sizeof(uint32_t), &boundary_flags)) {
        goto cleanup;
    }
    ((uint32_t *) counter.mapped)[0] = 0u;

    vkmesh_vk_buffer unique_buffers[4];
    unique_buffers[0] = edges_buffer;
    unique_buffers[1] = unique_edges;
    unique_buffers[2] = boundary_flags;
    unique_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) raw_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = 0u;
    uint32_t edge_groups = (uint32_t) ((raw_edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_UNIQUE_SIMPLIFY_EDGES, unique_buffers, &push, edge_groups)) goto cleanup;
    const uint32_t unique_edge_count = ((const uint32_t *) counter.mapped)[0];
    if ((size_t) unique_edge_count > raw_edge_count) goto cleanup;

    /* The sorted 3F edge records are only needed to construct the compact
       unique-edge array. Releasing them before the packed simplify scratch is
       allocated keeps the two largest transient phases from overlapping. */
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &edge_dummy);

    const uint64_t qem_base = 0u;
    const uint64_t offset_base = (uint64_t) vertex_count * 10ull;
    const uint64_t adjacency_base = offset_base + (uint64_t) offset_count;
    const uint64_t boundary_base = adjacency_base + (uint64_t) total_adj;
    const uint64_t cost_base = boundary_base + (uint64_t) vertex_count;
    const uint64_t best_cost_base = cost_base + (uint64_t) unique_edge_count;
    const uint64_t best_edge_base = best_cost_base + (uint64_t) face_count;
    const uint64_t face_keep_base = best_edge_base + (uint64_t) face_count;
    const uint64_t counter_base = face_keep_base + (uint64_t) face_count;
    const uint64_t scratch_count = counter_base + 1ull;
    (void) qem_base;
    if (scratch_count > UINT32_MAX) goto cleanup;
    if (!vk_buffer_create(vk, (size_t) scratch_count * sizeof(uint32_t), &scratch)) goto cleanup;

    vkmesh_vk_buffer copy_buffers[4];
    copy_buffers[0] = *scan_in;
    copy_buffers[1] = scratch;
    copy_buffers[2] = counter;
    copy_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    push.aux0 = 0u;
    push.aux1 = (uint32_t) offset_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, offset_groups)) goto cleanup;

    copy_buffers[0] = adjacency_buffer;
    copy_buffers[1] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = total_adj;
    push.aux0 = 0u;
    push.aux1 = (uint32_t) adjacency_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (total_adj + 127u) / 128u)) goto cleanup;

    copy_buffers[0] = boundary_flags;
    copy_buffers[1] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count;
    push.aux0 = 0u;
    push.aux1 = (uint32_t) boundary_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, vertex_groups)) goto cleanup;

    vkmesh_vk_buffer fill_buffers[4];
    fill_buffers[0] = scratch;
    fill_buffers[1] = counter;
    fill_buffers[2] = counter;
    fill_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = 0x7f800000u;
    push.aux1 = (uint32_t) best_cost_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, face_groups)) goto cleanup;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = UINT32_MAX;
    push.aux1 = (uint32_t) best_edge_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, face_groups)) goto cleanup;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = 1u;
    push.aux1 = (uint32_t) face_keep_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, face_groups)) goto cleanup;
    memset(&push, 0, sizeof(push));
    push.n = 1u;
    push.aux0 = 0u;
    push.aux1 = (uint32_t) counter_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, 1u)) goto cleanup;

    vkmesh_vk_buffer qem_buffers[4];
    qem_buffers[0] = faces_buffer;
    qem_buffers[1] = vertices_buffer;
    qem_buffers[2] = scratch;
    qem_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_QEM, qem_buffers, &push, vertex_groups)) goto cleanup;

    vkmesh_vk_buffer cost_buffers[4];
    cost_buffers[0] = unique_edges;
    cost_buffers[1] = vertices_buffer;
    cost_buffers[2] = faces_buffer;
    cost_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = unique_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    push.eps = lambda_edge_length;
    push.rel_eps = lambda_skinny;
    uint32_t unique_groups = (unique_edge_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_EDGE_COST, cost_buffers, &push, unique_groups)) goto cleanup;

    vkmesh_vk_buffer propagate_buffers[4];
    propagate_buffers[0] = unique_edges;
    propagate_buffers[1] = counter;
    propagate_buffers[2] = counter;
    propagate_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = unique_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_PROPAGATE_COST, propagate_buffers, &push, unique_groups)) goto cleanup;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_BEST_EDGE, propagate_buffers, &push, unique_groups)) goto cleanup;

    vkmesh_vk_buffer collapse_buffers[4];
    collapse_buffers[0] = unique_edges;
    collapse_buffers[1] = vertices_buffer;
    collapse_buffers[2] = faces_buffer;
    collapse_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = unique_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    push.eps = threshold;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_COLLAPSE_EDGES, collapse_buffers, &push, unique_groups)) goto cleanup;
    *collapsed_edges = (int) ((const uint32_t *) scratch.mapped)[counter_base];

    float * new_vertices = NULL;
    int64_t new_vertex_count = 0;
    int32_t * new_faces = NULL;
    int64_t new_face_count = 0;
    if (!compact_mesh_from_device_flags(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &scratch,
            (uint32_t) face_keep_base,
            &new_vertices,
            &new_vertex_count,
            &new_faces,
            &new_face_count)) {
        goto cleanup;
    }
    *removed_faces = (int) (mesh->n_faces - new_face_count);
    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = new_vertices;
    mesh->uvs = NULL;
    mesh->faces = new_faces;
    mesh->n_vertices = new_vertex_count;
    mesh->n_faces = new_face_count;
    mesh->vertex_capacity = new_vertex_count;
    mesh->face_capacity = new_face_count;
    mesh->has_uvs = 0;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &degree_buffer);
    vk_buffer_destroy(vk, &offset_a_buffer);
    vk_buffer_destroy(vk, &offset_b_buffer);
    vk_buffer_destroy(vk, &adjacency_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &edge_dummy);
    vk_buffer_destroy(vk, &unique_edges);
    vk_buffer_destroy(vk, &boundary_flags);
    vk_buffer_destroy(vk, &counter);
    vk_buffer_destroy(vk, &scratch);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_device_simplify_step(
    vkmesh_device_mesh * dm,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    uint32_t * collapsed_edges,
    uint32_t * removed_faces) {
    if (collapsed_edges != NULL) *collapsed_edges = 0u;
    if (removed_faces != NULL) *removed_faces = 0u;
    if (dm == NULL || dm->vk == NULL) return 0;
    if (dm->n_faces == 0u) return 1;
    if (dm->n_vertices == 0u || dm->n_faces > UINT32_MAX / 3u) return 0;

    vkmesh_vk * vk = dm->vk;
    vkmesh_mesh view;
    memset(&view, 0, sizeof(view));
    view.n_vertices = dm->n_vertices;
    view.n_faces = dm->n_faces;

    vkmesh_vk_buffer degree_buffer;
    vkmesh_vk_buffer offset_a_buffer;
    vkmesh_vk_buffer offset_b_buffer;
    vkmesh_vk_buffer adjacency_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer edge_dummy;
    vkmesh_vk_buffer unique_edges;
    vkmesh_vk_buffer boundary_flags;
    vkmesh_vk_buffer counter;
    vkmesh_vk_buffer scratch;
    memset(&degree_buffer, 0, sizeof(degree_buffer));
    memset(&offset_a_buffer, 0, sizeof(offset_a_buffer));
    memset(&offset_b_buffer, 0, sizeof(offset_b_buffer));
    memset(&adjacency_buffer, 0, sizeof(adjacency_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&edge_dummy, 0, sizeof(edge_dummy));
    memset(&unique_edges, 0, sizeof(unique_edges));
    memset(&boundary_flags, 0, sizeof(boundary_flags));
    memset(&counter, 0, sizeof(counter));
    memset(&scratch, 0, sizeof(scratch));

    const uint32_t vertex_count = dm->n_vertices;
    const uint32_t face_count = dm->n_faces;
    const uint32_t offset_count = vertex_count + 1u;
    const size_t degree_bytes = (size_t) vertex_count * sizeof(uint32_t);
    const size_t offset_bytes = (size_t) offset_count * sizeof(uint32_t);
    int ok = 0;

    if (!vk_buffer_create(vk, degree_bytes, &degree_buffer) ||
        !vk_buffer_create(vk, offset_bytes, &offset_a_buffer) ||
        !vk_buffer_create(vk, offset_bytes, &offset_b_buffer) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &counter)) {
        goto cleanup;
    }

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    uint32_t face_groups = (face_count + 127u) / 128u;
    uint32_t vertex_groups = (vertex_count + 127u) / 128u;

    vkmesh_vk_buffer degree_buffers[4];
    degree_buffers[0] = dm->faces;
    degree_buffers[1] = degree_buffer;
    degree_buffers[2] = counter;
    degree_buffers[3] = counter;
    push.n = face_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_FACE_DEGREE, degree_buffers, &push, face_groups)) goto cleanup;

    vkmesh_vk_buffer seed_buffers[4];
    seed_buffers[0] = degree_buffer;
    seed_buffers[1] = offset_a_buffer;
    seed_buffers[2] = counter;
    seed_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    uint32_t offset_groups = (offset_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SEED_VERTEX_OFFSETS, seed_buffers, &push, offset_groups)) goto cleanup;

    if (!vkmesh_scan_u32_inclusive(
            vk, &offset_a_buffer, &offset_b_buffer, offset_count, &counter)) goto cleanup;
    vkmesh_vk_buffer * scan_in = &offset_b_buffer;
    vkmesh_vk_buffer * scan_out = &offset_a_buffer;

    const uint32_t total_adj = ((const uint32_t *) scan_in->mapped)[vertex_count];
    if (total_adj != face_count * 3u) goto cleanup;
    if (!vk_buffer_create(vk, (size_t) total_adj * sizeof(uint32_t), &adjacency_buffer)) goto cleanup;

    vkmesh_vk_buffer cursor_copy_buffers[4];
    cursor_copy_buffers[0] = *scan_in;
    cursor_copy_buffers[1] = *scan_out;
    cursor_copy_buffers[2] = counter;
    cursor_copy_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, cursor_copy_buffers, &push, offset_groups)) goto cleanup;

    vkmesh_vk_buffer adj_buffers[4];
    adj_buffers[0] = dm->faces;
    adj_buffers[1] = *scan_out;
    adj_buffers[2] = adjacency_buffer;
    adj_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = vertex_count;
    push.aux1 = total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_FACE_ADJACENCY, adj_buffers, &push, face_groups)) goto cleanup;

    size_t raw_edge_count = 0;
    size_t raw_edge_sort_count = 0;
    if (!expand_edges_device(vk, &view, &dm->faces, &dm->vertices, &edges_buffer, &edge_dummy, &raw_edge_count, &raw_edge_sort_count)) {
        goto cleanup;
    }
    if (raw_edge_count > UINT32_MAX) goto cleanup;
    if (!vk_buffer_create(vk, raw_edge_count * 2u * sizeof(uint32_t), &unique_edges) ||
        !vk_buffer_create(vk, (size_t) vertex_count * sizeof(uint32_t), &boundary_flags)) {
        goto cleanup;
    }
    ((uint32_t *) counter.mapped)[0] = 0u;

    vkmesh_vk_buffer unique_buffers[4];
    unique_buffers[0] = edges_buffer;
    unique_buffers[1] = unique_edges;
    unique_buffers[2] = boundary_flags;
    unique_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) raw_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = 0u;
    uint32_t edge_groups = (uint32_t) ((raw_edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_UNIQUE_SIMPLIFY_EDGES, unique_buffers, &push, edge_groups)) goto cleanup;
    const uint32_t unique_edge_count = ((const uint32_t *) counter.mapped)[0];
    if ((size_t) unique_edge_count > raw_edge_count) goto cleanup;

    const uint64_t qem_base = 0u;
    const uint64_t offset_base = (uint64_t) vertex_count * 10ull;
    const uint64_t adjacency_base = offset_base + (uint64_t) offset_count;
    const uint64_t boundary_base = adjacency_base + (uint64_t) total_adj;
    const uint64_t cost_base = boundary_base + (uint64_t) vertex_count;
    const uint64_t best_cost_base = cost_base + (uint64_t) unique_edge_count;
    const uint64_t best_edge_base = best_cost_base + (uint64_t) face_count;
    const uint64_t face_keep_base = best_edge_base + (uint64_t) face_count;
    const uint64_t counter_base = face_keep_base + (uint64_t) face_count;
    const uint64_t scratch_count = counter_base + 1ull;
    (void) qem_base;
    if (scratch_count > UINT32_MAX) goto cleanup;
    if (!vk_buffer_create(vk, (size_t) scratch_count * sizeof(uint32_t), &scratch)) goto cleanup;

    vkmesh_vk_buffer copy_buffers[4];
    copy_buffers[0] = *scan_in;
    copy_buffers[1] = scratch;
    copy_buffers[2] = counter;
    copy_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    push.aux1 = (uint32_t) offset_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, offset_groups)) goto cleanup;

    copy_buffers[0] = adjacency_buffer;
    copy_buffers[1] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = total_adj;
    push.aux1 = (uint32_t) adjacency_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (total_adj + 127u) / 128u)) goto cleanup;

    copy_buffers[0] = boundary_flags;
    copy_buffers[1] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count;
    push.aux1 = (uint32_t) boundary_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, vertex_groups)) goto cleanup;

    /* Offsets, adjacency, and boundary flags now live in scratch. The QEM,
       cost, winner selection, and collapse kernels no longer reference the
       construction buffers directly. */
    vk_buffer_destroy(vk, &degree_buffer);
    vk_buffer_destroy(vk, &offset_a_buffer);
    vk_buffer_destroy(vk, &offset_b_buffer);
    vk_buffer_destroy(vk, &adjacency_buffer);
    vk_buffer_destroy(vk, &boundary_flags);

    vkmesh_vk_buffer fill_buffers[4];
    fill_buffers[0] = scratch;
    fill_buffers[1] = counter;
    fill_buffers[2] = counter;
    fill_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = 0x7f800000u;
    push.aux1 = (uint32_t) best_cost_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, face_groups)) goto cleanup;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = UINT32_MAX;
    push.aux1 = (uint32_t) best_edge_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, face_groups)) goto cleanup;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = 1u;
    push.aux1 = (uint32_t) face_keep_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, face_groups)) goto cleanup;
    memset(&push, 0, sizeof(push));
    push.n = 1u;
    push.aux0 = 0u;
    push.aux1 = (uint32_t) counter_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, 1u)) goto cleanup;

    vkmesh_vk_buffer qem_buffers[4];
    qem_buffers[0] = dm->faces;
    qem_buffers[1] = dm->vertices;
    qem_buffers[2] = scratch;
    qem_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_QEM, qem_buffers, &push, vertex_groups)) goto cleanup;

    vkmesh_vk_buffer cost_buffers[4];
    cost_buffers[0] = unique_edges;
    cost_buffers[1] = dm->vertices;
    cost_buffers[2] = dm->faces;
    cost_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = unique_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    push.eps = lambda_edge_length;
    push.rel_eps = lambda_skinny;
    uint32_t unique_groups = (unique_edge_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_EDGE_COST, cost_buffers, &push, unique_groups)) goto cleanup;

    vkmesh_vk_buffer propagate_buffers[4];
    propagate_buffers[0] = unique_edges;
    propagate_buffers[1] = counter;
    propagate_buffers[2] = counter;
    propagate_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = unique_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_PROPAGATE_COST, propagate_buffers, &push, unique_groups)) goto cleanup;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_BEST_EDGE, propagate_buffers, &push, unique_groups)) goto cleanup;

    vkmesh_vk_buffer collapse_buffers[4];
    collapse_buffers[0] = unique_edges;
    collapse_buffers[1] = dm->vertices;
    collapse_buffers[2] = dm->faces;
    collapse_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = unique_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    push.eps = threshold;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_COLLAPSE_EDGES, collapse_buffers, &push, unique_groups)) goto cleanup;

    uint32_t collapsed_u32 = ((const uint32_t *) scratch.mapped)[counter_base];
    uint32_t removed_u32 = 0u;
    if (collapsed_u32 > 0u &&
        !compact_device_mesh_from_flags(dm, &scratch, (uint32_t) face_keep_base, &removed_u32)) goto cleanup;
    if (collapsed_edges != NULL) *collapsed_edges = collapsed_u32;
    if (removed_faces != NULL) *removed_faces = removed_u32;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &degree_buffer);
    vk_buffer_destroy(vk, &offset_a_buffer);
    vk_buffer_destroy(vk, &offset_b_buffer);
    vk_buffer_destroy(vk, &adjacency_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &edge_dummy);
    vk_buffer_destroy(vk, &unique_edges);
    vk_buffer_destroy(vk, &boundary_flags);
    vk_buffer_destroy(vk, &counter);
    vk_buffer_destroy(vk, &scratch);
    return ok;
}

static int vkmesh_device_simplify(
    vkmesh_device_mesh * dm,
    int target_faces,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    int max_steps,
    int * total_collapsed,
    int * total_removed) {
    if (dm == NULL || dm->vk == NULL || dm->n_vertices == 0u || dm->n_faces == 0u) {
        return 0;
    }
    if (target_faces <= 0) target_faces = 1;

    uint32_t total_collapsed_u32 = 0u;
    uint32_t total_removed_u32 = 0u;

    int step = 0;
    int consecutive_no_progress = 0;
    while (dm->n_faces > (uint32_t) target_faces && (max_steps <= 0 || step < max_steps)) {
        uint32_t before = dm->n_faces;
        uint32_t collapsed = 0u;
        uint32_t removed = 0u;
        fprintf(stderr,
            "vkmesh: simplify step=%d begin faces=%u target=%d threshold=%.9g\n",
            step + 1,
            dm->n_faces,
            target_faces,
            threshold);
        fflush(stderr);
        if (!vkmesh_device_simplify_step(dm, lambda_edge_length, lambda_skinny, threshold, &collapsed, &removed)) return 0;
        if (total_collapsed_u32 > UINT32_MAX - collapsed || total_removed_u32 > UINT32_MAX - removed) return 0;
        total_collapsed_u32 += collapsed;
        total_removed_u32 += removed;
        ++step;
        fprintf(stderr,
            "vkmesh: simplify step=%d end collapsed=%u removed_faces=%u faces=%u\n",
            step,
            collapsed,
            removed,
            dm->n_faces);
        fflush(stderr);
        if (dm->n_faces <= (uint32_t) target_faces) break;
        if (collapsed == 0u || removed == 0u) {
            ++consecutive_no_progress;
        } else {
            consecutive_no_progress = 0;
        }
        if (consecutive_no_progress >= 16) {
            fprintf(stderr,
                "vkmesh: simplify stopped after %d consecutive no-progress rounds at %u faces\n",
                consecutive_no_progress,
                dm->n_faces);
            break;
        }
        if (collapsed == 0u || removed == 0u || (uint64_t) removed * 100ull < (uint64_t) before) {
            if (!isfinite(threshold) || threshold > FLT_MAX / 10.0f) {
                fprintf(stderr, "vkmesh: simplify stopped because threshold can no longer increase\n");
                break;
            }
            threshold *= 10.0f;
        }
    }

    if (total_collapsed_u32 > (uint32_t) INT_MAX || total_removed_u32 > (uint32_t) INT_MAX) return 0;
    *total_collapsed = (int) total_collapsed_u32;
    *total_removed = (int) total_removed_u32;
    return 1;
}

static int vkmesh_simplify_device_vulkan(
    vkmesh_mesh * mesh,
    int target_faces,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    int max_steps,
    int * total_collapsed,
    int * total_removed) {
    if (mesh == NULL || mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }
    if (target_faces <= 0) target_faces = 1;

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_device_mesh dm;
    memset(&dm, 0, sizeof(dm));
    int ok = 0;

    if (!vkmesh_device_mesh_upload(vk, mesh, &dm)) goto cleanup;
    if (!vkmesh_device_simplify(
            &dm,
            target_faces,
            lambda_edge_length,
            lambda_skinny,
            threshold,
            max_steps,
            total_collapsed,
            total_removed)) {
        goto cleanup;
    }
    if (!vkmesh_device_mesh_download(&dm, mesh)) goto cleanup;
    ok = 1;

cleanup:
    vkmesh_device_mesh_destroy(&dm);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_simplify_step_cpu(
    vkmesh_mesh * mesh,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    int * collapsed_edges,
    int * removed_faces) {
    *collapsed_edges = 0;
    *removed_faces = 0;
    if (mesh->n_faces <= 0) return 1;
    int * v2f_offset = NULL;
    int * v2f = NULL;
    vkmesh_simplify_edge * edges = NULL;
    int64_t edge_count = 0;
    uint8_t * boundary = NULL;
    vkmesh_qem * qems = NULL;
    int * best_edge = NULL;
    uint8_t * selected_vertex = NULL;
    uint8_t * selected_edge = NULL;
    uint8_t * keep_face = NULL;
    int ok = 0;

    if (!build_vertex_face_adjacency(mesh, &v2f_offset, &v2f)) goto cleanup;
    if (!get_unique_simplify_edges(mesh, &edges, &edge_count, &boundary)) goto cleanup;
    if (!build_vertex_qems(mesh, &qems)) goto cleanup;
    for (int64_t e = 0; e < edge_count; ++e) {
        edges[e].cost = simplify_edge_cost(mesh, v2f_offset, v2f, boundary, qems, lambda_edge_length, lambda_skinny, &edges[e]);
    }
    if (!propagate_edge_cost_to_faces(mesh, v2f_offset, v2f, edges, edge_count, &best_edge)) goto cleanup;
    selected_vertex = (uint8_t *) calloc((size_t) mesh->n_vertices, sizeof(uint8_t));
    selected_edge = (uint8_t *) calloc((size_t) edge_count, sizeof(uint8_t));
    keep_face = (uint8_t *) malloc((size_t) mesh->n_faces);
    if (selected_vertex == NULL || selected_edge == NULL || keep_face == NULL) goto cleanup;
    memset(keep_face, 1, (size_t) mesh->n_faces);

    for (int64_t e = 0; e < edge_count; ++e) {
        if (edges[e].cost > (double) threshold) continue;
        int32_t verts[2] = { edges[e].v0, edges[e].v1 };
        int can_collapse = !selected_vertex[verts[0]] && !selected_vertex[verts[1]];
        for (int side = 0; side < 2 && can_collapse; ++side) {
            int32_t v = verts[side];
            for (int i = v2f_offset[v]; i < v2f_offset[v + 1]; ++i) {
                if (best_edge[v2f[i]] != (int) e) {
                    can_collapse = 0;
                    break;
                }
            }
        }
        if (!can_collapse) continue;
        selected_edge[e] = 1u;
        selected_vertex[verts[0]] = 1u;
        selected_vertex[verts[1]] = 1u;
        ++(*collapsed_edges);
    }

    for (int64_t e = 0; e < edge_count; ++e) {
        if (!selected_edge[e]) continue;
        int32_t v0 = edges[e].v0;
        int32_t v1 = edges[e].v1;
        memcpy(mesh->vertices + (size_t) v0 * 3u, edges[e].pos, 3u * sizeof(float));
        for (int i = v2f_offset[v0]; i < v2f_offset[v0 + 1]; ++i) {
            int face_id = v2f[i];
            int32_t * f = mesh->faces + (size_t) face_id * 3u;
            if (face_contains_vertex(f, v1)) keep_face[face_id] = 0u;
        }
        for (int i = v2f_offset[v1]; i < v2f_offset[v1 + 1]; ++i) {
            int face_id = v2f[i];
            int32_t * f = mesh->faces + (size_t) face_id * 3u;
            if (f[0] == v1) f[0] = v0;
            if (f[1] == v1) f[1] = v0;
            if (f[2] == v1) f[2] = v0;
        }
    }
    if (!mesh_remove_faces_by_mask_vulkan(mesh, keep_face, removed_faces) &&
        !mesh_remove_faces_by_mask(mesh, keep_face, removed_faces)) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    free(v2f_offset);
    free(v2f);
    free(edges);
    free(boundary);
    free(qems);
    free(best_edge);
    free(selected_vertex);
    free(selected_edge);
    free(keep_face);
    return ok;
}

static int vkmesh_simplify_step(
    vkmesh_mesh * mesh,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    int * collapsed_edges,
    int * removed_faces) {
    if (vkmesh_simplify_step_vulkan(mesh, lambda_edge_length, lambda_skinny, threshold, collapsed_edges, removed_faces)) {
        return 1;
    }
    return vkmesh_simplify_step_cpu(mesh, lambda_edge_length, lambda_skinny, threshold, collapsed_edges, removed_faces);
}

static int vkmesh_simplify(
    vkmesh_mesh * mesh,
    int target_faces,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    int max_steps,
    int * total_collapsed,
    int * total_removed) {
    *total_collapsed = 0;
    *total_removed = 0;
    if (target_faces <= 0) target_faces = 1;
    if (vkmesh_simplify_device_vulkan(
            mesh,
            target_faces,
            lambda_edge_length,
            lambda_skinny,
            threshold,
            max_steps,
            total_collapsed,
            total_removed)) {
        return 1;
    }

    int step = 0;
    int consecutive_no_progress = 0;
    while (mesh->n_faces > target_faces && (max_steps <= 0 || step < max_steps)) {
        int before = (int) mesh->n_faces;
        int collapsed = 0;
        int removed = 0;
        fprintf(stderr,
            "vkmesh: simplify step=%d begin faces=%" PRId64 " target=%d threshold=%.9g\n",
            step + 1,
            mesh->n_faces,
            target_faces,
            threshold);
        fflush(stderr);
        if (!vkmesh_simplify_step(mesh, lambda_edge_length, lambda_skinny, threshold, &collapsed, &removed)) return 0;
        *total_collapsed += collapsed;
        *total_removed += removed;
        ++step;
        fprintf(stderr,
            "vkmesh: simplify step=%d end collapsed=%d removed_faces=%d faces=%" PRId64 "\n",
            step,
            collapsed,
            removed,
            mesh->n_faces);
        fflush(stderr);
        if (mesh->n_faces <= target_faces) break;
        if (collapsed == 0 || removed == 0) {
            ++consecutive_no_progress;
        } else {
            consecutive_no_progress = 0;
        }
        if (consecutive_no_progress >= 16) {
            fprintf(stderr,
                "vkmesh: simplify stopped after %d consecutive no-progress rounds at %" PRId64 " faces\n",
                consecutive_no_progress,
                mesh->n_faces);
            break;
        }
        if (collapsed == 0 || removed == 0 || (int64_t) removed * 100ll < (int64_t) before) {
            if (!isfinite(threshold) || threshold > FLT_MAX / 10.0f) {
                fprintf(stderr, "vkmesh: simplify stopped because threshold can no longer increase\n");
                break;
            }
            threshold *= 10.0f;
        }
    }
    return 1;
}

static int vkmesh_uv_unwrap(vkmesh_mesh * mesh, int texture_size, int * old_vertices, int * new_vertices, int * chart_count) {
    *old_vertices = (int) mesh->n_vertices;
    *new_vertices = (int) mesh->n_vertices;
    *chart_count = 0;
    if (mesh->n_vertices <= 0 || mesh->n_faces <= 0 ||
        mesh->n_vertices > UINT32_MAX || mesh->n_faces > UINT32_MAX / 3u) {
        return 0;
    }
    uint32_t * indices = (uint32_t *) malloc((size_t) mesh->n_faces * 3u * sizeof(uint32_t));
    if (indices == NULL) return 0;
    for (int64_t i = 0; i < mesh->n_faces * 3; ++i) indices[i] = (uint32_t) mesh->faces[i];

    /* Standalone meshbin UV unwrap uses xatlas directly on the CPU. */
    xatlasSetPrint(NULL, false);
    xatlasAtlas * atlas = xatlasCreate();
    if (atlas == NULL) {
        free(indices);
        return 0;
    }
    xatlasMeshDecl decl;
    xatlasMeshDeclInit(&decl);
    decl.vertexCount = (uint32_t) mesh->n_vertices;
    decl.vertexPositionData = mesh->vertices;
    decl.vertexPositionStride = 3u * sizeof(float);
    decl.indexCount = (uint32_t) mesh->n_faces * 3u;
    decl.indexData = indices;
    decl.indexFormat = XATLAS_INDEX_FORMAT_UINT32;
    xatlasAddMeshError add_error = xatlasAddMesh(atlas, &decl, 1);
    if (add_error != XATLAS_ADD_MESH_ERROR_SUCCESS) {
        fprintf(stderr, "vkmesh: xatlas AddMesh failed: %s\n", xatlasAddMeshErrorString(add_error));
        xatlasDestroy(atlas);
        free(indices);
        return 0;
    }
    xatlasPackOptions pack_options;
    xatlasPackOptionsInit(&pack_options);
    pack_options.resolution = texture_size > 0 ? (uint32_t) texture_size : 1024u;
    pack_options.padding = 4;
    pack_options.createImage = false;
    xatlasGenerate(atlas, NULL, &pack_options);
    if (atlas->meshCount != 1 || atlas->meshes == NULL || atlas->meshes[0].vertexCount == 0) {
        xatlasDestroy(atlas);
        free(indices);
        return 0;
    }
    const xatlasMesh * xm = &atlas->meshes[0];
    float * out_vertices = (float *) malloc((size_t) xm->vertexCount * 3u * sizeof(float));
    float * out_uvs = (float *) malloc((size_t) xm->vertexCount * 2u * sizeof(float));
    int32_t * out_faces = (int32_t *) malloc((size_t) xm->indexCount * sizeof(int32_t));
    if (out_vertices == NULL || out_uvs == NULL || out_faces == NULL) {
        free(out_vertices); free(out_uvs); free(out_faces);
        xatlasDestroy(atlas);
        free(indices);
        return 0;
    }
    float atlas_w = atlas->width > 0 ? (float) atlas->width : (float) pack_options.resolution;
    float atlas_h = atlas->height > 0 ? (float) atlas->height : (float) pack_options.resolution;
    for (uint32_t i = 0; i < xm->vertexCount; ++i) {
        const xatlasVertex * v = &xm->vertexArray[i];
        uint32_t src = v->xref;
        memcpy(out_vertices + (size_t) i * 3u, mesh->vertices + (size_t) src * 3u, 3u * sizeof(float));
        out_uvs[(size_t) i * 2u + 0u] = clamp01(v->uv[0] / atlas_w);
        out_uvs[(size_t) i * 2u + 1u] = clamp01(1.0f - v->uv[1] / atlas_h);
    }
    for (uint32_t i = 0; i < xm->indexCount; ++i) out_faces[i] = (int32_t) xm->indexArray[i];

    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = out_vertices;
    mesh->uvs = out_uvs;
    mesh->faces = out_faces;
    mesh->n_vertices = (int64_t) xm->vertexCount;
    mesh->n_faces = (int64_t) xm->indexCount / 3;
    mesh->vertex_capacity = mesh->n_vertices;
    mesh->face_capacity = mesh->n_faces;
    mesh->has_uvs = 1;
    *new_vertices = (int) mesh->n_vertices;
    *chart_count = (int) atlas->chartCount;
    xatlasDestroy(atlas);
    free(indices);
    return 1;
}

static void bvh_bounds_init(float bmin[3], float bmax[3]) {
    bmin[0] = bmin[1] = bmin[2] = FLT_MAX;
    bmax[0] = bmax[1] = bmax[2] = -FLT_MAX;
}

static void bvh_bounds_add(float bmin[3], float bmax[3], const float pmin[3], const float pmax[3]) {
    for (int i = 0; i < 3; ++i) {
        if (pmin[i] < bmin[i]) bmin[i] = pmin[i];
        if (pmax[i] > bmax[i]) bmax[i] = pmax[i];
    }
}

static uint32_t bvh_build_recursive(
    vkmesh_bvh_tri * tris,
    int start,
    int count,
    vkmesh_bvh_node * nodes,
    uint32_t * node_count) {
    uint32_t node_id = (*node_count)++;
    vkmesh_bvh_node * node = &nodes[node_id];
    bvh_bounds_init(node->bmin, node->bmax);
    for (int i = start; i < start + count; ++i) {
        bvh_bounds_add(node->bmin, node->bmax, tris[i].bmin, tris[i].bmax);
    }
    if (count <= 4) {
        node->left = (uint32_t) start;
        node->meta = 0x80000000u | (uint32_t) count;
        return node_id;
    }

    float cmin[3];
    float cmax[3];
    bvh_bounds_init(cmin, cmax);
    for (int i = start; i < start + count; ++i) {
        bvh_bounds_add(cmin, cmax, tris[i].centroid, tris[i].centroid);
    }
    float ex[3] = { cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2] };
    int axis = 0;
    if (ex[1] > ex[axis]) axis = 1;
    if (ex[2] > ex[axis]) axis = 2;
    float split = 0.5f * (cmin[axis] + cmax[axis]);
    int mid = start;
    for (int i = start; i < start + count; ++i) {
        if (tris[i].centroid[axis] < split) {
            vkmesh_bvh_tri tmp = tris[mid];
            tris[mid] = tris[i];
            tris[i] = tmp;
            ++mid;
        }
    }
    int left_count = mid - start;
    if (left_count <= 0 || left_count >= count) left_count = count / 2;
    uint32_t left = bvh_build_recursive(tris, start, left_count, nodes, node_count);
    uint32_t right = bvh_build_recursive(tris, start + left_count, count - left_count, nodes, node_count);
    node->left = left;
    node->meta = right;
    return node_id;
}

static int vkmesh_build_bvh(
    const vkmesh_mesh * mesh,
    vkmesh_bvh_node ** nodes_out,
    uint32_t * node_count_out,
    uint32_t ** tri_indices_out) {
    *nodes_out = NULL;
    *node_count_out = 0;
    *tri_indices_out = NULL;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 2u) return 0;
    vkmesh_bvh_tri * tris = (vkmesh_bvh_tri *) malloc((size_t) mesh->n_faces * sizeof(*tris));
    vkmesh_bvh_node * nodes = (vkmesh_bvh_node *) malloc((size_t) mesh->n_faces * 2u * sizeof(*nodes));
    uint32_t * tri_indices = (uint32_t *) malloc((size_t) mesh->n_faces * sizeof(uint32_t));
    if (tris == NULL || nodes == NULL || tri_indices == NULL) {
        free(tris); free(nodes); free(tri_indices);
        return 0;
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        bvh_bounds_init(tris[i].bmin, tris[i].bmax);
        float sum[3] = { 0.0f, 0.0f, 0.0f };
        for (int k = 0; k < 3; ++k) {
            const float * v = mesh->vertices + (size_t) f[k] * 3u;
            bvh_bounds_add(tris[i].bmin, tris[i].bmax, v, v);
            sum[0] += v[0];
            sum[1] += v[1];
            sum[2] += v[2];
        }
        tris[i].centroid[0] = sum[0] / 3.0f;
        tris[i].centroid[1] = sum[1] / 3.0f;
        tris[i].centroid[2] = sum[2] / 3.0f;
        tris[i].face = (uint32_t) i;
    }
    uint32_t node_count = 0;
    (void) bvh_build_recursive(tris, 0, (int) mesh->n_faces, nodes, &node_count);
    for (int64_t i = 0; i < mesh->n_faces; ++i) tri_indices[i] = tris[i].face;
    free(tris);
    vkmesh_bvh_node * compact_nodes =
        (vkmesh_bvh_node *) realloc(nodes, (size_t) node_count * sizeof(*nodes));
    if (compact_nodes != NULL) nodes = compact_nodes;
    *nodes_out = nodes;
    *node_count_out = node_count;
    *tri_indices_out = tri_indices;
    return 1;
}

static uint32_t u32_from_float(float value) {
    union {
        uint32_t u;
        float f;
    } v;
    v.f = value;
    return v.u;
}

static int load_points(const char * path, float ** points_out, int64_t * count_out) {
    *points_out = NULL;
    *count_out = 0;
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "vkmesh: failed to open points file %s\n", path);
        return 0;
    }
    float * points = NULL;
    int64_t count = 0;
    int64_t capacity = 0;
    char line[4096];
    int64_t line_no = 0;
    int ok = 1;
    while (fgets(line, sizeof(line), f) != NULL) {
        ++line_no;
        char * p = line;
        while (*p != '\0' && isspace((unsigned char) *p)) ++p;
        if (*p == '\0' || *p == '#') continue;
        float x, y, z;
        if (sscanf(p, "%f %f %f", &x, &y, &z) != 3) {
            fprintf(stderr, "vkmesh: bad point at %s:%" PRId64 "\n", path, line_no);
            ok = 0;
            break;
        }
        if (count == capacity) {
            int64_t next_cap = capacity > 0 ? capacity * 2 : 1024;
            float * next = (float *) realloc(points, (size_t) next_cap * 3u * sizeof(float));
            if (next == NULL) {
                ok = 0;
                break;
            }
            points = next;
            capacity = next_cap;
        }
        points[(size_t) count * 3u + 0u] = x;
        points[(size_t) count * 3u + 1u] = y;
        points[(size_t) count * 3u + 2u] = z;
        ++count;
    }
    fclose(f);
    if (!ok || count == 0) {
        free(points);
        if (ok) fprintf(stderr, "vkmesh: points file %s is empty\n", path);
        return 0;
    }
    *points_out = points;
    *count_out = count;
    return 1;
}

static float float_from_u32(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = bits;
    return v.f;
}

static void vkmesh_distance_query_destroy(vkmesh_distance_query * query) {
    if (query == NULL) return;
    if (query->vk != NULL) {
        for (uint32_t i = 0; i < 4; ++i) vk_buffer_destroy(query->vk, &query->buffers[i]);
    }
    memset(query, 0, sizeof(*query));
}

/* The unsigned-distance shader walks the whole BVH once per query point, so a
   batch sized only against memory (256 MiB of scratch = 13.4M points) becomes a
   single GPU job lasting minutes on a slow device. That overruns the kernel's
   compute-ring lockup timeout and wedges the GPU, taking the display with it on
   a card that also drives the desktop. Bound the batch by point count as well;
   the work is unchanged, it is just spread over more submissions. */
#define VKMESH_DEFAULT_MAX_BATCH_POINTS (1u << 20)

static size_t vkmesh_resolve_max_batch_points(void) {
    const char * env = getenv("TRELLIS_VKMESH_MAX_BATCH_POINTS");
    if (env != NULL && env[0] != '\0') {
        int parsed = 0;
        if (parse_int_arg(env, &parsed) && parsed > 0) {
            return (size_t) parsed;
        }
        fprintf(stderr,
            "vkmesh: ignoring invalid TRELLIS_VKMESH_MAX_BATCH_POINTS='%s'\n",
            env);
    }
    return VKMESH_DEFAULT_MAX_BATCH_POINTS;
}

static int vkmesh_distance_query_init(
    vkmesh_distance_query * query,
    vkmesh_vk * vk,
    const vkmesh_mesh * mesh,
    const vkmesh_bvh_node * nodes,
    uint32_t node_count,
    const uint32_t * tri_indices,
    int64_t max_point_count) {
    if (query == NULL || vk == NULL || mesh == NULL || nodes == NULL || tri_indices == NULL ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX ||
        node_count == 0 || max_point_count <= 0) {
        return 0;
    }
    memset(query, 0, sizeof(*query));
    query->vk = vk;
    query->face_count = (uint32_t) mesh->n_faces;
    query->node_count = node_count;

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    if ((size_t) node_count > SIZE_MAX / (8u * sizeof(uint32_t)) ||
        (size_t) mesh->n_faces > SIZE_MAX / sizeof(uint32_t)) {
        return 0;
    }
    const size_t node_words = (size_t) node_count * 8u;
    const size_t tri_words = (size_t) mesh->n_faces;
    if (node_words > UINT32_MAX || tri_words > UINT32_MAX - node_words) {
        fprintf(stderr, "vkmesh: BVH is too large for 32-bit shader addressing\n");
        return 0;
    }
    const size_t aux_bytes = (node_words + tri_words) * sizeof(uint32_t);

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(vk->physical_device, &properties);
    const VkDeviceSize max_range = properties.limits.maxStorageBufferRange;
    if ((VkDeviceSize) faces_bytes > max_range || (VkDeviceSize) vertices_bytes > max_range ||
        (VkDeviceSize) aux_bytes > max_range) {
        fprintf(stderr,
            "vkmesh: source geometry exceeds maxStorageBufferRange; faces=%.1f MiB vertices=%.1f MiB bvh=%.1f MiB limit=%.1f MiB\n",
            (double) faces_bytes / (1024.0 * 1024.0),
            (double) vertices_bytes / (1024.0 * 1024.0),
            (double) aux_bytes / (1024.0 * 1024.0),
            (double) max_range / (1024.0 * 1024.0));
        return 0;
    }

    if (!vk_buffer_create(vk, faces_bytes, &query->buffers[0]) ||
        !vk_buffer_create(vk, vertices_bytes, &query->buffers[1]) ||
        !vk_buffer_create(vk, aux_bytes, &query->buffers[3])) {
        goto fail;
    }
    memcpy(query->buffers[0].mapped, mesh->faces, faces_bytes);
    memcpy(query->buffers[1].mapped, mesh->vertices, vertices_bytes);
    uint32_t * aux = (uint32_t *) query->buffers[3].mapped;
    for (uint32_t i = 0; i < node_count; ++i) {
        const size_t base = (size_t) i * 8u;
        aux[base + 0u] = u32_from_float(nodes[i].bmin[0]);
        aux[base + 1u] = u32_from_float(nodes[i].bmin[1]);
        aux[base + 2u] = u32_from_float(nodes[i].bmin[2]);
        aux[base + 3u] = nodes[i].left;
        aux[base + 4u] = u32_from_float(nodes[i].bmax[0]);
        aux[base + 5u] = u32_from_float(nodes[i].bmax[1]);
        aux[base + 6u] = u32_from_float(nodes[i].bmax[2]);
        aux[base + 7u] = nodes[i].meta;
    }
    memcpy(aux + node_words, tri_indices, tri_words * sizeof(uint32_t));

    size_t available = vk->workspace_budget_bytes > vk->workspace_current_bytes ?
        vk->workspace_budget_bytes - vk->workspace_current_bytes : 0;
    size_t capacity = available / (5u * sizeof(uint32_t));
    const size_t scratch_cap_bytes = 256u * 1024u * 1024u;
    if (capacity > scratch_cap_bytes / (5u * sizeof(uint32_t))) {
        capacity = scratch_cap_bytes / (5u * sizeof(uint32_t));
    }
    size_t max_range_capacity = (size_t) (max_range / (5u * sizeof(uint32_t)));
    uint64_t max_dispatch_points = (uint64_t) properties.limits.maxComputeWorkGroupCount[0] * 128u;
    if (capacity > max_range_capacity) capacity = max_range_capacity;
    if ((uint64_t) capacity > max_dispatch_points) capacity = (size_t) max_dispatch_points;
    if (capacity > UINT32_MAX / 5u) capacity = UINT32_MAX / 5u;
    if ((uint64_t) capacity > (uint64_t) max_point_count) capacity = (size_t) max_point_count;
    const size_t max_batch_points = vkmesh_resolve_max_batch_points();
    if (capacity > max_batch_points) capacity = max_batch_points;
    if (capacity > 128u) capacity &= ~(size_t) 127u;
    if (capacity == 0) {
        fprintf(stderr, "vkmesh: GPU workspace budget leaves no room for distance-query batches\n");
        goto fail;
    }

    while (capacity > 0) {
        if (vk_buffer_create(vk, capacity * 5u * sizeof(uint32_t), &query->buffers[2])) break;
        if (capacity <= 128u) goto fail;
        capacity = (capacity / 2u) & ~(size_t) 127u;
    }
    query->point_capacity = (uint32_t) capacity;
    query->output_word_offset = query->point_capacity * 3u;
    fprintf(stderr,
        "vkmesh: distance workspace static=%.1f MiB batch=%.1f MiB max_batch_points=%u\n",
        (double) (faces_bytes + vertices_bytes + aux_bytes) / (1024.0 * 1024.0),
        (double) query->buffers[2].allocation_bytes / (1024.0 * 1024.0),
        query->point_capacity);
    return 1;

fail:
    vkmesh_distance_query_destroy(query);
    return 0;
}

static int vkmesh_distance_query_submit_loaded(
    vkmesh_distance_query * query,
    uint32_t count,
    float * distances,
    uint32_t * face_ids) {
    if (query == NULL || query->vk == NULL || count == 0 || count > query->point_capacity) {
        return 0;
    }
    uint32_t * io = (uint32_t *) query->buffers[2].mapped;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = count;
    push.aux0 = query->face_count;
    push.aux1 = query->node_count;
    push.aux2 = query->output_word_offset;
    const uint32_t groups = (count + 127u) / 128u;
    if (!vkmesh_dispatch(query->vk, VKMESH_PIPE_UNSIGNED_DISTANCE, query->buffers, &push, groups)) return 0;

    const uint32_t * raw = io + query->output_word_offset;
    for (uint32_t i = 0; i < count; ++i) {
        if (distances != NULL) distances[i] = float_from_u32(raw[(size_t) i * 2u + 0u]);
        if (face_ids != NULL) face_ids[i] = raw[(size_t) i * 2u + 1u];
    }
    query->batch_count += 1u;
    return 1;
}

static int vkmesh_distance_query_dispatch(
    vkmesh_distance_query * query,
    const float * points,
    uint32_t count,
    float * distances,
    uint32_t * face_ids) {
    if (query == NULL || points == NULL || count == 0 || count > query->point_capacity) return 0;
    uint32_t * io = (uint32_t *) query->buffers[2].mapped;
    for (uint32_t i = 0; i < count * 3u; ++i) io[i] = u32_from_float(points[i]);
    return vkmesh_distance_query_submit_loaded(query, count, distances, face_ids);
}

static int vkmesh_distance_query_points(
    vkmesh_distance_query * query,
    const float * points,
    int64_t point_count,
    float * distances,
    uint32_t * face_ids) {
    if (query == NULL || points == NULL || point_count <= 0 ||
        (distances == NULL && face_ids == NULL)) {
        return 0;
    }
    int64_t offset = 0;
    while (offset < point_count) {
        uint32_t batch = query->point_capacity;
        if ((int64_t) batch > point_count - offset) batch = (uint32_t) (point_count - offset);
        if (!vkmesh_distance_query_dispatch(
                query,
                points + (size_t) offset * 3u,
                batch,
                distances != NULL ? distances + offset : NULL,
                face_ids != NULL ? face_ids + offset : NULL)) {
            return 0;
        }
        offset += batch;
    }
    return 1;
}

static int vkmesh_unsigned_distance_vulkan_with_bvh(
    const vkmesh_mesh * mesh,
    const vkmesh_bvh_node * nodes,
    uint32_t node_count,
    const uint32_t * tri_indices,
    const float * points,
    int64_t point_count,
    float ** distances_out,
    uint32_t ** face_ids_out) {
    if (distances_out == NULL) return 0;
    *distances_out = NULL;
    if (face_ids_out != NULL) *face_ids_out = NULL;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX || point_count <= 0 || point_count > UINT32_MAX ||
        nodes == NULL || node_count == 0 || tri_indices == NULL || points == NULL) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }
    vkmesh_distance_query query;
    memset(&query, 0, sizeof(query));
    int ok = 0;
    if (!vkmesh_distance_query_init(&query, vk, mesh, nodes, node_count, tri_indices, point_count)) {
        goto cleanup;
    }

    float * distances = (float *) malloc((size_t) point_count * sizeof(float));
    uint32_t * face_ids = face_ids_out != NULL ? (uint32_t *) malloc((size_t) point_count * sizeof(uint32_t)) : NULL;
    if (distances == NULL || (face_ids_out != NULL && face_ids == NULL)) {
        free(distances);
        free(face_ids);
        goto cleanup;
    }
    if (!vkmesh_distance_query_points(&query, points, point_count, distances, face_ids)) {
        free(distances);
        free(face_ids);
        goto cleanup;
    }
    *distances_out = distances;
    if (face_ids_out != NULL) *face_ids_out = face_ids;
    ok = 1;

cleanup:
    if (query.batch_count > 0u) {
        fprintf(stderr, "vkmesh: distance query batches=%u\n", query.batch_count);
    }
    vkmesh_distance_query_destroy(&query);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_unsigned_distance_vulkan(
    const vkmesh_mesh * mesh,
    const float * points,
    int64_t point_count,
    float ** distances_out,
    uint32_t ** face_ids_out) {
    if (distances_out != NULL) *distances_out = NULL;
    if (face_ids_out != NULL) *face_ids_out = NULL;
    if (mesh == NULL || mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX ||
        point_count <= 0 || point_count > UINT32_MAX || points == NULL) {
        return 0;
    }

    vkmesh_bvh_node * nodes = NULL;
    uint32_t node_count = 0;
    uint32_t * tri_indices = NULL;
    int ok = 0;
    if (vkmesh_build_bvh(mesh, &nodes, &node_count, &tri_indices)) {
        ok = vkmesh_unsigned_distance_vulkan_with_bvh(
            mesh,
            nodes,
            node_count,
            tri_indices,
            points,
            point_count,
            distances_out,
            face_ids_out);
    }
    free(nodes);
    free(tri_indices);
    return ok;
}

static int write_distances(const char * path, const float * points, int64_t point_count, const float * distances, const uint32_t * face_ids) {
    FILE * f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "vkmesh: failed to open %s for writing\n", path);
        return 0;
    }
    fprintf(f, "# x y z unsigned_distance face_id\n");
    for (int64_t i = 0; i < point_count; ++i) {
        const float * p = points + (size_t) i * 3u;
        fprintf(f, "%.9g %.9g %.9g %.9g %u\n", p[0], p[1], p[2], distances[i], face_ids[i]);
    }
    fclose(f);
    return 1;
}

static uint64_t vkmesh_hash_u64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static int vkmesh_next_pow2_size(size_t need, size_t * out) {
    if (out == NULL) return 0;
    size_t cap = 16u;
    while (cap < need) {
        if (cap > SIZE_MAX / 2u) return 0;
        cap *= 2u;
    }
    *out = cap;
    return 1;
}

static int vkmesh_hash_init(vkmesh_u64_u32_hash * map, size_t expected_entries) {
    if (map == NULL) return 0;
    memset(map, 0, sizeof(*map));
    size_t cap = 0;
    if (expected_entries > (SIZE_MAX - 16u) / 2u) return 0;
    if (!vkmesh_next_pow2_size(expected_entries * 2u + 16u, &cap)) return 0;
    if (cap > SIZE_MAX / sizeof(uint64_t) || cap > SIZE_MAX / sizeof(uint32_t)) return 0;
    map->keys = (uint64_t *) malloc(cap * sizeof(uint64_t));
    map->vals = (uint32_t *) malloc(cap * sizeof(uint32_t));
    if (map->keys == NULL || map->vals == NULL) {
        free(map->keys);
        free(map->vals);
        memset(map, 0, sizeof(*map));
        return 0;
    }
    for (size_t i = 0; i < cap; ++i) map->keys[i] = UINT64_MAX;
    map->capacity = cap;
    return 1;
}

static void vkmesh_hash_destroy(vkmesh_u64_u32_hash * map) {
    if (map == NULL) return;
    free(map->keys);
    free(map->vals);
    memset(map, 0, sizeof(*map));
}

static uint32_t vkmesh_hash_lookup(const vkmesh_u64_u32_hash * map, uint64_t key);

static int vkmesh_hash_insert_no_grow(vkmesh_u64_u32_hash * map, uint64_t key, uint32_t value) {
    size_t mask = map->capacity - 1u;
    size_t slot = (size_t) vkmesh_hash_u64(key) & mask;
    for (size_t i = 0; i < map->capacity; ++i) {
        uint64_t prev = map->keys[slot];
        if (prev == key) {
            map->vals[slot] = value;
            return 1;
        }
        if (prev == UINT64_MAX) {
            map->keys[slot] = key;
            map->vals[slot] = value;
            map->count += 1u;
            return 1;
        }
        slot = (slot + 1u) & mask;
    }
    return 0;
}

static int vkmesh_hash_grow(vkmesh_u64_u32_hash * map) {
    if (map == NULL || map->capacity == 0 || map->capacity > SIZE_MAX / 2u) return 0;
    vkmesh_u64_u32_hash grown;
    memset(&grown, 0, sizeof(grown));
    const size_t new_capacity = map->capacity * 2u;
    if (new_capacity > SIZE_MAX / sizeof(uint64_t) || new_capacity > SIZE_MAX / sizeof(uint32_t)) return 0;
    grown.keys = (uint64_t *) malloc(new_capacity * sizeof(uint64_t));
    grown.vals = (uint32_t *) malloc(new_capacity * sizeof(uint32_t));
    if (grown.keys == NULL || grown.vals == NULL) {
        free(grown.keys);
        free(grown.vals);
        return 0;
    }
    for (size_t i = 0; i < new_capacity; ++i) grown.keys[i] = UINT64_MAX;
    grown.capacity = new_capacity;
    for (size_t i = 0; i < map->capacity; ++i) {
        if (map->keys[i] != UINT64_MAX &&
            !vkmesh_hash_insert_no_grow(&grown, map->keys[i], map->vals[i])) {
            vkmesh_hash_destroy(&grown);
            return 0;
        }
    }
    free(map->keys);
    free(map->vals);
    *map = grown;
    return 1;
}

static int vkmesh_hash_insert(vkmesh_u64_u32_hash * map, uint64_t key, uint32_t value) {
    if (map == NULL || map->keys == NULL || map->capacity == 0 || key == UINT64_MAX) return 0;
    if (vkmesh_hash_lookup(map, key) != UINT32_MAX) {
        return vkmesh_hash_insert_no_grow(map, key, value);
    }
    if (map->count + 1u > map->capacity - map->capacity / 4u && !vkmesh_hash_grow(map)) return 0;
    return vkmesh_hash_insert_no_grow(map, key, value);
}

static uint32_t vkmesh_hash_lookup(const vkmesh_u64_u32_hash * map, uint64_t key) {
    if (map == NULL || map->keys == NULL || map->capacity == 0 || key == UINT64_MAX) return UINT32_MAX;
    size_t mask = map->capacity - 1u;
    size_t slot = (size_t) vkmesh_hash_u64(key) & mask;
    for (size_t i = 0; i < map->capacity; ++i) {
        uint64_t prev = map->keys[slot];
        if (prev == UINT64_MAX) return UINT32_MAX;
        if (prev == key) return map->vals[slot];
        slot = (slot + 1u) & mask;
    }
    return UINT32_MAX;
}

static uint64_t vkmesh_remesh_coord_key(int32_t x, int32_t y, int32_t z, int dim) {
    if (dim <= 0 || x < 0 || y < 0 || z < 0 || x >= dim || y >= dim || z >= dim) return UINT64_MAX;
    uint64_t d = (uint64_t) dim;
    return ((uint64_t) x * d + (uint64_t) y) * d + (uint64_t) z;
}

static int vkmesh_remesh_coords_reserve(vkmesh_remesh_coord ** coords, int64_t * capacity, int64_t need) {
    if (coords == NULL || capacity == NULL || need < 0) return 0;
    if (need <= *capacity) return 1;
    int64_t cap = *capacity > 0 ? *capacity : 1024;
    while (cap < need) {
        if (cap > INT64_MAX / 2) return 0;
        cap *= 2;
    }
    if ((uint64_t) cap > (uint64_t) SIZE_MAX / sizeof(vkmesh_remesh_coord)) return 0;
    vkmesh_remesh_coord * next = (vkmesh_remesh_coord *) realloc(*coords, (size_t) cap * sizeof(vkmesh_remesh_coord));
    if (next == NULL) return 0;
    *coords = next;
    *capacity = cap;
    return 1;
}

static int vkmesh_remesh_coords_append(
    vkmesh_remesh_coord ** coords,
    int64_t * count,
    int64_t * capacity,
    int32_t x,
    int32_t y,
    int32_t z) {
    if (!vkmesh_remesh_coords_reserve(coords, capacity, *count + 1)) return 0;
    (*coords)[*count].x = x;
    (*coords)[*count].y = y;
    (*coords)[*count].z = z;
    *count += 1;
    return 1;
}

static int vkmesh_mesh_bounds(const vkmesh_mesh * mesh, float bmin[3], float bmax[3]) {
    if (mesh == NULL || mesh->vertices == NULL || mesh->n_vertices <= 0 || bmin == NULL || bmax == NULL) return 0;
    bmin[0] = bmax[0] = mesh->vertices[0];
    bmin[1] = bmax[1] = mesh->vertices[1];
    bmin[2] = bmax[2] = mesh->vertices[2];
    for (int64_t i = 1; i < mesh->n_vertices; ++i) {
        const float * v = mesh->vertices + (size_t) i * 3u;
        for (int k = 0; k < 3; ++k) {
            if (v[k] < bmin[k]) bmin[k] = v[k];
            if (v[k] > bmax[k]) bmax[k] = v[k];
        }
    }
    return 1;
}

static void vkmesh_distance_query_load_coords(
    vkmesh_distance_query * query,
    const vkmesh_remesh_coord * coords,
    int64_t offset,
    uint32_t count,
    int grid_resolution,
    const float center[3],
    float scale,
    int centers) {
    uint32_t * io = (uint32_t *) query->buffers[2].mapped;
    const float inv_res = 1.0f / (float) grid_resolution;
    const float point_offset = centers ? 0.5f : 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        const vkmesh_remesh_coord * coord = &coords[offset + i];
        io[(size_t) i * 3u + 0u] = u32_from_float(
            (((float) coord->x + point_offset) * inv_res - 0.5f) * scale + center[0]);
        io[(size_t) i * 3u + 1u] = u32_from_float(
            (((float) coord->y + point_offset) * inv_res - 0.5f) * scale + center[1]);
        io[(size_t) i * 3u + 2u] = u32_from_float(
            (((float) coord->z + point_offset) * inv_res - 0.5f) * scale + center[2]);
    }
}

static int vkmesh_distance_query_coord_values(
    vkmesh_distance_query * query,
    const vkmesh_remesh_coord * coords,
    int64_t count,
    int grid_resolution,
    const float center[3],
    float scale,
    int centers,
    float * values) {
    if (query == NULL || coords == NULL || count <= 0 || grid_resolution <= 0 || values == NULL) return 0;
    int64_t offset = 0;
    while (offset < count) {
        uint32_t batch = query->point_capacity;
        if ((int64_t) batch > count - offset) batch = (uint32_t) (count - offset);
        vkmesh_distance_query_load_coords(
            query, coords, offset, batch, grid_resolution, center, scale, centers);
        if (!vkmesh_distance_query_submit_loaded(query, batch, values + offset, NULL)) return 0;
        offset += batch;
    }
    return 1;
}

static int vkmesh_distance_query_filter_coords(
    vkmesh_distance_query * query,
    vkmesh_remesh_coord * coords,
    int64_t count,
    int grid_resolution,
    const float center[3],
    float scale,
    float eps,
    float threshold,
    int64_t * keep_count_out) {
    if (query == NULL || coords == NULL || count <= 0 || grid_resolution <= 0 || keep_count_out == NULL) return 0;
    int64_t offset = 0;
    int64_t keep_count = 0;
    while (offset < count) {
        uint32_t batch = query->point_capacity;
        if ((int64_t) batch > count - offset) batch = (uint32_t) (count - offset);
        vkmesh_distance_query_load_coords(
            query, coords, offset, batch, grid_resolution, center, scale, 1);
        if (!vkmesh_distance_query_submit_loaded(query, batch, NULL, NULL)) return 0;
        const uint32_t * raw =
            (const uint32_t *) query->buffers[2].mapped + query->output_word_offset;
        for (uint32_t i = 0; i < batch; ++i) {
            float distance = float_from_u32(raw[(size_t) i * 2u]);
            if (fabsf(distance - eps) < threshold) {
                coords[keep_count++] = coords[offset + i];
            }
        }
        offset += batch;
    }
    *keep_count_out = keep_count;
    return 1;
}

static int vkmesh_remesh_refine_sparse_grid(
    vkmesh_distance_query * query,
    int resolution,
    const float center[3],
    float scale,
    float band,
    vkmesh_remesh_coord ** coords_out,
    int64_t * count_out) {
    *coords_out = NULL;
    *count_out = 0;
    if (resolution <= 0 || band <= 0.0f) return 0;
    int base_resolution = resolution;
    while (base_resolution > 32) {
        if ((base_resolution & 1) != 0) return 0;
        base_resolution /= 2;
    }
    if (base_resolution <= 0) return 0;

    vkmesh_remesh_coord * coords = NULL;
    int64_t coord_count = 0;
    int64_t coord_capacity = 0;
    int64_t base_total = (int64_t) base_resolution * (int64_t) base_resolution * (int64_t) base_resolution;
    if (!vkmesh_remesh_coords_reserve(&coords, &coord_capacity, base_total)) return 0;
    for (int32_t x = 0; x < base_resolution; ++x) {
        for (int32_t y = 0; y < base_resolution; ++y) {
            for (int32_t z = 0; z < base_resolution; ++z) {
                coords[coord_count++] = (vkmesh_remesh_coord) { x, y, z };
            }
        }
    }

    const vkmesh_remesh_coord child_offsets[8] = {
        { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 },
        { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 },
    };
    const float eps = band * scale / (float) resolution;
    int ok = 0;
    int level = 0;
    while (1) {
        const float cell_size = scale / (float) base_resolution;
        int64_t keep_count = 0;
        if (!vkmesh_distance_query_filter_coords(
                query,
                coords,
                coord_count,
                base_resolution,
                center,
                scale,
                eps,
                0.87f * cell_size,
                &keep_count)) {
            goto cleanup;
        }
        fprintf(stderr,
            "vkmesh: remesh sparse_grid level=%d resolution=%d candidates=%" PRId64 " kept=%" PRId64 "\n",
            level,
            base_resolution,
            coord_count,
            keep_count);
        coord_count = keep_count;
        if (coord_count <= 0) goto cleanup;
        if (base_resolution >= resolution) break;

        if (coord_count > INT64_MAX / 8) goto cleanup;
        int64_t next_count = coord_count * 8;
        vkmesh_remesh_coord * next = NULL;
        int64_t next_capacity = 0;
        if (!vkmesh_remesh_coords_reserve(&next, &next_capacity, next_count)) {
            free(next);
            goto cleanup;
        }
        int64_t dst = 0;
        for (int64_t i = 0; i < coord_count; ++i) {
            int32_t bx = coords[i].x * 2;
            int32_t by = coords[i].y * 2;
            int32_t bz = coords[i].z * 2;
            for (int k = 0; k < 8; ++k) {
                next[dst++] = (vkmesh_remesh_coord) {
                    bx + child_offsets[k].x,
                    by + child_offsets[k].y,
                    bz + child_offsets[k].z,
                };
            }
        }
        free(coords);
        coords = next;
        coord_capacity = next_capacity;
        coord_count = next_count;
        base_resolution *= 2;
        ++level;
    }

    vkmesh_remesh_coord * compact_coords =
        (vkmesh_remesh_coord *) realloc(coords, (size_t) coord_count * sizeof(*coords));
    if (compact_coords != NULL) coords = compact_coords;
    *coords_out = coords;
    *count_out = coord_count;
    coords = NULL;
    ok = 1;

cleanup:
    free(coords);
    return ok;
}

static int vkmesh_remesh_build_active_hash(
    const vkmesh_remesh_coord * coords,
    int64_t coord_count,
    int resolution,
    vkmesh_u64_u32_hash * hash_out) {
    if (coord_count <= 0 || coord_count > UINT32_MAX) return 0;
    if (!vkmesh_hash_init(hash_out, (size_t) coord_count)) return 0;
    for (int64_t i = 0; i < coord_count; ++i) {
        uint64_t key = vkmesh_remesh_coord_key(coords[i].x, coords[i].y, coords[i].z, resolution);
        if (!vkmesh_hash_insert(hash_out, key, (uint32_t) i)) return 0;
    }
    return 1;
}

static int vkmesh_remesh_collect_grid_vertices(
    const vkmesh_remesh_coord * coords,
    int64_t coord_count,
    int resolution,
    vkmesh_remesh_coord ** grid_out,
    int64_t * grid_count_out,
    vkmesh_u64_u32_hash * vert_hash_out) {
    *grid_out = NULL;
    *grid_count_out = 0;
    if (coord_count <= 0 || coord_count > UINT32_MAX) return 0;
    if ((uint64_t) coord_count > (uint64_t) SIZE_MAX) return 0;
    /* Start near the observed unique-corner count and grow only for unusually sparse cells. */
    if (!vkmesh_hash_init(vert_hash_out, (size_t) coord_count)) return 0;

    vkmesh_remesh_coord * grid = NULL;
    int64_t grid_count = 0;
    int64_t grid_capacity = 0;
    int ok = 0;
    for (int64_t i = 0; i < coord_count; ++i) {
        for (int dx = 0; dx <= 1; ++dx) {
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dz = 0; dz <= 1; ++dz) {
                    int32_t x = coords[i].x + dx;
                    int32_t y = coords[i].y + dy;
                    int32_t z = coords[i].z + dz;
                    uint64_t key = vkmesh_remesh_coord_key(x, y, z, resolution + 1);
                    if (vkmesh_hash_lookup(vert_hash_out, key) != UINT32_MAX) continue;
                    if (grid_count >= UINT32_MAX) goto cleanup;
                    if (!vkmesh_remesh_coords_append(&grid, &grid_count, &grid_capacity, x, y, z) ||
                        !vkmesh_hash_insert(vert_hash_out, key, (uint32_t) (grid_count - 1))) {
                        goto cleanup;
                    }
                }
            }
        }
    }
    vkmesh_remesh_coord * compact_grid =
        (vkmesh_remesh_coord *) realloc(grid, (size_t) grid_count * sizeof(*grid));
    if (compact_grid != NULL) grid = compact_grid;
    *grid_out = grid;
    *grid_count_out = grid_count;
    grid = NULL;
    ok = 1;

cleanup:
    free(grid);
    if (!ok) vkmesh_hash_destroy(vert_hash_out);
    return ok;
}

static float vkmesh_remesh_grid_value(
    const vkmesh_u64_u32_hash * vert_hash,
    const float * values,
    int resolution,
    int32_t x,
    int32_t y,
    int32_t z,
    int * ok) {
    uint32_t idx = vkmesh_hash_lookup(vert_hash, vkmesh_remesh_coord_key(x, y, z, resolution + 1));
    if (idx == UINT32_MAX) {
        *ok = 0;
        return 0.0f;
    }
    return values[idx];
}

static int vkmesh_remesh_simple_dual_contour(
    const vkmesh_remesh_coord * coords,
    int64_t coord_count,
    const vkmesh_u64_u32_hash * vert_hash,
    const float * grid_values,
    int resolution,
    float ** dual_out,
    int8_t ** intersected_out) {
    *dual_out = NULL;
    *intersected_out = NULL;
    if (coord_count <= 0 || coord_count > UINT32_MAX) return 0;
    float * dual = (float *) malloc((size_t) coord_count * 3u * sizeof(float));
    int8_t * intersected = (int8_t *) calloc((size_t) coord_count * 3u, sizeof(int8_t));
    if (dual == NULL || intersected == NULL) {
        free(dual);
        free(intersected);
        return 0;
    }

    int ok = 1;
    for (int64_t i = 0; i < coord_count; ++i) {
        int32_t vx = coords[i].x;
        int32_t vy = coords[i].y;
        int32_t vz = coords[i].z;
        float sum[3] = { 0.0f, 0.0f, 0.0f };
        int count = 0;

        for (int u = 0; u <= 1; ++u) {
            for (int v = 0; v <= 1; ++v) {
                float val1 = vkmesh_remesh_grid_value(vert_hash, grid_values, resolution, vx, vy + u, vz + v, &ok);
                float val2 = vkmesh_remesh_grid_value(vert_hash, grid_values, resolution, vx + 1, vy + u, vz + v, &ok);
                if (!ok) goto fail;
                if ((val1 < 0.0f && val2 >= 0.0f) || (val1 >= 0.0f && val2 < 0.0f)) {
                    float t = -val1 / (val2 - val1);
                    sum[0] += (float) vx + t;
                    sum[1] += (float) (vy + u);
                    sum[2] += (float) (vz + v);
                    ++count;
                }
                if (u == 1 && v == 1) {
                    intersected[(size_t) i * 3u + 0u] =
                        (val1 < 0.0f && val2 >= 0.0f) ? 1 : ((val1 >= 0.0f && val2 < 0.0f) ? -1 : 0);
                }
            }
        }

        for (int u = 0; u <= 1; ++u) {
            for (int v = 0; v <= 1; ++v) {
                float val1 = vkmesh_remesh_grid_value(vert_hash, grid_values, resolution, vx + u, vy, vz + v, &ok);
                float val2 = vkmesh_remesh_grid_value(vert_hash, grid_values, resolution, vx + u, vy + 1, vz + v, &ok);
                if (!ok) goto fail;
                if ((val1 < 0.0f && val2 >= 0.0f) || (val1 >= 0.0f && val2 < 0.0f)) {
                    float t = -val1 / (val2 - val1);
                    sum[0] += (float) (vx + u);
                    sum[1] += (float) vy + t;
                    sum[2] += (float) (vz + v);
                    ++count;
                }
                if (u == 1 && v == 1) {
                    intersected[(size_t) i * 3u + 1u] =
                        (val1 < 0.0f && val2 >= 0.0f) ? 1 : ((val1 >= 0.0f && val2 < 0.0f) ? -1 : 0);
                }
            }
        }

        for (int u = 0; u <= 1; ++u) {
            for (int v = 0; v <= 1; ++v) {
                float val1 = vkmesh_remesh_grid_value(vert_hash, grid_values, resolution, vx + u, vy + v, vz, &ok);
                float val2 = vkmesh_remesh_grid_value(vert_hash, grid_values, resolution, vx + u, vy + v, vz + 1, &ok);
                if (!ok) goto fail;
                if ((val1 < 0.0f && val2 >= 0.0f) || (val1 >= 0.0f && val2 < 0.0f)) {
                    float t = -val1 / (val2 - val1);
                    sum[0] += (float) (vx + u);
                    sum[1] += (float) (vy + v);
                    sum[2] += (float) vz + t;
                    ++count;
                }
                if (u == 1 && v == 1) {
                    intersected[(size_t) i * 3u + 2u] =
                        (val1 < 0.0f && val2 >= 0.0f) ? 1 : ((val1 >= 0.0f && val2 < 0.0f) ? -1 : 0);
                }
            }
        }

        if (count > 0) {
            dual[(size_t) i * 3u + 0u] = sum[0] / (float) count;
            dual[(size_t) i * 3u + 1u] = sum[1] / (float) count;
            dual[(size_t) i * 3u + 2u] = sum[2] / (float) count;
        } else {
            dual[(size_t) i * 3u + 0u] = (float) vx + 0.5f;
            dual[(size_t) i * 3u + 1u] = (float) vy + 0.5f;
            dual[(size_t) i * 3u + 2u] = (float) vz + 0.5f;
        }
    }

    *dual_out = dual;
    *intersected_out = intersected;
    return 1;

fail:
    free(dual);
    free(intersected);
    return 0;
}

static void vkmesh_cross3(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static float vkmesh_dot3(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static float vkmesh_remesh_split_align(const float * vertices, const int32_t q[4], const int split[6]) {
    const float * v0 = vertices + (size_t) q[split[0]] * 3u;
    const float * v1 = vertices + (size_t) q[split[1]] * 3u;
    const float * v2 = vertices + (size_t) q[split[2]] * 3u;
    const float * v3 = vertices + (size_t) q[split[3]] * 3u;
    float a[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
    float b[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };
    float c[3] = { v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2] };
    float d[3] = { v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2] };
    float n0[3];
    float n1[3];
    vkmesh_cross3(a, b, n0);
    vkmesh_cross3(c, d, n1);
    return fabsf(vkmesh_dot3(n0, n1));
}

static int vkmesh_remesh_append_quad(
    int32_t ** quads,
    int32_t ** dirs,
    int64_t * count,
    int64_t * capacity,
    const int32_t q[4],
    int32_t dir) {
    if (*count >= *capacity) {
        int64_t cap = *capacity > 0 ? *capacity : 1024;
        while (cap <= *count) {
            if (cap > INT64_MAX / 2) return 0;
            cap *= 2;
        }
        if ((uint64_t) cap > (uint64_t) SIZE_MAX / (4u * sizeof(int32_t))) return 0;
        int32_t * next_quads = (int32_t *) realloc(*quads, (size_t) cap * 4u * sizeof(int32_t));
        if (next_quads == NULL) return 0;
        *quads = next_quads;
        int32_t * next_dirs = (int32_t *) realloc(*dirs, (size_t) cap * sizeof(int32_t));
        if (next_dirs == NULL) return 0;
        *dirs = next_dirs;
        *capacity = cap;
    }
    memcpy(*quads + (size_t) *count * 4u, q, 4u * sizeof(int32_t));
    (*dirs)[*count] = dir;
    *count += 1;
    return 1;
}

static int vkmesh_remesh_build_mesh_from_dc(
    const vkmesh_remesh_coord * coords,
    int64_t coord_count,
    const vkmesh_u64_u32_hash * active_hash,
    const float * dual,
    const int8_t * intersected,
    int resolution,
    const float center[3],
    float scale,
    vkmesh_mesh * out) {
    memset(out, 0, sizeof(*out));
    if (coord_count <= 0 || coord_count > UINT32_MAX) return 0;
    static const int32_t neighbor_offsets[3][4][3] = {
        { { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 1 }, { 0, 1, 0 } },
        { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 0, 1 }, { 0, 0, 1 } },
        { { 0, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, { 1, 0, 0 } },
    };
    static const int split_1_n[6] = { 0, 1, 2, 0, 2, 3 };
    static const int split_1_p[6] = { 0, 2, 1, 0, 3, 2 };
    static const int split_2_n[6] = { 0, 1, 3, 3, 1, 2 };
    static const int split_2_p[6] = { 0, 3, 1, 3, 2, 1 };

    int32_t * quads = NULL;
    int32_t * dirs = NULL;
    int64_t quad_count = 0;
    int64_t quad_capacity = 0;
    uint8_t * referenced = (uint8_t *) calloc((size_t) coord_count, sizeof(uint8_t));
    int32_t * vertex_map = (int32_t *) malloc((size_t) coord_count * sizeof(int32_t));
    if (referenced == NULL || vertex_map == NULL) {
        free(referenced);
        free(vertex_map);
        return 0;
    }

    int ok = 0;
    for (int64_t i = 0; i < coord_count; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            int32_t dir = (int32_t) intersected[(size_t) i * 3u + (size_t) axis];
            if (dir == 0) continue;
            int32_t q[4];
            int valid = 1;
            for (int k = 0; k < 4; ++k) {
                int32_t x = coords[i].x + neighbor_offsets[axis][k][0];
                int32_t y = coords[i].y + neighbor_offsets[axis][k][1];
                int32_t z = coords[i].z + neighbor_offsets[axis][k][2];
                uint32_t idx = vkmesh_hash_lookup(active_hash, vkmesh_remesh_coord_key(x, y, z, resolution));
                if (idx == UINT32_MAX) {
                    valid = 0;
                    break;
                }
                q[k] = (int32_t) idx;
            }
            if (!valid) continue;
            if (!vkmesh_remesh_append_quad(&quads, &dirs, &quad_count, &quad_capacity, q, dir)) goto cleanup;
            for (int k = 0; k < 4; ++k) referenced[q[k]] = 1u;
        }
    }

    int64_t vertex_count = 0;
    for (int64_t i = 0; i < coord_count; ++i) {
        if (referenced[i]) {
            if (vertex_count > INT32_MAX) goto cleanup;
            vertex_map[i] = (int32_t) vertex_count++;
        } else {
            vertex_map[i] = -1;
        }
    }
    if (vertex_count <= 0 || quad_count <= 0 || quad_count > INT64_MAX / 2) goto cleanup;
    if ((uint64_t) vertex_count > (uint64_t) SIZE_MAX / (3u * sizeof(float)) ||
        (uint64_t) quad_count > (uint64_t) SIZE_MAX / (6u * sizeof(int32_t))) {
        goto cleanup;
    }
    out->vertices = (float *) malloc((size_t) vertex_count * 3u * sizeof(float));
    out->faces = (int32_t *) malloc((size_t) quad_count * 6u * sizeof(int32_t));
    if (out->vertices == NULL || out->faces == NULL) goto cleanup;
    out->n_vertices = vertex_count;
    out->vertex_capacity = vertex_count;
    out->n_faces = quad_count * 2;
    out->face_capacity = out->n_faces;

    const float inv_res = 1.0f / (float) resolution;
    for (int64_t i = 0; i < coord_count; ++i) {
        int32_t dst = vertex_map[i];
        if (dst < 0) continue;
        out->vertices[(size_t) dst * 3u + 0u] = (dual[(size_t) i * 3u + 0u] * inv_res - 0.5f) * scale + center[0];
        out->vertices[(size_t) dst * 3u + 1u] = (dual[(size_t) i * 3u + 1u] * inv_res - 0.5f) * scale + center[1];
        out->vertices[(size_t) dst * 3u + 2u] = (dual[(size_t) i * 3u + 2u] * inv_res - 0.5f) * scale + center[2];
    }

    int64_t face_dst = 0;
    for (int64_t i = 0; i < quad_count; ++i) {
        int32_t q[4];
        for (int k = 0; k < 4; ++k) q[k] = vertex_map[quads[(size_t) i * 4u + (size_t) k]];
        const int * split1 = dirs[i] == 1 ? split_1_p : split_1_n;
        const int * split2 = dirs[i] == 1 ? split_2_p : split_2_n;
        float align0 = vkmesh_remesh_split_align(out->vertices, q, split1);
        float align1 = vkmesh_remesh_split_align(out->vertices, q, split2);
        const int * split = align0 > align1 ? split1 : split2;
        for (int k = 0; k < 6; ++k) {
            out->faces[(size_t) face_dst * 3u + (size_t) k] = q[split[k]];
        }
        face_dst += 2;
    }
    ok = 1;

cleanup:
    free(quads);
    free(dirs);
    free(referenced);
    free(vertex_map);
    if (!ok) mesh_free(out);
    return ok;
}

static void vkmesh_closest_point_triangle(
    const float p[3],
    const float a[3],
    const float b[3],
    const float c[3],
    float q[3]) {
    float ab[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
    float ac[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
    float ap[3] = { p[0] - a[0], p[1] - a[1], p[2] - a[2] };
    float d1 = vkmesh_dot3(ab, ap);
    float d2 = vkmesh_dot3(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        memcpy(q, a, 3u * sizeof(float));
        return;
    }

    float bp[3] = { p[0] - b[0], p[1] - b[1], p[2] - b[2] };
    float d3 = vkmesh_dot3(ab, bp);
    float d4 = vkmesh_dot3(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        memcpy(q, b, 3u * sizeof(float));
        return;
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        q[0] = a[0] + v * ab[0];
        q[1] = a[1] + v * ab[1];
        q[2] = a[2] + v * ab[2];
        return;
    }

    float cp[3] = { p[0] - c[0], p[1] - c[1], p[2] - c[2] };
    float d5 = vkmesh_dot3(ab, cp);
    float d6 = vkmesh_dot3(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        memcpy(q, c, 3u * sizeof(float));
        return;
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        q[0] = a[0] + w * ac[0];
        q[1] = a[1] + w * ac[1];
        q[2] = a[2] + w * ac[2];
        return;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        q[0] = b[0] + w * (c[0] - b[0]);
        q[1] = b[1] + w * (c[1] - b[1]);
        q[2] = b[2] + w * (c[2] - b[2]);
        return;
    }

    float denom = 1.0f / fmaxf(va + vb + vc, 1e-30f);
    float v = vb * denom;
    float w = vc * denom;
    q[0] = a[0] + ab[0] * v + ac[0] * w;
    q[1] = a[1] + ab[1] * v + ac[1] * w;
    q[2] = a[2] + ab[2] * v + ac[2] * w;
}

static int vkmesh_remesh_project_back(
    vkmesh_mesh * mesh,
    const vkmesh_mesh * source,
    vkmesh_distance_query * query,
    float project_back) {
    if (project_back <= 0.0f) return 1;
    if ((uint64_t) mesh->n_vertices > (uint64_t) SIZE_MAX / sizeof(uint32_t)) return 0;
    uint32_t * face_ids = (uint32_t *) malloc((size_t) mesh->n_vertices * sizeof(uint32_t));
    if (face_ids == NULL ||
        !vkmesh_distance_query_points(query, mesh->vertices, mesh->n_vertices, NULL, face_ids)) {
        free(face_ids);
        return 0;
    }
    for (int64_t i = 0; i < mesh->n_vertices; ++i) {
        uint32_t face_id = face_ids[i];
        if (face_id >= (uint32_t) source->n_faces) continue;
        const int32_t * f = source->faces + (size_t) face_id * 3u;
        const float * a = source->vertices + (size_t) f[0] * 3u;
        const float * b = source->vertices + (size_t) f[1] * 3u;
        const float * c = source->vertices + (size_t) f[2] * 3u;
        float * p = mesh->vertices + (size_t) i * 3u;
        float q[3];
        vkmesh_closest_point_triangle(p, a, b, c, q);
        p[0] -= project_back * (p[0] - q[0]);
        p[1] -= project_back * (p[1] - q[1]);
        p[2] -= project_back * (p[2] - q[2]);
    }
    free(face_ids);
    return 1;
}

static int vkmesh_remesh_narrow_band_dc_to(
    const vkmesh_mesh * mesh,
    int resolution,
    float band,
    float project_back,
    vkmesh_mesh * result) {
    if (result == NULL || mesh == result || mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }
    memset(result, 0, sizeof(*result));
    if (resolution <= 0) resolution = 1024;
    if (band <= 0.0f) band = 1.0f;
    if (project_back < 0.0f) project_back = 0.0f;

    float bmin[3];
    float bmax[3];
    if (!vkmesh_mesh_bounds(mesh, bmin, bmax)) return 0;
    float center[3] = {
        0.5f * (bmin[0] + bmax[0]),
        0.5f * (bmin[1] + bmax[1]),
        0.5f * (bmin[2] + bmax[2]),
    };
    float base_scale = fmaxf(fmaxf(bmax[0] - bmin[0], bmax[1] - bmin[1]), bmax[2] - bmin[2]);
    if (!(base_scale > 0.0f)) return 0;
    float scale = ((float) resolution + 3.0f * band) / (float) resolution * base_scale;
    float eps = band * scale / (float) resolution;

    fprintf(stderr,
        "vkmesh: remesh_narrow_band_dc begin resolution=%d band=%.9g project=%.9g scale=%.9g eps=%.9g\n",
        resolution,
        band,
        project_back,
        scale,
        eps);

    vkmesh_bvh_node * nodes = NULL;
    uint32_t node_count = 0;
    uint32_t * tri_indices = NULL;
    vkmesh_remesh_coord * coords = NULL;
    int64_t coord_count = 0;
    vkmesh_u64_u32_hash active_hash;
    vkmesh_u64_u32_hash vert_hash;
    memset(&active_hash, 0, sizeof(active_hash));
    memset(&vert_hash, 0, sizeof(vert_hash));
    vkmesh_remesh_coord * grid_verts = NULL;
    int64_t grid_vert_count = 0;
    float * grid_values = NULL;
    float * dual = NULL;
    int8_t * intersected = NULL;
    vkmesh_mesh out;
    memset(&out, 0, sizeof(out));
    vkmesh_distance_query distance_query;
    memset(&distance_query, 0, sizeof(distance_query));
    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    int ok = 0;

    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) goto cleanup;
    if (!vkmesh_build_bvh(mesh, &nodes, &node_count, &tri_indices)) goto cleanup;
    if (!vkmesh_distance_query_init(
            &distance_query,
            vk,
            mesh,
            nodes,
            node_count,
            tri_indices,
            UINT32_MAX / 5u)) {
        goto cleanup;
    }
    if (!vkmesh_remesh_refine_sparse_grid(
            &distance_query,
            resolution,
            center,
            scale,
            band,
            &coords,
            &coord_count)) {
        goto cleanup;
    }
    fprintf(stderr, "vkmesh: remesh active_voxels=%" PRId64 "\n", coord_count);
    if (!vkmesh_remesh_build_active_hash(coords, coord_count, resolution, &active_hash)) goto cleanup;
    if (!vkmesh_remesh_collect_grid_vertices(coords, coord_count, resolution, &grid_verts, &grid_vert_count, &vert_hash)) goto cleanup;
    fprintf(stderr, "vkmesh: remesh active_grid_vertices=%" PRId64 "\n", grid_vert_count);
    if ((uint64_t) grid_vert_count > (uint64_t) SIZE_MAX / sizeof(float)) goto cleanup;
    grid_values = (float *) malloc((size_t) grid_vert_count * sizeof(float));
    if (grid_values == NULL ||
        !vkmesh_distance_query_coord_values(
            &distance_query, grid_verts, grid_vert_count, resolution, center, scale, 0, grid_values)) {
        goto cleanup;
    }
    for (int64_t i = 0; i < grid_vert_count; ++i) grid_values[i] -= eps;

    if (project_back <= 0.0f) {
        if (distance_query.batch_count > 0u) {
            fprintf(stderr, "vkmesh: remesh distance batches=%u\n", distance_query.batch_count);
        }
        vkmesh_distance_query_destroy(&distance_query);
        free(nodes);
        nodes = NULL;
        free(tri_indices);
        tri_indices = NULL;
    }
    if (!vkmesh_remesh_simple_dual_contour(
            coords, coord_count, &vert_hash, grid_values, resolution, &dual, &intersected)) goto cleanup;
    vkmesh_hash_destroy(&vert_hash);
    free(grid_verts);
    grid_verts = NULL;
    free(grid_values);
    grid_values = NULL;
    if (!vkmesh_remesh_build_mesh_from_dc(
            coords, coord_count, &active_hash, dual, intersected, resolution, center, scale, &out)) goto cleanup;
    fprintf(stderr, "vkmesh: remesh after_dual_contour vertices=%" PRId64 " faces=%" PRId64 "\n", out.n_vertices, out.n_faces);
    free(coords);
    coords = NULL;
    vkmesh_hash_destroy(&active_hash);
    free(dual);
    dual = NULL;
    free(intersected);
    intersected = NULL;
    if (!vkmesh_remesh_project_back(&out, mesh, &distance_query, project_back)) goto cleanup;

    *result = out;
    memset(&out, 0, sizeof(out));
    fprintf(stderr, "vkmesh: remesh_narrow_band_dc done vertices=%" PRId64 " faces=%" PRId64 "\n", result->n_vertices, result->n_faces);
    ok = 1;

cleanup:
    if (distance_query.batch_count > 0u) {
        fprintf(stderr, "vkmesh: remesh distance batches=%u\n", distance_query.batch_count);
    }
    vkmesh_distance_query_destroy(&distance_query);
    if (owns_vk) vkmesh_vk_destroy(vk);
    mesh_free(&out);
    free(nodes);
    free(tri_indices);
    free(coords);
    vkmesh_hash_destroy(&active_hash);
    vkmesh_hash_destroy(&vert_hash);
    free(grid_verts);
    free(grid_values);
    free(dual);
    free(intersected);
    return ok;
}

static int vkmesh_remesh_narrow_band_dc(
    vkmesh_mesh * mesh,
    int resolution,
    float band,
    float project_back) {
    if (mesh == NULL) return 0;
    vkmesh_mesh result;
    memset(&result, 0, sizeof(result));
    if (!vkmesh_remesh_narrow_band_dc_to(mesh, resolution, band, project_back, &result)) return 0;
    mesh_free(mesh);
    *mesh = result;
    return 1;
}

static uint32_t vkmesh_find_parent_u32(const uint32_t * parent, uint32_t count, uint32_t x) {
    if (parent == NULL || x >= count) return UINT32_MAX;
    uint32_t r = x;
    for (uint32_t i = 0; i < 256u && parent[r] < count && parent[r] != r; ++i) r = parent[r];
    return r < count ? r : UINT32_MAX;
}

static int compute_hole_components_vulkan(
    const vkmesh_mesh * mesh,
    vkmesh_boundary_edge ** boundary_out,
    uint32_t ** parent_out,
    uint32_t ** comp_edges_out,
    uint32_t ** comp_invalid_out,
    float ** comp_perim_out,
    float ** comp_mid_out,
    int64_t * boundary_count_out) {
    *boundary_out = NULL;
    *parent_out = NULL;
    *comp_edges_out = NULL;
    *comp_invalid_out = NULL;
    *comp_perim_out = NULL;
    *comp_mid_out = NULL;
    *boundary_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer boundary_buffer;
    vkmesh_vk_buffer counter_buffer;
    vkmesh_vk_buffer hole_data;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&boundary_buffer, 0, sizeof(boundary_buffer));
    memset(&counter_buffer, 0, sizeof(counter_buffer));
    memset(&hole_data, 0, sizeof(hole_data));

    const uint32_t vertex_count = (uint32_t) mesh->n_vertices;
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t max_boundary_count = (size_t) mesh->n_faces * 3u;
    int ok = 0;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, max_boundary_count * sizeof(vkmesh_boundary_edge), &boundary_buffer) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer compact_buffers[4];
    compact_buffers[0] = edges_buffer;
    compact_buffers[1] = boundary_buffer;
    compact_buffers[2] = counter_buffer;
    compact_buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) edge_count;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_BOUNDARY_EDGES, compact_buffers, &push, edge_groups)) goto cleanup;

    uint32_t boundary_count = ((const uint32_t *) counter_buffer.mapped)[0];
    if ((size_t) boundary_count > max_boundary_count) goto cleanup;
    if (boundary_count == 0u) {
        ok = 1;
        goto cleanup;
    }

    size_t hole_words_size = (size_t) boundary_count * 7u + (size_t) vertex_count * 2u;
    if (hole_words_size > UINT32_MAX) goto cleanup;
    const uint32_t degree_off = boundary_count;
    const uint32_t owner_off = degree_off + vertex_count;
    const uint32_t comp_edges_off = owner_off + vertex_count;
    const uint32_t comp_invalid_off = comp_edges_off + boundary_count;
    const uint32_t comp_perim_off = comp_invalid_off + boundary_count;
    const uint32_t comp_mid_off = comp_perim_off + boundary_count;
    const size_t hole_words = hole_words_size;
    if (!vk_buffer_create(vk, hole_words * sizeof(uint32_t), &hole_data)) goto cleanup;
    memset((uint32_t *) hole_data.mapped + owner_off, 0xff, (size_t) vertex_count * sizeof(uint32_t));

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = hole_data;
    init_buffers[1] = counter_buffer;
    init_buffers[2] = counter_buffer;
    init_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    uint32_t boundary_groups = (boundary_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer degree_buffers[4];
    degree_buffers[0] = boundary_buffer;
    degree_buffers[1] = dummy_buffer;
    degree_buffers[2] = hole_data;
    degree_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_BOUNDARY_DEGREE_OWNER, degree_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = boundary_buffer;
    union_buffers[1] = dummy_buffer;
    union_buffers[2] = hole_data;
    union_buffers[3] = counter_buffer;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = hole_data;
    compress_buffers[1] = counter_buffer;
    compress_buffers[2] = counter_buffer;
    compress_buffers[3] = counter_buffer;
    int converged = 0;
    for (int iter = 0; iter < 64; ++iter) {
        ((uint32_t *) counter_buffer.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = boundary_count;
        push.aux0 = vertex_count;
        if (!vkmesh_dispatch_pair(
                vk, VKMESH_PIPE_UNION_BOUNDARY_EDGES, union_buffers, &push, boundary_groups,
                VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, boundary_count, boundary_groups)) goto cleanup;
        if (((const uint32_t *) counter_buffer.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    vkmesh_vk_buffer accum_buffers[4];
    accum_buffers[0] = boundary_buffer;
    accum_buffers[1] = vertices_buffer;
    accum_buffers[2] = hole_data;
    accum_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ACCUMULATE_HOLE_COMPONENTS, accum_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_boundary_edge * boundary = (vkmesh_boundary_edge *) malloc((size_t) boundary_count * sizeof(*boundary));
    uint32_t * parents = (uint32_t *) malloc((size_t) boundary_count * sizeof(*parents));
    uint32_t * comp_edges = (uint32_t *) malloc((size_t) boundary_count * sizeof(*comp_edges));
    uint32_t * comp_invalid = (uint32_t *) malloc((size_t) boundary_count * sizeof(*comp_invalid));
    float * comp_perim = (float *) malloc((size_t) boundary_count * sizeof(*comp_perim));
    float * comp_mid = (float *) malloc((size_t) boundary_count * 3u * sizeof(*comp_mid));
    if (boundary == NULL || parents == NULL || comp_edges == NULL || comp_invalid == NULL ||
        comp_perim == NULL || comp_mid == NULL) {
        free(boundary);
        free(parents);
        free(comp_edges);
        free(comp_invalid);
        free(comp_perim);
        free(comp_mid);
        goto cleanup;
    }
    uint32_t * hole_words_u32 = (uint32_t *) hole_data.mapped;
    memcpy(boundary, boundary_buffer.mapped, (size_t) boundary_count * sizeof(*boundary));
    memcpy(parents, hole_words_u32, (size_t) boundary_count * sizeof(*parents));
    memcpy(comp_edges, hole_words_u32 + comp_edges_off, (size_t) boundary_count * sizeof(*comp_edges));
    memcpy(comp_invalid, hole_words_u32 + comp_invalid_off, (size_t) boundary_count * sizeof(*comp_invalid));
    memcpy(comp_perim, hole_words_u32 + comp_perim_off, (size_t) boundary_count * sizeof(*comp_perim));
    memcpy(comp_mid, hole_words_u32 + comp_mid_off, (size_t) boundary_count * 3u * sizeof(*comp_mid));
    *boundary_out = boundary;
    *parent_out = parents;
    *comp_edges_out = comp_edges;
    *comp_invalid_out = comp_invalid;
    *comp_perim_out = comp_perim;
    *comp_mid_out = comp_mid;
    *boundary_count_out = (int64_t) boundary_count;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &boundary_buffer);
    vk_buffer_destroy(vk, &counter_buffer);
    vk_buffer_destroy(vk, &hole_data);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_fill_holes_cpu(vkmesh_mesh * mesh, float max_hole_perimeter, int * filled_loops, int * added_faces) {
    *filled_loops = 0;
    *added_faces = 0;
    vkmesh_boundary_edge * boundary = NULL;
    int64_t boundary_count = 0;
    if (!get_boundary_edges(mesh, &boundary, &boundary_count)) return 0;
    if (boundary_count == 0) {
        free(boundary);
        return 1;
    }

    int * degree = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(int));
    int * offset = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(int));
    int * cursor = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(int));
    int * parent = (int *) malloc((size_t) boundary_count * sizeof(int));
    if (degree == NULL || offset == NULL || cursor == NULL || parent == NULL) {
        free(boundary);
        free(degree);
        free(offset);
        free(cursor);
        free(parent);
        return 0;
    }
    for (int64_t i = 0; i < boundary_count; ++i) {
        ++degree[boundary[i].a];
        ++degree[boundary[i].b];
        parent[i] = (int) i;
    }
    for (int64_t v = 0; v < mesh->n_vertices; ++v) {
        offset[v + 1] = offset[v] + degree[v];
    }
    int * adj = (int *) malloc((size_t) offset[mesh->n_vertices] * sizeof(int));
    if (adj == NULL) {
        free(boundary);
        free(degree);
        free(offset);
        free(cursor);
        free(parent);
        return 0;
    }
    memcpy(cursor, offset, ((size_t) mesh->n_vertices + 1u) * sizeof(int));
    for (int64_t i = 0; i < boundary_count; ++i) {
        adj[cursor[boundary[i].a]++] = (int) i;
        adj[cursor[boundary[i].b]++] = (int) i;
    }
    for (int64_t v = 0; v < mesh->n_vertices; ++v) {
        int begin = offset[v];
        int end = offset[v + 1];
        for (int k = begin + 1; k < end; ++k) {
            uf_union(parent, adj[begin], adj[k]);
        }
    }

    uint8_t * valid = (uint8_t *) calloc((size_t) boundary_count, sizeof(uint8_t));
    int * comp_edges = (int *) calloc((size_t) boundary_count, sizeof(int));
    double * comp_perim = (double *) calloc((size_t) boundary_count, sizeof(double));
    double * comp_mid = (double *) calloc((size_t) boundary_count * 3u, sizeof(double));
    int32_t * comp_new_vertex = (int32_t *) malloc((size_t) boundary_count * sizeof(int32_t));
    if (valid == NULL || comp_edges == NULL || comp_perim == NULL || comp_mid == NULL || comp_new_vertex == NULL) {
        free(boundary); free(degree); free(offset); free(cursor); free(parent); free(adj);
        free(valid); free(comp_edges); free(comp_perim); free(comp_mid); free(comp_new_vertex);
        return 0;
    }
    memset(comp_new_vertex, 0xff, (size_t) boundary_count * sizeof(int32_t));
    for (int64_t i = 0; i < boundary_count; ++i) valid[uf_find(parent, (int) i)] = 1u;
    for (int64_t v = 0; v < mesh->n_vertices; ++v) {
        if (degree[v] == 0) continue;
        int root = uf_find(parent, adj[offset[v]]);
        if (degree[v] != 2) valid[root] = 0u;
    }
    for (int64_t i = 0; i < boundary_count; ++i) {
        int root = uf_find(parent, (int) i);
        const float * a = mesh->vertices + (size_t) boundary[i].a * 3u;
        const float * b = mesh->vertices + (size_t) boundary[i].b * 3u;
        ++comp_edges[root];
        comp_perim[root] += edge_length(mesh, boundary[i].a, boundary[i].b);
        comp_mid[(size_t) root * 3u + 0u] += ((double) a[0] + (double) b[0]) * 0.5;
        comp_mid[(size_t) root * 3u + 1u] += ((double) a[1] + (double) b[1]) * 0.5;
        comp_mid[(size_t) root * 3u + 2u] += ((double) a[2] + (double) b[2]) * 0.5;
    }
    for (int64_t root = 0; root < boundary_count; ++root) {
        if (!valid[root] || comp_edges[root] <= 0 || comp_perim[root] >= (double) max_hole_perimeter) continue;
        double inv = 1.0 / (double) comp_edges[root];
        if (!mesh_add_vertex(
                mesh,
                (float) (comp_mid[(size_t) root * 3u + 0u] * inv),
                (float) (comp_mid[(size_t) root * 3u + 1u] * inv),
                (float) (comp_mid[(size_t) root * 3u + 2u] * inv))) {
            free(boundary); free(degree); free(offset); free(cursor); free(parent); free(adj);
            free(valid); free(comp_edges); free(comp_perim); free(comp_mid); free(comp_new_vertex);
            return 0;
        }
        comp_new_vertex[root] = (int32_t) (mesh->n_vertices - 1);
        ++(*filled_loops);
    }
    for (int64_t i = 0; i < boundary_count; ++i) {
        int root = uf_find(parent, (int) i);
        int32_t center = comp_new_vertex[root];
        if (center < 0) continue;
        if (!mesh_add_face(mesh, boundary[i].b, boundary[i].a, center)) {
            free(boundary); free(degree); free(offset); free(cursor); free(parent); free(adj);
            free(valid); free(comp_edges); free(comp_perim); free(comp_mid); free(comp_new_vertex);
            return 0;
        }
        ++(*added_faces);
    }

    free(boundary);
    free(degree);
    free(offset);
    free(cursor);
    free(parent);
    free(adj);
    free(valid);
    free(comp_edges);
    free(comp_perim);
    free(comp_mid);
    free(comp_new_vertex);
    return 1;
}

static int vkmesh_scan_flags_to_offsets(
    vkmesh_vk * vk,
    vkmesh_vk_buffer * flags,
    uint32_t count,
    vkmesh_vk_buffer * offsets_a,
    vkmesh_vk_buffer * offsets_b,
    vkmesh_vk_buffer * dummy,
    vkmesh_vk_buffer ** offsets_out) {
    if (vk == NULL || flags == NULL || offsets_a == NULL || offsets_b == NULL || dummy == NULL || offsets_out == NULL) return 0;
    *offsets_out = NULL;
    vkmesh_vk_buffer seed_buffers[4];
    seed_buffers[0] = *flags;
    seed_buffers[1] = *offsets_a;
    seed_buffers[2] = *dummy;
    seed_buffers[3] = *dummy;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = count + 1u;
    uint32_t groups = (push.n + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SEED_VERTEX_OFFSETS, seed_buffers, &push, groups)) return 0;

    if (!vkmesh_scan_u32_inclusive(vk, offsets_a, offsets_b, count + 1u, dummy)) return 0;
    *offsets_out = offsets_b;
    return 1;
}

static int vkmesh_fill_holes_vulkan(vkmesh_mesh * mesh, float max_hole_perimeter, int * filled_loops, int * added_faces) {
    *filled_loops = 0;
    *added_faces = 0;
    if (mesh == NULL || mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer boundary_buffer;
    vkmesh_vk_buffer counter_buffer;
    vkmesh_vk_buffer hole_data;
    vkmesh_vk_buffer root_flags;
    vkmesh_vk_buffer root_offsets_a;
    vkmesh_vk_buffer root_offsets_b;
    vkmesh_vk_buffer face_flags;
    vkmesh_vk_buffer face_offsets_a;
    vkmesh_vk_buffer face_offsets_b;
    vkmesh_vk_buffer aux;
    vkmesh_vk_buffer out_vertices;
    vkmesh_vk_buffer out_faces;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&boundary_buffer, 0, sizeof(boundary_buffer));
    memset(&counter_buffer, 0, sizeof(counter_buffer));
    memset(&hole_data, 0, sizeof(hole_data));
    memset(&root_flags, 0, sizeof(root_flags));
    memset(&root_offsets_a, 0, sizeof(root_offsets_a));
    memset(&root_offsets_b, 0, sizeof(root_offsets_b));
    memset(&face_flags, 0, sizeof(face_flags));
    memset(&face_offsets_a, 0, sizeof(face_offsets_a));
    memset(&face_offsets_b, 0, sizeof(face_offsets_b));
    memset(&aux, 0, sizeof(aux));
    memset(&out_vertices, 0, sizeof(out_vertices));
    memset(&out_faces, 0, sizeof(out_faces));

    const uint32_t vertex_count = (uint32_t) mesh->n_vertices;
    const uint32_t face_count = (uint32_t) mesh->n_faces;
    const size_t faces_bytes = (size_t) face_count * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) vertex_count * 3u * sizeof(float);
    const size_t max_boundary_count = (size_t) face_count * 3u;
    int ok = 0;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, max_boundary_count * sizeof(vkmesh_boundary_edge), &boundary_buffer) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(vk, mesh, &faces_buffer, &vertices_buffer, &edges_buffer, &dummy_buffer, &edge_count, &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer compact_buffers[4];
    compact_buffers[0] = edges_buffer;
    compact_buffers[1] = boundary_buffer;
    compact_buffers[2] = counter_buffer;
    compact_buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) edge_count;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_BOUNDARY_EDGES, compact_buffers, &push, edge_groups)) goto cleanup;

    uint32_t boundary_count = ((const uint32_t *) counter_buffer.mapped)[0];
    if ((size_t) boundary_count > max_boundary_count) goto cleanup;
    if (boundary_count == 0u) {
        ok = 1;
        goto cleanup;
    }

    size_t hole_words_size = (size_t) boundary_count * 7u + (size_t) vertex_count * 2u;
    if (hole_words_size > UINT32_MAX) goto cleanup;
    const uint32_t degree_off = boundary_count;
    const uint32_t owner_off = degree_off + vertex_count;
    (void) owner_off;
    if (!vk_buffer_create(vk, hole_words_size * sizeof(uint32_t), &hole_data) ||
        !vk_buffer_create(vk, (size_t) boundary_count * sizeof(uint32_t), &root_flags) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &root_offsets_a) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &root_offsets_b) ||
        !vk_buffer_create(vk, (size_t) boundary_count * sizeof(uint32_t), &face_flags) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &face_offsets_a) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &face_offsets_b)) {
        goto cleanup;
    }
    memset((uint32_t *) hole_data.mapped + owner_off, 0xff, (size_t) vertex_count * sizeof(uint32_t));

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = hole_data;
    init_buffers[1] = dummy_buffer;
    init_buffers[2] = dummy_buffer;
    init_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    uint32_t boundary_groups = (boundary_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer degree_buffers[4];
    degree_buffers[0] = boundary_buffer;
    degree_buffers[1] = dummy_buffer;
    degree_buffers[2] = hole_data;
    degree_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_BOUNDARY_DEGREE_OWNER, degree_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = boundary_buffer;
    union_buffers[1] = dummy_buffer;
    union_buffers[2] = hole_data;
    union_buffers[3] = counter_buffer;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = hole_data;
    compress_buffers[1] = counter_buffer;
    compress_buffers[2] = counter_buffer;
    compress_buffers[3] = counter_buffer;
    int converged = 0;
    for (int iter = 0; iter < 64; ++iter) {
        ((uint32_t *) counter_buffer.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = boundary_count;
        push.aux0 = vertex_count;
        if (!vkmesh_dispatch_pair(
                vk, VKMESH_PIPE_UNION_BOUNDARY_EDGES, union_buffers, &push, boundary_groups,
                VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, boundary_count, boundary_groups)) goto cleanup;
        if (((const uint32_t *) counter_buffer.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    vkmesh_vk_buffer accum_buffers[4];
    accum_buffers[0] = boundary_buffer;
    accum_buffers[1] = vertices_buffer;
    accum_buffers[2] = hole_data;
    accum_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ACCUMULATE_HOLE_COMPONENTS, accum_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer mark_root_buffers[4];
    mark_root_buffers[0] = hole_data;
    mark_root_buffers[1] = root_flags;
    mark_root_buffers[2] = dummy_buffer;
    mark_root_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    push.eps = max_hole_perimeter;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_HOLE_ROOTS, mark_root_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer * root_offsets = NULL;
    if (!vkmesh_scan_flags_to_offsets(vk, &root_flags, boundary_count, &root_offsets_a, &root_offsets_b, &dummy_buffer, &root_offsets)) goto cleanup;
    uint32_t new_vertex_count = ((const uint32_t *) root_offsets->mapped)[boundary_count];
    if (new_vertex_count == 0u) {
        ok = 1;
        goto cleanup;
    }

    vkmesh_vk_buffer mark_face_buffers[4];
    mark_face_buffers[0] = hole_data;
    mark_face_buffers[1] = root_flags;
    mark_face_buffers[2] = face_flags;
    mark_face_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_HOLE_FACES, mark_face_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer * face_offsets = NULL;
    if (!vkmesh_scan_flags_to_offsets(vk, &face_flags, boundary_count, &face_offsets_a, &face_offsets_b, &dummy_buffer, &face_offsets)) goto cleanup;
    uint32_t new_face_count = ((const uint32_t *) face_offsets->mapped)[boundary_count];
    if (new_face_count == 0u ||
        vertex_count > UINT32_MAX - new_vertex_count ||
        face_count > UINT32_MAX - new_face_count) {
        goto cleanup;
    }

    const uint32_t out_vertex_count = vertex_count + new_vertex_count;
    const uint32_t out_face_count = face_count + new_face_count;
    const size_t aux_words = (size_t) boundary_count * 2u + 1u;
    if (!vk_buffer_create(vk, (size_t) out_vertex_count * 3u * sizeof(float), &out_vertices) ||
        !vk_buffer_create(vk, (size_t) out_face_count * 3u * sizeof(int32_t), &out_faces) ||
        !vk_buffer_create(vk, aux_words * sizeof(uint32_t), &aux)) {
        goto cleanup;
    }
    memset(aux.mapped, 0xff, (size_t) boundary_count * sizeof(uint32_t));

    vkmesh_vk_buffer copy_buffers[4];
    copy_buffers[0] = vertices_buffer;
    copy_buffers[1] = out_vertices;
    copy_buffers[2] = dummy_buffer;
    copy_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count * 3u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    copy_buffers[0] = faces_buffer;
    copy_buffers[1] = out_faces;
    memset(&push, 0, sizeof(push));
    push.n = face_count * 3u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    vkmesh_vk_buffer write_vertex_buffers[4];
    write_vertex_buffers[0] = hole_data;
    write_vertex_buffers[1] = *root_offsets;
    write_vertex_buffers[2] = aux;
    write_vertex_buffers[3] = out_vertices;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    push.aux1 = vertex_count;
    push.eps = max_hole_perimeter;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_WRITE_HOLE_VERTICES, write_vertex_buffers, &push, boundary_groups)) goto cleanup;

    copy_buffers[0] = *face_offsets;
    copy_buffers[1] = aux;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count + 1u;
    push.aux0 = 0u;
    push.aux1 = boundary_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    vkmesh_vk_buffer write_face_buffers[4];
    write_face_buffers[0] = boundary_buffer;
    write_face_buffers[1] = hole_data;
    write_face_buffers[2] = aux;
    write_face_buffers[3] = out_faces;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux1 = face_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_WRITE_HOLE_FACES, write_face_buffers, &push, boundary_groups)) goto cleanup;

    float * new_vertices = (float *) malloc((size_t) out_vertex_count * 3u * sizeof(float));
    int32_t * new_faces = (int32_t *) malloc((size_t) out_face_count * 3u * sizeof(int32_t));
    if (new_vertices == NULL || new_faces == NULL) {
        free(new_vertices);
        free(new_faces);
        goto cleanup;
    }
    memcpy(new_vertices, out_vertices.mapped, (size_t) out_vertex_count * 3u * sizeof(float));
    memcpy(new_faces, out_faces.mapped, (size_t) out_face_count * 3u * sizeof(int32_t));

    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = new_vertices;
    mesh->uvs = NULL;
    mesh->faces = new_faces;
    mesh->n_vertices = out_vertex_count;
    mesh->n_faces = out_face_count;
    mesh->vertex_capacity = out_vertex_count;
    mesh->face_capacity = out_face_count;
    mesh->has_uvs = 0;
    *filled_loops = (int) new_vertex_count;
    *added_faces = (int) new_face_count;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &boundary_buffer);
    vk_buffer_destroy(vk, &counter_buffer);
    vk_buffer_destroy(vk, &hole_data);
    vk_buffer_destroy(vk, &root_flags);
    vk_buffer_destroy(vk, &root_offsets_a);
    vk_buffer_destroy(vk, &root_offsets_b);
    vk_buffer_destroy(vk, &face_flags);
    vk_buffer_destroy(vk, &face_offsets_a);
    vk_buffer_destroy(vk, &face_offsets_b);
    vk_buffer_destroy(vk, &aux);
    vk_buffer_destroy(vk, &out_vertices);
    vk_buffer_destroy(vk, &out_faces);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_device_fill_holes(
    vkmesh_device_mesh * dm,
    float max_hole_perimeter,
    uint32_t * filled_loops,
    uint32_t * added_faces) {
    if (filled_loops != NULL) *filled_loops = 0u;
    if (added_faces != NULL) *added_faces = 0u;
    if (dm == NULL || dm->vk == NULL || dm->n_faces == 0u || dm->n_vertices == 0u ||
        dm->n_faces > UINT32_MAX / 3u) {
        return 0;
    }

    vkmesh_vk * vk = dm->vk;
    vkmesh_mesh view;
    memset(&view, 0, sizeof(view));
    view.n_vertices = dm->n_vertices;
    view.n_faces = dm->n_faces;

    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer boundary_buffer;
    vkmesh_vk_buffer counter_buffer;
    vkmesh_vk_buffer hole_data;
    vkmesh_vk_buffer root_flags;
    vkmesh_vk_buffer root_offsets_a;
    vkmesh_vk_buffer root_offsets_b;
    vkmesh_vk_buffer face_flags;
    vkmesh_vk_buffer face_offsets_a;
    vkmesh_vk_buffer face_offsets_b;
    vkmesh_vk_buffer aux;
    vkmesh_vk_buffer out_vertices;
    vkmesh_vk_buffer out_faces;
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&boundary_buffer, 0, sizeof(boundary_buffer));
    memset(&counter_buffer, 0, sizeof(counter_buffer));
    memset(&hole_data, 0, sizeof(hole_data));
    memset(&root_flags, 0, sizeof(root_flags));
    memset(&root_offsets_a, 0, sizeof(root_offsets_a));
    memset(&root_offsets_b, 0, sizeof(root_offsets_b));
    memset(&face_flags, 0, sizeof(face_flags));
    memset(&face_offsets_a, 0, sizeof(face_offsets_a));
    memset(&face_offsets_b, 0, sizeof(face_offsets_b));
    memset(&aux, 0, sizeof(aux));
    memset(&out_vertices, 0, sizeof(out_vertices));
    memset(&out_faces, 0, sizeof(out_faces));

    const uint32_t vertex_count = dm->n_vertices;
    const uint32_t face_count = dm->n_faces;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;
    int ok = 0;

    if (!vk_buffer_create(vk, sizeof(uint32_t), &counter_buffer)) {
        goto cleanup;
    }

    if (!expand_edges_device(vk, &view, &dm->faces, &dm->vertices, &edges_buffer, &dummy_buffer, &edge_count, &edge_sort_count)) {
        goto cleanup;
    }

    size_t boundary_count_size = 0;
    const vkmesh_edge * sorted_edges = (const vkmesh_edge *) edges_buffer.mapped;
    for (size_t begin = 0; begin < edge_count;) {
        size_t end = begin + 1u;
        while (end < edge_count &&
               sorted_edges[end].min_v == sorted_edges[begin].min_v &&
               sorted_edges[end].max_v == sorted_edges[begin].max_v) {
            ++end;
        }
        if (end == begin + 1u) ++boundary_count_size;
        begin = end;
    }
    if (boundary_count_size == 0u) {
        ok = 1;
        goto cleanup;
    }
    if (boundary_count_size > UINT32_MAX ||
        !vk_buffer_create(vk, boundary_count_size * sizeof(vkmesh_boundary_edge), &boundary_buffer)) {
        goto cleanup;
    }

    vkmesh_vk_buffer compact_buffers[4];
    compact_buffers[0] = edges_buffer;
    compact_buffers[1] = boundary_buffer;
    compact_buffers[2] = counter_buffer;
    compact_buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) edge_count;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_BOUNDARY_EDGES, compact_buffers, &push, edge_groups)) goto cleanup;

    uint32_t boundary_count = ((const uint32_t *) counter_buffer.mapped)[0];
    if ((size_t) boundary_count != boundary_count_size) goto cleanup;
    vk_buffer_destroy(vk, &edges_buffer);

    size_t hole_words_size = (size_t) boundary_count * 7u + (size_t) vertex_count * 2u;
    if (hole_words_size > UINT32_MAX) goto cleanup;
    const uint32_t degree_off = boundary_count;
    const uint32_t owner_off = degree_off + vertex_count;
    if (!vk_buffer_create(vk, hole_words_size * sizeof(uint32_t), &hole_data) ||
        !vk_buffer_create(vk, (size_t) boundary_count * sizeof(uint32_t), &root_flags) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &root_offsets_a) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &root_offsets_b) ||
        !vk_buffer_create(vk, (size_t) boundary_count * sizeof(uint32_t), &face_flags) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &face_offsets_a) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &face_offsets_b)) {
        goto cleanup;
    }
    memset((uint32_t *) hole_data.mapped + owner_off, 0xff, (size_t) vertex_count * sizeof(uint32_t));

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = hole_data;
    init_buffers[1] = dummy_buffer;
    init_buffers[2] = dummy_buffer;
    init_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    uint32_t boundary_groups = (boundary_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer degree_buffers[4];
    degree_buffers[0] = boundary_buffer;
    degree_buffers[1] = dummy_buffer;
    degree_buffers[2] = hole_data;
    degree_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_BOUNDARY_DEGREE_OWNER, degree_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = boundary_buffer;
    union_buffers[1] = dummy_buffer;
    union_buffers[2] = hole_data;
    union_buffers[3] = counter_buffer;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = hole_data;
    compress_buffers[1] = counter_buffer;
    compress_buffers[2] = counter_buffer;
    compress_buffers[3] = counter_buffer;
    int converged = 0;
    for (int iter = 0; iter < 64; ++iter) {
        ((uint32_t *) counter_buffer.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = boundary_count;
        push.aux0 = vertex_count;
        if (!vkmesh_dispatch_pair(
                vk, VKMESH_PIPE_UNION_BOUNDARY_EDGES, union_buffers, &push, boundary_groups,
                VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, boundary_count, boundary_groups)) goto cleanup;
        if (((const uint32_t *) counter_buffer.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    vkmesh_vk_buffer accum_buffers[4];
    accum_buffers[0] = boundary_buffer;
    accum_buffers[1] = dm->vertices;
    accum_buffers[2] = hole_data;
    accum_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ACCUMULATE_HOLE_COMPONENTS, accum_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer mark_root_buffers[4];
    mark_root_buffers[0] = hole_data;
    mark_root_buffers[1] = root_flags;
    mark_root_buffers[2] = dummy_buffer;
    mark_root_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    push.eps = max_hole_perimeter;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_HOLE_ROOTS, mark_root_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer * root_offsets = NULL;
    if (!vkmesh_scan_flags_to_offsets(vk, &root_flags, boundary_count, &root_offsets_a, &root_offsets_b, &dummy_buffer, &root_offsets)) goto cleanup;
    uint32_t new_vertex_count = ((const uint32_t *) root_offsets->mapped)[boundary_count];
    if (new_vertex_count == 0u) {
        ok = 1;
        goto cleanup;
    }

    vkmesh_vk_buffer mark_face_buffers[4];
    mark_face_buffers[0] = hole_data;
    mark_face_buffers[1] = root_flags;
    mark_face_buffers[2] = face_flags;
    mark_face_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_HOLE_FACES, mark_face_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer * face_offsets = NULL;
    if (!vkmesh_scan_flags_to_offsets(vk, &face_flags, boundary_count, &face_offsets_a, &face_offsets_b, &dummy_buffer, &face_offsets)) goto cleanup;
    uint32_t new_face_count = ((const uint32_t *) face_offsets->mapped)[boundary_count];
    if (new_face_count == 0u ||
        vertex_count > UINT32_MAX - new_vertex_count ||
        face_count > UINT32_MAX - new_face_count) {
        goto cleanup;
    }

    const uint32_t out_vertex_count = vertex_count + new_vertex_count;
    const uint32_t out_face_count = face_count + new_face_count;
    const size_t aux_words = (size_t) boundary_count * 2u + 1u;
    if (!vk_buffer_create(vk, (size_t) out_vertex_count * 3u * sizeof(float), &out_vertices) ||
        !vk_buffer_create(vk, (size_t) out_face_count * 3u * sizeof(int32_t), &out_faces) ||
        !vk_buffer_create(vk, aux_words * sizeof(uint32_t), &aux)) {
        goto cleanup;
    }
    memset(aux.mapped, 0xff, (size_t) boundary_count * sizeof(uint32_t));

    vkmesh_vk_buffer copy_buffers[4];
    copy_buffers[0] = dm->vertices;
    copy_buffers[1] = out_vertices;
    copy_buffers[2] = dummy_buffer;
    copy_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count * 3u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    copy_buffers[0] = dm->faces;
    copy_buffers[1] = out_faces;
    memset(&push, 0, sizeof(push));
    push.n = face_count * 3u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    vkmesh_vk_buffer write_vertex_buffers[4];
    write_vertex_buffers[0] = hole_data;
    write_vertex_buffers[1] = *root_offsets;
    write_vertex_buffers[2] = aux;
    write_vertex_buffers[3] = out_vertices;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    push.aux1 = vertex_count;
    push.eps = max_hole_perimeter;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_WRITE_HOLE_VERTICES, write_vertex_buffers, &push, boundary_groups)) goto cleanup;

    copy_buffers[0] = *face_offsets;
    copy_buffers[1] = aux;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count + 1u;
    push.aux0 = 0u;
    push.aux1 = boundary_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    vkmesh_vk_buffer write_face_buffers[4];
    write_face_buffers[0] = boundary_buffer;
    write_face_buffers[1] = hole_data;
    write_face_buffers[2] = aux;
    write_face_buffers[3] = out_faces;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux1 = face_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_WRITE_HOLE_FACES, write_face_buffers, &push, boundary_groups)) goto cleanup;

    vk_buffer_destroy(vk, &dm->vertices);
    vk_buffer_destroy(vk, &dm->faces);
    dm->vertices = out_vertices;
    dm->faces = out_faces;
    memset(&out_vertices, 0, sizeof(out_vertices));
    memset(&out_faces, 0, sizeof(out_faces));
    dm->n_vertices = out_vertex_count;
    dm->n_faces = out_face_count;
    if (filled_loops != NULL) *filled_loops = new_vertex_count;
    if (added_faces != NULL) *added_faces = new_face_count;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &boundary_buffer);
    vk_buffer_destroy(vk, &counter_buffer);
    vk_buffer_destroy(vk, &hole_data);
    vk_buffer_destroy(vk, &root_flags);
    vk_buffer_destroy(vk, &root_offsets_a);
    vk_buffer_destroy(vk, &root_offsets_b);
    vk_buffer_destroy(vk, &face_flags);
    vk_buffer_destroy(vk, &face_offsets_a);
    vk_buffer_destroy(vk, &face_offsets_b);
    vk_buffer_destroy(vk, &aux);
    vk_buffer_destroy(vk, &out_vertices);
    vk_buffer_destroy(vk, &out_faces);
    return ok;
}

static int vkmesh_fill_holes_device_vulkan(vkmesh_mesh * mesh, float max_hole_perimeter, int * filled_loops, int * added_faces) {
    *filled_loops = 0;
    *added_faces = 0;
    if (mesh == NULL || mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_device_mesh dm;
    memset(&dm, 0, sizeof(dm));
    uint32_t filled_u32 = 0u;
    uint32_t added_u32 = 0u;
    int ok = 0;
    if (!vkmesh_device_mesh_upload(vk, mesh, &dm)) goto cleanup;
    if (!vkmesh_device_fill_holes(&dm, max_hole_perimeter, &filled_u32, &added_u32)) goto cleanup;
    if (filled_u32 > (uint32_t) INT_MAX || added_u32 > (uint32_t) INT_MAX) goto cleanup;
    if (!vkmesh_device_mesh_download(&dm, mesh)) goto cleanup;
    *filled_loops = (int) filled_u32;
    *added_faces = (int) added_u32;
    ok = 1;

cleanup:
    vkmesh_device_mesh_destroy(&dm);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_fill_holes(vkmesh_mesh * mesh, float max_hole_perimeter, int * filled_loops, int * added_faces) {
    *filled_loops = 0;
    *added_faces = 0;
    if (vkmesh_fill_holes_device_vulkan(mesh, max_hole_perimeter, filled_loops, added_faces)) return 1;
    if (vkmesh_fill_holes_vulkan(mesh, max_hole_perimeter, filled_loops, added_faces)) return 1;
    vkmesh_boundary_edge * boundary = NULL;
    uint32_t * parents = NULL;
    uint32_t * comp_edges = NULL;
    uint32_t * comp_invalid = NULL;
    float * comp_perim = NULL;
    float * comp_mid = NULL;
    int64_t boundary_count_i64 = 0;
    if (!compute_hole_components_vulkan(
            mesh,
            &boundary,
            &parents,
            &comp_edges,
            &comp_invalid,
            &comp_perim,
            &comp_mid,
            &boundary_count_i64)) {
        return vkmesh_fill_holes_cpu(mesh, max_hole_perimeter, filled_loops, added_faces);
    }
    if (boundary_count_i64 <= 0) {
        free(boundary);
        free(parents);
        free(comp_edges);
        free(comp_invalid);
        free(comp_perim);
        free(comp_mid);
        return 1;
    }

    uint32_t boundary_count = (uint32_t) boundary_count_i64;
    int32_t * comp_new_vertex = (int32_t *) malloc((size_t) boundary_count * sizeof(*comp_new_vertex));
    if (comp_new_vertex == NULL) {
        free(boundary); free(parents); free(comp_edges); free(comp_invalid); free(comp_perim); free(comp_mid);
        return 0;
    }
    memset(comp_new_vertex, 0xff, (size_t) boundary_count * sizeof(*comp_new_vertex));

    for (uint32_t root = 0; root < boundary_count; ++root) {
        if (comp_edges[root] == 0u || comp_invalid[root] != 0u ||
            comp_perim[root] >= max_hole_perimeter) {
            continue;
        }
        float inv = 1.0f / (float) comp_edges[root];
        if (!mesh_add_vertex(
                mesh,
                comp_mid[(size_t) root * 3u + 0u] * inv,
                comp_mid[(size_t) root * 3u + 1u] * inv,
                comp_mid[(size_t) root * 3u + 2u] * inv)) {
            free(boundary); free(parents); free(comp_edges); free(comp_invalid); free(comp_perim); free(comp_mid);
            free(comp_new_vertex);
            return 0;
        }
        comp_new_vertex[root] = (int32_t) (mesh->n_vertices - 1);
        ++(*filled_loops);
    }

    for (uint32_t i = 0; i < boundary_count; ++i) {
        uint32_t root = vkmesh_find_parent_u32(parents, boundary_count, i);
        if (root == UINT32_MAX) continue;
        int32_t center = comp_new_vertex[root];
        if (center < 0) continue;
        if (!mesh_add_face(mesh, boundary[i].b, boundary[i].a, center)) {
            free(boundary); free(parents); free(comp_edges); free(comp_invalid); free(comp_perim); free(comp_mid);
            free(comp_new_vertex);
            return 0;
        }
        ++(*added_faces);
    }

    free(boundary);
    free(parents);
    free(comp_edges);
    free(comp_invalid);
    free(comp_perim);
    free(comp_mid);
    free(comp_new_vertex);
    return 1;
}

static int vkmesh_log_fill_holes(vkmesh_mesh * mesh, float max_hole_perimeter, const char * label) {
    int filled = 0;
    int added_faces = 0;
    int64_t before_v = mesh->n_vertices;
    int64_t before_f = mesh->n_faces;
    if (!vkmesh_fill_holes(mesh, max_hole_perimeter, &filled, &added_faces)) return 0;
    fprintf(stderr,
        "vkmesh: %s fill_holes loops=%d added_vertices=%" PRId64 " added_faces=%d faces=%" PRId64 "->%" PRId64 "\n",
        label,
        filled,
        mesh->n_vertices - before_v,
        added_faces,
        before_f,
        mesh->n_faces);
    return 1;
}

static int vkmesh_log_remove_duplicate_faces(vkmesh_mesh * mesh, const char * label) {
    int removed = 0;
    if (!vkmesh_remove_duplicate_faces(mesh, &removed)) return 0;
    fprintf(stderr, "vkmesh: %s remove_duplicate_faces removed=%d faces=%" PRId64 "\n", label, removed, mesh->n_faces);
    return 1;
}

static int vkmesh_log_remove_degenerate_faces(
    vkmesh_mesh * mesh,
    float degenerate_abs,
    float degenerate_rel,
    const char * label) {
    int removed = 0;
    if (!vkmesh_remove_degenerate_faces(mesh, degenerate_abs, degenerate_rel, &removed)) return 0;
    fprintf(stderr, "vkmesh: %s remove_degenerate_faces removed=%d faces=%" PRId64 "\n", label, removed, mesh->n_faces);
    return 1;
}

static int vkmesh_log_repair_non_manifold_edges(vkmesh_mesh * mesh, const char * label) {
    int old_v = 0;
    int new_v = 0;
    if (!vkmesh_repair_non_manifold_edges(mesh, &old_v, &new_v)) return 0;
    fprintf(stderr, "vkmesh: %s repair_non_manifold_edges vertices=%d->%d faces=%" PRId64 "\n", label, old_v, new_v, mesh->n_faces);
    return 1;
}

static int vkmesh_log_remove_small_connected_components(vkmesh_mesh * mesh, float min_component_area, const char * label) {
    int removed = 0;
    if (!vkmesh_remove_small_connected_components(mesh, min_component_area, &removed)) return 0;
    fprintf(stderr, "vkmesh: %s remove_small_connected_components removed=%d faces=%" PRId64 "\n", label, removed, mesh->n_faces);
    return 1;
}

static int vkmesh_log_simplify(
    vkmesh_mesh * mesh,
    int target_faces,
    float lambda_edge_length,
    float lambda_skinny,
    float simplify_threshold,
    int simplify_steps,
    const char * label);

static int vkmesh_log_small_components_fill_holes_device_cluster(
    vkmesh_mesh * mesh,
    float min_component_area,
    float max_hole_perimeter,
    const char * label) {
    if (mesh == NULL || label == NULL ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_device_mesh dm;
    memset(&dm, 0, sizeof(dm));
    uint32_t removed = 0u;
    uint32_t filled = 0u;
    uint32_t added_faces = 0u;
    uint32_t after_small_faces = 0u;
    uint32_t before_fill_vertices = 0u;
    uint32_t before_fill_faces = 0u;
    uint32_t after_fill_vertices = 0u;
    uint32_t after_fill_faces = 0u;
    int ok = 0;

    if (!vkmesh_device_mesh_upload(vk, mesh, &dm)) goto cleanup;
    if (!vkmesh_device_remove_small_connected_components(&dm, min_component_area, &removed)) goto cleanup;
    after_small_faces = dm.n_faces;
    before_fill_vertices = dm.n_vertices;
    before_fill_faces = dm.n_faces;
    if (!vkmesh_device_fill_holes(&dm, max_hole_perimeter, &filled, &added_faces)) goto cleanup;
    after_fill_vertices = dm.n_vertices;
    after_fill_faces = dm.n_faces;
    if (filled > (uint32_t) INT_MAX || added_faces > (uint32_t) INT_MAX) goto cleanup;
    if (!vkmesh_device_mesh_download(&dm, mesh)) goto cleanup;

    fprintf(stderr,
        "vkmesh: %s remove_small_connected_components removed=%u faces=%u\n",
        label,
        removed,
        after_small_faces);
    fprintf(stderr,
        "vkmesh: %s fill_holes loops=%u added_vertices=%u added_faces=%u faces=%u->%u\n",
        label,
        filled,
        after_fill_vertices - before_fill_vertices,
        added_faces,
        before_fill_faces,
        after_fill_faces);
    ok = 1;

cleanup:
    vkmesh_device_mesh_destroy(&dm);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

typedef struct vkmesh_trellis_cleanup_stats {
    uint32_t duplicate_removed;
    uint32_t duplicate_faces;
    uint32_t degenerate_removed;
    uint32_t degenerate_faces;
    uint32_t repair_old_vertices;
    uint32_t repair_new_vertices;
    uint32_t repair_faces;
    uint32_t small_removed;
    uint32_t small_faces;
    uint32_t filled;
    uint32_t fill_added_faces;
    uint32_t before_fill_vertices;
    uint32_t before_fill_faces;
    uint32_t after_fill_vertices;
    uint32_t after_fill_faces;
    uint32_t flipped_faces;
} vkmesh_trellis_cleanup_stats;

static int vkmesh_device_trellis_cleanup(
    vkmesh_device_mesh * dm,
    int run_degenerate_cleanup,
    float degenerate_abs,
    float degenerate_rel,
    float min_component_area,
    float max_hole_perimeter,
    int run_final_orientation,
    vkmesh_trellis_cleanup_stats * stats) {
    if (dm == NULL || stats == NULL) return 0;
    memset(stats, 0, sizeof(*stats));

    if (!vkmesh_device_remove_duplicate_faces(dm, &stats->duplicate_removed)) return 0;
    stats->duplicate_faces = dm->n_faces;

    if (run_degenerate_cleanup) {
        if (!vkmesh_device_remove_degenerate_faces(dm, degenerate_abs, degenerate_rel, &stats->degenerate_removed)) return 0;
        stats->degenerate_faces = dm->n_faces;
    }

    if (!vkmesh_device_repair_non_manifold_edges(dm, &stats->repair_old_vertices, &stats->repair_new_vertices)) return 0;
    stats->repair_faces = dm->n_faces;

    if (!vkmesh_device_remove_small_connected_components(dm, min_component_area, &stats->small_removed)) return 0;
    stats->small_faces = dm->n_faces;

    stats->before_fill_vertices = dm->n_vertices;
    stats->before_fill_faces = dm->n_faces;
    if (!vkmesh_device_fill_holes(dm, max_hole_perimeter, &stats->filled, &stats->fill_added_faces)) return 0;
    stats->after_fill_vertices = dm->n_vertices;
    stats->after_fill_faces = dm->n_faces;

    if (run_final_orientation) {
        if (!vkmesh_device_unify_face_orientations(dm, &stats->flipped_faces)) return 0;
    }
    return 1;
}

static void vkmesh_log_trellis_cleanup_stats(
    const char * label,
    int run_degenerate_cleanup,
    int run_final_orientation,
    const char * orientation_label,
    const vkmesh_trellis_cleanup_stats * stats) {
    fprintf(stderr, "vkmesh: %s remove_duplicate_faces removed=%u faces=%u\n", label, stats->duplicate_removed, stats->duplicate_faces);
    if (run_degenerate_cleanup) {
        fprintf(stderr, "vkmesh: %s remove_degenerate_faces removed=%u faces=%u\n", label, stats->degenerate_removed, stats->degenerate_faces);
    }
    fprintf(stderr,
        "vkmesh: %s repair_non_manifold_edges vertices=%u->%u faces=%u\n",
        label,
        stats->repair_old_vertices,
        stats->repair_new_vertices,
        stats->repair_faces);
    fprintf(stderr, "vkmesh: %s remove_small_connected_components removed=%u faces=%u\n", label, stats->small_removed, stats->small_faces);
    fprintf(stderr,
        "vkmesh: %s fill_holes loops=%u added_vertices=%u added_faces=%u faces=%u->%u\n",
        label,
        stats->filled,
        stats->after_fill_vertices - stats->before_fill_vertices,
        stats->fill_added_faces,
        stats->before_fill_faces,
        stats->after_fill_faces);
    if (run_final_orientation) {
        fprintf(stderr,
            "vkmesh: %s unify_face_orientations flipped=%u\n",
            orientation_label != NULL ? orientation_label : label,
            stats->flipped_faces);
    }
}

static int vkmesh_log_trellis_cleanup_device_cluster(
    vkmesh_mesh * mesh,
    int run_degenerate_cleanup,
    float degenerate_abs,
    float degenerate_rel,
    float min_component_area,
    float max_hole_perimeter,
    int run_final_orientation,
    const char * orientation_label,
    const char * label) {
    if (mesh == NULL || label == NULL ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_device_mesh dm;
    memset(&dm, 0, sizeof(dm));
    uint32_t duplicate_removed = 0u;
    uint32_t duplicate_faces = 0u;
    uint32_t degenerate_removed = 0u;
    uint32_t degenerate_faces = 0u;
    uint32_t repair_old_vertices = 0u;
    uint32_t repair_new_vertices = 0u;
    uint32_t repair_faces = 0u;
    uint32_t small_removed = 0u;
    uint32_t small_faces = 0u;
    uint32_t filled = 0u;
    uint32_t fill_added_faces = 0u;
    uint32_t before_fill_vertices = 0u;
    uint32_t before_fill_faces = 0u;
    uint32_t after_fill_vertices = 0u;
    uint32_t after_fill_faces = 0u;
    uint32_t flipped_faces = 0u;
    int ok = 0;

    if (!vkmesh_device_mesh_upload(vk, mesh, &dm)) goto cleanup;

    if (!vkmesh_device_remove_duplicate_faces(&dm, &duplicate_removed)) goto cleanup;
    duplicate_faces = dm.n_faces;

    if (run_degenerate_cleanup) {
        if (!vkmesh_device_remove_degenerate_faces(&dm, degenerate_abs, degenerate_rel, &degenerate_removed)) goto cleanup;
        degenerate_faces = dm.n_faces;
    }

    if (!vkmesh_device_repair_non_manifold_edges(&dm, &repair_old_vertices, &repair_new_vertices)) goto cleanup;
    repair_faces = dm.n_faces;

    if (!vkmesh_device_remove_small_connected_components(&dm, min_component_area, &small_removed)) goto cleanup;
    small_faces = dm.n_faces;

    before_fill_vertices = dm.n_vertices;
    before_fill_faces = dm.n_faces;
    if (!vkmesh_device_fill_holes(&dm, max_hole_perimeter, &filled, &fill_added_faces)) goto cleanup;
    after_fill_vertices = dm.n_vertices;
    after_fill_faces = dm.n_faces;
    if (filled > (uint32_t) INT_MAX || fill_added_faces > (uint32_t) INT_MAX) goto cleanup;

    if (run_final_orientation) {
        if (!vkmesh_device_unify_face_orientations(&dm, &flipped_faces)) goto cleanup;
        if (flipped_faces > (uint32_t) INT_MAX) goto cleanup;
    }

    if (!vkmesh_device_mesh_download(&dm, mesh)) goto cleanup;

    fprintf(stderr, "vkmesh: %s remove_duplicate_faces removed=%u faces=%u\n", label, duplicate_removed, duplicate_faces);
    if (run_degenerate_cleanup) {
        fprintf(stderr, "vkmesh: %s remove_degenerate_faces removed=%u faces=%u\n", label, degenerate_removed, degenerate_faces);
    }
    fprintf(stderr,
        "vkmesh: %s repair_non_manifold_edges vertices=%u->%u faces=%u\n",
        label,
        repair_old_vertices,
        repair_new_vertices,
        repair_faces);
    fprintf(stderr, "vkmesh: %s remove_small_connected_components removed=%u faces=%u\n", label, small_removed, small_faces);
    fprintf(stderr,
        "vkmesh: %s fill_holes loops=%u added_vertices=%u added_faces=%u faces=%u->%u\n",
        label,
        filled,
        after_fill_vertices - before_fill_vertices,
        fill_added_faces,
        before_fill_faces,
        after_fill_faces);
    if (run_final_orientation) {
        fprintf(stderr,
            "vkmesh: %s unify_face_orientations flipped=%u\n",
            orientation_label != NULL ? orientation_label : label,
            flipped_faces);
    }
    ok = 1;

cleanup:
    vkmesh_device_mesh_destroy(&dm);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_trellis_postprocess_device_inner(
    vkmesh_mesh * mesh,
    const char * projection_output,
    vkmesh_mesh * projection_mesh_out,
    int decimation_target,
    float max_hole_perimeter,
    float degenerate_abs,
    float degenerate_rel,
    float min_component_area,
    float lambda_edge_length,
    float lambda_skinny,
    float simplify_threshold,
    int simplify_steps,
    int run_degenerate_cleanup,
    int remesh,
    int remesh_resolution,
    float remesh_band,
    float remesh_project) {
    if (mesh == NULL || mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }
    if (decimation_target <= 0) decimation_target = 1000000;

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_device_mesh dm;
    memset(&dm, 0, sizeof(dm));
    vkmesh_mesh projection_candidate;
    memset(&projection_candidate, 0, sizeof(projection_candidate));
    vkmesh_mesh remesh_work;
    memset(&remesh_work, 0, sizeof(remesh_work));
    int ok = 0;
    if (!vkmesh_device_mesh_upload(vk, mesh, &dm)) goto cleanup;

    fprintf(stderr,
        "vkmesh: trellis_postprocess begin target=%d max_hole_perimeter=%.9g min_component_area=%.9g\n",
        decimation_target,
        max_hole_perimeter,
        min_component_area);

    uint32_t filled = 0u;
    uint32_t added_faces = 0u;
    uint32_t before_v = dm.n_vertices;
    uint32_t before_f = dm.n_faces;
    if (!vkmesh_device_fill_holes(&dm, max_hole_perimeter, &filled, &added_faces)) goto cleanup;
    fprintf(stderr,
        "vkmesh: trellis.pre fill_holes loops=%u added_vertices=%u added_faces=%u faces=%u->%u\n",
        filled,
        dm.n_vertices - before_v,
        added_faces,
        before_f,
        dm.n_faces);
    if ((projection_output != NULL && projection_output[0] != '\0') || projection_mesh_out != NULL) {
        if (!vkmesh_device_mesh_download(&dm, &projection_candidate)) {
            goto cleanup;
        }
        if (projection_output != NULL && projection_output[0] != '\0') {
            int wrote_projection = write_meshbin(projection_output, &projection_candidate);
            fprintf(stderr,
                "vkmesh: trellis.pre projection_mesh_output %s vertices=%" PRId64 " faces=%" PRId64 "\n",
                projection_output,
                projection_candidate.n_vertices,
                projection_candidate.n_faces);
            if (!wrote_projection) goto cleanup;
        }
    }

    if (remesh) {
        if (projection_candidate.vertices != NULL) {
            vkmesh_device_mesh_destroy(&dm);
            if (!vkmesh_remesh_narrow_band_dc_to(
                    &projection_candidate,
                    remesh_resolution,
                    remesh_band,
                    remesh_project,
                    &remesh_work)) goto cleanup;
        } else {
            if (!vkmesh_device_mesh_download(&dm, &remesh_work)) goto cleanup;
            vkmesh_device_mesh_destroy(&dm);
            if (!vkmesh_remesh_narrow_band_dc(
                    &remesh_work, remesh_resolution, remesh_band, remesh_project)) goto cleanup;
        }
        if (remesh_work.n_faces > (int64_t) decimation_target) {
            if (!vkmesh_log_simplify(
                    &remesh_work,
                    decimation_target,
                    lambda_edge_length,
                    lambda_skinny,
                    simplify_threshold,
                    simplify_steps,
                    "trellis.remesh")) {
                goto cleanup;
            }
        } else {
            fprintf(stderr, "vkmesh: trellis.remesh simplify skipped faces=%" PRId64 " target=%d\n", remesh_work.n_faces, decimation_target);
        }
        mesh_free(mesh);
        *mesh = remesh_work;
        memset(&remesh_work, 0, sizeof(remesh_work));
        if (projection_mesh_out != NULL && projection_candidate.vertices != NULL) {
            *projection_mesh_out = projection_candidate;
            memset(&projection_candidate, 0, sizeof(projection_candidate));
        }
        fprintf(stderr, "vkmesh: trellis_postprocess done vertices=%" PRId64 " faces=%" PRId64 "\n", mesh->n_vertices, mesh->n_faces);
        ok = 1;
        goto cleanup;
    }

    int64_t first_target = (int64_t) decimation_target * 3;
    if (first_target > INT_MAX) first_target = INT_MAX;
    if (dm.n_faces > (uint32_t) first_target) {
        int collapsed = 0;
        int removed = 0;
        fprintf(stderr, "vkmesh: trellis.pass1 simplify begin target=%" PRId64 " faces=%u\n", first_target, dm.n_faces);
        fflush(stderr);
        if (!vkmesh_device_simplify(
                &dm,
                (int) first_target,
                lambda_edge_length,
                lambda_skinny,
                simplify_threshold,
                simplify_steps,
                &collapsed,
                &removed)) {
            goto cleanup;
        }
        fprintf(stderr,
            "vkmesh: trellis.pass1 simplify target=%" PRId64 " collapsed=%d removed_faces=%d faces=%u\n",
            first_target,
            collapsed,
            removed,
            dm.n_faces);
    } else {
        fprintf(stderr, "vkmesh: trellis.pass1 simplify skipped faces=%u target=%" PRId64 "\n", dm.n_faces, first_target);
    }

    vkmesh_trellis_cleanup_stats pass1_stats;
    if (!vkmesh_device_trellis_cleanup(
            &dm,
            run_degenerate_cleanup,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            max_hole_perimeter,
            0,
            &pass1_stats)) {
        goto cleanup;
    }
    vkmesh_log_trellis_cleanup_stats("trellis.pass1", run_degenerate_cleanup, 0, NULL, &pass1_stats);

    if (dm.n_faces > (uint32_t) decimation_target) {
        int collapsed = 0;
        int removed = 0;
        fprintf(stderr, "vkmesh: trellis.pass2 simplify begin target=%d faces=%u\n", decimation_target, dm.n_faces);
        fflush(stderr);
        if (!vkmesh_device_simplify(
                &dm,
                decimation_target,
                lambda_edge_length,
                lambda_skinny,
                simplify_threshold,
                simplify_steps,
                &collapsed,
                &removed)) {
            goto cleanup;
        }
        fprintf(stderr,
            "vkmesh: trellis.pass2 simplify target=%d collapsed=%d removed_faces=%d faces=%u\n",
            decimation_target,
            collapsed,
            removed,
            dm.n_faces);
    } else {
        fprintf(stderr, "vkmesh: trellis.pass2 simplify skipped faces=%u target=%d\n", dm.n_faces, decimation_target);
    }

    vkmesh_trellis_cleanup_stats pass2_stats;
    if (!vkmesh_device_trellis_cleanup(
            &dm,
            run_degenerate_cleanup,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            max_hole_perimeter,
            1,
            &pass2_stats)) {
        goto cleanup;
    }
    vkmesh_log_trellis_cleanup_stats("trellis.pass2", run_degenerate_cleanup, 1, "trellis.final", &pass2_stats);

    if (!vkmesh_device_mesh_download(&dm, mesh)) goto cleanup;
    if (projection_mesh_out != NULL && projection_candidate.vertices != NULL) {
        *projection_mesh_out = projection_candidate;
        memset(&projection_candidate, 0, sizeof(projection_candidate));
    }
    fprintf(stderr, "vkmesh: trellis_postprocess done vertices=%" PRId64 " faces=%" PRId64 "\n", mesh->n_vertices, mesh->n_faces);
    ok = 1;

cleanup:
    mesh_free(&remesh_work);
    mesh_free(&projection_candidate);
    vkmesh_device_mesh_destroy(&dm);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_log_unify_face_orientations(vkmesh_mesh * mesh, const char * label) {
    int flipped = 0;
    if (!vkmesh_unify_face_orientations(mesh, &flipped)) return 0;
    fprintf(stderr, "vkmesh: %s unify_face_orientations flipped=%d\n", label, flipped);
    return 1;
}

static int vkmesh_log_simplify(
    vkmesh_mesh * mesh,
    int target_faces,
    float lambda_edge_length,
    float lambda_skinny,
    float simplify_threshold,
    int simplify_steps,
    const char * label) {
    int collapsed = 0;
    int removed = 0;
    if (target_faces <= 0) target_faces = 1;
    fprintf(stderr,
        "vkmesh: %s simplify begin target=%d faces=%" PRId64 "\n",
        label,
        target_faces,
        mesh->n_faces);
    fflush(stderr);
    if (!vkmesh_simplify(
            mesh,
            target_faces,
            lambda_edge_length,
            lambda_skinny,
            simplify_threshold,
            simplify_steps,
            &collapsed,
            &removed)) {
        return 0;
    }
    fprintf(stderr,
        "vkmesh: %s simplify target=%d collapsed=%d removed_faces=%d faces=%" PRId64 "\n",
        label,
        target_faces,
        collapsed,
        removed,
        mesh->n_faces);
    return 1;
}

static int vkmesh_trellis_postprocess_inner(
    vkmesh_mesh * mesh,
    const char * projection_output,
    vkmesh_mesh * projection_mesh_out,
    int decimation_target,
    float max_hole_perimeter,
    float degenerate_abs,
    float degenerate_rel,
    float min_component_area,
    float lambda_edge_length,
    float lambda_skinny,
    float simplify_threshold,
    int simplify_steps,
    int run_degenerate_cleanup,
    int remesh,
    int remesh_resolution,
    float remesh_band,
    float remesh_project) {
    if (vkmesh_trellis_postprocess_device_inner(
            mesh,
            projection_output,
            projection_mesh_out,
            decimation_target,
            max_hole_perimeter,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            lambda_edge_length,
            lambda_skinny,
            simplify_threshold,
            simplify_steps,
            run_degenerate_cleanup,
            remesh,
            remesh_resolution,
            remesh_band,
            remesh_project)) {
        return 1;
    }

    if (decimation_target <= 0) decimation_target = 1000000;
    fprintf(stderr,
        "vkmesh: trellis_postprocess begin target=%d max_hole_perimeter=%.9g min_component_area=%.9g\n",
        decimation_target,
        max_hole_perimeter,
        min_component_area);

    if (!vkmesh_log_fill_holes(mesh, max_hole_perimeter, "trellis.pre")) return 0;
    if ((projection_output != NULL && projection_output[0] != '\0') || projection_mesh_out != NULL) {
        if (projection_output != NULL && projection_output[0] != '\0') {
            if (!write_meshbin(projection_output, mesh)) return 0;
            fprintf(stderr,
                "vkmesh: trellis.pre projection_mesh_output %s vertices=%" PRId64 " faces=%" PRId64 "\n",
                projection_output,
                mesh->n_vertices,
                mesh->n_faces);
        }
        if (projection_mesh_out != NULL && !mesh_clone(mesh, projection_mesh_out)) return 0;
    }

    if (remesh) {
        if (!vkmesh_remesh_narrow_band_dc(mesh, remesh_resolution, remesh_band, remesh_project)) return 0;
        if (mesh->n_faces > decimation_target) {
            if (!vkmesh_log_simplify(
                    mesh,
                    decimation_target,
                    lambda_edge_length,
                    lambda_skinny,
                    simplify_threshold,
                    simplify_steps,
                    "trellis.remesh")) {
                return 0;
            }
        } else {
            fprintf(stderr, "vkmesh: trellis.remesh simplify skipped faces=%" PRId64 " target=%d\n", mesh->n_faces, decimation_target);
        }
        fprintf(stderr, "vkmesh: trellis_postprocess done vertices=%" PRId64 " faces=%" PRId64 "\n", mesh->n_vertices, mesh->n_faces);
        return 1;
    }

    int64_t first_target = (int64_t) decimation_target * 3;
    if (first_target > INT_MAX) first_target = INT_MAX;
    if (mesh->n_faces > first_target) {
        if (!vkmesh_log_simplify(
                mesh,
                (int) first_target,
                lambda_edge_length,
                lambda_skinny,
                simplify_threshold,
                simplify_steps,
                "trellis.pass1")) {
            return 0;
        }
    } else {
        fprintf(stderr, "vkmesh: trellis.pass1 simplify skipped faces=%" PRId64 " target=%" PRId64 "\n", mesh->n_faces, first_target);
    }

    if (!vkmesh_log_trellis_cleanup_device_cluster(
            mesh,
            run_degenerate_cleanup,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            max_hole_perimeter,
            0,
            NULL,
            "trellis.pass1")) {
        if (!vkmesh_log_duplicate_degenerate_device_cluster(
                mesh,
                run_degenerate_cleanup,
                degenerate_abs,
                degenerate_rel,
                "trellis.pass1")) {
            if (!vkmesh_log_remove_duplicate_faces(mesh, "trellis.pass1")) return 0;
            if (run_degenerate_cleanup && !vkmesh_log_remove_degenerate_faces(mesh, degenerate_abs, degenerate_rel, "trellis.pass1")) return 0;
        }
        if (!vkmesh_log_repair_non_manifold_edges(mesh, "trellis.pass1")) return 0;
        if (!vkmesh_log_remove_small_connected_components(mesh, min_component_area, "trellis.pass1")) return 0;
        if (!vkmesh_log_fill_holes(mesh, max_hole_perimeter, "trellis.pass1")) return 0;
    }

    if (mesh->n_faces > decimation_target) {
        if (!vkmesh_log_simplify(
                mesh,
                decimation_target,
                lambda_edge_length,
                lambda_skinny,
                simplify_threshold,
                simplify_steps,
                "trellis.pass2")) {
            return 0;
        }
    } else {
        fprintf(stderr, "vkmesh: trellis.pass2 simplify skipped faces=%" PRId64 " target=%d\n", mesh->n_faces, decimation_target);
    }

    int final_orientation_done = 0;
    if (vkmesh_log_trellis_cleanup_device_cluster(
            mesh,
            run_degenerate_cleanup,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            max_hole_perimeter,
            1,
            "trellis.final",
            "trellis.pass2")) {
        final_orientation_done = 1;
    } else {
        if (!vkmesh_log_duplicate_degenerate_device_cluster(
                mesh,
                run_degenerate_cleanup,
                degenerate_abs,
                degenerate_rel,
                "trellis.pass2")) {
            if (!vkmesh_log_remove_duplicate_faces(mesh, "trellis.pass2")) return 0;
            if (run_degenerate_cleanup && !vkmesh_log_remove_degenerate_faces(mesh, degenerate_abs, degenerate_rel, "trellis.pass2")) return 0;
        }
        if (!vkmesh_log_repair_non_manifold_edges(mesh, "trellis.pass2")) return 0;
        if (!vkmesh_log_remove_small_connected_components(mesh, min_component_area, "trellis.pass2")) return 0;
        if (!vkmesh_log_fill_holes(mesh, max_hole_perimeter, "trellis.pass2")) return 0;
    }
    if (!final_orientation_done && !vkmesh_log_unify_face_orientations(mesh, "trellis.final")) return 0;

    fprintf(stderr, "vkmesh: trellis_postprocess done vertices=%" PRId64 " faces=%" PRId64 "\n", mesh->n_vertices, mesh->n_faces);
    return 1;
}

static int vkmesh_trellis_postprocess(
    vkmesh_mesh * mesh,
    const char * projection_output,
    vkmesh_mesh * projection_mesh_out,
    int decimation_target,
    float max_hole_perimeter,
    float degenerate_abs,
    float degenerate_rel,
    float min_component_area,
    float lambda_edge_length,
    float lambda_skinny,
    float simplify_threshold,
    int simplify_steps,
    int run_degenerate_cleanup,
    int remesh,
    int remesh_resolution,
    float remesh_band,
    float remesh_project) {
    if (g_active_vkmesh_vk != NULL) {
        return vkmesh_trellis_postprocess_inner(
            mesh,
            projection_output,
            projection_mesh_out,
            decimation_target,
            max_hole_perimeter,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            lambda_edge_length,
            lambda_skinny,
            simplify_threshold,
            simplify_steps,
            run_degenerate_cleanup,
            remesh,
            remesh_resolution,
            remesh_band,
            remesh_project);
    }

    vkmesh_vk vk;
    if (!vkmesh_vk_init(&vk)) {
        vkmesh_vk_destroy(&vk);
        return 0;
    }
    g_active_vkmesh_vk = &vk;
    fprintf(stderr, "vkmesh: trellis_postprocess using one persistent Vulkan context\n");
    int ok = vkmesh_trellis_postprocess_inner(
        mesh,
        projection_output,
        projection_mesh_out,
        decimation_target,
        max_hole_perimeter,
        degenerate_abs,
        degenerate_rel,
        min_component_area,
        lambda_edge_length,
        lambda_skinny,
        simplify_threshold,
        simplify_steps,
        run_degenerate_cleanup,
        remesh,
        remesh_resolution,
        remesh_band,
        remesh_project);
    g_active_vkmesh_vk = NULL;
    vkmesh_vk_destroy(&vk);
    return ok;
}

#ifdef TRELLIS_VKMESH_LIBRARY
static trellis_status vkmesh_from_trellis_mesh(const trellis_mesh_host * src, vkmesh_mesh * dst) {
    if (src == NULL || dst == NULL || src->vertices == NULL || src->faces == NULL ||
        src->n_vertices <= 0 || src->n_faces <= 0 ||
        src->n_vertices > INT32_MAX ||
        (uint64_t) src->n_vertices > (uint64_t) SIZE_MAX / (3u * sizeof(float)) ||
        (uint64_t) src->n_faces > (uint64_t) SIZE_MAX / (3u * sizeof(int32_t))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    vkmesh_mesh out;
    memset(&out, 0, sizeof(out));
    const size_t vertex_count = (size_t) src->n_vertices * 3u;
    const size_t face_count = (size_t) src->n_faces * 3u;
    out.vertices = (float *) malloc(vertex_count * sizeof(float));
    out.faces = (int32_t *) malloc(face_count * sizeof(int32_t));
    if (out.vertices == NULL || out.faces == NULL) {
        mesh_free(&out);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    memcpy(out.vertices, src->vertices, vertex_count * sizeof(float));
    memcpy(out.faces, src->faces, face_count * sizeof(int32_t));
    for (size_t i = 0; i < vertex_count; ++i) {
        if (!isfinite(out.vertices[i])) {
            mesh_free(&out);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }
    for (size_t i = 0; i < face_count; ++i) {
        if (out.faces[i] < 0 || out.faces[i] >= src->n_vertices) {
            mesh_free(&out);
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }
    out.n_vertices = src->n_vertices;
    out.n_faces = src->n_faces;
    out.vertex_capacity = src->n_vertices;
    out.face_capacity = src->n_faces;
    *dst = out;
    return TRELLIS_STATUS_OK;
}

static trellis_status vkmesh_move_to_trellis_mesh(vkmesh_mesh * src, trellis_mesh_host * dst) {
    if (src == NULL || dst == NULL || src->vertices == NULL || src->faces == NULL ||
        src->n_vertices <= 0 || src->n_faces <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(dst, 0, sizeof(*dst));
    dst->vertices = src->vertices;
    dst->faces = src->faces;
    dst->n_vertices = src->n_vertices;
    dst->n_faces = src->n_faces;
    free(src->uvs);
    memset(src, 0, sizeof(*src));
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_vkmesh_postprocess(
    const trellis_mesh_host * mesh,
    trellis_mesh_host * mesh_out,
    trellis_mesh_host * projection_mesh_out,
    const trellis_vkmesh_postprocess_options * options) {
    if (mesh == NULL || mesh_out == NULL || mesh_out == mesh ||
        projection_mesh_out == mesh || projection_mesh_out == mesh_out) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (options != NULL &&
        (options->device < 0 || options->gpu_workspace_budget_mib < 0 ||
         options->decimation_target < 0 ||
         options->simplify_steps < 0 || options->remesh_resolution < 0 ||
         options->max_hole_perimeter < 0.0f || options->degenerate_abs < 0.0f ||
         options->degenerate_rel < 0.0f || options->min_component_area < 0.0f ||
         options->lambda_edge_length < 0.0f || options->lambda_skinny < 0.0f ||
         options->simplify_threshold < 0.0f || options->remesh_band < 0.0f ||
         options->remesh_project < 0.0f ||
         !isfinite(options->max_hole_perimeter) || !isfinite(options->degenerate_abs) ||
         !isfinite(options->degenerate_rel) || !isfinite(options->min_component_area) ||
         !isfinite(options->lambda_edge_length) || !isfinite(options->lambda_skinny) ||
         !isfinite(options->simplify_threshold) || !isfinite(options->remesh_band) ||
         !isfinite(options->remesh_project))) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(mesh_out, 0, sizeof(*mesh_out));
    if (projection_mesh_out != NULL) {
        memset(projection_mesh_out, 0, sizeof(*projection_mesh_out));
    }

    vkmesh_mesh work;
    vkmesh_mesh projection;
    memset(&work, 0, sizeof(work));
    memset(&projection, 0, sizeof(projection));
    trellis_status status = vkmesh_from_trellis_mesh(mesh, &work);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    const int no_simplify = options != NULL && options->no_simplify;
    int decimation_target = options != NULL ? options->decimation_target : 0;
    if (no_simplify) {
        decimation_target = INT_MAX;
    } else if (decimation_target <= 0) {
        decimation_target = 1000000;
    }

    const float max_hole_perimeter =
        options != NULL && options->max_hole_perimeter > 0.0f ? options->max_hole_perimeter : 3e-2f;
    const float degenerate_abs =
        options != NULL && options->degenerate_abs > 0.0f ? options->degenerate_abs : 1e-24f;
    const float degenerate_rel =
        options != NULL && options->degenerate_rel > 0.0f ? options->degenerate_rel : 1e-12f;
    const float min_component_area =
        options != NULL && options->min_component_area > 0.0f ? options->min_component_area : 1e-5f;
    const float lambda_edge_length =
        options != NULL && options->lambda_edge_length > 0.0f ? options->lambda_edge_length : 1e-2f;
    const float lambda_skinny =
        options != NULL && options->lambda_skinny > 0.0f ? options->lambda_skinny : 1e-3f;
    const float simplify_threshold =
        options != NULL && options->simplify_threshold > 0.0f ? options->simplify_threshold : 1e-8f;
    const int simplify_steps = options != NULL ? options->simplify_steps : 0;
    const int run_degenerate_cleanup = options != NULL && options->run_degenerate_cleanup;
    const int remesh = options != NULL && options->remesh;
    const int remesh_resolution =
        options != NULL && options->remesh_resolution > 0 ? options->remesh_resolution : 1024;
    const float remesh_band =
        options != NULL && options->remesh_band > 0.0f ? options->remesh_band : 1.0f;
    const float remesh_project =
        options != NULL && options->remesh_project > 0.0f ? options->remesh_project : 0.0f;
    const int previous_device = g_vkmesh_device_index;
    const int previous_workspace_budget_mib = g_vkmesh_workspace_budget_mib;
    g_vkmesh_device_index = options != NULL ? options->device : 0;
    g_vkmesh_workspace_budget_mib = options != NULL ? options->gpu_workspace_budget_mib : 0;
    g_vkmesh_last_error = VKMESH_ERROR_NONE;

    if (!vkmesh_trellis_postprocess(
            &work,
            NULL,
            projection_mesh_out != NULL ? &projection : NULL,
            decimation_target,
            max_hole_perimeter,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            lambda_edge_length,
            lambda_skinny,
            simplify_threshold,
            simplify_steps,
            run_degenerate_cleanup,
            remesh,
            remesh_resolution,
            remesh_band,
            remesh_project)) {
        switch (g_vkmesh_last_error) {
            case VKMESH_ERROR_INVALID_ARGUMENT:
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
                break;
            case VKMESH_ERROR_OUT_OF_MEMORY:
                status = TRELLIS_STATUS_OUT_OF_MEMORY;
                break;
            case VKMESH_ERROR_VULKAN_UNAVAILABLE:
                status = TRELLIS_STATUS_CUDA_UNAVAILABLE;
                break;
            default:
                status = TRELLIS_STATUS_ERROR;
                break;
        }
        goto cleanup;
    }

    status = vkmesh_move_to_trellis_mesh(&work, mesh_out);
    if (status != TRELLIS_STATUS_OK) {
        goto cleanup;
    }
    if (projection_mesh_out != NULL && projection.vertices != NULL) {
        status = vkmesh_move_to_trellis_mesh(&projection, projection_mesh_out);
        if (status != TRELLIS_STATUS_OK) {
            goto cleanup;
        }
    }

cleanup:
    g_vkmesh_device_index = previous_device;
    g_vkmesh_workspace_budget_mib = previous_workspace_budget_mib;
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_free(mesh_out);
        if (projection_mesh_out != NULL) {
            trellis_mesh_free(projection_mesh_out);
        }
    }
    mesh_free(&projection);
    mesh_free(&work);
    return status;
}
#endif

#ifndef TRELLIS_VKMESH_LIBRARY
typedef struct vkmesh_vulkan_probe_device {
    uint32_t physical_index;
    VkPhysicalDeviceProperties properties;
    uint8_t uuid[VK_UUID_SIZE];
    char driver_name[VK_MAX_DRIVER_NAME_SIZE];
    char driver_info[VK_MAX_DRIVER_INFO_SIZE];
    uint64_t device_local_memory_bytes;
    int has_compute_queue;
    int storage_buffer_16_bit_access;
    int usable;
} vkmesh_vulkan_probe_device;

static void vkmesh_print_json_string(const char * value) {
    const unsigned char * cursor = (const unsigned char *) (value != NULL ? value : "");
    putchar('"');
    while (*cursor != '\0') {
        switch (*cursor) {
            case '"': fputs("\\\"", stdout); break;
            case '\\': fputs("\\\\", stdout); break;
            case '\b': fputs("\\b", stdout); break;
            case '\f': fputs("\\f", stdout); break;
            case '\n': fputs("\\n", stdout); break;
            case '\r': fputs("\\r", stdout); break;
            case '\t': fputs("\\t", stdout); break;
            default:
                if (*cursor < 0x20u) {
                    fprintf(stdout, "\\u%04x", (unsigned int) *cursor);
                } else {
                    putchar((int) *cursor);
                }
                break;
        }
        ++cursor;
    }
    putchar('"');
}

static void vkmesh_format_vulkan_version(uint32_t version, char output[32]) {
    snprintf(
        output,
        32,
        "%u.%u.%u",
        VK_API_VERSION_MAJOR(version),
        VK_API_VERSION_MINOR(version),
        VK_API_VERSION_PATCH(version));
}

static const char * vkmesh_vulkan_device_type_name(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: return "other";
    }
}

static int vkmesh_vulkan_device_is_hardware(VkPhysicalDeviceType type) {
    return type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
        type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
        type == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU;
}

static int vkmesh_vulkan_device_has_compute_queue(VkPhysicalDevice physical_device) {
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, NULL);
    if (family_count == 0) return 0;
    VkQueueFamilyProperties * families =
        (VkQueueFamilyProperties *) calloc((size_t) family_count, sizeof(*families));
    if (families == NULL) return 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, families);
    int found = 0;
    for (uint32_t i = 0; i < family_count; ++i) {
        if (families[i].queueCount > 0 && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            found = 1;
            break;
        }
    }
    free(families);
    return found;
}

static uint64_t vkmesh_vulkan_device_local_memory(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceMemoryProperties memory;
    memset(&memory, 0, sizeof(memory));
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory);
    uint64_t largest = 0;
    for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
        if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        const uint64_t size = (uint64_t) memory.memoryHeaps[i].size;
        if (size > largest) largest = size;
    }
    return largest;
}

static void vkmesh_vulkan_probe_uuid(const uint8_t uuid[VK_UUID_SIZE], char output[VK_UUID_SIZE * 2 + 1]) {
    static const char digits[] = "0123456789abcdef";
    for (uint32_t i = 0; i < VK_UUID_SIZE; ++i) {
        output[i * 2u] = digits[uuid[i] >> 4u];
        output[i * 2u + 1u] = digits[uuid[i] & 0x0fu];
    }
    output[VK_UUID_SIZE * 2u] = '\0';
}

static const char * vkmesh_vulkan_probe_reason(
    const vkmesh_vulkan_probe_device * devices,
    uint32_t device_count) {
    int has_hardware = 0;
    int has_compute = 0;
    int has_vulkan_12_device = 0;
    for (uint32_t i = 0; i < device_count; ++i) {
        if (!vkmesh_vulkan_device_is_hardware(devices[i].properties.deviceType)) continue;
        has_hardware = 1;
        if (!devices[i].has_compute_queue) continue;
        has_compute = 1;
        if (devices[i].properties.apiVersion < VK_API_VERSION_1_2) continue;
        has_vulkan_12_device = 1;
        if (!devices[i].storage_buffer_16_bit_access) continue;
        return NULL;
    }
    if (!has_hardware) return "no_hardware_device";
    if (!has_compute) return "no_compute_queue";
    if (!has_vulkan_12_device) return "device_api_too_old";
    return "missing_storage_buffer_16_bit_access";
}

static int vkmesh_probe_vulkan_json(void) {
    uint32_t loader_version = VK_API_VERSION_1_0;
    PFN_vkEnumerateInstanceVersion enumerate_instance_version =
        (PFN_vkEnumerateInstanceVersion) vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    if (enumerate_instance_version != NULL && enumerate_instance_version(&loader_version) != VK_SUCCESS) {
        loader_version = VK_API_VERSION_1_0;
    }
    char loader_version_text[32];
    vkmesh_format_vulkan_version(loader_version, loader_version_text);
    if (loader_version < VK_API_VERSION_1_2) {
        fputs("{\"schemaVersion\":1,\"status\":\"unsupported\",\"loaderApiVersion\":", stdout);
        vkmesh_print_json_string(loader_version_text);
        fputs(",\"reasonCode\":\"loader_api_too_old\",\"devices\":[]}\n", stdout);
        return 0;
    }
    VkApplicationInfo application_info;
    memset(&application_info, 0, sizeof(application_info));
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pApplicationName = "Trellis Studio Vulkan probe";
    application_info.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application_info;
    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&instance_info, NULL, &instance);
    if (result != VK_SUCCESS) {
        fputs("{\"schemaVersion\":1,\"status\":\"unsupported\",\"loaderApiVersion\":", stdout);
        vkmesh_print_json_string(loader_version_text);
        fputs(",\"reasonCode\":\"instance_create_failed\",\"devices\":[]}\n", stdout);
        return 0;
    }

    uint32_t physical_count = 0;
    result = vkEnumeratePhysicalDevices(instance, &physical_count, NULL);
    if (result != VK_SUCCESS || physical_count == 0) {
        vkDestroyInstance(instance, NULL);
        fputs("{\"schemaVersion\":1,\"status\":\"unsupported\",\"loaderApiVersion\":", stdout);
        vkmesh_print_json_string(loader_version_text);
        fputs(",\"reasonCode\":\"no_physical_device\",\"devices\":[]}\n", stdout);
        return 0;
    }

    VkPhysicalDevice * physical_devices =
        (VkPhysicalDevice *) calloc((size_t) physical_count, sizeof(*physical_devices));
    vkmesh_vulkan_probe_device * devices =
        (vkmesh_vulkan_probe_device *) calloc((size_t) physical_count, sizeof(*devices));
    if (physical_devices == NULL || devices == NULL) {
        free(physical_devices);
        free(devices);
        vkDestroyInstance(instance, NULL);
        fputs("vkmesh: Vulkan probe ran out of memory\n", stderr);
        return 1;
    }
    result = vkEnumeratePhysicalDevices(instance, &physical_count, physical_devices);
    if (result != VK_SUCCESS) {
        free(physical_devices);
        free(devices);
        vkDestroyInstance(instance, NULL);
        fputs("vkmesh: Vulkan device enumeration failed\n", stderr);
        return 1;
    }

    uint32_t usable_count = 0;
    for (uint32_t i = 0; i < physical_count; ++i) {
        vkmesh_vulkan_probe_device * probe = &devices[i];
        probe->physical_index = i;
        VkPhysicalDeviceIDProperties id_properties;
        VkPhysicalDeviceDriverProperties driver_properties;
        VkPhysicalDeviceProperties2 properties2;
        memset(&id_properties, 0, sizeof(id_properties));
        memset(&driver_properties, 0, sizeof(driver_properties));
        memset(&properties2, 0, sizeof(properties2));
        id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        id_properties.pNext = &driver_properties;
        driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &id_properties;
        vkGetPhysicalDeviceProperties2(physical_devices[i], &properties2);
        probe->properties = properties2.properties;
        memcpy(probe->uuid, id_properties.deviceUUID, sizeof(probe->uuid));
        memcpy(probe->driver_name, driver_properties.driverName, sizeof(probe->driver_name));
        probe->driver_name[sizeof(probe->driver_name) - 1u] = '\0';
        memcpy(probe->driver_info, driver_properties.driverInfo, sizeof(probe->driver_info));
        probe->driver_info[sizeof(probe->driver_info) - 1u] = '\0';

        VkPhysicalDeviceVulkan11Features features11;
        VkPhysicalDeviceFeatures2 features2;
        memset(&features11, 0, sizeof(features11));
        memset(&features2, 0, sizeof(features2));
        features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &features11;
        vkGetPhysicalDeviceFeatures2(physical_devices[i], &features2);
        probe->has_compute_queue = vkmesh_vulkan_device_has_compute_queue(physical_devices[i]);
        probe->storage_buffer_16_bit_access = features11.storageBuffer16BitAccess == VK_TRUE;
        probe->device_local_memory_bytes = vkmesh_vulkan_device_local_memory(physical_devices[i]);
        probe->usable = vkmesh_vulkan_device_is_hardware(probe->properties.deviceType) &&
            probe->properties.apiVersion >= VK_API_VERSION_1_2 &&
            probe->has_compute_queue &&
            probe->storage_buffer_16_bit_access;
        if (probe->usable) ++usable_count;
    }
    free(physical_devices);
    vkDestroyInstance(instance, NULL);

    fputs("{\"schemaVersion\":1,\"status\":", stdout);
    vkmesh_print_json_string(usable_count > 0 ? "supported" : "unsupported");
    fputs(",\"loaderApiVersion\":", stdout);
    vkmesh_print_json_string(loader_version_text);
    if (usable_count == 0) {
        const char * reason = vkmesh_vulkan_probe_reason(devices, physical_count);
        fputs(",\"reasonCode\":", stdout);
        vkmesh_print_json_string(reason != NULL ? reason : "no_usable_device");
    }
    fputs(",\"devices\":[", stdout);
    for (uint32_t i = 0; i < physical_count; ++i) {
        const vkmesh_vulkan_probe_device * probe = &devices[i];
        char uuid[VK_UUID_SIZE * 2 + 1];
        char api_version[32];
        char driver_version[32];
        vkmesh_vulkan_probe_uuid(probe->uuid, uuid);
        vkmesh_format_vulkan_version(probe->properties.apiVersion, api_version);
        snprintf(driver_version, sizeof(driver_version), "%u", probe->properties.driverVersion);
        if (i > 0) putchar(',');
        fprintf(stdout, "{\"physicalIndex\":%u,\"name\":", probe->physical_index);
        vkmesh_print_json_string(probe->properties.deviceName);
        fputs(",\"deviceType\":", stdout);
        vkmesh_print_json_string(vkmesh_vulkan_device_type_name(probe->properties.deviceType));
        fputs(",\"uuid\":", stdout);
        vkmesh_print_json_string(uuid);
        fprintf(stdout, ",\"vendorId\":%u,\"driverName\":", probe->properties.vendorID);
        vkmesh_print_json_string(probe->driver_name);
        fputs(",\"driverInfo\":", stdout);
        vkmesh_print_json_string(probe->driver_info[0] != '\0' ? probe->driver_info : driver_version);
        fputs(",\"apiVersion\":", stdout);
        vkmesh_print_json_string(api_version);
        fprintf(
            stdout,
            ",\"memoryTotalBytes\":%" PRIu64
            ",\"hasComputeQueue\":%s,\"storageBuffer16BitAccess\":%s,\"usable\":%s}",
            probe->device_local_memory_bytes,
            probe->has_compute_queue ? "true" : "false",
            probe->storage_buffer_16_bit_access ? "true" : "false",
            probe->usable ? "true" : "false");
    }
    fputs("]}\n", stdout);
    free(devices);
    return 0;
}

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --input in.meshbin --output out.meshbin [options]\n"
        "       %s --probe-vulkan-json\n"
        "\n"
        "Postprocess stages:\n"
        "  --postprocess                 Run PyTorch/o-voxel TRELLIS postprocess preset\n"
        "  --trellis-postprocess         Alias of --postprocess\n"
        "  --cleanup                     Run one cleanup pass using selected primitive stages\n"
        "  --fill-holes                  Fill small manifold boundary loops (default)\n"
        "  --no-fill-holes               Disable default fill_holes\n"
        "  --remove-duplicate-faces      Remove duplicate triangle index sets\n"
        "  --remove-degenerate-faces     Remove duplicate-vertex / tiny-area faces\n"
        "  --repair-non-manifold-edges   Split vertices across non-manifold sheets\n"
        "  --remove-small-components     Remove components below --min-component-area\n"
        "  --unify-face-orientations     Make winding consistent per manifold component\n"
        "  --simplify                    Run CuMesh-style simplify loop\n"
        "  --no-simplify                 Disable simplify even if --target-faces was set\n"
        "  --remesh                      Run narrow-band dual-contouring remesh\n"
        "  --no-remesh                   Disable remesh in --postprocess preset\n"
        "  --uv-unwrap                   Run xatlas unwrap and store UVs in meshbin\n"
        "  --no-uv-unwrap                Disable UV unwrap after a preset enabled it\n"
        "\n"
        "Parameters:\n"
        "  --max-hole-perimeter X        Default 0.03, matching CuMesh/TRELLIS.2\n"
        "  --degenerate-abs X            Default 1e-24\n"
        "  --degenerate-rel X            Default 1e-12\n"
        "  --min-component-area X        Default 1e-5\n"
        "  --target-faces N              Enables simplify to N faces\n"
        "  --decimation-target N         Alias of --target-faces\n"
        "  --simplify-steps N            0 means no explicit step limit\n"
        "  --simplify-threshold X        Default 1e-8\n"
        "  --lambda-edge-length X        Default 1e-2\n"
        "  --lambda-skinny X             Default 1e-3\n"
        "  --remesh-resolution N         Dual-contouring resolution, default 1024\n"
        "  --remesh-band X               Narrow-band size in voxels, default 1\n"
        "  --remesh-project X            Project remesh vertices back to source, default 0\n"
        "  --texture-size N              xatlas pack resolution, default 1024\n"
        "  --device N                    Vulkan physical-device index, default 0\n"
        "  --gpu-workspace-budget-mib N  VkDeviceMemory workspace cap, default auto (max 2048 MiB)\n"
        "  --unsigned-distance pts.txt   Compute UDF for text points: x y z per line\n"
        "  --distance-output out.txt     Required with --unsigned-distance unless --output is enough for mesh only\n"
        "  --projection-mesh-output FILE With --postprocess, write post-fill source meshbin for texture projection\n"
        "\n",
        argv0,
        argv0);
}

int main(int argc, char ** argv) {
    if (argc == 2 && strcmp(argv[1], "--probe-vulkan-json") == 0) {
        return vkmesh_probe_vulkan_json();
    }
    const char * input = NULL;
    const char * output = NULL;
    const char * projection_output = NULL;
    const char * points_path = NULL;
    const char * distance_output = NULL;
    float max_hole_perimeter = 3e-2f;
    float degenerate_abs = 1e-24f;
    float degenerate_rel = 1e-12f;
    float min_component_area = 1e-5f;
    float simplify_threshold = 1e-8f;
    float lambda_edge_length = 1e-2f;
    float lambda_skinny = 1e-3f;
    float remesh_band = 1.0f;
    float remesh_project = 0.0f;
    int target_faces = 0;
    int simplify_steps = 0;
    int remesh_resolution = 1024;
    int texture_size = 1024;
    int device = 0;
    int gpu_workspace_budget_mib = 0;
    int trellis_postprocess = 0;
    int fill_holes = 1;
    int remove_duplicate_faces = 0;
    int remove_degenerate_faces = 0;
    int repair_non_manifold_edges = 0;
    int remove_small_components = 0;
    int unify_face_orientations = 0;
    int simplify = 0;
    int disable_simplify = 0;
    int simplify_option_seen = 0;
    int remesh = 0;
    int uv_unwrap = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--projection-mesh-output") == 0 && i + 1 < argc) {
            projection_output = argv[++i];
        } else if (strcmp(argv[i], "--postprocess") == 0 || strcmp(argv[i], "--trellis-postprocess") == 0) {
            trellis_postprocess = 1;
            remesh = 1;
            if (!simplify_option_seen) {
                disable_simplify = 1;
            }
            uv_unwrap = 1;
        } else if (strcmp(argv[i], "--cleanup") == 0) {
            fill_holes = 1;
            remove_duplicate_faces = 1;
            remove_degenerate_faces = 1;
            repair_non_manifold_edges = 1;
            remove_small_components = 1;
            unify_face_orientations = 1;
        } else if (strcmp(argv[i], "--max-hole-perimeter") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], &max_hole_perimeter)) {
                fprintf(stderr, "vkmesh: invalid --max-hole-perimeter\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--degenerate-abs") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], &degenerate_abs)) {
                fprintf(stderr, "vkmesh: invalid --degenerate-abs\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--degenerate-rel") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], &degenerate_rel)) {
                fprintf(stderr, "vkmesh: invalid --degenerate-rel\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--min-component-area") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], &min_component_area)) {
                fprintf(stderr, "vkmesh: invalid --min-component-area\n");
                return 2;
            }
        } else if ((strcmp(argv[i], "--target-faces") == 0 || strcmp(argv[i], "--decimation-target") == 0) && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &target_faces)) {
                fprintf(stderr, "vkmesh: invalid face target\n");
                return 2;
            }
            simplify = 1;
            disable_simplify = 0;
            simplify_option_seen = 1;
        } else if (strcmp(argv[i], "--simplify-steps") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &simplify_steps)) {
                fprintf(stderr, "vkmesh: invalid --simplify-steps\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--simplify-threshold") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], &simplify_threshold)) {
                fprintf(stderr, "vkmesh: invalid --simplify-threshold\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--lambda-edge-length") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], &lambda_edge_length)) {
                fprintf(stderr, "vkmesh: invalid --lambda-edge-length\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--lambda-skinny") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], &lambda_skinny)) {
                fprintf(stderr, "vkmesh: invalid --lambda-skinny\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--remesh-resolution") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &remesh_resolution)) {
                fprintf(stderr, "vkmesh: invalid --remesh-resolution\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--remesh-band") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], &remesh_band)) {
                fprintf(stderr, "vkmesh: invalid --remesh-band\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--remesh-project") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], &remesh_project)) {
                fprintf(stderr, "vkmesh: invalid --remesh-project\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--texture-size") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &texture_size)) {
                fprintf(stderr, "vkmesh: invalid --texture-size\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &device)) {
                fprintf(stderr, "vkmesh: invalid --device\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--gpu-workspace-budget-mib") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &gpu_workspace_budget_mib)) {
                fprintf(stderr, "vkmesh: invalid --gpu-workspace-budget-mib\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--unsigned-distance") == 0 && i + 1 < argc) {
            points_path = argv[++i];
        } else if (strcmp(argv[i], "--distance-output") == 0 && i + 1 < argc) {
            distance_output = argv[++i];
        } else if (strcmp(argv[i], "--fill-holes") == 0) {
            fill_holes = 1;
        } else if (strcmp(argv[i], "--no-fill-holes") == 0) {
            fill_holes = 0;
        } else if (strcmp(argv[i], "--remove-duplicate-faces") == 0) {
            remove_duplicate_faces = 1;
        } else if (strcmp(argv[i], "--remove-degenerate-faces") == 0) {
            remove_degenerate_faces = 1;
        } else if (strcmp(argv[i], "--repair-non-manifold-edges") == 0) {
            repair_non_manifold_edges = 1;
        } else if (strcmp(argv[i], "--remove-small-components") == 0) {
            remove_small_components = 1;
        } else if (strcmp(argv[i], "--unify-face-orientations") == 0) {
            unify_face_orientations = 1;
        } else if (strcmp(argv[i], "--simplify") == 0) {
            simplify = 1;
            disable_simplify = 0;
            simplify_option_seen = 1;
        } else if (strcmp(argv[i], "--no-simplify") == 0) {
            simplify = 0;
            disable_simplify = 1;
            simplify_option_seen = 1;
        } else if (strcmp(argv[i], "--remesh") == 0) {
            remesh = 1;
        } else if (strcmp(argv[i], "--no-remesh") == 0) {
            remesh = 0;
        } else if (strcmp(argv[i], "--uv-unwrap") == 0) {
            uv_unwrap = 1;
        } else if (strcmp(argv[i], "--no-uv-unwrap") == 0) {
            uv_unwrap = 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "vkmesh: unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }
    if (input == NULL || (output == NULL && distance_output == NULL)) {
        print_usage(argv[0]);
        return 2;
    }
    if (points_path != NULL && distance_output == NULL) {
        fprintf(stderr, "vkmesh: --distance-output is required with --unsigned-distance\n");
        return 2;
    }
    if (device < 0 || gpu_workspace_budget_mib < 0 || texture_size <= 0 || max_hole_perimeter < 0.0f ||
        degenerate_abs < 0.0f || degenerate_rel < 0.0f || min_component_area < 0.0f ||
        simplify_steps < 0 || simplify_threshold < 0.0f || lambda_edge_length < 0.0f ||
        lambda_skinny < 0.0f || target_faces < 0 ||
        (remesh && (remesh_resolution <= 0 || remesh_band <= 0.0f || remesh_project < 0.0f))) {
        fprintf(stderr, "vkmesh: invalid numeric parameter range\n");
        return 2;
    }
    g_vkmesh_device_index = device;
    g_vkmesh_workspace_budget_mib = gpu_workspace_budget_mib;

    vkmesh_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    if (!load_meshbin(input, &mesh)) {
        mesh_free(&mesh);
        return 1;
    }
    fprintf(stderr, "vkmesh: loaded %" PRId64 " vertices, %" PRId64 " faces\n", mesh.n_vertices, mesh.n_faces);
    if (trellis_postprocess) {
        int decimation_target = disable_simplify ? INT_MAX : target_faces;
        if (!vkmesh_trellis_postprocess(
                &mesh,
                projection_output,
                NULL,
                decimation_target,
                max_hole_perimeter,
                degenerate_abs,
                degenerate_rel,
                min_component_area,
                lambda_edge_length,
                lambda_skinny,
                simplify_threshold,
                simplify_steps,
                remove_degenerate_faces,
                remesh,
                remesh_resolution,
                remesh_band,
                remesh_project)) {
            mesh_free(&mesh);
            return 1;
        }
    } else {
        if (remove_duplicate_faces && !vkmesh_log_remove_duplicate_faces(&mesh, "stage")) {
            mesh_free(&mesh);
            return 1;
        }
        if (remove_degenerate_faces && !vkmesh_log_remove_degenerate_faces(&mesh, degenerate_abs, degenerate_rel, "stage")) {
            mesh_free(&mesh);
            return 1;
        }
        if (repair_non_manifold_edges && !vkmesh_log_repair_non_manifold_edges(&mesh, "stage")) {
            mesh_free(&mesh);
            return 1;
        }
        if (remove_small_components && !vkmesh_log_remove_small_connected_components(&mesh, min_component_area, "stage")) {
            mesh_free(&mesh);
            return 1;
        }
        /* Build hole boundaries only after malformed topology no longer pollutes adjacency. */
        if (fill_holes && !vkmesh_log_fill_holes(&mesh, max_hole_perimeter, "stage")) {
            mesh_free(&mesh);
            return 1;
        }
        if (unify_face_orientations && !vkmesh_log_unify_face_orientations(&mesh, "stage")) {
            mesh_free(&mesh);
            return 1;
        }
        if (remesh) {
            if (!vkmesh_remesh_narrow_band_dc(&mesh, remesh_resolution, remesh_band, remesh_project)) {
                mesh_free(&mesh);
                return 1;
            }
        }
        if (simplify && !disable_simplify) {
            if (target_faces <= 0) target_faces = (int) (mesh.n_faces / 2);
            if (target_faces <= 0) target_faces = 1;
            if (!vkmesh_log_simplify(
                    &mesh,
                    target_faces,
                    lambda_edge_length,
                    lambda_skinny,
                    simplify_threshold,
                    simplify_steps,
                    "stage")) {
                mesh_free(&mesh);
                return 1;
            }
        }
    }
    if (uv_unwrap) {
        int old_v = 0;
        int new_v = 0;
        int charts = 0;
        if (!vkmesh_uv_unwrap(&mesh, texture_size, &old_v, &new_v, &charts)) {
            mesh_free(&mesh);
            return 1;
        }
        fprintf(stderr, "vkmesh: uv_unwrap vertices=%d->%d charts=%d\n", old_v, new_v, charts);
    }
    if (points_path != NULL) {
        float * points = NULL;
        float * distances = NULL;
        uint32_t * face_ids = NULL;
        int64_t point_count = 0;
        if (!load_points(points_path, &points, &point_count) ||
            !vkmesh_unsigned_distance_vulkan(&mesh, points, point_count, &distances, &face_ids) ||
            !write_distances(distance_output, points, point_count, distances, face_ids)) {
            free(points);
            free(distances);
            free(face_ids);
            mesh_free(&mesh);
            return 1;
        }
        fprintf(stderr, "vkmesh: unsigned_distance points=%" PRId64 " wrote=%s\n", point_count, distance_output);
        free(points);
        free(distances);
        free(face_ids);
    }
    if (output != NULL) {
        if (!write_meshbin(output, &mesh)) {
            mesh_free(&mesh);
            return 1;
        }
        fprintf(stderr, "vkmesh: wrote %s (%" PRId64 " vertices, %" PRId64 " faces)\n", output, mesh.n_vertices, mesh.n_faces);
    }
    mesh_free(&mesh);
    return 0;
}
#endif
