#include "trellis.h"
#include "trellis_platform.h"
#include "image_to_3d_internal.h"

#include <vulkan/vulkan.h>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

typedef struct test_mesh {
    float * vertices;
    int32_t * faces;
    int64_t n_vertices;
    int64_t n_faces;
} test_mesh;

static int g_failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++g_failures; \
        return 0; \
    } \
} while (0)

static int close_f32(float actual, float expected) {
    return fabsf(actual - expected) <= 1e-6f;
}

static int read_gltf_position_bounds(
    const char * path,
    float min_out[3],
    float max_out[3]) {
    cgltf_options options;
    memset(&options, 0, sizeof(options));
    cgltf_data * data = NULL;
    if (cgltf_parse_file(&options, path, &data) != cgltf_result_success ||
        data == NULL || data->meshes_count == 0 ||
        data->meshes[0].primitives_count == 0) {
        cgltf_free(data);
        return 0;
    }
    const cgltf_primitive * primitive = &data->meshes[0].primitives[0];
    int found = 0;
    for (cgltf_size i = 0; i < primitive->attributes_count; ++i) {
        const cgltf_attribute * attribute = &primitive->attributes[i];
        if (attribute->type != cgltf_attribute_type_position ||
            attribute->data == NULL || !attribute->data->has_min ||
            !attribute->data->has_max) {
            continue;
        }
        for (int axis = 0; axis < 3; ++axis) {
            min_out[axis] = attribute->data->min[axis];
            max_out[axis] = attribute->data->max[axis];
        }
        found = 1;
        break;
    }
    cgltf_free(data);
    return found;
}

typedef struct gltf_texture_stats {
    int width;
    int height;
    size_t alpha_zero;
    size_t alpha_full;
    size_t alpha_partial;
} gltf_texture_stats;

static int read_gltf_texture_stats(
    const char * path,
    cgltf_size image_index,
    gltf_texture_stats * stats) {
    if (path == NULL || stats == NULL) return 0;
    memset(stats, 0, sizeof(*stats));
    cgltf_options options;
    memset(&options, 0, sizeof(options));
    cgltf_data * data = NULL;
    if (cgltf_parse_file(&options, path, &data) != cgltf_result_success || data == NULL ||
        cgltf_load_buffers(&options, data, path) != cgltf_result_success ||
        image_index >= data->images_count) {
        cgltf_free(data);
        return 0;
    }
    const cgltf_buffer_view * view = data->images[image_index].buffer_view;
    if (view == NULL || view->buffer == NULL || view->buffer->data == NULL ||
        view->size == 0 || view->size > INT_MAX || view->offset > view->buffer->size ||
        view->size > view->buffer->size - view->offset) {
        cgltf_free(data);
        return 0;
    }
    const unsigned char * encoded =
        (const unsigned char *) view->buffer->data + view->offset;
    int channels = 0;
    unsigned char * rgba = stbi_load_from_memory(
        encoded,
        (int) view->size,
        &stats->width,
        &stats->height,
        &channels,
        4);
    cgltf_free(data);
    if (rgba == NULL || stats->width <= 0 || stats->height <= 0) {
        stbi_image_free(rgba);
        return 0;
    }
    const size_t pixel_count = (size_t) stats->width * (size_t) stats->height;
    for (size_t i = 0; i < pixel_count; ++i) {
        const unsigned char alpha = rgba[i * 4u + 3u];
        if (alpha == 0u) {
            ++stats->alpha_zero;
        } else if (alpha == 255u) {
            ++stats->alpha_full;
        } else {
            ++stats->alpha_partial;
        }
    }
    stbi_image_free(rgba);
    return 1;
}

static void test_mesh_free(test_mesh * mesh) {
    if (mesh == NULL) return;
    free(mesh->vertices);
    free(mesh->faces);
    memset(mesh, 0, sizeof(*mesh));
}

static int write_meshbin(const char * path, const float * vertices, uint64_t n_vertices,
                         const int32_t * faces, uint64_t n_faces) {
    FILE * f = fopen(path, "wb");
    if (f == NULL) return 0;
    const char magic[8] = { 'T', 'R', 'L', 'M', 'E', 'S', 'H', '1' };
    const uint32_t flags = 0;
    const uint32_t reserved = 0;
    int ok =
        fwrite(magic, 1, sizeof(magic), f) == sizeof(magic) &&
        fwrite(&n_vertices, sizeof(n_vertices), 1, f) == 1 &&
        fwrite(&n_faces, sizeof(n_faces), 1, f) == 1 &&
        fwrite(&flags, sizeof(flags), 1, f) == 1 &&
        fwrite(&reserved, sizeof(reserved), 1, f) == 1 &&
        fwrite(vertices, sizeof(float), (size_t) n_vertices * 3u, f) == (size_t) n_vertices * 3u &&
        fwrite(faces, sizeof(int32_t), (size_t) n_faces * 3u, f) == (size_t) n_faces * 3u;
    if (fclose(f) != 0) ok = 0;
    return ok;
}

static int read_meshbin(const char * path, test_mesh * mesh_out) {
    memset(mesh_out, 0, sizeof(*mesh_out));
    FILE * f = fopen(path, "rb");
    if (f == NULL) return 0;
    char magic[8];
    uint64_t n_vertices = 0;
    uint64_t n_faces = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
    int ok =
        fread(magic, 1, sizeof(magic), f) == sizeof(magic) &&
        fread(&n_vertices, sizeof(n_vertices), 1, f) == 1 &&
        fread(&n_faces, sizeof(n_faces), 1, f) == 1 &&
        fread(&flags, sizeof(flags), 1, f) == 1 &&
        fread(&reserved, sizeof(reserved), 1, f) == 1 &&
        memcmp(magic, "TRLMESH1", 8) == 0 && n_vertices > 0 && n_faces > 0 &&
        n_vertices <= SIZE_MAX / (3u * sizeof(float)) &&
        n_faces <= SIZE_MAX / (3u * sizeof(int32_t));
    (void) reserved;
    test_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    if (ok) {
        mesh.vertices = (float *) malloc((size_t) n_vertices * 3u * sizeof(float));
        mesh.faces = (int32_t *) malloc((size_t) n_faces * 3u * sizeof(int32_t));
        mesh.n_vertices = (int64_t) n_vertices;
        mesh.n_faces = (int64_t) n_faces;
        ok = mesh.vertices != NULL && mesh.faces != NULL &&
            fread(mesh.vertices, sizeof(float), (size_t) n_vertices * 3u, f) == (size_t) n_vertices * 3u &&
            fread(mesh.faces, sizeof(int32_t), (size_t) n_faces * 3u, f) == (size_t) n_faces * 3u;
        if (ok && (flags & 1u) != 0) {
            float uv[2];
            for (uint64_t i = 0; i < n_vertices && ok; ++i) {
                ok = fread(uv, sizeof(float), 2u, f) == 2u;
            }
        }
    }
    if (fclose(f) != 0) ok = 0;
    if (!ok) {
        test_mesh_free(&mesh);
        return 0;
    }
    *mesh_out = mesh;
    return 1;
}

static void sort3(int32_t v[3]) {
    if (v[0] > v[1]) { int32_t t = v[0]; v[0] = v[1]; v[1] = t; }
    if (v[1] > v[2]) { int32_t t = v[1]; v[1] = v[2]; v[2] = t; }
    if (v[0] > v[1]) { int32_t t = v[0]; v[0] = v[1]; v[1] = t; }
}

static int mesh_is_closed_and_clean(const float * vertices, const int32_t * faces,
                                    int64_t n_vertices, int64_t n_faces) {
    (void) vertices;
    if (faces == NULL || n_vertices <= 0 || n_faces <= 0) return 0;
    for (int64_t i = 0; i < n_faces; ++i) {
        const int32_t * f = faces + (size_t) i * 3u;
        if (f[0] < 0 || f[1] < 0 || f[2] < 0 ||
            f[0] >= n_vertices || f[1] >= n_vertices || f[2] >= n_vertices ||
            f[0] == f[1] || f[1] == f[2] || f[2] == f[0]) return 0;
        int32_t key_i[3] = { f[0], f[1], f[2] };
        sort3(key_i);
        for (int64_t j = i + 1; j < n_faces; ++j) {
            const int32_t * g = faces + (size_t) j * 3u;
            int32_t key_j[3] = { g[0], g[1], g[2] };
            sort3(key_j);
            if (memcmp(key_i, key_j, sizeof(key_i)) == 0) return 0;
        }
    }
    for (int64_t i = 0; i < n_faces; ++i) {
        const int32_t * f = faces + (size_t) i * 3u;
        for (int e = 0; e < 3; ++e) {
            int32_t a = f[e];
            int32_t b = f[(e + 1) % 3];
            if (a > b) { int32_t t = a; a = b; b = t; }
            int count = 0;
            for (int64_t j = 0; j < n_faces; ++j) {
                const int32_t * g = faces + (size_t) j * 3u;
                for (int ge = 0; ge < 3; ++ge) {
                    int32_t ga = g[ge];
                    int32_t gb = g[(ge + 1) % 3];
                    if (ga > gb) { int32_t t = ga; ga = gb; gb = t; }
                    if (ga == a && gb == b) ++count;
                }
            }
            if (count != 2) return 0;
        }
    }
    return 1;
}

static int mesh_has_valid_triangles(const test_mesh * mesh) {
    if (mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0) return 0;
    uint8_t * referenced = (uint8_t *) calloc((size_t) mesh->n_vertices, 1u);
    if (referenced == NULL) return 0;
    int ok = 1;
    for (int64_t i = 0; i < mesh->n_vertices * 3 && ok; ++i) {
        if (!isfinite(mesh->vertices[i])) ok = 0;
    }
    for (int64_t i = 0; i < mesh->n_faces && ok; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        if (f[0] < 0 || f[1] < 0 || f[2] < 0 ||
            f[0] >= mesh->n_vertices || f[1] >= mesh->n_vertices || f[2] >= mesh->n_vertices ||
            f[0] == f[1] || f[1] == f[2] || f[2] == f[0]) {
            ok = 0;
            break;
        }
        referenced[f[0]] = 1u;
        referenced[f[1]] = 1u;
        referenced[f[2]] = 1u;
    }
    for (int64_t i = 0; i < mesh->n_vertices && ok; ++i) {
        if (!referenced[i]) ok = 0;
    }
    free(referenced);
    return ok;
}

static int mesh_has_well_conditioned_triangles(const test_mesh * mesh) {
    if (!mesh_has_valid_triangles(mesh)) return 0;
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        const float * a = mesh->vertices + (size_t) f[0] * 3u;
        const float * b = mesh->vertices + (size_t) f[1] * 3u;
        const float * c = mesh->vertices + (size_t) f[2] * 3u;
        const double ab[3] = {
            (double) b[0] - a[0], (double) b[1] - a[1], (double) b[2] - a[2],
        };
        const double ac[3] = {
            (double) c[0] - a[0], (double) c[1] - a[1], (double) c[2] - a[2],
        };
        const double bc[3] = {
            (double) c[0] - b[0], (double) c[1] - b[1], (double) c[2] - b[2],
        };
        const double cross[3] = {
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        };
        const double ab_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        const double ac_sq = ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2];
        const double bc_sq = bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2];
        const double max_edge_sq = fmax(ab_sq, fmax(ac_sq, bc_sq));
        const double cross_sq =
            cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
        if (!isfinite(max_edge_sq) || !isfinite(cross_sq) ||
            cross_sq <= 4e-12 * max_edge_sq * max_edge_sq) {
            return 0;
        }
    }
    return 1;
}

static int has_compute_device(int device_index) {
    VkInstance instance = VK_NULL_HANDLE;
    VkInstanceCreateInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    if (vkCreateInstance(&info, NULL, &instance) != VK_SUCCESS) return 0;
    uint32_t count = 0;
    int found = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, NULL) == VK_SUCCESS &&
        device_index >= 0 && (uint32_t) device_index < count) {
        VkPhysicalDevice * devices = (VkPhysicalDevice *) malloc((size_t) count * sizeof(*devices));
        if (devices != NULL && vkEnumeratePhysicalDevices(instance, &count, devices) == VK_SUCCESS) {
            uint32_t queue_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(devices[device_index], &queue_count, NULL);
            VkQueueFamilyProperties * queues =
                (VkQueueFamilyProperties *) malloc((size_t) queue_count * sizeof(*queues));
            if (queues != NULL) {
                vkGetPhysicalDeviceQueueFamilyProperties(devices[device_index], &queue_count, queues);
                for (uint32_t i = 0; i < queue_count; ++i) {
                    if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) found = 1;
                }
                free(queues);
            }
        }
        free(devices);
    }
    vkDestroyInstance(instance, NULL);
    return found;
}

static int test_cli_cleanup(const char * vkmesh_path) {
    static const float vertices[12] = {
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    static const int32_t dirty_faces[15] = {
        0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 2, 1, 0, 0, 1,
    };
    char input[PATH_MAX];
    char output[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(input, sizeof(input), "vkmesh_cleanup_in", ".meshbin"));
    CHECK_TRUE(trellis_make_temp_path(output, sizeof(output), "vkmesh_cleanup_out", ".meshbin"));
    CHECK_TRUE(write_meshbin(input, vertices, 4, dirty_faces, 5));
    char * cleanup_argv[] = {
        (char *) vkmesh_path, (char *) "--input", input, (char *) "--output", output,
        (char *) "--cleanup", (char *) "--max-hole-perimeter", (char *) "10",
        (char *) "--device", (char *) "0", NULL,
    };
    int ran = trellis_run_process_exact(cleanup_argv);
    test_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    int read_ok = ran && read_meshbin(output, &mesh);
    trellis_unlink(input);
    trellis_unlink(output);
    CHECK_TRUE(read_ok);
    CHECK_TRUE(mesh_is_closed_and_clean(mesh.vertices, mesh.faces, mesh.n_vertices, mesh.n_faces));
    test_mesh_free(&mesh);
    return 1;
}

static int test_cli_remove_relative_degenerate_face(const char * vkmesh_path) {
    static const float vertices[18] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        2.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f,
        2.0f, 1.0e-13f, 0.0f,
    };
    static const int32_t faces[6] = {
        0, 1, 2,
        3, 4, 5,
    };
    char input[PATH_MAX];
    char output[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(input, sizeof(input), "vkmesh_relative_degenerate_in", ".meshbin"));
    CHECK_TRUE(trellis_make_temp_path(output, sizeof(output), "vkmesh_relative_degenerate_out", ".meshbin"));
    CHECK_TRUE(write_meshbin(input, vertices, 6, faces, 2));
    char * args[] = {
        (char *) vkmesh_path, (char *) "--input", input, (char *) "--output", output,
        (char *) "--remove-degenerate-faces",
        (char *) "--degenerate-abs", (char *) "1e-24",
        (char *) "--degenerate-rel", (char *) "1e-12",
        (char *) "--no-fill-holes", (char *) "--no-uv-unwrap",
        (char *) "--device", (char *) "0", NULL,
    };
    int ran = trellis_run_process_exact(args);
    test_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    int read_ok = ran && read_meshbin(output, &mesh);
    trellis_unlink(input);
    trellis_unlink(output);
    CHECK_TRUE(read_ok);
    CHECK_TRUE(mesh.n_vertices == 3 && mesh.n_faces == 1);
    CHECK_TRUE(mesh_has_well_conditioned_triangles(&mesh));
    test_mesh_free(&mesh);
    return 1;
}

static int test_cli_simplify(const char * vkmesh_path) {
    enum { grid_size = 9, vertex_count = grid_size * grid_size,
           face_count = (grid_size - 1) * (grid_size - 1) * 2 };
    float vertices[vertex_count * 3];
    int32_t faces[face_count * 3];
    for (int y = 0; y < grid_size; ++y) {
        for (int x = 0; x < grid_size; ++x) {
            const int v = y * grid_size + x;
            vertices[v * 3 + 0] = (float) x / (float) (grid_size - 1);
            vertices[v * 3 + 1] = (float) y / (float) (grid_size - 1);
            vertices[v * 3 + 2] = 0.03f * sinf((float) x * 0.7f) * sinf((float) y * 0.5f);
        }
    }
    int face = 0;
    for (int y = 0; y + 1 < grid_size; ++y) {
        for (int x = 0; x + 1 < grid_size; ++x) {
            const int32_t a = y * grid_size + x;
            const int32_t b = a + 1;
            const int32_t c = a + grid_size;
            const int32_t d = c + 1;
            int32_t * f0 = faces + (size_t) face++ * 3u;
            int32_t * f1 = faces + (size_t) face++ * 3u;
            f0[0] = a; f0[1] = b; f0[2] = d;
            f1[0] = a; f1[1] = d; f1[2] = c;
        }
    }

    char input[PATH_MAX];
    char output[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(input, sizeof(input), "vkmesh_simplify_in", ".meshbin"));
    CHECK_TRUE(trellis_make_temp_path(output, sizeof(output), "vkmesh_simplify_out", ".meshbin"));
    CHECK_TRUE(write_meshbin(input, vertices, vertex_count, faces, face_count));
    char * args[] = {
        (char *) vkmesh_path, (char *) "--input", input, (char *) "--output", output,
        (char *) "--simplify", (char *) "--target-faces", (char *) "48",
        (char *) "--simplify-steps", (char *) "32", (char *) "--no-fill-holes",
        (char *) "--no-uv-unwrap", (char *) "--device", (char *) "0", NULL,
    };
    int ran = trellis_run_process_exact(args);
    test_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    int read_ok = ran && read_meshbin(output, &mesh);
    trellis_unlink(input);
    trellis_unlink(output);
    CHECK_TRUE(read_ok);
    CHECK_TRUE(mesh.n_faces > 0 && mesh.n_faces <= 48 && mesh.n_faces < face_count);
    CHECK_TRUE(mesh.n_vertices > 0 && mesh.n_vertices < vertex_count);
    CHECK_TRUE(mesh_has_valid_triangles(&mesh));
    test_mesh_free(&mesh);
    return 1;
}

static int test_cli_simplify_rejects_self_edge(const char * vkmesh_path) {
    static const float vertices[12] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    // The final face contributes a (0, 0) edge.  Without the self-edge
    // guards, its zero-cost "collapse" removes the whole star around v0.
    static const int32_t faces[15] = {
        0, 2, 1,
        0, 1, 3,
        1, 2, 3,
        2, 0, 3,
        0, 0, 1,
    };
    char input[PATH_MAX];
    char output[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(input, sizeof(input), "vkmesh_self_edge_in", ".meshbin"));
    CHECK_TRUE(trellis_make_temp_path(output, sizeof(output), "vkmesh_self_edge_out", ".meshbin"));
    CHECK_TRUE(write_meshbin(input, vertices, 4, faces, 5));
    char * args[] = {
        (char *) vkmesh_path, (char *) "--input", input, (char *) "--output", output,
        (char *) "--simplify", (char *) "--target-faces", (char *) "4",
        (char *) "--simplify-steps", (char *) "1",
        (char *) "--simplify-threshold", (char *) "1e-8",
        (char *) "--lambda-edge-length", (char *) "1",
        (char *) "--lambda-skinny", (char *) "0",
        (char *) "--no-fill-holes", (char *) "--no-uv-unwrap",
        (char *) "--device", (char *) "0", NULL,
    };
    int ran = trellis_run_process_exact(args);
    test_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    int read_ok = ran && read_meshbin(output, &mesh);
    trellis_unlink(input);
    trellis_unlink(output);
    CHECK_TRUE(read_ok);
    CHECK_TRUE(mesh.n_vertices == 4 && mesh.n_faces == 5);
    test_mesh_free(&mesh);
    return 1;
}

static int test_cli_simplify_rejects_collinear_collapse(const char * vkmesh_path) {
    static const float vertices[15] = {
        -0.01f,  0.0f, 0.0f,
         0.01f,  0.0f, 0.0f,
         0.0f,  -1.0f, 0.0f,
         0.0f,   1.0f, 0.0f,
         0.0f,   0.0f, 1.0f,
    };
    // Collapsing the short (0, 1) edge to its midpoint would put vertex 0 on
    // the line through vertices 2 and 3, flattening the retained first face.
    static const int32_t faces[6] = {
        0, 2, 3,
        0, 1, 4,
    };
    char input[PATH_MAX];
    char output[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(input, sizeof(input), "vkmesh_collinear_in", ".meshbin"));
    CHECK_TRUE(trellis_make_temp_path(output, sizeof(output), "vkmesh_collinear_out", ".meshbin"));
    CHECK_TRUE(write_meshbin(input, vertices, 5, faces, 2));
    char * args[] = {
        (char *) vkmesh_path, (char *) "--input", input, (char *) "--output", output,
        (char *) "--simplify", (char *) "--target-faces", (char *) "1",
        (char *) "--simplify-steps", (char *) "1",
        (char *) "--simplify-threshold", (char *) "0.001",
        (char *) "--lambda-edge-length", (char *) "1",
        (char *) "--lambda-skinny", (char *) "0",
        (char *) "--no-fill-holes", (char *) "--no-uv-unwrap",
        (char *) "--device", (char *) "0", NULL,
    };
    int ran = trellis_run_process_exact(args);
    test_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    int read_ok = ran && read_meshbin(output, &mesh);
    trellis_unlink(input);
    trellis_unlink(output);
    CHECK_TRUE(read_ok);
    CHECK_TRUE(mesh.n_vertices == 5 && mesh.n_faces == 2);
    CHECK_TRUE(mesh_has_well_conditioned_triangles(&mesh));
    test_mesh_free(&mesh);
    return 1;
}

static int test_cli_simplify_rejects_open_tetra_duplicate(const char * vkmesh_path) {
    static const float vertices[12] = {
        -0.0001f, 0.0f, 0.0f,
         0.0001f, 0.0f, 0.0f,
         0.0f,    1.0f, 0.0f,
         0.0f,    0.0f, 1.0f,
    };
    // Edge (0, 1) is much shorter than every other edge.  Contracting it
    // removes the first face and turns the remaining two into the same
    // unoriented triangle.  Vertex 3 is a forbidden common neighbor because
    // the only shared-face opposite vertex is 2.
    static const int32_t faces[9] = {
        0, 1, 2,
        0, 2, 3,
        1, 3, 2,
    };
    char input[PATH_MAX];
    char output[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(input, sizeof(input), "vkmesh_open_tetra_in", ".meshbin"));
    CHECK_TRUE(trellis_make_temp_path(output, sizeof(output), "vkmesh_open_tetra_out", ".meshbin"));
    CHECK_TRUE(write_meshbin(input, vertices, 4, faces, 3));
    char * args[] = {
        (char *) vkmesh_path, (char *) "--input", input, (char *) "--output", output,
        (char *) "--simplify", (char *) "--target-faces", (char *) "2",
        (char *) "--simplify-steps", (char *) "1",
        (char *) "--simplify-threshold", (char *) "0.01",
        (char *) "--lambda-edge-length", (char *) "1",
        (char *) "--lambda-skinny", (char *) "0",
        (char *) "--no-fill-holes", (char *) "--no-uv-unwrap",
        (char *) "--device", (char *) "0", NULL,
    };
    int ran = trellis_run_process_exact(args);
    test_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    int read_ok = ran && read_meshbin(output, &mesh);
    trellis_unlink(input);
    trellis_unlink(output);
    CHECK_TRUE(read_ok);
    CHECK_TRUE(mesh.n_vertices == 4 && mesh.n_faces == 3);
    CHECK_TRUE(mesh_has_valid_triangles(&mesh));
    test_mesh_free(&mesh);
    return 1;
}

static int test_cli_simplify_rejects_closed_tetra_duplicate(const char * vkmesh_path) {
    static const float vertices[12] = {
        -0.0001f, 0.0f, 0.0f,
         0.0001f, 0.0f, 0.0f,
         0.0f,    1.0f, 0.0f,
         0.0f,    0.0f, 1.0f,
    };
    // For a closed tetrahedron edge (0, 1), the common-neighbor vertex set is
    // exactly the two shared-face opposites {2, 3}.  A vertex-only link test
    // therefore passes, but the retained faces (0, 2, 3) and (1, 2, 3) become
    // duplicates.  The retained-face collision check must reject it.
    static const int32_t faces[12] = {
        0, 2, 1,
        0, 1, 3,
        1, 2, 3,
        2, 0, 3,
    };
    char input[PATH_MAX];
    char output[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(input, sizeof(input), "vkmesh_closed_tetra_in", ".meshbin"));
    CHECK_TRUE(trellis_make_temp_path(output, sizeof(output), "vkmesh_closed_tetra_out", ".meshbin"));
    CHECK_TRUE(write_meshbin(input, vertices, 4, faces, 4));
    char * args[] = {
        (char *) vkmesh_path, (char *) "--input", input, (char *) "--output", output,
        (char *) "--simplify", (char *) "--target-faces", (char *) "2",
        (char *) "--simplify-steps", (char *) "1",
        (char *) "--simplify-threshold", (char *) "0.01",
        (char *) "--lambda-edge-length", (char *) "1",
        (char *) "--lambda-skinny", (char *) "0",
        (char *) "--no-fill-holes", (char *) "--no-uv-unwrap",
        (char *) "--device", (char *) "0", NULL,
    };
    int ran = trellis_run_process_exact(args);
    test_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    int read_ok = ran && read_meshbin(output, &mesh);
    trellis_unlink(input);
    trellis_unlink(output);
    CHECK_TRUE(read_ok);
    CHECK_TRUE(mesh.n_vertices == 4 && mesh.n_faces == 4);
    CHECK_TRUE(mesh_is_closed_and_clean(mesh.vertices, mesh.faces, mesh.n_vertices, mesh.n_faces));
    test_mesh_free(&mesh);
    return 1;
}

static int test_cli_simplify_closed_manifold_progress(const char * vkmesh_path) {
    static const float vertices[18] = {
         0.0f,  0.0f,  1.0f,
         0.0f,  0.0f, -1.0f,
         1.0f,  0.0f,  0.0f,
         0.0f,  1.0f,  0.0f,
        -1.0f,  0.0f,  0.0f,
         0.0f, -1.0f,  0.0f,
    };
    static const int32_t faces[24] = {
        0, 2, 3,  0, 3, 4,  0, 4, 5,  0, 5, 2,
        1, 3, 2,  1, 4, 3,  1, 5, 4,  1, 2, 5,
    };
    char input[PATH_MAX];
    char output[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(input, sizeof(input), "vkmesh_manifold_in", ".meshbin"));
    CHECK_TRUE(trellis_make_temp_path(output, sizeof(output), "vkmesh_manifold_out", ".meshbin"));
    CHECK_TRUE(write_meshbin(input, vertices, 6, faces, 8));
    char * args[] = {
        (char *) vkmesh_path, (char *) "--input", input, (char *) "--output", output,
        (char *) "--simplify", (char *) "--target-faces", (char *) "6",
        (char *) "--simplify-steps", (char *) "1",
        (char *) "--simplify-threshold", (char *) "100",
        (char *) "--no-fill-holes", (char *) "--no-uv-unwrap",
        (char *) "--device", (char *) "0", NULL,
    };
    int ran = trellis_run_process_exact(args);
    test_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    int read_ok = ran && read_meshbin(output, &mesh);
    trellis_unlink(input);
    trellis_unlink(output);
    CHECK_TRUE(read_ok);
    CHECK_TRUE(mesh.n_faces >= 4 && mesh.n_faces < 8);
    CHECK_TRUE(mesh_is_closed_and_clean(mesh.vertices, mesh.faces, mesh.n_vertices, mesh.n_faces));
    CHECK_TRUE(mesh_has_well_conditioned_triangles(&mesh));
    test_mesh_free(&mesh);
    return 1;
}

static int test_cli_rejects_nan(const char * vkmesh_path) {
    char * args[] = {
        (char *) vkmesh_path, (char *) "--input", (char *) "unused.meshbin",
        (char *) "--output", (char *) "unused-output.meshbin",
        (char *) "--remesh-band", (char *) "nan", NULL,
    };
    CHECK_TRUE(!trellis_run_process_exact(args));
    return 1;
}

static int write_distance_points(const char * path, int count) {
    FILE * f = fopen(path, "w");
    if (f == NULL) return 0;
    int ok = 1;
    for (int i = 0; i < count; ++i) {
        const float x = (float) (i % 97) / 96.0f;
        const float y = (float) ((i / 97) % 89) / 88.0f;
        const float z = (float) ((i / (97 * 89)) % 83) / 82.0f;
        if (fprintf(f, "%.9g %.9g %.9g\n", x, y, z) < 0) {
            ok = 0;
            break;
        }
    }
    if (fclose(f) != 0) ok = 0;
    return ok;
}

static int read_distance_results(const char * path, int expected_count, float ** distances_out, uint32_t ** faces_out) {
    *distances_out = NULL;
    *faces_out = NULL;
    FILE * f = fopen(path, "r");
    if (f == NULL) return 0;
    float * distances = (float *) malloc((size_t) expected_count * sizeof(float));
    uint32_t * faces = (uint32_t *) malloc((size_t) expected_count * sizeof(uint32_t));
    if (distances == NULL || faces == NULL) {
        free(distances);
        free(faces);
        fclose(f);
        return 0;
    }
    char header[256];
    int ok = fgets(header, sizeof(header), f) != NULL;
    for (int i = 0; i < expected_count && ok; ++i) {
        float x, y, z;
        ok = fscanf(f, "%f %f %f %f %u", &x, &y, &z, &distances[i], &faces[i]) == 5;
    }
    if (ok) {
        float extra;
        ok = fscanf(f, "%f", &extra) != 1;
    }
    if (fclose(f) != 0) ok = 0;
    if (!ok) {
        free(distances);
        free(faces);
        return 0;
    }
    *distances_out = distances;
    *faces_out = faces;
    return 1;
}

static int test_udf_workspace_chunking(const char * vkmesh_path) {
    static const float vertices[12] = {
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    static const int32_t faces[12] = {
        0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3,
    };
    enum { point_count = 131072 };
    char mesh_path[PATH_MAX];
    char points_path[PATH_MAX];
    char small_path[PATH_MAX];
    char large_path[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(mesh_path, sizeof(mesh_path), "vkmesh_udf_mesh", ".meshbin"));
    CHECK_TRUE(trellis_make_temp_path(points_path, sizeof(points_path), "vkmesh_udf_points", ".txt"));
    CHECK_TRUE(trellis_make_temp_path(small_path, sizeof(small_path), "vkmesh_udf_small", ".txt"));
    CHECK_TRUE(trellis_make_temp_path(large_path, sizeof(large_path), "vkmesh_udf_large", ".txt"));
    CHECK_TRUE(write_meshbin(mesh_path, vertices, 4, faces, 4));
    CHECK_TRUE(write_distance_points(points_path, point_count));

    char * small_argv[] = {
        (char *) vkmesh_path, (char *) "--input", mesh_path,
        (char *) "--unsigned-distance", points_path, (char *) "--distance-output", small_path,
        (char *) "--gpu-workspace-budget-mib", (char *) "1", (char *) "--device", (char *) "0",
        (char *) "--no-fill-holes", NULL,
    };
    char * large_argv[] = {
        (char *) vkmesh_path, (char *) "--input", mesh_path,
        (char *) "--unsigned-distance", points_path, (char *) "--distance-output", large_path,
        (char *) "--gpu-workspace-budget-mib", (char *) "64", (char *) "--device", (char *) "0",
        (char *) "--no-fill-holes", NULL,
    };
    int ran = trellis_run_process_exact(small_argv) && trellis_run_process_exact(large_argv);
    float * small_distances = NULL;
    float * large_distances = NULL;
    uint32_t * small_faces = NULL;
    uint32_t * large_faces = NULL;
    int read_ok = ran &&
        read_distance_results(small_path, point_count, &small_distances, &small_faces) &&
        read_distance_results(large_path, point_count, &large_distances, &large_faces);
    trellis_unlink(mesh_path);
    trellis_unlink(points_path);
    trellis_unlink(small_path);
    trellis_unlink(large_path);
    CHECK_TRUE(read_ok);
    for (int i = 0; i < point_count; ++i) {
        CHECK_TRUE(fabsf(small_distances[i] - large_distances[i]) <= 1e-6f);
        CHECK_TRUE(small_faces[i] == large_faces[i]);
    }
    free(small_distances);
    free(large_distances);
    free(small_faces);
    free(large_faces);
    return 1;
}

static int test_remesh_workspace_chunking(const char * vkmesh_path) {
    static const float vertices[12] = {
        -0.35f, -0.35f, -0.35f, 0.35f, -0.35f, -0.35f,
        -0.35f, 0.35f, -0.35f, -0.35f, -0.35f, 0.35f,
    };
    static const int32_t faces[12] = {
        0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3,
    };
    char input[PATH_MAX];
    char small_path[PATH_MAX];
    char large_path[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(input, sizeof(input), "vkmesh_remesh_in", ".meshbin"));
    CHECK_TRUE(trellis_make_temp_path(small_path, sizeof(small_path), "vkmesh_remesh_small", ".meshbin"));
    CHECK_TRUE(trellis_make_temp_path(large_path, sizeof(large_path), "vkmesh_remesh_large", ".meshbin"));
    CHECK_TRUE(write_meshbin(input, vertices, 4, faces, 4));
    char * small_argv[] = {
        (char *) vkmesh_path, (char *) "--input", input, (char *) "--output", small_path,
        (char *) "--remesh", (char *) "--remesh-resolution", (char *) "128",
        (char *) "--remesh-band", (char *) "1", (char *) "--gpu-workspace-budget-mib", (char *) "1",
        (char *) "--no-fill-holes", (char *) "--device", (char *) "0", NULL,
    };
    char * large_argv[] = {
        (char *) vkmesh_path, (char *) "--input", input, (char *) "--output", large_path,
        (char *) "--remesh", (char *) "--remesh-resolution", (char *) "128",
        (char *) "--remesh-band", (char *) "1", (char *) "--gpu-workspace-budget-mib", (char *) "64",
        (char *) "--no-fill-holes", (char *) "--device", (char *) "0", NULL,
    };
    int ran = trellis_run_process_exact(small_argv) && trellis_run_process_exact(large_argv);
    test_mesh small_mesh;
    test_mesh large_mesh;
    memset(&small_mesh, 0, sizeof(small_mesh));
    memset(&large_mesh, 0, sizeof(large_mesh));
    int read_ok = ran && read_meshbin(small_path, &small_mesh) && read_meshbin(large_path, &large_mesh);
    trellis_unlink(input);
    trellis_unlink(small_path);
    trellis_unlink(large_path);
    CHECK_TRUE(read_ok);
    CHECK_TRUE(small_mesh.n_vertices == large_mesh.n_vertices);
    CHECK_TRUE(small_mesh.n_faces == large_mesh.n_faces);
    CHECK_TRUE(memcmp(
        small_mesh.vertices,
        large_mesh.vertices,
        (size_t) small_mesh.n_vertices * 3u * sizeof(float)) == 0);
    CHECK_TRUE(memcmp(
        small_mesh.faces,
        large_mesh.faces,
        (size_t) small_mesh.n_faces * 3u * sizeof(int32_t)) == 0);
    test_mesh_free(&small_mesh);
    test_mesh_free(&large_mesh);
    return 1;
}

static int test_gltf_bake(void) {
    float vertices[12] = {
        -0.40f, -0.10f, -0.20f, 0.30f, -0.10f, -0.20f,
        -0.40f, 0.25f, -0.20f, -0.40f, -0.10f, 0.45f,
    };
    int32_t faces[12] = {
        0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3,
    };
    int32_t coords[32];
    float attrs[48];
    int index = 0;
    for (int z = 1; z <= 2; ++z) {
        for (int y = 1; y <= 2; ++y) {
            for (int x = 1; x <= 2; ++x) {
                coords[index * 4 + 0] = 0;
                coords[index * 4 + 1] = x;
                coords[index * 4 + 2] = y;
                coords[index * 4 + 3] = z;
                attrs[index * 6 + 0] = (float) x * 0.25f;
                attrs[index * 6 + 1] = (float) y * 0.25f;
                attrs[index * 6 + 2] = (float) z * 0.25f;
                attrs[index * 6 + 3] = 0.1f;
                attrs[index * 6 + 4] = 0.7f;
                attrs[index * 6 + 5] = 1.0f;
                ++index;
            }
        }
    }
    trellis_mesh_host mesh;
    memset(&mesh, 0, sizeof(mesh));
    mesh.vertices = vertices;
    mesh.faces = faces;
    mesh.n_vertices = 4;
    mesh.n_faces = 4;
    trellis_pbr_voxels voxels;
    memset(&voxels, 0, sizeof(voxels));
    voxels.coords_bxyz = coords;
    voxels.attrs = attrs;
    voxels.n_coords = index;
    voxels.channels = 6;
    voxels.resolution = 4;
    char trellis_output[PATH_MAX];
    char pixal_output[PATH_MAX];
    char fragmented_output[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(
        trellis_output, sizeof(trellis_output), "vkmesh_bake_trellis", ".glb"));
    CHECK_TRUE(trellis_make_temp_path(
        pixal_output, sizeof(pixal_output), "vkmesh_bake_pixal", ".glb"));
    CHECK_TRUE(trellis_make_temp_path(
        fragmented_output, sizeof(fragmented_output), "vkmesh_bake_fragmented", ".glb"));
    trellis_status trellis_write_status = trellis_pipeline_write_gltf(
        trellis_output, &mesh, NULL, &voxels, 64, 0);
    trellis_status pixal_status = trellis_pipeline_write_gltf_ex(
        pixal_output,
        &mesh,
        NULL,
        &voxels,
        64,
        0,
        TRELLIS_PIPELINE_GLTF_COORDINATE_TRANSFORM_PIXAL3D);
    float fragmented_vertices[100u * 9u];
    int32_t fragmented_faces[100u * 3u];
    for (uint32_t face = 0; face < 100u; ++face) {
        memcpy(fragmented_vertices + (size_t) face * 9u, vertices, 9u * sizeof(float));
        fragmented_faces[(size_t) face * 3u + 0u] = (int32_t) (face * 3u + 0u);
        fragmented_faces[(size_t) face * 3u + 1u] = (int32_t) (face * 3u + 1u);
        fragmented_faces[(size_t) face * 3u + 2u] = (int32_t) (face * 3u + 2u);
    }
    trellis_mesh_host fragmented_mesh;
    memset(&fragmented_mesh, 0, sizeof(fragmented_mesh));
    fragmented_mesh.vertices = fragmented_vertices;
    fragmented_mesh.faces = fragmented_faces;
    fragmented_mesh.n_vertices = 300;
    fragmented_mesh.n_faces = 100;
    trellis_status fragmented_status = trellis_pipeline_write_gltf(
        fragmented_output, &fragmented_mesh, NULL, &voxels, 64, 0);
    FILE * f = trellis_write_status == TRELLIS_STATUS_OK ? fopen(trellis_output, "rb") : NULL;
    char magic[4] = {0, 0, 0, 0};
    int valid = f != NULL && fread(magic, 1, sizeof(magic), f) == sizeof(magic) &&
        memcmp(magic, "glTF", 4) == 0;
    if (f != NULL) fclose(f);
    float trellis_min[3] = {0};
    float trellis_max[3] = {0};
    float pixal_min[3] = {0};
    float pixal_max[3] = {0};
    const int trellis_bounds_ok = read_gltf_position_bounds(
        trellis_output, trellis_min, trellis_max);
    const int pixal_bounds_ok = read_gltf_position_bounds(
        pixal_output, pixal_min, pixal_max);
    gltf_texture_stats base_stats;
    gltf_texture_stats mr_stats;
    gltf_texture_stats fragmented_mr_stats;
    const int base_texture_ok = read_gltf_texture_stats(trellis_output, 0, &base_stats);
    const int mr_texture_ok = read_gltf_texture_stats(trellis_output, 1, &mr_stats);
    const int fragmented_texture_ok =
        read_gltf_texture_stats(fragmented_output, 1, &fragmented_mr_stats);
    trellis_unlink(trellis_output);
    trellis_unlink(pixal_output);
    trellis_unlink(fragmented_output);
    CHECK_TRUE(trellis_write_status == TRELLIS_STATUS_OK);
    CHECK_TRUE(pixal_status == TRELLIS_STATUS_OK);
    CHECK_TRUE(fragmented_status == TRELLIS_STATUS_OK);
    CHECK_TRUE(valid);
    CHECK_TRUE(trellis_bounds_ok && pixal_bounds_ok);
    CHECK_TRUE(base_texture_ok && mr_texture_ok);
    CHECK_TRUE(base_stats.width == 64 && base_stats.height == 64);
    CHECK_TRUE(mr_stats.width == 64 && mr_stats.height == 64);
    CHECK_TRUE(base_stats.alpha_zero > 0 && base_stats.alpha_full > 0);
    CHECK_TRUE(mr_stats.alpha_zero > 0 && mr_stats.alpha_full > 0);
    CHECK_TRUE(mr_stats.alpha_partial == 0);
    CHECK_TRUE(fragmented_texture_ok);
    CHECK_TRUE(fragmented_mr_stats.width == 64 && fragmented_mr_stats.height == 64);
    CHECK_TRUE(fragmented_mr_stats.alpha_zero > 0 && fragmented_mr_stats.alpha_full > 0);
    const float expected_trellis_min[3] = {-0.40f, -0.20f, -0.25f};
    const float expected_trellis_max[3] = {0.30f, 0.45f, 0.10f};
    const float expected_pixal_min[3] = {-0.30f, -0.10f, -0.45f};
    const float expected_pixal_max[3] = {0.40f, 0.25f, 0.20f};
    for (int axis = 0; axis < 3; ++axis) {
        CHECK_TRUE(close_f32(trellis_min[axis], expected_trellis_min[axis]));
        CHECK_TRUE(close_f32(trellis_max[axis], expected_trellis_max[axis]));
        CHECK_TRUE(close_f32(pixal_min[axis], expected_pixal_min[axis]));
        CHECK_TRUE(close_f32(pixal_max[axis], expected_pixal_max[axis]));
    }
    return 1;
}

typedef struct api_thread_args {
    const trellis_mesh_host * input;
    trellis_status status;
    int valid;
} api_thread_args;

static void run_api_thread(api_thread_args * args) {
    trellis_vkmesh_postprocess_options options;
    memset(&options, 0, sizeof(options));
    options.no_simplify = 1;
    options.max_hole_perimeter = 10.0f;
    options.device = 0;
    trellis_mesh_host output;
    memset(&output, 0, sizeof(output));
    args->status = trellis_vkmesh_postprocess(args->input, &output, NULL, &options);
    args->valid = args->status == TRELLIS_STATUS_OK &&
        mesh_is_closed_and_clean(output.vertices, output.faces, output.n_vertices, output.n_faces);
    trellis_mesh_free(&output);
}

#ifdef _WIN32
static DWORD WINAPI api_thread_entry(LPVOID opaque) {
    run_api_thread((api_thread_args *) opaque);
    return 0;
}
#else
static void * api_thread_entry(void * opaque) {
    run_api_thread((api_thread_args *) opaque);
    return NULL;
}
#endif

static int test_api_concurrent(void) {
    static float vertices[12] = {
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    static int32_t faces[12] = {
        0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3,
    };
    trellis_mesh_host input;
    memset(&input, 0, sizeof(input));
    input.vertices = vertices;
    input.faces = faces;
    input.n_vertices = 4;
    input.n_faces = 4;
    api_thread_args args[2] = { { &input, TRELLIS_STATUS_ERROR, 0 }, { &input, TRELLIS_STATUS_ERROR, 0 } };
#ifdef _WIN32
    HANDLE threads[2] = {
        CreateThread(NULL, 0, api_thread_entry, &args[0], 0, NULL),
        CreateThread(NULL, 0, api_thread_entry, &args[1], 0, NULL),
    };
    CHECK_TRUE(threads[0] != NULL && threads[1] != NULL);
    CHECK_TRUE(WaitForMultipleObjects(2, threads, TRUE, INFINITE) == WAIT_OBJECT_0);
    CloseHandle(threads[0]);
    CloseHandle(threads[1]);
#else
    pthread_t threads[2];
    CHECK_TRUE(pthread_create(&threads[0], NULL, api_thread_entry, &args[0]) == 0);
    CHECK_TRUE(pthread_create(&threads[1], NULL, api_thread_entry, &args[1]) == 0);
    CHECK_TRUE(pthread_join(threads[0], NULL) == 0);
    CHECK_TRUE(pthread_join(threads[1], NULL) == 0);
#endif
    CHECK_TRUE(args[0].status == TRELLIS_STATUS_OK && args[0].valid);
    CHECK_TRUE(args[1].status == TRELLIS_STATUS_OK && args[1].valid);
    return 1;
}

static int test_api_validation(void) {
    float vertices[12] = {
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    int32_t faces[12] = {
        0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3,
    };
    trellis_mesh_host input;
    trellis_mesh_host output;
    trellis_vkmesh_postprocess_options options;
    memset(&input, 0, sizeof(input));
    memset(&output, 0, sizeof(output));
    memset(&options, 0, sizeof(options));
    input.vertices = vertices;
    input.faces = faces;
    input.n_vertices = 4;
    input.n_faces = 4;
    options.device = INT_MAX;
    CHECK_TRUE(trellis_vkmesh_postprocess(&input, &output, NULL, &options) == TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(output.vertices == NULL && output.faces == NULL);
    options.device = 0;
    options.remesh_band = NAN;
    CHECK_TRUE(trellis_vkmesh_postprocess(&input, &output, NULL, &options) == TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(output.vertices == NULL && output.faces == NULL);
    options.remesh_band = 0.0f;
    vertices[0] = NAN;
    CHECK_TRUE(trellis_vkmesh_postprocess(&input, &output, NULL, &options) == TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(output.vertices == NULL && output.faces == NULL);
    vertices[0] = 0.0f;
    options.gpu_workspace_budget_mib = -1;
    CHECK_TRUE(trellis_vkmesh_postprocess(&input, &output, NULL, &options) == TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(output.vertices == NULL && output.faces == NULL);
    return 1;
}

static int test_api_workspace_oom_is_fatal(void) {
    enum { VERTEX_COUNT = 65536, FACE_COUNT = 100000 };
    const size_t vertex_words = (size_t) VERTEX_COUNT * 3u;
    const size_t face_words = (size_t) FACE_COUNT * 3u;
    const size_t vertex_bytes = vertex_words * sizeof(float);
    const size_t face_bytes = face_words * sizeof(int32_t);
    float * vertices = (float *) malloc(vertex_bytes);
    float * vertices_before = (float *) malloc(vertex_bytes);
    int32_t * faces = (int32_t *) malloc(face_bytes);
    int32_t * faces_before = (int32_t *) malloc(face_bytes);
    if (vertices == NULL || vertices_before == NULL || faces == NULL || faces_before == NULL) {
        free(vertices);
        free(vertices_before);
        free(faces);
        free(faces_before);
        CHECK_TRUE(0 && "workspace OOM test fixture allocation");
    }

    for (int i = 0; i < VERTEX_COUNT; ++i) {
        vertices[(size_t) i * 3u + 0u] = (float) (i % 257) / 257.0f;
        vertices[(size_t) i * 3u + 1u] = (float) ((i / 257) % 257) / 257.0f;
        vertices[(size_t) i * 3u + 2u] = (float) (i % 31) / 31.0f;
    }
    for (int i = 0; i < FACE_COUNT; ++i) {
        int32_t base = (int32_t) (((uint32_t) i * 3u) % (VERTEX_COUNT - 2u));
        faces[(size_t) i * 3u + 0u] = base;
        faces[(size_t) i * 3u + 1u] = base + 1;
        faces[(size_t) i * 3u + 2u] = base + 2;
    }
    memcpy(vertices_before, vertices, vertex_bytes);
    memcpy(faces_before, faces, face_bytes);

    trellis_mesh_host input;
    trellis_mesh_host output;
    trellis_mesh_host projection;
    trellis_vkmesh_postprocess_options options;
    memset(&input, 0, sizeof(input));
    memset(&output, 0, sizeof(output));
    memset(&projection, 0, sizeof(projection));
    memset(&options, 0, sizeof(options));
    input.vertices = vertices;
    input.faces = faces;
    input.n_vertices = VERTEX_COUNT;
    input.n_faces = FACE_COUNT;
    options.no_simplify = 1;
    options.device = 0;
    options.gpu_workspace_budget_mib = 1;

    trellis_status status = trellis_vkmesh_postprocess(
        &input, &output, &projection, &options);
    const int input_unchanged =
        input.vertices == vertices && input.faces == faces &&
        input.n_vertices == VERTEX_COUNT && input.n_faces == FACE_COUNT &&
        memcmp(vertices, vertices_before, vertex_bytes) == 0 &&
        memcmp(faces, faces_before, face_bytes) == 0;
    const int outputs_empty =
        output.vertices == NULL && output.faces == NULL &&
        output.n_vertices == 0 && output.n_faces == 0 &&
        projection.vertices == NULL && projection.faces == NULL &&
        projection.n_vertices == 0 && projection.n_faces == 0;

    trellis_mesh_free(&output);
    trellis_mesh_free(&projection);
    free(vertices);
    free(vertices_before);
    free(faces);
    free(faces_before);
    CHECK_TRUE(status == TRELLIS_STATUS_OUT_OF_MEMORY);
    CHECK_TRUE(input_unchanged);
    CHECK_TRUE(outputs_empty);
    return 1;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s /path/to/vkmesh\n", argv[0]);
        return 2;
    }
    if (!has_compute_device(0)) {
        fprintf(stderr, "vkmesh tests skipped: Vulkan device 0 has no compute queue\n");
        return 77;
    }
    (void) test_cli_cleanup(argv[1]);
    (void) test_cli_remove_relative_degenerate_face(argv[1]);
    (void) test_cli_simplify(argv[1]);
    (void) test_cli_simplify_rejects_self_edge(argv[1]);
    (void) test_cli_simplify_rejects_collinear_collapse(argv[1]);
    (void) test_cli_simplify_rejects_open_tetra_duplicate(argv[1]);
    (void) test_cli_simplify_rejects_closed_tetra_duplicate(argv[1]);
    (void) test_cli_simplify_closed_manifold_progress(argv[1]);
    (void) test_cli_rejects_nan(argv[1]);
    (void) test_udf_workspace_chunking(argv[1]);
    (void) test_remesh_workspace_chunking(argv[1]);
    (void) test_api_concurrent();
    (void) test_api_validation();
    (void) test_api_workspace_oom_is_fatal();
    (void) test_gltf_bake();
    if (g_failures != 0) {
        fprintf(stderr, "vkmesh tests failed: %d failure(s)\n", g_failures);
        return 1;
    }
    fprintf(stderr, "vkmesh tests passed\n");
    return 0;
}
