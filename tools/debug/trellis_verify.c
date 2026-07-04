#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_coord_dims = 3;

static void usage(const char * argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --coords-a A.i32 --coords-b B.i32 [--coord-dims 3|4] [--expect-exact-coords] [--min-iou X]\n"
        "  %s --f32-a A.f32 --f32-b B.f32 [--min-cos X] [--max-mean-abs X]\n",
        argv0, argv0);
}

static const char * arg_value(int argc, char ** argv, int * i) {
    if (*i + 1 >= argc) {
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

static int file_size(FILE * f, size_t * out) {
    if (fseek(f, 0, SEEK_END) != 0) {
        return 0;
    }
    long n = ftell(f);
    if (n < 0) {
        return 0;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        return 0;
    }
    *out = (size_t) n;
    return 1;
}

static int read_file_exact(const char * path, void ** data_out, size_t * size_out) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return 0;
    }
    size_t size = 0;
    if (!file_size(f, &size)) {
        fprintf(stderr, "failed to stat %s\n", path);
        fclose(f);
        return 0;
    }
    void * data = NULL;
    if (size > 0) {
        data = malloc(size);
        if (data == NULL) {
            fprintf(stderr, "allocation failed for %s (%zu bytes)\n", path, size);
            fclose(f);
            return 0;
        }
        if (fread(data, 1, size, f) != size) {
            fprintf(stderr, "failed to read %s\n", path);
            free(data);
            fclose(f);
            return 0;
        }
    }
    fclose(f);
    *data_out = data;
    *size_out = size;
    return 1;
}

static int compare_coord_row(const void * a, const void * b) {
    const int32_t * ca = (const int32_t *) a;
    const int32_t * cb = (const int32_t *) b;
    for (int i = 0; i < g_coord_dims; ++i) {
        if (ca[i] < cb[i]) return -1;
        if (ca[i] > cb[i]) return 1;
    }
    return 0;
}

static int run_coords(
    const char * a_path,
    const char * b_path,
    int expect_exact,
    double min_iou) {
    void * a_data = NULL;
    void * b_data = NULL;
    size_t a_size = 0;
    size_t b_size = 0;
    if (!read_file_exact(a_path, &a_data, &a_size) ||
        !read_file_exact(b_path, &b_data, &b_size)) {
        free(a_data);
        free(b_data);
        return 2;
    }
    if (g_coord_dims != 3 && g_coord_dims != 4) {
        fprintf(stderr, "coord dims must be 3 or 4\n");
        free(a_data);
        free(b_data);
        return 2;
    }
    const size_t row_size = (size_t) g_coord_dims * sizeof(int32_t);
    if (a_size % row_size != 0 || b_size % row_size != 0) {
        fprintf(stderr, "coord files must contain packed int32 rows with %d columns\n", g_coord_dims);
        free(a_data);
        free(b_data);
        return 2;
    }

    int exact = (a_size == b_size) &&
        (a_size == 0 || memcmp(a_data, b_data, a_size) == 0);
    size_t a_count = a_size / row_size;
    size_t b_count = b_size / row_size;
    int32_t * a = (int32_t *) a_data;
    int32_t * b = (int32_t *) b_data;
    qsort(a, a_count, row_size, compare_coord_row);
    qsort(b, b_count, row_size, compare_coord_row);

    size_t i = 0;
    size_t j = 0;
    size_t intersection = 0;
    while (i < a_count && j < b_count) {
        int cmp = compare_coord_row(a + i * (size_t) g_coord_dims, b + j * (size_t) g_coord_dims);
        if (cmp == 0) {
            intersection += 1;
            i += 1;
            j += 1;
        } else if (cmp < 0) {
            i += 1;
        } else {
            j += 1;
        }
    }
    const size_t uni = a_count + b_count - intersection;
    const double iou = uni == 0 ? 1.0 : (double) intersection / (double) uni;

    printf("coords: a=%zu b=%zu intersection=%zu union=%zu iou=%.9f exact_bytes=%s\n",
        a_count, b_count, intersection, uni, iou, exact ? "yes" : "no");

    int ok = 1;
    if (expect_exact && !exact) {
        fprintf(stderr, "coords exact byte match failed\n");
        ok = 0;
    }
    if (min_iou >= 0.0 && iou < min_iou) {
        fprintf(stderr, "coords IoU %.9f below threshold %.9f\n", iou, min_iou);
        ok = 0;
    }
    free(a_data);
    free(b_data);
    return ok ? 0 : 1;
}

static int run_f32(
    const char * a_path,
    const char * b_path,
    double min_cos,
    double max_mean_abs) {
    void * a_data = NULL;
    void * b_data = NULL;
    size_t a_size = 0;
    size_t b_size = 0;
    if (!read_file_exact(a_path, &a_data, &a_size) ||
        !read_file_exact(b_path, &b_data, &b_size)) {
        free(a_data);
        free(b_data);
        return 2;
    }
    if (a_size != b_size || a_size % sizeof(float) != 0) {
        fprintf(stderr, "f32 files must have the same number of float32 values\n");
        free(a_data);
        free(b_data);
        return 2;
    }
    const size_t n = a_size / sizeof(float);
    const float * a = (const float *) a_data;
    const float * b = (const float *) b_data;
    double dot = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    double mean_abs = 0.0;
    double max_abs = 0.0;
    double mse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double av = (double) a[i];
        const double bv = (double) b[i];
        const double d = av - bv;
        const double ad = fabs(d);
        dot += av * bv;
        aa += av * av;
        bb += bv * bv;
        mean_abs += ad;
        mse += d * d;
        if (ad > max_abs) {
            max_abs = ad;
        }
    }
    mean_abs = n == 0 ? 0.0 : mean_abs / (double) n;
    mse = n == 0 ? 0.0 : mse / (double) n;
    const double cos = (aa == 0.0 || bb == 0.0) ? 0.0 : dot / sqrt(aa * bb);
    const double rmse = sqrt(mse);
    printf("f32: n=%zu cosine=%.10f mean_abs=%.10g max_abs=%.10g rmse=%.10g\n",
        n, cos, mean_abs, max_abs, rmse);

    int ok = 1;
    if (min_cos >= -1.0 && cos < min_cos) {
        fprintf(stderr, "cosine %.10f below threshold %.10f\n", cos, min_cos);
        ok = 0;
    }
    if (max_mean_abs >= 0.0 && mean_abs > max_mean_abs) {
        fprintf(stderr, "mean_abs %.10g above threshold %.10g\n", mean_abs, max_mean_abs);
        ok = 0;
    }
    free(a_data);
    free(b_data);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    const char * coords_a = NULL;
    const char * coords_b = NULL;
    const char * f32_a = NULL;
    const char * f32_b = NULL;
    int expect_exact_coords = 0;
    int coord_dims = 3;
    double min_iou = -1.0;
    double min_cos = -2.0;
    double max_mean_abs = -1.0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--coords-a") == 0) {
            coords_a = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--coords-b") == 0) {
            coords_b = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--f32-a") == 0) {
            f32_a = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--f32-b") == 0) {
            f32_b = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--expect-exact-coords") == 0) {
            expect_exact_coords = 1;
        } else if (strcmp(argv[i], "--coord-dims") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            coord_dims = atoi(v);
        } else if (strcmp(argv[i], "--min-iou") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            min_iou = strtod(v, NULL);
        } else if (strcmp(argv[i], "--min-cos") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            min_cos = strtod(v, NULL);
        } else if (strcmp(argv[i], "--max-mean-abs") == 0) {
            const char * v = arg_value(argc, argv, &i);
            if (v == NULL) {
                usage(argv[0]);
                return 2;
            }
            max_mean_abs = strtod(v, NULL);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    const int have_coords = coords_a != NULL || coords_b != NULL;
    const int have_f32 = f32_a != NULL || f32_b != NULL;
    if (have_coords == have_f32) {
        usage(argv[0]);
        return 2;
    }
    if (have_coords) {
        if (coords_a == NULL || coords_b == NULL) {
            usage(argv[0]);
            return 2;
        }
        g_coord_dims = coord_dims;
        return run_coords(coords_a, coords_b, expect_exact_coords, min_iou);
    }
    if (f32_a == NULL || f32_b == NULL) {
        usage(argv[0]);
        return 2;
    }
    return run_f32(f32_a, f32_b, min_cos, max_mean_abs);
}
