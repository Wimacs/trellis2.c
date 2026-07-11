#include "pixal_naf.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
    NAF_ENCODER_BASE = 0,
    NAF_SEM_ENCODER_BASE = 18,
    NAF_ROPE_PERIODS = 36,
    NAF_BRANCH_TENSORS = 18,
    NAF_BLOCK_TENSORS = 8,
    NAF_ENCODER_BLOCKS = 2,
    NAF_GROUPS = 8,
};

static const char * const g_naf_weight_names[TRELLIS_PIXAL_NAF_WEIGHT_COUNT] = {
    "image_encoder.encoder.0.weight",
    "image_encoder.encoder.0.bias",
    "image_encoder.encoder.1.norm1.weight",
    "image_encoder.encoder.1.norm1.bias",
    "image_encoder.encoder.1.conv1.weight",
    "image_encoder.encoder.1.conv1.bias",
    "image_encoder.encoder.1.norm2.weight",
    "image_encoder.encoder.1.norm2.bias",
    "image_encoder.encoder.1.conv2.weight",
    "image_encoder.encoder.1.conv2.bias",
    "image_encoder.encoder.2.norm1.weight",
    "image_encoder.encoder.2.norm1.bias",
    "image_encoder.encoder.2.conv1.weight",
    "image_encoder.encoder.2.conv1.bias",
    "image_encoder.encoder.2.norm2.weight",
    "image_encoder.encoder.2.norm2.bias",
    "image_encoder.encoder.2.conv2.weight",
    "image_encoder.encoder.2.conv2.bias",
    "image_encoder.sem_encoder.0.weight",
    "image_encoder.sem_encoder.0.bias",
    "image_encoder.sem_encoder.1.norm1.weight",
    "image_encoder.sem_encoder.1.norm1.bias",
    "image_encoder.sem_encoder.1.conv1.weight",
    "image_encoder.sem_encoder.1.conv1.bias",
    "image_encoder.sem_encoder.1.norm2.weight",
    "image_encoder.sem_encoder.1.norm2.bias",
    "image_encoder.sem_encoder.1.conv2.weight",
    "image_encoder.sem_encoder.1.conv2.bias",
    "image_encoder.sem_encoder.2.norm1.weight",
    "image_encoder.sem_encoder.2.norm1.bias",
    "image_encoder.sem_encoder.2.conv1.weight",
    "image_encoder.sem_encoder.2.conv1.bias",
    "image_encoder.sem_encoder.2.norm2.weight",
    "image_encoder.sem_encoder.2.norm2.bias",
    "image_encoder.sem_encoder.2.conv2.weight",
    "image_encoder.sem_encoder.2.conv2.bias",
    "image_encoder.rope.periods",
};

static void set_issue(char * dst, size_t dst_size, const char * fmt, ...) {
    if (dst == NULL || dst_size == 0 || dst[0] != '\0') {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(dst, dst_size, fmt, args);
    va_end(args);
}

static int checked_mul_size(size_t a, size_t b, size_t * out) {
    if (out == NULL || (a != 0 && b > SIZE_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int checked_elements4(int a, int b, int c, int d, size_t * out) {
    size_t n = 0;
    if (a <= 0 || b <= 0 || c <= 0 || d <= 0 ||
        !checked_mul_size((size_t) a, (size_t) b, &n) ||
        !checked_mul_size(n, (size_t) c, &n) ||
        !checked_mul_size(n, (size_t) d, &n)) {
        return 0;
    }
    *out = n;
    return 1;
}

static size_t nchw_index(int b, int c, int y, int x, int channels, int height, int width) {
    return (((size_t) b * (size_t) channels + (size_t) c) * (size_t) height + (size_t) y) *
        (size_t) width + (size_t) x;
}

static size_t bhwc_index(int b, int y, int x, int c, int height, int width, int channels) {
    return (((size_t) b * (size_t) height + (size_t) y) * (size_t) width + (size_t) x) *
        (size_t) channels + (size_t) c;
}

const char * trellis_pixal_naf_weight_name(size_t index) {
    return index < TRELLIS_PIXAL_NAF_WEIGHT_COUNT ? g_naf_weight_names[index] : NULL;
}

static void expected_weight_shape(size_t index, int * n_dims, int64_t shape[4]) {
    memset(shape, 0, 4 * sizeof(shape[0]));
    if (index == NAF_ROPE_PERIODS) {
        *n_dims = 1;
        shape[0] = 16;
        return;
    }

    const size_t branch_base = index >= NAF_SEM_ENCODER_BASE ? NAF_SEM_ENCODER_BASE : NAF_ENCODER_BASE;
    const size_t local = index - branch_base;
    const int kernel = branch_base == NAF_SEM_ENCODER_BASE ? 3 : 1;
    if (local == 0) {
        *n_dims = 4;
        shape[0] = TRELLIS_PIXAL_NAF_ENCODER_CHANNELS;
        shape[1] = TRELLIS_PIXAL_NAF_IMAGE_CHANNELS;
        shape[2] = kernel;
        shape[3] = kernel;
        return;
    }
    if (local == 1) {
        *n_dims = 1;
        shape[0] = TRELLIS_PIXAL_NAF_ENCODER_CHANNELS;
        return;
    }

    const size_t block_field = (local - 2) % NAF_BLOCK_TENSORS;
    if (block_field == 2 || block_field == 6) {
        *n_dims = 4;
        shape[0] = TRELLIS_PIXAL_NAF_ENCODER_CHANNELS;
        shape[1] = TRELLIS_PIXAL_NAF_ENCODER_CHANNELS;
        shape[2] = kernel;
        shape[3] = kernel;
    } else {
        *n_dims = 1;
        shape[0] = TRELLIS_PIXAL_NAF_ENCODER_CHANNELS;
    }
}

void trellis_pixal_naf_free_weights(trellis_pixal_naf_weights * weights) {
    if (weights == NULL) {
        return;
    }
    for (size_t i = 0; i < TRELLIS_PIXAL_NAF_WEIGHT_COUNT; ++i) {
        free(weights->tensors[i].data);
    }
    memset(weights, 0, sizeof(*weights));
}

static int host_weights_are_empty(const trellis_pixal_naf_weights * weights) {
    if (weights == NULL || weights->n_tensors != 0) {
        return 0;
    }
    for (size_t i = 0; i < TRELLIS_PIXAL_NAF_WEIGHT_COUNT; ++i) {
        const trellis_pixal_naf_host_tensor * tensor = &weights->tensors[i];
        if (tensor->data != NULL || tensor->count != 0 || tensor->n_dims != 0) {
            return 0;
        }
        for (size_t d = 0; d < sizeof(tensor->shape) / sizeof(tensor->shape[0]); ++d) {
            if (tensor->shape[d] != 0) {
                return 0;
            }
        }
    }
    return 1;
}

trellis_status trellis_pixal_naf_load_weights_f32(
    const char * safetensors_path,
    trellis_pixal_naf_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    if (safetensors_path == NULL || weights == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (first_issue != NULL && first_issue_size != 0) {
        first_issue[0] = '\0';
    }
    if (!host_weights_are_empty(weights)) {
        set_issue(
            first_issue,
            first_issue_size,
            "NAF output weights must be empty/zero-initialized; free existing weights before loading");
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_pixal_naf_weights loaded_weights = {0};

    trellis_safetensors st;
    trellis_status status = trellis_safetensors_open(safetensors_path, &st);
    if (status != TRELLIS_STATUS_OK) {
        set_issue(first_issue, first_issue_size, "failed to open safetensors: %s", trellis_status_string(status));
        return status;
    }
    if (st.n_tensors != TRELLIS_PIXAL_NAF_WEIGHT_COUNT) {
        set_issue(
            first_issue,
            first_issue_size,
            "NAF tensor count mismatch: got %zu expected %d",
            st.n_tensors,
            TRELLIS_PIXAL_NAF_WEIGHT_COUNT);
        trellis_safetensors_close(&st);
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    for (size_t i = 0; i < TRELLIS_PIXAL_NAF_WEIGHT_COUNT; ++i) {
        const char * name = g_naf_weight_names[i];
        const trellis_safetensor_meta * meta = trellis_safetensors_find(&st, name);
        if (meta == NULL) {
            set_issue(first_issue, first_issue_size, "missing NAF tensor: %s", name);
            status = TRELLIS_STATUS_NOT_FOUND;
            break;
        }
        if (meta->dtype != TRELLIS_DTYPE_F32) {
            set_issue(
                first_issue,
                first_issue_size,
                "NAF tensor %s must be F32, got %s",
                name,
                trellis_dtype_name(meta->dtype));
            status = TRELLIS_STATUS_PARSE_ERROR;
            break;
        }

        int n_dims = 0;
        int64_t shape[4];
        expected_weight_shape(i, &n_dims, shape);
        if (meta->n_dims != n_dims) {
            set_issue(
                first_issue,
                first_issue_size,
                "NAF tensor %s rank mismatch: got %d expected %d",
                name,
                meta->n_dims,
                n_dims);
            status = TRELLIS_STATUS_PARSE_ERROR;
            break;
        }
        int shape_ok = 1;
        for (int d = 0; d < n_dims; ++d) {
            if (meta->shape[d] != shape[d]) {
                shape_ok = 0;
                set_issue(
                    first_issue,
                    first_issue_size,
                    "NAF tensor %s shape mismatch at dim%d: got %lld expected %lld",
                    name,
                    d,
                    (long long) meta->shape[d],
                    (long long) shape[d]);
                break;
            }
        }
        if (!shape_ok) {
            status = TRELLIS_STATUS_PARSE_ERROR;
            break;
        }

        const uint64_t count64 = trellis_safetensor_nelements(meta);
        if (count64 == 0 || count64 > (uint64_t) SIZE_MAX / sizeof(float)) {
            set_issue(first_issue, first_issue_size, "NAF tensor %s is too large", name);
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            break;
        }
        trellis_pixal_naf_host_tensor * tensor = &loaded_weights.tensors[i];
        tensor->data = (float *) malloc((size_t) count64 * sizeof(float));
        if (tensor->data == NULL) {
            set_issue(first_issue, first_issue_size, "out of memory loading NAF tensor %s", name);
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
            break;
        }
        tensor->count = (size_t) count64;
        tensor->n_dims = n_dims;
        memcpy(tensor->shape, shape, sizeof(shape));
        status = trellis_safetensors_read_f32(&st, meta, tensor->data, tensor->count);
        if (status != TRELLIS_STATUS_OK) {
            set_issue(
                first_issue,
                first_issue_size,
                "failed reading NAF tensor %s: %s",
                name,
                trellis_status_string(status));
            break;
        }
    }

    trellis_safetensors_close(&st);
    if (status != TRELLIS_STATUS_OK) {
        trellis_pixal_naf_free_weights(&loaded_weights);
        return status;
    }
    loaded_weights.n_tensors = TRELLIS_PIXAL_NAF_WEIGHT_COUNT;
    *weights = loaded_weights;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pixal_naf_bind_ggml_weights(
    trellis_tensor_store * store,
    trellis_pixal_naf_ggml_weights * weights,
    char * first_issue,
    size_t first_issue_size) {
    if (store == NULL || weights == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (first_issue != NULL && first_issue_size != 0) {
        first_issue[0] = '\0';
    }
    memset(weights, 0, sizeof(*weights));
    if (store->n_entries != TRELLIS_PIXAL_NAF_WEIGHT_COUNT) {
        set_issue(
            first_issue,
            first_issue_size,
            "NAF tensor store count mismatch: got %zu expected %d",
            store->n_entries,
            TRELLIS_PIXAL_NAF_WEIGHT_COUNT);
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    for (size_t i = 0; i < TRELLIS_PIXAL_NAF_WEIGHT_COUNT; ++i) {
        const char * name = g_naf_weight_names[i];
        struct ggml_tensor * tensor = trellis_tensor_store_get(store, name);
        if (tensor == NULL) {
            set_issue(first_issue, first_issue_size, "missing NAF tensor: %s", name);
            return TRELLIS_STATUS_NOT_FOUND;
        }
        if (tensor->type != GGML_TYPE_F32) {
            set_issue(first_issue, first_issue_size, "NAF tensor %s must be GGML F32", name);
            return TRELLIS_STATUS_PARSE_ERROR;
        }
        int n_dims = 0;
        int64_t shape[4];
        expected_weight_shape(i, &n_dims, shape);
        for (int d = 0; d < n_dims; ++d) {
            const int64_t expected = shape[n_dims - 1 - d];
            if (tensor->ne[d] != expected) {
                set_issue(
                    first_issue,
                    first_issue_size,
                    "NAF tensor %s GGML shape mismatch at dim%d: got %lld expected %lld",
                    name,
                    d,
                    (long long) tensor->ne[d],
                    (long long) expected);
                return TRELLIS_STATUS_PARSE_ERROR;
            }
        }
        for (int d = n_dims; d < GGML_MAX_DIMS; ++d) {
            if (tensor->ne[d] != 1) {
                set_issue(first_issue, first_issue_size, "NAF tensor %s rank mismatch", name);
                return TRELLIS_STATUS_PARSE_ERROR;
            }
        }
        weights->tensors[i] = tensor;
    }
    weights->n_tensors = TRELLIS_PIXAL_NAF_WEIGHT_COUNT;
    return TRELLIS_STATUS_OK;
}

static int weights_are_complete(const trellis_pixal_naf_weights * weights) {
    if (weights == NULL || weights->n_tensors != TRELLIS_PIXAL_NAF_WEIGHT_COUNT) {
        return 0;
    }
    for (size_t i = 0; i < TRELLIS_PIXAL_NAF_WEIGHT_COUNT; ++i) {
        if (weights->tensors[i].data == NULL) {
            return 0;
        }
    }
    return 1;
}

static int reflect_index(int index, int size) {
    while (index < 0 || index >= size) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * size - 2 - index;
        }
    }
    return index;
}

static void conv2d_reflect_nchw(
    const float * input,
    int batch,
    int in_channels,
    int height,
    int width,
    const float * weight,
    const float * bias,
    int out_channels,
    int kernel,
    float * output) {
    const int pad = kernel / 2;
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float sum = bias[oc];
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int ky = 0; ky < kernel; ++ky) {
                            const int iy = reflect_index(y + ky - pad, height);
                            for (int kx = 0; kx < kernel; ++kx) {
                                const int ix = reflect_index(x + kx - pad, width);
                                const size_t wi =
                                    (((size_t) oc * (size_t) in_channels + (size_t) ic) * (size_t) kernel +
                                     (size_t) ky) * (size_t) kernel + (size_t) kx;
                                sum += input[nchw_index(b, ic, iy, ix, in_channels, height, width)] * weight[wi];
                            }
                        }
                    }
                    output[nchw_index(b, oc, y, x, out_channels, height, width)] = sum;
                }
            }
        }
    }
}

static float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

static void group_norm_silu_nchw(
    const float * input,
    int batch,
    int channels,
    int height,
    int width,
    const float * gamma,
    const float * beta,
    float * output) {
    const int channels_per_group = channels / NAF_GROUPS;
    const size_t values_per_group =
        (size_t) channels_per_group * (size_t) height * (size_t) width;
    for (int b = 0; b < batch; ++b) {
        for (int group = 0; group < NAF_GROUPS; ++group) {
            const int c_begin = group * channels_per_group;
            const int c_end = c_begin + channels_per_group;
            double sum = 0.0;
            for (int c = c_begin; c < c_end; ++c) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        sum += input[nchw_index(b, c, y, x, channels, height, width)];
                    }
                }
            }
            const double mean = sum / (double) values_per_group;
            double variance_sum = 0.0;
            for (int c = c_begin; c < c_end; ++c) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const double delta =
                            (double) input[nchw_index(b, c, y, x, channels, height, width)] - mean;
                        variance_sum += delta * delta;
                    }
                }
            }
            const float inv_std = (float) (1.0 / sqrt(variance_sum / (double) values_per_group + 1e-5));
            for (int c = c_begin; c < c_end; ++c) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const size_t index = nchw_index(b, c, y, x, channels, height, width);
                        float value = ((float) ((double) input[index] - mean) * inv_std) * gamma[c] + beta[c];
                        output[index] = silu_f32(value);
                    }
                }
            }
        }
    }
}

static void adaptive_avg_pool_nchw_to_bhwc_into(
    const float * input,
    int batch,
    int channels,
    int input_h,
    int input_w,
    float * output,
    int output_channels,
    int output_channel_offset,
    int output_h,
    int output_w) {
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oy = 0; oy < output_h; ++oy) {
                const int y_begin = (int) (((int64_t) oy * input_h) / output_h);
                const int y_end = (int) ((((int64_t) (oy + 1) * input_h) + output_h - 1) / output_h);
                for (int ox = 0; ox < output_w; ++ox) {
                    const int x_begin = (int) (((int64_t) ox * input_w) / output_w);
                    const int x_end = (int) ((((int64_t) (ox + 1) * input_w) + output_w - 1) / output_w);
                    double sum = 0.0;
                    for (int iy = y_begin; iy < y_end; ++iy) {
                        for (int ix = x_begin; ix < x_end; ++ix) {
                            sum += input[nchw_index(b, c, iy, ix, channels, input_h, input_w)];
                        }
                    }
                    const int count = (y_end - y_begin) * (x_end - x_begin);
                    output[bhwc_index(
                        b,
                        oy,
                        ox,
                        output_channel_offset + c,
                        output_h,
                        output_w,
                        output_channels)] = (float) (sum / (double) count);
                }
            }
        }
    }
}

static void adaptive_avg_pool_bhwc(
    const float * input,
    int batch,
    int channels,
    int input_h,
    int input_w,
    float * output,
    int output_h,
    int output_w) {
    for (int b = 0; b < batch; ++b) {
        for (int oy = 0; oy < output_h; ++oy) {
            const int y_begin = (int) (((int64_t) oy * input_h) / output_h);
            const int y_end = (int) ((((int64_t) (oy + 1) * input_h) + output_h - 1) / output_h);
            for (int ox = 0; ox < output_w; ++ox) {
                const int x_begin = (int) (((int64_t) ox * input_w) / output_w);
                const int x_end = (int) ((((int64_t) (ox + 1) * input_w) + output_w - 1) / output_w);
                const int count = (y_end - y_begin) * (x_end - x_begin);
                for (int c = 0; c < channels; ++c) {
                    double sum = 0.0;
                    for (int iy = y_begin; iy < y_end; ++iy) {
                        for (int ix = x_begin; ix < x_end; ++ix) {
                            sum += input[bhwc_index(b, iy, ix, c, input_h, input_w, channels)];
                        }
                    }
                    output[bhwc_index(b, oy, ox, c, output_h, output_w, channels)] =
                        (float) (sum / (double) count);
                }
            }
        }
    }
}

static void bilinear_resize_align_corners_false_nchw(
    const float * input,
    int batch,
    int channels,
    int input_h,
    int input_w,
    float * output,
    int output_h,
    int output_w) {
    for (int oy = 0; oy < output_h; ++oy) {
        const float source_y = ((float) oy + 0.5f) * (float) input_h / (float) output_h - 0.5f;
        const int y_floor = (int) floorf(source_y);
        const float wy = source_y - (float) y_floor;
        const int y0 = y_floor < 0 ? 0 : (y_floor >= input_h ? input_h - 1 : y_floor);
        const int y1_raw = y_floor + 1;
        const int y1 = y1_raw < 0 ? 0 : (y1_raw >= input_h ? input_h - 1 : y1_raw);
        for (int ox = 0; ox < output_w; ++ox) {
            const float source_x = ((float) ox + 0.5f) * (float) input_w / (float) output_w - 0.5f;
            const int x_floor = (int) floorf(source_x);
            const float wx = source_x - (float) x_floor;
            const int x0 = x_floor < 0 ? 0 : (x_floor >= input_w ? input_w - 1 : x_floor);
            const int x1_raw = x_floor + 1;
            const int x1 = x1_raw < 0 ? 0 : (x1_raw >= input_w ? input_w - 1 : x1_raw);
            for (int b = 0; b < batch; ++b) {
                for (int c = 0; c < channels; ++c) {
                    const float v00 = input[nchw_index(b, c, y0, x0, channels, input_h, input_w)];
                    const float v01 = input[nchw_index(b, c, y0, x1, channels, input_h, input_w)];
                    const float v10 = input[nchw_index(b, c, y1, x0, channels, input_h, input_w)];
                    const float v11 = input[nchw_index(b, c, y1, x1, channels, input_h, input_w)];
                    const float top = v00 + (v01 - v00) * wx;
                    const float bottom = v10 + (v11 - v10) * wx;
                    output[nchw_index(b, c, oy, ox, channels, output_h, output_w)] =
                        top + (bottom - top) * wy;
                }
            }
        }
    }
}

static void apply_rope_2d_bhwc_inplace(
    float * queries,
    int batch,
    int height,
    int width,
    const float * periods) {
    const float two_pi = (float) (2.0 * M_PI);
    const size_t h_phase_count = (size_t) height * 16u;
    const size_t w_phase_count = (size_t) width * 16u;
    const size_t total_phase_count = 2u * (h_phase_count + w_phase_count);
    float * phase_cache = (float *) malloc(total_phase_count * sizeof(float));
    float * cos_h = phase_cache;
    float * sin_h = cos_h == NULL ? NULL : cos_h + h_phase_count;
    float * cos_w = sin_h == NULL ? NULL : sin_h + h_phase_count;
    float * sin_w = cos_w == NULL ? NULL : cos_w + w_phase_count;
    if (phase_cache != NULL) {
        for (int y = 0; y < height; ++y) {
            const float coord_h = 2.0f * (((float) y + 0.5f) / (float) height) - 1.0f;
            for (int d = 0; d < 16; ++d) {
                const float angle = two_pi * coord_h / periods[d];
                cos_h[(size_t) y * 16u + (size_t) d] = cosf(angle);
                sin_h[(size_t) y * 16u + (size_t) d] = sinf(angle);
            }
        }
        for (int x = 0; x < width; ++x) {
            const float coord_w = 2.0f * (((float) x + 0.5f) / (float) width) - 1.0f;
            for (int d = 0; d < 16; ++d) {
                const float angle = two_pi * coord_w / periods[d];
                cos_w[(size_t) x * 16u + (size_t) d] = cosf(angle);
                sin_w[(size_t) x * 16u + (size_t) d] = sinf(angle);
            }
        }
    }
    for (int y = 0; y < height; ++y) {
        const float coord_h = 2.0f * (((float) y + 0.5f) / (float) height) - 1.0f;
        for (int x = 0; x < width; ++x) {
            const float coord_w = 2.0f * (((float) x + 0.5f) / (float) width) - 1.0f;
            float cos_phase[32];
            float sin_phase[32];
            for (int d = 0; d < 32; ++d) {
                if (phase_cache != NULL) {
                    const size_t index = (size_t) (d & 15);
                    cos_phase[d] = d < 16 ?
                        cos_h[(size_t) y * 16u + index] :
                        cos_w[(size_t) x * 16u + index];
                    sin_phase[d] = d < 16 ?
                        sin_h[(size_t) y * 16u + index] :
                        sin_w[(size_t) x * 16u + index];
                } else {
                    const float coord = d < 16 ? coord_h : coord_w;
                    const float angle = two_pi * coord / periods[d & 15];
                    cos_phase[d] = cosf(angle);
                    sin_phase[d] = sinf(angle);
                }
            }
            for (int b = 0; b < batch; ++b) {
                for (int head = 0; head < TRELLIS_PIXAL_NAF_HEADS; ++head) {
                    const int channel_base = head * TRELLIS_PIXAL_NAF_HEAD_DIM;
                    for (int d = 0; d < 32; ++d) {
                        const size_t first = bhwc_index(
                            b, y, x, channel_base + d, height, width, TRELLIS_PIXAL_NAF_QUERY_CHANNELS);
                        const size_t second = bhwc_index(
                            b, y, x, channel_base + d + 32, height, width, TRELLIS_PIXAL_NAF_QUERY_CHANNELS);
                        const float a = queries[first];
                        const float z = queries[second];
                        queries[first] = a * cos_phase[d] - z * sin_phase[d];
                        queries[second] = z * cos_phase[d] + a * sin_phase[d];
                    }
                }
            }
        }
    }
    free(phase_cache);
}

static trellis_status run_encoder_branch(
    const trellis_pixal_naf_weights * weights,
    size_t branch_base,
    int kernel,
    const float * image,
    int batch,
    int image_h,
    int image_w,
    float * buffer_a,
    float * buffer_b,
    float * queries,
    int query_channel_offset,
    int output_h,
    int output_w) {
    const trellis_pixal_naf_host_tensor * input_w = &weights->tensors[branch_base];
    const trellis_pixal_naf_host_tensor * input_b = &weights->tensors[branch_base + 1];
    conv2d_reflect_nchw(
        image,
        batch,
        TRELLIS_PIXAL_NAF_IMAGE_CHANNELS,
        image_h,
        image_w,
        input_w->data,
        input_b->data,
        TRELLIS_PIXAL_NAF_ENCODER_CHANNELS,
        kernel,
        buffer_a);

    for (int block = 0; block < NAF_ENCODER_BLOCKS; ++block) {
        const size_t base = branch_base + 2 + (size_t) block * NAF_BLOCK_TENSORS;
        group_norm_silu_nchw(
            buffer_a,
            batch,
            TRELLIS_PIXAL_NAF_ENCODER_CHANNELS,
            image_h,
            image_w,
            weights->tensors[base + 0].data,
            weights->tensors[base + 1].data,
            buffer_b);
        conv2d_reflect_nchw(
            buffer_b,
            batch,
            TRELLIS_PIXAL_NAF_ENCODER_CHANNELS,
            image_h,
            image_w,
            weights->tensors[base + 2].data,
            weights->tensors[base + 3].data,
            TRELLIS_PIXAL_NAF_ENCODER_CHANNELS,
            kernel,
            buffer_a);
        group_norm_silu_nchw(
            buffer_a,
            batch,
            TRELLIS_PIXAL_NAF_ENCODER_CHANNELS,
            image_h,
            image_w,
            weights->tensors[base + 4].data,
            weights->tensors[base + 5].data,
            buffer_b);
        conv2d_reflect_nchw(
            buffer_b,
            batch,
            TRELLIS_PIXAL_NAF_ENCODER_CHANNELS,
            image_h,
            image_w,
            weights->tensors[base + 6].data,
            weights->tensors[base + 7].data,
            TRELLIS_PIXAL_NAF_ENCODER_CHANNELS,
            kernel,
            buffer_a);
    }

    adaptive_avg_pool_nchw_to_bhwc_into(
        buffer_a,
        batch,
        TRELLIS_PIXAL_NAF_ENCODER_CHANNELS,
        image_h,
        image_w,
        queries,
        TRELLIS_PIXAL_NAF_QUERY_CHANNELS,
        query_channel_offset,
        output_h,
        output_w);
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_pixal_naf_image_encoder_host(
    const trellis_pixal_naf_weights * weights,
    const float * image,
    int batch,
    int image_h,
    int image_w,
    int output_h,
    int output_w,
    float * queries,
    size_t queries_count) {
    size_t expected_queries = 0;
    if (!weights_are_complete(weights) || image == NULL || queries == NULL ||
        !checked_elements4(batch, TRELLIS_PIXAL_NAF_QUERY_CHANNELS, output_h, output_w, &expected_queries) ||
        expected_queries != queries_count || image_h <= 1 || image_w <= 1) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    int encoder_h = image_h;
    int encoder_w = image_w;
    if ((int64_t) image_h > 4ll * output_h || (int64_t) image_w > 4ll * output_w) {
        const int64_t h_limit = 4ll * output_h < 4ll * output_w ? 4ll * output_h : 4ll * output_w;
        const int64_t w_limit = 4ll * output_w < 4ll * output_h ? 4ll * output_w : 4ll * output_h;
        if ((int64_t) encoder_h > h_limit) encoder_h = (int) h_limit;
        if ((int64_t) encoder_w > w_limit) encoder_w = (int) w_limit;
    }
    if (encoder_h <= 1 || encoder_w <= 1) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    const float * encoder_image = image;
    float * resized_image = NULL;
    if (encoder_h != image_h || encoder_w != image_w) {
        size_t resized_count = 0;
        if (!checked_elements4(batch, TRELLIS_PIXAL_NAF_IMAGE_CHANNELS, encoder_h, encoder_w, &resized_count)) {
            return TRELLIS_STATUS_OUT_OF_MEMORY;
        }
        resized_image = (float *) malloc(resized_count * sizeof(float));
        if (resized_image == NULL) {
            return TRELLIS_STATUS_OUT_OF_MEMORY;
        }
        bilinear_resize_align_corners_false_nchw(
            image,
            batch,
            TRELLIS_PIXAL_NAF_IMAGE_CHANNELS,
            image_h,
            image_w,
            resized_image,
            encoder_h,
            encoder_w);
        encoder_image = resized_image;
    }

    size_t branch_count = 0;
    if (!checked_elements4(
            batch,
            TRELLIS_PIXAL_NAF_ENCODER_CHANNELS,
            encoder_h,
            encoder_w,
            &branch_count) ||
        branch_count > SIZE_MAX / sizeof(float)) {
        free(resized_image);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    float * buffer_a = (float *) malloc(branch_count * sizeof(float));
    float * buffer_b = (float *) malloc(branch_count * sizeof(float));
    if (buffer_a == NULL || buffer_b == NULL) {
        free(buffer_a);
        free(buffer_b);
        free(resized_image);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    trellis_status status = run_encoder_branch(
        weights,
        NAF_ENCODER_BASE,
        1,
        encoder_image,
        batch,
        encoder_h,
        encoder_w,
        buffer_a,
        buffer_b,
        queries,
        0,
        output_h,
        output_w);
    if (status == TRELLIS_STATUS_OK) {
        status = run_encoder_branch(
            weights,
            NAF_SEM_ENCODER_BASE,
            3,
            encoder_image,
            batch,
            encoder_h,
            encoder_w,
            buffer_a,
            buffer_b,
            queries,
            TRELLIS_PIXAL_NAF_ENCODER_CHANNELS,
            output_h,
            output_w);
    }
    if (status == TRELLIS_STATUS_OK) {
        apply_rope_2d_bhwc_inplace(
            queries,
            batch,
            output_h,
            output_w,
            weights->tensors[NAF_ROPE_PERIODS].data);
    }

    free(buffer_a);
    free(buffer_b);
    free(resized_image);
    return status;
}

trellis_status trellis_pixal_naf_query_key_forward_host(
    const trellis_pixal_naf_weights * weights,
    const float * image,
    int batch,
    int image_h,
    int image_w,
    int output_h,
    int output_w,
    int key_h,
    int key_w,
    float * query_bhwc,
    size_t query_count,
    float * key_bhwc,
    size_t key_count) {
    size_t expected_query = 0;
    size_t expected_key = 0;
    if (query_bhwc == NULL || key_bhwc == NULL ||
        !checked_elements4(batch, output_h, output_w, TRELLIS_PIXAL_NAF_QUERY_CHANNELS, &expected_query) ||
        !checked_elements4(batch, key_h, key_w, TRELLIS_PIXAL_NAF_QUERY_CHANNELS, &expected_key) ||
        expected_query != query_count || expected_key != key_count) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_status status = trellis_pixal_naf_image_encoder_host(
        weights,
        image,
        batch,
        image_h,
        image_w,
        output_h,
        output_w,
        query_bhwc,
        query_count);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    adaptive_avg_pool_bhwc(
        query_bhwc,
        batch,
        TRELLIS_PIXAL_NAF_QUERY_CHANNELS,
        output_h,
        output_w,
        key_bhwc,
        key_h,
        key_w);
    return TRELLIS_STATUS_OK;
}

static int ggml_weights_are_complete(const trellis_pixal_naf_ggml_weights * weights) {
    if (weights == NULL || weights->n_tensors != TRELLIS_PIXAL_NAF_WEIGHT_COUNT) {
        return 0;
    }
    for (size_t i = 0; i < TRELLIS_PIXAL_NAF_WEIGHT_COUNT; ++i) {
        if (weights->tensors[i] == NULL) {
            return 0;
        }
    }
    return 1;
}

static struct ggml_tensor * ggml_repeat_channel_param(
    struct ggml_context * ctx,
    struct ggml_tensor * parameter,
    struct ggml_tensor * target) {
    struct ggml_tensor * shaped = ggml_reshape_4d(ctx, parameter, 1, 1, parameter->ne[0], 1);
    return ggml_repeat(ctx, shaped, target);
}

static struct ggml_tensor * ggml_add_channel_bias(
    struct ggml_context * ctx,
    struct ggml_tensor * value,
    struct ggml_tensor * bias) {
    return ggml_add(ctx, value, ggml_repeat_channel_param(ctx, bias, value));
}

static struct ggml_tensor * ggml_reflect_pad2d_one(
    struct ggml_context * ctx,
    struct ggml_tensor * input) {
    if (ctx == NULL || input == NULL || input->ne[0] <= 1 || input->ne[1] <= 1) {
        return NULL;
    }

    struct ggml_tensor * left = ggml_view_4d(
        ctx,
        input,
        1,
        input->ne[1],
        input->ne[2],
        input->ne[3],
        input->nb[1],
        input->nb[2],
        input->nb[3],
        input->nb[0]);
    struct ggml_tensor * right = ggml_view_4d(
        ctx,
        input,
        1,
        input->ne[1],
        input->ne[2],
        input->ne[3],
        input->nb[1],
        input->nb[2],
        input->nb[3],
        (size_t) (input->ne[0] - 2) * input->nb[0]);
    struct ggml_tensor * width_padded = ggml_concat(ctx, ggml_concat(ctx, left, input, 0), right, 0);

    struct ggml_tensor * top = ggml_view_4d(
        ctx,
        width_padded,
        width_padded->ne[0],
        1,
        width_padded->ne[2],
        width_padded->ne[3],
        width_padded->nb[1],
        width_padded->nb[2],
        width_padded->nb[3],
        width_padded->nb[1]);
    struct ggml_tensor * bottom = ggml_view_4d(
        ctx,
        width_padded,
        width_padded->ne[0],
        1,
        width_padded->ne[2],
        width_padded->ne[3],
        width_padded->nb[1],
        width_padded->nb[2],
        width_padded->nb[3],
        (size_t) (width_padded->ne[1] - 2) * width_padded->nb[1]);
    return ggml_concat(ctx, ggml_concat(ctx, top, width_padded, 1), bottom, 1);
}

static struct ggml_tensor * ggml_conv_reflect(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * weight,
    struct ggml_tensor * bias,
    int kernel,
    int use_direct_conv) {
    if (kernel == 3) {
        input = ggml_reflect_pad2d_one(ctx, input);
        if (input == NULL) {
            return NULL;
        }
    } else if (kernel != 1) {
        return NULL;
    }
    struct ggml_tensor * output = use_direct_conv ?
        ggml_conv_2d_direct(ctx, weight, input, 1, 1, 0, 0, 1, 1) :
        ggml_conv_2d(ctx, weight, input, 1, 1, 0, 0, 1, 1);
    return output == NULL ? NULL : ggml_add_channel_bias(ctx, output, bias);
}

static struct ggml_tensor * ggml_group_norm_silu_affine(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * gamma,
    struct ggml_tensor * beta) {
    struct ggml_tensor * output = ggml_group_norm(ctx, input, NAF_GROUPS, 1e-5f);
    output = ggml_mul(ctx, output, ggml_repeat_channel_param(ctx, gamma, output));
    output = ggml_add(ctx, output, ggml_repeat_channel_param(ctx, beta, output));
    return ggml_silu(ctx, output);
}

static struct ggml_tensor * ggml_encoder_branch(
    struct ggml_context * ctx,
    const trellis_pixal_naf_ggml_weights * weights,
    size_t branch_base,
    int kernel,
    int use_direct_conv,
    struct ggml_tensor * image) {
    struct ggml_tensor * value = ggml_conv_reflect(
        ctx,
        image,
        weights->tensors[branch_base],
        weights->tensors[branch_base + 1],
        kernel,
        use_direct_conv);
    if (value == NULL) {
        return NULL;
    }
    for (int block = 0; block < NAF_ENCODER_BLOCKS; ++block) {
        const size_t base = branch_base + 2 + (size_t) block * NAF_BLOCK_TENSORS;
        value = ggml_group_norm_silu_affine(
            ctx, value, weights->tensors[base + 0], weights->tensors[base + 1]);
        value = ggml_conv_reflect(
            ctx, value, weights->tensors[base + 2], weights->tensors[base + 3], kernel, use_direct_conv);
        if (value == NULL) {
            return NULL;
        }
        value = ggml_group_norm_silu_affine(
            ctx, value, weights->tensors[base + 4], weights->tensors[base + 5]);
        value = ggml_conv_reflect(
            ctx, value, weights->tensors[base + 6], weights->tensors[base + 7], kernel, use_direct_conv);
        if (value == NULL) {
            return NULL;
        }
    }
    return value;
}

static struct ggml_tensor * ggml_adaptive_pool_divisible(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    int output_w,
    int output_h) {
    if (input == NULL || output_w <= 0 || output_h <= 0 ||
        input->ne[0] < output_w || input->ne[1] < output_h ||
        input->ne[0] % output_w != 0 || input->ne[1] % output_h != 0) {
        return NULL;
    }
    if (input->ne[0] == output_w && input->ne[1] == output_h) {
        return input;
    }
    const int kernel_w = (int) input->ne[0] / output_w;
    const int kernel_h = (int) input->ne[1] / output_h;
    return ggml_pool_2d(
        ctx,
        input,
        GGML_OP_POOL_AVG,
        kernel_w,
        kernel_h,
        kernel_w,
        kernel_h,
        0.0f,
        0.0f);
}

trellis_status trellis_pixal_naf_query_key_forward_ggml_host(
    const trellis_backend_context * backend,
    const trellis_pixal_naf_ggml_weights * weights,
    const float * image_nchw,
    int batch,
    int image_h,
    int image_w,
    int output_h,
    int output_w,
    int key_h,
    int key_w,
    float * query_bhwc,
    size_t query_count,
    float * key_bhwc,
    size_t key_count) {
    size_t expected_image = 0;
    size_t expected_query = 0;
    size_t expected_key = 0;
    if (backend == NULL || backend->backend == NULL || !ggml_weights_are_complete(weights) ||
        image_nchw == NULL || query_bhwc == NULL || key_bhwc == NULL || image_h <= 1 || image_w <= 1 ||
        !checked_elements4(batch, TRELLIS_PIXAL_NAF_IMAGE_CHANNELS, image_h, image_w, &expected_image) ||
        !checked_elements4(batch, output_h, output_w, TRELLIS_PIXAL_NAF_QUERY_CHANNELS, &expected_query) ||
        !checked_elements4(batch, key_h, key_w, TRELLIS_PIXAL_NAF_QUERY_CHANNELS, &expected_key) ||
        expected_query != query_count || expected_key != key_count) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    int encoder_h = image_h;
    int encoder_w = image_w;
    if ((int64_t) image_h > 4ll * output_h || (int64_t) image_w > 4ll * output_w) {
        const int64_t limit = 4ll * output_h < 4ll * output_w ? 4ll * output_h : 4ll * output_w;
        if ((int64_t) encoder_h > limit) encoder_h = (int) limit;
        if ((int64_t) encoder_w > limit) encoder_w = (int) limit;
    }
    if (encoder_h <= 1 || encoder_w <= 1 || encoder_w < output_w || encoder_h < output_h ||
        encoder_w % output_w != 0 || encoder_h % output_h != 0) {
        /* GGML has regular AVG_POOL_2D but no general adaptive pooling op. */
        return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }

    const size_t graph_nodes = 4096;
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * graph_nodes +
            ggml_graph_overhead_custom(graph_nodes, false) + 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    struct ggml_tensor * image = ggml_new_tensor_4d(
        ctx,
        GGML_TYPE_F32,
        image_w,
        image_h,
        TRELLIS_PIXAL_NAF_IMAGE_CHANNELS,
        batch);
    struct ggml_tensor * encoder_image = image;
    if (encoder_w != image_w || encoder_h != image_h) {
        encoder_image = ggml_interpolate(
            ctx,
            image,
            encoder_w,
            encoder_h,
            TRELLIS_PIXAL_NAF_IMAGE_CHANNELS,
            batch,
            GGML_SCALE_MODE_BILINEAR);
    }
    const int use_direct_conv = backend->kind != TRELLIS_BACKEND_VULKAN;
    struct ggml_tensor * encoder = ggml_encoder_branch(
        ctx, weights, NAF_ENCODER_BASE, 1, use_direct_conv, encoder_image);
    struct ggml_tensor * sem_encoder = ggml_encoder_branch(
        ctx, weights, NAF_SEM_ENCODER_BASE, 3, use_direct_conv, encoder_image);
    struct ggml_tensor * query_whcn =
        encoder == NULL || sem_encoder == NULL ? NULL : ggml_concat(ctx, encoder, sem_encoder, 2);
    query_whcn = ggml_adaptive_pool_divisible(ctx, query_whcn, output_w, output_h);
    struct ggml_tensor * query_cwhn = query_whcn == NULL ? NULL :
        ggml_cont(ctx, ggml_permute(ctx, query_whcn, 1, 2, 0, 3));
    if (image == NULL || query_cwhn == NULL || query_cwhn->ne[0] != TRELLIS_PIXAL_NAF_QUERY_CHANNELS ||
        query_cwhn->ne[1] != output_w || query_cwhn->ne[2] != output_h || query_cwhn->ne[3] != batch) {
        ggml_free(ctx);
        return TRELLIS_STATUS_ERROR;
    }
    ggml_set_input(image);
    ggml_set_output(query_cwhn);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    if (graph == NULL) {
        ggml_free(ctx);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_build_forward_expand(graph, query_cwhn);
    ggml_gallocr_t allocator = trellis_backend_new_graph_allocator(backend);
    if (allocator == NULL || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (allocator != NULL) ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    TRELLIS_INFO(
        "Pixal NAF encoder: backend=%s image=%dx%d query=%dx%d key=%dx%d graph_workspace=%.1f MiB query_host=%.1f MiB key_host=%.1f MiB",
        trellis_backend_kind_name(backend->kind),
        image_h,
        image_w,
        output_h,
        output_w,
        key_h,
        key_w,
        (double) ggml_gallocr_get_buffer_size(allocator, 0) / (1024.0 * 1024.0),
        (double) (query_count * sizeof(float)) / (1024.0 * 1024.0),
        (double) (key_count * sizeof(float)) / (1024.0 * 1024.0));

    ggml_backend_tensor_set(image, image_nchw, 0, expected_image * sizeof(float));
    trellis_status status = trellis_backend_compute_graph(backend, graph);
    if (status == TRELLIS_STATUS_OK) {
        ggml_backend_tensor_get(query_cwhn, query_bhwc, 0, query_count * sizeof(float));
        float periods[16];
        ggml_backend_tensor_get(weights->tensors[NAF_ROPE_PERIODS], periods, 0, sizeof(periods));
        apply_rope_2d_bhwc_inplace(query_bhwc, batch, output_h, output_w, periods);
        adaptive_avg_pool_bhwc(
            query_bhwc,
            batch,
            TRELLIS_PIXAL_NAF_QUERY_CHANNELS,
            output_h,
            output_w,
            key_bhwc,
            key_h,
            key_w);
    }

    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return status;
}
