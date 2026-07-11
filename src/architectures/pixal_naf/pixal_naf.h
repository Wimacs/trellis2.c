#ifndef TRELLIS2_C_ARCHITECTURES_PIXAL_NAF_H
#define TRELLIS2_C_ARCHITECTURES_PIXAL_NAF_H

#include "trellis.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    TRELLIS_PIXAL_NAF_WEIGHT_COUNT = 37,
    TRELLIS_PIXAL_NAF_IMAGE_CHANNELS = 3,
    TRELLIS_PIXAL_NAF_ENCODER_CHANNELS = 128,
    TRELLIS_PIXAL_NAF_QUERY_CHANNELS = 256,
    TRELLIS_PIXAL_NAF_HEADS = 4,
    TRELLIS_PIXAL_NAF_HEAD_DIM = 64,
    TRELLIS_PIXAL_NAF_KERNEL_SIZE = 9,
};

typedef struct trellis_pixal_naf_host_tensor {
    float * data;
    size_t count;
    int n_dims;
    int64_t shape[4]; /* safetensors/PyTorch order */
} trellis_pixal_naf_host_tensor;

typedef struct trellis_pixal_naf_weights {
    trellis_pixal_naf_host_tensor tensors[TRELLIS_PIXAL_NAF_WEIGHT_COUNT];
    size_t n_tensors;
} trellis_pixal_naf_weights;

typedef struct trellis_pixal_naf_ggml_weights {
    struct ggml_tensor * tensors[TRELLIS_PIXAL_NAF_WEIGHT_COUNT];
    size_t n_tensors;
} trellis_pixal_naf_ggml_weights;

/* Returns the unmodified ValeoAI state_dict name for an index, or NULL. */
const char * trellis_pixal_naf_weight_name(size_t index);

/* Loads the converter output directly to host F32 memory. weights must be
 * zero-initialized (and freed before reuse). The file must contain exactly the
 * 37 official tensors, with their complete names and shapes. */
trellis_status trellis_pixal_naf_load_weights_f32(
    const char * safetensors_path,
    trellis_pixal_naf_weights * weights,
    char * first_issue,
    size_t first_issue_size);

void trellis_pixal_naf_free_weights(trellis_pixal_naf_weights * weights);

/* Strictly binds a tensor store produced from the same converter output. */
trellis_status trellis_pixal_naf_bind_ggml_weights(
    trellis_tensor_store * store,
    trellis_pixal_naf_ggml_weights * weights,
    char * first_issue,
    size_t first_issue_size);

/* Exact inference-mode ImageEncoder reference. Input is NCHW and output is
 * BHWC, matching the projected-condition token layout.
 * queries_count must be batch * 256 * output_h * output_w. */
trellis_status trellis_pixal_naf_image_encoder_host(
    const trellis_pixal_naf_weights * weights,
    const float * image,
    int batch,
    int image_h,
    int image_w,
    int output_h,
    int output_w,
    float * queries,
    size_t queries_count);

/* Exact small-size host reference for query/key production. Both outputs are
 * BHWC. Key is adaptive_avg_pool2d(query, [key_h,key_w]) after 2D RoPE. */
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
    size_t key_count);

/* Device-backed encoder path. Convolution, GroupNorm, SiLU, reflect padding,
 * concatenation and divisible adaptive pooling are built as a GGML graph.
 * Only the 256-channel unrotated query is copied to host; exact inference RoPE
 * and arbitrary adaptive key pooling are then applied on host. No DINO value or
 * 1024-channel NAF output is materialized here. Outputs are BHWC. */
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
    size_t key_count);

#ifdef __cplusplus
}
#endif

#endif
