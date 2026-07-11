#ifndef TRELLIS2_C_MESH_RIGGING_SAMPLER_H
#define TRELLIS2_C_MESH_RIGGING_SAMPLER_H

#include "trellis.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trellis_tokenskin_sampler_options {
    /* Explicit greedy mode.  A non-positive temperature also selects greedy. */
    bool greedy;
    float temperature;
    float repetition_penalty;
    /* Zero disables top-k.  Values larger than the active vocabulary are
     * clamped to the active vocabulary, matching Transformers. */
    size_t top_k;
    /* One disables top-p; stochastic sampling requires 0 < top_p <= 1. */
    float top_p;
    /* Transformers keeps eos_count+1 candidates for beam sampling (2 for
     * TokenSkin's single global EOS); ordinary sampling keeps one. */
    size_t min_tokens_to_keep;
} trellis_tokenskin_sampler_options;

#define TRELLIS_TOKENSKIN_SAMPLER_OPTIONS_INIT { \
    false, 1.0f, 1.0f, 0u, 1.0f, 1u \
}

/* Reusable per-sequence CPU workspace.  One instance is not thread-safe. */
typedef struct trellis_tokenskin_sampler {
    size_t vocab_capacity;
    uint64_t rng_state;
    float * scores;
    void * sorted_items;
    uint8_t * seen;
} trellis_tokenskin_sampler;

trellis_status trellis_tokenskin_sampler_init(
    trellis_tokenskin_sampler * sampler,
    size_t vocab_capacity,
    uint64_t seed);

void trellis_tokenskin_sampler_free(trellis_tokenskin_sampler * sampler);

void trellis_tokenskin_sampler_seed(
    trellis_tokenskin_sampler * sampler,
    uint64_t seed);

/* Applies the same ordering as HF generation:
 *   RepetitionPenalty -> grammar mask -> Temperature -> TopK -> TopP.
 * generated_ids must contain only IDs returned by generate(), excluding the
 * 512 mesh-prefix embeddings and manually prepended BOS/class/skeleton IDs.
 * Duplicate generated IDs are penalized once, as in torch.scatter.
 *
 * NaN logits and masked logits become -INFINITY.  If no usable candidate
 * remains, TRELLIS_STATUS_PARSE_ERROR is returned.  output_count and
 * vocab_size must match.  The workspace may have a larger capacity. */
trellis_status trellis_tokenskin_sampler_process_logits(
    trellis_tokenskin_sampler * sampler,
    const float * logits,
    const uint8_t * grammar_mask,
    size_t vocab_size,
    const int32_t * generated_ids,
    size_t generated_count,
    const trellis_tokenskin_sampler_options * options,
    float * output_logits,
    size_t output_count);

/* Processes logits and either takes the first argmax (greedy) or samples from
 * the final categorical distribution using the instance's deterministic
 * SplitMix64 stream. */
trellis_status trellis_tokenskin_sampler_sample(
    trellis_tokenskin_sampler * sampler,
    const float * logits,
    const uint8_t * grammar_mask,
    size_t vocab_size,
    const int32_t * generated_ids,
    size_t generated_count,
    const trellis_tokenskin_sampler_options * options,
    int32_t * token_out);

#ifdef __cplusplus
}
#endif

#endif
