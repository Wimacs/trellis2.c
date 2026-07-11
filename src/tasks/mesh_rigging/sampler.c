#include "sampler.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct sampler_item {
    float score;
    float probability;
    size_t token;
} sampler_item;

static bool sampler_options_are_valid(
    const trellis_tokenskin_sampler_options * options) {
    if (options == NULL || !isfinite(options->repetition_penalty) ||
        options->repetition_penalty <= 0.0f) {
        return false;
    }
    const bool greedy = options->greedy || options->temperature <= 0.0f;
    if (!greedy && (!isfinite(options->temperature) || options->temperature <= 0.0f)) {
        return false;
    }
    return isfinite(options->top_p) && options->top_p > 0.0f &&
        options->top_p <= 1.0f && options->min_tokens_to_keep > 0u;
}

static bool sampler_workspace_is_valid(
    const trellis_tokenskin_sampler * sampler,
    size_t vocab_size) {
    return sampler != NULL && vocab_size != 0u &&
        vocab_size <= sampler->vocab_capacity &&
        sampler->scores != NULL && sampler->sorted_items != NULL && sampler->seen != NULL;
}

trellis_status trellis_tokenskin_sampler_init(
    trellis_tokenskin_sampler * sampler,
    size_t vocab_capacity,
    uint64_t seed) {
    if (sampler == NULL || vocab_capacity == 0u ||
        vocab_capacity > SIZE_MAX / sizeof(float) ||
        vocab_capacity > SIZE_MAX / sizeof(sampler_item)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_tokenskin_sampler initialized;
    memset(&initialized, 0, sizeof(initialized));
    initialized.vocab_capacity = vocab_capacity;
    initialized.rng_state = seed;
    initialized.scores = (float *) malloc(vocab_capacity * sizeof(float));
    initialized.sorted_items = malloc(vocab_capacity * sizeof(sampler_item));
    initialized.seen = (uint8_t *) malloc(vocab_capacity * sizeof(uint8_t));
    if (initialized.scores == NULL || initialized.sorted_items == NULL ||
        initialized.seen == NULL) {
        trellis_tokenskin_sampler_free(&initialized);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    *sampler = initialized;
    return TRELLIS_STATUS_OK;
}

void trellis_tokenskin_sampler_free(trellis_tokenskin_sampler * sampler) {
    if (sampler == NULL) return;
    free(sampler->scores);
    free(sampler->sorted_items);
    free(sampler->seen);
    memset(sampler, 0, sizeof(*sampler));
}

void trellis_tokenskin_sampler_seed(
    trellis_tokenskin_sampler * sampler,
    uint64_t seed) {
    if (sampler != NULL) sampler->rng_state = seed;
}

static int item_compare_descending(const void * left, const void * right) {
    const sampler_item * a = (const sampler_item *) left;
    const sampler_item * b = (const sampler_item *) right;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    if (a->token < b->token) return -1;
    if (a->token > b->token) return 1;
    return 0;
}

static int item_compare_ascending(const void * left, const void * right) {
    const sampler_item * a = (const sampler_item *) left;
    const sampler_item * b = (const sampler_item *) right;
    if (a->score < b->score) return -1;
    if (a->score > b->score) return 1;
    if (a->token < b->token) return -1;
    if (a->token > b->token) return 1;
    return 0;
}

static bool scores_have_candidate(const float * scores, size_t vocab_size) {
    for (size_t i = 0; i < vocab_size; ++i) {
        if (scores[i] > -INFINITY && !isnan(scores[i])) return true;
    }
    return false;
}

static void apply_top_k(
    float * scores,
    sampler_item * items,
    size_t vocab_size,
    size_t top_k) {
    if (top_k == 0u || top_k >= vocab_size) return;
    for (size_t i = 0; i < vocab_size; ++i) {
        items[i].score = scores[i];
        items[i].probability = 0.0f;
        items[i].token = i;
    }
    qsort(items, vocab_size, sizeof(items[0]), item_compare_descending);
    const float threshold = items[top_k - 1u].score;
    for (size_t i = 0; i < vocab_size; ++i) {
        /* Strict comparison preserves all ties at the kth score, matching
         * scores < torch.topk(...)[..., -1]. */
        if (scores[i] < threshold) scores[i] = -INFINITY;
    }
}

static trellis_status assign_softmax_probabilities(
    sampler_item * items,
    const float * scores,
    size_t vocab_size) {
    size_t positive_infinities = 0u;
    float maximum = -INFINITY;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (isinf(scores[i]) && scores[i] > 0.0f) {
            ++positive_infinities;
        } else if (isfinite(scores[i]) && scores[i] > maximum) {
            maximum = scores[i];
        }
    }

    float sum = 0.0f;
    if (positive_infinities == 0u) {
        if (!isfinite(maximum)) return TRELLIS_STATUS_PARSE_ERROR;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (isfinite(scores[i])) sum += expf(scores[i] - maximum);
        }
        if (!(sum > 0.0f) || !isfinite(sum)) return TRELLIS_STATUS_PARSE_ERROR;
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        items[i].score = scores[i];
        items[i].token = i;
        if (positive_infinities != 0u) {
            items[i].probability = isinf(scores[i]) && scores[i] > 0.0f ?
                1.0f / (float) positive_infinities : 0.0f;
        } else {
            items[i].probability = isfinite(scores[i]) ?
                expf(scores[i] - maximum) / sum : 0.0f;
        }
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status apply_top_p(
    float * scores,
    sampler_item * items,
    size_t vocab_size,
    float top_p,
    size_t min_tokens_to_keep) {
    if (top_p >= 1.0f) return TRELLIS_STATUS_OK;
    trellis_status status = assign_softmax_probabilities(items, scores, vocab_size);
    if (status != TRELLIS_STATUS_OK) return status;
    qsort(items, vocab_size, sizeof(items[0]), item_compare_ascending);

    const float remove_probability = 1.0f - top_p;
    float cumulative_probability = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cumulative_probability += items[i].probability;
        /* HF uses <= and then unconditionally keeps the final sorted token. */
        const bool remove = cumulative_probability <= remove_probability &&
            i + min_tokens_to_keep < vocab_size;
        if (remove) scores[items[i].token] = -INFINITY;
    }
    return scores_have_candidate(scores, vocab_size) ?
        TRELLIS_STATUS_OK : TRELLIS_STATUS_PARSE_ERROR;
}

static trellis_status process_into_workspace(
    trellis_tokenskin_sampler * sampler,
    const float * logits,
    const uint8_t * grammar_mask,
    size_t vocab_size,
    const int32_t * generated_ids,
    size_t generated_count,
    const trellis_tokenskin_sampler_options * options) {
    if (!sampler_workspace_is_valid(sampler, vocab_size) || logits == NULL ||
        grammar_mask == NULL || (generated_ids == NULL && generated_count != 0u) ||
        !sampler_options_are_valid(options)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        sampler->scores[i] = isnan(logits[i]) ? -INFINITY : logits[i];
    }
    memset(sampler->seen, 0, vocab_size * sizeof(sampler->seen[0]));

    /* RepetitionPenaltyLogitsProcessor gathers and scatters token IDs.  A
     * boolean set is equivalent and makes duplicate IDs explicitly idempotent. */
    for (size_t i = 0; i < generated_count; ++i) {
        const int32_t token = generated_ids[i];
        if (token < 0 || (size_t) token >= vocab_size) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        if (sampler->seen[token]) continue;
        sampler->seen[token] = 1u;
        float score = sampler->scores[token];
        score = score < 0.0f ? score * options->repetition_penalty :
            score / options->repetition_penalty;
        sampler->scores[token] = isnan(score) ? -INFINITY : score;
    }

    /* VocabSwitchingLogitsProcessor is a custom logits processor, so it runs
     * after repetition and before the sampling warpers. */
    for (size_t i = 0; i < vocab_size; ++i) {
        if (grammar_mask[i] == 0u) sampler->scores[i] = -INFINITY;
    }
    if (!scores_have_candidate(sampler->scores, vocab_size)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    const bool greedy = options->greedy || options->temperature <= 0.0f;
    if (greedy) return TRELLIS_STATUS_OK;

    for (size_t i = 0; i < vocab_size; ++i) {
        if (sampler->scores[i] > -INFINITY) {
            const float tempered = sampler->scores[i] / options->temperature;
            sampler->scores[i] = isnan(tempered) ? -INFINITY : tempered;
        }
    }
    if (!scores_have_candidate(sampler->scores, vocab_size)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    sampler_item * items = (sampler_item *) sampler->sorted_items;
    size_t top_k = options->top_k;
    if (top_k != 0u && top_k < options->min_tokens_to_keep) {
        top_k = options->min_tokens_to_keep;
    }
    if (top_k > vocab_size) top_k = vocab_size;
    apply_top_k(sampler->scores, items, vocab_size, top_k);
    if (!scores_have_candidate(sampler->scores, vocab_size)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    const size_t min_tokens_to_keep = options->min_tokens_to_keep < vocab_size ?
        options->min_tokens_to_keep : vocab_size;
    return apply_top_p(
        sampler->scores, items, vocab_size, options->top_p, min_tokens_to_keep);
}

trellis_status trellis_tokenskin_sampler_process_logits(
    trellis_tokenskin_sampler * sampler,
    const float * logits,
    const uint8_t * grammar_mask,
    size_t vocab_size,
    const int32_t * generated_ids,
    size_t generated_count,
    const trellis_tokenskin_sampler_options * options,
    float * output_logits,
    size_t output_count) {
    if (output_logits == NULL || output_count != vocab_size) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const trellis_status status = process_into_workspace(
        sampler, logits, grammar_mask, vocab_size,
        generated_ids, generated_count, options);
    if (status != TRELLIS_STATUS_OK) return status;
    memcpy(output_logits, sampler->scores, vocab_size * sizeof(output_logits[0]));
    return TRELLIS_STATUS_OK;
}

static uint64_t splitmix64_next(uint64_t * state) {
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

static double random_unit(uint64_t * state) {
    return (double) (splitmix64_next(state) >> 11u) *
        (1.0 / 9007199254740992.0);
}

static trellis_status choose_greedy(
    const float * scores,
    size_t vocab_size,
    int32_t * token_out) {
    bool found = false;
    float best = -INFINITY;
    size_t best_token = 0u;
    for (size_t i = 0; i < vocab_size; ++i) {
        const float score = scores[i];
        if (isnan(score) || score == -INFINITY) continue;
        if (!found || score > best) {
            found = true;
            best = score;
            best_token = i;
        }
    }
    if (!found || best_token > INT32_MAX) return TRELLIS_STATUS_PARSE_ERROR;
    *token_out = (int32_t) best_token;
    return TRELLIS_STATUS_OK;
}

static trellis_status choose_categorical(
    trellis_tokenskin_sampler * sampler,
    size_t vocab_size,
    int32_t * token_out) {
    const float * scores = sampler->scores;
    size_t positive_infinities = 0u;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (isinf(scores[i]) && scores[i] > 0.0f) ++positive_infinities;
    }
    if (positive_infinities != 0u) {
        size_t selected = (size_t) (random_unit(&sampler->rng_state) *
            (double) positive_infinities);
        if (selected >= positive_infinities) selected = positive_infinities - 1u;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (!(isinf(scores[i]) && scores[i] > 0.0f)) continue;
            if (selected-- == 0u) {
                if (i > INT32_MAX) return TRELLIS_STATUS_PARSE_ERROR;
                *token_out = (int32_t) i;
                return TRELLIS_STATUS_OK;
            }
        }
        return TRELLIS_STATUS_PARSE_ERROR;
    }

    float maximum = -INFINITY;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (isfinite(scores[i]) && scores[i] > maximum) maximum = scores[i];
    }
    if (!isfinite(maximum)) return TRELLIS_STATUS_PARSE_ERROR;

    double sum = 0.0;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (isfinite(scores[i])) sum += (double) expf(scores[i] - maximum);
    }
    if (!(sum > 0.0) || !isfinite(sum)) return TRELLIS_STATUS_PARSE_ERROR;

    const double target = random_unit(&sampler->rng_state) * sum;
    double cumulative = 0.0;
    size_t last_candidate = SIZE_MAX;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (!isfinite(scores[i])) continue;
        cumulative += (double) expf(scores[i] - maximum);
        last_candidate = i;
        if (target < cumulative) {
            if (i > INT32_MAX) return TRELLIS_STATUS_PARSE_ERROR;
            *token_out = (int32_t) i;
            return TRELLIS_STATUS_OK;
        }
    }
    if (last_candidate == SIZE_MAX || last_candidate > INT32_MAX) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    *token_out = (int32_t) last_candidate;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_tokenskin_sampler_sample(
    trellis_tokenskin_sampler * sampler,
    const float * logits,
    const uint8_t * grammar_mask,
    size_t vocab_size,
    const int32_t * generated_ids,
    size_t generated_count,
    const trellis_tokenskin_sampler_options * options,
    int32_t * token_out) {
    if (token_out == NULL) return TRELLIS_STATUS_INVALID_ARGUMENT;
    const trellis_status status = process_into_workspace(
        sampler, logits, grammar_mask, vocab_size,
        generated_ids, generated_count, options);
    if (status != TRELLIS_STATUS_OK) return status;
    if (options->greedy || options->temperature <= 0.0f) {
        return choose_greedy(sampler->scores, vocab_size, token_out);
    }
    return choose_categorical(sampler, vocab_size, token_out);
}
