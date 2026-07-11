#include "sampler.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++g_failures; \
        goto cleanup; \
    } \
} while (0)

static void fill_mask(uint8_t * mask, size_t count) {
    memset(mask, 1, count * sizeof(mask[0]));
}

static int nearly_equal(float actual, float expected) {
    return fabsf(actual - expected) <= 2e-6f;
}

static void test_hf_processing_golden(void) {
    /* Generated with Transformers 4.56.1 processors in this exact order:
     * RepetitionPenalty(1.5), mask token 6, Temperature(.8), TopK(4),
     * TopP(.75). */
    const float logits[] = {2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 1.0f, 4.0f, 0.0f};
    const int32_t generated[] = {0, 3, 3, 6};
    uint8_t mask[8];
    float output[8];
    trellis_tokenskin_sampler sampler;
    memset(&sampler, 0, sizeof(sampler));
    fill_mask(mask, 8u);
    mask[6] = 0u;
    trellis_tokenskin_sampler_options options =
        TRELLIS_TOKENSKIN_SAMPLER_OPTIONS_INIT;
    options.temperature = 0.8f;
    options.repetition_penalty = 1.5f;
    options.top_k = 4u;
    options.top_p = 0.75f;

    CHECK_TRUE(trellis_tokenskin_sampler_init(&sampler, 8u, 123u) == TRELLIS_STATUS_OK);
    CHECK_TRUE(trellis_tokenskin_sampler_process_logits(
        &sampler, logits, mask, 8u,
        generated, sizeof(generated) / sizeof(generated[0]),
        &options, output, 8u) == TRELLIS_STATUS_OK);
    CHECK_TRUE(nearly_equal(output[0], 1.6666667461395264f));
    CHECK_TRUE(nearly_equal(output[3], 2.5f));
    for (size_t i = 0; i < 8u; ++i) {
        if (i == 0u || i == 3u) continue;
        CHECK_TRUE(isinf(output[i]) && output[i] < 0.0f);
    }

cleanup:
    trellis_tokenskin_sampler_free(&sampler);
}

static void test_repetition_sign_and_unique_ids(void) {
    const float logits[] = {2.0f, -2.0f, 1.0f, 0.0f};
    const int32_t generated[] = {0, 1, 0, 1};
    const uint8_t mask[] = {1, 1, 1, 1};
    float output[4];
    trellis_tokenskin_sampler sampler;
    memset(&sampler, 0, sizeof(sampler));
    trellis_tokenskin_sampler_options options =
        TRELLIS_TOKENSKIN_SAMPLER_OPTIONS_INIT;
    options.temperature = 2.0f;
    options.repetition_penalty = 2.0f;

    CHECK_TRUE(trellis_tokenskin_sampler_init(&sampler, 4u, 0u) == TRELLIS_STATUS_OK);
    CHECK_TRUE(trellis_tokenskin_sampler_process_logits(
        &sampler, logits, mask, 4u,
        generated, sizeof(generated) / sizeof(generated[0]),
        &options, output, 4u) == TRELLIS_STATUS_OK);
    CHECK_TRUE(nearly_equal(output[0], 0.5f));
    CHECK_TRUE(nearly_equal(output[1], -2.0f));
    CHECK_TRUE(nearly_equal(output[2], 0.5f));
    CHECK_TRUE(nearly_equal(output[3], 0.0f));

cleanup:
    trellis_tokenskin_sampler_free(&sampler);
}

static void test_top_k_ties_and_top_p(void) {
    const uint8_t mask[] = {1, 1, 1, 1};
    const float top_k_logits[] = {3.0f, 2.0f, 2.0f, 1.0f};
    const float top_p_logits[] = {
        -2.995732273553991f, /* log(.05) */
        -2.302585092994046f, /* log(.10) */
        -1.609437912434100f, /* log(.20) */
        -0.430782916092454f, /* log(.65) */
    };
    float output[4];
    trellis_tokenskin_sampler sampler;
    memset(&sampler, 0, sizeof(sampler));
    trellis_tokenskin_sampler_options options =
        TRELLIS_TOKENSKIN_SAMPLER_OPTIONS_INIT;

    CHECK_TRUE(trellis_tokenskin_sampler_init(&sampler, 4u, 9u) == TRELLIS_STATUS_OK);
    options.top_k = 2u;
    CHECK_TRUE(trellis_tokenskin_sampler_process_logits(
        &sampler, top_k_logits, mask, 4u, NULL, 0u,
        &options, output, 4u) == TRELLIS_STATUS_OK);
    CHECK_TRUE(nearly_equal(output[0], 3.0f));
    CHECK_TRUE(nearly_equal(output[1], 2.0f));
    CHECK_TRUE(nearly_equal(output[2], 2.0f));
    CHECK_TRUE(isinf(output[3]) && output[3] < 0.0f);

    options.top_k = 0u;
    options.top_p = 0.8f;
    CHECK_TRUE(trellis_tokenskin_sampler_process_logits(
        &sampler, top_p_logits, mask, 4u, NULL, 0u,
        &options, output, 4u) == TRELLIS_STATUS_OK);
    CHECK_TRUE(isinf(output[0]) && output[0] < 0.0f);
    CHECK_TRUE(isinf(output[1]) && output[1] < 0.0f);
    CHECK_TRUE(nearly_equal(output[2], top_p_logits[2]));
    CHECK_TRUE(nearly_equal(output[3], top_p_logits[3]));

cleanup:
    trellis_tokenskin_sampler_free(&sampler);
}

static void test_beam_min_tokens_to_keep(void) {
    const uint8_t mask[] = {1, 1, 1, 1};
    const float logits[] = {10.0f, 3.0f, 2.0f, 1.0f};
    float output[4];
    trellis_tokenskin_sampler sampler;
    memset(&sampler, 0, sizeof(sampler));
    trellis_tokenskin_sampler_options options =
        TRELLIS_TOKENSKIN_SAMPLER_OPTIONS_INIT;
    options.min_tokens_to_keep = 2u;
    CHECK_TRUE(trellis_tokenskin_sampler_init(&sampler, 4u, 9u) ==
        TRELLIS_STATUS_OK);

    options.top_k = 1u;
    CHECK_TRUE(trellis_tokenskin_sampler_process_logits(
        &sampler, logits, mask, 4u, NULL, 0u,
        &options, output, 4u) == TRELLIS_STATUS_OK);
    CHECK_TRUE(nearly_equal(output[0], logits[0]));
    CHECK_TRUE(nearly_equal(output[1], logits[1]));
    CHECK_TRUE(isinf(output[2]) && output[2] < 0.0f);
    CHECK_TRUE(isinf(output[3]) && output[3] < 0.0f);

    options.top_k = 0u;
    options.top_p = 0.01f;
    CHECK_TRUE(trellis_tokenskin_sampler_process_logits(
        &sampler, logits, mask, 4u, NULL, 0u,
        &options, output, 4u) == TRELLIS_STATUS_OK);
    CHECK_TRUE(nearly_equal(output[0], logits[0]));
    CHECK_TRUE(nearly_equal(output[1], logits[1]));
    CHECK_TRUE(isinf(output[2]) && output[2] < 0.0f);
    CHECK_TRUE(isinf(output[3]) && output[3] < 0.0f);

cleanup:
    trellis_tokenskin_sampler_free(&sampler);
}

static void test_grammar_forces_global_eos(void) {
    const size_t vocab = TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE;
    float * logits = (float *) malloc(vocab * sizeof(float));
    uint8_t * mask = (uint8_t *) calloc(vocab, sizeof(uint8_t));
    trellis_tokenskin_sampler sampler;
    memset(&sampler, 0, sizeof(sampler));
    trellis_tokenskin_sampler_options options =
        TRELLIS_TOKENSKIN_SAMPLER_OPTIONS_INIT;
    int32_t token = -1;
    CHECK_TRUE(logits != NULL && mask != NULL);
    for (size_t i = 0; i < vocab; ++i) logits[i] = (float) (i % 101u);
    mask[TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS] = 1u;

    CHECK_TRUE(trellis_tokenskin_sampler_init(&sampler, vocab, UINT64_C(42)) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(trellis_tokenskin_sampler_sample(
        &sampler, logits, mask, vocab, NULL, 0u, &options, &token) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(token == TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS);

cleanup:
    trellis_tokenskin_sampler_free(&sampler);
    free(logits);
    free(mask);
}

static void test_deterministic_rng_and_reseed(void) {
    const float logits[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    const uint8_t mask[] = {1, 1, 1, 1, 1};
    trellis_tokenskin_sampler a;
    trellis_tokenskin_sampler b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    trellis_tokenskin_sampler_options options =
        TRELLIS_TOKENSKIN_SAMPLER_OPTIONS_INIT;
    int32_t first_pass[16];
    CHECK_TRUE(trellis_tokenskin_sampler_init(&a, 5u, UINT64_C(1234567)) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(trellis_tokenskin_sampler_init(&b, 5u, UINT64_C(1234567)) ==
        TRELLIS_STATUS_OK);
    for (size_t i = 0; i < 16u; ++i) {
        int32_t x = -1;
        int32_t y = -1;
        CHECK_TRUE(trellis_tokenskin_sampler_sample(
            &a, logits, mask, 5u, NULL, 0u, &options, &x) == TRELLIS_STATUS_OK);
        CHECK_TRUE(trellis_tokenskin_sampler_sample(
            &b, logits, mask, 5u, NULL, 0u, &options, &y) == TRELLIS_STATUS_OK);
        CHECK_TRUE(x == y);
        first_pass[i] = x;
    }
    trellis_tokenskin_sampler_seed(&a, UINT64_C(1234567));
    for (size_t i = 0; i < 16u; ++i) {
        int32_t x = -1;
        CHECK_TRUE(trellis_tokenskin_sampler_sample(
            &a, logits, mask, 5u, NULL, 0u, &options, &x) == TRELLIS_STATUS_OK);
        CHECK_TRUE(x == first_pass[i]);
    }

cleanup:
    trellis_tokenskin_sampler_free(&a);
    trellis_tokenskin_sampler_free(&b);
}

static void test_greedy_and_invalid_logits(void) {
    const float logits[] = {NAN, -INFINITY, 2.0f, 1.0f};
    const float infinite_logits[] = {1.0f, INFINITY, -INFINITY, INFINITY};
    const uint8_t mask[] = {1, 1, 1, 1};
    const uint8_t empty_mask[] = {0, 0, 0, 0};
    trellis_tokenskin_sampler sampler;
    memset(&sampler, 0, sizeof(sampler));
    trellis_tokenskin_sampler_options options =
        TRELLIS_TOKENSKIN_SAMPLER_OPTIONS_INIT;
    int32_t token = -1;
    CHECK_TRUE(trellis_tokenskin_sampler_init(&sampler, 4u, 7u) == TRELLIS_STATUS_OK);

    options.temperature = 0.0f;
    CHECK_TRUE(trellis_tokenskin_sampler_sample(
        &sampler, logits, mask, 4u, NULL, 0u, &options, &token) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(token == 2);

    options.temperature = 1.0f;
    options.greedy = true;
    token = -1;
    CHECK_TRUE(trellis_tokenskin_sampler_sample(
        &sampler, logits, mask, 4u, NULL, 0u, &options, &token) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(token == 2);

    options.greedy = false;
    CHECK_TRUE(trellis_tokenskin_sampler_sample(
        &sampler, logits, empty_mask, 4u, NULL, 0u, &options, &token) ==
        TRELLIS_STATUS_PARSE_ERROR);

    /* Positive infinities form a well-defined uniform categorical subset
     * instead of producing NaN through inf-inf during softmax. */
    trellis_tokenskin_sampler_seed(&sampler, 7u);
    CHECK_TRUE(trellis_tokenskin_sampler_sample(
        &sampler, infinite_logits, mask, 4u, NULL, 0u, &options, &token) ==
        TRELLIS_STATUS_OK);
    CHECK_TRUE(token == 1 || token == 3);

cleanup:
    trellis_tokenskin_sampler_free(&sampler);
}

int main(void) {
    test_hf_processing_golden();
    test_repetition_sign_and_unique_ids();
    test_top_k_ties_and_top_p();
    test_beam_min_tokens_to_keep();
    test_grammar_forces_global_eos();
    test_deterministic_rng_and_reseed();
    test_greedy_and_invalid_logits();
    if (g_failures != 0) {
        fprintf(stderr, "%d TokenSkin sampler test(s) failed\n", g_failures);
        return 1;
    }
    puts("TokenSkin sampler tests passed");
    return 0;
}
