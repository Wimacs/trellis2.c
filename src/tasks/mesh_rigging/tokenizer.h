#ifndef TRELLIS2_C_MESH_RIGGING_TOKENIZER_H
#define TRELLIS2_C_MESH_RIGGING_TOKENIZER_H

#include "trellis.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed vocabulary from the released articulation_xl_quantization_256_token_4
 * checkpoint.  These values are model ABI, not configurable defaults. */
enum {
    TRELLIS_TOKENSKIN_NUM_DISCRETE = 256,
    TRELLIS_TOKENSKIN_TOKEN_BRANCH = 256,
    TRELLIS_TOKENSKIN_TOKEN_BOS = 257,
    TRELLIS_TOKENSKIN_TOKEN_SKELETON_EOS = 258,
    TRELLIS_TOKENSKIN_TOKEN_PAD = 259,
    TRELLIS_TOKENSKIN_TOKEN_SPRING = 260,
    TRELLIS_TOKENSKIN_TOKEN_BODY = 261,
    TRELLIS_TOKENSKIN_TOKEN_HAND = 262,
    TRELLIS_TOKENSKIN_TOKEN_CLASS_NONE = 263,
    TRELLIS_TOKENSKIN_TOKEN_CLASS_RIGNET = 264,
    TRELLIS_TOKENSKIN_TOKEN_CLASS_VROID = 265,
    TRELLIS_TOKENSKIN_TOKEN_CLASS_ARTICULATION = 266,
    TRELLIS_TOKENSKIN_SKELETON_VOCAB_SIZE = 267,
    TRELLIS_TOKENSKIN_FSQ_CODEBOOK_SIZE = 32768,
    TRELLIS_TOKENSKIN_TOKEN_GLOBAL_EOS = 33035,
    TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE = 33036,
    TRELLIS_TOKENSKIN_TOKENS_PER_SKIN = 4,
    TRELLIS_TOKENSKIN_FSQ_CODE_DIM = 5,
};

typedef enum trellis_tokenskin_grammar_state {
    TRELLIS_TOKENSKIN_GRAMMAR_INVALID = 0,
    TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_BOS,
    TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_CLASS_PART_OR_JOINT,
    TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_PART_OR_JOINT,
    TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT_2,
    TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT_3,
    TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_BRANCH_PART_OR_JOINT,
    TRELLIS_TOKENSKIN_GRAMMAR_EXPECT_JOINT,
    TRELLIS_TOKENSKIN_GRAMMAR_SKELETON_COMPLETE,
    TRELLIS_TOKENSKIN_GRAMMAR_SEQUENCE_COMPLETE,
} trellis_tokenskin_grammar_state;

typedef enum trellis_tokenskin_eos_mode {
    /* Reproduces VocabSwitchingLogitsProcessor exactly: global EOS is emitted
     * after 4*J-1 real skin tokens, then decoded as FSQ index 32768. */
    TRELLIS_TOKENSKIN_EOS_OFFICIAL_COMPAT = 0,
    /* Emits all 4*J real skin tokens before global EOS. */
    TRELLIS_TOKENSKIN_EOS_CORRECTED = 1,
} trellis_tokenskin_eos_mode;

typedef struct trellis_tokenskin_mask_info {
    trellis_tokenskin_grammar_state state;
    size_t bone_count;
    size_t skeleton_eos_index;
    size_t skin_token_count;
    bool has_skeleton_eos;
    bool has_global_eos;
    bool next_is_global_eos;
} trellis_tokenskin_mask_info;

typedef struct trellis_tokenskin_skeleton {
    /* Joint heads in normalized model space, shape (joint_count, 3). */
    float * joints_xyz;
    /* Resolved parent joint heads, shape (joint_count, 3).  The root repeats
     * its own position, matching TokenizerPart.make_skeleton. */
    float * parent_joints_xyz;
    /* Parent indices, shape (joint_count); root is -1. */
    int32_t * parents;
    size_t joint_count;
    /* One of CLASS_NONE/RIGNET/VROID/ARTICULATION. */
    int32_t class_token_id;
} trellis_tokenskin_skeleton;

/* Builds a 0/1 validity mask for the next autoregressive token.  ids is the
 * full token sequence including manually prepended start tokens, as seen by
 * the Python logits processor after concatenating self.init and input_ids.
 * mask_count must be at least TRELLIS_TOKENSKIN_MODEL_VOCAB_SIZE. */
trellis_status trellis_tokenskin_tokenizer_next_mask(
    const int32_t * ids,
    size_t count,
    trellis_tokenskin_eos_mode eos_mode,
    uint8_t * mask,
    size_t mask_count,
    trellis_tokenskin_mask_info * info_out);

/* Counts completed bones using TokenizerPart.bones_in_sequence semantics.
 * Parsing stops at the first skeleton EOS.  Prefixes without EOS are valid. */
trellis_status trellis_tokenskin_tokenizer_bones_in_sequence(
    const int32_t * ids,
    size_t count,
    size_t * bones_out,
    bool * has_skeleton_eos_out,
    size_t * skeleton_eos_index_out,
    trellis_tokenskin_grammar_state * state_out);

/* Decodes the skeleton up to the first skeleton EOS.  Branch parent positions
 * are resolved to the nearest preceding joint, scanning preceding joints in
 * reverse order exactly like TokenizerPart.make_skeleton. */
trellis_status trellis_tokenskin_tokenizer_decode_skeleton(
    const int32_t * ids,
    size_t count,
    trellis_tokenskin_skeleton * skeleton_out);

void trellis_tokenskin_skeleton_free(trellis_tokenskin_skeleton * skeleton);

/* Extracts the 4*J FSQ indices expected by the decoder from a completed
 * sequence.  In OFFICIAL_COMPAT mode the final global EOS is deliberately
 * included and becomes raw FSQ index 32768. */
trellis_status trellis_tokenskin_tokenizer_extract_skin_indices(
    const int32_t * ids,
    size_t count,
    trellis_tokenskin_eos_mode eos_mode,
    int32_t * indices_out,
    size_t indices_capacity,
    size_t * indices_count_out);

/* Converts an FSQ index into the five normalized level values used before
 * project_out.  OFFICIAL_COMPAT accepts index 32768, whose base-8 digits wrap
 * to the same all-zero digits as index 0. */
trellis_status trellis_tokenskin_tokenizer_fsq_code(
    int32_t fsq_index,
    trellis_tokenskin_eos_mode eos_mode,
    float code_out[TRELLIS_TOKENSKIN_FSQ_CODE_DIM]);

float trellis_tokenskin_tokenizer_undiscretize(int32_t token);

#ifdef __cplusplus
}
#endif

#endif
