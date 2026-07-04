#ifndef TRELLIS2_C_TOOLS_DEBUG_TRELLIS_CHECKPOINT_VALIDATE_H
#define TRELLIS2_C_TOOLS_DEBUG_TRELLIS_CHECKPOINT_VALIDATE_H

#include "trellis.h"

typedef struct trellis_checkpoint_report {
    size_t expected_tensors;
    size_t actual_tensors;
    size_t found_tensors;
    size_t missing_tensors;
    size_t shape_mismatches;
    size_t dtype_mismatches;
    size_t extra_tensors;
    uint64_t expected_elements;
    uint64_t expected_bytes;
    char first_issue[256];
} trellis_checkpoint_report;

void trellis_checkpoint_report_clear(trellis_checkpoint_report * report);

trellis_status trellis_ss_flow_validate_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report);

trellis_status trellis_shape_slat_flow_validate_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report);

trellis_status trellis_ss_decoder_validate_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report);

trellis_status trellis_shape_decoder_validate_checkpoint(
    const char * safetensors_path,
    trellis_checkpoint_report * report);

#endif
