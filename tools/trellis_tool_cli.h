#ifndef TRELLIS2_C_TOOLS_TRELLIS_TOOL_CLI_H
#define TRELLIS2_C_TOOLS_TRELLIS_TOOL_CLI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum trellis_tool_log_level {
    TRELLIS_TOOL_LOG_DEBUG = 0,
    TRELLIS_TOOL_LOG_INFO = 1,
    TRELLIS_TOOL_LOG_WARN = 2,
    TRELLIS_TOOL_LOG_ERROR = 3,
} trellis_tool_log_level;

void trellis_tool_set_verbose(int verbose);
void trellis_tool_log(trellis_tool_log_level level, const char * fmt, ...);

void trellis_tool_progress_bytes(
    const char * label,
    int step,
    int steps,
    uint64_t bytes_processed,
    uint64_t bytes_total,
    int64_t elapsed_us);

void trellis_tool_progress_steps(
    const char * label,
    int step,
    int steps,
    int64_t step_us,
    const char * detail);

#define TRELLIS_TOOL_DEBUG(...) trellis_tool_log(TRELLIS_TOOL_LOG_DEBUG, __VA_ARGS__)
#define TRELLIS_TOOL_INFO(...) trellis_tool_log(TRELLIS_TOOL_LOG_INFO, __VA_ARGS__)
#define TRELLIS_TOOL_WARN(...) trellis_tool_log(TRELLIS_TOOL_LOG_WARN, __VA_ARGS__)
#define TRELLIS_TOOL_ERROR(...) trellis_tool_log(TRELLIS_TOOL_LOG_ERROR, __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif
