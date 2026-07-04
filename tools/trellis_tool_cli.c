#define _POSIX_C_SOURCE 200809L

#include "trellis_tool_cli.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#define TRELLIS_TOOL_ISATTY _isatty
#define TRELLIS_TOOL_FILENO _fileno
#else
#include <unistd.h>
#define TRELLIS_TOOL_ISATTY isatty
#define TRELLIS_TOOL_FILENO fileno
#endif

static int g_trellis_tool_verbose = 0;

void trellis_tool_set_verbose(int verbose) {
    g_trellis_tool_verbose = verbose != 0;
}

static const char * log_level_name(trellis_tool_log_level level) {
    switch (level) {
        case TRELLIS_TOOL_LOG_DEBUG: return "DEBUG";
        case TRELLIS_TOOL_LOG_INFO: return "INFO ";
        case TRELLIS_TOOL_LOG_WARN: return "WARN ";
        case TRELLIS_TOOL_LOG_ERROR: return "ERROR";
        default: return "?????";
    }
}

void trellis_tool_log(trellis_tool_log_level level, const char * fmt, ...) {
    if (level == TRELLIS_TOOL_LOG_DEBUG && !g_trellis_tool_verbose) {
        return;
    }
    FILE * out = level == TRELLIS_TOOL_LOG_ERROR ? stderr : stdout;
    fprintf(out, "[%s] ", log_level_name(level));
    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);
    size_t n = fmt == NULL ? 0 : strlen(fmt);
    if (n == 0 || fmt[n - 1] != '\n') {
        fputc('\n', out);
    }
    fflush(out);
}

static void build_progress_bar(char * dst, size_t dst_size, int step, int steps, char progress_char, int show_head) {
    if (dst == NULL || dst_size == 0) {
        return;
    }
    const int width = 50;
    int current = 0;
    if (steps > 0) {
        current = (int) ((double) step * (double) width / (double) steps);
        if (current < 0) current = 0;
        if (current >= width) current = width - 1;
    }

    size_t pos = 0;
    if (pos < dst_size) dst[pos++] = ' ';
    if (pos < dst_size) dst[pos++] = ' ';
    if (pos < dst_size) dst[pos++] = '|';
    for (int i = 0; i < width && pos + 1 < dst_size; ++i) {
        if (i > current) {
            dst[pos++] = ' ';
        } else if (show_head && i == current && i != width - 1) {
            dst[pos++] = '>';
        } else {
            dst[pos++] = progress_char;
        }
    }
    if (pos < dst_size) dst[pos++] = '|';
    if (pos < dst_size) {
        dst[pos] = '\0';
    } else {
        dst[dst_size - 1] = '\0';
    }
}

static double bytes_to_mib(uint64_t bytes) {
    return (double) bytes / (1024.0 * 1024.0);
}

static void print_progress_line(
    const char * label,
    int step,
    int steps,
    const char * speed_text,
    const char * detail,
    char progress_char,
    int show_head) {
    if (step <= 0 || steps <= 0) {
        return;
    }
    char bar[64];
    build_progress_bar(bar, sizeof(bar), step, steps, progress_char, show_head);
    const int done = step >= steps;
    const int interactive = TRELLIS_TOOL_ISATTY(TRELLIS_TOOL_FILENO(stdout));
    if (interactive) {
        printf("\r%s %d/%d - %s", bar, step, steps, speed_text == NULL ? "" : speed_text);
        if (label != NULL && label[0] != '\0') {
            printf(" - %s", label);
        }
        if (detail != NULL && detail[0] != '\0') {
            printf(" - %s", detail);
        }
        printf("\033[K%s", done ? "\n" : "");
    } else if (done || step == 1 || (steps >= 10 && (step % (steps / 10)) == 0)) {
        printf("%s %d/%d - %s", label == NULL ? "progress" : label, step, steps, speed_text == NULL ? "" : speed_text);
        if (detail != NULL && detail[0] != '\0') {
            printf(" - %s", detail);
        }
        printf("\n");
    }
    fflush(stdout);
}

void trellis_tool_progress_bytes(
    const char * label,
    int step,
    int steps,
    uint64_t bytes_processed,
    uint64_t bytes_total,
    int64_t elapsed_us) {
    char speed_text[96];
    double seconds = elapsed_us <= 0 ? 0.0 : (double) elapsed_us / 1000000.0;
    double mib_s = seconds <= 0.0 ? 0.0 : bytes_to_mib(bytes_processed) / seconds;
    if (mib_s >= 1024.0) {
        snprintf(speed_text, sizeof(speed_text), "%.2fGB/s", mib_s / 1024.0);
    } else {
        snprintf(speed_text, sizeof(speed_text), "%.2fMB/s", mib_s);
    }
    char detail[96];
    if (bytes_total > 0) {
        snprintf(detail, sizeof(detail), "%.1f/%.1f MiB", bytes_to_mib(bytes_processed), bytes_to_mib(bytes_total));
    } else {
        snprintf(detail, sizeof(detail), "%.1f MiB", bytes_to_mib(bytes_processed));
    }
    print_progress_line(label, step, steps, speed_text, detail, '#', 0);
}

void trellis_tool_progress_steps(
    const char * label,
    int step,
    int steps,
    int64_t step_us,
    const char * detail) {
    char speed_text[96];
    double seconds = step_us <= 0 ? 0.0 : (double) step_us / 1000000.0;
    if (seconds > 0.0 && seconds < 1.0) {
        snprintf(speed_text, sizeof(speed_text), "%.2fit/s", 1.0 / seconds);
    } else {
        snprintf(speed_text, sizeof(speed_text), "%.2fs/it", seconds);
    }
    print_progress_line(label, step, steps, speed_text, detail, '=', 1);
}
