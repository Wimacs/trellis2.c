#ifndef TRELLIS_FILE_SEEK_H
#define TRELLIS_FILE_SEEK_H

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#ifdef _WIN32
#include <io.h>
#endif

static inline int trellis_file_seek_set_u64(FILE * f, uint64_t offset) {
    if (f == NULL || offset > (uint64_t) INT64_MAX) {
        errno = EOVERFLOW;
        return -1;
    }
#ifdef _WIN32
    return _fseeki64(f, (__int64) offset, SEEK_SET);
#else
    if (offset > (uint64_t) LONG_MAX) {
        errno = EOVERFLOW;
        return -1;
    }
    return fseek(f, (long) offset, SEEK_SET);
#endif
}

static inline int trellis_file_seek_set_sum_u64(FILE * f, uint64_t base, uint64_t offset) {
    if (offset > UINT64_MAX - base) {
        errno = EOVERFLOW;
        return -1;
    }
    return trellis_file_seek_set_u64(f, base + offset);
}

#endif
