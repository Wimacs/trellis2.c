#ifndef TRELLIS_PLATFORM_H
#define TRELLIS_PLATFORM_H

#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#include <windows.h>
#define TRELLIS_PATH_SEP '\\'
#define TRELLIS_PATH_LIST_SEP ';'
#define TRELLIS_EXE_SUFFIX ".exe"
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#define TRELLIS_PATH_SEP '/'
#define TRELLIS_PATH_LIST_SEP ':'
#define TRELLIS_EXE_SUFFIX ""
#ifdef __cplusplus
extern "C" {
int setenv(const char * name, const char * value, int overwrite) noexcept;
int unsetenv(const char * name) noexcept;
FILE * fdopen(int fd, const char * mode) noexcept;
ssize_t readlink(const char * path, char * buf, size_t bufsiz) noexcept;
#else
int setenv(const char * name, const char * value, int overwrite);
int unsetenv(const char * name);
FILE * fdopen(int fd, const char * mode);
ssize_t readlink(const char * path, char * buf, size_t bufsiz);
#endif
#ifdef __cplusplus
}
#endif
#endif

static inline int trellis_path_is_sep(char c) {
#ifdef _WIN32
    return c == '/' || c == '\\';
#else
    return c == '/';
#endif
}

static inline int trellis_path_has_sep(const char * path) {
    if (path == NULL) {
        return 0;
    }
    for (const char * p = path; *p != '\0'; ++p) {
        if (trellis_path_is_sep(*p)) {
            return 1;
        }
    }
    return 0;
}

static inline char * trellis_path_last_sep(char * path) {
    if (path == NULL) {
        return NULL;
    }
    char * last = NULL;
    for (char * p = path; *p != '\0'; ++p) {
        if (trellis_path_is_sep(*p)) {
            last = p;
        }
    }
    return last;
}

static inline const char * trellis_path_last_sep_const(const char * path) {
    if (path == NULL) {
        return NULL;
    }
    const char * last = NULL;
    for (const char * p = path; *p != '\0'; ++p) {
        if (trellis_path_is_sep(*p)) {
            last = p;
        }
    }
    return last;
}

static inline int trellis_unlink(const char * path) {
#ifdef _WIN32
    return _unlink(path);
#else
    return unlink(path);
#endif
}

static inline long trellis_getpid(void) {
#ifdef _WIN32
    return (long) _getpid();
#else
    return (long) getpid();
#endif
}

static inline int trellis_access_exists(const char * path) {
#ifdef _WIN32
    return path != NULL && _access(path, 0) == 0;
#else
    return path != NULL && access(path, F_OK) == 0;
#endif
}

static inline int trellis_access_read(const char * path) {
#ifdef _WIN32
    return path != NULL && _access(path, 4) == 0;
#else
    return path != NULL && access(path, R_OK) == 0;
#endif
}

static inline int trellis_access_executable(const char * path) {
#ifdef _WIN32
    return trellis_access_exists(path);
#else
    return path != NULL && access(path, X_OK) == 0;
#endif
}

static inline char * trellis_strdup(const char * s) {
    if (s == NULL) {
        return NULL;
    }
    size_t n = strlen(s) + 1u;
    char * copy = (char *) malloc(n);
    if (copy != NULL) {
        memcpy(copy, s, n);
    }
    return copy;
}

static inline int trellis_setenv(const char * name, const char * value, int overwrite) {
    if (name == NULL || name[0] == '\0' || value == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (!overwrite && getenv(name) != NULL) {
        return 0;
    }
#ifdef _WIN32
    return _putenv_s(name, value);
#else
    return setenv(name, value, overwrite);
#endif
}

static inline int trellis_unsetenv(const char * name) {
    if (name == NULL || name[0] == '\0') {
        errno = EINVAL;
        return -1;
    }
#ifdef _WIN32
    return _putenv_s(name, "");
#else
    return unsetenv(name);
#endif
}

static inline int trellis_mkdir_one(const char * path) {
    if (path == NULL || path[0] == '\0') {
        return 1;
    }
#ifdef _WIN32
    if (_mkdir(path) == 0) {
        return 1;
    }
#else
    if (mkdir(path, 0775) == 0) {
        return 1;
    }
#endif
    return errno == EEXIST;
}

static inline int trellis_path_is_root_prefix(const char * path, size_t len) {
    if (len == 0) {
        return 1;
    }
#ifdef _WIN32
    if (len == 2 && ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z')) && path[1] == ':') {
        return 1;
    }
    if (len == 3 && ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z')) &&
        path[1] == ':' && trellis_path_is_sep(path[2])) {
        return 1;
    }
#endif
    return len == 1 && trellis_path_is_sep(path[0]);
}

static inline int trellis_mkdir_p(const char * path) {
    if (path == NULL || path[0] == '\0') {
        return 1;
    }
    char tmp[PATH_MAX];
    int n = snprintf(tmp, sizeof(tmp), "%s", path);
    if (n < 0 || (size_t) n >= sizeof(tmp)) {
        errno = ENAMETOOLONG;
        return 0;
    }

    char * p = tmp;
#ifdef _WIN32
    if (((tmp[0] >= 'A' && tmp[0] <= 'Z') || (tmp[0] >= 'a' && tmp[0] <= 'z')) && tmp[1] == ':') {
        p = tmp + 2;
    }
#endif
    while (trellis_path_is_sep(*p)) {
        ++p;
    }

    for (; *p != '\0'; ++p) {
        if (!trellis_path_is_sep(*p)) {
            continue;
        }
        char saved = *p;
        *p = '\0';
        if (!trellis_path_is_root_prefix(tmp, (size_t) (p - tmp)) && !trellis_mkdir_one(tmp)) {
            *p = saved;
            return 0;
        }
        *p = saved;
    }
    if (!trellis_path_is_root_prefix(tmp, strlen(tmp)) && !trellis_mkdir_one(tmp)) {
        return 0;
    }
    return 1;
}

static inline int trellis_mkdir_parent(const char * path) {
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    char dir[PATH_MAX];
    int n = snprintf(dir, sizeof(dir), "%s", path);
    if (n < 0 || (size_t) n >= sizeof(dir)) {
        errno = ENAMETOOLONG;
        return 0;
    }
    char * slash = trellis_path_last_sep(dir);
    if (slash == NULL || slash == dir) {
        return 1;
    }
#ifdef _WIN32
    if (slash == dir + 2 && dir[1] == ':') {
        return 1;
    }
#endif
    *slash = '\0';
    return trellis_mkdir_p(dir);
}

static inline int trellis_make_temp_path(char * dst, size_t dst_size, const char * prefix, const char * suffix) {
    if (dst == NULL || dst_size == 0 || prefix == NULL) {
        errno = EINVAL;
        return 0;
    }
    if (suffix == NULL) {
        suffix = "";
    }
#ifdef _WIN32
    char dir_buf[MAX_PATH + 1];
    DWORD got = GetTempPathA((DWORD) sizeof(dir_buf), dir_buf);
    const char * dir = got > 0 && got < sizeof(dir_buf) ? dir_buf : ".";
#else
    const char * dir = getenv("TMPDIR");
    if (dir == NULL || dir[0] == '\0') {
        dir = "/tmp";
    }
#endif
    size_t dir_len = strlen(dir);
    int needs_sep = dir_len > 0 && !trellis_path_is_sep(dir[dir_len - 1]);
    static unsigned long counter = 0;
    unsigned long id = ++counter;
    int n = snprintf(
        dst,
        dst_size,
        "%s%s%s_%ld_%lu%s",
        dir,
        needs_sep ? (TRELLIS_PATH_SEP == '\\' ? "\\" : "/") : "",
        prefix,
        trellis_getpid(),
        id,
        suffix);
    return n >= 0 && (size_t) n < dst_size;
}

static inline FILE * trellis_open_temp_file(char * path_out, size_t path_out_size, const char * prefix, const char * suffix) {
    if (path_out == NULL || path_out_size == 0) {
        errno = EINVAL;
        return NULL;
    }
    for (int attempt = 0; attempt < 100; ++attempt) {
        if (!trellis_make_temp_path(path_out, path_out_size, prefix, suffix)) {
            return NULL;
        }
#ifdef _WIN32
        int fd = _open(
            path_out,
            _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY,
            _S_IREAD | _S_IWRITE);
        if (fd >= 0) {
            return _fdopen(fd, "wb");
        }
#else
        int fd = open(path_out, O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd >= 0) {
            return fdopen(fd, "wb");
        }
#endif
        if (errno != EEXIST) {
            return NULL;
        }
    }
    errno = EEXIST;
    return NULL;
}

static inline int trellis_current_executable_path(char * dst, size_t dst_size) {
    if (dst == NULL || dst_size == 0) {
        return 0;
    }
#ifdef _WIN32
    DWORD n = GetModuleFileNameA(NULL, dst, (DWORD) dst_size);
    if (n == 0 || n >= dst_size) {
        return 0;
    }
    dst[n] = '\0';
    return 1;
#elif defined(__linux__)
    ssize_t n = readlink("/proc/self/exe", dst, dst_size - 1u);
    if (n <= 0 || (size_t) n >= dst_size) {
        return 0;
    }
    dst[n] = '\0';
    return 1;
#else
    (void) dst;
    (void) dst_size;
    return 0;
#endif
}

static inline int trellis_command_available(const char * name) {
    if (name == NULL || name[0] == '\0') {
        return 0;
    }
    if (trellis_path_has_sep(name)) {
        return trellis_access_executable(name);
    }
#ifdef _WIN32
    char found[MAX_PATH];
    const char * ext = strrchr(name, '.') == NULL ? ".exe" : NULL;
    return SearchPathA(NULL, name, ext, (DWORD) sizeof(found), found, NULL) > 0;
#else
    const char * path = getenv("PATH");
    if (path == NULL || path[0] == '\0') {
        path = "/bin:/usr/bin";
    }
    const char * p = path;
    while (*p != '\0') {
        const char * end = strchr(p, TRELLIS_PATH_LIST_SEP);
        size_t dir_len = end != NULL ? (size_t) (end - p) : strlen(p);
        char candidate[PATH_MAX];
        if (dir_len == 0) {
            int n = snprintf(candidate, sizeof(candidate), "%s", name);
            if (n >= 0 && (size_t) n < sizeof(candidate) && trellis_access_executable(candidate)) {
                return 1;
            }
        } else if (dir_len + 1u + strlen(name) < sizeof(candidate)) {
            memcpy(candidate, p, dir_len);
            candidate[dir_len] = '/';
            strcpy(candidate + dir_len + 1u, name);
            if (trellis_access_executable(candidate)) {
                return 1;
            }
        }
        if (end == NULL) {
            break;
        }
        p = end + 1;
    }
    return 0;
#endif
}

static inline int trellis_run_process_search_path(char * const argv[]) {
    if (argv == NULL || argv[0] == NULL) {
        errno = EINVAL;
        return 0;
    }
#ifdef _WIN32
    intptr_t rc = _spawnvp(_P_WAIT, argv[0], (const char * const *) argv);
    return rc == 0;
#else
    pid_t pid = fork();
    if (pid < 0) {
        return 0;
    }
    if (pid == 0) {
        execvp(argv[0], argv);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return 0;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

static inline int trellis_run_process_exact(char * const argv[]) {
    if (argv == NULL || argv[0] == NULL) {
        errno = EINVAL;
        return 0;
    }
#ifdef _WIN32
    intptr_t rc = _spawnv(_P_WAIT, argv[0], (const char * const *) argv);
    return rc == 0;
#else
    pid_t pid = fork();
    if (pid < 0) {
        return 0;
    }
    if (pid == 0) {
        execv(argv[0], argv);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return 0;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

static inline int64_t trellis_now_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    if (QueryPerformanceFrequency(&freq) && QueryPerformanceCounter(&counter) && freq.QuadPart > 0) {
        return (int64_t) ((counter.QuadPart * 1000000ll) / freq.QuadPart);
    }
    return (int64_t) GetTickCount64() * 1000ll;
#elif defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (int64_t) ts.tv_sec * 1000000ll + (int64_t) (ts.tv_nsec / 1000ll);
    }
    return (int64_t) time(NULL) * 1000000ll;
#else
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) == TIME_UTC) {
        return (int64_t) ts.tv_sec * 1000000ll + (int64_t) (ts.tv_nsec / 1000ll);
    }
    return (int64_t) time(NULL) * 1000000ll;
#endif
}

#endif
