#ifndef TRELLIS2_C_REGISTRY_H
#define TRELLIS2_C_REGISTRY_H

#include "trellis.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRELLIS_REGISTRY_ABI_VERSION 1u

struct trellis_model_package;

typedef struct trellis_model_family_descriptor {
    uint32_t abi_version;
    size_t struct_size;
    const char * id;
    const char * default_task;
    const char * default_profile;
} trellis_model_family_descriptor;

typedef struct trellis_architecture_descriptor {
    uint32_t abi_version;
    size_t struct_size;
    const char * id;
    const char * family; /* NULL means shared by multiple families. */
} trellis_architecture_descriptor;

enum {
    TRELLIS_TASK_FLAG_TEST_FIXTURE = 1u << 0,
};

typedef struct trellis_task_descriptor {
    uint32_t abi_version;
    size_t struct_size;
    const char * id;
    const char * input_kind;
    const char * output_kind;
    uint32_t flags;
} trellis_task_descriptor;

size_t trellis_registry_family_count(void);
const trellis_model_family_descriptor * trellis_registry_family_at(size_t index);
const trellis_model_family_descriptor * trellis_registry_find_family(const char * id);

size_t trellis_registry_architecture_count(void);
const trellis_architecture_descriptor * trellis_registry_architecture_at(size_t index);
const trellis_architecture_descriptor * trellis_registry_find_architecture(const char * id);

size_t trellis_registry_task_count(void);
const trellis_task_descriptor * trellis_registry_task_at(size_t index);
const trellis_task_descriptor * trellis_registry_find_task(const char * id);

trellis_status trellis_registry_validate_package(const struct trellis_model_package * package);

#ifdef __cplusplus
}
#endif

#endif
