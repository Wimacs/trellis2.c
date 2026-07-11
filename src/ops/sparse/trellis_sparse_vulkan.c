#include "trellis_sparse_backend.h"

#ifdef TRELLIS_HAS_SPARSE_VULKAN
#include "trellis_sparse_vk_add_spv.h"
#include "trellis_sparse_vk_c2s_count_spv.h"
#include "trellis_sparse_vk_c2s_fill_spv.h"
#include "trellis_sparse_vk_c2s_gather_spv.h"
#include "trellis_sparse_vk_fill_bias_spv.h"
#include "trellis_sparse_vk_linear_coop_spv.h"
#include "trellis_sparse_vk_linear_mat_spv.h"
#include "trellis_sparse_vk_linear_silu_coop_spv.h"
#include "trellis_sparse_vk_linear_silu_mat_spv.h"
#include "trellis_sparse_vk_pixal_naf_attention_project_spv.h"
#include "trellis_sparse_vk_row_norm_silu_spv.h"
#include "trellis_sparse_vk_row_norm_spv.h"
#include "trellis_sparse_vk_rulebook_dispatch_spv.h"
#include "trellis_sparse_vk_rulebook_fill_spv.h"
#include "trellis_sparse_vk_rulebook_hash_insert_spv.h"
#include "trellis_sparse_vk_rulebook_mask_init_spv.h"
#include "trellis_sparse_vk_rulebook_tile_valid_spv.h"
#include "trellis_sparse_vk_rulebook_valid_sorted_count_spv.h"
#include "trellis_sparse_vk_rulebook_valid_sorted_scatter_spv.h"
#include "trellis_sparse_vk_scan_i32_add_block_offsets_spv.h"
#include "trellis_sparse_vk_scan_i32_block_spv.h"
#include "trellis_sparse_vk_scan_i32_stride_spv.h"
#include "trellis_sparse_vk_silu_spv.h"
#include "trellis_sparse_vk_skip_repeat_spv.h"
#include "trellis_sparse_vk_sort_bitonic_spv.h"
#include "trellis_sparse_vk_sparse_conv_masked_mat_spv.h"
#include "trellis_sparse_vk_sparse_conv_masked_sorted_mat_spv.h"
#include "trellis_sparse_vk_sparse_conv_offset_coop_spv.h"
#include "trellis_sparse_vk_sparse_conv_offset_mat_spv.h"

#include <vulkan/vulkan.h>

#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef enum sparse_vk_pipeline_id {
    SPARSE_VK_PIPE_ROW_NORM,
    SPARSE_VK_PIPE_ROW_NORM_SILU,
    SPARSE_VK_PIPE_SILU,
    SPARSE_VK_PIPE_ADD,
    SPARSE_VK_PIPE_C2S_GATHER,
    SPARSE_VK_PIPE_SKIP_REPEAT,
    SPARSE_VK_PIPE_FILL_BIAS,
    SPARSE_VK_PIPE_RULEBOOK_HASH_INSERT,
    SPARSE_VK_PIPE_RULEBOOK_FILL,
    SPARSE_VK_PIPE_C2S_COUNT,
    SPARSE_VK_PIPE_C2S_FILL,
    SPARSE_VK_PIPE_SCAN_I32_STRIDE,
    SPARSE_VK_PIPE_SCAN_I32_BLOCK,
    SPARSE_VK_PIPE_SCAN_I32_ADD_BLOCK_OFFSETS,
    SPARSE_VK_PIPE_RULEBOOK_DISPATCH,
    SPARSE_VK_PIPE_RULEBOOK_TILE_VALID,
    SPARSE_VK_PIPE_RULEBOOK_MASK_INIT,
    SPARSE_VK_PIPE_SORT_BITONIC,
    SPARSE_VK_PIPE_RULEBOOK_VALID_SORTED_COUNT,
    SPARSE_VK_PIPE_RULEBOOK_VALID_SORTED_SCATTER,
    SPARSE_VK_PIPE_LINEAR_MAT,
    SPARSE_VK_PIPE_LINEAR_SILU_MAT,
    SPARSE_VK_PIPE_SPARSE_CONV_OFFSET_MAT,
    SPARSE_VK_PIPE_SPARSE_CONV_MASKED_MAT,
    SPARSE_VK_PIPE_SPARSE_CONV_MASKED_SORTED_MAT,
    SPARSE_VK_PIPE_LINEAR_COOP,
    SPARSE_VK_PIPE_LINEAR_SILU_COOP,
    SPARSE_VK_PIPE_SPARSE_CONV_OFFSET_COOP,
    SPARSE_VK_PIPE_PIXAL_NAF_ATTENTION_PROJECT,
    SPARSE_VK_PIPE_COUNT,
} sparse_vk_pipeline_id;

enum {
    VK_BINDING_COUNT = 10,
};

enum {
    VK_SCAN_BLOCK_SIZE = 128,
};

typedef struct sparse_vk_push {
    uint32_t reserved;
    uint32_t n;
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t total;
    uint32_t table_mask;
    uint32_t flags;
    float eps;
} sparse_vk_push;

struct trellis_sparse_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void * mapped;
    size_t bytes;
    int device_local;
    struct trellis_sparse_buffer * next_free;
};

struct trellis_sparse_rulebook {
    int32_t * coords_host;
    int64_t n;
    uint32_t table_mask;
    int owns_coords;
    int64_t offset_starts[27];
    int64_t offset_counts[27];
    int64_t total_pairs;
    trellis_sparse_buffer * coords;
    trellis_sparse_buffer * hash_keys;
    trellis_sparse_buffer * hash_values;
    trellis_sparse_buffer * src_rows;
    trellis_sparse_buffer * dst_rows;
    trellis_sparse_buffer * neighbors;
    trellis_sparse_buffer * tile_valid_counts;
    trellis_sparse_buffer * tile_valid_offsets;
    trellis_sparse_buffer * sort_keys;
    trellis_sparse_buffer * sorted_idx;
    trellis_sparse_buffer * gray_code;
    trellis_sparse_buffer * reduced_code;
    trellis_sparse_buffer * valid_kernel;
    trellis_sparse_buffer * valid_kernel_seg;
    trellis_sparse_buffer * valid_kernel_seg_tmp;
};

typedef struct sparse_vk_weight {
    const float * ptr;
    size_t count;
    trellis_sparse_buffer * buffer;
    struct sparse_vk_weight * next;
} sparse_vk_weight;

typedef struct sparse_vk_c2s_cache_entry sparse_vk_c2s_cache_entry;

struct trellis_sparse_c2s_device_map {
    sparse_vk_c2s_cache_entry * entry;
};

struct sparse_vk_c2s_cache_entry {
    const int32_t * coords_ptr;
    const int32_t * parent_ptr;
    const int32_t * subidx_ptr;
    int64_t n;
    trellis_sparse_buffer * coords;
    trellis_sparse_buffer * parent;
    trellis_sparse_buffer * subidx;
    trellis_sparse_c2s_device_map * device_map;
    int owns_buffers;
    struct sparse_vk_c2s_cache_entry * next;
};

typedef struct sparse_vk_descriptor_node {
    VkDescriptorSet set;
    struct sparse_vk_descriptor_node * next;
} sparse_vk_descriptor_node;

typedef struct sparse_vk_command_node {
    VkCommandBuffer command;
    struct sparse_vk_command_node * next;
} sparse_vk_command_node;

typedef struct trellis_sparse_vk_backend {
    trellis_sparse_backend base;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    uint32_t max_workgroup_count[3];
    VkCommandPool command_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipelines[SPARSE_VK_PIPE_COUNT];
    VkDescriptorPool descriptor_pool;
    trellis_sparse_buffer * dummy;
    sparse_vk_weight * weights;
    sparse_vk_c2s_cache_entry * c2s_caches;
    trellis_sparse_buffer * free_buffers;
    sparse_vk_descriptor_node * free_descriptors;
    sparse_vk_command_node * free_commands;
    sparse_vk_descriptor_node * pending_descriptors;
    sparse_vk_command_node * pending_commands;
    int use_sparse_conv_indirect;
    int coopmat_supported;
    int destroying;
} trellis_sparse_vk_backend;

static void vk_reclaim_pending(trellis_sparse_vk_backend * vk);

static uint32_t vk_hash_u64(uint32_t lo, uint32_t hi) {
    uint32_t x = lo ^ (hi * 0x9e3779b9u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static void vk_pack_coord4(int b, int x, int y, int z, uint32_t * lo, uint32_t * hi) {
    *hi = (((uint32_t) b & 0xffffu) << 16) | ((uint32_t) x & 0xffffu);
    *lo = (((uint32_t) y & 0xffffu) << 16) | ((uint32_t) z & 0xffffu);
    *lo += 1u;
    if (*lo == 0u) {
        *hi += 1u;
    }
}

static int64_t vk_next_power_of_two(int64_t x) {
    int64_t v = 1;
    while (v < x && v < INT64_MAX / 2) {
        v <<= 1;
    }
    return v;
}

static int vk_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static int vk_env_disabled(const char * name) {
    const char * value = getenv(name);
    return value != NULL && strcmp(value, "0") == 0;
}

static int vk_physical_device_has_extension(VkPhysicalDevice physical_device, const char * name) {
    if (physical_device == VK_NULL_HANDLE || name == NULL) {
        return 0;
    }
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, NULL, &count, NULL) != VK_SUCCESS || count == 0) {
        return 0;
    }
    VkExtensionProperties * props = (VkExtensionProperties *) calloc(count, sizeof(*props));
    if (props == NULL) {
        return 0;
    }
    int found = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, NULL, &count, props) == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; ++i) {
            if (strcmp(props[i].extensionName, name) == 0) {
                found = 1;
                break;
            }
        }
    }
    free(props);
    return found;
}

static int vk_physical_device_supports_sparse_coopmat(VkInstance instance, VkPhysicalDevice physical_device) {
    if (instance == VK_NULL_HANDLE || physical_device == VK_NULL_HANDLE || vk_env_disabled("TRELLIS_VK_COOPMAT")) {
        return 0;
    }
    if (!vk_physical_device_has_extension(physical_device, VK_KHR_16BIT_STORAGE_EXTENSION_NAME) ||
        !vk_physical_device_has_extension(physical_device, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) ||
        !vk_physical_device_has_extension(physical_device, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
        return 0;
    }

    VkPhysicalDeviceVulkan11Features features11;
    VkPhysicalDeviceVulkan12Features features12;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_features;
    VkPhysicalDeviceFeatures2 features2;
    memset(&features11, 0, sizeof(features11));
    memset(&features12, 0, sizeof(features12));
    memset(&coop_features, 0, sizeof(coop_features));
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features11;
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features11.pNext = &features12;
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = &coop_features;
    coop_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    if (!features11.storageBuffer16BitAccess || !features12.shaderFloat16 || !coop_features.cooperativeMatrix) {
        return 0;
    }

    VkPhysicalDeviceSubgroupProperties subgroup_props;
    VkPhysicalDeviceProperties2 props2;
    memset(&subgroup_props, 0, sizeof(subgroup_props));
    memset(&props2, 0, sizeof(props2));
    subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup_props;
    vkGetPhysicalDeviceProperties2(physical_device, &props2);
    if (subgroup_props.subgroupSize != 32u ||
        (subgroup_props.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
        return 0;
    }

    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR get_coop_props =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (get_coop_props == NULL) {
        return 0;
    }
    uint32_t prop_count = 0;
    if (get_coop_props(physical_device, &prop_count, NULL) != VK_SUCCESS || prop_count == 0) {
        return 0;
    }
    VkCooperativeMatrixPropertiesKHR * props =
        (VkCooperativeMatrixPropertiesKHR *) calloc(prop_count, sizeof(*props));
    if (props == NULL) {
        return 0;
    }
    for (uint32_t i = 0; i < prop_count; ++i) {
        props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    }
    int supported = 0;
    if (get_coop_props(physical_device, &prop_count, props) == VK_SUCCESS) {
        for (uint32_t i = 0; i < prop_count; ++i) {
            if (props[i].MSize == 16u &&
                props[i].NSize == 16u &&
                props[i].KSize == 16u &&
                props[i].AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                props[i].BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                props[i].CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                props[i].ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                props[i].saturatingAccumulation == VK_FALSE &&
                props[i].scope == VK_SCOPE_SUBGROUP_KHR) {
                supported = 1;
                break;
            }
        }
    }
    free(props);
    return supported;
}

static int32_t vk_find_coord_host(
    const uint32_t * keys,
    const int32_t * values,
    uint32_t table_mask,
    int b,
    int x,
    int y,
    int z) {
    uint32_t lo, hi;
    vk_pack_coord4(b, x, y, z, &lo, &hi);
    uint32_t slot = vk_hash_u64(lo, hi) & table_mask;
    for (uint32_t probe = 0; probe <= table_mask; ++probe) {
        const uint32_t key_lo = keys[2u * slot + 0u];
        const uint32_t key_hi = keys[2u * slot + 1u];
        if (key_lo == lo && key_hi == hi) {
            return values[slot];
        }
        if (key_lo == 0u && key_hi == 0u) {
            return -1;
        }
        slot = (slot + 1u) & table_mask;
    }
    return -1;
}

static trellis_status vk_status(VkResult result) {
    if (result == VK_SUCCESS) {
        return TRELLIS_STATUS_OK;
    }
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY ||
        result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    return TRELLIS_STATUS_ERROR;
}

static uint32_t find_memory_type(
    trellis_sparse_vk_backend * vk,
    uint32_t type_bits,
    VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(vk->physical_device, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) != 0 &&
            (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

static trellis_status vk_alloc_buffer(
    trellis_sparse_vk_backend * vk,
    size_t bytes,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memory_flags,
    int map_memory,
    trellis_sparse_buffer ** out) {
    if (vk == NULL || out == NULL || bytes == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    trellis_sparse_buffer * b = (trellis_sparse_buffer *) calloc(1, sizeof(*b));
    if (b == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    VkBufferCreateInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = (VkDeviceSize) bytes;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(vk->device, &buffer_info, NULL, &b->buffer);
    if (result != VK_SUCCESS) {
        free(b);
        return vk_status(result);
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk->device, b->buffer, &req);
    uint32_t memory_type = find_memory_type(vk, req.memoryTypeBits, memory_flags);
    if (memory_type == UINT32_MAX) {
        vkDestroyBuffer(vk->device, b->buffer, NULL);
        free(b);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = memory_type;
    result = vkAllocateMemory(vk->device, &alloc_info, NULL, &b->memory);
    if (result == VK_SUCCESS) {
        result = vkBindBufferMemory(vk->device, b->buffer, b->memory, 0);
    }
    if (result == VK_SUCCESS && map_memory) {
        result = vkMapMemory(vk->device, b->memory, 0, req.size, 0, &b->mapped);
    }
    if (result != VK_SUCCESS) {
        if (b->mapped != NULL) vkUnmapMemory(vk->device, b->memory);
        if (b->memory != VK_NULL_HANDLE) vkFreeMemory(vk->device, b->memory, NULL);
        vkDestroyBuffer(vk->device, b->buffer, NULL);
        free(b);
        return vk_status(result);
    }
    b->bytes = bytes;
    b->device_local = (memory_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 &&
        (memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0;
    if (b->mapped != NULL) {
        memset(b->mapped, 0, bytes);
    }
    *out = b;
    return TRELLIS_STATUS_OK;
}

static void vk_destroy_buffer_raw(trellis_sparse_vk_backend * vk, trellis_sparse_buffer * buffer) {
    if (vk == NULL || buffer == NULL) {
        return;
    }
    if (buffer->mapped != NULL) vkUnmapMemory(vk->device, buffer->memory);
    if (buffer->memory != VK_NULL_HANDLE) vkFreeMemory(vk->device, buffer->memory, NULL);
    if (buffer->buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk->device, buffer->buffer, NULL);
    free(buffer);
}

static void vk_add_released_bytes(size_t * total, size_t bytes) {
    if (total == NULL) {
        return;
    }
    *total = bytes > SIZE_MAX - *total ? SIZE_MAX : *total + bytes;
}

static trellis_status vk_trim(
    trellis_sparse_backend * backend,
    unsigned flags,
    size_t * released_bytes) {
    if (released_bytes != NULL) {
        *released_bytes = 0;
    }
    if (backend == NULL || (flags & ~(unsigned) TRELLIS_SPARSE_TRIM_ALL) != 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (flags == 0) {
        return TRELLIS_STATUS_OK;
    }

    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    if (vk->device != VK_NULL_HANDLE) {
        VkResult result = vkDeviceWaitIdle(vk->device);
        trellis_status status = vk_status(result);
        if (status != TRELLIS_STATUS_OK) {
            return status;
        }
        vk_reclaim_pending(vk);
    }

    size_t released = 0;
    if ((flags & TRELLIS_SPARSE_TRIM_WEIGHTS) != 0) {
        sparse_vk_weight * weight = vk->weights;
        vk->weights = NULL;
        while (weight != NULL) {
            sparse_vk_weight * next = weight->next;
            if (weight->buffer != NULL) {
                vk_add_released_bytes(&released, weight->buffer->bytes);
                vk_destroy_buffer_raw(vk, weight->buffer);
            }
            free(weight);
            weight = next;
        }
    }
    if ((flags & TRELLIS_SPARSE_TRIM_FREE_BUFFERS) != 0) {
        trellis_sparse_buffer * buffer = vk->free_buffers;
        vk->free_buffers = NULL;
        while (buffer != NULL) {
            trellis_sparse_buffer * next = buffer->next_free;
            vk_add_released_bytes(&released, buffer->bytes);
            vk_destroy_buffer_raw(vk, buffer);
            buffer = next;
        }
    }
    if (released_bytes != NULL) {
        *released_bytes = released;
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_alloc_buffer_with_trim_retry(
    trellis_sparse_vk_backend * vk,
    size_t bytes,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memory_flags,
    int map_memory,
    trellis_sparse_buffer ** out) {
    trellis_status status = vk_alloc_buffer(
        vk,
        bytes,
        usage,
        memory_flags,
        map_memory,
        out);
    if (status != TRELLIS_STATUS_OUT_OF_MEMORY || vk->free_buffers == NULL) {
        return status;
    }
    status = vk_trim(&vk->base, TRELLIS_SPARSE_TRIM_FREE_BUFFERS, NULL);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    return vk_alloc_buffer(
        vk,
        bytes,
        usage,
        memory_flags,
        map_memory,
        out);
}

static int vk_buffer_is_pooled(const trellis_sparse_vk_backend * vk, const trellis_sparse_buffer * buffer) {
    if (vk == NULL || buffer == NULL) {
        return 0;
    }
    for (const trellis_sparse_buffer * it = vk->free_buffers; it != NULL; it = it->next_free) {
        if (it == buffer) {
            return 1;
        }
    }
    return 0;
}

static int vk_remove_pooled_buffer(trellis_sparse_vk_backend * vk, const trellis_sparse_buffer * buffer) {
    if (vk == NULL || buffer == NULL) {
        return 0;
    }
    trellis_sparse_buffer ** link = &vk->free_buffers;
    while (*link != NULL) {
        if (*link == buffer) {
            *link = (*link)->next_free;
            return 1;
        }
        link = &(*link)->next_free;
    }
    return 0;
}

static int vk_pooled_buffer_is_reasonable_fit(size_t available, size_t requested) {
    if (available < requested) {
        return 0;
    }
    const size_t max_absolute_waste = 16u * 1024u * 1024u;
    if (available - requested <= max_absolute_waste) {
        return 1;
    }
    return requested > SIZE_MAX / 2u || available <= requested * 2u;
}

static trellis_sparse_buffer * vk_take_pooled_buffer(trellis_sparse_vk_backend * vk, size_t bytes) {
    if (vk == NULL) {
        return NULL;
    }
    trellis_sparse_buffer ** best_link = NULL;
    for (trellis_sparse_buffer ** link = &vk->free_buffers;
         *link != NULL;
         link = &(*link)->next_free) {
        trellis_sparse_buffer * candidate = *link;
        if (vk_pooled_buffer_is_reasonable_fit(candidate->bytes, bytes) &&
            (best_link == NULL || candidate->bytes < (*best_link)->bytes)) {
            best_link = link;
            if (candidate->bytes == bytes) {
                break;
            }
        }
    }
    if (best_link == NULL) {
        return NULL;
    }
    trellis_sparse_buffer * best = *best_link;
    *best_link = best->next_free;
    best->next_free = NULL;
    return best;
}

static trellis_status vk_alloc_bytes(
    trellis_sparse_vk_backend * vk,
    size_t bytes,
    trellis_sparse_buffer ** out) {
    if (out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    trellis_sparse_buffer * pooled = vk_take_pooled_buffer(vk, bytes);
    if (pooled != NULL) {
        *out = pooled;
        return TRELLIS_STATUS_OK;
    }
    return vk_alloc_buffer_with_trim_retry(
        vk,
        bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        0,
        out);
}

static trellis_status vk_alloc_staging(
    trellis_sparse_vk_backend * vk,
    size_t bytes,
    VkBufferUsageFlags usage,
    trellis_sparse_buffer ** out) {
    return vk_alloc_buffer_with_trim_retry(
        vk,
        bytes,
        usage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        1,
        out);
}

static void vk_free_buffer(trellis_sparse_backend * backend, trellis_sparse_buffer * buffer) {
    if (backend == NULL || buffer == NULL) {
        return;
    }
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    if (!vk->destroying && buffer->device_local && buffer->mapped == NULL && buffer != vk->dummy) {
        if (vk_buffer_is_pooled(vk, buffer)) {
            return;
        }
        buffer->next_free = vk->free_buffers;
        vk->free_buffers = buffer;
        return;
    }
    vk_remove_pooled_buffer(vk, buffer);
    if (vk->device != VK_NULL_HANDLE && !vk->destroying) {
        vkDeviceWaitIdle(vk->device);
    }
    vk_destroy_buffer_raw(vk, buffer);
}

static void vk_clear_c2s_cache(trellis_sparse_vk_backend * vk) {
    if (vk == NULL) {
        return;
    }
    sparse_vk_c2s_cache_entry * entry = vk->c2s_caches;
    while (entry != NULL) {
        sparse_vk_c2s_cache_entry * next = entry->next;
        if (entry->owns_buffers) {
            vk_free_buffer(&vk->base, entry->coords);
            vk_free_buffer(&vk->base, entry->parent);
            vk_free_buffer(&vk->base, entry->subidx);
        }
        free(entry->device_map);
        free(entry);
        entry = next;
    }
    vk->c2s_caches = NULL;
}

static const sparse_vk_c2s_cache_entry * vk_c2s_entry_from_map(
    const trellis_sparse_vk_backend * vk,
    const trellis_sparse_c2s_device_map * map) {
    if (vk == NULL || map == NULL || map->entry == NULL) {
        return NULL;
    }
    for (const sparse_vk_c2s_cache_entry * entry = vk->c2s_caches; entry != NULL; entry = entry->next) {
        if (entry == map->entry && entry->coords != NULL && entry->parent != NULL && entry->subidx != NULL) {
            return entry;
        }
    }
    return NULL;
}

static const sparse_vk_c2s_cache_entry * vk_find_c2s_cache_for_coords(
    const trellis_sparse_vk_backend * vk,
    const int32_t * coords,
    int64_t n) {
    if (vk == NULL || coords == NULL || n <= 0) {
        return NULL;
    }
    for (const sparse_vk_c2s_cache_entry * entry = vk->c2s_caches; entry != NULL; entry = entry->next) {
        if ((entry->coords_ptr == coords || (const int32_t *) entry->device_map == coords) &&
            entry->n == n && entry->coords != NULL) {
            return entry;
        }
    }
    return NULL;
}

static const sparse_vk_c2s_cache_entry * vk_find_c2s_cache_for_parent_subidx(
    const trellis_sparse_vk_backend * vk,
    const int32_t * parent,
    const int32_t * subidx,
    int64_t n) {
    if (vk == NULL || parent == NULL || subidx == NULL || n <= 0) {
        return NULL;
    }
    for (const sparse_vk_c2s_cache_entry * entry = vk->c2s_caches; entry != NULL; entry = entry->next) {
        if (entry->parent_ptr == parent && entry->subidx_ptr == subidx &&
            entry->n == n && entry->parent != NULL && entry->subidx != NULL) {
            return entry;
        }
    }
    return NULL;
}

static trellis_status vk_add_c2s_cache_entry(
    trellis_sparse_vk_backend * vk,
    const int32_t * coords,
    const int32_t * parent,
    const int32_t * subidx,
    int64_t n,
    trellis_sparse_buffer * coords_buf,
    trellis_sparse_buffer * parent_buf,
    trellis_sparse_buffer * subidx_buf,
    int owns_buffers,
    sparse_vk_c2s_cache_entry ** entry_out) {
    if (entry_out != NULL) {
        *entry_out = NULL;
    }
    if (vk == NULL || n <= 0 ||
        coords_buf == NULL || parent_buf == NULL || subidx_buf == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    sparse_vk_c2s_cache_entry * entry = (sparse_vk_c2s_cache_entry *) calloc(1, sizeof(*entry));
    if (entry == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    entry->coords_ptr = coords;
    entry->parent_ptr = parent;
    entry->subidx_ptr = subidx;
    entry->n = n;
    entry->coords = coords_buf;
    entry->parent = parent_buf;
    entry->subidx = subidx_buf;
    entry->owns_buffers = owns_buffers;
    if (owns_buffers) {
        entry->device_map = (trellis_sparse_c2s_device_map *) calloc(1, sizeof(*entry->device_map));
        if (entry->device_map == NULL) {
            free(entry);
            return TRELLIS_STATUS_OUT_OF_MEMORY;
        }
        entry->device_map->entry = entry;
    }
    entry->next = vk->c2s_caches;
    vk->c2s_caches = entry;
    if (entry_out != NULL) {
        *entry_out = entry;
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_copy_buffer_range(
    trellis_sparse_vk_backend * vk,
    trellis_sparse_buffer * src,
    trellis_sparse_buffer * dst,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes) {
    if (vk == NULL || src == NULL || dst == NULL || bytes == 0 ||
        src_offset > src->bytes || dst_offset > dst->bytes ||
        bytes > src->bytes - src_offset || bytes > dst->bytes - dst_offset) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    VkCommandBuffer command;
    VkCommandBufferAllocateInfo cmd_alloc;
    memset(&cmd_alloc, 0, sizeof(cmd_alloc));
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = vk->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    VkResult result = vkAllocateCommandBuffers(vk->device, &cmd_alloc, &command);
    if (result != VK_SUCCESS) {
        return vk_status(result);
    }
    VkCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(command, &begin);
    if (result == VK_SUCCESS) {
        VkBufferMemoryBarrier before[2];
        memset(before, 0, sizeof(before));
        before[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        before[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        before[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        before[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before[0].buffer = src->buffer;
        before[0].offset = (VkDeviceSize) src_offset;
        before[0].size = (VkDeviceSize) bytes;
        before[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        before[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        before[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        before[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before[1].buffer = dst->buffer;
        before[1].offset = (VkDeviceSize) dst_offset;
        before[1].size = (VkDeviceSize) bytes;
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            NULL,
            2,
            before,
            0,
            NULL);
        VkBufferCopy region;
        memset(&region, 0, sizeof(region));
        region.srcOffset = (VkDeviceSize) src_offset;
        region.dstOffset = (VkDeviceSize) dst_offset;
        region.size = (VkDeviceSize) bytes;
        vkCmdCopyBuffer(command, src->buffer, dst->buffer, 1, &region);
        VkBufferMemoryBarrier after;
        memset(&after, 0, sizeof(after));
        after.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
        after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        after.buffer = dst->buffer;
        after.offset = (VkDeviceSize) dst_offset;
        after.size = (VkDeviceSize) bytes;
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            0,
            0,
            NULL,
            1,
            &after,
            0,
            NULL);
        result = vkEndCommandBuffer(command);
    }
    if (result == VK_SUCCESS) {
        VkSubmitInfo submit;
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        result = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE);
    }
    if (result == VK_SUCCESS) {
        result = vkQueueWaitIdle(vk->queue);
        if (result == VK_SUCCESS) {
            vk_reclaim_pending(vk);
        }
    }
    vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &command);
    return vk_status(result);
}

static trellis_status vk_copy_buffer(
    trellis_sparse_vk_backend * vk,
    trellis_sparse_buffer * src,
    trellis_sparse_buffer * dst,
    size_t bytes) {
    return vk_copy_buffer_range(vk, src, dst, 0, 0, bytes);
}

static trellis_status vk_upload_bytes(
    trellis_sparse_vk_backend * vk,
    const void * src,
    size_t bytes,
    trellis_sparse_buffer ** out) {
    if (out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    const size_t alloc_bytes = bytes == 0 ? 4 : bytes;
    trellis_status status = vk_alloc_bytes(vk, alloc_bytes, out);
    if (status == TRELLIS_STATUS_OK && bytes != 0 && src != NULL) {
        trellis_sparse_buffer * staging = NULL;
        status = vk_alloc_staging(vk, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &staging);
        if (status == TRELLIS_STATUS_OK) {
            memcpy(staging->mapped, src, bytes);
            status = vk_copy_buffer(vk, staging, *out, bytes);
        }
        vk_free_buffer(&vk->base, staging);
        if (status != TRELLIS_STATUS_OK) {
            vk_free_buffer(&vk->base, *out);
            *out = NULL;
        }
    }
    return status;
}

static trellis_status vk_alloc_f32(trellis_sparse_backend * backend, size_t count, trellis_sparse_buffer ** out) {
    if (count > SIZE_MAX / sizeof(float)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    return vk_alloc_bytes((trellis_sparse_vk_backend *) backend, count * sizeof(float), out);
}

static trellis_status vk_upload_f32(trellis_sparse_backend * backend, const float * src, size_t count, trellis_sparse_buffer ** out) {
    if (src == NULL && count != 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return vk_upload_bytes((trellis_sparse_vk_backend *) backend, src, count * sizeof(float), out);
}

static trellis_status vk_upload_i32(trellis_sparse_backend * backend, const int32_t * src, size_t count, trellis_sparse_buffer ** out) {
    if (src == NULL && count != 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return vk_upload_bytes((trellis_sparse_vk_backend *) backend, src, count * sizeof(int32_t), out);
}

static trellis_status vk_download_f32(trellis_sparse_backend * backend, const trellis_sparse_buffer * src, float * dst, size_t count) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    if (vk == NULL || src == NULL || dst == NULL || src->bytes < count * sizeof(float)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t bytes = count * sizeof(float);
    trellis_sparse_buffer * staging = NULL;
    trellis_status status = vk_alloc_staging(vk, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &staging);
    if (status == TRELLIS_STATUS_OK) {
        status = vk_copy_buffer(vk, (trellis_sparse_buffer *) src, staging, bytes);
    }
    if (status == TRELLIS_STATUS_OK) {
        memcpy(dst, staging->mapped, bytes);
    }
    vk_free_buffer(backend, staging);
    return status;
}

static trellis_status vk_download_bytes(
    trellis_sparse_vk_backend * vk,
    const trellis_sparse_buffer * src,
    void * dst,
    size_t bytes) {
    if (vk == NULL || src == NULL || dst == NULL || bytes == 0 || src->bytes < bytes) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_sparse_buffer * staging = NULL;
    trellis_status status = vk_alloc_staging(vk, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &staging);
    if (status == TRELLIS_STATUS_OK) {
        status = vk_copy_buffer(vk, (trellis_sparse_buffer *) src, staging, bytes);
    }
    if (status == TRELLIS_STATUS_OK) {
        memcpy(dst, staging->mapped, bytes);
    }
    vk_free_buffer(&vk->base, staging);
    return status;
}

static trellis_status vk_download_bytes_at(
    trellis_sparse_vk_backend * vk,
    const trellis_sparse_buffer * src,
    size_t src_offset,
    void * dst,
    size_t bytes) {
    if (vk == NULL || src == NULL || dst == NULL || bytes == 0 ||
        src_offset > src->bytes || bytes > src->bytes - src_offset) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_sparse_buffer * staging = NULL;
    trellis_status status = vk_alloc_staging(vk, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &staging);
    if (status == TRELLIS_STATUS_OK) {
        status = vk_copy_buffer_range(vk, (trellis_sparse_buffer *) src, staging, src_offset, 0, bytes);
    }
    if (status == TRELLIS_STATUS_OK) {
        memcpy(dst, staging->mapped, bytes);
    }
    vk_free_buffer(&vk->base, staging);
    return status;
}

static trellis_status vk_get_weight(
    trellis_sparse_vk_backend * vk,
    const float * ptr,
    size_t count,
    trellis_sparse_buffer ** out) {
    if (vk == NULL || out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = vk->dummy;
    if (ptr == NULL || count == 0) {
        return TRELLIS_STATUS_OK;
    }
    for (sparse_vk_weight * w = vk->weights; w != NULL; w = w->next) {
        if (w->ptr == ptr && w->count == count) {
            *out = w->buffer;
            return TRELLIS_STATUS_OK;
        }
    }
    sparse_vk_weight * node = (sparse_vk_weight *) calloc(1, sizeof(*node));
    if (node == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    trellis_status status = vk_upload_bytes(vk, ptr, count * sizeof(float), &node->buffer);
    if (status != TRELLIS_STATUS_OK) {
        free(node);
        return status;
    }
    node->ptr = ptr;
    node->count = count;
    node->next = vk->weights;
    vk->weights = node;
    *out = node->buffer;
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_acquire_descriptor_set(
    trellis_sparse_vk_backend * vk,
    VkDescriptorSet * descriptor_set) {
    if (vk == NULL || descriptor_set == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *descriptor_set = VK_NULL_HANDLE;
    sparse_vk_descriptor_node * node = vk->free_descriptors;
    if (node != NULL) {
        vk->free_descriptors = node->next;
        *descriptor_set = node->set;
        free(node);
        return TRELLIS_STATUS_OK;
    }
    VkDescriptorSetAllocateInfo set_alloc;
    memset(&set_alloc, 0, sizeof(set_alloc));
    set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc.descriptorPool = vk->descriptor_pool;
    set_alloc.descriptorSetCount = 1;
    set_alloc.pSetLayouts = &vk->descriptor_set_layout;
    VkResult result = vkAllocateDescriptorSets(vk->device, &set_alloc, descriptor_set);
    return vk_status(result);
}

static void vk_release_descriptor_set(trellis_sparse_vk_backend * vk, VkDescriptorSet descriptor_set) {
    if (vk == NULL || descriptor_set == VK_NULL_HANDLE || vk->destroying) {
        return;
    }
    sparse_vk_descriptor_node * node = (sparse_vk_descriptor_node *) malloc(sizeof(*node));
    if (node == NULL) {
        vkFreeDescriptorSets(vk->device, vk->descriptor_pool, 1, &descriptor_set);
        return;
    }
    node->set = descriptor_set;
    node->next = vk->free_descriptors;
    vk->free_descriptors = node;
}

static void vk_defer_descriptor_set(trellis_sparse_vk_backend * vk, VkDescriptorSet descriptor_set) {
    if (vk == NULL || descriptor_set == VK_NULL_HANDLE || vk->destroying) {
        return;
    }
    sparse_vk_descriptor_node * node = (sparse_vk_descriptor_node *) malloc(sizeof(*node));
    if (node == NULL) {
        return;
    }
    node->set = descriptor_set;
    node->next = vk->pending_descriptors;
    vk->pending_descriptors = node;
}

static trellis_status vk_acquire_command_buffer(
    trellis_sparse_vk_backend * vk,
    VkCommandBuffer * command) {
    if (vk == NULL || command == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *command = VK_NULL_HANDLE;
    sparse_vk_command_node * node = vk->free_commands;
    if (node != NULL) {
        vk->free_commands = node->next;
        *command = node->command;
        free(node);
        VkResult result = vkResetCommandBuffer(*command, 0);
        return vk_status(result);
    }
    VkCommandBufferAllocateInfo cmd_alloc;
    memset(&cmd_alloc, 0, sizeof(cmd_alloc));
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = vk->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    VkResult result = vkAllocateCommandBuffers(vk->device, &cmd_alloc, command);
    return vk_status(result);
}

static void vk_release_command_buffer(trellis_sparse_vk_backend * vk, VkCommandBuffer command) {
    if (vk == NULL || command == VK_NULL_HANDLE || vk->destroying) {
        return;
    }
    sparse_vk_command_node * node = (sparse_vk_command_node *) malloc(sizeof(*node));
    if (node == NULL) {
        vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &command);
        return;
    }
    node->command = command;
    node->next = vk->free_commands;
    vk->free_commands = node;
}

static void vk_defer_command_buffer(trellis_sparse_vk_backend * vk, VkCommandBuffer command) {
    if (vk == NULL || command == VK_NULL_HANDLE || vk->destroying) {
        return;
    }
    sparse_vk_command_node * node = (sparse_vk_command_node *) malloc(sizeof(*node));
    if (node == NULL) {
        return;
    }
    node->command = command;
    node->next = vk->pending_commands;
    vk->pending_commands = node;
}

static void vk_reclaim_pending(trellis_sparse_vk_backend * vk) {
    if (vk == NULL) {
        return;
    }
    while (vk->pending_descriptors != NULL) {
        sparse_vk_descriptor_node * node = vk->pending_descriptors;
        vk->pending_descriptors = node->next;
        if (vk->device != VK_NULL_HANDLE &&
            vk->descriptor_pool != VK_NULL_HANDLE &&
            node->set != VK_NULL_HANDLE) {
            vkFreeDescriptorSets(vk->device, vk->descriptor_pool, 1, &node->set);
        }
        free(node);
    }
    while (vk->pending_commands != NULL) {
        sparse_vk_command_node * node = vk->pending_commands;
        vk->pending_commands = node->next;
        if (vk->device != VK_NULL_HANDLE &&
            vk->command_pool != VK_NULL_HANDLE &&
            node->command != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &node->command);
        }
        free(node);
    }
}

static trellis_status vk_record_dispatch(
    trellis_sparse_vk_backend * vk,
    VkCommandBuffer command,
    VkPipeline pipeline,
    const sparse_vk_push * push,
    trellis_sparse_buffer * f0,
    trellis_sparse_buffer * f1,
    trellis_sparse_buffer * f2,
    trellis_sparse_buffer * f3,
    trellis_sparse_buffer * i0,
    trellis_sparse_buffer * i1,
    trellis_sparse_buffer * u0,
    trellis_sparse_buffer * i2,
    trellis_sparse_buffer * i3,
    trellis_sparse_buffer * i4,
    uint32_t groups_x,
    uint32_t groups_y,
    uint32_t groups_z,
    VkDescriptorSet * descriptor_set_out) {
    if (vk == NULL || command == VK_NULL_HANDLE || pipeline == VK_NULL_HANDLE || push == NULL ||
        descriptor_set_out == NULL || groups_x == 0 || groups_y == 0 || groups_z == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *descriptor_set_out = VK_NULL_HANDLE;
    trellis_sparse_buffer * buffers[VK_BINDING_COUNT] = {
        f0 != NULL ? f0 : vk->dummy,
        f1 != NULL ? f1 : vk->dummy,
        f2 != NULL ? f2 : vk->dummy,
        f3 != NULL ? f3 : vk->dummy,
        i0 != NULL ? i0 : vk->dummy,
        i1 != NULL ? i1 : vk->dummy,
        u0 != NULL ? u0 : vk->dummy,
        i2 != NULL ? i2 : vk->dummy,
        i3 != NULL ? i3 : vk->dummy,
        i4 != NULL ? i4 : vk->dummy,
    };
    VkDescriptorSet descriptor_set;
    trellis_status status = vk_acquire_descriptor_set(vk, &descriptor_set);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    VkDescriptorBufferInfo infos[VK_BINDING_COUNT];
    VkWriteDescriptorSet writes[VK_BINDING_COUNT];
    memset(infos, 0, sizeof(infos));
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < VK_BINDING_COUNT; ++i) {
        infos[i].buffer = buffers[i]->buffer;
        infos[i].offset = 0;
        infos[i].range = buffers[i]->bytes;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(vk->device, VK_BINDING_COUNT, writes, 0, NULL);

    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
    vkCmdPushConstants(command, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*push), push);
    vkCmdDispatch(command, groups_x, groups_y, groups_z);
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT |
            VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(
        command,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        0,
        1,
        &barrier,
        0,
        NULL,
        0,
        NULL);
    *descriptor_set_out = descriptor_set;
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_record_dispatch_indirect(
    trellis_sparse_vk_backend * vk,
    VkCommandBuffer command,
    VkPipeline pipeline,
    const sparse_vk_push * push,
    trellis_sparse_buffer * f0,
    trellis_sparse_buffer * f1,
    trellis_sparse_buffer * f2,
    trellis_sparse_buffer * f3,
    trellis_sparse_buffer * i0,
    trellis_sparse_buffer * i1,
    trellis_sparse_buffer * u0,
    trellis_sparse_buffer * i2,
    trellis_sparse_buffer * i3,
    trellis_sparse_buffer * i4,
    trellis_sparse_buffer * indirect,
    size_t indirect_offset,
    VkDescriptorSet * descriptor_set_out) {
    if (vk == NULL || command == VK_NULL_HANDLE || pipeline == VK_NULL_HANDLE || push == NULL ||
        indirect == NULL || indirect_offset > indirect->bytes ||
        indirect->bytes - indirect_offset < 3u * sizeof(uint32_t) ||
        descriptor_set_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *descriptor_set_out = VK_NULL_HANDLE;
    trellis_sparse_buffer * buffers[VK_BINDING_COUNT] = {
        f0 != NULL ? f0 : vk->dummy,
        f1 != NULL ? f1 : vk->dummy,
        f2 != NULL ? f2 : vk->dummy,
        f3 != NULL ? f3 : vk->dummy,
        i0 != NULL ? i0 : vk->dummy,
        i1 != NULL ? i1 : vk->dummy,
        u0 != NULL ? u0 : vk->dummy,
        i2 != NULL ? i2 : vk->dummy,
        i3 != NULL ? i3 : vk->dummy,
        i4 != NULL ? i4 : vk->dummy,
    };
    VkDescriptorSet descriptor_set;
    trellis_status status = vk_acquire_descriptor_set(vk, &descriptor_set);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    VkDescriptorBufferInfo infos[VK_BINDING_COUNT];
    VkWriteDescriptorSet writes[VK_BINDING_COUNT];
    memset(infos, 0, sizeof(infos));
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < VK_BINDING_COUNT; ++i) {
        infos[i].buffer = buffers[i]->buffer;
        infos[i].offset = 0;
        infos[i].range = buffers[i]->bytes;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(vk->device, VK_BINDING_COUNT, writes, 0, NULL);

    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
    vkCmdPushConstants(command, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*push), push);
    vkCmdDispatchIndirect(command, indirect->buffer, (VkDeviceSize) indirect_offset);
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT |
            VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(
        command,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        0,
        1,
        &barrier,
        0,
        NULL,
        0,
        NULL);
    *descriptor_set_out = descriptor_set;
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_dispatch_pipeline(
    trellis_sparse_vk_backend * vk,
    VkPipeline pipeline,
    const sparse_vk_push * push,
    trellis_sparse_buffer * f0,
    trellis_sparse_buffer * f1,
    trellis_sparse_buffer * f2,
    trellis_sparse_buffer * f3,
    trellis_sparse_buffer * i0,
    trellis_sparse_buffer * i1,
    trellis_sparse_buffer * u0,
    trellis_sparse_buffer * i2,
    trellis_sparse_buffer * i3,
    trellis_sparse_buffer * i4,
    uint32_t groups_x,
    uint32_t groups_y,
    uint32_t groups_z) {
    if (vk == NULL || pipeline == VK_NULL_HANDLE || push == NULL ||
        groups_x == 0 || groups_y == 0 || groups_z == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    VkCommandBuffer command;
    trellis_status status = vk_acquire_command_buffer(vk, &command);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    VkCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult result = vkBeginCommandBuffer(command, &begin);
    if (result == VK_SUCCESS) {
        status = vk_record_dispatch(
            vk,
            command,
            pipeline,
            push,
            f0,
            f1,
            f2,
            f3,
            i0,
            i1,
            u0,
            i2,
            i3,
            i4,
            groups_x,
            groups_y,
            groups_z,
            &descriptor_set);
        if (status == TRELLIS_STATUS_OK) {
            result = vkEndCommandBuffer(command);
        } else {
            result = VK_ERROR_UNKNOWN;
        }
    }
    if (result == VK_SUCCESS) {
        VkSubmitInfo submit;
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        result = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE);
    }
    if (result == VK_SUCCESS) {
        result = vkQueueWaitIdle(vk->queue);
        if (result == VK_SUCCESS) {
            vk_reclaim_pending(vk);
        }
    }
    vk_release_command_buffer(vk, command);
    vk_release_descriptor_set(vk, descriptor_set);
    return status == TRELLIS_STATUS_OK ? vk_status(result) : status;
}

static trellis_status vk_dispatch(
    trellis_sparse_vk_backend * vk,
    sparse_vk_pipeline_id pipeline_id,
    const sparse_vk_push * push,
    trellis_sparse_buffer * f0,
    trellis_sparse_buffer * f1,
    trellis_sparse_buffer * f2,
    trellis_sparse_buffer * f3,
    trellis_sparse_buffer * i0,
    trellis_sparse_buffer * i1,
    trellis_sparse_buffer * u0,
    uint32_t work_items) {
    if (work_items == 0 || pipeline_id >= SPARSE_VK_PIPE_COUNT) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return vk_dispatch_pipeline(
        vk,
        vk->pipelines[pipeline_id],
        push,
        f0,
        f1,
        f2,
        f3,
        i0,
        i1,
        u0,
        NULL,
        NULL,
        NULL,
        (work_items + 127u) / 128u,
        1,
        1);
}

static trellis_status vk_mat_dispatch_dims(
    const trellis_sparse_vk_backend * vk,
    uint32_t cols,
    uint32_t rows,
    uint32_t * groups_x,
    uint32_t * groups_y,
    uint32_t * groups_z) {
    if (vk == NULL || cols == 0 || rows == 0 ||
        groups_x == NULL || groups_y == NULL || groups_z == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *groups_x = (cols + 15u) / 16u;
    const uint32_t row_tiles = (rows + 15u) / 16u;
    const uint32_t max_y = vk->max_workgroup_count[1] == 0 ? 65535u : vk->max_workgroup_count[1];
    const uint32_t max_z = vk->max_workgroup_count[2] == 0 ? 65535u : vk->max_workgroup_count[2];
    *groups_y = row_tiles < max_y ? row_tiles : max_y;
    *groups_z = (row_tiles + *groups_y - 1u) / *groups_y;
    if (*groups_x == 0 || *groups_y == 0 || *groups_z == 0 ||
        *groups_x > vk->max_workgroup_count[0] || *groups_z > max_z) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_coop_dispatch_dims(
    const trellis_sparse_vk_backend * vk,
    uint32_t cols,
    uint32_t rows,
    uint32_t * groups_x,
    uint32_t * groups_y,
    uint32_t * groups_z) {
    if (vk == NULL || cols == 0 || rows == 0 ||
        groups_x == NULL || groups_y == NULL || groups_z == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *groups_x = (cols + 31u) / 32u;
    const uint32_t row_tiles = (rows + 31u) / 32u;
    const uint32_t max_y = vk->max_workgroup_count[1] == 0 ? 65535u : vk->max_workgroup_count[1];
    const uint32_t max_z = vk->max_workgroup_count[2] == 0 ? 65535u : vk->max_workgroup_count[2];
    *groups_y = row_tiles < max_y ? row_tiles : max_y;
    *groups_z = (row_tiles + *groups_y - 1u) / *groups_y;
    if (*groups_x == 0 || *groups_y == 0 || *groups_z == 0 ||
        *groups_x > vk->max_workgroup_count[0] || *groups_z > max_z) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_dispatch_mat(
    trellis_sparse_vk_backend * vk,
    sparse_vk_pipeline_id pipeline_id,
    const sparse_vk_push * push,
    trellis_sparse_buffer * f0,
    trellis_sparse_buffer * f1,
    trellis_sparse_buffer * f2,
    trellis_sparse_buffer * f3,
    trellis_sparse_buffer * i0,
    trellis_sparse_buffer * i1,
    uint32_t cols,
    uint32_t rows) {
    if (vk == NULL || pipeline_id >= SPARSE_VK_PIPE_COUNT || cols == 0 || rows == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    uint32_t groups_x = 0;
    uint32_t groups_y = 0;
    uint32_t groups_z = 0;
    trellis_status status = vk_mat_dispatch_dims(vk, cols, rows, &groups_x, &groups_y, &groups_z);
    if (status != TRELLIS_STATUS_OK) return status;
    return vk_dispatch_pipeline(
        vk,
        vk->pipelines[pipeline_id],
        push,
        f0,
        f1,
        f2,
        f3,
        i0,
        i1,
        NULL,
        NULL,
        NULL,
        NULL,
        groups_x,
        groups_y,
        groups_z);
}

static trellis_status vk_dispatch_coop_mat(
    trellis_sparse_vk_backend * vk,
    sparse_vk_pipeline_id pipeline_id,
    const sparse_vk_push * push,
    trellis_sparse_buffer * f0,
    trellis_sparse_buffer * f1,
    trellis_sparse_buffer * f2,
    trellis_sparse_buffer * f3,
    trellis_sparse_buffer * i0,
    trellis_sparse_buffer * i1,
    uint32_t cols,
    uint32_t rows) {
    if (vk == NULL || pipeline_id >= SPARSE_VK_PIPE_COUNT ||
        !vk->coopmat_supported || vk->pipelines[pipeline_id] == VK_NULL_HANDLE ||
        cols == 0 || rows == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    uint32_t groups_x = 0;
    uint32_t groups_y = 0;
    uint32_t groups_z = 0;
    trellis_status status = vk_coop_dispatch_dims(vk, cols, rows, &groups_x, &groups_y, &groups_z);
    if (status != TRELLIS_STATUS_OK) return status;
    return vk_dispatch_pipeline(
        vk,
        vk->pipelines[pipeline_id],
        push,
        f0,
        f1,
        f2,
        f3,
        i0,
        i1,
        NULL,
        NULL,
        NULL,
        NULL,
        groups_x,
        groups_y,
        groups_z);
}

static trellis_status vk_linear(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const float * weight,
    const float * bias,
    trellis_sparse_buffer * y,
    int64_t n,
    int in_channels,
    int out_channels) {
    if (n <= 0 || n > UINT32_MAX || in_channels <= 0 || out_channels <= 0 ||
        (uint64_t) n * (uint64_t) out_channels > UINT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * w = NULL;
    trellis_sparse_buffer * b = NULL;
    trellis_status status = vk_get_weight(vk, weight, (size_t) in_channels * (size_t) out_channels, &w);
    if (status == TRELLIS_STATUS_OK) status = vk_get_weight(vk, bias, (size_t) out_channels, &b);
    sparse_vk_push push = {
        0, (uint32_t) n, (uint32_t) in_channels, (uint32_t) out_channels,
        (uint32_t) ((uint64_t) n * (uint64_t) out_channels), 0, bias != NULL ? 1u : 0u, 0.0f,
    };
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (vk_env_enabled("TRELLIS_VK_LINEAR_COOPMAT") &&
        vk->coopmat_supported &&
        vk->pipelines[SPARSE_VK_PIPE_LINEAR_COOP] != VK_NULL_HANDLE) {
        return vk_dispatch_coop_mat(
            vk, SPARSE_VK_PIPE_LINEAR_COOP, &push, (trellis_sparse_buffer *) x, w, b, y, NULL, NULL,
            (uint32_t) out_channels, (uint32_t) n);
    }
    return vk_dispatch_mat(
        vk,
        SPARSE_VK_PIPE_LINEAR_MAT,
        &push,
        (trellis_sparse_buffer *) x,
        w,
        b,
        y,
        NULL,
        NULL,
        (uint32_t) out_channels,
        (uint32_t) n);
}

static trellis_status vk_linear_silu(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const float * weight,
    const float * bias,
    trellis_sparse_buffer * y,
    int64_t n,
    int in_channels,
    int out_channels) {
    if (n <= 0 || n > UINT32_MAX || in_channels <= 0 || out_channels <= 0 ||
        (uint64_t) n * (uint64_t) out_channels > UINT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * w = NULL;
    trellis_sparse_buffer * b = NULL;
    trellis_status status = vk_get_weight(vk, weight, (size_t) in_channels * (size_t) out_channels, &w);
    if (status == TRELLIS_STATUS_OK) status = vk_get_weight(vk, bias, (size_t) out_channels, &b);
    sparse_vk_push push = {
        0, (uint32_t) n, (uint32_t) in_channels, (uint32_t) out_channels,
        (uint32_t) ((uint64_t) n * (uint64_t) out_channels), 0, bias != NULL ? 1u : 0u, 0.0f,
    };
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    if (vk_env_enabled("TRELLIS_VK_LINEAR_COOPMAT") &&
        vk->coopmat_supported &&
        vk->pipelines[SPARSE_VK_PIPE_LINEAR_SILU_COOP] != VK_NULL_HANDLE) {
        return vk_dispatch_coop_mat(
            vk, SPARSE_VK_PIPE_LINEAR_SILU_COOP, &push, (trellis_sparse_buffer *) x, w, b, y, NULL, NULL,
            (uint32_t) out_channels, (uint32_t) n);
    }
    return vk_dispatch_mat(
        vk,
        SPARSE_VK_PIPE_LINEAR_SILU_MAT,
        &push,
        (trellis_sparse_buffer *) x,
        w,
        b,
        y,
        NULL,
        NULL,
        (uint32_t) out_channels,
        (uint32_t) n);
}

static trellis_status vk_row_norm(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const float * gamma,
    const float * beta,
    trellis_sparse_buffer * y,
    int64_t n,
    int channels,
    float eps) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * g = NULL;
    trellis_sparse_buffer * b = NULL;
    trellis_status status = vk_get_weight(vk, gamma, (size_t) channels, &g);
    if (status == TRELLIS_STATUS_OK) status = vk_get_weight(vk, beta, (size_t) channels, &b);
    sparse_vk_push push = {
        0, (uint32_t) n, (uint32_t) channels, (uint32_t) channels,
        (uint32_t) n, 0, (gamma != NULL ? 1u : 0u) | (beta != NULL ? 2u : 0u), eps,
    };
    return status == TRELLIS_STATUS_OK ?
        vk_dispatch(vk, SPARSE_VK_PIPE_ROW_NORM, &push, (trellis_sparse_buffer *) x, g, b, y, NULL, NULL, NULL, (uint32_t) n) :
        status;
}

static trellis_status vk_row_norm_silu(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const float * gamma,
    const float * beta,
    trellis_sparse_buffer * y,
    int64_t n,
    int channels,
    float eps) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * g = NULL;
    trellis_sparse_buffer * b = NULL;
    trellis_status status = vk_get_weight(vk, gamma, (size_t) channels, &g);
    if (status == TRELLIS_STATUS_OK) status = vk_get_weight(vk, beta, (size_t) channels, &b);
    sparse_vk_push push = {
        0, (uint32_t) n, (uint32_t) channels, (uint32_t) channels,
        (uint32_t) n, 0, (gamma != NULL ? 1u : 0u) | (beta != NULL ? 2u : 0u), eps,
    };
    return status == TRELLIS_STATUS_OK ?
        vk_dispatch(vk, SPARSE_VK_PIPE_ROW_NORM_SILU, &push, (trellis_sparse_buffer *) x, g, b, y, NULL, NULL, NULL, (uint32_t) n) :
        status;
}

static trellis_status vk_silu_inplace(trellis_sparse_backend * backend, trellis_sparse_buffer * x, size_t count) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    sparse_vk_push push = { 0, 0, 0, 0, (uint32_t) count, 0, 0, 0.0f };
    return vk_dispatch(vk, SPARSE_VK_PIPE_SILU, &push, NULL, NULL, NULL, x, NULL, NULL, NULL, (uint32_t) count);
}

static trellis_status vk_add(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * a,
    const trellis_sparse_buffer * b,
    trellis_sparse_buffer * y,
    size_t count) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    sparse_vk_push push = { 0, 0, 0, 0, (uint32_t) count, 0, 0, 0.0f };
    return vk_dispatch(vk, SPARSE_VK_PIPE_ADD, &push, (trellis_sparse_buffer *) a, (trellis_sparse_buffer *) b, NULL, y, NULL, NULL, NULL, (uint32_t) count);
}

static trellis_status vk_record_sparse_dispatch_checked(
    trellis_sparse_vk_backend * vk,
    VkCommandBuffer command,
    sparse_vk_pipeline_id pipeline_id,
    const sparse_vk_push * push,
    trellis_sparse_buffer * f0,
    trellis_sparse_buffer * f1,
    trellis_sparse_buffer * f2,
    trellis_sparse_buffer * f3,
    trellis_sparse_buffer * i0,
    trellis_sparse_buffer * i1,
    trellis_sparse_buffer * u0,
    trellis_sparse_buffer * i2,
    trellis_sparse_buffer * i3,
    trellis_sparse_buffer * i4,
    uint32_t groups_x,
    VkDescriptorSet * descriptor_sets,
    uint32_t descriptor_capacity,
    uint32_t * descriptor_count) {
    if (descriptor_count == NULL || *descriptor_count >= descriptor_capacity ||
        pipeline_id >= SPARSE_VK_PIPE_COUNT) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_status status = vk_record_dispatch(
        vk,
        command,
        vk->pipelines[pipeline_id],
        push,
        f0,
        f1,
        f2,
        f3,
        i0,
        i1,
        u0,
        i2,
        i3,
        i4,
        groups_x,
        1,
        1,
        &descriptor_sets[*descriptor_count]);
    if (status == TRELLIS_STATUS_OK) {
        ++(*descriptor_count);
    }
    return status;
}

static trellis_status vk_build_sorted_masked_metadata(
    trellis_sparse_vk_backend * vk,
    trellis_sparse_rulebook * r,
    uint32_t n,
    uint32_t sort_count,
    uint32_t row_tiles) {
    if (vk == NULL || r == NULL || n == 0 || sort_count == 0 || row_tiles == 0 ||
        r->neighbors == NULL || r->sort_keys == NULL || r->sorted_idx == NULL ||
        r->gray_code == NULL || r->reduced_code == NULL ||
        r->valid_kernel == NULL || r->valid_kernel_seg == NULL ||
        r->valid_kernel_seg_tmp == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    enum { SORT_DESCRIPTOR_CAPACITY = 1024 };
    VkDescriptorSet descriptor_sets[SORT_DESCRIPTOR_CAPACITY];
    memset(descriptor_sets, 0, sizeof(descriptor_sets));
    uint32_t descriptor_count = 0;
    VkCommandBuffer command = VK_NULL_HANDLE;
    trellis_status status = vk_acquire_command_buffer(vk, &command);
    VkResult result = VK_SUCCESS;
    if (status == TRELLIS_STATUS_OK) {
        VkCommandBufferBeginInfo begin;
        memset(&begin, 0, sizeof(begin));
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        result = vkBeginCommandBuffer(command, &begin);
        status = vk_status(result);
    }

    if (status == TRELLIS_STATUS_OK) {
        sparse_vk_push init_push = {
            0,
            n,
            0,
            0,
            sort_count,
            0,
            0,
            0.0f,
        };
        status = vk_record_sparse_dispatch_checked(
            vk,
            command,
            SPARSE_VK_PIPE_RULEBOOK_MASK_INIT,
            &init_push,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            r->sorted_idx,
            r->sort_keys,
            r->gray_code,
            NULL,
            r->neighbors,
            (sort_count + 127u) / 128u,
            descriptor_sets,
            SORT_DESCRIPTOR_CAPACITY,
            &descriptor_count);
    }

    for (uint32_t k = 2u; status == TRELLIS_STATUS_OK && k <= sort_count; k <<= 1u) {
        for (uint32_t j = k >> 1u; status == TRELLIS_STATUS_OK && j > 0u; j >>= 1u) {
            sparse_vk_push sort_push = {
                0,
                n,
                0,
                0,
                sort_count,
                j,
                k,
                0.0f,
            };
            status = vk_record_sparse_dispatch_checked(
                vk,
                command,
                SPARSE_VK_PIPE_SORT_BITONIC,
                &sort_push,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                r->sorted_idx,
                r->sort_keys,
                NULL,
                NULL,
                NULL,
                (sort_count + 127u) / 128u,
                descriptor_sets,
                SORT_DESCRIPTOR_CAPACITY,
                &descriptor_count);
        }
        if (k > UINT32_MAX / 2u) {
            break;
        }
    }

    if (status == TRELLIS_STATUS_OK) {
        sparse_vk_push count_push = {
            0,
            n,
            0,
            0,
            row_tiles,
            0,
            16u,
            0.0f,
        };
        status = vk_record_sparse_dispatch_checked(
            vk,
            command,
            SPARSE_VK_PIPE_RULEBOOK_VALID_SORTED_COUNT,
            &count_push,
            NULL,
            NULL,
            NULL,
            NULL,
            r->sorted_idx,
            r->gray_code,
            NULL,
            r->reduced_code,
            r->valid_kernel_seg,
            NULL,
            (row_tiles + 127u) / 128u,
            descriptor_sets,
            SORT_DESCRIPTOR_CAPACITY,
            &descriptor_count);
    }

    trellis_sparse_buffer * scan_in = r->valid_kernel_seg;
    trellis_sparse_buffer * scan_out = r->valid_kernel_seg_tmp;
    const uint32_t seg_len = row_tiles + 1u;
    for (uint32_t stride = 1u; status == TRELLIS_STATUS_OK && stride < seg_len; stride <<= 1u) {
        sparse_vk_push scan_push = {
            0,
            seg_len,
            0,
            0,
            seg_len,
            0,
            stride,
            0.0f,
        };
        status = vk_record_sparse_dispatch_checked(
            vk,
            command,
            SPARSE_VK_PIPE_SCAN_I32_STRIDE,
            &scan_push,
            NULL,
            NULL,
            NULL,
            NULL,
            scan_in,
            scan_out,
            NULL,
            NULL,
            NULL,
            NULL,
            (seg_len + 127u) / 128u,
            descriptor_sets,
            SORT_DESCRIPTOR_CAPACITY,
            &descriptor_count);
        if (status == TRELLIS_STATUS_OK) {
            trellis_sparse_buffer * tmp = scan_in;
            scan_in = scan_out;
            scan_out = tmp;
        }
        if (stride > UINT32_MAX / 2u) {
            break;
        }
    }
    if (status == TRELLIS_STATUS_OK && scan_in != r->valid_kernel_seg) {
        trellis_sparse_buffer * tmp = r->valid_kernel_seg;
        r->valid_kernel_seg = r->valid_kernel_seg_tmp;
        r->valid_kernel_seg_tmp = tmp;
    }

    if (status == TRELLIS_STATUS_OK) {
        sparse_vk_push scatter_push = {
            0,
            row_tiles,
            0,
            0,
            row_tiles,
            0,
            0,
            0.0f,
        };
        status = vk_record_sparse_dispatch_checked(
            vk,
            command,
            SPARSE_VK_PIPE_RULEBOOK_VALID_SORTED_SCATTER,
            &scatter_push,
            NULL,
            NULL,
            NULL,
            NULL,
            r->reduced_code,
            r->valid_kernel_seg,
            NULL,
            r->valid_kernel,
            NULL,
            NULL,
            (row_tiles + 127u) / 128u,
            descriptor_sets,
            SORT_DESCRIPTOR_CAPACITY,
            &descriptor_count);
    }

    if (status == TRELLIS_STATUS_OK) {
        result = vkEndCommandBuffer(command);
        status = vk_status(result);
    }
    if (status == TRELLIS_STATUS_OK) {
        VkSubmitInfo submit;
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        result = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE);
        status = vk_status(result);
    }
    if (status == TRELLIS_STATUS_OK) {
        if (command != VK_NULL_HANDLE) {
            vk_defer_command_buffer(vk, command);
            command = VK_NULL_HANDLE;
        }
        for (uint32_t i = 0; i < descriptor_count; ++i) {
            vk_defer_descriptor_set(vk, descriptor_sets[i]);
            descriptor_sets[i] = VK_NULL_HANDLE;
        }
    } else {
        if (command != VK_NULL_HANDLE) {
            vk_release_command_buffer(vk, command);
        }
        for (uint32_t i = 0; i < descriptor_count; ++i) {
            vk_release_descriptor_set(vk, descriptor_sets[i]);
        }
    }
    return status;
}

static trellis_status vk_build_rulebook(
    trellis_sparse_backend * backend,
    const int32_t * coords_bxyz,
    int64_t n,
    trellis_sparse_rulebook ** out) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    if (coords_bxyz == NULL || n <= 0 || out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (n > UINT32_MAX || n > INT32_MAX || n > UINT32_MAX / 27 || n > INT64_MAX / 27) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const sparse_vk_c2s_cache_entry * c2s_cache = vk_find_c2s_cache_for_coords(vk, coords_bxyz, n);
    if (c2s_cache == NULL) {
        for (int64_t row = 0; row < n; ++row) {
            const int32_t * c = coords_bxyz + row * 4;
            if (c[0] < 0 || c[0] > 3 ||
                c[1] < 0 || c[1] > 1023 ||
                c[2] < 0 || c[2] > 1023 ||
                c[3] < 0 || c[3] > 1023) {
                return TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    trellis_sparse_rulebook * r = (trellis_sparse_rulebook *) calloc(1, sizeof(*r));
    if (r == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    const int64_t table_size = vk_next_power_of_two(n * 2);
    if (table_size <= 0 || table_size > UINT32_MAX || 27ull * (uint64_t) n > SIZE_MAX / sizeof(int32_t)) {
        free(r);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t keys_count = (size_t) table_size + 32u;
    const int build_sorted_masked = vk_env_enabled("TRELLIS_VK_SPARSE_CONV_MASKED_SORTED");
    const int build_tile_masked = vk_env_enabled("TRELLIS_VK_SPARSE_CONV_MASKED") && !build_sorted_masked;
    const int build_neighbors = build_tile_masked || build_sorted_masked;
    const uint32_t row_tiles = (uint32_t) (((uint64_t) n + 15u) / 16u);
    const int64_t sort_count_i64 = vk_next_power_of_two(n);
    if (sort_count_i64 <= 0 || sort_count_i64 > UINT32_MAX) {
        free(r);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const uint32_t sort_count = (uint32_t) sort_count_i64;
    const size_t tile_valid_slots = (size_t) row_tiles * 27u;
    r->n = n;
    r->table_mask = (uint32_t) (table_size - 1);

    trellis_status status = TRELLIS_STATUS_OK;
    if (c2s_cache != NULL) {
        r->coords = c2s_cache->coords;
        r->owns_coords = 0;
    } else {
        status = vk_upload_i32(backend, coords_bxyz, (size_t) n * 4u, &r->coords);
        if (status == TRELLIS_STATUS_OK) {
            r->owns_coords = 1;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, keys_count * sizeof(uint32_t), &r->hash_keys);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) table_size * sizeof(int32_t), &r->hash_values);
    }
    const size_t rulebook_slots = (size_t) n * 27u;
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, rulebook_slots * sizeof(int32_t), &r->src_rows);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, rulebook_slots * sizeof(int32_t), &r->dst_rows);
    }
    if (status == TRELLIS_STATUS_OK && build_neighbors) {
        status = vk_alloc_bytes(vk, rulebook_slots * sizeof(int32_t), &r->neighbors);
    }
    if (status == TRELLIS_STATUS_OK && build_tile_masked) {
        status = vk_alloc_bytes(vk, (size_t) row_tiles * sizeof(uint32_t), &r->tile_valid_counts);
    }
    if (status == TRELLIS_STATUS_OK && build_tile_masked) {
        status = vk_alloc_bytes(vk, tile_valid_slots * sizeof(int32_t), &r->tile_valid_offsets);
    }
    if (status == TRELLIS_STATUS_OK && build_sorted_masked) {
        status = vk_alloc_bytes(vk, (size_t) sort_count * sizeof(uint32_t), &r->sort_keys);
    }
    if (status == TRELLIS_STATUS_OK && build_sorted_masked) {
        status = vk_alloc_bytes(vk, (size_t) sort_count * sizeof(int32_t), &r->sorted_idx);
    }
    if (status == TRELLIS_STATUS_OK && build_sorted_masked) {
        status = vk_alloc_bytes(vk, (size_t) sort_count * sizeof(int32_t), &r->gray_code);
    }
    if (status == TRELLIS_STATUS_OK && build_sorted_masked) {
        status = vk_alloc_bytes(vk, (size_t) row_tiles * sizeof(int32_t), &r->reduced_code);
    }
    if (status == TRELLIS_STATUS_OK && build_sorted_masked) {
        status = vk_alloc_bytes(vk, tile_valid_slots * sizeof(int32_t), &r->valid_kernel);
    }
    if (status == TRELLIS_STATUS_OK && build_sorted_masked) {
        status = vk_alloc_bytes(vk, (size_t) (row_tiles + 1u) * sizeof(int32_t), &r->valid_kernel_seg);
    }
    if (status == TRELLIS_STATUS_OK && build_sorted_masked) {
        status = vk_alloc_bytes(vk, (size_t) (row_tiles + 1u) * sizeof(int32_t), &r->valid_kernel_seg_tmp);
    }
    if (status == TRELLIS_STATUS_OK) {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_sets[3] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
        uint32_t descriptor_count = 0;
        status = vk_acquire_command_buffer(vk, &command);
        VkResult result = VK_SUCCESS;
        if (status == TRELLIS_STATUS_OK) {
            VkCommandBufferBeginInfo begin;
            memset(&begin, 0, sizeof(begin));
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            result = vkBeginCommandBuffer(command, &begin);
            status = vk_status(result);
        }
        if (status == TRELLIS_STATUS_OK) {
            vkCmdFillBuffer(command, r->hash_keys->buffer, 0, (VkDeviceSize) (keys_count * sizeof(uint32_t)), 0u);
            if (r->neighbors != NULL) {
                vkCmdFillBuffer(command, r->neighbors->buffer, 0, (VkDeviceSize) (rulebook_slots * sizeof(int32_t)), 0xffffffffu);
                if (r->tile_valid_counts != NULL) {
                    vkCmdFillBuffer(command, r->tile_valid_counts->buffer, 0, (VkDeviceSize) ((size_t) row_tiles * sizeof(uint32_t)), 0u);
                }
            }
            VkBufferMemoryBarrier zero_barriers[3];
            memset(zero_barriers, 0, sizeof(zero_barriers));
            zero_barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            zero_barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            zero_barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            zero_barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            zero_barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            zero_barriers[0].buffer = r->hash_keys->buffer;
            zero_barriers[0].offset = 0;
            zero_barriers[0].size = (VkDeviceSize) (keys_count * sizeof(uint32_t));
            uint32_t zero_barrier_count = 1;
            if (r->neighbors != NULL) {
                zero_barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                zero_barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                zero_barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                zero_barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                zero_barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                zero_barriers[1].buffer = r->neighbors->buffer;
                zero_barriers[1].offset = 0;
                zero_barriers[1].size = (VkDeviceSize) (rulebook_slots * sizeof(int32_t));
                zero_barrier_count = 2;
                if (r->tile_valid_counts != NULL) {
                    zero_barriers[2].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    zero_barriers[2].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    zero_barriers[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    zero_barriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    zero_barriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    zero_barriers[2].buffer = r->tile_valid_counts->buffer;
                    zero_barriers[2].offset = 0;
                    zero_barriers[2].size = (VkDeviceSize) ((size_t) row_tiles * sizeof(uint32_t));
                    zero_barrier_count = 3;
                }
            }
            vkCmdPipelineBarrier(
                command,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                NULL,
                zero_barrier_count,
                zero_barriers,
                0,
                NULL);
        }
        if (status == TRELLIS_STATUS_OK) {
            sparse_vk_push insert_push = {
                0, (uint32_t) n, 0, 0, (uint32_t) n, r->table_mask, 0, 0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_RULEBOOK_HASH_INSERT],
                &insert_push,
                NULL,
                NULL,
                NULL,
                NULL,
                r->coords,
                r->hash_values,
                r->hash_keys,
                NULL,
                NULL,
                NULL,
                ((uint32_t) n + 127u) / 128u,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
        if (status == TRELLIS_STATUS_OK) {
            sparse_vk_push fill_push = {
                0,
                (uint32_t) n,
                0,
                0,
                (uint32_t) ((uint64_t) n * 27u),
                r->table_mask,
                r->neighbors != NULL ? 1u : 0u,
                0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_RULEBOOK_FILL],
                &fill_push,
                NULL,
                NULL,
                NULL,
                NULL,
                r->coords,
                r->hash_values,
                r->hash_keys,
                r->src_rows,
                r->dst_rows,
                r->neighbors,
                (fill_push.total + 127u) / 128u,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
        if (status == TRELLIS_STATUS_OK && r->tile_valid_counts != NULL && r->tile_valid_offsets != NULL) {
            sparse_vk_push tile_valid_push = {
                0,
                (uint32_t) n,
                0,
                0,
                (uint32_t) tile_valid_slots,
                0,
                0,
                0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_RULEBOOK_TILE_VALID],
                &tile_valid_push,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                r->tile_valid_offsets,
                r->tile_valid_counts,
                NULL,
                NULL,
                r->neighbors,
                (tile_valid_push.total + 127u) / 128u,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
        if (status == TRELLIS_STATUS_OK) {
            result = vkEndCommandBuffer(command);
            status = vk_status(result);
        }
        if (status == TRELLIS_STATUS_OK) {
            VkSubmitInfo submit;
            memset(&submit, 0, sizeof(submit));
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            result = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE);
            status = vk_status(result);
        }
        if (status == TRELLIS_STATUS_OK) {
            if (command != VK_NULL_HANDLE) {
                vk_defer_command_buffer(vk, command);
                command = VK_NULL_HANDLE;
            }
            for (uint32_t i = 0; i < descriptor_count; ++i) {
                vk_defer_descriptor_set(vk, descriptor_sets[i]);
                descriptor_sets[i] = VK_NULL_HANDLE;
            }
        } else {
            if (command != VK_NULL_HANDLE) {
                vk_release_command_buffer(vk, command);
            }
            for (uint32_t i = 0; i < descriptor_count; ++i) {
                vk_release_descriptor_set(vk, descriptor_sets[i]);
            }
        }
    }
    if (status == TRELLIS_STATUS_OK && build_sorted_masked) {
        status = vk_build_sorted_masked_metadata(vk, r, (uint32_t) n, sort_count, row_tiles);
    }
    if (status == TRELLIS_STATUS_OK) {
        for (int offset = 0; offset < 27; ++offset) {
            r->offset_starts[offset] = (int64_t) offset * n;
            r->offset_counts[offset] = n;
        }
        r->total_pairs = n * 27;
    }
    if (status != TRELLIS_STATUS_OK) {
        if (r->coords != NULL && r->owns_coords) vk_free_buffer(backend, r->coords);
        if (r->hash_keys != NULL) vk_free_buffer(backend, r->hash_keys);
        if (r->hash_values != NULL) vk_free_buffer(backend, r->hash_values);
        if (r->src_rows != NULL) vk_free_buffer(backend, r->src_rows);
        if (r->dst_rows != NULL) vk_free_buffer(backend, r->dst_rows);
        if (r->neighbors != NULL) vk_free_buffer(backend, r->neighbors);
        if (r->tile_valid_counts != NULL) vk_free_buffer(backend, r->tile_valid_counts);
        if (r->tile_valid_offsets != NULL) vk_free_buffer(backend, r->tile_valid_offsets);
        if (r->sort_keys != NULL) vk_free_buffer(backend, r->sort_keys);
        if (r->sorted_idx != NULL) vk_free_buffer(backend, r->sorted_idx);
        if (r->gray_code != NULL) vk_free_buffer(backend, r->gray_code);
        if (r->reduced_code != NULL) vk_free_buffer(backend, r->reduced_code);
        if (r->valid_kernel != NULL) vk_free_buffer(backend, r->valid_kernel);
        if (r->valid_kernel_seg != NULL) vk_free_buffer(backend, r->valid_kernel_seg);
        if (r->valid_kernel_seg_tmp != NULL) vk_free_buffer(backend, r->valid_kernel_seg_tmp);
        free(r->coords_host);
        free(r);
        return status;
    }
    *out = r;
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_sparse_conv3d(
    trellis_sparse_backend * backend,
    const trellis_sparse_rulebook * rulebook,
    const trellis_sparse_buffer * feats,
    const float * weight,
    const float * bias,
    trellis_sparse_buffer * out,
    int64_t n,
    int in_channels,
    int out_channels) {
    if (rulebook == NULL || feats == NULL || weight == NULL || out == NULL || n <= 0 || n > UINT32_MAX ||
        in_channels <= 0 || out_channels <= 0 ||
        (uint64_t) n * (uint64_t) out_channels > UINT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * w = NULL;
    trellis_sparse_buffer * b = NULL;
    trellis_sparse_buffer * indirect = NULL;
    const int use_masked_sorted =
        vk_env_enabled("TRELLIS_VK_SPARSE_CONV_MASKED_SORTED") &&
        rulebook->neighbors != NULL &&
        rulebook->sorted_idx != NULL &&
        rulebook->valid_kernel != NULL &&
        rulebook->valid_kernel_seg != NULL;
    const int use_masked_implicit =
        !use_masked_sorted &&
        vk_env_enabled("TRELLIS_VK_SPARSE_CONV_MASKED") &&
        rulebook->neighbors != NULL &&
        rulebook->tile_valid_counts != NULL &&
        rulebook->tile_valid_offsets != NULL;
    const int use_coopmat =
        vk_env_enabled("TRELLIS_VK_SPARSE_CONV_COOPMAT") &&
        !use_masked_sorted &&
        !use_masked_implicit &&
        vk->coopmat_supported &&
        vk->pipelines[SPARSE_VK_PIPE_SPARSE_CONV_OFFSET_COOP] != VK_NULL_HANDLE;
    VkPipeline conv_pipeline = vk->pipelines[
        use_coopmat ? SPARSE_VK_PIPE_SPARSE_CONV_OFFSET_COOP : SPARSE_VK_PIPE_SPARSE_CONV_OFFSET_MAT];
    trellis_status status = vk_get_weight(vk, weight, (size_t) out_channels * 27u * (size_t) in_channels, &w);
    if (status == TRELLIS_STATUS_OK) status = vk_get_weight(vk, bias, (size_t) out_channels, &b);
    if (status == TRELLIS_STATUS_OK && !use_masked_sorted && !use_masked_implicit && !use_coopmat && vk->use_sparse_conv_indirect) {
        status = vk_alloc_bytes(vk, 27u * 3u * sizeof(uint32_t), &indirect);
    }
    sparse_vk_push fill_push = {
        0, (uint32_t) n, 0, (uint32_t) out_channels,
        (uint32_t) ((uint64_t) n * (uint64_t) out_channels), 0, bias != NULL ? 1u : 0u, 0.0f,
    };
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_sets[32];
    uint32_t descriptor_count = 0;
    memset(descriptor_sets, 0, sizeof(descriptor_sets));
    if (status == TRELLIS_STATUS_OK) {
        status = vk_acquire_command_buffer(vk, &command);
    }
    VkResult result = VK_SUCCESS;
    if (status == TRELLIS_STATUS_OK) {
        VkCommandBufferBeginInfo begin;
        memset(&begin, 0, sizeof(begin));
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        result = vkBeginCommandBuffer(command, &begin);
        status = vk_status(result);
    }
    if (status == TRELLIS_STATUS_OK && !use_masked_sorted && !use_masked_implicit && !use_coopmat && vk->use_sparse_conv_indirect) {
        sparse_vk_push indirect_push = {
            0,
            27u,
            (uint32_t) out_channels,
            0,
            27u,
            vk->max_workgroup_count[1],
            0,
            0.0f,
        };
        status = vk_record_dispatch(
            vk,
            command,
            vk->pipelines[SPARSE_VK_PIPE_RULEBOOK_DISPATCH],
            &indirect_push,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            rulebook->hash_keys,
            indirect,
            NULL,
            NULL,
            1,
            1,
            1,
            &descriptor_sets[descriptor_count]);
        if (status == TRELLIS_STATUS_OK) ++descriptor_count;
    }
    if (status == TRELLIS_STATUS_OK && !use_masked_sorted && !use_masked_implicit) {
        status = vk_record_dispatch(
            vk,
            command,
            vk->pipelines[SPARSE_VK_PIPE_FILL_BIAS],
            &fill_push,
            NULL,
            NULL,
            b,
            out,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            (fill_push.total + 127u) / 128u,
            1,
            1,
            &descriptor_sets[descriptor_count]);
        if (status == TRELLIS_STATUS_OK) ++descriptor_count;
    }
    if (status == TRELLIS_STATUS_OK && use_masked_implicit) {
        uint32_t groups_x = 0;
        uint32_t groups_y = 0;
        uint32_t groups_z = 0;
        status = vk_mat_dispatch_dims(
            vk,
            (uint32_t) out_channels,
            (uint32_t) n,
            &groups_x,
            &groups_y,
            &groups_z);
        if (status == TRELLIS_STATUS_OK) {
            sparse_vk_push conv_push = {
                0,
                (uint32_t) n,
                (uint32_t) in_channels,
                (uint32_t) out_channels,
                (uint32_t) ((uint64_t) n * (uint64_t) out_channels),
                0,
                bias != NULL ? 1u : 0u,
                0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_SPARSE_CONV_MASKED_MAT],
                &conv_push,
                (trellis_sparse_buffer *) feats,
                w,
                b,
                out,
                rulebook->neighbors,
                rulebook->tile_valid_offsets,
                rulebook->tile_valid_counts,
                NULL,
                NULL,
                NULL,
                groups_x,
                groups_y,
                groups_z,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
    }
    if (status == TRELLIS_STATUS_OK && use_masked_sorted) {
        uint32_t groups_x = 0;
        uint32_t groups_y = 0;
        uint32_t groups_z = 0;
        status = vk_mat_dispatch_dims(
            vk,
            (uint32_t) out_channels,
            (uint32_t) n,
            &groups_x,
            &groups_y,
            &groups_z);
        if (status == TRELLIS_STATUS_OK) {
            sparse_vk_push conv_push = {
                0,
                (uint32_t) n,
                (uint32_t) in_channels,
                (uint32_t) out_channels,
                (uint32_t) ((uint64_t) n * (uint64_t) out_channels),
                0,
                bias != NULL ? 1u : 0u,
                0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_SPARSE_CONV_MASKED_SORTED_MAT],
                &conv_push,
                (trellis_sparse_buffer *) feats,
                w,
                b,
                out,
                rulebook->neighbors,
                rulebook->valid_kernel,
                rulebook->sorted_idx,
                rulebook->valid_kernel_seg,
                NULL,
                NULL,
                groups_x,
                groups_y,
                groups_z,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
    }
    for (int offset = 0; status == TRELLIS_STATUS_OK && !use_masked_sorted && !use_masked_implicit && offset < 27; ++offset) {
        const int64_t pair_count = n;
        const int64_t pair_start = rulebook->offset_starts[offset];
        if (pair_count > UINT32_MAX || pair_start > UINT32_MAX ||
            (uint64_t) pair_count * (uint64_t) out_channels > UINT32_MAX) {
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
            break;
        }
        sparse_vk_push conv_push = {
            0,
            (uint32_t) pair_count,
            (uint32_t) in_channels,
            (uint32_t) out_channels,
            (uint32_t) ((uint64_t) pair_count * (uint64_t) out_channels),
            (uint32_t) offset,
            (uint32_t) pair_start,
            0.0f,
        };
        if (!use_coopmat && vk->use_sparse_conv_indirect) {
            status = vk_record_dispatch_indirect(
                vk,
                command,
                conv_pipeline,
                &conv_push,
                (trellis_sparse_buffer *) feats,
                w,
                NULL,
                out,
                rulebook->src_rows,
                rulebook->dst_rows,
                rulebook->hash_keys,
                NULL,
                NULL,
                NULL,
                indirect,
                (size_t) offset * 3u * sizeof(uint32_t),
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        } else {
            uint32_t groups_x = 0;
            uint32_t groups_y = 0;
            uint32_t groups_z = 0;
            status = use_coopmat ?
                vk_coop_dispatch_dims(
                    vk,
                    (uint32_t) out_channels,
                    (uint32_t) pair_count,
                    &groups_x,
                    &groups_y,
                    &groups_z) :
                vk_mat_dispatch_dims(
                    vk,
                    (uint32_t) out_channels,
                    (uint32_t) pair_count,
                    &groups_x,
                    &groups_y,
                    &groups_z);
            if (status == TRELLIS_STATUS_OK) {
                status = vk_record_dispatch(
                    vk,
                    command,
                    conv_pipeline,
                    &conv_push,
                    (trellis_sparse_buffer *) feats,
                    w,
                    NULL,
                    out,
                    rulebook->src_rows,
                    rulebook->dst_rows,
                    rulebook->hash_keys,
                    NULL,
                    NULL,
                    NULL,
                    groups_x,
                    groups_y,
                    groups_z,
                    &descriptor_sets[descriptor_count]);
                if (status == TRELLIS_STATUS_OK) ++descriptor_count;
            }
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        result = vkEndCommandBuffer(command);
        status = vk_status(result);
    }
    if (status == TRELLIS_STATUS_OK) {
        VkSubmitInfo submit;
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        result = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE);
        status = vk_status(result);
    }
    if (status == TRELLIS_STATUS_OK) {
        result = vkQueueWaitIdle(vk->queue);
        status = vk_status(result);
        if (status == TRELLIS_STATUS_OK) {
            vk_reclaim_pending(vk);
        }
    }
    if (command != VK_NULL_HANDLE) {
        vk_release_command_buffer(vk, command);
    }
    for (uint32_t i = 0; i < descriptor_count; ++i) {
        vk_release_descriptor_set(vk, descriptor_sets[i]);
    }
    vk_free_buffer(backend, indirect);
    return status;
}

static void vk_free_rulebook(trellis_sparse_backend * backend, trellis_sparse_rulebook * rulebook) {
    if (rulebook == NULL) {
        return;
    }
    if (rulebook->owns_coords) {
        vk_free_buffer(backend, rulebook->coords);
    }
    vk_free_buffer(backend, rulebook->hash_keys);
    vk_free_buffer(backend, rulebook->hash_values);
    vk_free_buffer(backend, rulebook->src_rows);
    vk_free_buffer(backend, rulebook->dst_rows);
    vk_free_buffer(backend, rulebook->neighbors);
    vk_free_buffer(backend, rulebook->tile_valid_counts);
    vk_free_buffer(backend, rulebook->tile_valid_offsets);
    vk_free_buffer(backend, rulebook->sort_keys);
    vk_free_buffer(backend, rulebook->sorted_idx);
    vk_free_buffer(backend, rulebook->gray_code);
    vk_free_buffer(backend, rulebook->reduced_code);
    vk_free_buffer(backend, rulebook->valid_kernel);
    vk_free_buffer(backend, rulebook->valid_kernel_seg);
    vk_free_buffer(backend, rulebook->valid_kernel_seg_tmp);
    free(rulebook->coords_host);
    free(rulebook);
}

static trellis_status vk_c2s_gather(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const int32_t * parent,
    const int32_t * subidx,
    trellis_sparse_buffer * y,
    int64_t n_out,
    int out_channels) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * parent_buf = NULL;
    trellis_sparse_buffer * subidx_buf = NULL;
    int owns_parent = 1;
    int owns_subidx = 1;
    const sparse_vk_c2s_cache_entry * c2s_cache = vk_find_c2s_cache_for_parent_subidx(vk, parent, subidx, n_out);
    trellis_status status = TRELLIS_STATUS_OK;
    if (c2s_cache != NULL) {
        parent_buf = c2s_cache->parent;
        subidx_buf = c2s_cache->subidx;
        owns_parent = 0;
        owns_subidx = 0;
    } else {
        status = vk_upload_i32(backend, parent, (size_t) n_out, &parent_buf);
        if (status == TRELLIS_STATUS_OK) status = vk_upload_i32(backend, subidx, (size_t) n_out, &subidx_buf);
    }
    sparse_vk_push push = {
        0, (uint32_t) n_out, 0, (uint32_t) out_channels,
        (uint32_t) ((uint64_t) n_out * (uint64_t) out_channels), 0, 0, 0.0f,
    };
    if (status == TRELLIS_STATUS_OK) {
        status = vk_dispatch(
            vk,
            SPARSE_VK_PIPE_C2S_GATHER,
            &push,
            (trellis_sparse_buffer *) x,
            NULL,
            NULL,
            y,
            parent_buf,
            subidx_buf,
            NULL,
            push.total);
    }
    if (owns_parent) vk_free_buffer(backend, parent_buf);
    if (owns_subidx) vk_free_buffer(backend, subidx_buf);
    return status;
}

static trellis_status vk_skip_repeat(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const int32_t * parent,
    const int32_t * subidx,
    trellis_sparse_buffer * y,
    int64_t n_out,
    int in_channels,
    int out_channels) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * parent_buf = NULL;
    trellis_sparse_buffer * subidx_buf = NULL;
    int owns_parent = 1;
    int owns_subidx = 1;
    const sparse_vk_c2s_cache_entry * c2s_cache = vk_find_c2s_cache_for_parent_subidx(vk, parent, subidx, n_out);
    trellis_status status = TRELLIS_STATUS_OK;
    if (c2s_cache != NULL) {
        parent_buf = c2s_cache->parent;
        subidx_buf = c2s_cache->subidx;
        owns_parent = 0;
        owns_subidx = 0;
    } else {
        status = vk_upload_i32(backend, parent, (size_t) n_out, &parent_buf);
        if (status == TRELLIS_STATUS_OK) status = vk_upload_i32(backend, subidx, (size_t) n_out, &subidx_buf);
    }
    sparse_vk_push push = {
        0, (uint32_t) n_out, (uint32_t) in_channels, (uint32_t) out_channels,
        (uint32_t) ((uint64_t) n_out * (uint64_t) out_channels), 0, 0, 0.0f,
    };
    if (status == TRELLIS_STATUS_OK) {
        status = vk_dispatch(
            vk,
            SPARSE_VK_PIPE_SKIP_REPEAT,
            &push,
            (trellis_sparse_buffer *) x,
            NULL,
            NULL,
            y,
            parent_buf,
            subidx_buf,
            NULL,
            push.total);
    }
    if (owns_parent) vk_free_buffer(backend, parent_buf);
    if (owns_subidx) vk_free_buffer(backend, subidx_buf);
    return status;
}

static trellis_status vk_alias_c2s_map(
    trellis_sparse_backend * backend,
    const int32_t * coords_bxyz,
    const int32_t * parent,
    const int32_t * subidx,
    int64_t n,
    const int32_t * alias_coords_bxyz,
    const int32_t * alias_parent,
    const int32_t * alias_subidx) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    if (vk == NULL || coords_bxyz == NULL || parent == NULL || subidx == NULL ||
        alias_coords_bxyz == NULL || alias_parent == NULL || alias_subidx == NULL || n <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const sparse_vk_c2s_cache_entry * entry = vk_find_c2s_cache_for_parent_subidx(vk, parent, subidx, n);
    if (entry == NULL || entry->coords_ptr != coords_bxyz) {
        return TRELLIS_STATUS_OK;
    }
    return vk_add_c2s_cache_entry(
        vk,
        alias_coords_bxyz,
        alias_parent,
        alias_subidx,
        n,
        entry->coords,
        entry->parent,
        entry->subidx,
        0,
        NULL);
}

static trellis_status vk_build_c2s_map_device_impl(
    trellis_sparse_backend * backend,
    const int32_t * coords_bxyz,
    const trellis_sparse_c2s_device_map * coords_map,
    const trellis_sparse_buffer * logits,
    int64_t n,
    trellis_sparse_c2s_device_map ** map_out,
    int64_t * n_out) {
    if (backend == NULL || logits == NULL || map_out == NULL || n_out == NULL ||
        (coords_bxyz == NULL && coords_map == NULL) ||
        n <= 0 || n > UINT32_MAX / 8 || n > INT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *map_out = NULL;
    *n_out = 0;
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * coords_dev = NULL;
    trellis_sparse_buffer * counts_dev = NULL;
    trellis_sparse_buffer * prefix_dev = NULL;
    trellis_sparse_buffer * block_sums_dev = NULL;
    trellis_sparse_buffer * block_prefix_dev = NULL;
    trellis_sparse_buffer * out_coords_dev = NULL;
    trellis_sparse_buffer * parent_dev = NULL;
    trellis_sparse_buffer * subidx_dev = NULL;
    trellis_sparse_buffer * final_prefix_dev = NULL;
    int64_t m = 0;
    int owns_coords_dev = 1;
    sparse_vk_c2s_cache_entry * output_entry = NULL;

    trellis_status status = TRELLIS_STATUS_OK;
    if (coords_map != NULL) {
        const sparse_vk_c2s_cache_entry * entry = vk_c2s_entry_from_map(vk, coords_map);
        if (entry == NULL || entry->n != n) {
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
        } else {
            coords_dev = entry->coords;
            owns_coords_dev = 0;
        }
    } else {
        const sparse_vk_c2s_cache_entry * input_cache = vk_find_c2s_cache_for_coords(vk, coords_bxyz, n);
        if (input_cache != NULL) {
            coords_dev = input_cache->coords;
            owns_coords_dev = 0;
        } else {
            status = vk_upload_i32(backend, coords_bxyz, (size_t) n * 4u, &coords_dev);
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) n * sizeof(int32_t), &counts_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) n * sizeof(int32_t), &prefix_dev);
    }
    const int use_block_scan = !vk_env_disabled("TRELLIS_VK_C2S_BLOCK_SCAN");
    const uint32_t scan_blocks = (uint32_t) (((uint64_t) n + VK_SCAN_BLOCK_SIZE - 1u) / VK_SCAN_BLOCK_SIZE);
    if (status == TRELLIS_STATUS_OK && use_block_scan && scan_blocks > 1u) {
        status = vk_alloc_bytes(vk, (size_t) scan_blocks * sizeof(int32_t), &block_sums_dev);
    }
    if (status == TRELLIS_STATUS_OK && use_block_scan && scan_blocks > 1u) {
        status = vk_alloc_bytes(vk, (size_t) scan_blocks * sizeof(int32_t), &block_prefix_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_sets[64];
        uint32_t descriptor_count = 0;
        memset(descriptor_sets, 0, sizeof(descriptor_sets));
        status = vk_acquire_command_buffer(vk, &command);
        VkResult result = VK_SUCCESS;
        if (status == TRELLIS_STATUS_OK) {
            VkCommandBufferBeginInfo begin;
            memset(&begin, 0, sizeof(begin));
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            result = vkBeginCommandBuffer(command, &begin);
            status = vk_status(result);
        }
        if (status == TRELLIS_STATUS_OK) {
            sparse_vk_push count_push = {
                0, (uint32_t) n, 0, 0, (uint32_t) n, 0, 0, 0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_C2S_COUNT],
                &count_push,
                (trellis_sparse_buffer *) logits,
                NULL,
                NULL,
                NULL,
                NULL,
                counts_dev,
                NULL,
                NULL,
                NULL,
                NULL,
                ((uint32_t) n + 127u) / 128u,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
        if (use_block_scan && status == TRELLIS_STATUS_OK) {
            if (descriptor_count >= (uint32_t) (sizeof(descriptor_sets) / sizeof(descriptor_sets[0]))) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
        if (use_block_scan && status == TRELLIS_STATUS_OK) {
            sparse_vk_push scan_block_push = {
                0, (uint32_t) n, 0, 0, (uint32_t) n, 0, 0, 0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_SCAN_I32_BLOCK],
                &scan_block_push,
                NULL,
                NULL,
                NULL,
                NULL,
                counts_dev,
                prefix_dev,
                NULL,
                block_sums_dev,
                NULL,
                NULL,
                scan_blocks,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
        trellis_sparse_buffer * block_scan_in = block_sums_dev;
        trellis_sparse_buffer * block_scan_out = block_prefix_dev;
        for (uint32_t stride = 1; use_block_scan && status == TRELLIS_STATUS_OK && scan_blocks > 1u && stride < scan_blocks; stride <<= 1) {
            if (descriptor_count >= (uint32_t) (sizeof(descriptor_sets) / sizeof(descriptor_sets[0]))) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
                break;
            }
            sparse_vk_push scan_push = {
                0, scan_blocks, 0, 0, scan_blocks, 0, stride, 0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_SCAN_I32_STRIDE],
                &scan_push,
                NULL,
                NULL,
                NULL,
                NULL,
                block_scan_in,
                block_scan_out,
                NULL,
                NULL,
                NULL,
                NULL,
                (scan_blocks + 127u) / 128u,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
            trellis_sparse_buffer * tmp = block_scan_in;
            block_scan_in = block_scan_out;
            block_scan_out = tmp;
            if (stride > UINT32_MAX / 2u) {
                break;
            }
        }
        if (use_block_scan && status == TRELLIS_STATUS_OK && scan_blocks > 1u) {
            if (descriptor_count >= (uint32_t) (sizeof(descriptor_sets) / sizeof(descriptor_sets[0]))) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
        if (use_block_scan && status == TRELLIS_STATUS_OK && scan_blocks > 1u) {
            sparse_vk_push add_block_push = {
                0, (uint32_t) n, 0, 0, (uint32_t) n, 0, 0, 0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_SCAN_I32_ADD_BLOCK_OFFSETS],
                &add_block_push,
                NULL,
                NULL,
                NULL,
                NULL,
                prefix_dev,
                counts_dev,
                NULL,
                block_scan_in,
                NULL,
                NULL,
                scan_blocks,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
        if (use_block_scan) {
            final_prefix_dev = scan_blocks > 1u ? counts_dev : prefix_dev;
        } else {
            trellis_sparse_buffer * scan_in = counts_dev;
            trellis_sparse_buffer * scan_out = prefix_dev;
            for (uint32_t stride = 1; status == TRELLIS_STATUS_OK && stride < (uint32_t) n; stride <<= 1) {
                if (descriptor_count >= (uint32_t) (sizeof(descriptor_sets) / sizeof(descriptor_sets[0]))) {
                    status = TRELLIS_STATUS_INVALID_ARGUMENT;
                    break;
                }
                sparse_vk_push scan_push = {
                    0, (uint32_t) n, 0, 0, (uint32_t) n, 0, stride, 0.0f,
                };
                status = vk_record_dispatch(
                    vk,
                    command,
                    vk->pipelines[SPARSE_VK_PIPE_SCAN_I32_STRIDE],
                    &scan_push,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    scan_in,
                    scan_out,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    ((uint32_t) n + 127u) / 128u,
                    1,
                    1,
                    &descriptor_sets[descriptor_count]);
                if (status == TRELLIS_STATUS_OK) ++descriptor_count;
                trellis_sparse_buffer * tmp = scan_in;
                scan_in = scan_out;
                scan_out = tmp;
                if (stride > UINT32_MAX / 2u) {
                    break;
                }
            }
            final_prefix_dev = scan_in;
        }
        if (status == TRELLIS_STATUS_OK) {
            result = vkEndCommandBuffer(command);
            status = vk_status(result);
        }
        if (status == TRELLIS_STATUS_OK) {
            VkSubmitInfo submit;
            memset(&submit, 0, sizeof(submit));
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            result = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE);
            status = vk_status(result);
        }
        if (status == TRELLIS_STATUS_OK) {
            result = vkQueueWaitIdle(vk->queue);
            status = vk_status(result);
            if (status == TRELLIS_STATUS_OK) {
                vk_reclaim_pending(vk);
            }
        }
        if (command != VK_NULL_HANDLE) {
            vk_release_command_buffer(vk, command);
        }
        for (uint32_t i = 0; i < descriptor_count; ++i) {
            vk_release_descriptor_set(vk, descriptor_sets[i]);
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        int32_t m32 = 0;
        status = vk_download_bytes_at(
            vk,
            final_prefix_dev,
            ((size_t) n - 1u) * sizeof(int32_t),
            &m32,
            sizeof(m32));
        if (status == TRELLIS_STATUS_OK) {
            if (m32 <= 0 || m32 > INT32_MAX || (int64_t) m32 > n * 8) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
            } else {
                m = (int64_t) m32;
            }
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) m * 4u * sizeof(int32_t), &out_coords_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) m * sizeof(int32_t), &parent_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) m * sizeof(int32_t), &subidx_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        sparse_vk_push fill_push = {
            0,
            (uint32_t) n,
            0,
            0,
            (uint32_t) ((uint64_t) n * 8u),
            0,
            0,
            0.0f,
        };
        status = vk_dispatch_pipeline(
            vk,
            vk->pipelines[SPARSE_VK_PIPE_C2S_FILL],
            &fill_push,
            (trellis_sparse_buffer *) logits,
            NULL,
            NULL,
            NULL,
            coords_dev,
            final_prefix_dev,
            NULL,
            out_coords_dev,
            parent_dev,
            subidx_dev,
            (fill_push.total + 127u) / 128u,
            1,
            1);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_add_c2s_cache_entry(
            vk,
            NULL,
            NULL,
            NULL,
            m,
            out_coords_dev,
            parent_dev,
            subidx_dev,
            1,
            &output_entry);
        if (status == TRELLIS_STATUS_OK) {
            out_coords_dev = NULL;
            parent_dev = NULL;
            subidx_dev = NULL;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        *map_out = output_entry->device_map;
        *n_out = m;
    }
    vk_free_buffer(backend, subidx_dev);
    vk_free_buffer(backend, parent_dev);
    vk_free_buffer(backend, out_coords_dev);
    vk_free_buffer(backend, prefix_dev);
    vk_free_buffer(backend, block_prefix_dev);
    vk_free_buffer(backend, block_sums_dev);
    vk_free_buffer(backend, counts_dev);
    if (owns_coords_dev) vk_free_buffer(backend, coords_dev);
    return status;
}

static trellis_status vk_build_c2s_map(
    trellis_sparse_backend * backend,
    const int32_t * coords_bxyz,
    const trellis_sparse_buffer * logits,
    int64_t n,
    int32_t ** coords_out,
    int32_t ** parent_out,
    int32_t ** subidx_out,
    int64_t * n_out) {
    if (backend == NULL || coords_bxyz == NULL || logits == NULL ||
        coords_out == NULL || parent_out == NULL || subidx_out == NULL || n_out == NULL ||
        n <= 0 || n > UINT32_MAX / 8 || n > INT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *coords_out = NULL;
    *parent_out = NULL;
    *subidx_out = NULL;
    *n_out = 0;
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * coords_dev = NULL;
    trellis_sparse_buffer * counts_dev = NULL;
    trellis_sparse_buffer * prefix_dev = NULL;
    trellis_sparse_buffer * block_sums_dev = NULL;
    trellis_sparse_buffer * block_prefix_dev = NULL;
    trellis_sparse_buffer * out_coords_dev = NULL;
    trellis_sparse_buffer * parent_dev = NULL;
    trellis_sparse_buffer * subidx_dev = NULL;
    trellis_sparse_buffer * final_prefix_dev = NULL;
    int32_t * out_coords = NULL;
    int32_t * out_parent = NULL;
    int32_t * out_subidx = NULL;
    int64_t m = 0;
    int owns_coords_dev = 1;

    const sparse_vk_c2s_cache_entry * input_cache = vk_find_c2s_cache_for_coords(vk, coords_bxyz, n);
    trellis_status status = TRELLIS_STATUS_OK;
    if (input_cache != NULL) {
        coords_dev = input_cache->coords;
        owns_coords_dev = 0;
    } else {
        status = vk_upload_i32(backend, coords_bxyz, (size_t) n * 4u, &coords_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) n * sizeof(int32_t), &counts_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) n * sizeof(int32_t), &prefix_dev);
    }
    const int use_block_scan = !vk_env_disabled("TRELLIS_VK_C2S_BLOCK_SCAN");
    const uint32_t scan_blocks = (uint32_t) (((uint64_t) n + VK_SCAN_BLOCK_SIZE - 1u) / VK_SCAN_BLOCK_SIZE);
    if (status == TRELLIS_STATUS_OK && use_block_scan && scan_blocks > 1u) {
        status = vk_alloc_bytes(vk, (size_t) scan_blocks * sizeof(int32_t), &block_sums_dev);
    }
    if (status == TRELLIS_STATUS_OK && use_block_scan && scan_blocks > 1u) {
        status = vk_alloc_bytes(vk, (size_t) scan_blocks * sizeof(int32_t), &block_prefix_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_sets[64];
        uint32_t descriptor_count = 0;
        memset(descriptor_sets, 0, sizeof(descriptor_sets));
        status = vk_acquire_command_buffer(vk, &command);
        VkResult result = VK_SUCCESS;
        if (status == TRELLIS_STATUS_OK) {
            VkCommandBufferBeginInfo begin;
            memset(&begin, 0, sizeof(begin));
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            result = vkBeginCommandBuffer(command, &begin);
            status = vk_status(result);
        }
        if (status == TRELLIS_STATUS_OK) {
            sparse_vk_push count_push = {
                0, (uint32_t) n, 0, 0, (uint32_t) n, 0, 0, 0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_C2S_COUNT],
                &count_push,
                (trellis_sparse_buffer *) logits,
                NULL,
                NULL,
                NULL,
                NULL,
                counts_dev,
                NULL,
                NULL,
                NULL,
                NULL,
                ((uint32_t) n + 127u) / 128u,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
        if (use_block_scan && status == TRELLIS_STATUS_OK) {
            if (descriptor_count >= (uint32_t) (sizeof(descriptor_sets) / sizeof(descriptor_sets[0]))) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
        if (use_block_scan && status == TRELLIS_STATUS_OK) {
            sparse_vk_push scan_block_push = {
                0, (uint32_t) n, 0, 0, (uint32_t) n, 0, 0, 0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_SCAN_I32_BLOCK],
                &scan_block_push,
                NULL,
                NULL,
                NULL,
                NULL,
                counts_dev,
                prefix_dev,
                NULL,
                block_sums_dev,
                NULL,
                NULL,
                scan_blocks,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
        trellis_sparse_buffer * block_scan_in = block_sums_dev;
        trellis_sparse_buffer * block_scan_out = block_prefix_dev;
        for (uint32_t stride = 1; use_block_scan && status == TRELLIS_STATUS_OK && scan_blocks > 1u && stride < scan_blocks; stride <<= 1) {
            if (descriptor_count >= (uint32_t) (sizeof(descriptor_sets) / sizeof(descriptor_sets[0]))) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
                break;
            }
            sparse_vk_push scan_push = {
                0, scan_blocks, 0, 0, scan_blocks, 0, stride, 0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_SCAN_I32_STRIDE],
                &scan_push,
                NULL,
                NULL,
                NULL,
                NULL,
                block_scan_in,
                block_scan_out,
                NULL,
                NULL,
                NULL,
                NULL,
                (scan_blocks + 127u) / 128u,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
            trellis_sparse_buffer * tmp = block_scan_in;
            block_scan_in = block_scan_out;
            block_scan_out = tmp;
            if (stride > UINT32_MAX / 2u) {
                break;
            }
        }
        if (use_block_scan && status == TRELLIS_STATUS_OK && scan_blocks > 1u) {
            if (descriptor_count >= (uint32_t) (sizeof(descriptor_sets) / sizeof(descriptor_sets[0]))) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
            }
        }
        if (use_block_scan && status == TRELLIS_STATUS_OK && scan_blocks > 1u) {
            sparse_vk_push add_block_push = {
                0, (uint32_t) n, 0, 0, (uint32_t) n, 0, 0, 0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipelines[SPARSE_VK_PIPE_SCAN_I32_ADD_BLOCK_OFFSETS],
                &add_block_push,
                NULL,
                NULL,
                NULL,
                NULL,
                prefix_dev,
                counts_dev,
                NULL,
                block_scan_in,
                NULL,
                NULL,
                scan_blocks,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
        if (use_block_scan) {
            final_prefix_dev = scan_blocks > 1u ? counts_dev : prefix_dev;
        } else {
            trellis_sparse_buffer * scan_in = counts_dev;
            trellis_sparse_buffer * scan_out = prefix_dev;
            for (uint32_t stride = 1; status == TRELLIS_STATUS_OK && stride < (uint32_t) n; stride <<= 1) {
                if (descriptor_count >= (uint32_t) (sizeof(descriptor_sets) / sizeof(descriptor_sets[0]))) {
                    status = TRELLIS_STATUS_INVALID_ARGUMENT;
                    break;
                }
                sparse_vk_push scan_push = {
                    0, (uint32_t) n, 0, 0, (uint32_t) n, 0, stride, 0.0f,
                };
                status = vk_record_dispatch(
                    vk,
                    command,
                    vk->pipelines[SPARSE_VK_PIPE_SCAN_I32_STRIDE],
                    &scan_push,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    scan_in,
                    scan_out,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    ((uint32_t) n + 127u) / 128u,
                    1,
                    1,
                    &descriptor_sets[descriptor_count]);
                if (status == TRELLIS_STATUS_OK) ++descriptor_count;
                trellis_sparse_buffer * tmp = scan_in;
                scan_in = scan_out;
                scan_out = tmp;
                if (stride > UINT32_MAX / 2u) {
                    break;
                }
            }
            final_prefix_dev = scan_in;
        }
        if (status == TRELLIS_STATUS_OK) {
            result = vkEndCommandBuffer(command);
            status = vk_status(result);
        }
        if (status == TRELLIS_STATUS_OK) {
            VkSubmitInfo submit;
            memset(&submit, 0, sizeof(submit));
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            result = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE);
            status = vk_status(result);
        }
        if (status == TRELLIS_STATUS_OK) {
            result = vkQueueWaitIdle(vk->queue);
            status = vk_status(result);
            if (status == TRELLIS_STATUS_OK) {
                vk_reclaim_pending(vk);
            }
        }
        if (command != VK_NULL_HANDLE) {
            vk_release_command_buffer(vk, command);
        }
        for (uint32_t i = 0; i < descriptor_count; ++i) {
            vk_release_descriptor_set(vk, descriptor_sets[i]);
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        int32_t m32 = 0;
        status = vk_download_bytes_at(
            vk,
            final_prefix_dev,
            ((size_t) n - 1u) * sizeof(int32_t),
            &m32,
            sizeof(m32));
        if (status == TRELLIS_STATUS_OK) {
            if (m32 <= 0 || m32 > INT32_MAX || (int64_t) m32 > n * 8) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
            } else {
                m = (int64_t) m32;
            }
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) m * 4u * sizeof(int32_t), &out_coords_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) m * sizeof(int32_t), &parent_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) m * sizeof(int32_t), &subidx_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        sparse_vk_push fill_push = {
            0,
            (uint32_t) n,
            0,
            0,
            (uint32_t) ((uint64_t) n * 8u),
            0,
            0,
            0.0f,
        };
        status = vk_dispatch_pipeline(
            vk,
            vk->pipelines[SPARSE_VK_PIPE_C2S_FILL],
            &fill_push,
            (trellis_sparse_buffer *) logits,
            NULL,
            NULL,
            NULL,
            coords_dev,
            final_prefix_dev,
            NULL,
            out_coords_dev,
            parent_dev,
            subidx_dev,
            (fill_push.total + 127u) / 128u,
            1,
            1);
    }
    if (status == TRELLIS_STATUS_OK) {
        out_coords = (int32_t *) malloc((size_t) m * 4u * sizeof(int32_t));
        out_parent = (int32_t *) malloc((size_t) m * sizeof(int32_t));
        out_subidx = (int32_t *) malloc((size_t) m * sizeof(int32_t));
        if (out_coords == NULL || out_parent == NULL || out_subidx == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_download_bytes(vk, out_coords_dev, out_coords, (size_t) m * 4u * sizeof(int32_t));
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_download_bytes(vk, parent_dev, out_parent, (size_t) m * sizeof(int32_t));
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_download_bytes(vk, subidx_dev, out_subidx, (size_t) m * sizeof(int32_t));
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_add_c2s_cache_entry(
            vk,
            out_coords,
            out_parent,
            out_subidx,
            m,
            out_coords_dev,
            parent_dev,
            subidx_dev,
            1,
            NULL);
        if (status == TRELLIS_STATUS_OK) {
            out_coords_dev = NULL;
            parent_dev = NULL;
            subidx_dev = NULL;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        *coords_out = out_coords;
        *parent_out = out_parent;
        *subidx_out = out_subidx;
        *n_out = m;
        out_coords = NULL;
        out_parent = NULL;
        out_subidx = NULL;
    }
    free(out_coords);
    free(out_parent);
    free(out_subidx);
    vk_free_buffer(backend, subidx_dev);
    vk_free_buffer(backend, parent_dev);
    vk_free_buffer(backend, out_coords_dev);
    vk_free_buffer(backend, prefix_dev);
    vk_free_buffer(backend, block_prefix_dev);
    vk_free_buffer(backend, block_sums_dev);
    vk_free_buffer(backend, counts_dev);
    if (owns_coords_dev) vk_free_buffer(backend, coords_dev);
    return status;
}

static trellis_status vk_build_c2s_map_device(
    trellis_sparse_backend * backend,
    const int32_t * coords_bxyz,
    const trellis_sparse_c2s_device_map * coords_map,
    const trellis_sparse_buffer * logits,
    int64_t n,
    trellis_sparse_c2s_device_map ** map_out,
    int64_t * n_out) {
    return vk_build_c2s_map_device_impl(backend, coords_bxyz, coords_map, logits, n, map_out, n_out);
}

static trellis_status vk_build_rulebook_for_c2s_map(
    trellis_sparse_backend * backend,
    const trellis_sparse_c2s_device_map * map,
    trellis_sparse_rulebook ** out) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    const sparse_vk_c2s_cache_entry * entry = vk_c2s_entry_from_map(vk, map);
    if (entry == NULL || out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return vk_build_rulebook(backend, (const int32_t *) map, entry->n, out);
}

static trellis_status vk_c2s_gather_device(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const trellis_sparse_c2s_device_map * map,
    trellis_sparse_buffer * y,
    int64_t n_out,
    int out_channels) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    const sparse_vk_c2s_cache_entry * entry = vk_c2s_entry_from_map(vk, map);
    if (entry == NULL || x == NULL || y == NULL || n_out != entry->n || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    sparse_vk_push push = {
        0, (uint32_t) n_out, 0, (uint32_t) out_channels,
        (uint32_t) ((uint64_t) n_out * (uint64_t) out_channels), 0, 0, 0.0f,
    };
    return vk_dispatch(
        vk,
        SPARSE_VK_PIPE_C2S_GATHER,
        &push,
        (trellis_sparse_buffer *) x,
        NULL,
        NULL,
        y,
        entry->parent,
        entry->subidx,
        NULL,
        push.total);
}

static trellis_status vk_skip_repeat_device(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const trellis_sparse_c2s_device_map * map,
    trellis_sparse_buffer * y,
    int64_t n_out,
    int in_channels,
    int out_channels) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    const sparse_vk_c2s_cache_entry * entry = vk_c2s_entry_from_map(vk, map);
    if (entry == NULL || x == NULL || y == NULL || n_out != entry->n ||
        in_channels <= 0 || out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    sparse_vk_push push = {
        0, (uint32_t) n_out, (uint32_t) in_channels, (uint32_t) out_channels,
        (uint32_t) ((uint64_t) n_out * (uint64_t) out_channels), 0, 0, 0.0f,
    };
    return vk_dispatch(
        vk,
        SPARSE_VK_PIPE_SKIP_REPEAT,
        &push,
        (trellis_sparse_buffer *) x,
        NULL,
        NULL,
        y,
        entry->parent,
        entry->subidx,
        NULL,
        push.total);
}

static trellis_status vk_download_c2s_coords(
    trellis_sparse_backend * backend,
    const trellis_sparse_c2s_device_map * map,
    int32_t * coords_out,
    int64_t n) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    const sparse_vk_c2s_cache_entry * entry = vk_c2s_entry_from_map(vk, map);
    if (entry == NULL || coords_out == NULL || n != entry->n) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return vk_download_bytes(vk, entry->coords, coords_out, (size_t) n * 4u * sizeof(int32_t));
}

static trellis_status vk_download_c2s_map(
    trellis_sparse_backend * backend,
    const trellis_sparse_c2s_device_map * map,
    int32_t * coords_out,
    int32_t * parent_out,
    int32_t * subidx_out,
    int64_t n) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    const sparse_vk_c2s_cache_entry * entry = vk_c2s_entry_from_map(vk, map);
    if (entry == NULL || coords_out == NULL || parent_out == NULL || subidx_out == NULL || n != entry->n) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_status status = vk_download_bytes(vk, entry->coords, coords_out, (size_t) n * 4u * sizeof(int32_t));
    if (status == TRELLIS_STATUS_OK) {
        status = vk_download_bytes(vk, entry->parent, parent_out, (size_t) n * sizeof(int32_t));
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_download_bytes(vk, entry->subidx, subidx_out, (size_t) n * sizeof(int32_t));
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_add_c2s_cache_entry(
            vk,
            coords_out,
            parent_out,
            subidx_out,
            n,
            entry->coords,
            entry->parent,
            entry->subidx,
            0,
            NULL);
    }
    return status;
}

static void vk_destroy(trellis_sparse_backend * backend) {
    if (backend == NULL) {
        return;
    }
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    if (vk->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(vk->device);
        vk_reclaim_pending(vk);
    }
    vk->destroying = 1;
    vk_clear_c2s_cache(vk);
    sparse_vk_weight * w = vk->weights;
    while (w != NULL) {
        sparse_vk_weight * next = w->next;
        vk_free_buffer(backend, w->buffer);
        free(w);
        w = next;
    }
    vk_free_buffer(backend, vk->dummy);
    while (vk->free_buffers != NULL) {
        trellis_sparse_buffer * next = vk->free_buffers->next_free;
        vk_destroy_buffer_raw(vk, vk->free_buffers);
        vk->free_buffers = next;
    }
    while (vk->free_commands != NULL) {
        sparse_vk_command_node * next = vk->free_commands->next;
        vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &vk->free_commands->command);
        free(vk->free_commands);
        vk->free_commands = next;
    }
    while (vk->free_descriptors != NULL) {
        sparse_vk_descriptor_node * next = vk->free_descriptors->next;
        vkFreeDescriptorSets(vk->device, vk->descriptor_pool, 1, &vk->free_descriptors->set);
        free(vk->free_descriptors);
        vk->free_descriptors = next;
    }
    for (uint32_t i = 0; i < (uint32_t) SPARSE_VK_PIPE_COUNT; ++i) {
        if (vk->pipelines[i] != VK_NULL_HANDLE) {
            vkDestroyPipeline(vk->device, vk->pipelines[i], NULL);
        }
    }
    if (vk->pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(vk->device, vk->pipeline_layout, NULL);
    if (vk->descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk->device, vk->descriptor_set_layout, NULL);
    if (vk->descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(vk->device, vk->descriptor_pool, NULL);
    if (vk->command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(vk->device, vk->command_pool, NULL);
    if (vk->device != VK_NULL_HANDLE) vkDestroyDevice(vk->device, NULL);
    if (vk->instance != VK_NULL_HANDLE) vkDestroyInstance(vk->instance, NULL);
    free(vk);
}

#include "trellis_pixal_naf_vulkan.inc"

static const trellis_sparse_backend_ops g_vk_ops = {
    vk_destroy,
    vk_alloc_f32,
    vk_upload_f32,
    vk_upload_i32,
    vk_download_f32,
    vk_free_buffer,
    vk_linear,
    vk_linear_silu,
    vk_row_norm,
    vk_row_norm_silu,
    vk_silu_inplace,
    vk_add,
    vk_build_rulebook,
    vk_sparse_conv3d,
    vk_free_rulebook,
    vk_c2s_gather,
    vk_skip_repeat,
    vk_build_c2s_map,
    vk_alias_c2s_map,
    vk_build_c2s_map_device,
    vk_build_rulebook_for_c2s_map,
    vk_c2s_gather_device,
    vk_skip_repeat_device,
    vk_download_c2s_coords,
    vk_download_c2s_map,
    vk_trim,
    vk_pixal_naf_attention_project_sparse,
};

static trellis_status vk_create_pipeline_from_spv(
    trellis_sparse_vk_backend * vk,
    const unsigned char * spv,
    unsigned int spv_len,
    VkPipeline * pipeline_out) {
    if (vk == NULL || spv == NULL || spv_len == 0 || pipeline_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    VkShaderModuleCreateInfo shader_info;
    memset(&shader_info, 0, sizeof(shader_info));
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spv_len;
    shader_info.pCode = (const uint32_t *) spv;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkResult result = vkCreateShaderModule(vk->device, &shader_info, NULL, &shader);
    if (result != VK_SUCCESS) return vk_status(result);

    VkPipelineShaderStageCreateInfo stage;
    memset(&stage, 0, sizeof(stage));
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = vk->pipeline_layout;
    result = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, pipeline_out);
    vkDestroyShaderModule(vk->device, shader, NULL);
    return vk_status(result);
}

typedef struct sparse_vk_pipeline_spec {
    sparse_vk_pipeline_id id;
    const unsigned char * spv;
    unsigned int spv_len;
    int requires_coopmat;
} sparse_vk_pipeline_spec;

static const sparse_vk_pipeline_spec g_sparse_vk_pipeline_specs[] = {
    { SPARSE_VK_PIPE_ROW_NORM, trellis_sparse_vk_row_norm_spv, sizeof(trellis_sparse_vk_row_norm_spv), 0 },
    { SPARSE_VK_PIPE_ROW_NORM_SILU, trellis_sparse_vk_row_norm_silu_spv, sizeof(trellis_sparse_vk_row_norm_silu_spv), 0 },
    { SPARSE_VK_PIPE_SILU, trellis_sparse_vk_silu_spv, sizeof(trellis_sparse_vk_silu_spv), 0 },
    { SPARSE_VK_PIPE_ADD, trellis_sparse_vk_add_spv, sizeof(trellis_sparse_vk_add_spv), 0 },
    { SPARSE_VK_PIPE_C2S_GATHER, trellis_sparse_vk_c2s_gather_spv, sizeof(trellis_sparse_vk_c2s_gather_spv), 0 },
    { SPARSE_VK_PIPE_SKIP_REPEAT, trellis_sparse_vk_skip_repeat_spv, sizeof(trellis_sparse_vk_skip_repeat_spv), 0 },
    { SPARSE_VK_PIPE_FILL_BIAS, trellis_sparse_vk_fill_bias_spv, sizeof(trellis_sparse_vk_fill_bias_spv), 0 },
    {
        SPARSE_VK_PIPE_RULEBOOK_HASH_INSERT,
        trellis_sparse_vk_rulebook_hash_insert_spv,
        sizeof(trellis_sparse_vk_rulebook_hash_insert_spv),
        0,
    },
    { SPARSE_VK_PIPE_RULEBOOK_FILL, trellis_sparse_vk_rulebook_fill_spv, sizeof(trellis_sparse_vk_rulebook_fill_spv), 0 },
    { SPARSE_VK_PIPE_C2S_COUNT, trellis_sparse_vk_c2s_count_spv, sizeof(trellis_sparse_vk_c2s_count_spv), 0 },
    { SPARSE_VK_PIPE_C2S_FILL, trellis_sparse_vk_c2s_fill_spv, sizeof(trellis_sparse_vk_c2s_fill_spv), 0 },
    { SPARSE_VK_PIPE_SCAN_I32_STRIDE, trellis_sparse_vk_scan_i32_stride_spv, sizeof(trellis_sparse_vk_scan_i32_stride_spv), 0 },
    { SPARSE_VK_PIPE_SCAN_I32_BLOCK, trellis_sparse_vk_scan_i32_block_spv, sizeof(trellis_sparse_vk_scan_i32_block_spv), 0 },
    {
        SPARSE_VK_PIPE_SCAN_I32_ADD_BLOCK_OFFSETS,
        trellis_sparse_vk_scan_i32_add_block_offsets_spv,
        sizeof(trellis_sparse_vk_scan_i32_add_block_offsets_spv),
        0,
    },
    {
        SPARSE_VK_PIPE_RULEBOOK_DISPATCH,
        trellis_sparse_vk_rulebook_dispatch_spv,
        sizeof(trellis_sparse_vk_rulebook_dispatch_spv),
        0,
    },
    {
        SPARSE_VK_PIPE_RULEBOOK_TILE_VALID,
        trellis_sparse_vk_rulebook_tile_valid_spv,
        sizeof(trellis_sparse_vk_rulebook_tile_valid_spv),
        0,
    },
    {
        SPARSE_VK_PIPE_RULEBOOK_MASK_INIT,
        trellis_sparse_vk_rulebook_mask_init_spv,
        sizeof(trellis_sparse_vk_rulebook_mask_init_spv),
        0,
    },
    { SPARSE_VK_PIPE_SORT_BITONIC, trellis_sparse_vk_sort_bitonic_spv, sizeof(trellis_sparse_vk_sort_bitonic_spv), 0 },
    {
        SPARSE_VK_PIPE_RULEBOOK_VALID_SORTED_COUNT,
        trellis_sparse_vk_rulebook_valid_sorted_count_spv,
        sizeof(trellis_sparse_vk_rulebook_valid_sorted_count_spv),
        0,
    },
    {
        SPARSE_VK_PIPE_RULEBOOK_VALID_SORTED_SCATTER,
        trellis_sparse_vk_rulebook_valid_sorted_scatter_spv,
        sizeof(trellis_sparse_vk_rulebook_valid_sorted_scatter_spv),
        0,
    },
    { SPARSE_VK_PIPE_LINEAR_MAT, trellis_sparse_vk_linear_mat_spv, sizeof(trellis_sparse_vk_linear_mat_spv), 0 },
    { SPARSE_VK_PIPE_LINEAR_SILU_MAT, trellis_sparse_vk_linear_silu_mat_spv, sizeof(trellis_sparse_vk_linear_silu_mat_spv), 0 },
    {
        SPARSE_VK_PIPE_SPARSE_CONV_OFFSET_MAT,
        trellis_sparse_vk_sparse_conv_offset_mat_spv,
        sizeof(trellis_sparse_vk_sparse_conv_offset_mat_spv),
        0,
    },
    {
        SPARSE_VK_PIPE_SPARSE_CONV_MASKED_MAT,
        trellis_sparse_vk_sparse_conv_masked_mat_spv,
        sizeof(trellis_sparse_vk_sparse_conv_masked_mat_spv),
        0,
    },
    {
        SPARSE_VK_PIPE_SPARSE_CONV_MASKED_SORTED_MAT,
        trellis_sparse_vk_sparse_conv_masked_sorted_mat_spv,
        sizeof(trellis_sparse_vk_sparse_conv_masked_sorted_mat_spv),
        0,
    },
    { SPARSE_VK_PIPE_LINEAR_COOP, trellis_sparse_vk_linear_coop_spv, sizeof(trellis_sparse_vk_linear_coop_spv), 1 },
    {
        SPARSE_VK_PIPE_LINEAR_SILU_COOP,
        trellis_sparse_vk_linear_silu_coop_spv,
        sizeof(trellis_sparse_vk_linear_silu_coop_spv),
        1,
    },
    {
        SPARSE_VK_PIPE_SPARSE_CONV_OFFSET_COOP,
        trellis_sparse_vk_sparse_conv_offset_coop_spv,
        sizeof(trellis_sparse_vk_sparse_conv_offset_coop_spv),
        1,
    },
    {
        SPARSE_VK_PIPE_PIXAL_NAF_ATTENTION_PROJECT,
        trellis_sparse_vk_pixal_naf_attention_project_spv,
        sizeof(trellis_sparse_vk_pixal_naf_attention_project_spv),
        0,
    },
};

static trellis_status vk_create_pipelines(trellis_sparse_vk_backend * vk) {
    if (vk == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < (uint32_t) (sizeof(g_sparse_vk_pipeline_specs) / sizeof(g_sparse_vk_pipeline_specs[0])); ++i) {
        const sparse_vk_pipeline_spec * spec = &g_sparse_vk_pipeline_specs[i];
        if (spec->requires_coopmat && !vk->coopmat_supported) {
            continue;
        }
        trellis_status status = vk_create_pipeline_from_spv(
            vk,
            spec->spv,
            spec->spv_len,
            &vk->pipelines[spec->id]);
        if (status != TRELLIS_STATUS_OK) {
            if (!spec->requires_coopmat) {
                return status;
            }
            vk->coopmat_supported = 0;
            for (uint32_t p = (uint32_t) SPARSE_VK_PIPE_LINEAR_COOP; p < (uint32_t) SPARSE_VK_PIPE_COUNT; ++p) {
                if (vk->pipelines[p] != VK_NULL_HANDLE) {
                    vkDestroyPipeline(vk->device, vk->pipelines[p], NULL);
                    vk->pipelines[p] = VK_NULL_HANDLE;
                }
            }
            return TRELLIS_STATUS_OK;
        }
    }
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_sparse_vulkan_backend_create(int device, trellis_sparse_backend ** out) {
    if (out == NULL || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) calloc(1, sizeof(*vk));
    if (vk == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    vk->base.ops = &g_vk_ops;
    vk->base.kind = TRELLIS_SPARSE_BACKEND_VULKAN;
    vk->base.device = device;
    vk->use_sparse_conv_indirect = !vk_env_disabled("TRELLIS_VK_SPARSE_CONV_INDIRECT");

    VkApplicationInfo app;
    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "trellis2.c sparse";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    VkResult result = vkCreateInstance(&instance_info, NULL, &vk->instance);
    if (result != VK_SUCCESS) {
        vk_destroy(&vk->base);
        return vk_status(result);
    }
    uint32_t physical_count = 0;
    vkEnumeratePhysicalDevices(vk->instance, &physical_count, NULL);
    if (physical_count == 0 || (uint32_t) device >= physical_count) {
        vk_destroy(&vk->base);
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }
    VkPhysicalDevice * physical = (VkPhysicalDevice *) malloc((size_t) physical_count * sizeof(VkPhysicalDevice));
    if (physical == NULL) {
        vk_destroy(&vk->base);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    vkEnumeratePhysicalDevices(vk->instance, &physical_count, physical);
    vk->physical_device = physical[device];
    free(physical);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(vk->physical_device, &props);
    vk->max_workgroup_count[0] = props.limits.maxComputeWorkGroupCount[0];
    vk->max_workgroup_count[1] = props.limits.maxComputeWorkGroupCount[1];
    vk->max_workgroup_count[2] = props.limits.maxComputeWorkGroupCount[2];
    vk->coopmat_supported = vk_physical_device_supports_sparse_coopmat(vk->instance, vk->physical_device);

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->physical_device, &family_count, NULL);
    VkQueueFamilyProperties * families = (VkQueueFamilyProperties *) malloc((size_t) family_count * sizeof(*families));
    if (families == NULL) {
        vk_destroy(&vk->base);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(vk->physical_device, &family_count, families);
    vk->queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            vk->queue_family = i;
            break;
        }
    }
    free(families);
    if (vk->queue_family == UINT32_MAX) {
        vk_destroy(&vk->base);
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = vk->queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    const char * coop_extensions[] = {
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
    };
    VkPhysicalDeviceFeatures2 enabled_features2;
    VkPhysicalDeviceVulkan11Features enabled_features11;
    VkPhysicalDeviceVulkan12Features enabled_features12;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR enabled_coop_features;
    memset(&enabled_features2, 0, sizeof(enabled_features2));
    memset(&enabled_features11, 0, sizeof(enabled_features11));
    memset(&enabled_features12, 0, sizeof(enabled_features12));
    memset(&enabled_coop_features, 0, sizeof(enabled_coop_features));
    if (vk->coopmat_supported) {
        enabled_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        enabled_features2.pNext = &enabled_features11;
        enabled_features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        enabled_features11.storageBuffer16BitAccess = VK_TRUE;
        enabled_features11.pNext = &enabled_features12;
        enabled_features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        enabled_features12.shaderFloat16 = VK_TRUE;
        enabled_features12.pNext = &enabled_coop_features;
        enabled_coop_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        enabled_coop_features.cooperativeMatrix = VK_TRUE;
        device_info.enabledExtensionCount = (uint32_t) (sizeof(coop_extensions) / sizeof(coop_extensions[0]));
        device_info.ppEnabledExtensionNames = coop_extensions;
        device_info.pNext = &enabled_features2;
    }
    result = vkCreateDevice(vk->physical_device, &device_info, NULL, &vk->device);
    if (result != VK_SUCCESS) {
        vk_destroy(&vk->base);
        return vk_status(result);
    }
    vkGetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);

    VkCommandPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = vk->queue_family;
    result = vkCreateCommandPool(vk->device, &pool_info, NULL, &vk->command_pool);
    if (result != VK_SUCCESS) {
        vk_destroy(&vk->base);
        return vk_status(result);
    }

    VkDescriptorSetLayoutBinding bindings[VK_BINDING_COUNT];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < VK_BINDING_COUNT; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = VK_BINDING_COUNT;
    layout_info.pBindings = bindings;
    result = vkCreateDescriptorSetLayout(vk->device, &layout_info, NULL, &vk->descriptor_set_layout);
    if (result != VK_SUCCESS) {
        vk_destroy(&vk->base);
        return vk_status(result);
    }

    VkPushConstantRange push_range;
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(sparse_vk_push);
    VkPipelineLayoutCreateInfo pipeline_layout_info;
    memset(&pipeline_layout_info, 0, sizeof(pipeline_layout_info));
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &vk->descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    result = vkCreatePipelineLayout(vk->device, &pipeline_layout_info, NULL, &vk->pipeline_layout);
    if (result != VK_SUCCESS) {
        vk_destroy(&vk->base);
        return vk_status(result);
    }
    trellis_status status = vk_create_pipelines(vk);
    if (status != TRELLIS_STATUS_OK) {
        vk_destroy(&vk->base);
        return status;
    }
    if (vk_env_enabled("TRELLIS_VK_SPARSE_CONV_COOPMAT")) {
        TRELLIS_INFO(
            "sparse vulkan: cooperative matrix sparse conv %s",
            vk->coopmat_supported && vk->pipelines[SPARSE_VK_PIPE_SPARSE_CONV_OFFSET_COOP] != VK_NULL_HANDLE ?
                "enabled" :
                "unavailable");
    }

    VkDescriptorPoolSize pool_size;
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = VK_BINDING_COUNT * 4096;
    VkDescriptorPoolCreateInfo descriptor_pool_info;
    memset(&descriptor_pool_info, 0, sizeof(descriptor_pool_info));
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptor_pool_info.maxSets = 4096;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &pool_size;
    result = vkCreateDescriptorPool(vk->device, &descriptor_pool_info, NULL, &vk->descriptor_pool);
    if (result != VK_SUCCESS) {
        vk_destroy(&vk->base);
        return vk_status(result);
    }
    status = vk_alloc_bytes(vk, 4, &vk->dummy);
    if (status != TRELLIS_STATUS_OK) {
        vk_destroy(&vk->base);
        return status;
    }
    *out = &vk->base;
    return TRELLIS_STATUS_OK;
}

#else

trellis_status trellis_sparse_vulkan_backend_create(int device, trellis_sparse_backend ** out) {
    (void) device;
    if (out == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = 0;
    return TRELLIS_STATUS_NOT_IMPLEMENTED;
}

#endif
