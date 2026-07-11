cmake_minimum_required(VERSION 3.18)

if (NOT DEFINED TRELLIS2_C_BACKEND)
    if (DEFINED TRELLIS2_C_GGML_BACKEND)
        set(TRELLIS2_C_BACKEND "${TRELLIS2_C_GGML_BACKEND}" CACHE STRING "Unified graph+sparse backend: cuda or vulkan" FORCE)
    else()
        set(TRELLIS2_C_BACKEND "cuda" CACHE STRING "Unified graph+sparse backend: cuda or vulkan")
    endif()
endif()
set_property(CACHE TRELLIS2_C_BACKEND PROPERTY STRINGS cuda vulkan)
if (NOT TRELLIS2_C_BACKEND MATCHES "^(cuda|vulkan)$")
    message(FATAL_ERROR "TRELLIS2_C_BACKEND must be one of: cuda, vulkan")
endif()

if (TRELLIS2_C_BACKEND STREQUAL "cuda")
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures for trellis2.c kernels" FORCE)
    endif()
    project(trellis2_c LANGUAGES C CXX CUDA)
else()
    project(trellis2_c LANGUAGES C CXX)
endif()

set(TRELLIS2_C_GGML_BACKEND "${TRELLIS2_C_BACKEND}" CACHE STRING "Deprecated; use TRELLIS2_C_BACKEND" FORCE)
set_property(CACHE TRELLIS2_C_GGML_BACKEND PROPERTY STRINGS cuda vulkan)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
if (TRELLIS2_C_BACKEND STREQUAL "cuda")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

option(TRELLIS2_C_BUILD_TESTS "Build trellis2.c tests" ON)
option(TRELLIS2_C_BUILD_TOOLS "Build trellis2.c command line tools" ON)
option(TRELLIS2_C_BUILD_RAYLIB_VIEWER "Build the raylib GUI" ON)
option(TRELLIS2_C_ENABLE_TENSOR_CORES "Prefer tensor-core capable kernels when 16-bit data and hardware support are available" ON)
option(TRELLIS2_C_ENABLE_GLTF_VULKAN_BAKE "Build Vulkan glTF texture baking independent of the inference backend" ON)

if (TRELLIS2_C_BACKEND STREQUAL "cuda")
    set(GGML_CUDA ON CACHE BOOL "Build ggml CUDA backend" FORCE)
    set(GGML_VULKAN OFF CACHE BOOL "Build ggml Vulkan backend" FORCE)
    if (TRELLIS2_C_ENABLE_TENSOR_CORES)
        set(GGML_CUDA_FORCE_CUBLAS ON CACHE BOOL "Prefer cuBLAS tensor-op GEMMs for ggml CUDA matmul" FORCE)
        set(GGML_CUDA_FORCE_MMQ OFF CACHE BOOL "Do not force ggml CUDA MMQ kernels when tensor cores are preferred" FORCE)
        set(GGML_CUDA_FA ON CACHE BOOL "Build ggml CUDA FlashAttention kernels" FORCE)
    else()
        set(GGML_CUDA_FORCE_CUBLAS OFF CACHE BOOL "Prefer cuBLAS tensor-op GEMMs for ggml CUDA matmul" FORCE)
        set(GGML_CUDA_FORCE_MMQ OFF CACHE BOOL "Do not force ggml CUDA MMQ kernels when tensor cores are preferred" FORCE)
    endif()
else()
    set(GGML_CUDA OFF CACHE BOOL "Build ggml CUDA backend" FORCE)
    set(GGML_VULKAN ON CACHE BOOL "Build ggml Vulkan backend" FORCE)
    set(GGML_CUDA_FORCE_CUBLAS OFF CACHE BOOL "Prefer cuBLAS tensor-op GEMMs for ggml CUDA matmul" FORCE)
    set(GGML_CUDA_FORCE_MMQ OFF CACHE BOOL "Do not force ggml CUDA MMQ kernels when tensor cores are preferred" FORCE)
endif()
set(GGML_NATIVE OFF CACHE BOOL "Avoid host-specific codegen in the embedded ggml build" FORCE)
set(GGML_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
message(STATUS "TRELLIS2_C_ENABLE_TENSOR_CORES=${TRELLIS2_C_ENABLE_TENSOR_CORES}")

set(TRELLIS2_C_HAS_VKMESH_C_API OFF)
if (TRELLIS2_C_BUILD_TOOLS AND TRELLIS2_C_BACKEND STREQUAL "vulkan")
    set(TRELLIS2_C_HAS_VKMESH_C_API ON)
endif()

set(TRELLIS2_C_HAS_GLTF_VULKAN_BAKE OFF)
if (TRELLIS2_C_ENABLE_GLTF_VULKAN_BAKE)
    set(TRELLIS2_C_HAS_GLTF_VULKAN_BAKE ON)
endif()
message(STATUS "TRELLIS2_C_ENABLE_GLTF_VULKAN_BAKE=${TRELLIS2_C_ENABLE_GLTF_VULKAN_BAKE}")

if (GGML_VULKAN OR TRELLIS2_C_HAS_GLTF_VULKAN_BAKE)
    find_package(Vulkan REQUIRED)
    if (NOT Vulkan_GLSLC_EXECUTABLE)
        find_program(Vulkan_GLSLC_EXECUTABLE glslc REQUIRED)
    endif()
endif()

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/3rd/ggml" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

add_library(xatlas STATIC
    3rd/xatlas/source/xatlas/xatlas.cpp
)
target_include_directories(xatlas PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/3rd/xatlas/source/xatlas"
)
target_compile_definitions(xatlas PRIVATE XATLAS_C_API=1)

if (TRELLIS2_C_BUILD_RAYLIB_VIEWER AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/3rd/raylib/CMakeLists.txt")
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/3rd/raylib" "${CMAKE_CURRENT_BINARY_DIR}/raylib")
endif()

set(TRELLIS2_C_SOURCES
    src/runtime/trellis_runtime.c
    src/runtime/trellis_model_package.c
    src/runtime/trellis_registry.c
    src/tasks/image_to_3d/adapter.c
    src/tasks/image_to_3d/adapters/trellis2.c
    src/tasks/image_to_3d/adapters/pixal3d.c
    src/ops/sparse/guides.c
    src/architectures/sparse_structure_decoder/decoder.c
    src/architectures/dinov3/dinov3.c
    src/architectures/pixal_naf/pixal_naf.c
    src/architectures/birefnet/birefnet.c
    src/ops/mesh/flex_dual_grid.c
    src/architectures/dit_flow/dit_flow.c
    src/ops/ggml/ggml_layers.c
    src/tasks/image_to_3d/image_to_3d.c
    src/tasks/image_to_3d/model_cache.c
    src/architectures/dit_flow/dit_flow_executor.c
    src/ops/sampling/flow_sampler.c
    src/ops/projection/pixal_projection.c
    src/ops/projected_attention/projected_attention.c
    src/tasks/image_to_3d/stages/sparse_structure.c
    src/tasks/image_to_3d/stages/structured_latent.c
    src/tasks/image_to_3d/export/gltf.c
    src/weights/safetensors.c
    src/architectures/sparse_unet_decoder/decoder.c
    src/architectures/sparse_unet_decoder/executor.c
    src/ops/sparse/trellis_sparse_cpu.c
    src/ops/sparse/trellis_pixal_naf_backend.c
    src/weights/tensor_store.c
)

if (NOT TRELLIS2_C_HAS_VKMESH_C_API)
    list(APPEND TRELLIS2_C_SOURCES
        src/ops/mesh/vkmesh_stubs.c
    )
endif()

if (TRELLIS2_C_BACKEND STREQUAL "cuda")
    list(APPEND TRELLIS2_C_SOURCES
        src/ops/sparse/trellis_sparse_vulkan_stubs.c
        src/ops/sparse/cuda/kernels.cu
        src/ops/sparse/cuda/forward.cu
        src/ops/projected_attention/projected_attention_cuda.cu
    )
else()
    list(APPEND TRELLIS2_C_SOURCES
        src/ops/sparse/trellis_sparse_vulkan.c
        src/ops/sparse/cuda/stubs.c
        src/ops/projected_attention/projected_attention_cuda_stubs.c
    )
endif()

add_library(trellis2_c
    ${TRELLIS2_C_SOURCES}
)
if (WIN32)
    set_target_properties(trellis2_c PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if (GGML_VULKAN)
    set(TRELLIS_SPARSE_SHADER_DIR "${CMAKE_CURRENT_LIST_DIR}/src/ops/sparse/shaders")
    set(TRELLIS_SPARSE_SHADER_COMMON_DEPS
        "${TRELLIS_SPARSE_SHADER_DIR}/sparse_common.inc"
        "${TRELLIS_SPARSE_SHADER_DIR}/sparse_mat_common.inc"
        "${TRELLIS_SPARSE_SHADER_DIR}/sparse_coop_common.inc")
    set(TRELLIS_SPARSE_VK_HEADERS)
    function(add_trellis_sparse_vk_shader NAME SOURCE SYMBOL TARGET_ENV)
        set(SPV "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv")
        set(HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/${SYMBOL}.h")
        add_custom_command(
            OUTPUT "${SPV}"
            COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=${TARGET_ENV}
                    -I "${TRELLIS_SPARSE_SHADER_DIR}"
                    "${CMAKE_CURRENT_LIST_DIR}/${SOURCE}"
                    -o "${SPV}"
            DEPENDS "${CMAKE_CURRENT_LIST_DIR}/${SOURCE}" ${TRELLIS_SPARSE_SHADER_COMMON_DEPS}
            COMMENT "Compile ${NAME} Vulkan shader")
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
            COMMAND "${CMAKE_COMMAND}"
                    -DINPUT="${SPV}"
                    -DOUTPUT="${HEADER}"
                    -DSYMBOL=${SYMBOL}
                    -P "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
            DEPENDS "${SPV}" "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
            COMMENT "Generate ${NAME} Vulkan shader header")
        list(APPEND TRELLIS_SPARSE_VK_HEADERS "${HEADER}")
        set(TRELLIS_SPARSE_VK_HEADERS "${TRELLIS_SPARSE_VK_HEADERS}" PARENT_SCOPE)
    endfunction()

    add_trellis_sparse_vk_shader(trellis_sparse_vk_row_norm src/ops/sparse/shaders/row_norm.comp trellis_sparse_vk_row_norm_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_row_norm_silu src/ops/sparse/shaders/row_norm_silu.comp trellis_sparse_vk_row_norm_silu_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_silu src/ops/sparse/shaders/silu.comp trellis_sparse_vk_silu_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_add src/ops/sparse/shaders/add.comp trellis_sparse_vk_add_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_c2s_gather src/ops/sparse/shaders/c2s_gather.comp trellis_sparse_vk_c2s_gather_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_skip_repeat src/ops/sparse/shaders/skip_repeat.comp trellis_sparse_vk_skip_repeat_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_fill_bias src/ops/sparse/shaders/fill_bias.comp trellis_sparse_vk_fill_bias_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_rulebook_hash_insert src/ops/sparse/shaders/rulebook_hash_insert.comp trellis_sparse_vk_rulebook_hash_insert_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_rulebook_fill src/ops/sparse/shaders/rulebook_fill.comp trellis_sparse_vk_rulebook_fill_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_c2s_count src/ops/sparse/shaders/c2s_count.comp trellis_sparse_vk_c2s_count_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_c2s_fill src/ops/sparse/shaders/c2s_fill.comp trellis_sparse_vk_c2s_fill_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_scan_i32_stride src/ops/sparse/shaders/scan_i32_stride.comp trellis_sparse_vk_scan_i32_stride_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_scan_i32_block src/ops/sparse/shaders/scan_i32_block.comp trellis_sparse_vk_scan_i32_block_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_scan_i32_add_block_offsets src/ops/sparse/shaders/scan_i32_add_block_offsets.comp trellis_sparse_vk_scan_i32_add_block_offsets_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_rulebook_dispatch src/ops/sparse/shaders/rulebook_dispatch.comp trellis_sparse_vk_rulebook_dispatch_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_rulebook_tile_valid src/ops/sparse/shaders/rulebook_tile_valid.comp trellis_sparse_vk_rulebook_tile_valid_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_rulebook_mask_init src/ops/sparse/shaders/rulebook_mask_init.comp trellis_sparse_vk_rulebook_mask_init_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_sort_bitonic src/ops/sparse/shaders/sort_bitonic.comp trellis_sparse_vk_sort_bitonic_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_rulebook_valid_sorted_count src/ops/sparse/shaders/rulebook_valid_sorted_count.comp trellis_sparse_vk_rulebook_valid_sorted_count_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_rulebook_valid_sorted_scatter src/ops/sparse/shaders/rulebook_valid_sorted_scatter.comp trellis_sparse_vk_rulebook_valid_sorted_scatter_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_linear_mat src/ops/sparse/shaders/linear_mat.comp trellis_sparse_vk_linear_mat_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_linear_silu_mat src/ops/sparse/shaders/linear_silu_mat.comp trellis_sparse_vk_linear_silu_mat_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_sparse_conv_offset_mat src/ops/sparse/shaders/sparse_conv_offset_mat.comp trellis_sparse_vk_sparse_conv_offset_mat_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_sparse_conv_masked_mat src/ops/sparse/shaders/sparse_conv_masked_mat.comp trellis_sparse_vk_sparse_conv_masked_mat_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_sparse_conv_masked_sorted_mat src/ops/sparse/shaders/sparse_conv_masked_sorted_mat.comp trellis_sparse_vk_sparse_conv_masked_sorted_mat_spv vulkan1.2)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_linear_coop src/ops/sparse/shaders/linear_coop.comp trellis_sparse_vk_linear_coop_spv vulkan1.3)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_linear_silu_coop src/ops/sparse/shaders/linear_silu_coop.comp trellis_sparse_vk_linear_silu_coop_spv vulkan1.3)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_sparse_conv_offset_coop src/ops/sparse/shaders/sparse_conv_offset_coop.comp trellis_sparse_vk_sparse_conv_offset_coop_spv vulkan1.3)
    add_trellis_sparse_vk_shader(trellis_sparse_vk_pixal_naf_attention_project src/ops/sparse/shaders/pixal_naf_attention_project.comp trellis_sparse_vk_pixal_naf_attention_project_spv vulkan1.2)

    set(TRELLIS_GLTF_BAKE_VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/trellis_gltf_bake_vert.spv")
    set(TRELLIS_GLTF_BAKE_VERT_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/trellis_gltf_bake_vert_spv.h")
    set(TRELLIS_GLTF_BAKE_FRAG_SPV "${CMAKE_CURRENT_BINARY_DIR}/trellis_gltf_bake_frag.spv")
    set(TRELLIS_GLTF_BAKE_FRAG_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/trellis_gltf_bake_frag_spv.h")
    set(TRELLIS_GLTF_DILATE_SPV "${CMAKE_CURRENT_BINARY_DIR}/trellis_gltf_dilate.spv")
    set(TRELLIS_GLTF_DILATE_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/trellis_gltf_dilate_spv.h")
    set(TRELLIS_GLTF_FILL_EMPTY_SPV "${CMAKE_CURRENT_BINARY_DIR}/trellis_gltf_fill_empty.spv")
    set(TRELLIS_GLTF_FILL_EMPTY_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/trellis_gltf_fill_empty_spv.h")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_BAKE_VERT_SPV}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=vertex --target-env=vulkan1.2
                "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_bake.vert"
                -o "${TRELLIS_GLTF_BAKE_VERT_SPV}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_bake.vert"
        COMMENT "Compile TRELLIS glTF Vulkan texture bake vertex shader")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_BAKE_FRAG_SPV}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=fragment --target-env=vulkan1.2
                "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_bake.frag"
                -o "${TRELLIS_GLTF_BAKE_FRAG_SPV}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_bake.frag"
        COMMENT "Compile TRELLIS glTF Vulkan texture bake fragment shader")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_DILATE_SPV}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2
                "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_dilate.comp"
                -o "${TRELLIS_GLTF_DILATE_SPV}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_dilate.comp"
        COMMENT "Compile TRELLIS glTF Vulkan texture dilation shader")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_FILL_EMPTY_SPV}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2
                "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_fill_empty.comp"
                -o "${TRELLIS_GLTF_FILL_EMPTY_SPV}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_fill_empty.comp"
        COMMENT "Compile TRELLIS glTF Vulkan texture fill shader")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_BAKE_VERT_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT="${TRELLIS_GLTF_BAKE_VERT_SPV}"
                -DOUTPUT="${TRELLIS_GLTF_BAKE_VERT_HEADER}"
                -DSYMBOL=trellis_gltf_bake_vert_spv
                -P "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        DEPENDS "${TRELLIS_GLTF_BAKE_VERT_SPV}" "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        COMMENT "Generate TRELLIS glTF Vulkan texture bake vertex shader header")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_BAKE_FRAG_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT="${TRELLIS_GLTF_BAKE_FRAG_SPV}"
                -DOUTPUT="${TRELLIS_GLTF_BAKE_FRAG_HEADER}"
                -DSYMBOL=trellis_gltf_bake_frag_spv
                -P "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        DEPENDS "${TRELLIS_GLTF_BAKE_FRAG_SPV}" "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        COMMENT "Generate TRELLIS glTF Vulkan texture bake fragment shader header")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_DILATE_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT="${TRELLIS_GLTF_DILATE_SPV}"
                -DOUTPUT="${TRELLIS_GLTF_DILATE_HEADER}"
                -DSYMBOL=trellis_gltf_dilate_spv
                -P "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        DEPENDS "${TRELLIS_GLTF_DILATE_SPV}" "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        COMMENT "Generate TRELLIS glTF Vulkan texture dilation shader header")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_FILL_EMPTY_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT="${TRELLIS_GLTF_FILL_EMPTY_SPV}"
                -DOUTPUT="${TRELLIS_GLTF_FILL_EMPTY_HEADER}"
                -DSYMBOL=trellis_gltf_fill_empty_spv
                -P "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        DEPENDS "${TRELLIS_GLTF_FILL_EMPTY_SPV}" "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        COMMENT "Generate TRELLIS glTF Vulkan texture fill shader header")
    add_custom_target(trellis_sparse_vk_shader DEPENDS
        ${TRELLIS_SPARSE_VK_HEADERS})
    add_custom_target(trellis_gltf_vk_shader DEPENDS
        "${TRELLIS_GLTF_BAKE_VERT_HEADER}"
        "${TRELLIS_GLTF_BAKE_FRAG_HEADER}"
        "${TRELLIS_GLTF_DILATE_HEADER}"
        "${TRELLIS_GLTF_FILL_EMPTY_HEADER}")
    add_dependencies(trellis2_c trellis_sparse_vk_shader)
    add_dependencies(trellis2_c trellis_gltf_vk_shader)
endif()

if (TRELLIS2_C_HAS_GLTF_VULKAN_BAKE AND NOT GGML_VULKAN)
    set(TRELLIS_GLTF_BAKE_VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/trellis_gltf_bake_vert.spv")
    set(TRELLIS_GLTF_BAKE_VERT_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/trellis_gltf_bake_vert_spv.h")
    set(TRELLIS_GLTF_BAKE_FRAG_SPV "${CMAKE_CURRENT_BINARY_DIR}/trellis_gltf_bake_frag.spv")
    set(TRELLIS_GLTF_BAKE_FRAG_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/trellis_gltf_bake_frag_spv.h")
    set(TRELLIS_GLTF_DILATE_SPV "${CMAKE_CURRENT_BINARY_DIR}/trellis_gltf_dilate.spv")
    set(TRELLIS_GLTF_DILATE_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/trellis_gltf_dilate_spv.h")
    set(TRELLIS_GLTF_FILL_EMPTY_SPV "${CMAKE_CURRENT_BINARY_DIR}/trellis_gltf_fill_empty.spv")
    set(TRELLIS_GLTF_FILL_EMPTY_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/trellis_gltf_fill_empty_spv.h")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_BAKE_VERT_SPV}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=vertex --target-env=vulkan1.2
                "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_bake.vert"
                -o "${TRELLIS_GLTF_BAKE_VERT_SPV}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_bake.vert"
        COMMENT "Compile TRELLIS glTF Vulkan texture bake vertex shader")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_BAKE_FRAG_SPV}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=fragment --target-env=vulkan1.2
                "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_bake.frag"
                -o "${TRELLIS_GLTF_BAKE_FRAG_SPV}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_bake.frag"
        COMMENT "Compile TRELLIS glTF Vulkan texture bake fragment shader")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_DILATE_SPV}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2
                "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_dilate.comp"
                -o "${TRELLIS_GLTF_DILATE_SPV}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_dilate.comp"
        COMMENT "Compile TRELLIS glTF Vulkan texture dilation shader")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_FILL_EMPTY_SPV}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2
                "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_fill_empty.comp"
                -o "${TRELLIS_GLTF_FILL_EMPTY_SPV}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/export/shaders/trellis_gltf_fill_empty.comp"
        COMMENT "Compile TRELLIS glTF Vulkan texture fill shader")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_BAKE_VERT_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT="${TRELLIS_GLTF_BAKE_VERT_SPV}"
                -DOUTPUT="${TRELLIS_GLTF_BAKE_VERT_HEADER}"
                -DSYMBOL=trellis_gltf_bake_vert_spv
                -P "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        DEPENDS "${TRELLIS_GLTF_BAKE_VERT_SPV}" "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        COMMENT "Generate TRELLIS glTF Vulkan texture bake vertex shader header")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_BAKE_FRAG_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT="${TRELLIS_GLTF_BAKE_FRAG_SPV}"
                -DOUTPUT="${TRELLIS_GLTF_BAKE_FRAG_HEADER}"
                -DSYMBOL=trellis_gltf_bake_frag_spv
                -P "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        DEPENDS "${TRELLIS_GLTF_BAKE_FRAG_SPV}" "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        COMMENT "Generate TRELLIS glTF Vulkan texture bake fragment shader header")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_DILATE_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT="${TRELLIS_GLTF_DILATE_SPV}"
                -DOUTPUT="${TRELLIS_GLTF_DILATE_HEADER}"
                -DSYMBOL=trellis_gltf_dilate_spv
                -P "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        DEPENDS "${TRELLIS_GLTF_DILATE_SPV}" "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        COMMENT "Generate TRELLIS glTF Vulkan texture dilation shader header")
    add_custom_command(
        OUTPUT "${TRELLIS_GLTF_FILL_EMPTY_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT="${TRELLIS_GLTF_FILL_EMPTY_SPV}"
                -DOUTPUT="${TRELLIS_GLTF_FILL_EMPTY_HEADER}"
                -DSYMBOL=trellis_gltf_fill_empty_spv
                -P "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        DEPENDS "${TRELLIS_GLTF_FILL_EMPTY_SPV}" "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        COMMENT "Generate TRELLIS glTF Vulkan texture fill shader header")
    add_custom_target(trellis_gltf_vk_shader DEPENDS
        "${TRELLIS_GLTF_BAKE_VERT_HEADER}"
        "${TRELLIS_GLTF_BAKE_FRAG_HEADER}"
        "${TRELLIS_GLTF_DILATE_HEADER}"
        "${TRELLIS_GLTF_FILL_EMPTY_HEADER}")
    add_dependencies(trellis2_c trellis_gltf_vk_shader)
endif()

target_include_directories(trellis2_c
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/runtime"
        "${CMAKE_CURRENT_LIST_DIR}/src/ops"
        "${CMAKE_CURRENT_LIST_DIR}/src/ops/sparse"
        "${CMAKE_CURRENT_LIST_DIR}/src/ops/sparse/cuda"
        "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d"
        "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/adapters"
        "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d/stages"
        "${CMAKE_CURRENT_LIST_DIR}/src/architectures/dit_flow"
        "${CMAKE_CURRENT_LIST_DIR}/src/architectures/pixal_naf"
        "${CMAKE_CURRENT_LIST_DIR}/src/ops/projected_attention"
        "${CMAKE_CURRENT_LIST_DIR}/src/ops/projection"
        "${CMAKE_CURRENT_LIST_DIR}/src/weights"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${CMAKE_CURRENT_LIST_DIR}/3rd/cgltf"
        "${CMAKE_CURRENT_LIST_DIR}/3rd/stb"
        "${CMAKE_CURRENT_LIST_DIR}/3rd/raylib/src/external"
        "${CMAKE_CURRENT_LIST_DIR}/3rd/xatlas/source/xatlas"
)

target_link_libraries(trellis2_c PUBLIC ggml xatlas)
if (TRELLIS2_C_BACKEND STREQUAL "cuda")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(trellis2_c PUBLIC CUDA::cudart CUDA::cublas)
endif()
if (GGML_VULKAN OR TRELLIS2_C_HAS_GLTF_VULKAN_BAKE)
    target_link_libraries(trellis2_c PRIVATE Vulkan::Vulkan)
endif()
if (UNIX)
    target_link_libraries(trellis2_c PUBLIC m)
endif()

if (TRELLIS2_C_BACKEND STREQUAL "cuda")
    set(TRELLIS_HAS_CUDA_DECODER_VALUE 1)
else()
    set(TRELLIS_HAS_CUDA_DECODER_VALUE 0)
endif()
if (TRELLIS2_C_HAS_GLTF_VULKAN_BAKE)
    set(TRELLIS_HAS_GLTF_VULKAN_BAKE_VALUE 1)
else()
    set(TRELLIS_HAS_GLTF_VULKAN_BAKE_VALUE 0)
endif()
if (TRELLIS2_C_HAS_VKMESH_C_API)
    set(TRELLIS_HAS_VKMESH_C_API_VALUE 1)
else()
    set(TRELLIS_HAS_VKMESH_C_API_VALUE 0)
endif()
target_compile_definitions(trellis2_c PUBLIC
    TRELLIS_DEFAULT_BACKEND="${TRELLIS2_C_BACKEND}"
    TRELLIS_DEFAULT_GGML_BACKEND="${TRELLIS2_C_BACKEND}"
    TRELLIS_HAS_CUDA_DECODER=${TRELLIS_HAS_CUDA_DECODER_VALUE}
    TRELLIS_HAS_GLTF_VULKAN_BAKE=${TRELLIS_HAS_GLTF_VULKAN_BAKE_VALUE}
    TRELLIS_HAS_VKMESH_C_API=${TRELLIS_HAS_VKMESH_C_API_VALUE}
)
if (GGML_CUDA)
    target_compile_definitions(trellis2_c PUBLIC GGML_USE_CUDA)
endif()
if (GGML_VULKAN)
    target_compile_definitions(trellis2_c PUBLIC GGML_USE_VULKAN)
    target_compile_definitions(trellis2_c PRIVATE TRELLIS_HAS_SPARSE_VULKAN=1)
endif()

if (TRELLIS2_C_BUILD_TESTS)
    enable_testing()

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        add_test(
            NAME trellis2_c_glb_verifier_tests
            COMMAND "${Python3_EXECUTABLE}" -m unittest -v tests/test_verify_glb.py)
        set_tests_properties(trellis2_c_glb_verifier_tests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")
    endif()

    add_executable(trellis2_c_model_package_tests
        tests/test_model_package.c
    )
    target_compile_definitions(trellis2_c_model_package_tests PRIVATE
        TRELLIS_TEST_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}")
    target_include_directories(trellis2_c_model_package_tests PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/runtime")
    target_link_libraries(trellis2_c_model_package_tests PRIVATE trellis2_c)
    add_test(NAME trellis2_c_model_package_tests COMMAND trellis2_c_model_package_tests)

    add_executable(trellis2_c_model_registry_tests
        tests/test_model_registry.c
    )
    target_include_directories(trellis2_c_model_registry_tests PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/runtime")
    target_link_libraries(trellis2_c_model_registry_tests PRIVATE trellis2_c)
    add_test(NAME trellis2_c_model_registry_tests COMMAND trellis2_c_model_registry_tests)

    add_executable(trellis2_c_attention_policy_tests
        tests/test_attention_policy.c
    )
    target_link_libraries(trellis2_c_attention_policy_tests PRIVATE trellis2_c ggml)
    add_test(NAME trellis2_c_attention_policy_tests COMMAND trellis2_c_attention_policy_tests)

    add_executable(trellis2_c_image_to_3d_adapter_tests
        tests/test_image_to_3d_adapter.c
    )
    target_compile_definitions(trellis2_c_image_to_3d_adapter_tests PRIVATE
        TRELLIS_TEST_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}")
    target_include_directories(trellis2_c_image_to_3d_adapter_tests PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/runtime"
        "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d")
    target_link_libraries(trellis2_c_image_to_3d_adapter_tests PRIVATE trellis2_c ggml)
    add_test(NAME trellis2_c_image_to_3d_adapter_tests COMMAND trellis2_c_image_to_3d_adapter_tests)

    add_executable(trellis2_c_gltf_axes_tests
        tests/test_gltf_axes.c
    )
    target_include_directories(trellis2_c_gltf_axes_tests PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d")
    target_link_libraries(trellis2_c_gltf_axes_tests PRIVATE trellis2_c)
    add_test(NAME trellis2_c_gltf_axes_tests COMMAND trellis2_c_gltf_axes_tests)

    if (TRELLIS2_C_BACKEND STREQUAL "cuda")
        add_executable(trellis2_c_tests
            tests/test_trellis.c
            tools/debug/trellis_checkpoint_validate.c
            tools/debug/trellis_sparse_reference.c
        )
        target_compile_definitions(trellis2_c_tests PRIVATE _POSIX_C_SOURCE=200809L)
        target_include_directories(trellis2_c_tests PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/src/ops"
            "${CMAKE_CURRENT_LIST_DIR}/src/ops/sparse/cuda"
            "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d"
            "${CMAKE_CURRENT_LIST_DIR}/src/architectures/dit_flow"
            "${CMAKE_CURRENT_LIST_DIR}/tools/debug")
        target_link_libraries(trellis2_c_tests PRIVATE trellis2_c ggml)
        add_test(NAME trellis2_c_tests COMMAND trellis2_c_tests)
        set_tests_properties(trellis2_c_tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(trellis2_c_backend_ops_tests
        tests/test_ggml_backend_ops.c
    )
    target_link_libraries(trellis2_c_backend_ops_tests PRIVATE trellis2_c ggml)
    add_test(
        NAME trellis2_c_backend_ops_${TRELLIS2_C_BACKEND}_tests
        COMMAND trellis2_c_backend_ops_tests ${TRELLIS2_C_BACKEND})
    set_tests_properties(trellis2_c_backend_ops_${TRELLIS2_C_BACKEND}_tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(trellis2_c_pixal_condition_tests
        tests/test_pixal_condition.c
    )
    target_include_directories(trellis2_c_pixal_condition_tests PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/ops/projection")
    target_link_libraries(trellis2_c_pixal_condition_tests PRIVATE trellis2_c)
    add_test(NAME trellis2_c_pixal_condition_tests COMMAND trellis2_c_pixal_condition_tests)

    add_executable(trellis2_c_pixal_naf_attention_tests
        tests/test_pixal_naf_attention.c
    )
    target_include_directories(trellis2_c_pixal_naf_attention_tests PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/ops/projection"
        "${CMAKE_CURRENT_LIST_DIR}/src/ops/projected_attention")
    target_link_libraries(trellis2_c_pixal_naf_attention_tests PRIVATE trellis2_c)
    add_test(NAME trellis2_c_pixal_naf_attention_tests COMMAND trellis2_c_pixal_naf_attention_tests)

    add_executable(trellis2_c_pixal_naf_encoder_tests
        tests/test_pixal_naf_encoder.c
    )
    target_include_directories(trellis2_c_pixal_naf_encoder_tests PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/architectures/pixal_naf")
    target_link_libraries(trellis2_c_pixal_naf_encoder_tests PRIVATE trellis2_c ggml)
    add_test(
        NAME trellis2_c_pixal_naf_encoder_${TRELLIS2_C_BACKEND}_tests
        COMMAND trellis2_c_pixal_naf_encoder_tests ${TRELLIS2_C_BACKEND})
    set_tests_properties(
        trellis2_c_pixal_naf_encoder_${TRELLIS2_C_BACKEND}_tests
        PROPERTIES SKIP_RETURN_CODE 77)

    if (TRELLIS2_C_BACKEND STREQUAL "vulkan")
        add_executable(trellis2_c_pixal_naf_vulkan_tests
            tests/test_pixal_naf_vulkan.c
        )
        target_include_directories(trellis2_c_pixal_naf_vulkan_tests PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/src/ops"
            "${CMAKE_CURRENT_LIST_DIR}/src/ops/projection"
            "${CMAKE_CURRENT_LIST_DIR}/src/ops/projected_attention")
        target_link_libraries(trellis2_c_pixal_naf_vulkan_tests PRIVATE trellis2_c)
        add_test(NAME trellis2_c_pixal_naf_vulkan_tests COMMAND trellis2_c_pixal_naf_vulkan_tests)
        set_tests_properties(trellis2_c_pixal_naf_vulkan_tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    if (TRELLIS2_C_BACKEND STREQUAL "cuda")
        add_executable(trellis2_c_pixal_naf_cuda_tests
            tests/test_pixal_naf_cuda.c
        )
        target_include_directories(trellis2_c_pixal_naf_cuda_tests PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/src/ops/projection"
            "${CMAKE_CURRENT_LIST_DIR}/src/ops/projected_attention")
        target_link_libraries(trellis2_c_pixal_naf_cuda_tests PRIVATE trellis2_c)
        add_test(NAME trellis2_c_pixal_naf_cuda_tests COMMAND trellis2_c_pixal_naf_cuda_tests)
        set_tests_properties(trellis2_c_pixal_naf_cuda_tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(trellis2_c_sparse_decoder_backend_tests
        tests/test_sparse_decoder_backend.c
    )
    target_include_directories(trellis2_c_sparse_decoder_backend_tests PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/ops")
    target_link_libraries(trellis2_c_sparse_decoder_backend_tests PRIVATE trellis2_c ggml)
    add_test(NAME trellis2_c_sparse_decoder_backend_cpu_tests COMMAND trellis2_c_sparse_decoder_backend_tests cpu)
    set_tests_properties(trellis2_c_sparse_decoder_backend_cpu_tests PROPERTIES SKIP_RETURN_CODE 77)
    if (TRELLIS2_C_BACKEND STREQUAL "vulkan")
        add_test(NAME trellis2_c_sparse_decoder_backend_vulkan_tests COMMAND trellis2_c_sparse_decoder_backend_tests vulkan)
        set_tests_properties(trellis2_c_sparse_decoder_backend_vulkan_tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

if (TRELLIS2_C_BUILD_TOOLS)
    set(TRELLIS_TOOL_COMMON_SOURCES
        tools/trellis_tool_cli.c
        tools/trellis_tool_model.c
    )

    add_executable(trellis-info tools/trellis_info.c)
    target_link_libraries(trellis-info PRIVATE trellis2_c ggml)

    add_executable(trellis-birefnet-rgba tools/trellis_birefnet_rgba.c)
    target_link_libraries(trellis-birefnet-rgba PRIVATE trellis2_c ggml)
    if (UNIX)
        target_link_libraries(trellis-birefnet-rgba PRIVATE m)
    endif()

    if (TRELLIS2_C_BACKEND STREQUAL "cuda")
        add_executable(trellis-sparse-structure
            tools/debug/trellis_sparse_structure.c
            tools/debug/trellis_checkpoint_validate.c
        )
        target_link_libraries(trellis-sparse-structure PRIVATE trellis2_c ggml)

        add_executable(trellis-structured-latent
            tools/debug/trellis_structured_latent.c
            tools/debug/trellis_checkpoint_validate.c
        )
        target_link_libraries(trellis-structured-latent PRIVATE trellis2_c ggml)

        add_executable(trellis-infer tools/debug/trellis_infer.c ${TRELLIS_TOOL_COMMON_SOURCES})
        target_include_directories(trellis-infer PRIVATE "${CMAKE_CURRENT_LIST_DIR}/tools")
        target_link_libraries(trellis-infer PRIVATE trellis2_c ggml)
    endif()

    add_executable(trellis-image-to-gltf
        tools/trellis_image_to_gltf.c
    )
    target_link_libraries(trellis-image-to-gltf PRIVATE trellis2_c ggml)
    if (UNIX)
        target_link_libraries(trellis-image-to-gltf PRIVATE m)
    endif()

    add_executable(trellis-rebake-gltf
        tools/vkmesh/trellis_rebake_gltf.c
    )
    target_include_directories(trellis-rebake-gltf PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d")
    target_link_libraries(trellis-rebake-gltf PRIVATE trellis2_c ggml)
    if (UNIX)
        target_link_libraries(trellis-rebake-gltf PRIVATE m)
    endif()

    add_executable(trellis-verify tools/debug/trellis_verify.c)
    if (UNIX)
        target_link_libraries(trellis-verify PRIVATE m)
    endif()

    if (GGML_VULKAN)
        function(add_vkmesh_shader NAME SOURCE SYMBOL)
            set(SPV "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv")
            set(HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/${SYMBOL}.h")
            add_custom_command(
                OUTPUT "${SPV}"
                COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2
                        "${CMAKE_CURRENT_LIST_DIR}/${SOURCE}"
                        -o "${SPV}"
                DEPENDS "${CMAKE_CURRENT_LIST_DIR}/${SOURCE}"
                COMMENT "Compile ${NAME} Vulkan shader")
            add_custom_command(
                OUTPUT "${HEADER}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
                COMMAND "${CMAKE_COMMAND}"
                        -DINPUT="${SPV}"
                        -DOUTPUT="${HEADER}"
                        -DSYMBOL=${SYMBOL}
                        -P "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
                DEPENDS "${SPV}" "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
                COMMENT "Generate ${NAME} Vulkan shader header")
            list(APPEND VKMESH_SHADER_HEADERS "${HEADER}")
            set(VKMESH_SHADER_HEADERS "${VKMESH_SHADER_HEADERS}" PARENT_SCOPE)
        endfunction()
        set(VKMESH_SHADER_HEADERS)
        set(VKMESH_SHADER_DIR tools/vkmesh/shaders)
        add_vkmesh_shader(vkmesh_expand_edges ${VKMESH_SHADER_DIR}/vkmesh_expand_edges.comp vkmesh_expand_edges_spv)
        add_vkmesh_shader(vkmesh_degenerate_faces ${VKMESH_SHADER_DIR}/vkmesh_degenerate_faces.comp vkmesh_degenerate_faces_spv)
        add_vkmesh_shader(vkmesh_face_keys ${VKMESH_SHADER_DIR}/vkmesh_face_keys.comp vkmesh_face_keys_spv)
        add_vkmesh_shader(vkmesh_sort_edges ${VKMESH_SHADER_DIR}/vkmesh_sort_edges.comp vkmesh_sort_edges_spv)
        add_vkmesh_shader(vkmesh_sort_face_keys ${VKMESH_SHADER_DIR}/vkmesh_sort_face_keys.comp vkmesh_sort_face_keys_spv)
        add_vkmesh_shader(vkmesh_mark_duplicate_faces ${VKMESH_SHADER_DIR}/vkmesh_mark_duplicate_faces.comp vkmesh_mark_duplicate_faces_spv)
        add_vkmesh_shader(vkmesh_mark_boundary_edges ${VKMESH_SHADER_DIR}/vkmesh_mark_boundary_edges.comp vkmesh_mark_boundary_edges_spv)
        add_vkmesh_shader(vkmesh_boundary_degree_owner ${VKMESH_SHADER_DIR}/vkmesh_boundary_degree_owner.comp vkmesh_boundary_degree_owner_spv)
        add_vkmesh_shader(vkmesh_compact_boundary_edges ${VKMESH_SHADER_DIR}/vkmesh_compact_boundary_edges.comp vkmesh_compact_boundary_edges_spv)
        add_vkmesh_shader(vkmesh_compact_face_pairs ${VKMESH_SHADER_DIR}/vkmesh_compact_face_pairs.comp vkmesh_compact_face_pairs_spv)
        add_vkmesh_shader(vkmesh_compact_faces ${VKMESH_SHADER_DIR}/vkmesh_compact_faces.comp vkmesh_compact_faces_spv)
        add_vkmesh_shader(vkmesh_compact_unique_simplify_edges ${VKMESH_SHADER_DIR}/vkmesh_compact_unique_simplify_edges.comp vkmesh_compact_unique_simplify_edges_spv)
        add_vkmesh_shader(vkmesh_copy_u32 ${VKMESH_SHADER_DIR}/vkmesh_copy_u32.comp vkmesh_copy_u32_spv)
        add_vkmesh_shader(vkmesh_accumulate_hole_components ${VKMESH_SHADER_DIR}/vkmesh_accumulate_hole_components.comp vkmesh_accumulate_hole_components_spv)
        add_vkmesh_shader(vkmesh_apply_orientation_flips ${VKMESH_SHADER_DIR}/vkmesh_apply_orientation_flips.comp vkmesh_apply_orientation_flips_spv)
        add_vkmesh_shader(vkmesh_assign_vertex_map ${VKMESH_SHADER_DIR}/vkmesh_assign_vertex_map.comp vkmesh_assign_vertex_map_spv)
        add_vkmesh_shader(vkmesh_fill_u32 ${VKMESH_SHADER_DIR}/vkmesh_fill_u32.comp vkmesh_fill_u32_spv)
        add_vkmesh_shader(vkmesh_remap_faces ${VKMESH_SHADER_DIR}/vkmesh_remap_faces.comp vkmesh_remap_faces_spv)
        add_vkmesh_shader(vkmesh_compact_vertices ${VKMESH_SHADER_DIR}/vkmesh_compact_vertices.comp vkmesh_compact_vertices_spv)
        add_vkmesh_shader(vkmesh_init_u32_sequence ${VKMESH_SHADER_DIR}/vkmesh_init_u32_sequence.comp vkmesh_init_u32_sequence_spv)
        add_vkmesh_shader(vkmesh_init_orientation_state ${VKMESH_SHADER_DIR}/vkmesh_init_orientation_state.comp vkmesh_init_orientation_state_spv)
        add_vkmesh_shader(vkmesh_seed_vertex_offsets ${VKMESH_SHADER_DIR}/vkmesh_seed_vertex_offsets.comp vkmesh_seed_vertex_offsets_spv)
        add_vkmesh_shader(vkmesh_scan_u32_stride ${VKMESH_SHADER_DIR}/vkmesh_scan_u32_stride.comp vkmesh_scan_u32_stride_spv)
        add_vkmesh_shader(vkmesh_union_face_edges ${VKMESH_SHADER_DIR}/vkmesh_union_face_edges.comp vkmesh_union_face_edges_spv)
        add_vkmesh_shader(vkmesh_union_boundary_edges ${VKMESH_SHADER_DIR}/vkmesh_union_boundary_edges.comp vkmesh_union_boundary_edges_spv)
        add_vkmesh_shader(vkmesh_union_corner_edges ${VKMESH_SHADER_DIR}/vkmesh_union_corner_edges.comp vkmesh_union_corner_edges_spv)
        add_vkmesh_shader(vkmesh_union_orientation_edges ${VKMESH_SHADER_DIR}/vkmesh_union_orientation_edges.comp vkmesh_union_orientation_edges_spv)
        add_vkmesh_shader(vkmesh_compress_parents ${VKMESH_SHADER_DIR}/vkmesh_compress_parents.comp vkmesh_compress_parents_spv)
        add_vkmesh_shader(vkmesh_compress_orientation_state ${VKMESH_SHADER_DIR}/vkmesh_compress_orientation_state.comp vkmesh_compress_orientation_state_spv)
        add_vkmesh_shader(vkmesh_vertex_face_degree ${VKMESH_SHADER_DIR}/vkmesh_vertex_face_degree.comp vkmesh_vertex_face_degree_spv)
        add_vkmesh_shader(vkmesh_vertex_face_adjacency ${VKMESH_SHADER_DIR}/vkmesh_vertex_face_adjacency.comp vkmesh_vertex_face_adjacency_spv)
        add_vkmesh_shader(vkmesh_vertex_qem ${VKMESH_SHADER_DIR}/vkmesh_vertex_qem.comp vkmesh_vertex_qem_spv)
        add_vkmesh_shader(vkmesh_simplify_edge_cost ${VKMESH_SHADER_DIR}/vkmesh_simplify_edge_cost.comp vkmesh_simplify_edge_cost_spv)
        add_vkmesh_shader(vkmesh_simplify_propagate_cost ${VKMESH_SHADER_DIR}/vkmesh_simplify_propagate_cost.comp vkmesh_simplify_propagate_cost_spv)
        add_vkmesh_shader(vkmesh_simplify_best_edge ${VKMESH_SHADER_DIR}/vkmesh_simplify_best_edge.comp vkmesh_simplify_best_edge_spv)
        add_vkmesh_shader(vkmesh_simplify_collapse_edges ${VKMESH_SHADER_DIR}/vkmesh_simplify_collapse_edges.comp vkmesh_simplify_collapse_edges_spv)
        add_vkmesh_shader(vkmesh_assign_corner_vertices ${VKMESH_SHADER_DIR}/vkmesh_assign_corner_vertices.comp vkmesh_assign_corner_vertices_spv)
        add_vkmesh_shader(vkmesh_write_repaired_mesh ${VKMESH_SHADER_DIR}/vkmesh_write_repaired_mesh.comp vkmesh_write_repaired_mesh_spv)
        add_vkmesh_shader(vkmesh_component_area ${VKMESH_SHADER_DIR}/vkmesh_component_area.comp vkmesh_component_area_spv)
        add_vkmesh_shader(vkmesh_mark_component_keep ${VKMESH_SHADER_DIR}/vkmesh_mark_component_keep.comp vkmesh_mark_component_keep_spv)
        add_vkmesh_shader(vkmesh_mark_hole_roots ${VKMESH_SHADER_DIR}/vkmesh_mark_hole_roots.comp vkmesh_mark_hole_roots_spv)
        add_vkmesh_shader(vkmesh_mark_hole_faces ${VKMESH_SHADER_DIR}/vkmesh_mark_hole_faces.comp vkmesh_mark_hole_faces_spv)
        add_vkmesh_shader(vkmesh_write_hole_vertices ${VKMESH_SHADER_DIR}/vkmesh_write_hole_vertices.comp vkmesh_write_hole_vertices_spv)
        add_vkmesh_shader(vkmesh_write_hole_faces ${VKMESH_SHADER_DIR}/vkmesh_write_hole_faces.comp vkmesh_write_hole_faces_spv)
        add_vkmesh_shader(vkmesh_unsigned_distance ${VKMESH_SHADER_DIR}/vkmesh_unsigned_distance.comp vkmesh_unsigned_distance_spv)
        add_custom_target(vkmesh_shader DEPENDS ${VKMESH_SHADER_HEADERS})
        if (TRELLIS2_C_HAS_VKMESH_C_API)
            add_library(trellis_vkmesh_api OBJECT tools/vkmesh/vkmesh.c)
            target_compile_definitions(trellis_vkmesh_api PRIVATE TRELLIS_VKMESH_LIBRARY)
            target_include_directories(trellis_vkmesh_api PRIVATE
                "${CMAKE_CURRENT_LIST_DIR}/include"
                "${CMAKE_CURRENT_BINARY_DIR}/generated"
                "${CMAKE_CURRENT_LIST_DIR}/3rd/ggml/include"
                "${CMAKE_CURRENT_LIST_DIR}/3rd/xatlas/source/xatlas")
            target_link_libraries(trellis_vkmesh_api PRIVATE Vulkan::Vulkan xatlas)
            set_target_properties(trellis_vkmesh_api PROPERTIES POSITION_INDEPENDENT_CODE ON)
            add_dependencies(trellis_vkmesh_api vkmesh_shader)
            target_sources(trellis2_c PRIVATE $<TARGET_OBJECTS:trellis_vkmesh_api>)
            add_dependencies(trellis2_c trellis_vkmesh_api)
        endif()
        add_executable(vkmesh tools/vkmesh/vkmesh.c)
        add_dependencies(vkmesh vkmesh_shader)
        target_include_directories(vkmesh PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/generated"
            "${CMAKE_CURRENT_LIST_DIR}/3rd/xatlas/source/xatlas")
        target_link_libraries(vkmesh PRIVATE Vulkan::Vulkan xatlas)
        set_target_properties(vkmesh PROPERTIES LINKER_LANGUAGE CXX)
        if (UNIX)
            target_link_libraries(vkmesh PRIVATE m)
        endif()
        if (TRELLIS2_C_BUILD_TESTS AND TRELLIS2_C_HAS_VKMESH_C_API)
            find_package(Threads REQUIRED)
            add_executable(trellis2_c_vkmesh_tests tests/test_vkmesh.c)
            target_include_directories(trellis2_c_vkmesh_tests PRIVATE
                "${CMAKE_CURRENT_LIST_DIR}/src/ops"
                "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d"
                "${CMAKE_CURRENT_LIST_DIR}/src/architectures/dit_flow"
                "${CMAKE_CURRENT_LIST_DIR}/3rd/cgltf")
            target_link_libraries(trellis2_c_vkmesh_tests PRIVATE
                trellis2_c ggml Vulkan::Vulkan Threads::Threads)
            add_dependencies(trellis2_c_vkmesh_tests vkmesh)
            add_test(NAME trellis2_c_vkmesh_tests
                COMMAND trellis2_c_vkmesh_tests $<TARGET_FILE:vkmesh>)
            set_tests_properties(trellis2_c_vkmesh_tests PROPERTIES SKIP_RETURN_CODE 77)
        endif()
        if (TARGET trellis-image-to-gltf)
            add_dependencies(trellis-image-to-gltf vkmesh)
        endif()
    endif()

    if (TARGET raylib)
        if (WIN32)
            set_source_files_properties(tools/trellis_viewer.c PROPERTIES COMPILE_DEFINITIONS "NOGDI;NOUSER")
        endif()
        set(TRELLIS_GUI_SOURCES tools/trellis_viewer.c)
        if (WIN32)
            list(APPEND TRELLIS_GUI_SOURCES tools/trellis_win_file_dialog.c)
        endif()
        add_executable(trellis-gui ${TRELLIS_GUI_SOURCES})
        target_include_directories(trellis-gui PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/src/ops"
            "${CMAKE_CURRENT_LIST_DIR}/src/ops/sparse"
            "${CMAKE_CURRENT_LIST_DIR}/src/tasks/image_to_3d"
            "${CMAKE_CURRENT_LIST_DIR}/src/architectures/dit_flow")
        target_link_libraries(trellis-gui PRIVATE trellis2_c ggml raylib)
        if (WIN32)
            target_link_libraries(trellis-gui PRIVATE comdlg32 shell32)
            set_target_properties(trellis-gui PROPERTIES WIN32_EXECUTABLE TRUE)
            target_link_options(trellis-gui PRIVATE /ENTRY:mainCRTStartup)
            add_custom_command(TARGET trellis-gui POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:trellis-gui>
                    $<TARGET_FILE_DIR:trellis-gui>
                COMMAND_EXPAND_LISTS)
        endif()
        if (UNIX)
            target_link_libraries(trellis-gui PRIVATE m pthread)
        endif()
    endif()
endif()
