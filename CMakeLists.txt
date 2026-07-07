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

option(TRELLIS2_C_BUILD_TESTS "Build trellis2.c tests" ON)
option(TRELLIS2_C_BUILD_TOOLS "Build trellis2.c command line tools" ON)
option(TRELLIS2_C_BUILD_RAYLIB_VIEWER "Build the raylib snapshot viewers" ON)
option(TRELLIS2_C_ENABLE_TENSOR_CORES "Prefer tensor-core capable kernels when 16-bit data and hardware support are available" ON)

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

if (GGML_VULKAN)
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

if (TRELLIS2_C_BACKEND STREQUAL "cuda" AND TRELLIS2_C_BUILD_RAYLIB_VIEWER AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/3rd/raylib/CMakeLists.txt")
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/3rd/raylib" "${CMAKE_CURRENT_BINARY_DIR}/raylib")
endif()

set(TRELLIS2_C_SOURCES
    src/runtime/trellis_runtime.c
    src/model/trellis_sparse_guides.c
    src/model/trellis_sparse_structure_decoder.c
    src/model/trellis_dino.c
    src/mesh/trellis_flex_dual_grid.c
    src/model/trellis_slat_flow_model.c
    src/model/trellis_ggml_layers.c
    src/pipeline/trellis_pipeline.c
    src/pipeline/trellis_dit_flow_executor.c
    src/pipeline/trellis_flow_sampler.c
    src/pipeline/trellis_sparse_structure_pipeline.c
    src/pipeline/trellis_structured_latent_pipeline.c
    src/export/trellis_gltf_export.c
    src/io/trellis_safetensors.c
    src/model/trellis_sparse_unet_vae_decoder.c
    src/model/trellis_sparse_unet_vae_decoder_exec.c
    src/model/sparse/trellis_sparse_cpu.c
    src/io/trellis_weights.c
)

if (TRELLIS2_C_BACKEND STREQUAL "cuda")
    list(APPEND TRELLIS2_C_SOURCES
        src/model/sparse/trellis_sparse_vulkan_stubs.c
        src/model/trellis_cuda_kernels.cu
        src/model/trellis_cuda_forward.cu
    )
else()
    list(APPEND TRELLIS2_C_SOURCES
        src/model/sparse/trellis_sparse_vulkan.c
        src/model/trellis_cuda_stubs.c
    )
endif()

add_library(trellis2_c
    ${TRELLIS2_C_SOURCES}
)

if (GGML_VULKAN)
    set(TRELLIS_SPARSE_VK_SPV "${CMAKE_CURRENT_BINARY_DIR}/trellis_sparse_vk.spv")
    set(TRELLIS_SPARSE_VK_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/trellis_sparse_vk_spv.h")
    set(TRELLIS_SPARSE_VK_MAT_SPV "${CMAKE_CURRENT_BINARY_DIR}/trellis_sparse_vk_mat.spv")
    set(TRELLIS_SPARSE_VK_MAT_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/trellis_sparse_vk_mat_spv.h")
    add_custom_command(
        OUTPUT "${TRELLIS_SPARSE_VK_SPV}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2
                "${CMAKE_CURRENT_LIST_DIR}/src/model/sparse/trellis_sparse_vk.comp"
                -o "${TRELLIS_SPARSE_VK_SPV}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/model/sparse/trellis_sparse_vk.comp"
        COMMENT "Compile TRELLIS sparse Vulkan shader")
    add_custom_command(
        OUTPUT "${TRELLIS_SPARSE_VK_MAT_SPV}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2
                "${CMAKE_CURRENT_LIST_DIR}/src/model/sparse/trellis_sparse_vk_mat.comp"
                -o "${TRELLIS_SPARSE_VK_MAT_SPV}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/model/sparse/trellis_sparse_vk_mat.comp"
        COMMENT "Compile TRELLIS sparse Vulkan tiled matmul shader")
    add_custom_command(
        OUTPUT "${TRELLIS_SPARSE_VK_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT="${TRELLIS_SPARSE_VK_SPV}"
                -DOUTPUT="${TRELLIS_SPARSE_VK_HEADER}"
                -DSYMBOL=trellis_sparse_vk_spv
                -P "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        DEPENDS "${TRELLIS_SPARSE_VK_SPV}" "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        COMMENT "Generate TRELLIS sparse Vulkan shader header")
    add_custom_command(
        OUTPUT "${TRELLIS_SPARSE_VK_MAT_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT="${TRELLIS_SPARSE_VK_MAT_SPV}"
                -DOUTPUT="${TRELLIS_SPARSE_VK_MAT_HEADER}"
                -DSYMBOL=trellis_sparse_vk_mat_spv
                -P "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        DEPENDS "${TRELLIS_SPARSE_VK_MAT_SPV}" "${CMAKE_CURRENT_LIST_DIR}/cmake/bin2c.cmake"
        COMMENT "Generate TRELLIS sparse Vulkan tiled matmul shader header")
    add_custom_target(trellis_sparse_vk_shader DEPENDS "${TRELLIS_SPARSE_VK_HEADER}" "${TRELLIS_SPARSE_VK_MAT_HEADER}")
    add_dependencies(trellis2_c trellis_sparse_vk_shader)
endif()

target_include_directories(trellis2_c
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/model"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${CMAKE_CURRENT_LIST_DIR}/3rd/cgltf"
        "${CMAKE_CURRENT_LIST_DIR}/3rd/stb"
        "${CMAKE_CURRENT_LIST_DIR}/3rd/xatlas/source/xatlas"
)

target_link_libraries(trellis2_c PUBLIC ggml xatlas)
if (TRELLIS2_C_BACKEND STREQUAL "cuda")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(trellis2_c PUBLIC CUDA::cudart CUDA::cublas)
endif()
if (GGML_VULKAN)
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
target_compile_definitions(trellis2_c PUBLIC
    TRELLIS_DEFAULT_BACKEND="${TRELLIS2_C_BACKEND}"
    TRELLIS_DEFAULT_GGML_BACKEND="${TRELLIS2_C_BACKEND}"
    TRELLIS_HAS_CUDA_DECODER=${TRELLIS_HAS_CUDA_DECODER_VALUE}
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
    if (TRELLIS2_C_BACKEND STREQUAL "cuda")
        add_executable(trellis2_c_tests
            tests/test_trellis.c
            tools/debug/trellis_checkpoint_validate.c
            tools/debug/trellis_sparse_reference.c
        )
        target_compile_definitions(trellis2_c_tests PRIVATE _POSIX_C_SOURCE=200809L)
        target_include_directories(trellis2_c_tests PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/src/model"
            "${CMAKE_CURRENT_LIST_DIR}/tools/debug")
        target_link_libraries(trellis2_c_tests PRIVATE trellis2_c ggml)
        add_test(NAME trellis2_c_tests COMMAND trellis2_c_tests)
        set_tests_properties(trellis2_c_tests PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(trellis2_c_backend_ops_tests
            tests/test_ggml_backend_ops.c
        )
        target_link_libraries(trellis2_c_backend_ops_tests PRIVATE trellis2_c ggml)
        add_test(NAME trellis2_c_backend_ops_tests COMMAND trellis2_c_backend_ops_tests)
        set_tests_properties(trellis2_c_backend_ops_tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(trellis2_c_sparse_decoder_backend_tests
        tests/test_sparse_decoder_backend.c
    )
    target_include_directories(trellis2_c_sparse_decoder_backend_tests PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/model")
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

    add_executable(trellis-image-to-obj
        tools/trellis_image_to_obj.c
    )
    target_link_libraries(trellis-image-to-obj PRIVATE trellis2_c ggml)
    if (UNIX)
        target_link_libraries(trellis-image-to-obj PRIVATE m)
    endif()

    add_executable(trellis-verify tools/debug/trellis_verify.c)
    if (UNIX)
        target_link_libraries(trellis-verify PRIVATE m)
    endif()

    if (TARGET raylib)
        add_executable(trellis-live
            tools/viewers/trellis_live.c
            tools/debug/trellis_infer.c
            tools/viewers/trellis_structured_latent_mesh_live_viewer.c
            ${TRELLIS_TOOL_COMMON_SOURCES}
        )
        target_compile_definitions(trellis-live PRIVATE TRELLIS_TOOL_LIBRARY)
        target_include_directories(trellis-live PRIVATE "${CMAKE_CURRENT_LIST_DIR}/tools")
        target_link_libraries(trellis-live PRIVATE trellis2_c ggml raylib)
        if (UNIX)
            target_link_libraries(trellis-live PRIVATE m)
        endif()

        add_executable(trellis-voxel-viewer tools/viewers/trellis_voxel_viewer.c)
        target_link_libraries(trellis-voxel-viewer PRIVATE raylib)
        if (UNIX)
            target_link_libraries(trellis-voxel-viewer PRIVATE m)
        endif()

        add_executable(trellis-mesh-viewer tools/viewers/trellis_mesh_viewer.c)
        target_link_libraries(trellis-mesh-viewer PRIVATE raylib)
        if (UNIX)
            target_link_libraries(trellis-mesh-viewer PRIVATE m)
        endif()

        add_executable(trellis-structured-latent-mesh-live-viewer tools/viewers/trellis_structured_latent_mesh_live_viewer.c)
        target_include_directories(trellis-structured-latent-mesh-live-viewer PRIVATE "${CMAKE_CURRENT_LIST_DIR}/tools")
        target_link_libraries(trellis-structured-latent-mesh-live-viewer PRIVATE trellis2_c ggml raylib)
        if (UNIX)
            target_link_libraries(trellis-structured-latent-mesh-live-viewer PRIVATE m)
        endif()
    endif()
endif()
