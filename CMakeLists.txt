cmake_minimum_required(VERSION 3.18)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures for trellis2.c kernels" FORCE)
endif()

project(trellis2_c LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TRELLIS2_C_BUILD_TESTS "Build trellis2.c tests" ON)
option(TRELLIS2_C_BUILD_TOOLS "Build trellis2.c command line tools" ON)
option(TRELLIS2_C_BUILD_RAYLIB_VIEWER "Build the raylib snapshot viewers" ON)

# TRELLIS.2 inference is CUDA-only in this port. CPU is still built by ggml as a
# small runtime dependency, but all trellis2.c execution entry points initialize
# ggml's CUDA backend and fail when CUDA is unavailable.
set(GGML_CUDA ON CACHE BOOL "Build ggml CUDA backend" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Avoid host-specific codegen in the embedded ggml build" FORCE)
set(GGML_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/3rd/ggml" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

if (TRELLIS2_C_BUILD_RAYLIB_VIEWER AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/3rd/raylib/CMakeLists.txt")
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/3rd/raylib" "${CMAKE_CURRENT_BINARY_DIR}/raylib")
endif()

add_library(trellis2_c
    src/runtime/trellis_runtime.c
    src/model/trellis_cuda_kernels.cu
    src/model/trellis_cuda_forward.cu
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
    src/io/trellis_safetensors.c
    src/model/trellis_sparse_unet_vae_decoder.c
    src/io/trellis_weights.c
)

target_include_directories(trellis2_c
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/include"
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(trellis2_c PUBLIC ggml CUDA::cudart CUDA::cublas)
if (UNIX)
    target_link_libraries(trellis2_c PUBLIC m)
endif()

if (GGML_CUDA)
    target_compile_definitions(trellis2_c PUBLIC GGML_USE_CUDA TRELLIS_CUDA_ONLY=1)
endif()

if (TRELLIS2_C_BUILD_TESTS)
    enable_testing()
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
endif()

if (TRELLIS2_C_BUILD_TOOLS)
    set(TRELLIS_TOOL_COMMON_SOURCES
        tools/trellis_tool_cli.c
        tools/trellis_tool_model.c
    )

    add_executable(trellis-info tools/trellis_info.c)
    target_link_libraries(trellis-info PRIVATE trellis2_c ggml)

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
