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
    src/trellis_backend.c
    src/custom_ops/trellis_cuda_ops.cu
    src/trellis_cuda_inference.cu
    src/trellis_decoder.c
    src/trellis_dino.c
    src/trellis_flex_dual_grid.c
    src/trellis_flow_model.c
    src/trellis_ops.c
    src/trellis_pipeline.c
    src/trellis_safetensors.c
    src/trellis_shape_decoder.c
    src/trellis_sparse.c
    src/trellis_stage1.c
    src/trellis_weights.c
)

target_include_directories(trellis2_c
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/include"
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(trellis2_c PUBLIC ggml CUDA::cudart)
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
    )
    target_compile_definitions(trellis2_c_tests PRIVATE _POSIX_C_SOURCE=200809L)
    target_link_libraries(trellis2_c_tests PRIVATE trellis2_c ggml)
    add_test(NAME trellis2_c_tests COMMAND trellis2_c_tests)
    set_tests_properties(trellis2_c_tests PROPERTIES SKIP_RETURN_CODE 77)
endif()

if (TRELLIS2_C_BUILD_TOOLS)
    add_executable(trellis-info tools/trellis_info.c)
    target_link_libraries(trellis-info PRIVATE trellis2_c ggml)

    add_executable(trellis-stage1 tools/trellis_stage1.c)
    target_link_libraries(trellis-stage1 PRIVATE trellis2_c ggml)

    add_executable(trellis-stage2 tools/trellis_stage2.c)
    target_link_libraries(trellis-stage2 PRIVATE trellis2_c ggml)

    add_executable(trellis-infer tools/trellis_infer.c)
    target_link_libraries(trellis-infer PRIVATE trellis2_c ggml)

    add_executable(trellis-verify tools/trellis_verify.c)
    if (UNIX)
        target_link_libraries(trellis-verify PRIVATE m)
    endif()

    if (TARGET raylib)
        add_executable(trellis-live
            tools/trellis_live.c
            tools/trellis_infer.c
            tools/trellis_stage2_mesh_live_viewer.c
        )
        target_compile_definitions(trellis-live PRIVATE TRELLIS_TOOL_LIBRARY)
        target_link_libraries(trellis-live PRIVATE trellis2_c ggml raylib)
        if (UNIX)
            target_link_libraries(trellis-live PRIVATE m)
        endif()

        add_executable(trellis-voxel-viewer tools/trellis_voxel_viewer.c)
        target_link_libraries(trellis-voxel-viewer PRIVATE raylib)
        if (UNIX)
            target_link_libraries(trellis-voxel-viewer PRIVATE m)
        endif()

        add_executable(trellis-mesh-viewer tools/trellis_mesh_viewer.c)
        target_link_libraries(trellis-mesh-viewer PRIVATE raylib)
        if (UNIX)
            target_link_libraries(trellis-mesh-viewer PRIVATE m)
        endif()

        add_executable(trellis-stage2-mesh-live-viewer tools/trellis_stage2_mesh_live_viewer.c)
        target_link_libraries(trellis-stage2-mesh-live-viewer PRIVATE trellis2_c ggml raylib)
        if (UNIX)
            target_link_libraries(trellis-stage2-mesh-live-viewer PRIVATE m)
        endif()
    endif()
endif()
