cmake_minimum_required(VERSION 3.24)
project(trtbench VERSION 0.1.0 LANGUAGES CXX)

option(TB_ENABLE_CUDA "Build the CUDA timing runner, built-in workloads and CLI" OFF)
option(TB_BUILD_TESTS "Build unit tests" ON)
option(TB_WARNINGS_AS_ERRORS "Treat host compiler warnings as errors" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(cmake/Dependencies.cmake)
include(cmake/GitSha.cmake)

set(TB_HOST_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
if(TB_WARNINGS_AS_ERRORS)
  list(APPEND TB_HOST_WARNINGS -Werror)
endif()

# ---------------------------------------------------------------------------
# Core: pure C++17. Statistics, clock audit, roofline, the lie detector, the
# comparison guard and JSON. No CUDA dependency, so every decision about
# whether a measurement is believable is unit-tested on a CPU-only runner.
# ---------------------------------------------------------------------------
add_library(trtbench_core
  src/core/config.cpp
  src/core/stats.cpp
  src/core/clocks.cpp
  src/core/roofline.cpp
  src/core/audit.cpp
  src/core/result.cpp
  src/core/compare.cpp
  src/core/json.cpp
  src/core/traps.cpp
  src/core/traps_synthetic.cpp
)
target_include_directories(trtbench_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(trtbench_core PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(trtbench_core PRIVATE ${TB_HOST_WARNINGS})
target_compile_definitions(trtbench_core PRIVATE
  TRTBENCH_GIT_SHA="${TB_GIT_SHA}"
  TRTBENCH_VERSION="${PROJECT_VERSION}")

# ---------------------------------------------------------------------------
# CUDA layer: event-timed runner, NVML clock source, clock locking, built-in
# calibration workloads and the CLI. Needs the CUDA toolkit to build; needs a
# GPU only to run.
# ---------------------------------------------------------------------------
if(TB_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # T4, A100, A10/A10G, L4, H100.
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)

  add_library(trtbench_cuda
    src/cuda/device_context.cpp
    src/cuda/runner.cpp
    src/cuda/nvml.cpp
    src/cuda/clock_lock.cpp
    src/cuda/builtin_workloads.cu
    src/cuda/traps_gpu.cu
  )
  target_link_libraries(trtbench_cuda
    PUBLIC trtbench_core CUDA::cudart_static
    PRIVATE ${CMAKE_DL_LIBS})
  # NVML is dlopen'ed, not linked: the binaries must start (and report "clocks
  # unavailable") on machines without a driver, such as CI runners.
  target_compile_options(trtbench_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${TB_HOST_WARNINGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra --Werror=all-warnings -lineinfo>)
endif()

# The CLI builds either way. Without CUDA only --self-test is available, and
# its GPU traps report SKIP with the reason.
add_executable(trtbench tools/trtbench_cli.cpp)
target_compile_options(trtbench PRIVATE ${TB_HOST_WARNINGS})
if(TB_ENABLE_CUDA)
  target_link_libraries(trtbench PRIVATE trtbench_cuda)
  target_compile_definitions(trtbench PRIVATE TRTBENCH_HAVE_CUDA=1)
else()
  target_link_libraries(trtbench PRIVATE trtbench_core)
endif()

if(TB_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
  # The trap suite is a test of the harness: a failing trap fails ctest.
  add_test(NAME self_test COMMAND trtbench --self-test)
endif()
