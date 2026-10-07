cmake_minimum_required(VERSION 3.18)
project(hedgespa_spectral_graphs VERSION 0.1.0 LANGUAGES CXX)

option(ENABLE_CUDA "Build the CUDA correlation kernel (requires CUDA toolkit)" OFF)
option(ENABLE_NATIVE_ARCH "Compile C++ with -march=native (fairer CPU baseline; binaries not portable)" OFF)
option(BUILD_TESTS "Build unit tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA is configured before any target is created so that CMAKE_CUDA_ARCHITECTURES is
# picked up as the default CUDA_ARCHITECTURES property of hs_core.
if(ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # "native" needs CMake >= 3.24 and a visible GPU; otherwise pass -DCMAKE_CUDA_ARCHITECTURES=<sm>.
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      set(CMAKE_CUDA_ARCHITECTURES native)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA correlation kernel: ENABLED (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
else()
  message(STATUS "CUDA correlation kernel: disabled (CPU only)")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Warning/arch flags apply to C++ sources only: nvcc rejects -Wpedantic and friends.
add_library(hs_flags INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(HS_CXX_FLAGS -Wall -Wextra -Wpedantic)
  if(ENABLE_NATIVE_ARCH)
    list(APPEND HS_CXX_FLAGS -march=native)
  endif()
  target_compile_options(hs_flags INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:${HS_CXX_FLAGS}>")
endif()

set(HS_SOURCES src/data.cpp src/correlation.cpp src/graph.cpp src/spectral.cpp src/clustering.cpp)
if(ENABLE_CUDA)
  list(APPEND HS_SOURCES cuda/correlation.cu)
endif()

add_library(hs_core STATIC ${HS_SOURCES})
target_include_directories(hs_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hs_core PUBLIC Eigen3::Eigen PRIVATE hs_flags)
if(ENABLE_CUDA)
  set_target_properties(hs_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(hs_core PUBLIC HS_WITH_CUDA=1)
  target_link_libraries(hs_core PUBLIC CUDA::cudart)
endif()

add_executable(hs_demo src/main.cpp)
target_link_libraries(hs_demo PRIVATE hs_core hs_flags)

add_executable(hs_bench src/bench_main.cpp)
target_link_libraries(hs_bench PRIVATE hs_core hs_flags)

if(BUILD_TESTS)
  enable_testing()
  add_executable(hs_tests tests/test_main.cpp)
  target_link_libraries(hs_tests PRIVATE hs_core hs_flags)
  add_test(NAME unit_tests COMMAND hs_tests)
  add_test(NAME demo_synthetic COMMAND hs_demo --mst)
  add_test(NAME demo_csv
           COMMAND hs_demo --csv ${CMAKE_CURRENT_SOURCE_DIR}/examples/sample_returns.csv
                   --clusters 3 --threshold 0.3)
  if(NOT ENABLE_CUDA)  # without CUDA, --gpu must fail loudly rather than silently run on CPU
    add_test(NAME demo_gpu_unavailable COMMAND hs_demo --gpu)
    set_tests_properties(demo_gpu_unavailable PROPERTIES WILL_FAIL TRUE)
  endif()
endif()
