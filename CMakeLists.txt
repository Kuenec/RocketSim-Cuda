cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
  find_program(RSC_CLANG_CXX clang++)
  if(RSC_CLANG_CXX)
    set(CMAKE_CXX_COMPILER "${RSC_CLANG_CXX}")
  endif()
endif()

project(RocketSimCuda VERSION 3.0.0 LANGUAGES CXX)

option(RSC_ENABLE_CUDA "Build the CUDA backend" ON)
option(RSC_BUILD_TOOLS "Build verification and benchmark tools" ON)
option(RSC_NATIVE_ARCH "Optimize host code for the CPU of the build machine" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(RSC_HOST_FLAGS -ffp-contract=off -fno-fast-math -fno-finite-math-only
  -fno-math-errno)
if(RSC_NATIVE_ARCH)
  list(APPEND RSC_HOST_FLAGS -march=native)
endif()
set(RSC_WARNING_FLAGS -Wall -Wextra)

if(RSC_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
endif()

function(rsc_set_host_options target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${RSC_HOST_FLAGS} ${RSC_WARNING_FLAGS} -Wpedantic>)
  if(RSC_ENABLE_CUDA)
    foreach(flag IN LISTS RSC_HOST_FLAGS RSC_WARNING_FLAGS)
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
    endforeach()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -prec-div=true -prec-sqrt=true -ftz=false --Werror all-warnings>)
  endif()
endfunction()

add_library(rocketsim_cuda STATIC
  src/rsc/host/builder.cpp
  src/rocketsim_cuda/api.cpp)
target_include_directories(rocketsim_cuda PUBLIC src)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC)
  target_sources(rocketsim_cuda PRIVATE src/rocketsim_cuda/cpu_step.cpp)
  target_compile_definitions(rocketsim_cuda PRIVATE RSC_CPU_STEP)
  if(NOT RSC_NATIVE_ARCH AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    target_sources(rocketsim_cuda PRIVATE src/rocketsim_cuda/cpu_step_avx2.cpp)
    target_compile_definitions(rocketsim_cuda PRIVATE RSC_CPU_STEP_AVX2)
  endif()
endif()
set_target_properties(rocketsim_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(rocketsim_cuda PROPERTIES CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
rsc_set_host_options(rocketsim_cuda)
if(RSC_ENABLE_CUDA)
  target_sources(rocketsim_cuda PRIVATE src/rocketsim_cuda/cuda_batch.cu
    src/rocketsim_cuda/persistent_step.cu)
  target_compile_definitions(rocketsim_cuda PUBLIC RSC_HAS_CUDA)
  target_link_libraries(rocketsim_cuda PUBLIC CUDA::cudart)
endif()

add_library(rocketsim_cuda_c SHARED src/rocketsim_cuda/c_api.cpp)
set_target_properties(rocketsim_cuda_c PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_link_libraries(rocketsim_cuda_c PRIVATE rocketsim_cuda)
rsc_set_host_options(rocketsim_cuda_c)

if(RSC_BUILD_TOOLS AND RSC_ENABLE_CUDA)
  add_executable(rsc_libm_check tools/libm_check.cu)
  target_include_directories(rsc_libm_check PRIVATE src)
  rsc_set_host_options(rsc_libm_check)
endif()
