cmake_minimum_required(VERSION 3.20)
project(RocketSimCudaPort CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(REF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/RocketSim-ORIGINAL)
set(WORK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/RocketSim-CUDA)

if(NOT EXISTS ${REF_DIR}/src/RocketSim.h)
	message(FATAL_ERROR
		"Pristine reference tree missing. Restore it with:\n"
		"  git clone https://github.com/ZealanL/RocketSim RocketSim-ORIGINAL\n"
		"  git -C RocketSim-ORIGINAL checkout f80f9d90f922315c5877a81987845d3242ff1780")
endif()

find_package(Threads REQUIRED)

# Both trees define the same symbols; each bench links exactly one of them.
function(add_rocketsim_lib name tree)
	file(GLOB_RECURSE SRC CONFIGURE_DEPENDS
		${tree}/src/*.cpp ${tree}/src/*.h
		${tree}/libsrc/bullet3-3.24/*.cpp ${tree}/libsrc/bullet3-3.24/*.h)
	add_library(${name} STATIC ${SRC})
	target_include_directories(${name} PUBLIC ${tree}/src)
	target_link_libraries(${name} PUBLIC Threads::Threads)
endfunction()

add_rocketsim_lib(rocketsim_ref ${REF_DIR})
add_rocketsim_lib(rocketsim_work ${WORK_DIR})

# GPU path: work copy only. Detected, never assumed - CPU-only builds stay valid
# (--gpu-step then reports it is unavailable). On this machine configure with:
#   cmake -B build -G Ninja -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
#         -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15
# (login shells get both via /etc/profile.d/cuda.sh, see Persistent_Memory.md)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	find_package(CUDAToolkit REQUIRED)
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
		set(CMAKE_CUDA_ARCHITECTURES native)
	endif()
	file(GLOB_RECURSE WORK_CUDA_SRC CONFIGURE_DEPENDS ${WORK_DIR}/src/Cuda/*.cu)
	target_sources(rocketsim_work PRIVATE ${WORK_CUDA_SRC})
	set_target_properties(rocketsim_work PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
	target_compile_definitions(rocketsim_work PUBLIC RS_CUDA=1)
	# -fmad=false: GPU math must be plain IEEE mul+add to match the -ffp-contract=off CPU reference
	target_compile_options(rocketsim_work PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
	target_link_libraries(rocketsim_work PUBLIC CUDA::cudart_static)
	message(STATUS "CUDA enabled (arch: ${CMAKE_CUDA_ARCHITECTURES})")
else()
	message(STATUS "CUDA toolkit not found - GPU path disabled, --gpu-step unavailable")
endif()

add_executable(rocketsim_reference_bench tools/bench/main.cpp)
target_link_libraries(rocketsim_reference_bench PRIVATE rocketsim_ref)

add_executable(rocketsim_bench tools/bench_work/main.cpp)
target_link_libraries(rocketsim_bench PRIVATE rocketsim_work)

foreach(t rocketsim_ref rocketsim_work rocketsim_reference_bench rocketsim_bench)
	set_target_properties(${t} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
	if(NOT MSVC)
		# 2026-06-10: no FMA contraction, so the CPU reference defines plain mul+add
		# semantics that the GPU port can reproduce exactly (nvcc -fmad=false).
		# MSVC /fp:precise already does not contract.
		target_compile_options(${t} PRIVATE -ffp-contract=off)
	endif()
endforeach()
