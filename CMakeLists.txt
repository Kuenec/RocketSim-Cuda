cmake_minimum_required (VERSION 3.20)

project("RocketSim" LANGUAGES CXX)

# RocketSim-CUDA work tree (v2.2.1 + the bit-exact CUDA port).
# FP flags are part of the validated spec: the CPU reference is compiled with
# -ffp-contract=off and the GPU with --fmad=false so both sides do plain IEEE
# mul+add. Do not change them - the GPU/CPU bit-exactness gates depend on it.

file(GLOB_RECURSE FILES_ROCKETSIM "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.h")
file(GLOB_RECURSE FILES_BULLET "${PROJECT_SOURCE_DIR}/libsrc/bullet3-3.24/*.cpp" "${PROJECT_SOURCE_DIR}/libsrc/bullet3-3.24/*.h")

add_library(RocketSim STATIC ${FILES_ROCKETSIM} ${FILES_BULLET})
target_include_directories(RocketSim PUBLIC "${PROJECT_SOURCE_DIR}/src")

set_target_properties(RocketSim PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RocketSim PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
# linked into the GigaLearnCPP shared library
set_target_properties(RocketSim PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
	target_compile_options(RocketSim PRIVATE -ffp-contract=off)
endif()

find_package(Threads REQUIRED)
target_link_libraries(RocketSim PUBLIC Threads::Threads)

# GPU simulation path (detected, never assumed - CPU-only builds stay valid)
include(CheckLanguage)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES native)
endif()
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	find_package(CUDAToolkit REQUIRED)
	file(GLOB_RECURSE FILES_CUDA "${PROJECT_SOURCE_DIR}/src/Cuda/*.cu")
	target_sources(RocketSim PRIVATE ${FILES_CUDA})
	# the target predates enable_language(CUDA), so set the arch on it explicitly
	set_target_properties(RocketSim PROPERTIES
		CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
		CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
	target_compile_definitions(RocketSim PUBLIC RS_CUDA=1)
	target_compile_options(RocketSim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
	# in-kernel per-phase cycle profiler (GPUSIM_PROF at runtime); its
	# accumulators cost registers in the 255-register step kernel, so it is
	# compiled in only on request
	option(RS_GPU_PHASE_PROF "Compile the GPU step kernel's phase profiler" OFF)
	target_compile_definitions(RocketSim PRIVATE $<$<BOOL:${RS_GPU_PHASE_PROF}>:RS_GPU_PHASE_PROF>)
	target_link_libraries(RocketSim PUBLIC CUDA::cudart_static)
	message(STATUS "RocketSim: CUDA simulation enabled (arch: ${CMAKE_CUDA_ARCHITECTURES})")
else()
	message(STATUS "RocketSim: CUDA toolkit not found - GPU simulation disabled")
endif()
