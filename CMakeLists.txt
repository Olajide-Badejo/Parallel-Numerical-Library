# Parallel Numerical Lab.
#
# CMAKE_CUDA_HOST_COMPILER must be set before project() enables the CUDA
# language, because compiler identification runs inside project() and will fail
# against a host compiler nvcc rejects. Setting it afterwards is too late; that
# cost an afternoon and is recorded in the engineering log.
cmake_minimum_required(VERSION 3.28)

set(PNL_CUDA_HOST_COMPILER "g++-14" CACHE STRING
    "Host compiler nvcc drives. CUDA 13.3 rejects GCC newer than 15.")

option(PNL_ENABLE_OPENMP "Build the OpenMP backend" ON)
option(PNL_ENABLE_MPI    "Build the MPI and hybrid backends" ON)
option(PNL_ENABLE_CUDA   "Build the CUDA backend" ON)
option(PNL_BUILD_TESTS   "Build the test suite" ON)
option(PNL_WERROR        "Treat warnings as errors" ON)

# Decide on CUDA before project(), and only enable the language if a compiler is
# actually present, so a machine without CUDA configures cleanly instead of
# failing.
set(PNL_PROJECT_LANGUAGES CXX)
if(PNL_ENABLE_CUDA)
    include(CheckLanguage OPTIONAL RESULT_VARIABLE PNL_HAVE_CHECK_LANGUAGE)
    find_program(PNL_NVCC_EXECUTABLE nvcc)
    if(PNL_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_HOST_COMPILER "${PNL_CUDA_HOST_COMPILER}" CACHE FILEPATH "" FORCE)
        list(APPEND PNL_PROJECT_LANGUAGES CUDA)
    else()
        message(STATUS "pnl: nvcc not found, the CUDA backend will be skipped")
        set(PNL_ENABLE_CUDA OFF)
    endif()
endif()

project(parallel_numerical_lab
        VERSION 1.0.0
        DESCRIPTION "C++23 numerical library with pluggable parallel execution backends"
        HOMEPAGE_URL "https://github.com/olajidebadejo/parallel-numerical-lab"
        LANGUAGES ${PNL_PROJECT_LANGUAGES})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# At C++20 and above CMake scans every translation unit for module
# dependencies. This project uses headers throughout and no modules, so the scan
# is pure build time cost and it drags in the experimental -fmodules-ts path on
# GCC. Turning it off is both faster and one less moving part.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Optimisation flags. -O3 with the native architecture is what a benchmark of
# this kind must use; without -march=native the vector width is whatever the
# baseline x86-64 target allows and the bandwidth study would be measuring the
# compiler's conservatism rather than the machine. Fast math is deliberately NOT
# enabled: the equivalence suite asserts bit identical reductions, and
# -ffast-math would license the compiler to reassociate them.
add_library(pnl_flags INTERFACE)
target_compile_options(pnl_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3;-march=native>
)
if(PNL_WERROR)
    target_compile_options(pnl_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(pnl_core
    src/backend/factory.cpp
    src/backend/pthreads_pool.cpp
)
target_include_directories(pnl_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(pnl_core PUBLIC pnl_flags)

find_package(Threads REQUIRED)
target_link_libraries(pnl_core PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
if(PNL_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(pnl_core PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(pnl_core PUBLIC PNL_WITH_OPENMP=1)
        message(STATUS "pnl: OpenMP ${OpenMP_CXX_VERSION} enabled, spec date ${OpenMP_CXX_SPEC_DATE}")
    else()
        message(STATUS "pnl: OpenMP not found, that backend will be skipped")
        set(PNL_ENABLE_OPENMP OFF)
    endif()
endif()

# ---------------------------------------------------------------------------
# MPI
# ---------------------------------------------------------------------------
if(PNL_ENABLE_MPI)
    find_package(MPI COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        target_sources(pnl_core PRIVATE src/backend/mpi_runtime.cpp)
        target_link_libraries(pnl_core PUBLIC MPI::MPI_CXX)
        target_compile_definitions(pnl_core PUBLIC PNL_WITH_MPI=1)

        # OpenMPI 5.0.10 advertises MPI 3.1 through MPI_VERSION even though the
        # MPI 4.0 entry points are present and linkable. Guarding on the macro
        # would therefore disable features that work. Probe for the symbol
        # instead and record what was found.
        include(CheckCXXSymbolExists)
        set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
        set(CMAKE_REQUIRED_LIBRARIES ${MPI_CXX_LIBRARIES})
        check_cxx_symbol_exists(MPI_Isendrecv "mpi.h" PNL_HAVE_MPI_ISENDRECV)
        unset(CMAKE_REQUIRED_INCLUDES)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(PNL_HAVE_MPI_ISENDRECV)
            target_compile_definitions(pnl_core PUBLIC PNL_HAVE_MPI_ISENDRECV=1)
        endif()
        message(STATUS "pnl: MPI ${MPI_CXX_VERSION} enabled (${MPIEXEC_EXECUTABLE})")
    else()
        message(STATUS "pnl: MPI not found, those backends will be skipped")
        set(PNL_ENABLE_MPI OFF)
    endif()
endif()

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
if(PNL_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # sm_120 is the RTX 5070 in the target machine. native would be more
        # portable but fails on a build host with no GPU, which CI is.
        set(CMAKE_CUDA_ARCHITECTURES 120)
    endif()
    add_library(pnl_cuda STATIC
        src/cuda/jacobi_sweep.cu
        src/cuda/rb_gauss_seidel.cu
        src/cuda/stream_probe.cu
    )
    set_target_properties(pnl_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(pnl_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_compile_options(pnl_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
    target_link_libraries(pnl_core PUBLIC pnl_cuda)
    target_compile_definitions(pnl_core PUBLIC PNL_WITH_CUDA=1)
    message(STATUS "pnl: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}, host ${CMAKE_CUDA_HOST_COMPILER}")
endif()

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
add_executable(pnl src/main.cpp)
target_link_libraries(pnl PRIVATE pnl_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(PNL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "pnl: build type ${CMAKE_BUILD_TYPE}, C++ compiler ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
