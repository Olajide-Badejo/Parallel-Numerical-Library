# Parallel Numerical Library.
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
#
# -ffp-contract=off deserves its own note. Without it the compiler may fuse
# a*b + c*d into an FMA, which is more accurate and entirely legal, but it is a
# choice the compiler makes per target and per expression. The SOR update has
# exactly that shape, so the host contracted it and the GPU, compiled with
# --fmad=false, did not, and the two disagreed in the last bit. Since the
# central claim of this library is that the same numerics run over every
# execution model, "bit identical unless the compiler felt like contracting" is
# not good enough. Turning contraction off on both sides makes the claim true
# and costs nothing measurable on kernels that are bandwidth bound anyway.
add_library(pnl_flags INTERFACE)
target_compile_options(pnl_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -ffp-contract=off>
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

    # sm_120 is the RTX 5070 in the target machine. This has to be a forced
    # assignment, not a guarded one: project() already gives
    # CMAKE_CUDA_ARCHITECTURES a default (75 here), so a NOT DEFINED guard never
    # fires and the build silently targets the wrong architecture.
    set(PNL_CUDA_ARCH "120" CACHE STRING "CUDA architectures to target")
    set(CMAKE_CUDA_ARCHITECTURES "${PNL_CUDA_ARCH}")

    # The CUDA runtime as an ordinary library, so the final link does not have
    # to be driven by nvcc. See below.
    find_package(CUDAToolkit REQUIRED)

    # Drop the CUDA host compiler's GCC library directory from the CUDA implicit
    # link directories.
    #
    # CMake detects those directories by asking nvcc, which is driving g++-14,
    # so the list contains /usr/lib/gcc/x86_64-linux-gnu/14. That -L lands on
    # the link line ahead of the search path g++-16 supplies for itself, so
    # -lstdc++ resolves to GCC 14's libstdc++, which has none of the atomic wait
    # and notify symbols GCC 16 emits for std::jthread and std::barrier. The
    # link then fails on std::__detail::__wait_impl and friends, which is
    # confusing because nothing in the CUDA code uses threads.
    #
    # Dropping every version specific GCC directory is safe precisely because
    # the link is driven by the C++ compiler, which contributes its own.
    # See ENV-05 in the engineering log.
    set(PNL_FILTERED_CUDA_LINK_DIRS "")
    foreach(pnl_dir IN LISTS CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES)
        if(NOT pnl_dir MATCHES "/lib/gcc/")
            list(APPEND PNL_FILTERED_CUDA_LINK_DIRS "${pnl_dir}")
        else()
            message(STATUS "pnl: dropping ${pnl_dir} from the CUDA implicit link directories")
        endif()
    endforeach()
    set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "${PNL_FILTERED_CUDA_LINK_DIRS}")

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
    # --fmad=false disables fused multiply add contraction. On a bandwidth bound
    # stencil it costs nothing measurable, and it buys something worth more: the
    # device sweep then evaluates exactly the arithmetic the host sweep does, so
    # "the GPU agrees with the CPU" is an equality rather than a tolerance. The
    # reductions still differ, because a tree association is not an ordered sum,
    # and the CUDA tests say which comparison is which.
    target_compile_options(pnl_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;--fmad=false;-lineinfo>)
    target_link_libraries(pnl_cuda PUBLIC CUDA::cudart)
    target_link_libraries(pnl_core PUBLIC pnl_cuda)
    target_compile_definitions(pnl_core PUBLIC PNL_WITH_CUDA=1)

    # Force the final link to be driven by the C++ compiler.
    #
    # Without this, CMake sees a CUDA source in the dependency graph and drives
    # the link with the CUDA host compiler, which is g++-14. That links g++-14's
    # libstdc++, which has none of the atomic wait and notify symbols GCC 16
    # emits for std::jthread and std::barrier, and the link fails on
    # std::__detail::__wait_impl and friends. The C ABI boundary that lets the
    # two compilers coexist only works if the C++ half is linked by the C++ half's
    # compiler. See ENV-05 in the engineering log.
    set_target_properties(pnl_cuda PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
        LINKER_LANGUAGE CXX)

    message(STATUS "pnl: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}, host ${CMAKE_CUDA_HOST_COMPILER}")
endif()

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
add_executable(pnl src/main.cpp)
target_link_libraries(pnl PRIVATE pnl_core)
set_target_properties(pnl PROPERTIES LINKER_LANGUAGE CXX)

# Stamp the commit into the binary so every result row can name the code that
# produced it, as Section 7 requires. Resolved at configure time, so a rebuild
# after new commits without reconfiguring would carry a stale hash; the sweep
# driver reconfigures, and a dirty tree is reported as such.
find_package(Git QUIET)
set(PNL_GIT_COMMIT "unknown")
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    OUTPUT_VARIABLE PNL_GIT_COMMIT
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    OUTPUT_VARIABLE PNL_GIT_DIRTY
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(NOT PNL_GIT_COMMIT)
        set(PNL_GIT_COMMIT "unknown")
    elseif(PNL_GIT_DIRTY)
        set(PNL_GIT_COMMIT "${PNL_GIT_COMMIT}.dirty")
    endif()
endif()
target_compile_definitions(pnl PRIVATE PNL_GIT_COMMIT="${PNL_GIT_COMMIT}")
message(STATUS "pnl: results will be stamped with commit ${PNL_GIT_COMMIT}")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(PNL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "pnl: build type ${CMAKE_BUILD_TYPE}, C++ compiler ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
