# SPDX-License-Identifier: MIT
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
# OFF by default, and deliberately so. A warnings as errors policy is a
# developer's contract with their own tree, not something a consumer of an
# installed copy should inherit: a newer compiler with a new diagnostic would
# turn into a build failure in code they did not write. The Makefile's
# configure target turns it on, so the developer build here stays strict, and
# CI turns it on explicitly. See BUILD-02 in docs/ENGINEERING_LOG.md.
option(PNL_WERROR        "Treat warnings as errors" OFF)

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
        DESCRIPTION "C++20 numerical library with pluggable parallel execution backends"
        # From `git remote get-url origin`. The previous value named a
        # repository that does not exist, which a reader would only discover by
        # following it.
        HOMEPAGE_URL "https://github.com/Olajide-Badejo/Parallel-Numerical-Library"
        LANGUAGES ${PNL_PROJECT_LANGUAGES})

set(CMAKE_CXX_STANDARD 20)
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

# Included here rather than beside the install rules because the install
# interface of pnl_core, written a few lines below, already needs
# CMAKE_INSTALL_INCLUDEDIR.
include(GNUInstallDirs)

# Two flag targets, and where the line between them falls is the whole of this
# library's compile time contract with a consumer.
#
# pnl_flags is public, is exported, and carries only what a consumer must
# compile with for the central claim of the library to hold. That is
# -ffp-contract=off and the language standard, and nothing else. Every flag that
# is a property of this machine or of this developer's taste belongs in the
# other target.
#
# -ffp-contract=off deserves its own note. Without it the compiler may fuse
# a*b + c*d into an FMA, which is more accurate and entirely legal, but it is a
# choice the compiler makes per target and per expression. The SOR update has
# exactly that shape, so the host contracted it and the GPU, compiled with
# --fmad=false, did not, and the two disagreed in the last bit. Since the
# central claim of this library is that the same numerics run over every
# execution model, "bit identical unless the compiler felt like contracting" is
# not good enough. Turning contraction off on both sides makes the claim true
# and costs nothing measurable on kernels that are bandwidth bound anyway. The
# library is about three quarters headers, so that flag has to travel with the
# headers or it does not reach the code it is about.
#
# The flag itself is a GNU driver spelling, so the block that sets it is guarded
# by compiler identity and every other toolchain is given the equivalent by its
# own name. "GNU|Clang" matches AppleClang as well, which is correct: it is a
# clang driver and takes -ffp-contract unchanged. An unknown compiler gets a
# warning rather than silence, because the central claim of the library is what
# is missing if the flag is not set.
add_library(pnl_flags INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pnl_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    )
elseif(MSVC)
    # /fp:strict is the MSVC spelling of the same contract. It disables
    # contraction, which is what this library needs from it; the default
    # /fp:precise permits the compiler to fuse. Untested: nobody has built this
    # tree with MSVC, and the supported platform statement in README.md says so.
    target_compile_options(pnl_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:/fp:strict>
    )
else()
    message(WARNING
        "pnl: no contraction flag is known for ${CMAKE_CXX_COMPILER_ID}, so the compiler may "
        "fuse a multiply and an add and the bit identity claim does not hold in this build")
endif()
# The standard is part of the exported interface, not just of this build. A
# consumer that compiles against these headers needs C++20 for std::jthread,
# std::barrier and std::span, and nothing above it: the tree contains no C++23
# only construct. See decision 20 in docs/DESIGN_DECISIONS.md.
target_compile_features(pnl_flags INTERFACE cxx_std_20)

# pnl_dev_flags is private, is never installed and is populated only when this
# project is the top level one, so a FetchContent or add_subdirectory consumer
# inherits none of it.
#
# -march=native is the reason the split exists. It is mandatory here, because
# without it the vector width is whatever the baseline x86-64 target allows and
# the bandwidth study would be measuring the compiler's conservatism rather than
# the machine; and it is indefensible in an exported interface, because a binary
# built with it on a build farm faults on the machine it is shipped to. Fast
# math is deliberately NOT enabled anywhere: the equivalence suite asserts bit
# identical reductions, and -ffast-math would license the compiler to
# reassociate them. See BUILD-02 in docs/ENGINEERING_LOG.md.
#
# The trap in the other direction is just as expensive: -O3 and -march=native
# must still reach the measured binary after this split, or every number
# recorded from here on quietly changes. Everything that is measured, which is
# pnl, pnl_core and the tests, links this target, and the phase B1 gate greps
# the compile line of src/main.cpp for all three flags.
add_library(pnl_dev_flags INTERFACE)
if(PROJECT_IS_TOP_LEVEL)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(pnl_dev_flags INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3;-march=native>
        )
        if(PNL_WERROR)
            target_compile_options(pnl_dev_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
        endif()
    elseif(MSVC)
        # /W4 is the MSVC warning level that corresponds to -Wall -Wextra; its
        # own /Wall is every warning the compiler has, including several that
        # fire in the standard library headers, and is not usable. There is no
        # Release optimisation or architecture line here on purpose: MSVC's
        # Release configuration already sets /O2, and MSVC has no -march=native,
        # only fixed /arch: levels, so a build here would be measuring the
        # baseline architecture and must not be mistaken for a measured one.
        target_compile_options(pnl_dev_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
        if(PNL_WERROR)
            target_compile_options(pnl_dev_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
# builtin_backends.cpp defines backend_registry(), which factory.cpp calls, so
# the archive member is always pulled in and the built in registrations always
# run. A registration object that nothing references would be linked out of a
# static library without a diagnostic; the file's own header comment says so.
add_library(pnl_core
    src/backend/builtin_backends.cpp
    src/backend/factory.cpp
    src/backend/pthreads_pool.cpp
)
target_include_directories(pnl_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
# The standard travels with pnl_flags, which carries it for the consumer too.
target_link_libraries(pnl_core PUBLIC pnl_flags)
# $<BUILD_INTERFACE:> around a PRIVATE dependency of a static library, and it is
# load bearing. A static library does not link its own dependencies, so CMake
# records even a PRIVATE one in INTERFACE_LINK_LIBRARIES as $<LINK_ONLY:...> for
# whoever links the archive later. install(EXPORT) then refuses the whole export
# set because pnl_dev_flags is not in it, and pnl_dev_flags must never be in it.
# Wrapping the entry makes it evaluate to nothing in the install interface while
# leaving the build unchanged. See BUILD-04 in docs/ENGINEERING_LOG.md.
target_link_libraries(pnl_core PRIVATE $<BUILD_INTERFACE:pnl_dev_flags>)

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
    # the link line ahead of the search path the C++ compiler supplies for
    # itself, so -lstdc++ resolves to GCC 14's libstdc++, which has none of the
    # atomic wait and notify symbols a newer GCC emits for std::jthread and
    # std::barrier. The link then fails on std::__detail::__wait_impl and
    # friends, which is confusing because nothing in the CUDA code uses threads.
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
    #
    # The host warning set is spelled out here rather than inherited from
    # pnl_dev_flags on purpose. Every option that target carries is guarded by
    # $<COMPILE_LANGUAGE:CXX> and this target compiles only CUDA sources, so
    # linking it in would contribute exactly nothing to a compile line while
    # adding pnl_dev_flags to this target's link interface, and from there to
    # the export set that install(EXPORT) validates. A flag that cannot apply is
    # not worth an entanglement with the install rules.
    target_compile_options(pnl_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;--fmad=false;-lineinfo>)
    target_link_libraries(pnl_cuda PUBLIC CUDA::cudart)
    target_link_libraries(pnl_core PUBLIC pnl_cuda)
    target_compile_definitions(pnl_core PUBLIC PNL_WITH_CUDA=1)

    # Force the final link to be driven by the C++ compiler.
    #
    # Without this, CMake sees a CUDA source in the dependency graph and drives
    # the link with the CUDA host compiler, which is g++-14. That links g++-14's
    # libstdc++, which has none of the atomic wait and notify symbols the C++
    # compiler emits for std::jthread and std::barrier, and the link fails on
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
# This is the measured binary, so it is the one the developer flags matter most
# on. It links them directly rather than inheriting them, because pnl_core
# carries them PRIVATE and a private usage requirement does not propagate.
target_link_libraries(pnl PRIVATE pnl_core pnl_dev_flags)
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
    # Only sources that determine the binary's behaviour count towards dirtiness.
    # Regenerated report assets and measurement output must not be able to stamp a
    # result row .dirty; that is the direct cause of every published row carrying
    # the stamp, see PROV-01 in the engineering log.
    execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=normal --
                            CMakeLists.txt cmake include src tests benchmarks scripts Makefile
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    OUTPUT_VARIABLE PNL_GIT_DIRTY
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT PNL_GIT_COMMIT)
        set(PNL_GIT_COMMIT "unknown")
    elseif(PNL_GIT_DIRTY)
        set(PNL_GIT_COMMIT "${PNL_GIT_COMMIT}.dirty")
    endif()
endif()
target_compile_definitions(pnl PRIVATE PNL_GIT_COMMIT="${PNL_GIT_COMMIT}")
message(STATUS "pnl: results will be stamped with commit ${PNL_GIT_COMMIT}")

# ---------------------------------------------------------------------------
# Install and export
# ---------------------------------------------------------------------------
# The first public API this library has ever had. Release 1.0.0 shipped no
# install rule, no export set, no package configuration file, no namespaced
# target and no version header, so there was nothing a consumer could have
# depended on and nothing this release can break. Creating an API where there
# was none is additive, which is why 1.1.0 is a minor bump; section 0.1.1 of the
# version 2 specification is where that argument lives, and CONTRIBUTING.md
# states the policy that follows from it.
include(CMakePackageConfigHelpers)

# What this build actually enabled, for the package configuration file to guard
# its find_dependency calls on. pnl_core links all three PUBLIC where they are
# present, so their imported targets are named inside the generated
# pnlTargets.cmake and a consumer who has not found them fails while CMake is
# still reading that file.
set(PNL_WITH_OPENMP ${PNL_ENABLE_OPENMP})
set(PNL_WITH_MPI    ${PNL_ENABLE_MPI})
set(PNL_WITH_CUDA   ${PNL_ENABLE_CUDA})

# The version header is generated into the build tree, never into the source
# tree. The reasoning is in cmake/version.hpp.in and it is not a matter of
# tidiness: the dirty check a few lines above watches `include`, so a generated
# header there would stamp every measured row .dirty on every configure.
set(PNL_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/generated/include")
configure_file(cmake/version.hpp.in "${PNL_GENERATED_INCLUDE_DIR}/pnl/version.hpp" @ONLY)
target_include_directories(pnl_core PUBLIC
    $<BUILD_INTERFACE:${PNL_GENERATED_INCLUDE_DIR}>)

# pnl::core resolves in the build tree as well as in an installed one, so a
# consumer who switches between find_package and add_subdirectory changes
# nothing. EXPORT_NAME is what makes the installed spelling pnl::core rather
# than pnl::pnl_core; the other targets keep their own names, because nothing is
# meant to link them directly.
add_library(pnl::core ALIAS pnl_core)
set_target_properties(pnl_core PROPERTIES EXPORT_NAME core)

# pnl_cuda has to be in the export set on any machine that built it. CMake
# refuses to export a target whose link interface names a static library outside
# the set, and pnl_core links pnl_cuda PUBLIC, so without this guard the install
# fails outright wherever nvcc is present and succeeds everywhere else. See
# BUILD-03 in docs/ENGINEERING_LOG.md. pnl_dev_flags is never in this list.
set(PNL_EXPORT_TARGETS pnl_core pnl_flags)
if(TARGET pnl_cuda)
    list(APPEND PNL_EXPORT_TARGETS pnl_cuda)
endif()

install(TARGETS ${PNL_EXPORT_TARGETS}
        EXPORT pnlTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${PNL_GENERATED_INCLUDE_DIR}/pnl/version.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pnl)

install(EXPORT pnlTargets
        FILE pnlTargets.cmake
        NAMESPACE pnl::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pnl)

configure_package_config_file(
    cmake/pnlConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/pnlConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pnl)

# SameMajorVersion. From 1.1.0 onward the source API follows semantic
# versioning, so a consumer asking for 1.1 is satisfied by any later 1.x and
# never by 2.x. There is no ABI promise across minor versions and CONTRIBUTING.md
# says why: three quarters of this library is headers, so a consumer compiles it
# rather than links a settled binary layout.
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/pnlConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pnlConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/pnlConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pnl)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(PNL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "pnl: build type ${CMAKE_BUILD_TYPE}, C++ compiler ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
