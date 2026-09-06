# SPDX-License-Identifier: MIT
# Test suite.
#
# Each level from Section 10 gets its own binary so a failure names the gate it
# belongs to, and so the MPI level can be launched under mpirun without dragging
# the rest of the suite through an MPI runtime it does not need.

add_library(pnl_test_main STATIC test_main.cpp)
target_include_directories(pnl_test_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pnl_test_main PUBLIC pnl_core)
target_link_libraries(pnl_test_main PRIVATE pnl_dev_flags)

# add_pnl_test(<name> <source> [LABEL <label>])
function(add_pnl_test name source)
    cmake_parse_arguments(ARG "" "LABEL" "" ${ARGN})
    add_executable(${name} ${source})
    # pnl_dev_flags explicitly, not through pnl_core, which carries it PRIVATE.
    # The tests have to be built with the same optimisation and architecture
    # flags as the measured binary: the equivalence gate asserts bit identical
    # iterates, and a test compiled at a different vector width would be
    # asserting it of arithmetic the driver never runs.
    target_link_libraries(${name} PRIVATE pnl_test_main pnl_dev_flags)
    # Link with the C++ compiler even when CUDA is in the dependency graph; see
    # the note in the top level CMakeLists.
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
        LABELS "${ARG_LABEL}"
        TIMEOUT 900
    )
endfunction()

add_pnl_test(test_contract    unit/test_contract.cpp        LABEL unit)
add_pnl_test(test_numerics    unit/test_numerics.cpp        LABEL unit)
# Its own executable because it replaces the global operator new and operator
# delete, which is a program wide change no other test should have to live with.
add_pnl_test(test_no_allocation unit/test_no_allocation.cpp LABEL unit)
add_pnl_test(test_preconditions unit/test_preconditions.cpp LABEL unit)
add_pnl_test(test_registry    unit/test_registry.cpp        LABEL unit)
add_pnl_test(test_solvers     unit/test_solvers.cpp         LABEL unit)
# Reaches pnl/version.hpp through pnl_core's build interface, which is where the
# generated header lives; there is no copy of it in the source tree.
add_pnl_test(test_version     unit/test_version.cpp         LABEL unit)
add_pnl_test(test_convergence convergence/test_convergence.cpp LABEL convergence)
add_pnl_test(test_equivalence equivalence/test_equivalence.cpp LABEL equivalence)

# The contraction probe compiled the other way round, which is the only thing
# that proves the probe is connected to anything.
#
# It cannot go through add_pnl_test. That function links pnl_test_main, which
# links pnl_core PUBLIC, which links pnl_flags PUBLIC, and pnl_flags is where
# -ffp-contract=off comes from. A target that must compile with contraction ON
# therefore cannot inherit any of them, so this one compiles test_main.cpp for
# itself and sets the standard by hand.
#
# It does have to link the archive, because phase B4 added a case that calls
# make_backend from this fused translation unit and expects it to throw, and
# make_backend now calls a non inline lookup in factory.cpp. $<LINK_ONLY:> is
# what makes that possible: it links pnl_core and everything pnl_core links,
# and propagates none of its compile usage requirements, so -ffp-contract=off
# does not arrive with it. CMake appends inherited interface options after a
# target's own, so a plain link would put -ffp-contract=off last on the compile
# line and win. The include directories are already given by hand below, which
# is the only usage requirement this target actually wants.
#
# -O2 rather than -O3 only because the specification names -O2; the compile line
# CMake builds puts CMAKE_CXX_FLAGS_RELEASE first and these options after it, so
# whichever is written here is the one that takes effect. -march=native is what
# makes an FMA instruction available in the first place. If a machine ever
# stops fusing this expression the test fails loudly instead of passing
# vacuously, which is the point of it.
add_executable(test_contract_detects_fma
    ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp
    unit/test_contract.cpp)
target_include_directories(test_contract_detects_fma PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include)
#
# pnl_dev_flags arrives through $<LINK_ONLY:> as well, and only for its link
# options. This is the one target in the tree that does not link that target
# outright, so under PNL_SANITIZE it would be the one binary whose link line
# lacked the sanitizer runtime while the archive it links is full of calls into
# it, and the link would fail on __asan_ symbols in code this test has no
# opinion about. LINK_ONLY brings the runtime and leaves the compile line, which
# is the whole point of this target, exactly as spelled below.
target_link_libraries(test_contract_detects_fma
    PRIVATE $<LINK_ONLY:pnl_core> $<LINK_ONLY:pnl_dev_flags>)
target_compile_features(test_contract_detects_fma PRIVATE cxx_std_20)
target_compile_definitions(test_contract_detects_fma PRIVATE PNL_CONTRACT_EXPECT_FUSED=1)
target_compile_options(test_contract_detects_fma PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-O2;-march=native;-ffp-contract=fast>)
set_target_properties(test_contract_detects_fma PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME test_contract_detects_fma COMMAND test_contract_detects_fma)
set_tests_properties(test_contract_detects_fma PROPERTIES LABELS unit TIMEOUT 300)

# A header that is not self contained is a compile error for the first consumer
# who includes it on its own and for nobody else, so the assertion has to be a
# compile of exactly that one include. An OBJECT library is enough: it is built
# by `make build`, it links to nothing, and there is no runtime behaviour to
# assert, which is why it is not an add_pnl_test. pnl_flags rather than pnl_core,
# because the point is to compile the header with the flags a consumer gets and
# without anything else the library would have dragged in.
add_library(chunking_alone OBJECT unit/chunking_alone.cpp)
target_include_directories(chunking_alone PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(chunking_alone PRIVATE pnl_flags pnl_dev_flags)

# The style gate runs the dash checker over the whole tree. It is a test rather
# than only a make target so that CI cannot pass while the rule is broken.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME test_no_dashes
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_SOURCE_DIR}/scripts/check_no_dashes.py
                     ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_no_dashes PROPERTIES LABELS style TIMEOUT 300)

    add_test(NAME test_dash_checker_self
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_SOURCE_DIR}/tests/style/check_linter.py)
    set_tests_properties(test_dash_checker_self PROPERTIES LABELS style TIMEOUT 120)

    # The other half of ground rule 8 for the host: a build with -ffast-math has
    # to fail, and has to say why. It is a test that runs the compiler because a
    # guard that stops a compile cannot be observed from inside one. The
    # configured compiler is passed in rather than looked up, so this checks the
    # compiler the project is built with.
    add_test(NAME test_fast_math_rejected
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_SOURCE_DIR}/tests/style/check_fast_math.py
                     ${CMAKE_CXX_COMPILER}
                     ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_fast_math_rejected PROPERTIES LABELS style TIMEOUT 300)

    # The schema migration holds the only copy of every measurement in the
    # repository while it runs, so it is tested here rather than trusted at the
    # start of a sweep. It needs no build: the header it migrates towards is
    # passed to it.
    add_test(NAME test_migrate_summary
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_SOURCE_DIR}/tests/sweep/test_migrate_summary.py)
    set_tests_properties(test_migrate_summary PROPERTIES LABELS unit TIMEOUT 120)

    # Ground rule 7 lives entirely inside the asset generator: it decides which
    # differences the report may print as a number and which print as a phrase,
    # and how wide the knee interval is. Building the report proves none of that,
    # because the report builds whatever the generator decides, so the rules are
    # tested here against a summary with known values.
    add_test(NAME test_gen_report_assets
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_SOURCE_DIR}/tests/report/test_gen_report_assets.py)
    set_tests_properties(test_gen_report_assets PROPERTIES LABELS unit TIMEOUT 300)
endif()

# MPI tests run under mpirun at one, two and four ranks. Section 10 requires
# agreement at all three, and the remainder aware decomposition means the odd
# sizes are the interesting ones.
if(PNL_ENABLE_MPI AND MPI_CXX_FOUND)
    # This one brings its own main: it has to initialise MPI before any case
    # runs, silence every rank but zero, and combine the exit status across
    # ranks so a failure anywhere fails the test. So it links pnl_core rather
    # than pnl_test_main, which would supply a competing main.
    add_executable(test_mpi mpi/test_mpi.cpp)
    target_include_directories(test_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_mpi PRIVATE pnl_core pnl_dev_flags)
    set_target_properties(test_mpi PROPERTIES LINKER_LANGUAGE CXX)
    foreach(ranks IN ITEMS 1 2 4)
        add_test(NAME test_mpi_${ranks}rank
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                         --oversubscribe $<TARGET_FILE:test_mpi>)
        set_tests_properties(test_mpi_${ranks}rank PROPERTIES LABELS mpi TIMEOUT 900)
    endforeach()
endif()

if(PNL_ENABLE_CUDA)
    add_pnl_test(test_cuda cuda/test_cuda.cpp LABEL cuda)
endif()
