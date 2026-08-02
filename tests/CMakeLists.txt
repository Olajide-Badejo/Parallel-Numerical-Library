# Test suite.
#
# Each level from Section 10 gets its own binary so a failure names the gate it
# belongs to, and so the MPI level can be launched under mpirun without dragging
# the rest of the suite through an MPI runtime it does not need.

add_library(pnl_test_main STATIC test_main.cpp)
target_include_directories(pnl_test_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pnl_test_main PUBLIC pnl_core)

# add_pnl_test(<name> <source> [LABEL <label>])
function(add_pnl_test name source)
    cmake_parse_arguments(ARG "" "LABEL" "" ${ARGN})
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE pnl_test_main)
    # Link with the C++ compiler even when CUDA is in the dependency graph; see
    # the note in the top level CMakeLists.
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
        LABELS "${ARG_LABEL}"
        TIMEOUT 900
    )
endfunction()

add_pnl_test(test_numerics    unit/test_numerics.cpp        LABEL unit)
add_pnl_test(test_solvers     unit/test_solvers.cpp         LABEL unit)
add_pnl_test(test_convergence convergence/test_convergence.cpp LABEL convergence)
add_pnl_test(test_equivalence equivalence/test_equivalence.cpp LABEL equivalence)

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
    target_link_libraries(test_mpi PRIVATE pnl_core)
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
