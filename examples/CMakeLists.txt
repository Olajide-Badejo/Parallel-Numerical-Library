# SPDX-License-Identifier: MIT
# The examples, which are also the install test.
#
# This is a project of its own, not a subdirectory of the library, and it is
# built against a staged install through find_package rather than against the
# build tree. That is the whole point: a consumer is a stranger with a prefix,
# not someone who has this repository.
#
# LANGUAGES CXX and nothing else, deliberately. Every optional half of this
# library is compiled in another language, and an exported target that quietly
# demands one of them fails here and nowhere else. Do not add CUDA, and do not
# add Fortran when part C brings it; the day this file needs them is the day the
# export is wrong.
cmake_minimum_required(VERSION 3.28)

project(pnl_examples LANGUAGES CXX)

find_package(pnl REQUIRED CONFIG)

add_executable(poisson poisson.cpp)
target_link_libraries(poisson PRIVATE pnl::core)

# The phase B4 extension point, exercised the way a stranger would: this file
# registers a backend of its own against the installed package and gets the
# whole solver zoo over it. It links pnl::core and nothing else, which is the
# claim being made.
add_executable(custom_backend custom_backend.cpp)
target_link_libraries(custom_backend PRIVATE pnl::core)

message(STATUS "pnl_examples: building against pnl ${pnl_VERSION}")
