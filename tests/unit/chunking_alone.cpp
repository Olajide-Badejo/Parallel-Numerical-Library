// SPDX-License-Identifier: MIT
/// \file chunking_alone.cpp
/// One include, and the compile is the whole test.
///
/// pnl/backend/chunking.hpp used Schedule without including the header that
/// defines it, so a consumer who included this one file first got a compile
/// error naming a type they had never heard of. Nothing in the suite could see
/// it, because every other translation unit reaches chunking.hpp through a
/// backend header that has already included backend.hpp. Section 4.7 records it.
///
/// This file therefore includes that header and nothing else, on purpose, and
/// is registered as an object library rather than a test executable: there is
/// no runtime assertion to make, the compile either succeeds or the header is
/// not self contained. Do not add a second include here, and do not link it
/// into anything. Adding a file of the same shape for any other public header
/// is the way to extend the check.

#include <pnl/backend/chunking.hpp>

namespace {

/// One use of each entry point, so the header is instantiated rather than only
/// parsed, and so the file has an external effect and cannot be optimised into
/// an empty object.
constexpr pnl::Index CHUNKS = pnl::backend::reduction_chunk_count(1024);

static_assert(CHUNKS == pnl::backend::DETERMINISTIC_CHUNKS);
static_assert(pnl::backend::reduction_chunk(1024, 0).size() == 2);
static_assert(pnl::backend::for_chunk_count(1024, 4, pnl::backend::Schedule::Static, 8) == 4);
static_assert(pnl::backend::for_chunk(1024, 4, pnl::backend::Schedule::Static, 8, 0).size() == 256);

}  // namespace
