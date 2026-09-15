// SPDX-License-Identifier: MIT
/// \file fuzz_libfuzzer.cpp
/// The libFuzzer entry point, built only when `PNL_FUZZ=ON` and the compiler is
/// clang.
///
/// **Why it is not part of the default build.** `-fsanitize=fuzzer` is a clang
/// feature. Making the tests depend on it would turn "the tests build" into "you
/// have two compilers installed", which is a cost every contributor and every CI
/// job pays for a search most of them will never run. The dependency free
/// driver beside this file is what runs by default; this is what a person runs
/// deliberately, for hours, when they want a search rather than a gate.
///
/// One binary per target, because libFuzzer's whole model is one entry point
/// per process and a corpus per entry point. Which target this translation unit
/// is compiled as is decided by `PNL_FUZZ_TARGET` on the compile line, so the
/// three binaries share one source file and cannot drift apart.
///
/// To run one:
///
///     cmake -S . -B build-fuzz -G Ninja -DCMAKE_CXX_COMPILER=clang++ \
///           -DPNL_FUZZ=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
///           -DPNL_ENABLE_CUDA=OFF
///     cmake --build build-fuzz -j 6
///     mkdir -p corpus/brackets
///     build-fuzz/tests/fuzz_brackets_libfuzzer corpus/brackets -max_total_time=600
///
/// A crash writes the input to `crash-<hash>` in the working directory, and that
/// file is what turns a find into a regression test: the bytes go into a case in
/// `tests/fuzz/fuzz_random_driver.cpp`, or the seed there is changed in a commit
/// that says why.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <pnl_test.hpp>

#include "fuzz_targets.hpp"

#if !defined(PNL_FUZZ_TARGET)
#error "PNL_FUZZ_TARGET must name the target this binary fuzzes: cli, thomas or brackets"
#endif

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    try {
        return ::pnl::fuzz::PNL_FUZZ_TARGET::fuzz_one(data, size);
    } catch (const ::pnl::test::Failure& failure) {
        // A property violation is a finding, so it has to stop the run and be
        // written to a crash file. std::abort is what libFuzzer watches for;
        // returning non zero is not, and an exception that escapes here would
        // reach std::terminate with the message lost.
        std::fprintf(stderr, "pnl fuzz: %s\n", failure.message.c_str());
        std::fflush(stderr);
        std::abort();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "pnl fuzz: unexpected exception: %s\n", error.what());
        std::fflush(stderr);
        std::abort();
    }
}
