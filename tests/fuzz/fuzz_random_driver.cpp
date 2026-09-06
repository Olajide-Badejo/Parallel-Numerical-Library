// SPDX-License-Identifier: MIT
/// \file fuzz_random_driver.cpp
/// The dependency free driver: a fixed number of inputs from a fixed seed.
///
/// This is what CTest runs, under the `unit` label, on every machine and in
/// every job. It needs no clang, no `-fsanitize=fuzzer` and no corpus, it takes
/// a fraction of a second, and it gives the same verdict twice, which is the
/// property a gate needs and a search does not have.
///
/// It is not a substitute for the real search and does not pretend to be one.
/// A pseudo random byte string is a poor way to find a deep path, because it has
/// no coverage feedback and no notion of a corpus to mutate. What it is good at
/// is exactly what a regression test is for: a defect that a fuzzer once found
/// stays found, because the same seed produces the same inputs, and a hundred
/// thousand cheap draws over three small targets is a wide net for a shallow
/// hole. `tests/fuzz/fuzz_libfuzzer.cpp` is the search; this is the gate.
///
/// The seed is printed on every run and named in every failure message, and the
/// failing input is printed as hexadecimal, so a failure here is a reproduction
/// recipe rather than a report that something once went wrong.

#include <cstdint>
#include <cstdio>
#include <pnl_test.hpp>
#include <random>
#include <string>
#include <vector>

#include "fuzz_targets.hpp"

using namespace pnl;

namespace {

/// The seed. Fixed, printed, and part of the contract of this file: changing it
/// changes what is tested, so it changes in a commit that says so.
constexpr std::uint64_t FUZZ_SEED = 20260906;

/// Inputs per target. Enough to be a wide net over three small targets and
/// cheap enough that nobody is tempted to take it out of the default run.
constexpr int DRAWS = 20000;

/// Longest input. Every target reads a bounded number of bytes and the cursor
/// zero fills past the end, so a longer input buys nothing; what matters is
/// that short ones occur, which is where a target that assumed it had bytes
/// would fail.
constexpr std::size_t MAX_INPUT = 96;

/// The failing input, as a reviewer would paste it back in.
[[nodiscard]] std::string as_hex(const std::vector<std::uint8_t>& input) {
    std::string text;
    text.reserve(input.size() * 2);
    for (const std::uint8_t byte : input) {
        char pair[3];
        std::snprintf(pair, sizeof(pair), "%02x", byte);
        text += pair;
    }
    return text;
}

/// Run one target over DRAWS inputs.
void sweep(const char* name, int (*target)(const std::uint8_t*, std::size_t)) {
    std::mt19937_64 engine(FUZZ_SEED);
    std::uniform_int_distribution<int> length_draw(0, static_cast<int>(MAX_INPUT));
    std::uniform_int_distribution<int> byte_draw(0, 255);

    std::vector<std::uint8_t> input;
    for (int draw = 0; draw < DRAWS; ++draw) {
        input.resize(static_cast<std::size_t>(length_draw(engine)));
        for (auto& byte : input) byte = static_cast<std::uint8_t>(byte_draw(engine));

        try {
            (void)target(input.data(), input.size());
        } catch (const test::Failure& failure) {
            throw test::Failure{std::string(name) + " failed on draw " + std::to_string(draw) +
                                " of seed " + std::to_string(FUZZ_SEED) + ", input " +
                                as_hex(input) + "\n        " + failure.message};
        }
    }
    std::printf("        %s: %d inputs from seed %llu, no violation\n",
                name,
                DRAWS,
                static_cast<unsigned long long>(FUZZ_SEED));
}

}  // namespace

PNL_TEST("fuzz/parse refuses every command line by exception or accepts it") {
    sweep("parse", &fuzz::cli::fuzz_one);
}

PNL_TEST("fuzz/thomas_solve refuses or solves, and never writes outside its right hand side") {
    sweep("thomas_solve", &fuzz::thomas::fuzz_one);
}

PNL_TEST(
    "fuzz/the bracketing root finders never call a converged run on an interval with no root") {
    sweep("brackets", &fuzz::brackets::fuzz_one);
}

PNL_TEST("fuzz/the empty input is a legal input for every target") {
    // The shortest input there is, and the one a corpus starts from. Every
    // target has to survive it, which is what the zero filling cursor in
    // fuzz_targets.hpp is for.
    PNL_REQUIRE(fuzz::cli::fuzz_one(nullptr, 0) == 0);
    PNL_REQUIRE(fuzz::thomas::fuzz_one(nullptr, 0) == 0);
    PNL_REQUIRE(fuzz::brackets::fuzz_one(nullptr, 0) == 0);
}
