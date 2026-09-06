// SPDX-License-Identifier: MIT
/// \file fuzz_parse.cpp
/// Fuzz target for the command line parser of `src/main.cpp`.
///
/// **How this reaches `parse()`.** That function is `static` inside an anonymous
/// namespace in `src/main.cpp`, which is right: it is the driver's own parser
/// and nothing else has any business calling it. So this translation unit
/// includes that file, with `main` renamed by a macro, and calls the parser
/// where it lives. The alternative would have been to lift `parse()` into a
/// header for the sake of a test, which changes the shape of the driver to suit
/// its test and makes an internal function part of an interface. Including the
/// file is honest about what it is: a test that reaches inside one translation
/// unit, and it fuzzes the parser the driver actually runs rather than a copy of
/// it.
///
/// Every header `src/main.cpp` includes is included here first, so that by the
/// time the macro is in force those files have already been read and the only
/// token it can rewrite is the definition at the bottom of `main.cpp`.
///
/// **What is generated, and the one thing that is not.** Inputs are built as a
/// flag from the driver's own list followed by an arbitrary byte string, and the
/// byte string is where the interesting values live: `--size` and friends go
/// through `std::from_chars`, which has to refuse `12abc`, `+`, `-`, an empty
/// string, a number with a thousand digits, and every kind of leading and
/// trailing rubbish, and `--tolerance` and `--omega` go through the floating
/// point overload of the same, which is a different parser. Flags with no value
/// at all are generated too, because a flag at the end of the command line is a
/// path of its own.
///
/// `--help` and `-h` are excluded, and that is not a gap. They call `usage(0)`,
/// which exits the process with status zero by design: a request to stop is not
/// an input to parse, and generating one would end the fuzzer's run rather than
/// test anything. Every other refusal in `parse()` is an exception as of this
/// phase, which is what makes the property below assertable at all; see
/// `argument_value` in `src/main.cpp` and CLI-02.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/stream_probe.hpp>
#include <pnl/backend/topology.hpp>
#include <pnl/bench/timed_solve.hpp>
#include <pnl/core/error.hpp>
#include <pnl/problems/dense_generator.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>
#include <pnl/version.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <iterator>
#include <memory>
#include <numeric>
#include <pnl_test.hpp>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "fuzz_targets.hpp"

// The driver's own main is not wanted here; the fuzz driver supplies one. The
// macro is in force for exactly one file and, because every header that file
// includes has already been read above, the only token it can rewrite is that
// definition.
#define main pnl_driver_main_not_used
#include "../../src/main.cpp"  // NOLINT(bugprone-suspicious-include)
#undef main

namespace pnl::fuzz::cli {

namespace {

/// Every flag `parse()` recognises that takes a value.
constexpr const char* VALUE_FLAGS[] = {
    "--solver",
    "--backend",
    "--problem",
    "--rhs",
    "--size",
    "--workers",
    "--threads-per-rank",
    "--pinning",
    "--reduction",
    "--schedule",
    "--mode",
    "--iterations",
    "--tolerance",
    "--omega",
    "--blocks",
    "--check-interval",
    "--reps",
    "--seed",
    "--label",
};

/// Every flag that takes none. `--help` and `-h` are absent on purpose; see the
/// file header.
constexpr const char* BARE_FLAGS[] = {
    "--progress",
    "--header",
    "--list",
    "--topology",
    "--bandwidth",
    "--version",
};

}  // namespace

int fuzz_one(const std::uint8_t* data, std::size_t size) {
    Bytes bytes(data, size);

    // A command line of a few arguments, so a flag that consumes the next one
    // can be followed by another flag and the interaction is exercised rather
    // than only the single flag case.
    std::vector<std::string> arguments;
    arguments.emplace_back("pnl");

    const std::size_t words = 1 + bytes.below(4);
    for (std::size_t word = 0; word < words; ++word) {
        const std::uint8_t kind = bytes.byte();
        if (kind % 4 == 0) {
            arguments.emplace_back(BARE_FLAGS[bytes.below(std::size(BARE_FLAGS))]);
            continue;
        }

        arguments.emplace_back(VALUE_FLAGS[bytes.below(std::size(VALUE_FLAGS))]);
        if (kind % 4 == 1) {
            // A flag with nothing after it, which used to leave the process by
            // std::exit and now throws.
            break;
        }

        // The value: an arbitrary byte string, NUL excluded because argv is a
        // C string array and a NUL is the end of one rather than a character in
        // it. Length is drawn so that the empty string and long values both
        // occur.
        const std::size_t length = bytes.below(24);
        std::string value;
        value.reserve(length);
        for (std::size_t i = 0; i < length; ++i) {
            const std::uint8_t raw = bytes.byte();
            value.push_back(static_cast<char>(raw == 0 ? '0' : raw));
        }
        arguments.push_back(std::move(value));
    }

    // argv as parse() expects it: an array of pointers into strings this
    // function owns for the length of the call.
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) argv.push_back(argument.data());

    try {
        const Options options = parse(static_cast<int>(argv.size()), argv.data());
        // The one postcondition worth asserting on a successful parse, because
        // it is the one CLI-01 was about: a repetition count below one indexes
        // an empty timings vector three times, and parse() is where that is
        // refused.
        if (options.repetitions < 1) {
            throw test::Failure{"parse() accepted a repetition count of " +
                                std::to_string(options.repetitions) +
                                ", which indexes an empty timings vector three times over"};
        }
    } catch (const Error&) {
        // The contract. Every refusal is a pnl::Error and a caller who catches
        // that type has covered the whole of what parse() can say no to.
    } catch (const test::Failure&) {
        throw;
    } catch (const std::exception& error) {
        std::string command;
        for (const auto& argument : arguments) {
            command += argument;
            command += ' ';
        }
        throw test::Failure{
            std::string("parse() threw an exception that is not derived from pnl::Error on the "
                        "command line '") +
            command + "': " + error.what()};
    } catch (...) {
        throw test::Failure{"parse() threw something that is not a std::exception at all"};
    }
    return 0;
}

}  // namespace pnl::fuzz::cli
