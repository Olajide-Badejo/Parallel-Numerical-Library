// SPDX-License-Identifier: MIT
#pragma once

/// \file fuzz_targets.hpp
/// The three fuzz targets, and the property all three assert.
///
/// **The property, stated once because it is the same for all of them.** For
/// any input at all, the target must produce either a result or an exception
/// derived from `pnl::Error`. No crash, no `std::terminate`, no sanitizer
/// report, no `std::exit`, and no exception of any other type: a
/// `std::out_of_range` from a container or a `std::invalid_argument` from a
/// conversion is a defect here even though it is technically an exception,
/// because the library's contract is that a caller catches `pnl::Error` and has
/// covered everything the library can refuse.
///
/// **Each target is one function `int fuzz_one(const uint8_t*, size_t)`,** which
/// is the shape libFuzzer's `LLVMFuzzerTestOneInput` takes, so the same body
/// serves both drivers. The return is 0 when the input was consumed, whatever
/// the outcome, and it is the drivers that decide what to do about a violation:
/// the target itself throws `pnl::test::Failure` when the property is broken.
/// They live in namespaces of their own rather than under three different names
/// because the requirement is the signature, and one binary has to hold all
/// three.
///
/// **Two drivers, and the dependency free one is not a placeholder.**
///
///   - `tests/fuzz/fuzz_random_driver.cpp` is a CTest entry under the `unit`
///     label. It draws a fixed number of inputs from a fixed seed, so it runs on
///     every machine, in every CI job, needs no clang, and gives the same
///     verdict twice. It is a regression test made of random inputs rather than
///     a search.
///   - `tests/fuzz/fuzz_libfuzzer.cpp` is the real search, built only when
///     `PNL_FUZZ=ON` and the compiler is clang, because `-fsanitize=fuzzer` is
///     a clang feature. **It is deliberately not a requirement of the default
///     build.** Making a GCC build depend on a clang only sanitizer would turn
///     "the tests build" into "you have two compilers", which is a cost every
///     contributor pays for a search most of them will never run.
///
/// The three targets are chosen because each takes untrusted or awkward input
/// and each has a row in Section 4.7 behind it: `parse()` terminated the process
/// on a mistyped flag and indexed an empty vector on `--reps 0` (CLI-01),
/// `thomas_solve` wrote past the end of an empty system (NUM-08), and the root
/// finder accepted a bracket whose ordinate product underflowed to zero
/// (NUM-09). Each fix is a precondition, and a precondition is exactly the kind
/// of thing a fuzzer is good at walking around.

#include <pnl/core/error.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>

namespace pnl::fuzz {

/// Fuzz `parse()` of `src/main.cpp`.
namespace cli {
[[nodiscard]] int fuzz_one(const std::uint8_t* data, std::size_t size);
}  // namespace cli

/// Fuzz `pnl::numerics::thomas_solve`.
namespace thomas {
[[nodiscard]] int fuzz_one(const std::uint8_t* data, std::size_t size);
}  // namespace thomas

/// Fuzz the bracketing root finders, `bisection` and `brent`.
namespace brackets {
[[nodiscard]] int fuzz_one(const std::uint8_t* data, std::size_t size);
}  // namespace brackets

/// A cursor over the fuzzer's bytes.
///
/// Deliberately total: every read succeeds, and one past the end returns zero
/// rather than failing, so a target never has to branch on "did I have enough
/// input" and a short input exercises the same code as a long one with the tail
/// zeroed. libFuzzer's corpus starts with very short inputs, so this is the
/// common case rather than the corner.
class Bytes {
 public:
    Bytes(const std::uint8_t* data, std::size_t size) noexcept : data_(data), size_(size) {}

    [[nodiscard]] std::uint8_t byte() noexcept {
        if (at_ >= size_) return 0;
        return data_[at_++];
    }

    /// A value in `[0, bound)`, or 0 when \p bound is not positive.
    [[nodiscard]] std::size_t below(std::size_t bound) noexcept {
        if (bound == 0) return 0;
        return static_cast<std::size_t>(byte()) % bound;
    }

    /// Eight bytes reinterpreted as a double, which is the point: the fuzzer
    /// gets to produce infinities, NaNs, denormals and negative zero, and every
    /// one of those is a value a caller can pass.
    [[nodiscard]] double real() noexcept {
        std::uint64_t bits = 0;
        for (int i = 0; i < 8; ++i) {
            bits |= static_cast<std::uint64_t>(byte()) << (8 * i);
        }
        static_assert(sizeof(double) == sizeof(bits));
        return std::bit_cast<double>(bits);
    }

    [[nodiscard]] bool exhausted() const noexcept { return at_ >= size_; }

    [[nodiscard]] std::size_t consumed() const noexcept { return at_; }

 private:
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t at_ = 0;
};

}  // namespace pnl::fuzz
