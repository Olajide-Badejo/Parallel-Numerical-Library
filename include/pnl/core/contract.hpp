// SPDX-License-Identifier: MIT
#pragma once

/// \file contract.hpp
/// The runtime half of the numerical contract: a probe that fails when this
/// build fused a multiply and an add.
///
/// Why a runtime probe at all. The library is about three quarters headers, so
/// most of the arithmetic the bit identity claim is about is compiled in the
/// consumer's translation units, not in libpnl_core.a. `pnl_flags` carries
/// `-ffp-contract=off` publicly so that a consumer who uses the package gets it,
/// but there is no predefined macro for `-ffp-contract` and `-ffp-contract=fast`
/// is GCC's default. A consumer who copies the headers, or who writes their own
/// compile line, therefore gets contraction with no diagnostic at all. The
/// `#if defined(__FAST_MATH__)` guard in types.hpp cannot see it, because
/// contraction has no macro to see. So the check has to happen while the program
/// runs, and it does, from `make_backend`.
///
/// Why it cannot be written the obvious way. "Compute a contraction sensitive
/// expression, compute the unfused value beside it, compare" is worthless in a
/// translation unit compiled with contraction on: either both sides contract and
/// agree, or the compiler folds both at compile time with correct rounding and
/// they agree again. The comparison has to be between a computation the compiler
/// is forced to emit and a **literal** of the correctly rounded unfused result.
/// `volatile` is what forces the emission; the constants below are the literal.
///
/// Where it runs, which release 1.1.0 changed. When this file was written
/// `make_backend` was an out of line function in `factory.cpp` and called the
/// probe from there. That does not work, and the reason is the reason the probe
/// exists: `assert_no_contraction` is inline, so the call was inlined into the
/// copy `factory.cpp` compiled, which is the copy built with this library's
/// flags, and a consumer whose own compile line was wrong was caught only if
/// their copy happened to be the one the linker kept. Phase B4 made the public
/// `make_backend` an inline function in `pnl/backend/backend.hpp` that calls
/// this probe and then a non inline `detail::make_backend_impl` in
/// `factory.cpp`. The probe is now emitted into the translation unit that
/// constructs the backend and therefore compiled with the flags that apply to
/// the arithmetic that translation unit is about to run.
///
/// It runs on every construction rather than once behind a flag, deliberately.
/// A function local static inside an inline function is one object for the whole
/// program, not one per translation unit, so a guard would check whichever
/// translation unit built the first backend and silently exempt every other one.
/// The probe is three volatile stores, a multiply, an add and a compare against
/// a call that starts a thread pool, so there is nothing to buy by guarding it.
///
/// What it still does not catch. A translation unit that runs the library's
/// arithmetic without ever constructing a backend of its own. Calling this
/// function yourself, from a translation unit of your own, is the way to check
/// such a compile line, and that is why it is public rather than an
/// implementation detail of the factory.

#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>

namespace pnl {

/// First operand of the contraction probe, exactly 1 + 2^-27.
///
/// These four values are named rather than written inline because release 1.2.0
/// runs the same probe on the Fortran side, as assertion 4 of Section 9.8 of the
/// specification, and a number that has to be the same in two languages must
/// have one definition in each and no third spelling anywhere.
inline constexpr Real CONTRACTION_PROBE_A = 1.0 + 0x1p-27;

/// Second operand, equal to the first.
inline constexpr Real CONTRACTION_PROBE_B = CONTRACTION_PROBE_A;

/// Addend, exactly -1.
inline constexpr Real CONTRACTION_PROBE_C = -1.0;

/// The correctly rounded value of `a * b + c` evaluated in two steps, exactly
/// 2^-26.
///
/// The exact product is 1 + 2^-26 + 2^-54. One ulp at that magnitude is 2^-52,
/// so the 2^-54 term is a quarter of an ulp and round to nearest discards it,
/// leaving 1 + 2^-26; adding -1 is then exact. A fused multiply add rounds once
/// instead of twice and keeps the term, giving 2^-26 + 2^-54, which is itself
/// exactly representable. The two answers are therefore distinguishable by `==`
/// rather than by a tolerance, which is the property the probe needs.
inline constexpr Real CONTRACTION_PROBE_EXPECTED = 0x1p-26;

/// Throw unless this translation unit was compiled without floating point
/// contraction.
///
/// \throws ConfigurationError if the compiler fused the multiply and the add.
inline void assert_no_contraction() {
    // a = b = 1 + 2^-27, c = -1. Two step gives exactly 2^-26; a fused multiply
    // add keeps the 2^-54 term. volatile forces the expression to be emitted.
    volatile Real a = CONTRACTION_PROBE_A;
    volatile Real b = CONTRACTION_PROBE_B;
    volatile Real c = CONTRACTION_PROBE_C;
    if (a * b + c != CONTRACTION_PROBE_EXPECTED) {
        throw ConfigurationError(
            "pnl was compiled with floating point contraction enabled; the bit identity "
            "guarantee does not hold");
    }
}

}  // namespace pnl
