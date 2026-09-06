// SPDX-License-Identifier: MIT
/// \file fuzz_thomas.cpp
/// Fuzz target for `pnl::numerics::thomas_solve`, the tridiagonal solve.
///
/// Two preconditions guard it and a fuzzer's whole job is to walk around a
/// precondition: the four spans must be the same length, and every row must be
/// diagonally dominant. NUM-08 is the row of Section 4.7 behind this target: an
/// empty system wrote `c_prime[0]` and read `upper[0]` and `diagonal[0]` before
/// the guard existed, which is a heap write past the end of a zero length
/// allocation and is exactly what the address sanitizer sees and a passing test
/// does not.
///
/// **Two kinds of input are generated and the difference is the point.** Most
/// draws build a system that satisfies both preconditions, by taking arbitrary
/// off diagonals from the fuzzer and then choosing the diagonal to dominate
/// them, so the solve runs to completion and the fuzzer is exploring the
/// arithmetic rather than the guard. The rest hand over four arbitrary spans,
/// which is what a caller who got it wrong would do, and those must be refused
/// by an exception rather than by a write past the end of something.
///
/// A dominant system can still produce a non finite answer, because the fuzzer
/// is free to hand over infinities and NaNs as coefficients, and that is not a
/// defect: `thomas_solve` promises to solve the system it was given, and the
/// solution of a system containing an infinity may well be a NaN. What is
/// asserted is that the sizes are unchanged and that nothing outside the right
/// hand side was touched, which is the property a caller relies on.

#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>
#include <pnl/numerics/lu.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <pnl_test.hpp>
#include <string>
#include <vector>

#include "fuzz_targets.hpp"

namespace pnl::fuzz::thomas {

namespace {

/// A canary either side of the right hand side, so a write one element outside
/// it is a value this function can see rather than a sanitizer report the
/// dependency free driver would never notice.
///
/// The sanitizer build catches it too and catches more; this exists so that the
/// ordinary Release run of the CTest entry is not silent about the one failure
/// mode NUM-08 was.
constexpr Real CANARY = -12345.5;

}  // namespace

int fuzz_one(const std::uint8_t* data, std::size_t size) {
    Bytes bytes(data, size);

    const std::size_t n = bytes.below(17);
    const bool make_dominant = (bytes.byte() % 4) != 0;

    Vector lower(n, 0.0);
    Vector diagonal(n, 0.0);
    Vector upper(n, 0.0);
    // Two canaries around the right hand side, and the view handed to the solve
    // covers only the middle.
    Vector rhs_storage(n + 2, CANARY);

    for (std::size_t i = 0; i < n; ++i) {
        lower[i] = bytes.real();
        upper[i] = bytes.real();
        rhs_storage[i + 1] = bytes.real();
        if (make_dominant) {
            // The precondition, satisfied by construction: the diagonal is at
            // least the sum of the two off diagonal magnitudes. std::abs of an
            // infinity is an infinity and of a NaN is a NaN, and both are values
            // the solve is allowed to be given, so no attempt is made to keep
            // this finite.
            const Real off = std::abs(lower[i]) + std::abs(upper[i]);
            diagonal[i] = off + 1.0;
        } else {
            diagonal[i] = bytes.real();
        }
    }

    // A length mismatch, which is the other precondition, on a fraction of the
    // draws. Shortening rather than lengthening, because a solve that reads
    // past the end of the short span is the failure this is looking for.
    const bool mismatch_lengths = (bytes.byte() % 8) == 0;
    if (mismatch_lengths && n > 0) {
        upper.pop_back();
    }

    const VectorView rhs_view(rhs_storage.data() + 1, n);

    bool refused = false;
    try {
        numerics::thomas_solve(
            ConstVectorView(lower), ConstVectorView(diagonal), ConstVectorView(upper), rhs_view);
    } catch (const Error&) {
        // The contract: a precondition that does not hold is a pnl::Error. The
        // checks below still run, because a guard that threw after it had
        // already written would be a defect wearing a diagnostic.
        refused = true;
    } catch (const test::Failure&) {
        throw;
    } catch (const std::exception& error) {
        throw test::Failure{std::string("thomas_solve threw an exception that is not derived "
                                        "from pnl::Error at n = ") +
                            std::to_string(n) + ": " + error.what()};
    } catch (...) {
        throw test::Failure{"thomas_solve threw something that is not a std::exception at all"};
    }

    // A length mismatch is not optional to refuse: the solve indexes all four
    // spans to the length of the diagonal. A non dominant system may be refused
    // or may happen to be dominant anyway, since the fuzzer chose the diagonal,
    // so nothing is asserted about that one.
    if (mismatch_lengths && n > 0 && !refused) {
        throw test::Failure{
            "thomas_solve accepted four spans that are not the same length at "
            "n = " +
            std::to_string(n)};
    }

    if (rhs_storage.front() != CANARY || rhs_storage.back() != CANARY) {
        throw test::Failure{
            "thomas_solve wrote outside the right hand side it was given, at "
            "n = " +
            std::to_string(n) + "; the canary before it reads " +
            test::format(rhs_storage.front()) + " and the one after it " +
            test::format(rhs_storage.back())};
    }
    if (lower.size() != n || diagonal.size() != n) {
        throw test::Failure{"thomas_solve changed the length of a span it was given"};
    }
    return 0;
}

}  // namespace pnl::fuzz::thomas
