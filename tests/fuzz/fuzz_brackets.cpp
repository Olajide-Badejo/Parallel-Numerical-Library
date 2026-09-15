// SPDX-License-Identifier: MIT
/// \file fuzz_brackets.cpp
/// Fuzz target for the bracketing root finders, `bisection` and `brent`.
///
/// NUM-09 is the row behind this one. The bracket test used to be
/// `f(a) * f(b) <= 0`, and two ordinates of the same sign whose magnitudes are
/// small enough multiply to positive zero: `1e-200 * 1e-200` underflows, the
/// product is `+0.0`, `+0.0 <= 0.0` is true, and bisection returned the midpoint
/// of an interval containing no root with `converged = true` on it. A fuzzer
/// that is allowed to choose the endpoints and the function is the right shape
/// of test for a guard like that, because the failing inputs are not the ones
/// anybody writes by hand.
///
/// **What is generated.** The two endpoints come from the fuzzer as raw bit
/// patterns, so infinities, NaNs, denormals and negative zero all occur. The
/// function is drawn from a small family whose members are chosen for the
/// trouble they cause rather than for realism: one that underflows on both
/// sides, which is NUM-09 itself; one that is zero everywhere, so every point is
/// a root and the bracket test has to decide something about a degenerate case;
/// one that returns NaN, which is not a bracket and must not be treated as one;
/// one that is discontinuous, so a sign change exists but no root does; and two
/// ordinary ones so that most draws actually converge.
///
/// **The property.** Either a `Result<Real>` or a `pnl::Error`, and one thing
/// more that is specific to these functions and is the whole of NUM-09: **a run
/// that reports `converged` must have a root to show for it.** A converged
/// answer is checked against the function, and a bracket that never had a sign
/// change must have been refused rather than solved. That is the assertion the
/// product spelling would fail and the sign comparison passes.

#include <pnl/core/diagnostics.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>
#include <pnl/numerics/roots.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <pnl_test.hpp>
#include <string>

#include "fuzz_targets.hpp"

namespace pnl::fuzz::brackets {

namespace {

/// The awkward function family. Each is total: it returns a value for every
/// argument, including an infinity, so the fuzzer never has to avoid one.
[[nodiscard]] Real evaluate(int which, Real x) {
    switch (which % 6) {
        case 0:
            // An honest root at 0.5, which most draws will bracket or not.
            return x - 0.5;
        case 1:
            // A cubic with three roots, so the interpolation steps of Brent are
            // exercised rather than only its bisection fallback.
            return x * x * x - x;
        case 2:
            // NUM-09 exactly: ordinates so small that their product underflows
            // to positive zero while both keep the same sign.
            return 1.0e-200 * (2.0 + std::sin(x));
        case 3:
            // Zero everywhere. Every point is a root, and every pair of
            // endpoints is a bracket by the "either ordinate is zero" rule.
            return 0.0;
        case 4:
            // Not a number, which is not a bracket however it is spelled.
            return std::numeric_limits<Real>::quiet_NaN();
        default:
            // A step: a sign change with no root in between, which is the case
            // bisection cannot detect and is not asked to. What matters is that
            // it terminates and reports a bracket half width rather than
            // looping or lying about the value.
            return x < 0.25 ? -1.0 : 1.0;
    }
}

}  // namespace

int fuzz_one(const std::uint8_t* data, std::size_t size) {
    Bytes bytes(data, size);

    const int which = static_cast<int>(bytes.below(6));
    const Real a = bytes.real();
    const Real b = bytes.real();

    numerics::RootOptions options;
    // The tolerance, and the draw is split because the two halves support
    // different assertions.
    //
    // Most draws get a sane one, which is the regime where "converged" means
    // what a reader expects it to mean and the residual can be checked. The
    // rest get an arbitrary bit pattern, including a negative tolerance, a NaN
    // and an enormous one, because a caller can pass any of those and the loops
    // compare against them; there the only assertions are termination and the
    // exception type.
    //
    // The split exists because the first version of this file asserted the
    // residual unconditionally and the fuzzer refuted it in forty draws, with a
    // tolerance of about 3.5e101. Brent reported convergence with an error
    // estimate of 1.8e101, which is correct and honest: `tolerance` in
    // RootOptions is an absolute bound on the **bracket half width**, not on
    // the residual, and a caller who asks for 1e101 is told the bracket is that
    // wide and that the request was met. The harness was wrong, not the
    // library, and this comment is here so nobody tightens it back.
    const bool sane_tolerance = (bytes.byte() % 4) != 0;
    options.tolerance = sane_tolerance ? 1.0e-12 : bytes.real();
    // Bounded, so a draw cannot turn into an unbounded run. The cap is a
    // property of this harness and not of the library.
    options.max_iterations = sane_tolerance ? 200 : static_cast<Index>(bytes.below(64));

    const numerics::ScalarFunction f = [which](Real x) { return evaluate(which, x); };

    const Real fa = evaluate(which, a);
    const Real fb = evaluate(which, b);
    const bool is_bracket = numerics::detail::brackets_root(fa, fb);

    for (const bool use_brent : {false, true}) {
        Result<Real> result;
        bool refused = false;
        try {
            result = use_brent ? numerics::brent(f, a, b, options)
                               : numerics::bisection(f, a, b, options);
        } catch (const Error&) {
            // The contract: a bracket that is not one is a pnl::Error.
            refused = true;
        } catch (const test::Failure&) {
            throw;
        } catch (const std::exception& error) {
            throw test::Failure{std::string("a root finder threw an exception that is not "
                                            "derived from pnl::Error: ") +
                                error.what()};
        } catch (...) {
            throw test::Failure{"a root finder threw something that is not a std::exception"};
        }

        const std::string who = use_brent ? "brent" : "bisection";

        // NUM-09. A pair of ordinates that do not change sign is not a bracket,
        // whatever their product underflows to, and must be refused rather than
        // solved.
        if (!is_bracket && !refused) {
            throw test::Failure{who + " accepted endpoints whose ordinates are " +
                                test::format(fa) + " and " + test::format(fb) +
                                ", which do not change sign, and returned " +
                                test::format(result.value) +
                                " instead of refusing them. A product that underflows to zero "
                                "is not a sign change."};
        }
        if (is_bracket && refused) {
            throw test::Failure{who + " refused endpoints whose ordinates are " + test::format(fa) +
                                " and " + test::format(fb) + ", which do change sign"};
        }
        if (refused) continue;

        // A converged answer has to have something to show for it, and only a
        // sane tolerance makes that a fair question; see the note on the draw.
        //
        // The step function is excluded and the exclusion is a real one rather
        // than a convenience: it has a sign change and no root at all, so a
        // bracket half width below the tolerance is a correct report about an
        // interval and there is no value for it to be a claim about. Endpoints
        // that are not finite are excluded for the same reason: bisecting
        // between two infinities produces a NaN midpoint and the answer is
        // meaningless rather than wrong.
        if (result.diagnostics.converged && sane_tolerance && (which % 6) != 5 &&
            std::isfinite(a) && std::isfinite(b)) {
            const Real value = evaluate(which, result.value);
            const Real width = result.diagnostics.error_estimate;
            const bool plausible = std::abs(value) <= 1.0e-6 || !std::isfinite(value) ||
                                   !std::isfinite(width) || !std::isfinite(result.value);
            if (!plausible) {
                throw test::Failure{who + " reported convergence at " + test::format(result.value) +
                                    " where the function is " + test::format(value) +
                                    ", with a reported error estimate of " + test::format(width) +
                                    ", at a tolerance of " + test::format(options.tolerance)};
            }
        }

        // What "converged" means whatever the tolerance was: the reported error
        // estimate is a bound on the bracket half width and the run met the
        // bound it was given. The 2 * EPSILON * |value| slack is Brent's own
        // spelling of the tolerance and is not optional; bisection has no such
        // term and satisfies the comparison without it.
        if (result.diagnostics.converged && std::isfinite(options.tolerance)) {
            const Real width = result.diagnostics.error_estimate;
            const Real allowed =
                std::abs(options.tolerance) + 2.0 * EPSILON * std::abs(result.value);
            if (std::isfinite(width) && width > allowed && std::isfinite(result.value)) {
                throw test::Failure{who + " reported convergence with a bracket half width of " +
                                    test::format(width) + ", above the tolerance of " +
                                    test::format(options.tolerance) + " it was given"};
            }
        }
        if (result.diagnostics.evaluations < 2) {
            throw test::Failure{who + " reported " +
                                std::to_string(result.diagnostics.evaluations) +
                                " function evaluations, and it has to make at least the two "
                                "that decide whether the endpoints bracket a root"};
        }
    }
    return 0;
}

}  // namespace pnl::fuzz::brackets
