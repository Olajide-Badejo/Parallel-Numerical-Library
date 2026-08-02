#pragma once

/// \file roots.hpp
/// Scalar root finding: bisection, Newton, and Brent.
///
/// Reference: Burden and Faires, "Numerical Analysis", 10th ed., Cengage,
/// chapter 2; Brent, "Algorithms for Minimization without Derivatives",
/// Prentice Hall 1973, chapter 4.

#include <pnl/core/diagnostics.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>

#include <algorithm>
#include <cmath>
#include <functional>

namespace pnl::numerics {

/// A real function of one real variable.
using ScalarFunction = std::function<Real(Real)>;

/// Stopping controls shared by the root finders.
struct RootOptions {
    /// Absolute tolerance on the bracket width or step size.
    Real tolerance = 1.0e-12;
    Index max_iterations = 200;
};

/// Bisection on a sign changing bracket.
///
/// Convergence order: 1, linear with rate exactly 1/2. The bracket width halves
/// every iteration, so the iteration count needed is known in advance and the
/// method cannot fail once a bracket exists. That reliability is why it is kept
/// alongside faster methods and why Brent falls back to it.
///
/// error_estimate is the half width of the final bracket, a genuine bound on
/// the distance to the root rather than an estimate.
///
/// \throws InvalidArgument if f(a) and f(b) do not have opposite signs.
[[nodiscard]] inline Result<Real> bisection(const ScalarFunction& f, Real a, Real b,
                                            const RootOptions& options = {}) {
    Real fa = f(a);
    Real fb = f(b);
    Index evaluations = 2;
    require(fa * fb <= 0.0, "bisection needs a bracket whose endpoints differ in sign");

    if (fa == 0.0) return make_result(a, Diagnostics{0.0, 0, evaluations, true,
                                                     StopReason::Converged});
    if (fb == 0.0) return make_result(b, Diagnostics{0.0, 0, evaluations, true,
                                                     StopReason::Converged});

    Diagnostics diagnostics;
    Real midpoint = a;
    Index iteration = 0;
    for (; iteration < options.max_iterations; ++iteration) {
        midpoint = a + 0.5 * (b - a);
        const Real half_width = 0.5 * std::abs(b - a);
        if (half_width <= options.tolerance) {
            diagnostics.converged = true;
            diagnostics.reason = StopReason::Converged;
            diagnostics.error_estimate = half_width;
            break;
        }
        const Real fm = f(midpoint);
        ++evaluations;
        if (fm == 0.0) {
            diagnostics.converged = true;
            diagnostics.reason = StopReason::Converged;
            diagnostics.error_estimate = 0.0;
            break;
        }
        if ((fa < 0.0) == (fm < 0.0)) {
            a = midpoint;
            fa = fm;
        } else {
            b = midpoint;
            fb = fm;
        }
        diagnostics.error_estimate = 0.5 * std::abs(b - a);
    }

    diagnostics.iterations = iteration;
    diagnostics.evaluations = evaluations;
    if (!diagnostics.converged) diagnostics.reason = StopReason::IterationCap;
    return make_result(midpoint, diagnostics);
}

/// Newton iteration with an analytic derivative.
///
/// Convergence order: 2 for a simple root, provided the initial guess is close
/// enough and the derivative does not vanish there. Degrades to order 1 at a
/// multiple root, which the diagnostics expose through the iteration count
/// rather than hiding.
///
/// error_estimate is the magnitude of the last Newton step, which for a
/// quadratically converging iteration bounds the remaining error to leading
/// order.
///
/// \throws NumericalFailure if the derivative vanishes at an iterate.
[[nodiscard]] inline Result<Real> newton(const ScalarFunction& f, const ScalarFunction& df,
                                         Real x0, const RootOptions& options = {}) {
    Diagnostics diagnostics;
    Real x = x0;
    Index evaluations = 0;
    Index iteration = 0;
    Real step = 0.0;

    for (; iteration < options.max_iterations; ++iteration) {
        const Real value = f(x);
        const Real slope = df(x);
        evaluations += 2;
        if (!std::isfinite(value) || !std::isfinite(slope)) {
            diagnostics.reason = StopReason::Diverged;
            break;
        }
        if (slope == 0.0) {
            throw NumericalFailure("newton: the derivative vanished at x = " +
                                   std::to_string(x));
        }
        step = value / slope;
        x -= step;
        diagnostics.error_estimate = std::abs(step);
        if (std::abs(step) <= options.tolerance) {
            diagnostics.converged = true;
            diagnostics.reason = StopReason::Converged;
            ++iteration;
            break;
        }
    }

    diagnostics.iterations = iteration;
    diagnostics.evaluations = evaluations;
    if (!diagnostics.converged && diagnostics.reason != StopReason::Diverged) {
        diagnostics.reason = StopReason::IterationCap;
    }
    return make_result(x, diagnostics);
}

/// Brent's method: inverse quadratic interpolation with a guaranteed bisection
/// fallback.
///
/// Convergence order: superlinear in practice, roughly 1.84 when the inverse
/// quadratic step is accepted, while never converging more slowly than
/// bisection because the bracket is maintained at every step. This combination
/// of speed and guaranteed termination is why it is the default here.
///
/// error_estimate is the final bracket half width.
///
/// \throws InvalidArgument if f(a) and f(b) do not have opposite signs.
[[nodiscard]] inline Result<Real> brent(const ScalarFunction& f, Real a, Real b,
                                        const RootOptions& options = {}) {
    Real fa = f(a);
    Real fb = f(b);
    Index evaluations = 2;
    require(fa * fb <= 0.0, "brent needs a bracket whose endpoints differ in sign");

    if (fa == 0.0) return make_result(a, Diagnostics{0.0, 0, evaluations, true,
                                                     StopReason::Converged});
    if (fb == 0.0) return make_result(b, Diagnostics{0.0, 0, evaluations, true,
                                                     StopReason::Converged});

    // Keep b as the best estimate.
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    Real c = a;
    Real fc = fa;
    Real d = b - a;
    Real e = d;
    bool used_bisection = true;

    Diagnostics diagnostics;
    Index iteration = 0;
    for (; iteration < options.max_iterations; ++iteration) {
        if (fb * fc > 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        const Real tolerance = 2.0 * EPSILON * std::abs(b) + 0.5 * options.tolerance;
        const Real midpoint = 0.5 * (c - b);
        diagnostics.error_estimate = std::abs(midpoint);

        if (std::abs(midpoint) <= tolerance || fb == 0.0) {
            diagnostics.converged = true;
            diagnostics.reason = StopReason::Converged;
            break;
        }

        if (std::abs(e) >= tolerance && std::abs(fa) > std::abs(fb)) {
            // Attempt interpolation: inverse quadratic when three distinct
            // ordinates are available, otherwise the secant.
            const Real s = fb / fa;
            Real p = 0.0;
            Real q = 0.0;
            if (a == c) {
                p = 2.0 * midpoint * s;
                q = 1.0 - s;
            } else {
                const Real r = fb / fc;
                const Real t = fa / fc;
                p = s * (2.0 * midpoint * t * (t - r) - (b - a) * (r - 1.0));
                q = (t - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = std::abs(p);

            const Real limit = std::min(3.0 * midpoint * q - std::abs(tolerance * q),
                                        std::abs(e * q));
            if (2.0 * p < limit) {
                e = d;
                d = p / q;
                used_bisection = false;
            } else {
                d = midpoint;
                e = d;
                used_bisection = true;
            }
        } else {
            d = midpoint;
            e = d;
            used_bisection = true;
        }

        a = b;
        fa = fb;
        b += std::abs(d) > tolerance ? d : (midpoint > 0.0 ? tolerance : -tolerance);
        fb = f(b);
        ++evaluations;
    }

    (void)used_bisection;
    diagnostics.iterations = iteration;
    diagnostics.evaluations = evaluations;
    if (!diagnostics.converged) diagnostics.reason = StopReason::IterationCap;
    return make_result(b, diagnostics);
}

}  // namespace pnl::numerics
