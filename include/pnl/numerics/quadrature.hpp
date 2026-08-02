#pragma once

/// \file quadrature.hpp
/// Numerical integration: adaptive Simpson, Gauss Legendre, and Romberg.
///
/// Reference: Burden and Faires, "Numerical Analysis", 10th ed., Cengage,
/// chapter 4; Davis and Rabinowitz, "Methods of Numerical Integration", 2nd
/// ed., Academic Press 1984, for the Gauss Legendre nodes and the Romberg
/// extrapolation table.

#include <pnl/core/diagnostics.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>
#include <pnl/numerics/roots.hpp>

#include <cmath>
#include <numbers>
#include <vector>

namespace pnl::numerics {

/// Stopping controls shared by the quadrature rules.
struct QuadratureOptions {
    Real tolerance = 1.0e-10;
    /// Maximum recursion depth for the adaptive rule, or maximum table size for
    /// Romberg.
    Index max_depth = 50;
};

namespace detail {

/// Simpson's rule on a single interval.
[[nodiscard]] inline Real simpson_rule(Real fa, Real fm, Real fb, Real h) {
    return h * (fa + 4.0 * fm + fb) / 6.0;
}

/// Recursive half of the adaptive Simpson rule.
inline Real adaptive_simpson_step(const ScalarFunction& f, Real a, Real b, Real fa, Real fm,
                                  Real fb, Real whole, Real tolerance, Index depth,
                                  Index max_depth, Index& evaluations, Index& deepest,
                                  bool& hit_depth_limit) {
    const Real m = 0.5 * (a + b);
    const Real lm = 0.5 * (a + m);
    const Real rm = 0.5 * (m + b);
    const Real flm = f(lm);
    const Real frm = f(rm);
    evaluations += 2;

    const Real left = simpson_rule(fa, flm, fm, m - a);
    const Real right = simpson_rule(fm, frm, fb, b - m);
    const Real difference = left + right - whole;

    deepest = std::max(deepest, depth);

    // The classical Richardson criterion for Simpson: the composite pair is
    // more accurate than the whole by a factor of 15, so difference / 15
    // estimates the error of the refined value.
    if (depth >= max_depth) {
        hit_depth_limit = true;
        return left + right + difference / 15.0;
    }
    if (std::abs(difference) <= 15.0 * tolerance) {
        return left + right + difference / 15.0;
    }

    return adaptive_simpson_step(f, a, m, fa, flm, fm, left, 0.5 * tolerance, depth + 1,
                                 max_depth, evaluations, deepest, hit_depth_limit) +
           adaptive_simpson_step(f, m, b, fm, frm, fb, right, 0.5 * tolerance, depth + 1,
                                 max_depth, evaluations, deepest, hit_depth_limit);
}

}  // namespace detail

/// Adaptive Simpson quadrature.
///
/// Convergence order: the underlying Simpson rule is O(h^4), exact for cubics.
/// Adaptivity concentrates evaluations where the fourth derivative is large, so
/// the achieved accuracy per evaluation is far better than the fixed rule on
/// functions with local structure.
///
/// error_estimate is the accumulated Richardson estimate; iterations reports the
/// deepest recursion reached and evaluations the true function call count.
[[nodiscard]] inline Result<Real> adaptive_simpson(const ScalarFunction& f, Real a, Real b,
                                                   const QuadratureOptions& options = {}) {
    require(b >= a, "adaptive_simpson needs an interval with b at least a");
    if (a == b) return make_result(0.0, Diagnostics{0.0, 0, 0, true, StopReason::Converged});

    const Real m = 0.5 * (a + b);
    const Real fa = f(a);
    const Real fm = f(m);
    const Real fb = f(b);
    Index evaluations = 3;
    Index deepest = 0;
    bool hit_depth_limit = false;

    const Real whole = detail::simpson_rule(fa, fm, fb, b - a);
    const Real value =
        detail::adaptive_simpson_step(f, a, b, fa, fm, fb, whole, options.tolerance, 1,
                                      options.max_depth, evaluations, deepest, hit_depth_limit);

    Diagnostics diagnostics;
    diagnostics.iterations = deepest;
    diagnostics.evaluations = evaluations;
    diagnostics.error_estimate = std::abs(value - whole) / 15.0;
    diagnostics.converged = !hit_depth_limit;
    diagnostics.reason = hit_depth_limit ? StopReason::IterationCap : StopReason::Converged;
    return make_result(value, diagnostics);
}

/// Gauss Legendre nodes and weights on [-1, 1] for a given point count.
///
/// The nodes are the roots of the Legendre polynomial P_n, found by Newton
/// iteration from the Chebyshev approximation to their positions, and the
/// weights follow from the standard formula 2 / ((1 - x^2) P_n'(x)^2). Computing
/// them rather than tabulating them means any order is available and the
/// derivation is visible.
struct GaussLegendreRule {
    Vector nodes;
    Vector weights;
};

/// \throws InvalidArgument if \p points is less than one.
[[nodiscard]] inline GaussLegendreRule gauss_legendre_rule(Index points) {
    require(points >= 1, "gauss_legendre_rule needs at least one point");
    GaussLegendreRule rule;
    rule.nodes.assign(static_cast<std::size_t>(points), 0.0);
    rule.weights.assign(static_cast<std::size_t>(points), 0.0);

    const Real pi = std::numbers::pi_v<Real>;
    const auto n = static_cast<Real>(points);

    for (Index i = 0; i < (points + 1) / 2; ++i) {
        // Chebyshev starting guess, accurate enough that Newton converges in a
        // handful of steps for every order used here.
        Real x = std::cos(pi * (static_cast<Real>(i) + 0.75) / (n + 0.5));
        Real derivative = 0.0;
        for (Index iteration = 0; iteration < 100; ++iteration) {
            // Legendre recurrence: (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}.
            Real p0 = 1.0;
            Real p1 = 0.0;
            for (Index k = 0; k < points; ++k) {
                const Real p2 = p1;
                p1 = p0;
                const auto kk = static_cast<Real>(k);
                p0 = ((2.0 * kk + 1.0) * x * p1 - kk * p2) / (kk + 1.0);
            }
            derivative = n * (x * p0 - p1) / (x * x - 1.0);
            const Real step = p0 / derivative;
            x -= step;
            if (std::abs(step) <= 2.0 * EPSILON * (1.0 + std::abs(x))) break;
        }
        const auto lo = static_cast<std::size_t>(i);
        const auto hi = static_cast<std::size_t>(points - 1 - i);
        rule.nodes[lo] = -x;
        rule.nodes[hi] = x;
        const Real weight = 2.0 / ((1.0 - x * x) * derivative * derivative);
        rule.weights[lo] = weight;
        rule.weights[hi] = weight;
    }
    return rule;
}

/// Fixed order Gauss Legendre quadrature on [a, b].
///
/// Convergence order: exact for polynomials of degree up to 2n-1 with n points,
/// which is the maximum any n point rule can achieve. For analytic integrands
/// the error falls geometrically in n rather than algebraically in h.
///
/// error_estimate compares the requested rule against the rule with half the
/// points, which for a smooth integrand is a conservative bound.
[[nodiscard]] inline Result<Real> gauss_legendre(const ScalarFunction& f, Real a, Real b,
                                                 Index points = 20) {
    require(points >= 2, "gauss_legendre needs at least two points to estimate its own error");
    const Real half = 0.5 * (b - a);
    const Real centre = 0.5 * (a + b);

    auto integrate = [&](const GaussLegendreRule& rule) {
        Real total = 0.0;
        for (std::size_t k = 0; k < rule.nodes.size(); ++k) {
            total += rule.weights[k] * f(centre + half * rule.nodes[k]);
        }
        return half * total;
    };

    const GaussLegendreRule full = gauss_legendre_rule(points);
    const GaussLegendreRule coarse = gauss_legendre_rule(points / 2);
    const Real value = integrate(full);
    const Real coarse_value = integrate(coarse);

    Diagnostics diagnostics;
    diagnostics.iterations = 1;
    diagnostics.evaluations = points + points / 2;
    diagnostics.error_estimate = std::abs(value - coarse_value);
    diagnostics.converged = true;
    diagnostics.reason = StopReason::Converged;
    return make_result(value, diagnostics);
}

/// Romberg integration.
///
/// Method: repeated trapezoid refinement combined by Richardson extrapolation.
/// Row k of the table is O(h^{2k+2}) accurate for a smooth integrand, so the
/// diagonal converges very fast; the price is that the whole scheme relies on
/// the Euler Maclaurin expansion and therefore degrades on integrands whose
/// derivatives are not bounded.
///
/// error_estimate is the difference between the last two diagonal entries.
[[nodiscard]] inline Result<Real> romberg(const ScalarFunction& f, Real a, Real b,
                                          const QuadratureOptions& options = {}) {
    const Index max_rows = std::min(options.max_depth, Index{25});
    require(max_rows >= 2, "romberg needs at least two rows");

    std::vector<Vector> table;
    table.reserve(static_cast<std::size_t>(max_rows));
    Index evaluations = 2;

    Real h = b - a;
    Vector first{0.5 * h * (f(a) + f(b))};
    table.push_back(first);

    Diagnostics diagnostics;
    Real value = table[0][0];
    Index row = 1;

    for (; row < max_rows; ++row) {
        h *= 0.5;
        // Add only the new midpoints, which is what makes Romberg cheap.
        Real sum = 0.0;
        const Index new_points = Index{1} << (row - 1);
        for (Index k = 0; k < new_points; ++k) {
            sum += f(a + (2.0 * static_cast<Real>(k) + 1.0) * h);
            ++evaluations;
        }
        Vector current(static_cast<std::size_t>(row + 1), 0.0);
        current[0] = 0.5 * table[static_cast<std::size_t>(row - 1)][0] + h * sum;

        Real power = 1.0;
        for (Index j = 1; j <= row; ++j) {
            power *= 4.0;
            const auto jj = static_cast<std::size_t>(j);
            current[jj] = current[jj - 1] +
                          (current[jj - 1] - table[static_cast<std::size_t>(row - 1)][jj - 1]) /
                              (power - 1.0);
        }
        table.push_back(current);

        const Real previous = value;
        value = current[static_cast<std::size_t>(row)];
        diagnostics.error_estimate = std::abs(value - previous);
        if (row >= 2 && diagnostics.error_estimate <= options.tolerance) {
            diagnostics.converged = true;
            diagnostics.reason = StopReason::Converged;
            break;
        }
    }

    diagnostics.iterations = row;
    diagnostics.evaluations = evaluations;
    if (!diagnostics.converged) diagnostics.reason = StopReason::IterationCap;
    return make_result(value, diagnostics);
}

}  // namespace pnl::numerics
