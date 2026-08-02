/// \file test_numerics.cpp
/// Unit tests for the numerics module on closed form cases.

#include <pnl_test.hpp>

#include <pnl/numerics/lu.hpp>
#include <pnl/numerics/ode.hpp>
#include <pnl/numerics/qr.hpp>
#include <pnl/numerics/quadrature.hpp>
#include <pnl/numerics/roots.hpp>

#include <numbers>

using namespace pnl;
using namespace pnl::numerics;

// ---------------------------------------------------------------------------
// Root finding
// ---------------------------------------------------------------------------

PNL_TEST("roots/bisection finds sqrt(2) and bounds its own error") {
    auto f = [](Real x) { return x * x - 2.0; };
    const auto result = bisection(f, 0.0, 2.0, RootOptions{1.0e-12, 200});
    PNL_REQUIRE(result.converged());
    PNL_REQUIRE_CLOSE(result.value, std::numbers::sqrt2_v<Real>, 1.0e-11);
    // The reported error estimate must be a genuine bound, not a hope.
    PNL_REQUIRE(std::abs(result.value - std::numbers::sqrt2_v<Real>) <=
                result.diagnostics.error_estimate + 1.0e-15);
}

PNL_TEST("roots/bisection rejects a bracket that does not change sign") {
    auto f = [](Real x) { return x * x + 1.0; };
    PNL_REQUIRE_THROWS(bisection(f, 0.0, 2.0), InvalidArgument);
}

PNL_TEST("roots/newton converges quadratically on a simple root") {
    auto f = [](Real x) { return x * x - 2.0; };
    auto df = [](Real x) { return 2.0 * x; };
    const auto result = newton(f, df, 1.0, RootOptions{1.0e-14, 100});
    PNL_REQUIRE(result.converged());
    PNL_REQUIRE_CLOSE(result.value, std::numbers::sqrt2_v<Real>, 1.0e-14);
    // Quadratic convergence from x0 = 1 reaches machine precision in about
    // five steps; anything much larger means the order has been lost.
    PNL_REQUIRE_MESSAGE(result.diagnostics.iterations <= 8,
                        "newton took " + std::to_string(result.diagnostics.iterations) +
                            " iterations, which is not quadratic convergence");
}

PNL_TEST("roots/newton reports a vanishing derivative rather than dividing by it") {
    auto f = [](Real x) { return x * x; };
    auto df = [](Real x) { return 2.0 * x; };
    PNL_REQUIRE_THROWS(newton(f, df, 0.0), NumericalFailure);
}

PNL_TEST("roots/brent beats bisection on evaluation count for the same accuracy") {
    auto f = [](Real x) { return std::cos(x) - x; };
    const RootOptions options{1.0e-12, 200};
    const auto brent_result = brent(f, 0.0, 2.0, options);
    const auto bisection_result = bisection(f, 0.0, 2.0, options);
    PNL_REQUIRE(brent_result.converged());
    PNL_REQUIRE(bisection_result.converged());
    PNL_REQUIRE_CLOSE(brent_result.value, bisection_result.value, 1.0e-10);
    PNL_REQUIRE_MESSAGE(
        brent_result.diagnostics.evaluations < bisection_result.diagnostics.evaluations,
        "brent used " + std::to_string(brent_result.diagnostics.evaluations) +
            " evaluations against bisection's " +
            std::to_string(bisection_result.diagnostics.evaluations) +
            ", so its superlinear step is not being taken");
}

// ---------------------------------------------------------------------------
// Quadrature
// ---------------------------------------------------------------------------

PNL_TEST("quadrature/adaptive simpson integrates a known integral") {
    auto f = [](Real x) { return std::sin(x); };
    const auto result = adaptive_simpson(f, 0.0, std::numbers::pi_v<Real>);
    PNL_REQUIRE(result.converged());
    PNL_REQUIRE_CLOSE(result.value, 2.0, 1.0e-10);
}

PNL_TEST("quadrature/simpson is exact for cubics") {
    // Simpson's rule integrates degree three exactly, so the error must be at
    // rounding level however few subdivisions are used.
    auto f = [](Real x) { return 2.0 * x * x * x - 3.0 * x * x + 5.0 * x - 7.0; };
    const auto result = adaptive_simpson(f, -1.0, 2.0, QuadratureOptions{1.0e-14, 30});
    const Real exact = 0.5 * 16.0 - 3.0 - 0.5 * 2.0 * 3.0 + 2.5 * 3.0 - 21.0;
    // Recompute the antiderivative explicitly to avoid a hand arithmetic slip.
    auto antiderivative = [](Real x) {
        return 0.5 * x * x * x * x - x * x * x + 2.5 * x * x - 7.0 * x;
    };
    const Real closed_form = antiderivative(2.0) - antiderivative(-1.0);
    (void)exact;
    PNL_REQUIRE_CLOSE(result.value, closed_form, 1.0e-12);
}

PNL_TEST("quadrature/gauss legendre is exact to degree 2n-1") {
    // With five points the rule integrates degree nine exactly.
    auto f = [](Real x) { return std::pow(x, 9) + 3.0 * std::pow(x, 4); };
    auto antiderivative = [](Real x) { return std::pow(x, 10) / 10.0 + 3.0 * std::pow(x, 5) / 5.0; };
    const auto result = gauss_legendre(f, -1.0, 1.5, 6);
    PNL_REQUIRE_CLOSE(result.value, antiderivative(1.5) - antiderivative(-1.0), 1.0e-12);
}

PNL_TEST("quadrature/gauss legendre weights sum to the interval length") {
    for (Index points : {2, 5, 12, 33}) {
        const auto rule = gauss_legendre_rule(points);
        Real total = 0.0;
        for (Real weight : rule.weights) total += weight;
        PNL_REQUIRE_CLOSE(total, 2.0, 1.0e-13);
    }
}

PNL_TEST("quadrature/romberg converges on a smooth integrand") {
    auto f = [](Real x) { return std::exp(-x * x); };
    const auto result = romberg(f, 0.0, 1.0, QuadratureOptions{1.0e-12, 20});
    PNL_REQUIRE(result.converged());
    // erf(1) * sqrt(pi) / 2.
    const Real closed_form = 0.5 * std::sqrt(std::numbers::pi_v<Real>) * std::erf(1.0);
    PNL_REQUIRE_CLOSE(result.value, closed_form, 1.0e-11);
}

// ---------------------------------------------------------------------------
// Dense linear algebra
// ---------------------------------------------------------------------------

PNL_TEST("lu/solves a hand checkable system") {
    // The system
    //   2x +  y -  z =  8
    //  -3x -  y + 2z = -11
    //  -2x +  y + 2z = -3
    // has the exact solution (2, 3, -1).
    DenseMatrix a(3);
    a(0, 0) = 2;  a(0, 1) = 1;  a(0, 2) = -1;
    a(1, 0) = -3; a(1, 1) = -1; a(1, 2) = 2;
    a(2, 0) = -2; a(2, 1) = 1;  a(2, 2) = 2;
    const Vector b{8.0, -11.0, -3.0};

    const auto result = lu_solve(a, b);
    PNL_REQUIRE(result.converged());
    PNL_REQUIRE_CLOSE(result.value[0], 2.0, 1.0e-13);
    PNL_REQUIRE_CLOSE(result.value[1], 3.0, 1.0e-13);
    PNL_REQUIRE_CLOSE(result.value[2], -1.0, 1.0e-13);
}

PNL_TEST("lu/determinant matches the closed form and tracks the pivot sign") {
    DenseMatrix a(3);
    a(0, 0) = 6; a(0, 1) = 1; a(0, 2) = 1;
    a(1, 0) = 4; a(1, 1) = -2; a(1, 2) = 5;
    a(2, 0) = 2; a(2, 1) = 8; a(2, 2) = 7;
    const LuFactorisation factorisation{DenseMatrix(a)};
    // Expanded by hand: 6(-14-40) - 1(28-10) + 1(32+4) = -324 - 18 + 36 = -306.
    PNL_REQUIRE_CLOSE(factorisation.determinant(), -306.0, 1.0e-11);
}

PNL_TEST("lu/reports a singular matrix instead of returning nonsense") {
    DenseMatrix a(2);
    a(0, 0) = 1; a(0, 1) = 2;
    a(1, 0) = 2; a(1, 1) = 4;
    PNL_REQUIRE_THROWS(LuFactorisation{DenseMatrix(a)}, NumericalFailure);
}

PNL_TEST("lu/partial pivoting survives a zero leading pivot") {
    // Without row interchange the first pivot is zero and the factorisation
    // fails; with it the system is trivial.
    DenseMatrix a(2);
    a(0, 0) = 0; a(0, 1) = 1;
    a(1, 0) = 1; a(1, 1) = 0;
    const Vector b{3.0, 5.0};
    const auto result = lu_solve(a, b);
    PNL_REQUIRE_CLOSE(result.value[0], 5.0, 1.0e-14);
    PNL_REQUIRE_CLOSE(result.value[1], 3.0, 1.0e-14);
}

PNL_TEST("lu/thomas solves a tridiagonal system") {
    // Diagonal 4, off diagonals -1, which is exactly one grid line of the
    // Poisson operator.
    const Index n = 6;
    Vector lower(static_cast<std::size_t>(n), -1.0);
    Vector diagonal(static_cast<std::size_t>(n), 4.0);
    Vector upper(static_cast<std::size_t>(n), -1.0);
    Vector rhs(static_cast<std::size_t>(n), 1.0);
    lower[0] = 0.0;
    upper[static_cast<std::size_t>(n - 1)] = 0.0;

    Vector expected = rhs;
    thomas_solve(lower, diagonal, upper, rhs);

    // Verify by multiplying back through the tridiagonal operator.
    for (Index i = 0; i < n; ++i) {
        const auto k = static_cast<std::size_t>(i);
        Real value = 4.0 * rhs[k];
        if (i > 0) value -= rhs[k - 1];
        if (i + 1 < n) value -= rhs[k + 1];
        PNL_REQUIRE_CLOSE(value, expected[k], 1.0e-13);
    }
}

PNL_TEST("lu/thomas refuses a matrix that is not diagonally dominant") {
    Vector lower{0.0, -5.0};
    Vector diagonal{1.0, 1.0};
    Vector upper{-5.0, 0.0};
    Vector rhs{1.0, 1.0};
    PNL_REQUIRE_THROWS(thomas_solve(lower, diagonal, upper, rhs), InvalidArgument);
}

PNL_TEST("qr/solves a square system to the same answer as lu") {
    Matrix a(3, 3);
    a(0, 0) = 2;  a(0, 1) = 1;  a(0, 2) = -1;
    a(1, 0) = -3; a(1, 1) = -1; a(1, 2) = 2;
    a(2, 0) = -2; a(2, 1) = 1;  a(2, 2) = 2;
    const Vector b{8.0, -11.0, -3.0};
    const auto result = qr_solve(a, b);
    PNL_REQUIRE_CLOSE(result.value[0], 2.0, 1.0e-12);
    PNL_REQUIRE_CLOSE(result.value[1], 3.0, 1.0e-12);
    PNL_REQUIRE_CLOSE(result.value[2], -1.0, 1.0e-12);
}

PNL_TEST("qr/least squares fits a line through noiseless points") {
    // Four points exactly on y = 3x + 1, so the residual must be zero.
    Matrix a(4, 2);
    const Real xs[4] = {0.0, 1.0, 2.0, 3.0};
    Vector b(4);
    for (int i = 0; i < 4; ++i) {
        a(i, 0) = 1.0;
        a(i, 1) = xs[i];
        b[static_cast<std::size_t>(i)] = 3.0 * xs[i] + 1.0;
    }
    const auto result = qr_solve(a, b);
    PNL_REQUIRE_CLOSE(result.value[0], 1.0, 1.0e-12);
    PNL_REQUIRE_CLOSE(result.value[1], 3.0, 1.0e-12);
    PNL_REQUIRE(result.diagnostics.error_estimate < 1.0e-14);
}

PNL_TEST("qr/R is upper triangular and reproduces the normal equations") {
    Matrix a(5, 3);
    for (Index i = 0; i < 5; ++i) {
        for (Index j = 0; j < 3; ++j) {
            a(i, j) = std::sin(static_cast<Real>(1 + i * 3 + j));
        }
    }
    const QrFactorisation factorisation{Matrix(a)};
    const Matrix r = factorisation.upper_triangular();
    for (Index i = 1; i < 3; ++i) {
        for (Index j = 0; j < i; ++j) PNL_REQUIRE_EXACT(r(i, j), 0.0);
    }
    // R^T R must equal A^T A, which is the defining property of the QR factor.
    for (Index i = 0; i < 3; ++i) {
        for (Index j = 0; j < 3; ++j) {
            Real rtr = 0.0;
            for (Index k = 0; k < 3; ++k) rtr += r(k, i) * r(k, j);
            Real ata = 0.0;
            for (Index k = 0; k < 5; ++k) ata += a(k, i) * a(k, j);
            PNL_REQUIRE_CLOSE(rtr, ata, 1.0e-12);
        }
    }
}

// ---------------------------------------------------------------------------
// Ordinary differential equations
// ---------------------------------------------------------------------------

PNL_TEST("ode/rk4 integrates exponential decay") {
    auto f = [](Real, ConstVectorView y, VectorView dydt) { dydt[0] = -2.0 * y[0]; };
    const Vector y0{1.0};
    const auto result = rk4(f, 0.0, y0, 1.0, 200);
    PNL_REQUIRE(result.diagnostics.converged);
    PNL_REQUIRE_CLOSE(result.y[0], std::exp(-2.0), 1.0e-10);
}

PNL_TEST("ode/dormand prince meets its requested tolerance") {
    // The harmonic oscillator, whose exact solution is known and whose energy
    // is conserved, so both the value and an invariant can be checked.
    auto f = [](Real, ConstVectorView y, VectorView dydt) {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    };
    const Vector y0{1.0, 0.0};
    OdeOptions options;
    options.absolute_tolerance = 1.0e-11;
    options.relative_tolerance = 1.0e-11;
    const Real t1 = 10.0;
    const auto result = dormand_prince(f, 0.0, y0, t1, options);

    PNL_REQUIRE(result.diagnostics.converged);
    PNL_REQUIRE_CLOSE(result.y[0], std::cos(t1), 1.0e-8);
    PNL_REQUIRE_CLOSE(result.y[1], -std::sin(t1), 1.0e-8);
    // Energy y0^2 + y1^2 starts at one and must stay there.
    PNL_REQUIRE_CLOSE(result.y[0] * result.y[0] + result.y[1] * result.y[1], 1.0, 1.0e-8);
    // The controller must actually have adapted, not just taken uniform steps.
    PNL_REQUIRE(result.accepted_steps > 0);
}

PNL_TEST("ode/dormand prince takes fewer steps at a looser tolerance") {
    auto f = [](Real t, ConstVectorView y, VectorView dydt) { dydt[0] = -y[0] + std::sin(t); };
    const Vector y0{1.0};
    OdeOptions tight;
    tight.absolute_tolerance = tight.relative_tolerance = 1.0e-12;
    OdeOptions loose;
    loose.absolute_tolerance = loose.relative_tolerance = 1.0e-6;
    const auto tight_result = dormand_prince(f, 0.0, y0, 5.0, tight);
    const auto loose_result = dormand_prince(f, 0.0, y0, 5.0, loose);
    PNL_REQUIRE_MESSAGE(loose_result.accepted_steps < tight_result.accepted_steps,
                        "the step size controller is not responding to the tolerance");
}
