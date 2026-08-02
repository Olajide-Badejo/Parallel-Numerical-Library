/// \file test_convergence.cpp
/// Empirical convergence orders and the theory ratios of Section 10.
///
/// These are the tests that make the report's mathematical claims checkable
/// rather than decorative. Each one measures something the theory predicts in
/// closed form and compares against that prediction, so a regression in the
/// numerics shows up as a failed theorem and not merely as a slower run.

#include <pnl_test.hpp>

#include <pnl/backend/serial.hpp>
#include <pnl/numerics/ode.hpp>
#include <pnl/numerics/quadrature.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

#include <numbers>

using namespace pnl;
using namespace pnl::solvers;

namespace {

backend::SerialBackend make_serial() { return backend::SerialBackend{backend::Config{}}; }

/// Slope of a straight line fitted to (log h, log error), which estimates the
/// convergence order.
[[nodiscard]] Real fitted_order(const std::vector<Real>& steps,
                                const std::vector<Real>& errors) {
    const auto n = static_cast<Real>(steps.size());
    Real sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (std::size_t i = 0; i < steps.size(); ++i) {
        const Real x = std::log(steps[i]);
        const Real y = std::log(errors[i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    return (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
}

/// Iterations a solver needs, measured exactly.
[[nodiscard]] Index iterations_for(const std::string& name, problems::Poisson2D& problem,
                                   Real tolerance, Real omega = 0.0) {
    auto serial = make_serial();
    auto solver = make_solver(name);
    SolverOptions options;
    options.tolerance = tolerance;
    options.max_iterations = 2000000;
    options.check_interval = 1;
    options.relaxation = omega;
    const auto result = solver->solve(problem, serial, options);
    if (!result.converged()) return -1;
    return result.diagnostics.iterations;
}

}  // namespace

// ---------------------------------------------------------------------------
// Classical convergence orders
// ---------------------------------------------------------------------------

PNL_TEST("convergence/rk4 exhibits fourth order global error") {
    auto f = [](Real, ConstVectorView y, VectorView dydt) { dydt[0] = -2.0 * y[0]; };
    const Vector y0{1.0};
    const Real exact = std::exp(-2.0);

    std::vector<Real> steps;
    std::vector<Real> errors;
    for (Index n : {10, 20, 40, 80}) {
        const auto result = numerics::rk4(f, 0.0, y0, 1.0, n);
        steps.push_back(1.0 / static_cast<Real>(n));
        errors.push_back(std::abs(result.y[0] - exact));
    }
    const Real order = fitted_order(steps, errors);
    PNL_REQUIRE_MESSAGE(test::close_absolute(order, 4.0, 0.15),
                        "rk4 measured order " + test::format(order) + ", expected 4");
}

PNL_TEST("convergence/composite simpson exhibits fourth order error") {
    // The adaptive rule hides the order behind its refinement, so the order is
    // measured on the fixed composite rule the adaptive one is built from.
    auto f = [](Real x) { return std::exp(x); };
    const Real exact = std::numbers::e_v<Real> - 1.0;

    std::vector<Real> steps;
    std::vector<Real> errors;
    for (Index panels : {4, 8, 16, 32}) {
        const Real h = 1.0 / static_cast<Real>(panels);
        Real total = 0.0;
        for (Index k = 0; k < panels; ++k) {
            const Real a = static_cast<Real>(k) * h;
            const Real b = a + h;
            total += (b - a) / 6.0 * (f(a) + 4.0 * f(0.5 * (a + b)) + f(b));
        }
        steps.push_back(h);
        errors.push_back(std::abs(total - exact));
    }
    const Real order = fitted_order(steps, errors);
    PNL_REQUIRE_MESSAGE(test::close_absolute(order, 4.0, 0.15),
                        "simpson measured order " + test::format(order) + ", expected 4");
}

PNL_TEST("convergence/the five point stencil is second order accurate") {
    std::vector<Real> steps;
    std::vector<Real> errors;
    for (Index n : {15, 31, 63}) {
        problems::Poisson2D problem(n);
        auto serial = make_serial();
        auto solver = make_solver("sor");
        SolverOptions options;
        // Far below the discretisation error being measured, which is O(1e-4)
        // here, and comfortably above the rounding floor of a stationary
        // iteration on this operator, which was measured at 3e-13 at n = 63.
        // Asking for 1e-13 would be asking the iteration to beat its own
        // arithmetic.
        options.tolerance = 1.0e-11;
        options.max_iterations = 200000;
        const auto result = solver->solve(problem, serial, options);
        PNL_REQUIRE(result.converged());

        const ConstVectorView exact = problem.exact_solution();
        Real worst = 0.0;
        for (Index i = 1; i <= n; ++i) {
            for (Index j = 1; j <= n; ++j) {
                const auto k = static_cast<std::size_t>(problem.at(i, j));
                worst = std::max(worst, std::abs(result.solution[k] - exact[k]));
            }
        }
        steps.push_back(1.0 / static_cast<Real>(n + 1));
        errors.push_back(worst);
    }
    const Real order = fitted_order(steps, errors);
    PNL_REQUIRE_MESSAGE(test::close_absolute(order, 2.0, 0.1),
                        "the discretisation measured order " + test::format(order) +
                            ", expected 2");
}

// ---------------------------------------------------------------------------
// Theory ratios for the splitting family
// ---------------------------------------------------------------------------

PNL_TEST("convergence/the Jacobi spectral radius matches cos(pi h)") {
    // Measure the asymptotic contraction factor directly from the residual
    // history and compare against the closed form.
    for (Index n : {31, 63}) {
        problems::Poisson2D problem(n);
        auto serial = make_serial();
        auto solver = make_solver("jacobi");
        SolverOptions options;
        options.tolerance = 1.0e-6;
        options.max_iterations = 200000;
        options.check_interval = 1;
        options.record_history = true;
        const auto result = solver->solve(problem, serial, options);
        PNL_REQUIRE(result.converged());

        const Real measured = result.convergence_factor(200);
        const Real predicted = problem.theory().jacobi_spectral_radius;
        PNL_REQUIRE_MESSAGE(
            test::close_relative(measured, predicted, 1.0e-3),
            "at n = " + std::to_string(n) + " the measured Jacobi contraction factor is " +
                test::format(measured) + " but cos(pi h) is " + test::format(predicted));
    }
}

PNL_TEST("convergence/Gauss Seidel needs half the iterations of Jacobi") {
    // Young's theorem: for a consistently ordered matrix the Gauss Seidel
    // iteration matrix has spectral radius equal to the square of the Jacobi
    // one, so the iteration counts stand in the ratio two to one.
    for (Index n : {31, 63}) {
        problems::Poisson2D problem(n);
        const Index jacobi = iterations_for("jacobi", problem, 1.0e-8);
        const Index gauss_seidel = iterations_for("gauss_seidel_f", problem, 1.0e-8);
        PNL_REQUIRE(jacobi > 0 && gauss_seidel > 0);

        const Real ratio = static_cast<Real>(jacobi) / static_cast<Real>(gauss_seidel);
        const Real predicted = problem.theory().jacobi_to_gauss_seidel_ratio();
        PNL_REQUIRE_MESSAGE(
            test::close_relative(ratio, predicted, 0.02),
            "at n = " + std::to_string(n) + " Jacobi took " + std::to_string(jacobi) +
                " iterations and Gauss Seidel " + std::to_string(gauss_seidel) +
                ", a ratio of " + test::format(ratio) + " against the predicted " +
                test::format(predicted));
        PNL_REQUIRE_MESSAGE(test::close_absolute(predicted, 2.0, 0.01),
                            "the predicted ratio should itself be two");
    }
}

PNL_TEST("convergence/forward and backward Gauss Seidel converge at the same rate") {
    // On a symmetric matrix the two iteration matrices are similar, so their
    // spectral radii and hence their iteration counts must agree.
    problems::Poisson2D problem(31);
    const Index forward = iterations_for("gauss_seidel_f", problem, 1.0e-8);
    const Index backward = iterations_for("gauss_seidel_b", problem, 1.0e-8);
    PNL_REQUIRE(forward > 0 && backward > 0);
    PNL_REQUIRE_MESSAGE(
        std::abs(forward - backward) <= std::max<Index>(2, forward / 100),
        "forward Gauss Seidel took " + std::to_string(forward) + " iterations and backward " +
            std::to_string(backward) + ", which should be equal on a symmetric operator");
}

PNL_TEST("convergence/SOR at the closed form optimum beats every nearby factor") {
    // Young's closed form omega* = 2 / (1 + sin(pi h)) is claimed to minimise
    // the spectral radius. Test it as a minimum: perturbing omega either way
    // must cost iterations.
    problems::Poisson2D problem(63);
    const Real optimal = problem.theory().optimal_relaxation;

    const Index at_optimum = iterations_for("sor", problem, 1.0e-8, optimal);
    PNL_REQUIRE(at_optimum > 0);

    for (Real delta : {-0.05, -0.02, 0.02, 0.05}) {
        const Real omega = optimal + delta;
        if (omega <= 0.0 || omega >= 2.0) continue;
        const Index perturbed = iterations_for("sor", problem, 1.0e-8, omega);
        PNL_REQUIRE(perturbed > 0);
        PNL_REQUIRE_MESSAGE(
            perturbed >= at_optimum,
            "SOR at omega = " + test::format(omega) + " took " + std::to_string(perturbed) +
                " iterations, fewer than the " + std::to_string(at_optimum) +
                " at the supposedly optimal " + test::format(optimal));
    }
}

PNL_TEST("convergence/optimal SOR changes the order of the iteration count") {
    // Jacobi and Gauss Seidel need O(n^2) iterations; optimal SOR needs O(n).
    // Doubling n should therefore roughly quadruple the first and merely double
    // the second, which is the single most important quantitative claim the
    // report makes about the stationary family.
    problems::Poisson2D coarse(31);
    problems::Poisson2D fine(63);

    const Index jacobi_coarse = iterations_for("jacobi", coarse, 1.0e-8);
    const Index jacobi_fine = iterations_for("jacobi", fine, 1.0e-8);
    const Index sor_coarse = iterations_for("sor", coarse, 1.0e-8);
    const Index sor_fine = iterations_for("sor", fine, 1.0e-8);
    PNL_REQUIRE(jacobi_coarse > 0 && jacobi_fine > 0 && sor_coarse > 0 && sor_fine > 0);

    const Real jacobi_growth = static_cast<Real>(jacobi_fine) / static_cast<Real>(jacobi_coarse);
    const Real sor_growth = static_cast<Real>(sor_fine) / static_cast<Real>(sor_coarse);

    PNL_REQUIRE_MESSAGE(jacobi_growth > 3.0,
                        "Jacobi iteration count grew by only " + test::format(jacobi_growth) +
                            " when the grid doubled, which is not the expected O(n^2)");
    PNL_REQUIRE_MESSAGE(sor_growth < 2.6,
                        "optimal SOR iteration count grew by " + test::format(sor_growth) +
                            " when the grid doubled, which is worse than the expected O(n)");
}

PNL_TEST("convergence/red black ordering costs iterations but keeps the rate order") {
    // Section 5 predicts the red black reordering converges at a typically
    // slightly worse per iteration rate than natural ordering while remaining
    // convergent. Measure that penalty rather than assert it.
    problems::Poisson2D problem(63);
    const Index natural = iterations_for("gauss_seidel_f", problem, 1.0e-8);
    const Index red_black = iterations_for("gauss_seidel_rb", problem, 1.0e-8);
    PNL_REQUIRE(natural > 0 && red_black > 0);

    const Real penalty = static_cast<Real>(red_black) / static_cast<Real>(natural);
    PNL_REQUIRE_MESSAGE(penalty >= 0.95 && penalty <= 1.3,
                        "red black Gauss Seidel took " + std::to_string(red_black) +
                            " iterations against natural ordering's " + std::to_string(natural) +
                            ", a ratio of " + test::format(penalty) +
                            " which is outside the expected small penalty");
}

PNL_TEST("convergence/line relaxation beats point relaxation") {
    // Solving each grid line exactly rather than each point should cut the
    // iteration count roughly in half at the same coupling structure.
    problems::Poisson2D problem(63);
    const Index point_jacobi = iterations_for("jacobi", problem, 1.0e-8);
    const Index line_jacobi = iterations_for("block_jacobi", problem, 1.0e-8);
    PNL_REQUIRE(point_jacobi > 0 && line_jacobi > 0);
    PNL_REQUIRE_MESSAGE(line_jacobi < point_jacobi,
                        "line Jacobi took " + std::to_string(line_jacobi) +
                            " iterations against point Jacobi's " + std::to_string(point_jacobi) +
                            ", so the exact line solve is buying nothing");

    const Index line_gauss_seidel = iterations_for("block_gauss_seidel", problem, 1.0e-8);
    PNL_REQUIRE(line_gauss_seidel > 0);
    const Real ratio = static_cast<Real>(line_jacobi) / static_cast<Real>(line_gauss_seidel);
    PNL_REQUIRE_MESSAGE(test::close_absolute(ratio, 2.0, 0.15),
                        "line Jacobi to line Gauss Seidel ratio is " + test::format(ratio) +
                            ", expected about two by the same argument as the point case");
}

PNL_TEST("convergence/conjugate gradient needs O(n) iterations on the model problem") {
    // The condition number of the five point stencil is O(h^-2), so the CG
    // bound gives O(h^-1) = O(n) iterations. Doubling the grid should roughly
    // double the count, not quadruple it.
    //
    // The spectrally rich right hand side is essential here. With the single
    // mode manufactured source the Krylov subspace is one dimensional and CG
    // converges in one iteration whatever the grid size, which measures the
    // degeneracy of the problem rather than the method.
    problems::Poisson2D coarse(31, problems::PoissonRhs::SpectrallyRich);
    problems::Poisson2D fine(63, problems::PoissonRhs::SpectrallyRich);
    const Index cg_coarse = iterations_for("cg", coarse, 1.0e-8);
    const Index cg_fine = iterations_for("cg", fine, 1.0e-8);
    PNL_REQUIRE(cg_coarse > 0 && cg_fine > 0);

    const Real growth = static_cast<Real>(cg_fine) / static_cast<Real>(cg_coarse);
    PNL_REQUIRE_MESSAGE(growth > 1.4 && growth < 2.6,
                        "conjugate gradient took " + std::to_string(cg_coarse) + " then " +
                            std::to_string(cg_fine) + " iterations, a growth of " +
                            test::format(growth) + " when the grid doubled, expected about two");
}

PNL_TEST("convergence/a single mode source makes conjugate gradient terminate at once") {
    // This records the trap rather than only avoiding it. The eigenvectors of
    // the five point stencil are sin(p pi x) sin(q pi y), so the manufactured
    // source is exactly one of them and the Krylov subspace it generates is one
    // dimensional. If this ever stops being true the manufactured solution has
    // changed and the discretisation error test needs revisiting.
    for (Index n : {31, 63, 127}) {
        problems::Poisson2D problem(n, problems::PoissonRhs::ManufacturedSine);
        const Index iterations = iterations_for("cg", problem, 1.0e-8);
        PNL_REQUIRE_MESSAGE(iterations == 1,
                            "conjugate gradient took " + std::to_string(iterations) +
                                " iterations on the single mode source at n = " +
                                std::to_string(n) + ", expected exactly one");
    }
}

PNL_TEST("convergence/Richardson converges from the Gershgorin step") {
    // Richardson with omega = 1 / bound must converge, since the bound
    // overestimates lambda_max and so the step is inside (0, 2 / lambda_max).
    problems::Poisson2D problem(31);
    const Index iterations = iterations_for("richardson", problem, 1.0e-6);
    PNL_REQUIRE_MESSAGE(iterations > 0,
                        "Richardson did not converge from its default Gershgorin step");

    // The step is 1/8 against the Jacobi 1/4, so it should need roughly twice
    // as many iterations as Jacobi, not diverge and not stall.
    const Index jacobi = iterations_for("jacobi", problem, 1.0e-6);
    const Real ratio = static_cast<Real>(iterations) / static_cast<Real>(jacobi);
    PNL_REQUIRE_MESSAGE(ratio > 1.2 && ratio < 3.0,
                        "Richardson took " + std::to_string(iterations) +
                            " iterations against Jacobi's " + std::to_string(jacobi) +
                            ", a ratio of " + test::format(ratio) +
                            " which is outside the expected factor of about two");
}
