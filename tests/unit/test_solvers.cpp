/// \file test_solvers.cpp
/// Every solver in the zoo on hand checkable systems.
///
/// Section 10 asks for each solver on a small system whose answer can be
/// verified by hand. The 4 by 4 case below is strictly diagonally dominant and
/// symmetric, so every method in the zoo including conjugate gradient applies to
/// it, and its solution is known because the right hand side was built from it.

#include <pnl_test.hpp>

#include <pnl/backend/serial.hpp>
#include <pnl/problems/dense_generator.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

using namespace pnl;
using namespace pnl::solvers;

namespace {

/// A serial backend for the unit level, where parallelism is not under test.
backend::SerialBackend make_serial() {
    return backend::SerialBackend{backend::Config{}};
}

SolverOptions tight_options() {
    SolverOptions options;
    options.tolerance = 1.0e-10;
    options.max_iterations = 200000;
    options.check_interval = 1;
    return options;
}

}  // namespace

PNL_TEST("solvers/every solver in the registry solves a 4x4 system") {
    // A symmetric, strictly diagonally dominant 4 by 4 system, small enough to
    // check by hand and admissible for every method including CG.
    auto problem = problems::DenseProblem(4, 12345, problems::DenseKind::SymmetricPositiveDefinite,
                                          2);
    auto serial = make_serial();
    const ConstVectorView exact = problem.exact_solution();

    int checked = 0;
    for (const auto& solver : all_solvers()) {
        if (!solver->applicable_to(problem)) {
            // Only the red black variants should decline a dense system.
            PNL_REQUIRE_MESSAGE(
                std::string(solver->name()).find("_rb") != std::string::npos,
                std::string("solver ") + std::string(solver->name()) +
                    " unexpectedly declined a symmetric positive definite dense system");
            continue;
        }
        const auto result = solver->solve(problem, serial, tight_options());
        PNL_REQUIRE_MESSAGE(result.converged(),
                            std::string("solver ") + std::string(solver->name()) +
                                " did not converge on the 4 by 4 system, stopping because " +
                                std::string(to_string(result.diagnostics.reason)));
        for (Index i = 0; i < problem.unknown_count(); ++i) {
            const auto k = static_cast<std::size_t>(i);
            PNL_REQUIRE_MESSAGE(
                test::close_absolute(result.solution[k], exact[k], 1.0e-7),
                std::string("solver ") + std::string(solver->name()) + " component " +
                    std::to_string(i) + " is " + test::format(result.solution[k]) +
                    " but should be " + test::format(exact[k]));
        }
        ++checked;
    }
    PNL_REQUIRE_MESSAGE(checked >= 10, "expected at least ten applicable solvers, ran " +
                                           std::to_string(checked));
}

PNL_TEST("solvers/every solver in the registry solves the Poisson problem") {
    problems::Poisson2D problem(15);
    auto serial = make_serial();

    SolverOptions options = tight_options();
    options.tolerance = 1.0e-9;

    // Solve once with a direct enough method to have a reference: SOR at the
    // closed form optimum converges fastest here.
    auto reference_solver = make_solver("sor");
    const auto reference = reference_solver->solve(problem, serial, options);
    PNL_REQUIRE(reference.converged());

    for (const auto& solver : all_solvers()) {
        if (!solver->applicable_to(problem)) continue;
        const auto result = solver->solve(problem, serial, options);
        PNL_REQUIRE_MESSAGE(result.converged(),
                            std::string("solver ") + std::string(solver->name()) +
                                " did not converge on the 15 by 15 Poisson problem");
        // All methods solve the same linear system, so they must agree on its
        // solution to well within the tolerance they each stopped at.
        for (Index i = 1; i <= problem.side(); ++i) {
            for (Index j = 1; j <= problem.side(); ++j) {
                const auto k = static_cast<std::size_t>(problem.at(i, j));
                PNL_REQUIRE_MESSAGE(
                    test::close_absolute(result.solution[k], reference.solution[k], 1.0e-6),
                    std::string("solver ") + std::string(solver->name()) +
                        " disagrees with the reference solution at grid point (" +
                        std::to_string(i) + ", " + std::to_string(j) + ")");
            }
        }
    }
}

PNL_TEST("solvers/conjugate gradient refuses a system that is not symmetric") {
    problems::DenseProblem problem(8, 999, problems::DenseKind::DiagonallyDominant, 2);
    auto serial = make_serial();
    auto solver = make_solver("cg");
    PNL_REQUIRE(!solver->applicable_to(problem));
    PNL_REQUIRE_THROWS(solver->solve(problem, serial, tight_options()), InvalidArgument);
}

PNL_TEST("solvers/red black methods refuse a dense system") {
    problems::DenseProblem problem(8, 999, problems::DenseKind::SymmetricPositiveDefinite, 2);
    auto serial = make_serial();
    for (const char* name : {"gauss_seidel_rb", "sor_rb"}) {
        auto solver = make_solver(name);
        PNL_REQUIRE(!solver->applicable_to(problem));
        PNL_REQUIRE_THROWS(solver->solve(problem, serial, tight_options()), InvalidArgument);
    }
}

PNL_TEST("solvers/SOR rejects a relaxation factor outside the Kahan interval") {
    problems::Poisson2D problem(7);
    auto serial = make_serial();
    auto solver = make_solver("sor");
    SolverOptions options = tight_options();
    for (Real omega : {2.5, 3.0}) {
        options.relaxation = omega;
        PNL_REQUIRE_THROWS(solver->solve(problem, serial, options), InvalidArgument);
    }
}

PNL_TEST("solvers/conjugate gradient terminates within n steps in exact arithmetic") {
    // The Krylov argument says CG reaches the exact solution in at most n
    // iterations. Rounding spoils that in general, but on a small well
    // conditioned system the bound should still hold comfortably.
    const Index n = 20;
    problems::DenseProblem problem(n, 4242, problems::DenseKind::SymmetricPositiveDefinite, 4);
    auto serial = make_serial();
    auto solver = make_solver("cg");
    SolverOptions options = tight_options();
    options.tolerance = 1.0e-12;
    const auto result = solver->solve(problem, serial, options);
    PNL_REQUIRE(result.converged());
    PNL_REQUIRE_MESSAGE(result.diagnostics.iterations <= n,
                        "conjugate gradient took " +
                            std::to_string(result.diagnostics.iterations) +
                            " iterations on an order " + std::to_string(n) +
                            " system, exceeding the Krylov bound");
}

PNL_TEST("solvers/a non converged result reports itself as such") {
    problems::Poisson2D problem(63);
    auto serial = make_serial();
    auto solver = make_solver("jacobi");
    SolverOptions options;
    options.max_iterations = 5;  // Nowhere near enough.
    options.tolerance = 1.0e-12;
    const auto result = solver->solve(problem, serial, options);
    PNL_REQUIRE(!result.converged());
    PNL_REQUIRE(result.diagnostics.reason == StopReason::IterationCap);
    PNL_REQUIRE_THROWS(result.diagnostics.require_converged("jacobi"), ConvergenceFailure);
}

PNL_TEST("solvers/fixed iteration mode runs exactly the requested count") {
    problems::Poisson2D problem(31);
    auto serial = make_serial();
    auto solver = make_solver("jacobi");
    SolverOptions options;
    options.mode = RunMode::FixedIterations;
    options.max_iterations = 37;
    options.check_interval = 1000000;  // Never checks, so nothing can stop early.
    const auto result = solver->solve(problem, serial, options);
    PNL_REQUIRE_EXACT(static_cast<Real>(result.diagnostics.iterations), 37.0);
    PNL_REQUIRE(!result.converged());
}

PNL_TEST("solvers/the Poisson solution matches the manufactured solution to O(h^2)") {
    // With the algebraic residual driven far below the discretisation error,
    // what is left is the truncation error of the five point stencil, which for
    // u = sin(pi x) sin(pi y) has the closed form leading constant pi^2 / 12.
    for (Index n : {31, 63}) {
        problems::Poisson2D problem(n);
        auto serial = make_serial();
        auto solver = make_solver("sor");
        SolverOptions options;
        // See the note in the convergence suite: 1e-11 is well below the
        // discretisation error and above the measured rounding floor.
        options.tolerance = 1.0e-11;
        options.max_iterations = 100000;
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
        const Real h = 1.0 / static_cast<Real>(n + 1);
        const Real predicted = std::numbers::pi_v<Real> * std::numbers::pi_v<Real> / 12.0 * h * h;
        PNL_REQUIRE_MESSAGE(
            test::close_relative(worst, predicted, 0.05),
            "at n = " + std::to_string(n) + " the discretisation error is " +
                test::format(worst) + " but the truncation analysis predicts " +
                test::format(predicted));
    }
}
