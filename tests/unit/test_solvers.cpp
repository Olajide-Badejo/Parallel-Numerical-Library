/// \file test_solvers.cpp
/// Every solver in the zoo on hand checkable systems.
///
/// Section 10 asks for each solver on a small system whose answer can be
/// verified by hand. The 4 by 4 case below is strictly diagonally dominant and
/// symmetric, so every method in the zoo including conjugate gradient applies to
/// it, and its solution is known because the right hand side was built from it.

#include <pnl/backend/serial.hpp>
#include <pnl/problems/dense_generator.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

#include <pnl_test.hpp>

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
    auto problem =
        problems::DenseProblem(4, 12345, problems::DenseKind::SymmetricPositiveDefinite, 2);
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
            PNL_REQUIRE_MESSAGE(test::close_absolute(result.solution[k], exact[k], 1.0e-7),
                                std::string("solver ") + std::string(solver->name()) +
                                    " component " + std::to_string(i) + " is " +
                                    test::format(result.solution[k]) + " but should be " +
                                    test::format(exact[k]));
        }
        ++checked;
    }
    PNL_REQUIRE_MESSAGE(checked >= 10,
                        "expected at least ten applicable solvers, ran " + std::to_string(checked));
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
                        "conjugate gradient took " + std::to_string(result.diagnostics.iterations) +
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

PNL_TEST("solvers/every solver reports its work unit and its operator applications") {
    // The result row says what work it timed, and this is where that claim is
    // checked. `sweeps` is updates per unknown per iteration, `passes` is
    // streams over the array, and they differ for exactly the red black
    // methods: one sweep of work in two passes over memory. `evaluations` is
    // every application of the operator that was performed, which for the
    // driven methods is the initial residual, one per sweep, and one per
    // residual the check interval asked for, and for Richardson and conjugate
    // gradient is one per iteration plus the initial residual.
    struct Expected {
        const char* name;
        Index sweeps;
        Index passes;
        Index evaluations;
        bool relaxed;
        /// Whether the reported residual is b - A x at the returned iterate.
        /// True for every method that measures it, and false for conjugate
        /// gradient alone, which reports the recurrence residual
        /// r_{k+1} = r_k - alpha A p_k. That is the standard choice and is what
        /// makes the method cost one operator application per iteration, but it
        /// drifts from the true residual as the recurrence rounds. See MEAS-05.
        bool true_residual;
    };

    // At ten fixed iterations with the residual checked every iteration.
    constexpr Expected table[] = {
        {"richardson", 1, 2, 11, true, true},
        {"jacobi", 1, 1, 21, false, true},
        {"gauss_seidel_f", 1, 1, 21, false, true},
        {"gauss_seidel_b", 1, 1, 21, false, true},
        {"gauss_seidel_s", 2, 2, 31, false, true},
        {"gauss_seidel_rb", 1, 2, 21, false, true},
        {"sor", 1, 1, 21, true, true},
        {"ssor", 2, 2, 31, true, true},
        {"sor_rb", 1, 2, 21, true, true},
        {"block_jacobi", 1, 2, 21, false, true},
        {"block_gauss_seidel", 1, 1, 21, false, true},
        {"cg", 1, 6, 11, false, false},
    };

    problems::Poisson2D problem(63);
    auto serial = make_serial();
    SolverOptions options;
    options.mode = RunMode::FixedIterations;
    options.max_iterations = 10;
    options.check_interval = 1;

    const auto names = all_solver_names();
    PNL_REQUIRE_MESSAGE(names.size() == std::size(table),
                        "the registry holds " + std::to_string(names.size()) +
                            " solvers but this test declares " + std::to_string(std::size(table)));

    Vector recomputed = problem.make_state();
    const Real rhs_norm = problem.rhs_norm(serial);

    std::printf("%-20s %7s %7s %13s %10s\n", "solver", "sweeps", "passes", "evaluations", "omega");
    for (const Expected& expected : table) {
        auto solver = make_solver(expected.name);
        auto result = solver->solve(problem, serial, options);
        const Diagnostics& d = result.diagnostics;
        const Real omega = solver->relaxation_factor(problem, options);
        std::printf("%-20s %7td %7td %13td %10s\n",
                    expected.name,
                    d.sweeps,
                    d.passes,
                    d.evaluations,
                    omega > 0.0 ? test::format(omega).c_str() : "");

        const std::string where = std::string("solver ") + expected.name;
        PNL_REQUIRE_MESSAGE(d.iterations == options.max_iterations,
                            where + " ran " + std::to_string(d.iterations) +
                                " iterations where a fixed run asked for " +
                                std::to_string(options.max_iterations));
        PNL_REQUIRE_MESSAGE(d.sweeps == expected.sweeps,
                            where + " reports " + std::to_string(d.sweeps) +
                                " sweeps per iteration where the work unit is " +
                                std::to_string(expected.sweeps));
        PNL_REQUIRE_MESSAGE(d.passes == expected.passes,
                            where + " reports " + std::to_string(d.passes) +
                                " passes per iteration where it makes " +
                                std::to_string(expected.passes));
        PNL_REQUIRE_MESSAGE(d.evaluations == expected.evaluations,
                            where + " reports " + std::to_string(d.evaluations) +
                                " operator applications where it performed " +
                                std::to_string(expected.evaluations));
        PNL_REQUIRE_MESSAGE(solver->work_unit().sweeps == expected.sweeps &&
                                solver->work_unit().passes == expected.passes,
                            where + " declares a work unit its result does not report");
        PNL_REQUIRE_MESSAGE(
            (omega > 0.0) == expected.relaxed,
            where + (expected.relaxed ? " reports no relaxation factor although it uses one"
                                      : " reports a relaxation factor it never uses"));

        // The fairness property the shared driver exists for: a method that
        // measures its residual measures it at the iterate it returns, not one
        // iteration earlier. This is what a residual hint threaded into the
        // driver would have broken for Richardson, and it is why Richardson
        // carries its residual vector instead.
        if (expected.true_residual) {
            const Real at_the_answer =
                problem.residual(serial, result.solution, recomputed) / rhs_norm;
            PNL_REQUIRE_MESSAGE(d.error_estimate == at_the_answer,
                                where + " reports relative residual " +
                                    test::format(d.error_estimate) +
                                    " but the residual at the iterate it returned is " +
                                    test::format(at_the_answer));
        }
    }
}

PNL_TEST("solvers/Richardson applies the operator once per iteration") {
    // MEAS-02. Richardson used to evaluate the residual inside its sweep and
    // then have the driver evaluate it again at the next iterate, which at the
    // default check interval is twice per iteration for a method that needs it
    // once. Carrying the residual across iterations removes the second one and
    // must not move an iterate: the residual is a pure function of the iterate,
    // so the same axpy runs over the same vector in the same order.
    problems::Poisson2D problem(31);
    auto serial = make_serial();
    auto richardson = make_solver("richardson");

    SolverOptions options;
    options.mode = RunMode::FixedIterations;
    options.max_iterations = 25;
    options.check_interval = 1;
    auto result = richardson->solve(problem, serial, options);

    PNL_REQUIRE_MESSAGE(result.diagnostics.evaluations == options.max_iterations + 1,
                        "richardson applied the operator " +
                            std::to_string(result.diagnostics.evaluations) + " times over " +
                            std::to_string(options.max_iterations) +
                            " iterations, where one per iteration plus the initial residual is " +
                            std::to_string(options.max_iterations + 1));

    // A residual carried across iterations is only sound if it is still the
    // residual at the iterate the run returns. Sharing one evaluation the naive
    // way would report Richardson at x_k while every other method reports at
    // x_{k+1}, which is a shift of one iteration and is invisible in any test
    // that only asks whether the method converged.
    Vector recomputed = problem.make_state();
    const Real at_the_answer =
        problem.residual(serial, result.solution, recomputed) / problem.rhs_norm(serial);
    PNL_REQUIRE_EXACT(result.diagnostics.error_estimate, at_the_answer);
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
        PNL_REQUIRE_MESSAGE(test::close_relative(worst, predicted, 0.05),
                            "at n = " + std::to_string(n) + " the discretisation error is " +
                                test::format(worst) + " but the truncation analysis predicts " +
                                test::format(predicted));
    }
}
