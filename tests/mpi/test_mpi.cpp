/// \file test_mpi.cpp
/// Distributed correctness at 1, 2 and 4 ranks.
///
/// What is asserted, and what is deliberately not.
///
/// Within a fixed rank count the deterministic reduction is bit identical
/// across runs, because the per rank partials are gathered and summed in rank
/// order. Across rank counts it cannot be, because the rank boundaries are
/// chosen for load balance and so do not align with the fixed chunk grid the
/// shared memory backends use, which regroups the additions. Those comparisons
/// are therefore made to reduction tolerance, and the difference is a measured
/// property rather than a hidden one.
///
/// The natural ordering sweeps are the exception: they are asserted bit
/// identical against the serial backend at every rank count, because the
/// pipelined token chain reproduces the sequential recurrence exactly rather
/// than approximating it.

#include <pnl_test.hpp>

#include <pnl/backend/mpi.hpp>
#include <pnl/backend/serial.hpp>
#include <pnl/problems/dense_generator.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

#include <mpi.h>

#include <cstdio>
#include <numbers>

using namespace pnl;
using namespace pnl::solvers;

namespace {

int world_rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int world_size() {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

SolverOptions fixed_options(Index iterations = 20) {
    SolverOptions options;
    options.mode = RunMode::FixedIterations;
    options.max_iterations = iterations;
    options.check_interval = 1000000;
    return options;
}

/// Largest absolute difference between two vectors.
Real worst_difference(const Vector& a, const Vector& b) {
    Real worst = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
}

/// Solvers whose sweep is order independent, so a distributed run reproduces
/// the serial one exactly except for reduction regrouping.
const char* const ORDER_FREE_SOLVERS[] = {"jacobi", "gauss_seidel_rb", "sor_rb", "block_jacobi",
                                          "richardson"};

/// Solvers with a sequential recurrence, reproduced exactly by the token chain.
const char* const ORDERED_SOLVERS[] = {"gauss_seidel_f", "gauss_seidel_b", "gauss_seidel_s",
                                       "sor", "block_gauss_seidel"};

}  // namespace

PNL_TEST("mpi/the row decomposition covers the grid exactly once") {
    const int ranks = world_size();
    for (Index rows : {1, 7, 63, 64, 1000}) {
        Index covered = 0;
        Index previous_end = 0;
        for (int r = 0; r < ranks; ++r) {
            const Range band = block_partition(rows, ranks, r);
            PNL_REQUIRE(band.begin == previous_end);
            previous_end = band.end;
            covered += band.size();
        }
        PNL_REQUIRE_MESSAGE(covered == rows,
                            "at " + std::to_string(ranks) + " ranks the decomposition of " +
                                std::to_string(rows) + " rows covered " +
                                std::to_string(covered));
    }
}

PNL_TEST("mpi/order free solvers agree with serial on the Poisson problem") {
    // Sizes chosen so the row count leaves a remainder at 2 and 4 ranks, which
    // is where an off by one in the decomposition would show.
    for (Index n : {63, 65, 127}) {
        problems::Poisson2D problem(n, problems::PoissonRhs::SpectrallyRich);

        backend::Config serial_config;
        backend::SerialBackend serial(serial_config);

        backend::Config mpi_config;
        backend::MpiBackend distributed(mpi_config, backend::TopologyReport{});

        for (const char* name : ORDER_FREE_SOLVERS) {
            auto solver = make_solver(name);
            if (!solver->applicable_to(problem)) continue;
            const Vector expected = solver->solve(problem, serial, fixed_options()).solution;
            const Vector actual = solver->solve(problem, distributed, fixed_options()).solution;
            const Real difference = worst_difference(expected, actual);
            PNL_REQUIRE_MESSAGE(
                difference <= 1.0e-12,
                std::string("solver ") + name + " at n = " + std::to_string(n) + " on " +
                    std::to_string(world_size()) + " ranks differs from serial by " +
                    test::format(difference));
        }
    }
}

PNL_TEST("mpi/ordered solvers reproduce the sequential recurrence exactly") {
    // The pipelined token chain preserves natural ordering, so these must be
    // bit identical to serial, not merely close.
    for (Index n : {31, 63, 65}) {
        problems::Poisson2D problem(n, problems::PoissonRhs::SpectrallyRich);

        backend::Config serial_config;
        backend::SerialBackend serial(serial_config);

        backend::Config mpi_config;
        backend::MpiBackend distributed(mpi_config, backend::TopologyReport{});

        for (const char* name : ORDERED_SOLVERS) {
            auto solver = make_solver(name);
            if (!solver->applicable_to(problem)) continue;
            const Vector expected = solver->solve(problem, serial, fixed_options(10)).solution;
            const Vector actual = solver->solve(problem, distributed, fixed_options(10)).solution;
            const Real difference = worst_difference(expected, actual);
            PNL_REQUIRE_MESSAGE(
                difference == 0.0,
                std::string("solver ") + name + " at n = " + std::to_string(n) + " on " +
                    std::to_string(world_size()) +
                    " ranks is not bit identical to serial, worst difference " +
                    test::format(difference) +
                    "; the pipelined ordering is meant to be exact");
        }
    }
}

PNL_TEST("mpi/dense systems agree with serial") {
    problems::DenseProblem problem(150, 20260802,
                                   problems::DenseKind::SymmetricPositiveDefinite, 6);

    backend::Config serial_config;
    backend::SerialBackend serial(serial_config);

    backend::Config mpi_config;
    backend::MpiBackend distributed(mpi_config, backend::TopologyReport{});

    for (const auto& name : all_solver_names()) {
        auto solver = make_solver(name);
        if (!solver->applicable_to(problem)) continue;
        const Vector expected = solver->solve(problem, serial, fixed_options(8)).solution;
        const Vector actual = solver->solve(problem, distributed, fixed_options(8)).solution;
        const Real difference = worst_difference(expected, actual);
        PNL_REQUIRE_MESSAGE(difference <= 1.0e-12,
                            "solver " + name + " on " + std::to_string(world_size()) +
                                " ranks differs from serial by " + test::format(difference));
    }
}

PNL_TEST("mpi/the deterministic reduction is reproducible within a rank count") {
    const Index n = 200000;
    Vector data(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) {
        data[static_cast<std::size_t>(i)] =
            std::sin(static_cast<Real>(i)) * std::pow(10.0, (i % 21) - 10);
    }

    backend::Config config;
    backend::MpiBackend distributed(config, backend::TopologyReport{});
    const Range mine = distributed.local_rows(n);

    auto run = [&] {
        return distributed.reduce(mine.size(), 0.0, [&](Range chunk) {
            Real partial = 0.0;
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                partial += data[static_cast<std::size_t>(mine.begin + k)];
            }
            return partial;
        });
    };

    const Real first = run();
    for (int repeat = 0; repeat < 5; ++repeat) {
        PNL_REQUIRE_MESSAGE(run() == first,
                            "the deterministic reduction is not reproducible at " +
                                std::to_string(world_size()) + " ranks");
    }

    // Every rank must also agree with every other, which the ordered allgather
    // guarantees and MPI_Allreduce would not.
    Real from_root = first;
    MPI_Bcast(&from_root, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    PNL_REQUIRE_MESSAGE(from_root == first,
                        "rank " + std::to_string(world_rank()) +
                            " disagrees with rank 0 on the reduction value");
}

PNL_TEST("mpi/solvers converge to the same solution whatever the rank count") {
    // The end to end statement: run to tolerance, and check the answer against
    // the manufactured solution rather than against another run.
    problems::Poisson2D problem(63);
    backend::Config config;
    backend::MpiBackend distributed(config, backend::TopologyReport{});

    auto solver = make_solver("cg");
    SolverOptions options;
    options.tolerance = 1.0e-11;
    options.max_iterations = 50000;
    const auto result = solver->solve(problem, distributed, options);
    PNL_REQUIRE_MESSAGE(result.converged(),
                        "conjugate gradient did not converge at " +
                            std::to_string(world_size()) + " ranks");

    const ConstVectorView exact = problem.exact_solution();
    Real worst = 0.0;
    for (Index i = 1; i <= problem.side(); ++i) {
        for (Index j = 1; j <= problem.side(); ++j) {
            const auto k = static_cast<std::size_t>(problem.at(i, j));
            worst = std::max(worst, std::abs(result.solution[k] - exact[k]));
        }
    }
    const Real h = 1.0 / static_cast<Real>(problem.side() + 1);
    const Real predicted = std::numbers::pi_v<Real> * std::numbers::pi_v<Real> / 12.0 * h * h;
    PNL_REQUIRE_MESSAGE(test::close_relative(worst, predicted, 0.05),
                        "at " + std::to_string(world_size()) +
                            " ranks the discretisation error is " + test::format(worst) +
                            " against the predicted " + test::format(predicted));
}

PNL_TEST("mpi/communication time is measured and non zero when there is communication") {
    problems::Poisson2D problem(127, problems::PoissonRhs::SpectrallyRich);
    backend::Config config;
    backend::MpiBackend distributed(config, backend::TopologyReport{});
    distributed.reset_timing();

    auto solver = make_solver("cg");
    (void)solver->solve(problem, distributed, fixed_options(50));

    const auto& timing = distributed.timing();
    PNL_REQUIRE(timing.reductions > 0);
    if (world_size() > 1) {
        PNL_REQUIRE_MESSAGE(timing.halo_exchanges > 0,
                            "no halo exchange was recorded on a multi rank run");
        PNL_REQUIRE_MESSAGE(timing.total_seconds() > 0.0,
                            "communication time measured as zero on a multi rank run");
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    const int rank = world_rank();
    // Only rank 0 writes, per Section 9. Every rank runs every case, and the
    // exit status is combined so a failure on any rank fails the test.
    if (rank != 0) {
        // Losing this redirect is harmless, it would only mean a noisier log,
        // so the failure is acknowledged rather than treated as fatal.
        if (std::freopen("/dev/null", "w", stdout) == nullptr) {
            std::fprintf(stderr, "rank %d could not silence stdout\n", rank);
        }
    } else {
        std::printf("running at %d rank(s)\n", world_size());
    }

    const int local_status = pnl::test::run_all();
    int global_status = 0;
    MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Finalize();
    return global_status;
}
