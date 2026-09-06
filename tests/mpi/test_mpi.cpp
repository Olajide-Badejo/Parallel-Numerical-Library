// SPDX-License-Identifier: MIT
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

#include <pnl/backend/hybrid.hpp>
#include <pnl/backend/mpi.hpp>
#include <pnl/backend/serial.hpp>
#include <pnl/problems/dense_generator.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <pnl_test.hpp>
#include <string>

#include <mpi.h>

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

/// Solvers whose sweep is order independent, so a distributed run reproduces
/// the serial one exactly except for reduction regrouping.
const char* const ORDER_FREE_SOLVERS[] = {
    "jacobi", "gauss_seidel_rb", "sor_rb", "block_jacobi", "richardson"};

/// Solvers with a sequential recurrence, reproduced exactly by the token chain.
const char* const ORDERED_SOLVERS[] = {
    "gauss_seidel_f", "gauss_seidel_b", "gauss_seidel_s", "sor", "block_gauss_seidel"};

/// Fail on one rank on purpose, when `PNL_TEST_FAIL_RANK` names it.
///
/// This is the gate for the failure mode itself rather than for any numerics.
/// It is called from inside a case that goes on to make collectives, which is
/// the shape that matters: the assertion throws, this rank leaves the case, and
/// the other ranks are left in an exchange that has lost a participant. Before
/// phase B6 that was a hang to the 900 second CTest timeout. Unset, which is
/// every ordinary run, this does nothing.
void fail_if_injected() {
    const char* const requested = std::getenv("PNL_TEST_FAIL_RANK");
    if (requested == nullptr) return;
    if (std::atoi(requested) != world_rank()) return;
    throw test::Failure{"injected failure on rank " + std::to_string(world_rank()) +
                        ", requested by PNL_TEST_FAIL_RANK"};
}

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
                                std::to_string(rows) + " rows covered " + std::to_string(covered));
    }
}

PNL_TEST("mpi/order free solvers agree with serial on the Poisson problem") {
    // The injection point of the failure mode gate. It is here rather than in
    // the case above because this one goes on to exchange halos, so a rank that
    // leaves early takes a participant out of a collective the others are
    // already in.
    fail_if_injected();

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
            const Real difference = test::worst_difference(expected, actual);
            PNL_REQUIRE_MESSAGE(difference <= 1.0e-12,
                                std::string("solver ") + name + " at n = " + std::to_string(n) +
                                    " on " + std::to_string(world_size()) +
                                    " ranks differs from serial by " + test::format(difference));
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
            const Real difference = test::worst_difference(expected, actual);
            PNL_REQUIRE_MESSAGE(difference == 0.0,
                                std::string("solver ") + name + " at n = " + std::to_string(n) +
                                    " on " + std::to_string(world_size()) +
                                    " ranks is not bit identical to serial, worst difference " +
                                    test::format(difference) +
                                    "; the pipelined ordering is meant to be exact");
        }
    }
}

PNL_TEST("mpi/dense systems agree with serial") {
    problems::DenseProblem problem(
        150, 20260802, problems::DenseKind::SymmetricPositiveDefinite, 6);

    backend::Config serial_config;
    backend::SerialBackend serial(serial_config);

    backend::Config mpi_config;
    backend::MpiBackend distributed(mpi_config, backend::TopologyReport{});

    for (const auto& name : all_solver_names()) {
        auto solver = make_solver(name);
        if (!solver->applicable_to(problem)) continue;
        const Vector expected = solver->solve(problem, serial, fixed_options(8)).solution;
        const Vector actual = solver->solve(problem, distributed, fixed_options(8)).solution;
        const Real difference = test::worst_difference(expected, actual);
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
    PNL_REQUIRE_MESSAGE(
        from_root == first,
        "rank " + std::to_string(world_rank()) + " disagrees with rank 0 on the reduction value");
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
    PNL_REQUIRE_MESSAGE(
        result.converged(),
        "conjugate gradient did not converge at " + std::to_string(world_size()) + " ranks");

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

PNL_TEST("mpi/default_args reach the distributed override through either static type") {
    // The distributed half of what tests/unit/test_default_args.cpp asserts of
    // the interface. This is the pair Section 4.7 names: the defaults were on
    // Backend::run_ordered and MpiBackend's override omitted them, so the two
    // argument call below compiled through a Backend reference and did not
    // compile at all through an MpiBackend reference to the same object. That
    // it compiles here is most of the assertion; that both spellings run the
    // work once is the rest.
    backend::Config config;
    backend::MpiBackend distributed(config, backend::TopologyReport{});
    backend::Backend& base = distributed;

    int ran = 0;
    distributed.run_ordered([&] { ++ran; }, true);
    PNL_REQUIRE_MESSAGE(ran == 1, "the ordered work did not run through the derived type");
    base.run_ordered([&] { ++ran; }, true);
    PNL_REQUIRE_MESSAGE(ran == 2, "the ordered work did not run through the base type");

    // Backward as well, since that is the other half of the token chain and
    // takes the same defaults.
    distributed.run_ordered([&] { ++ran; }, false);
    base.run_ordered([&] { ++ran; }, false);
    PNL_REQUIRE_MESSAGE(ran == 4, "a backward ordered sweep did not run on every rank");
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

#if defined(PNL_WITH_OPENMP)

namespace {

/// Threads inside each rank for the hybrid cases. Two, so that at four ranks
/// the job is eight workers and the product is not equal to either factor,
/// which is what makes the worker count assertion below say something.
constexpr int HYBRID_THREADS = 2;

[[nodiscard]] backend::Config hybrid_config() {
    backend::Config config;
    config.threads_per_rank = HYBRID_THREADS;
    return config;
}

}  // namespace

PNL_TEST("mpi/hybrid reports ranks times threads as its worker count") {
    // A6 and MEAS-09. worker_count() used to be inherited from MpiBackend,
    // which returns the rank count, while the constructor set config_.workers
    // to the product and a comment said the product was the number the curve
    // wanted. The result row takes its workers column from this function, so
    // every hybrid row recorded its rank count and the scaling curve was
    // plotted against a quantity a factor of threads_per_rank too small.
    backend::HybridBackend hybrid(hybrid_config(), backend::TopologyReport{});

    const int expected = world_size() * HYBRID_THREADS;
    PNL_REQUIRE_MESSAGE(hybrid.worker_count() == expected,
                        "the hybrid backend at " + std::to_string(world_size()) + " ranks and " +
                            std::to_string(HYBRID_THREADS) + " threads reports " +
                            std::to_string(hybrid.worker_count()) + " workers rather than " +
                            std::to_string(expected));
    PNL_REQUIRE(hybrid.threads_per_rank() == HYBRID_THREADS);
    PNL_REQUIRE_MESSAGE(hybrid.rank_count() == world_size(),
                        "the hybrid backend disagrees with MPI about the rank count");

    // The config the result row also reads has to agree with the function, or
    // the row and the curve disagree with each other.
    PNL_REQUIRE_MESSAGE(
        hybrid.config().workers == expected,
        "the hybrid backend's config reports " + std::to_string(hybrid.config().workers) +
            " workers where worker_count() reports " + std::to_string(hybrid.worker_count()));

    // Through the factory as well, which is the path the driver takes.
    backend::Config config = hybrid_config();
    auto built = backend::make_backend("hybrid", config);
    PNL_REQUIRE(built->name() == "hybrid");
    PNL_REQUIRE_MESSAGE(built->worker_count() == expected,
                        "the hybrid backend built through make_backend reports " +
                            std::to_string(built->worker_count()) + " workers rather than " +
                            std::to_string(expected));
}

PNL_TEST("mpi/hybrid solves agree with serial to reduction tolerance") {
    // The whole of the hybrid backend below the interface is inherited from
    // MpiBackend: the halo exchange, the ordered pass and the deterministic
    // reduction are literally the same code. What is new is that a rank spreads
    // its band across an OpenMP team instead of walking it on one thread, and
    // this is the assertion that the spreading changes no answer.
    //
    // Reduction tolerance rather than bit identity, and for the reason the file
    // header gives: rank boundaries are chosen for load balance and do not align
    // with the fixed chunk grid, so a reduction over four ranks groups its
    // partials differently from one over two. That is a property of the
    // decomposition and not of the threading.
    for (Index n : {63, 65, 127}) {
        problems::Poisson2D problem(n, problems::PoissonRhs::SpectrallyRich);

        backend::Config serial_config;
        backend::SerialBackend serial(serial_config);
        backend::HybridBackend hybrid(hybrid_config(), backend::TopologyReport{});

        for (const char* name : ORDER_FREE_SOLVERS) {
            auto solver = make_solver(name);
            if (!solver->applicable_to(problem)) continue;
            const Vector expected = solver->solve(problem, serial, fixed_options()).solution;
            const Vector actual = solver->solve(problem, hybrid, fixed_options()).solution;
            const Real difference = test::worst_difference(expected, actual);
            PNL_REQUIRE_MESSAGE(difference <= 1.0e-12,
                                std::string("solver ") + name + " at n = " + std::to_string(n) +
                                    " on hybrid at " + std::to_string(world_size()) +
                                    " ranks times " + std::to_string(HYBRID_THREADS) +
                                    " threads differs from serial by " + test::format(difference));
        }
    }
}

PNL_TEST("mpi/hybrid reproduces the sequential recurrence exactly") {
    // The ordered sweeps are the strong claim, and they are stronger on hybrid
    // than the tolerance above: the pipelined token chain preserves natural
    // ordering across ranks, and run_ordered runs the rank's whole band on one
    // thread, so an OpenMP team inside the rank must not disturb it. If a future
    // change threaded the ordered pass, this is what would notice.
    for (Index n : {31, 63, 65}) {
        problems::Poisson2D problem(n, problems::PoissonRhs::SpectrallyRich);

        backend::Config serial_config;
        backend::SerialBackend serial(serial_config);
        backend::HybridBackend hybrid(hybrid_config(), backend::TopologyReport{});

        for (const char* name : ORDERED_SOLVERS) {
            auto solver = make_solver(name);
            if (!solver->applicable_to(problem)) continue;
            const Vector expected = solver->solve(problem, serial, fixed_options(10)).solution;
            const Vector actual = solver->solve(problem, hybrid, fixed_options(10)).solution;
            const Real difference = test::worst_difference(expected, actual);
            PNL_REQUIRE_MESSAGE(difference == 0.0,
                                std::string("solver ") + name + " at n = " + std::to_string(n) +
                                    " on hybrid at " + std::to_string(world_size()) +
                                    " ranks is not bit identical to serial, worst difference " +
                                    test::format(difference));
        }
    }
}

PNL_TEST("mpi/hybrid reduces identically to the pure distributed backend") {
    // The control that makes the two comparable. Both backends use the same
    // rank decomposition and the same fixed chunk grid inside a rank, so at the
    // same rank count they must agree bit for bit whatever the thread count is.
    // Any difference the sweep measures between mpi and hybrid is then the
    // threading, which is the comparison the study wants.
    const Index n = 200000;
    Vector data(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) {
        data[static_cast<std::size_t>(i)] =
            std::sin(static_cast<Real>(i)) * std::pow(10.0, (i % 21) - 10);
    }

    backend::Config plain_config;
    backend::MpiBackend distributed(plain_config, backend::TopologyReport{});
    backend::HybridBackend hybrid(hybrid_config(), backend::TopologyReport{});

    auto reduce_on = [&](backend::Backend& execution) {
        const Range mine = execution.local_rows(n);
        return execution.reduce(mine.size(), 0.0, [&](Range chunk) {
            Real partial = 0.0;
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                partial += data[static_cast<std::size_t>(mine.begin + k)];
            }
            return partial;
        });
    };

    const Real plain = reduce_on(distributed);
    const Real threaded = reduce_on(hybrid);
    PNL_REQUIRE_MESSAGE(plain == threaded,
                        "at " + std::to_string(world_size()) + " ranks the hybrid reduction gave " +
                            test::format(threaded) + " and the pure distributed one gave " +
                            test::format(plain) +
                            "; the two share a chunk grid and must agree bit for bit");

    // And it is reproducible within the hybrid backend itself, which is what
    // says the OpenMP team is not contributing an arrival order.
    for (int repeat = 0; repeat < 5; ++repeat) {
        PNL_REQUIRE_MESSAGE(reduce_on(hybrid) == threaded,
                            "the hybrid deterministic reduction is not reproducible at " +
                                std::to_string(world_size()) + " ranks");
    }
}

PNL_TEST("mpi/a body that throws inside a rank's team surfaces on the dispatching thread") {
    // B6 wrapped every worker body in a relay because an exception may not leave
    // an OpenMP structured block: before that, a throwing body terminated the
    // rank. The hybrid backend has a relay of its own, since its team is nested
    // inside the rank rather than around it, and this is the case that exercises
    // it. It is the hybrid half of what tests/unit/test_throwing_body.cpp
    // asserts of the shared memory pools.
    backend::HybridBackend hybrid(hybrid_config(), backend::TopologyReport{});

    bool caught = false;
    try {
        hybrid.parallel_for(1000, [](Range chunk) {
            if (chunk.begin == 0) throw NumericalFailure("deliberate, from the first chunk");
        });
    } catch (const NumericalFailure& failure) {
        caught = true;
        PNL_REQUIRE_MESSAGE(
            std::string(failure.what()).find("deliberate") != std::string::npos,
            std::string("the message did not survive the thread boundary: ") + failure.what());
    }
    PNL_REQUIRE_MESSAGE(caught,
                        "a body that threw inside the hybrid backend's OpenMP team did not "
                        "surface on the dispatching thread");

    // The backend is usable afterwards, which is the other half of the relay's
    // contract: it clears itself on rethrow.
    int chunks = 0;
    hybrid.parallel_for(1000, [&](Range) { ++chunks; });
    PNL_REQUIRE_MESSAGE(chunks > 0, "the hybrid backend did not dispatch after an exception");
}

#endif  // PNL_WITH_OPENMP

namespace {

/// How long a rank that has failed waits for the others to reach the same point
/// before it gives up on them and aborts the job.
///
/// Any value far below the 900 second CTest timeout and far above the skew
/// between ranks that have just finished the same case would do; the ranks
/// arrive within milliseconds of each other when they arrive at all.
constexpr double FAILURE_DEADLINE_SECONDS = 5.0;

/// Combine the pass or fail flag across ranks after every case.
///
/// Two mechanisms, because there are two ways one rank's failure hurts.
///
/// A failure every rank sees, a numeric comparison for instance, is combined
/// here and stops the run at the same case everywhere, so no rank enters the
/// next case with a different view of the world than its neighbours.
///
/// A failure only one rank sees is worse. The assertion throws out of the
/// middle of a case, past the collectives that case still owed, and the ranks
/// that passed are already parked in an exchange whose participant has left.
/// A rank that failed therefore cannot simply block here: it gives the others a
/// bounded moment to arrive and aborts the job when they do not. That is the
/// difference between a job that fails in seconds and the hang to the CTest
/// timeout that Section 4.7 records against this file.
///
/// The reduction is the nonblocking form on every path, failing or not, because
/// a blocking collective and a nonblocking one do not match each other.
int combine_case_status(const std::string& name, int status) {
    int combined = 0;
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Iallreduce(&status, &combined, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD, &request);

    if (status == 0) {
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        return combined;
    }

    int done = 0;
    const double deadline = MPI_Wtime() + FAILURE_DEADLINE_SECONDS;
    while (done == 0 && MPI_Wtime() < deadline) {
        MPI_Test(&request, &done, MPI_STATUS_IGNORE);
    }
    if (done == 0) {
        std::fprintf(stderr,
                     "rank %d failed '%s' and the other ranks did not reach the end of that "
                     "case within %.0f seconds, so they are waiting in a collective this rank "
                     "has left. Aborting the job.\n",
                     world_rank(),
                     name.c_str(),
                     FAILURE_DEADLINE_SECONDS);
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return combined > status ? combined : status;
}

}  // namespace

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

    // The status is combined after every case rather than only at the end. At
    // the end alone it still fails the job when every rank gets there, and the
    // whole difficulty is that a rank failure is exactly the thing that stops
    // them getting there.
    const int local_status = pnl::test::run_all({}, combine_case_status);
    int global_status = 0;
    MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Finalize();
    return global_status;
}
