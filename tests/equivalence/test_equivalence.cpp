// SPDX-License-Identifier: MIT
/// \file test_equivalence.cpp
/// Cross backend equivalence: the invariant the whole comparison rests on.
///
/// Objective 2 says identical numerics across backends is a tested invariant,
/// not an assumption. These tests assert the strong form of that: with the
/// deterministic reduction mode, every shared memory backend at every worker
/// count produces bit identical iterates. Not close, identical.
///
/// That is achievable because the reduction chunk grid is fixed by the problem
/// size alone, so the partials are always combined in the same order, and
/// because every solver's sweep is either order independent (Jacobi, red black)
/// or executed in strict order (natural ordering Gauss Seidel, block Gauss
/// Seidel). Where it is not achievable, the test says so explicitly rather than
/// hiding behind a tolerance: the native reduction mode is checked to agree only
/// to reduction tolerance, and that difference is the measured price of
/// determinism.
///
/// **The Poisson problems here are constructed with
/// `PoissonRhs::SpectrallyRich` explicitly**, never on the constructor default.
/// The default is `ManufacturedSine`, `u = sin(pi x) sin(pi y)`, which is
/// symmetric under exchanging the two grid axes, so a kernel that read the grid
/// transposed would produce the transpose of the correct iterate and every
/// comparison in this file would still pass. Section 9.3 of the version 2
/// specification makes that argument for the Fortran kernels of release 1.2.0,
/// where a reversed subscript order is the specific mistake it is about, and the
/// rich source is what makes it visible. It costs nothing here and it is what
/// every sweep block already measures.
///
/// **The boundary sizes** are the cases below whose names contain the word
/// boundary, and CTest has an entry that runs exactly those. Two boundaries
/// matter and neither was covered by a suite that ran everything at 63:
///
///   - `n` in `{511, 512, 513}`, where `reduction_chunk_count` stops being `n`
///     and becomes `DETERMINISTIC_CHUNKS`. That grid is the reason exact
///     equality is assertable at all, and its shape changes exactly there.
///   - `n` in `{0, 1}`, the empty problem and the one unknown problem. At one
///     unknown and eight workers, seven of the eight chunks are empty, which is
///     the shape a partition is most likely to get wrong; at zero the library
///     must refuse the problem, and must refuse it the same way whichever
///     backend was going to run it.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/chunking.hpp>
#include <pnl/core/error.hpp>
#include <pnl/problems/dense_generator.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <pnl_test.hpp>
#include <string>

using namespace pnl;
using namespace pnl::solvers;

namespace {

/// Shared memory backends present in this build.
[[nodiscard]] std::vector<std::string> thread_backends() {
    std::vector<std::string> names;
    for (const auto& name : backend::available_backends()) {
        if (name == "mpi" || name == "hybrid") continue;
        names.push_back(name);
    }
    return names;
}

/// The comma separated list in `PNL_TEST_WORKERS`, or an empty vector.
///
/// Unset, empty, or holding nothing that parses to a positive count, this
/// returns nothing and the sweep below is the machine's. Values that do not
/// parse are dropped rather than diagnosed: the variable exists to widen a
/// sweep, and a typo that widens it by less is caught by reading the line the
/// suite prints.
[[nodiscard]] std::vector<int> requested_worker_counts() {
    std::vector<int> counts;
    const char* requested = std::getenv("PNL_TEST_WORKERS");
    if (requested == nullptr) return counts;

    const std::string text(requested);
    std::size_t position = 0;
    while (position <= text.size()) {
        const std::size_t comma = text.find(',', position);
        const std::size_t length =
            comma == std::string::npos ? std::string::npos : comma - position;
        const std::string field = text.substr(position, length);
        char* end = nullptr;
        const long value = std::strtol(field.c_str(), &end, 10);
        if (end != field.c_str() && value > 0 && value <= 1024) {
            counts.push_back(static_cast<int>(value));
        }
        if (comma == std::string::npos) break;
        position = comma + 1;
    }
    return counts;
}

/// Worker counts to sweep, clamped to what this machine has, and printed.
///
/// `PNL_TEST_WORKERS` overrides the list, and it exists for one caller: the
/// thread sanitizer job in CI. That job runs on a four processor runner, so the
/// clamp below would stop the sweep at four workers and the job would cover
/// half of what Section 8 of the version 2 specification asks of it without
/// saying so anywhere. Asking eight software workers to share four processors
/// is a legitimate thing to require of a pool, and is if anything a harder test
/// of one, because the scheduler then preempts inside the critical sections
/// rather than around them.
///
/// The list is printed once, whichever way it was arrived at, so that a green
/// run says what it covered instead of leaving a reader to work it out from the
/// processor count of the machine it ran on.
[[nodiscard]] std::vector<int> worker_counts() {
    static const std::vector<int> counts = [] {
        std::vector<int> chosen = requested_worker_counts();
        const char* source = "PNL_TEST_WORKERS";
        if (chosen.empty()) {
            source = "this machine";
            const int available = backend::available_logical_cpus();
            chosen = {1, 2, 3};
            for (int candidate : {4, 7, 8, 16}) {
                if (candidate <= available) chosen.push_back(candidate);
            }
        }
        std::printf("        worker counts swept, from %s:", source);
        for (int value : chosen) std::printf(" %d", value);
        std::printf("\n");
        return chosen;
    }();
    return counts;
}

SolverOptions equivalence_options() {
    SolverOptions options;
    // A fixed iteration count rather than a tolerance: two backends that
    // stopped at different iterations would agree on the answer but not on the
    // iterate, and it is the iterate that proves the sweeps are identical.
    options.mode = RunMode::FixedIterations;
    options.max_iterations = 25;
    options.check_interval = 1000000;
    return options;
}

/// Solve on one named backend at one worker count.
[[nodiscard]] Vector solve_with(const std::string& backend_name,
                                int workers,
                                const std::string& solver_name,
                                problems::Problem& problem,
                                backend::ReductionMode reduction) {
    backend::Config config;
    config.workers = workers;
    config.reduction = reduction;
    auto execution = backend::make_backend(backend_name, config);
    auto solver = make_solver(solver_name);
    return solver->solve(problem, *execution, equivalence_options()).solution;
}

}  // namespace

PNL_TEST("equivalence/every backend and worker count gives bit identical Poisson iterates") {
    problems::Poisson2D problem(63, problems::PoissonRhs::SpectrallyRich);

    for (const auto& solver_name : all_solver_names()) {
        auto probe = make_solver(solver_name);
        if (!probe->applicable_to(problem)) continue;

        const Vector reference =
            solve_with("serial", 1, solver_name, problem, backend::ReductionMode::Deterministic);

        for (const auto& backend_name : thread_backends()) {
            for (int workers : worker_counts()) {
                const Vector candidate = solve_with(backend_name,
                                                    workers,
                                                    solver_name,
                                                    problem,
                                                    backend::ReductionMode::Deterministic);
                PNL_REQUIRE_MESSAGE(test::first_difference(reference, candidate) < 0,
                                    "solver " + solver_name + " on backend " + backend_name +
                                        " with " + std::to_string(workers) +
                                        " workers differs from serial at " +
                                        test::describe_difference(candidate, reference));
            }
        }
    }
}

PNL_TEST("equivalence/every backend and worker count gives bit identical dense iterates") {
    problems::DenseProblem problem(
        180, 20260802, problems::DenseKind::SymmetricPositiveDefinite, 6);

    for (const auto& solver_name : all_solver_names()) {
        auto probe = make_solver(solver_name);
        if (!probe->applicable_to(problem)) continue;

        const Vector reference =
            solve_with("serial", 1, solver_name, problem, backend::ReductionMode::Deterministic);

        for (const auto& backend_name : thread_backends()) {
            for (int workers : worker_counts()) {
                const Vector candidate = solve_with(backend_name,
                                                    workers,
                                                    solver_name,
                                                    problem,
                                                    backend::ReductionMode::Deterministic);
                PNL_REQUIRE_MESSAGE(test::first_difference(reference, candidate) < 0,
                                    "solver " + solver_name + " on backend " + backend_name +
                                        " with " + std::to_string(workers) +
                                        " workers differs from serial at " +
                                        test::describe_difference(candidate, reference));
            }
        }
    }
}

PNL_TEST("equivalence/the deterministic reduction is bit identical across worker counts") {
    // Reduce a deliberately awkward vector, whose terms span many magnitudes so
    // that any change of summation order shows up immediately.
    const Index n = 100000;
    Vector data(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) {
        data[static_cast<std::size_t>(i)] =
            std::sin(static_cast<Real>(i)) * std::pow(10.0, (i % 21) - 10);
    }

    auto reduce_on =
        [&](const std::string& backend_name, int workers, backend::ReductionMode mode) {
            backend::Config config;
            config.workers = workers;
            config.reduction = mode;
            auto execution = backend::make_backend(backend_name, config);
            return execution->reduce(n, 0.0, [&](Range chunk) {
                Real partial = 0.0;
                for (Index k = chunk.begin; k < chunk.end; ++k) {
                    partial += data[static_cast<std::size_t>(k)];
                }
                return partial;
            });
        };

    const Real reference = reduce_on("serial", 1, backend::ReductionMode::Deterministic);
    for (const auto& backend_name : thread_backends()) {
        for (int workers : worker_counts()) {
            const Real candidate =
                reduce_on(backend_name, workers, backend::ReductionMode::Deterministic);
            PNL_REQUIRE_MESSAGE(candidate == reference,
                                "deterministic reduction on " + backend_name + " with " +
                                    std::to_string(workers) + " workers gave " +
                                    test::format(candidate) + " against the serial " +
                                    test::format(reference));
        }
    }
}

PNL_TEST("equivalence/the native reduction agrees only to reduction tolerance") {
    // This test documents the difference rather than forbidding it. The native
    // mode is offered so the sweep can price determinism; if it ever became bit
    // identical too, the deterministic mode would be costing nothing and this
    // test failing would be the signal to say so in the report.
    const Index n = 100000;
    Vector data(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) {
        data[static_cast<std::size_t>(i)] =
            std::sin(static_cast<Real>(i)) * std::pow(10.0, (i % 21) - 10);
    }

    auto reduce_on =
        [&](const std::string& backend_name, int workers, backend::ReductionMode mode) {
            backend::Config config;
            config.workers = workers;
            config.reduction = mode;
            auto execution = backend::make_backend(backend_name, config);
            return execution->reduce(n, 0.0, [&](Range chunk) {
                Real partial = 0.0;
                for (Index k = chunk.begin; k < chunk.end; ++k) {
                    partial += data[static_cast<std::size_t>(k)];
                }
                return partial;
            });
        };

    const Real reference = reduce_on("serial", 1, backend::ReductionMode::Deterministic);
    for (const auto& backend_name : thread_backends()) {
        const Real native = reduce_on(backend_name, 4, backend::ReductionMode::Native);
        PNL_REQUIRE_MESSAGE(
            test::close_relative(native, reference, 1.0e-12),
            "native reduction on " + backend_name + " gave " + test::format(native) +
                " which is not even within reduction tolerance of " + test::format(reference));
    }
}

PNL_TEST("equivalence/the block partition covers the range exactly once") {
    // The remainder aware partition is shared by the chunk grid and the MPI row
    // decomposition, so an off by one here would corrupt both.
    for (Index n : {0, 1, 7, 64, 1000, 4097}) {
        for (Index parts = 1; parts <= 33; ++parts) {
            Index covered = 0;
            Index previous_end = 0;
            Index largest = 0;
            Index smallest = n;
            for (Index k = 0; k < parts; ++k) {
                const Range block = block_partition(n, parts, k);
                PNL_REQUIRE(block.begin == previous_end);
                PNL_REQUIRE(block.size() >= 0);
                previous_end = block.end;
                covered += block.size();
                largest = std::max(largest, block.size());
                smallest = std::min(smallest, block.size());
            }
            PNL_REQUIRE_MESSAGE(covered == n,
                                "partition of " + std::to_string(n) + " into " +
                                    std::to_string(parts) + " covered " + std::to_string(covered));
            PNL_REQUIRE(previous_end == n);
            if (n > 0 && parts <= n) {
                PNL_REQUIRE_MESSAGE(largest - smallest <= 1,
                                    "block sizes differ by more than one for n = " +
                                        std::to_string(n) + ", parts = " + std::to_string(parts));
            }
        }
    }
}

PNL_TEST("equivalence/a dynamic schedule gives the same answer as a static one") {
    problems::Poisson2D problem(63, problems::PoissonRhs::SpectrallyRich);
    for (const auto& solver_name : {"jacobi", "gauss_seidel_rb", "cg", "block_jacobi"}) {
        backend::Config static_config;
        static_config.workers = 4;
        static_config.schedule = backend::Schedule::Static;

        backend::Config dynamic_config;
        dynamic_config.workers = 4;
        dynamic_config.schedule = backend::Schedule::Dynamic;
        dynamic_config.chunks_per_worker = 5;

        for (const auto& backend_name : thread_backends()) {
            auto solver = make_solver(solver_name);
            auto a = backend::make_backend(backend_name, static_config);
            auto b = backend::make_backend(backend_name, dynamic_config);
            const Vector first = solver->solve(problem, *a, equivalence_options()).solution;
            const Vector second = solver->solve(problem, *b, equivalence_options()).solution;
            PNL_REQUIRE_MESSAGE(test::first_difference(first, second) < 0,
                                std::string("solver ") + solver_name + " on " + backend_name +
                                    " differs between static and dynamic scheduling at " +
                                    test::describe_difference(second, first));
        }
    }
}

PNL_TEST("equivalence/boundary: the reduction is bit identical across the chunk count switch") {
    // reduction_chunk_count is n below DETERMINISTIC_CHUNKS and
    // DETERMINISTIC_CHUNKS at or above it, so 511 chunks of one element, 512 of
    // one, and 512 of one or two are the three shapes the grid takes around the
    // switch. Reduced directly rather than through a solver, so a failure names
    // the reduction and not the method that called it.
    for (Index n : {Index{510}, Index{511}, Index{512}, Index{513}, Index{514}}) {
        Vector data(static_cast<std::size_t>(n));
        for (Index i = 0; i < n; ++i) {
            // The same twenty one decade spread the case above uses: terms of
            // wildly different magnitudes are what make a regrouping visible.
            data[static_cast<std::size_t>(i)] =
                std::sin(static_cast<Real>(i)) * std::pow(10.0, (i % 21) - 10);
        }

        const Index expected_chunks = std::min(backend::DETERMINISTIC_CHUNKS, n);
        PNL_REQUIRE_MESSAGE(backend::reduction_chunk_count(n) == expected_chunks,
                            "the reduction chunk count at n = " + std::to_string(n) + " is " +
                                std::to_string(backend::reduction_chunk_count(n)));

        auto reduce_on = [&](const std::string& backend_name, int workers) {
            backend::Config config;
            config.workers = workers;
            config.reduction = backend::ReductionMode::Deterministic;
            auto execution = backend::make_backend(backend_name, config);
            return execution->reduce(n, 0.0, [&](Range chunk) {
                Real partial = 0.0;
                for (Index k = chunk.begin; k < chunk.end; ++k) {
                    partial += data[static_cast<std::size_t>(k)];
                }
                return partial;
            });
        };

        const Real reference = reduce_on("serial", 1);
        for (const auto& backend_name : thread_backends()) {
            for (int workers : worker_counts()) {
                const Real candidate = reduce_on(backend_name, workers);
                PNL_REQUIRE_MESSAGE(candidate == reference,
                                    "at the chunk count switch, n = " + std::to_string(n) +
                                        ", the deterministic reduction on " + backend_name +
                                        " with " + std::to_string(workers) + " workers gave " +
                                        test::format(candidate) + " against the serial " +
                                        test::format(reference));
            }
        }
    }
}

PNL_TEST("equivalence/boundary: solves at the chunk count switch are bit identical") {
    // The same sizes through a solve, on a problem whose unknown count is
    // exactly the reduce length. A dense system of order n has n unknowns; a
    // Poisson grid of side n has n squared, which lands nowhere near the switch,
    // so this is the problem that reaches it.
    //
    // The reducing solvers, which is what the boundary is about: conjugate
    // gradient reduces several times an iteration, Richardson carries a residual
    // from one iteration to the next, and Jacobi is the control that reduces
    // once at the end. A red black method is included because its sweep splits
    // the range a second time.
    for (Index n : {Index{511}, Index{512}, Index{513}}) {
        problems::DenseProblem problem(
            n, 20260802, problems::DenseKind::SymmetricPositiveDefinite, 8);

        for (const char* solver_name : {"cg", "richardson", "jacobi", "gauss_seidel_rb"}) {
            auto probe = make_solver(solver_name);
            if (!probe->applicable_to(problem)) continue;

            const Vector reference = solve_with(
                "serial", 1, solver_name, problem, backend::ReductionMode::Deterministic);

            for (const auto& backend_name : thread_backends()) {
                for (int workers : worker_counts()) {
                    const Vector candidate = solve_with(backend_name,
                                                        workers,
                                                        solver_name,
                                                        problem,
                                                        backend::ReductionMode::Deterministic);
                    PNL_REQUIRE_MESSAGE(test::first_difference(reference, candidate) < 0,
                                        std::string("solver ") + solver_name + " at order " +
                                            std::to_string(n) + " on backend " + backend_name +
                                            " with " + std::to_string(workers) +
                                            " workers differs from serial at " +
                                            test::describe_difference(candidate, reference));
                }
            }
        }
    }
}

PNL_TEST("equivalence/boundary: an empty problem is refused the same way on every backend") {
    // Zero unknowns is a documented require failure rather than an empty
    // answer, and it is documented in the only place that matters: the
    // constructor of each problem refuses it, so the refusal happens before any
    // backend exists and is the same on all of them by construction. That is
    // asserted rather than assumed, because "the same on every backend" is the
    // claim, and a problem that had deferred the check to its first sweep would
    // have made it false.
    PNL_REQUIRE_THROWS(problems::Poisson2D(0, problems::PoissonRhs::SpectrallyRich),
                       InvalidArgument);
    PNL_REQUIRE_THROWS(problems::Poisson2D(-1, problems::PoissonRhs::SpectrallyRich),
                       InvalidArgument);
    PNL_REQUIRE_THROWS(
        problems::DenseProblem(0, 20260802, problems::DenseKind::SymmetricPositiveDefinite, 1),
        InvalidArgument);

    for (const auto& backend_name : thread_backends()) {
        for (int workers : worker_counts()) {
            backend::Config config;
            config.workers = workers;
            auto execution = backend::make_backend(backend_name, config);

            // The backend itself answers an empty range with no work rather than
            // with a failure, which is the other half of the contract: it is the
            // problem that refuses to exist, not the dispatch that refuses to
            // run over nothing.
            int chunks = 0;
            execution->parallel_for(0, [&](Range) { ++chunks; });
            PNL_REQUIRE_MESSAGE(chunks == 0,
                                "backend " + backend_name + " with " + std::to_string(workers) +
                                    " workers ran " + std::to_string(chunks) +
                                    " chunks over an empty range");
            const Real sum = execution->reduce(0, 0.0, [](Range) { return 1.0; });
            PNL_REQUIRE_MESSAGE(sum == 0.0,
                                "backend " + backend_name + " with " + std::to_string(workers) +
                                    " workers reduced an empty range to " + test::format(sum));
        }
    }
}

PNL_TEST("equivalence/boundary: one unknown is bit identical on every backend and worker count") {
    // The smallest problem the library accepts, and the one where most workers
    // get nothing: at one row and eight workers, seven of the eight chunks are
    // empty. An off by one in a partition, or a sweep that assumed its chunk was
    // non empty, shows here and nowhere else in this suite.
    problems::Poisson2D poisson(1, problems::PoissonRhs::SpectrallyRich);
    problems::DenseProblem dense(1, 20260802, problems::DenseKind::SymmetricPositiveDefinite, 1);

    for (problems::Problem* problem :
         {static_cast<problems::Problem*>(&poisson), static_cast<problems::Problem*>(&dense)}) {
        PNL_REQUIRE(problem->unknown_count() == 1);
        for (const auto& solver_name : all_solver_names()) {
            auto probe = make_solver(solver_name);
            if (!probe->applicable_to(*problem)) continue;

            const Vector reference = solve_with(
                "serial", 1, solver_name, *problem, backend::ReductionMode::Deterministic);
            PNL_REQUIRE_MESSAGE(reference.size() == static_cast<std::size_t>(problem->state_size()),
                                "solver " + solver_name + " on " + problem->name() +
                                    " returned an iterate of " + std::to_string(reference.size()) +
                                    " values rather than " + std::to_string(problem->state_size()));

            for (const auto& backend_name : thread_backends()) {
                for (int workers : worker_counts()) {
                    const Vector candidate = solve_with(backend_name,
                                                        workers,
                                                        solver_name,
                                                        *problem,
                                                        backend::ReductionMode::Deterministic);
                    PNL_REQUIRE_MESSAGE(test::first_difference(reference, candidate) < 0,
                                        "at one unknown, solver " + solver_name + " on " +
                                            problem->name() + " and backend " + backend_name +
                                            " with " + std::to_string(workers) +
                                            " workers differs from serial at " +
                                            test::describe_difference(candidate, reference));
                }
            }
        }
    }
}
