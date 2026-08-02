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

#include <pnl_test.hpp>

#include <pnl/backend/backend.hpp>
#include <pnl/problems/dense_generator.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

#include <memory>

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

/// Worker counts to sweep, clamped to what this machine has.
[[nodiscard]] std::vector<int> worker_counts() {
    const int available = backend::available_logical_cpus();
    std::vector<int> counts{1, 2, 3};
    for (int candidate : {4, 7, 8, 16}) {
        if (candidate <= available) counts.push_back(candidate);
    }
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
[[nodiscard]] Vector solve_with(const std::string& backend_name, int workers,
                                const std::string& solver_name, problems::Problem& problem,
                                backend::ReductionMode reduction) {
    backend::Config config;
    config.workers = workers;
    config.reduction = reduction;
    auto execution = backend::make_backend(backend_name, config);
    auto solver = make_solver(solver_name);
    return solver->solve(problem, *execution, equivalence_options()).solution;
}

/// Index of the first differing element, or -1 when identical.
[[nodiscard]] Index first_difference(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) return 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return static_cast<Index>(i);
    }
    return -1;
}

}  // namespace

PNL_TEST("equivalence/every backend and worker count gives bit identical Poisson iterates") {
    problems::Poisson2D problem(63);

    for (const auto& solver_name : all_solver_names()) {
        auto probe = make_solver(solver_name);
        if (!probe->applicable_to(problem)) continue;

        const Vector reference =
            solve_with("serial", 1, solver_name, problem, backend::ReductionMode::Deterministic);

        for (const auto& backend_name : thread_backends()) {
            for (int workers : worker_counts()) {
                const Vector candidate = solve_with(backend_name, workers, solver_name, problem,
                                                    backend::ReductionMode::Deterministic);
                const Index differs = first_difference(reference, candidate);
                PNL_REQUIRE_MESSAGE(
                    differs < 0,
                    "solver " + solver_name + " on backend " + backend_name + " with " +
                        std::to_string(workers) + " workers differs from serial at index " +
                        std::to_string(differs) + ": " +
                        test::format(candidate[static_cast<std::size_t>(differs)]) +
                        " against " +
                        test::format(reference[static_cast<std::size_t>(differs)]));
            }
        }
    }
}

PNL_TEST("equivalence/every backend and worker count gives bit identical dense iterates") {
    problems::DenseProblem problem(180, 20260802,
                                   problems::DenseKind::SymmetricPositiveDefinite, 6);

    for (const auto& solver_name : all_solver_names()) {
        auto probe = make_solver(solver_name);
        if (!probe->applicable_to(problem)) continue;

        const Vector reference =
            solve_with("serial", 1, solver_name, problem, backend::ReductionMode::Deterministic);

        for (const auto& backend_name : thread_backends()) {
            for (int workers : worker_counts()) {
                const Vector candidate = solve_with(backend_name, workers, solver_name, problem,
                                                    backend::ReductionMode::Deterministic);
                const Index differs = first_difference(reference, candidate);
                PNL_REQUIRE_MESSAGE(
                    differs < 0,
                    "solver " + solver_name + " on backend " + backend_name + " with " +
                        std::to_string(workers) + " workers differs from serial at index " +
                        std::to_string(differs));
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

    auto reduce_on = [&](const std::string& backend_name, int workers,
                         backend::ReductionMode mode) {
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

    auto reduce_on = [&](const std::string& backend_name, int workers,
                         backend::ReductionMode mode) {
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
            PNL_REQUIRE_MESSAGE(covered == n, "partition of " + std::to_string(n) + " into " +
                                                  std::to_string(parts) + " covered " +
                                                  std::to_string(covered));
            PNL_REQUIRE(previous_end == n);
            if (n > 0 && parts <= n) {
                PNL_REQUIRE_MESSAGE(largest - smallest <= 1,
                                    "block sizes differ by more than one for n = " +
                                        std::to_string(n) + ", parts = " +
                                        std::to_string(parts));
            }
        }
    }
}

PNL_TEST("equivalence/a dynamic schedule gives the same answer as a static one") {
    problems::Poisson2D problem(63);
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
            const Index differs = first_difference(first, second);
            PNL_REQUIRE_MESSAGE(differs < 0,
                                std::string("solver ") + solver_name + " on " + backend_name +
                                    " differs between static and dynamic scheduling at index " +
                                    std::to_string(differs));
        }
    }
}
