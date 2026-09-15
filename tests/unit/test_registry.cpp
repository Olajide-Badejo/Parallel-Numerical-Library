// SPDX-License-Identifier: MIT
/// \file test_registry.cpp
/// The phase B4 gate: a backend registered from a test translation unit is
/// constructible by name, and the built in names keep the order everything else
/// reads.
///
/// The registration below is the whole point of the file. It happens here, in a
/// translation unit the library has never seen, through the same public
/// BackendRegistration a consumer writes, and then make_backend() builds the
/// result. That is the extension point working end to end.
///
/// It also proves something less obvious that no other test could. pnl_core is
/// a static library and a linker drops an archive member nothing references, so
/// if the built in registrations lived in an object file that only defined
/// namespace scope objects, that file would be silently dropped, the registry
/// would hold nothing but what this file put in it, and the built in name
/// assertions below would fail. They pass because backend_registry() itself is
/// defined in src/backend/builtin_backends.cpp and factory.cpp calls it.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/chunking.hpp>
#include <pnl/backend/registry.hpp>
#include <pnl/backend/topology.hpp>
#include <pnl/core/error.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>
#include <pnl/solvers/splitting.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pnl_test.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace pnl;

/// A backend of the kind a third party would write: serial underneath, counting
/// what it was asked to do so the test can prove the dispatches went through it
/// and not through something else that happened to answer to the name.
class CountingBackend final : public backend::Backend {
 public:
    explicit CountingBackend(const backend::Config& config) : config_(config) {
        config_.workers = 1;
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "test_counting"; }

    [[nodiscard]] int worker_count() const noexcept override { return 1; }

    [[nodiscard]] std::string pinning_status() const override { return "not_requested"; }

    void parallel_for(Index n, const backend::RangeBody& body) override {
        ++dispatches_;
        const Index chunks =
            backend::for_chunk_count(n, 1, config_.schedule, config_.chunks_per_worker);
        for (Index k = 0; k < chunks; ++k) {
            body(backend::for_chunk(n, 1, config_.schedule, config_.chunks_per_worker, k));
        }
    }

    [[nodiscard]] Real reduce(Index n, Real init, const backend::RangeReducer& reducer) override {
        ++dispatches_;
        const Index chunks = backend::reduction_chunk_count(n);
        Real total = init;
        for (Index k = 0; k < chunks; ++k) total += reducer(backend::reduction_chunk(n, k));
        return total;
    }

    void barrier() override {}

    [[nodiscard]] const backend::Config& config() const noexcept override { return config_; }

    /// Dispatches this object has served, over its whole life.
    [[nodiscard]] long long dispatches() const noexcept { return dispatches_; }

 private:
    backend::Config config_;
    long long dispatches_ = 0;
};

/// A solver of the kind a third party would write: one Jacobi sweep per
/// iteration, which is the simplest thing that drives the shared stationary
/// loop and therefore proves the registered object is really being solved with.
class CountingSolver final : public solvers::Solver {
 public:
    [[nodiscard]] std::string_view name() const noexcept override { return "test_counting"; }

    [[nodiscard]] std::string_view splitting() const noexcept override {
        return "M = D, registered from a test translation unit";
    }

    [[nodiscard]] bool applicable_to(const problems::Problem&) const override { return true; }

    [[nodiscard]] solvers::WorkUnit work_unit() const noexcept override { return {1, 1}; }

    using Solver::solve;

    [[nodiscard]] solvers::SolveReport solve(problems::Problem& problem,
                                             backend::Backend& execution,
                                             const solvers::SolverOptions& options,
                                             solvers::SolverWorkspace& workspace) const override {
        auto sweep = [&](VectorView x, VectorView work) {
            problem.jacobi_sweep(execution, x, work);
            return work;
        };
        return solvers::detail::run_stationary(
            problem, execution, options, "test_counting", work_unit(), workspace, sweep);
    }
};

/// The one line a consumer writes. At namespace scope, so it runs before main
/// and the registry is populated by the time any case below looks at it.
const backend::BackendRegistration counting_backend_registration(
    "test_counting", [](const backend::Config& config, const backend::TopologyReport&) {
        return std::make_unique<CountingBackend>(config);
    });

const solvers::SolverRegistration counting_solver_registration("test_counting", [] {
    return std::make_unique<CountingSolver>();
});

/// The built in backend names, under the same guards builtin_backends.cpp uses,
/// in the order the fixed list held before the registry existed.
[[nodiscard]] std::vector<std::string> expected_builtin_backends() {
    std::vector<std::string> names{"serial"};
#if defined(PNL_WITH_OPENMP)
    names.emplace_back("openmp");
#endif
    names.emplace_back("pthreads");
    names.emplace_back("jthread");
#if defined(PNL_WITH_MPI)
    names.emplace_back("mpi");
#if defined(PNL_WITH_OPENMP)
    names.emplace_back("hybrid");
#endif
#endif
    return names;
}

/// The twelve built in solvers, in the order the report presents them.
[[nodiscard]] std::vector<std::string> expected_builtin_solvers() {
    return {"richardson",
            "jacobi",
            "gauss_seidel_f",
            "gauss_seidel_b",
            "gauss_seidel_s",
            "gauss_seidel_rb",
            "sor",
            "ssor",
            "sor_rb",
            "block_jacobi",
            "block_gauss_seidel",
            "cg"};
}

[[nodiscard]] std::string joined(const std::vector<std::string>& names) {
    std::string text;
    for (const auto& name : names) {
        if (!text.empty()) text += ", ";
        text += name;
    }
    return text;
}

}  // namespace

PNL_TEST("registry/a backend registered here is constructible through make_backend") {
    backend::Config config;
    config.workers = 1;
    const std::unique_ptr<backend::Backend> execution =
        backend::make_backend("test_counting", config);
    PNL_REQUIRE(execution->name() == "test_counting");
    PNL_REQUIRE(execution->worker_count() == 1);

    // And it really is the object this file defined, doing the work.
    const auto* counting = dynamic_cast<const CountingBackend*>(execution.get());
    PNL_REQUIRE(counting != nullptr);

    problems::Poisson2D problem(15, problems::PoissonRhs::SpectrallyRich);
    Vector x = problem.make_state();
    Vector out = problem.make_state();
    problem.jacobi_sweep(*execution, x, out);
    PNL_REQUIRE(counting->dispatches() >= 1);
}

PNL_TEST("registry/a registered backend appears in available_backends") {
    const std::vector<std::string> names = backend::available_backends();
    PNL_REQUIRE(std::find(names.begin(), names.end(), "test_counting") != names.end());
}

PNL_TEST("registry/the built in backend names keep their original order") {
    const std::vector<std::string> names = backend::available_backends();
    const std::vector<std::string> expected = expected_builtin_backends();
    PNL_REQUIRE_MESSAGE(names.size() > expected.size(),
                        "available_backends returned " + joined(names) +
                            ", which does not contain the built ins plus this test's backend");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        PNL_REQUIRE_MESSAGE(names[i] == expected[i],
                            "available_backends returned " + joined(names) + " where position " +
                                std::to_string(i) + " should be " + expected[i]);
    }
    // A registration from outside the library lands after every built in, so
    // nothing that was in the list before this registry existed can move.
    PNL_REQUIRE(names[expected.size()] == "test_counting");
}

PNL_TEST("registry/an unknown backend name says what is available") {
    backend::Config config;
    bool threw = false;
    try {
        (void)backend::make_backend("no_such_backend", config);
    } catch (const InvalidArgument& error) {
        threw = true;
        const std::string what = error.what();
        PNL_REQUIRE_MESSAGE(what.find("is not available in this build; available backends are") !=
                                std::string::npos,
                            "the message lost its wording: " + what);
        PNL_REQUIRE_MESSAGE(what.find("serial") != std::string::npos,
                            "the message does not list the available names: " + what);
    }
    PNL_REQUIRE(threw);
}

PNL_TEST("registry/a backend name cannot be registered twice") {
    PNL_REQUIRE_THROWS(backend::backend_registry().register_factory(
                           "serial",
                           [](const backend::Config& config, const backend::TopologyReport&) {
                               return std::make_unique<CountingBackend>(config);
                           }),
                       InvalidArgument);
}

PNL_TEST("registry/a solver registered here is constructible through make_solver") {
    const std::unique_ptr<solvers::Solver> solver = solvers::make_solver("test_counting");
    PNL_REQUIRE(solver->name() == "test_counting");

    // And it solves, over a built in backend, with no solver code knowing it is
    // not one of the twelve.
    backend::Config config;
    config.workers = 1;
    const std::unique_ptr<backend::Backend> execution = backend::make_backend("serial", config);
    problems::Poisson2D problem(15, problems::PoissonRhs::SpectrallyRich);
    solvers::SolverOptions options;
    options.mode = solvers::RunMode::FixedIterations;
    options.max_iterations = 5;
    const SolveResult result = solver->solve(problem, *execution, options);
    PNL_REQUIRE(result.diagnostics.iterations == 5);
    PNL_REQUIRE(std::isfinite(result.diagnostics.error_estimate));
}

PNL_TEST("registry/the built in solver names keep their original order") {
    const std::vector<std::string> names = solvers::all_solver_names();
    const std::vector<std::string> expected = expected_builtin_solvers();
    PNL_REQUIRE_MESSAGE(names.size() == expected.size() + 1,
                        "all_solver_names returned " + joined(names));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        PNL_REQUIRE_MESSAGE(names[i] == expected[i],
                            "all_solver_names returned " + joined(names) + " where position " +
                                std::to_string(i) + " should be " + expected[i]);
    }
    PNL_REQUIRE(names[expected.size()] == "test_counting");
}

PNL_TEST("registry/every solver answers to the name it is registered under") {
    const std::vector<std::string> names = solvers::all_solver_names();
    const std::vector<std::unique_ptr<solvers::Solver>> solvers_built = solvers::all_solvers();
    PNL_REQUIRE(names.size() == solvers_built.size());
    for (std::size_t i = 0; i < names.size(); ++i) {
        PNL_REQUIRE_MESSAGE(solvers_built[i]->name() == names[i],
                            "solver " + std::to_string(i) + " is registered as " + names[i] +
                                " and calls itself " + std::string(solvers_built[i]->name()));
    }
}

PNL_TEST("registry/an unknown solver name says what is known") {
    bool threw = false;
    try {
        (void)solvers::make_solver("no_such_solver");
    } catch (const InvalidArgument& error) {
        threw = true;
        const std::string what = error.what();
        PNL_REQUIRE_MESSAGE(
            what.find("unknown solver 'no_such_solver'; known solvers are") != std::string::npos,
            "the message lost its wording: " + what);
    }
    PNL_REQUIRE(threw);
}

PNL_TEST("registry/the shared topology a caller holds is not rewritten underneath it") {
    // Section 4.7: the cheap pass and the probing pass shared one cached
    // report, and the probing pass reassigned it wholesale. Every backend
    // constructor copies a TopologyReport out of the reference this function
    // returns, so on another thread that reassignment frees the vectors a copy
    // in progress is reading. Single threaded it is still wrong, and this is
    // the single threaded half of it: a caller who took the cheap report found
    // its contents replaced by a later call it had nothing to do with.
    //
    // The probe below is the expensive part of this file, a few seconds of
    // timing one kernel on every logical processor. It runs once per process
    // and there is no cheaper way to reach the second cache.
    const backend::TopologyReport& cheap = backend::shared_topology(false);
    const int cpus = cheap.logical_cpus;
    const std::size_t leaders = cheap.core_leaders.size();
    const std::string verdict = cheap.verdict;
    const void* const address = &cheap;

    const backend::TopologyReport& probed = backend::shared_topology(true);

    PNL_REQUIRE_MESSAGE(&cheap == address, "the cheap report moved");
    PNL_REQUIRE_MESSAGE(cheap.logical_cpus == cpus,
                        "the processor count changed from " + std::to_string(cpus) + " to " +
                            std::to_string(cheap.logical_cpus) + " under a held reference");
    PNL_REQUIRE_MESSAGE(cheap.core_leaders.size() == leaders,
                        "the core leader list was reallocated under a held reference");
    PNL_REQUIRE_MESSAGE(cheap.verdict == verdict,
                        "the verdict changed from '" + verdict + "' to '" + cheap.verdict +
                            "' under a held reference");

    // The probing pass still answers with the cheap facts where it has nothing
    // better, which is what the pcore refusal message quotes.
    PNL_REQUIRE(probed.logical_cpus == cpus);
    PNL_REQUIRE_MESSAGE(!probed.verdict.empty(), "the probed report has no verdict to quote");

    // And the caches are stable: asking twice returns the same objects.
    PNL_REQUIRE(&backend::shared_topology(false) == &cheap);
    PNL_REQUIRE(&backend::shared_topology(true) == &probed);
}
