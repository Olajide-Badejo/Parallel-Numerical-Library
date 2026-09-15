// SPDX-License-Identifier: MIT
/// \file custom_backend.cpp
/// Register an execution backend of your own, and get the whole solver zoo.
///
/// This is the phase B4 extension point used the way a stranger would use it:
/// against an installed prefix, from a project that declares LANGUAGES CXX, with
/// no part of this repository in sight. The backend below is deliberately dull,
/// a serial one that counts the dispatches it serves, because the interesting
/// claim is not what it computes but that nothing else had to change. No solver,
/// no problem and no line of library code knows it exists, and the answer it
/// produces is bit identical to the built in serial backend's.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/chunking.hpp>
#include <pnl/backend/registry.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>

namespace {

/// Serial execution that counts what it was asked to do.
class CountingBackend final : public pnl::backend::Backend {
 public:
    explicit CountingBackend(const pnl::backend::Config& config) : config_(config) {
        config_.workers = 1;
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "counting"; }

    [[nodiscard]] int worker_count() const noexcept override { return 1; }

    [[nodiscard]] std::string pinning_status() const override { return "not_requested"; }

    void parallel_for(pnl::Index n, const pnl::backend::RangeBody& body) override {
        ++dispatches_;
        const pnl::Index chunks =
            pnl::backend::for_chunk_count(n, 1, config_.schedule, config_.chunks_per_worker);
        for (pnl::Index k = 0; k < chunks; ++k) {
            body(pnl::backend::for_chunk(n, 1, config_.schedule, config_.chunks_per_worker, k));
        }
    }

    /// The deterministic chunk grid, walked in ascending order. Walking it is
    /// what makes the reduction bit identical to every other backend's: the
    /// grid depends on the problem size alone, so the partials are the same
    /// numbers summed in the same order whoever computed them.
    [[nodiscard]] pnl::Real reduce(pnl::Index n,
                                   pnl::Real init,
                                   const pnl::backend::RangeReducer& reducer) override {
        ++dispatches_;
        const pnl::Index chunks = pnl::backend::reduction_chunk_count(n);
        pnl::Real total = init;
        for (pnl::Index k = 0; k < chunks; ++k) {
            total += reducer(pnl::backend::reduction_chunk(n, k));
        }
        return total;
    }

    void barrier() override {}

    [[nodiscard]] const pnl::backend::Config& config() const noexcept override { return config_; }

    [[nodiscard]] long long dispatches() const noexcept { return dispatches_; }

 private:
    pnl::backend::Config config_;
    long long dispatches_ = 0;
};

/// The one line. At namespace scope, so it runs before main and the name is in
/// the registry by the time anything asks for it.
const pnl::backend::BackendRegistration counting_registration(
    "counting", [](const pnl::backend::Config& config, const pnl::backend::TopologyReport&) {
        return std::make_unique<CountingBackend>(config);
    });

/// Twenty five conjugate gradient iterations on a small Poisson problem.
[[nodiscard]] pnl::Vector solve_with(const std::string& backend_name) {
    const pnl::backend::Config config;
    const std::unique_ptr<pnl::backend::Backend> execution =
        pnl::backend::make_backend(backend_name, config);
    pnl::problems::Poisson2D problem(63, pnl::problems::PoissonRhs::SpectrallyRich);
    pnl::solvers::SolverOptions options;
    options.mode = pnl::solvers::RunMode::FixedIterations;
    options.max_iterations = 25;
    const std::unique_ptr<pnl::solvers::Solver> solver = pnl::solvers::make_solver("cg");
    return solver->solve(problem, *execution, options).solution;
}

}  // namespace

int main() {
    std::printf("registered backends:");
    for (const auto& name : pnl::backend::available_backends()) {
        std::printf(" %s", name.c_str());
    }
    std::printf("\n");

    const pnl::Vector mine = solve_with("counting");
    const pnl::Vector reference = solve_with("serial");

    if (mine.size() != reference.size()) {
        std::printf("the two iterates are different lengths\n");
        return 1;
    }
    for (std::size_t i = 0; i < mine.size(); ++i) {
        if (mine[i] != reference[i]) {
            std::printf("the iterates differ at element %zu: %.17g against %.17g\n",
                        i,
                        mine[i],
                        reference[i]);
            return 1;
        }
    }
    std::printf("25 conjugate gradient iterations on a 63 by 63 Poisson problem:\n");
    std::printf("  the registered backend's iterate is bit identical to the serial one,\n");
    std::printf("  all %zu values, compared with == and not with a tolerance.\n", mine.size());
    return 0;
}
