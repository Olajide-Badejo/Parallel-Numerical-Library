// SPDX-License-Identifier: MIT
#pragma once

/// \file golden_config.hpp
/// The one definition of the golden file configuration.
///
/// `tests/unit/test_golden.cpp` reads the committed files and
/// `tests/unit/write_golden.cpp` writes them. If those two disagreed about the
/// grid size, the right hand side, the seed, the iteration count, the backend
/// or the reduction mode, the test would compare one computation against a file
/// produced by another and would fail for a reason that has nothing to do with
/// the arithmetic it exists to watch. So neither of them spells the
/// configuration out; both call this.

#include <pnl/backend/backend.hpp>
#include <pnl/core/types.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace pnl::golden {

/// Interior points per side. Small on purpose: 31 gives 1089 padded values per
/// file and twelve files of about 26 KB, which is a size a reviewer can open
/// and a `git diff` can show, and 25 sweeps of it is a few milliseconds.
inline constexpr Index SIDE = 31;

/// The seed of the rich right hand side, and the seed every result row in this
/// repository carries.
inline constexpr std::uint64_t SEED = 20260802;

/// Sweeps. Fixed, never a tolerance: two builds that stopped at different
/// iterations would agree about the answer and disagree about the iterate, and
/// it is the iterate that carries the arithmetic.
inline constexpr Index ITERATIONS = 25;

/// The problem, built the same way in both programs.
///
/// `PoissonRhs::SpectrallyRich` rather than the constructor default, for two
/// reasons. It is seeded from `std::mt19937_64` and a
/// `std::uniform_real_distribution`, so its construction makes no `libm` call
/// at all and the files hold on any glibc; `ManufacturedSine` calls `std::sin`
/// twice per interior point, and `sin` is not correctly rounded and is not
/// required to give the same last bit across glibc versions, so a golden file
/// built from it would be a record of this machine's libm as much as of this
/// library. It is also the source Section 9.3 asks for, because it is not
/// symmetric under exchanging the two grid axes and a transposed layout is
/// therefore visible in the numbers.
[[nodiscard]] inline problems::Poisson2D problem() {
    return problems::Poisson2D(SIDE, problems::PoissonRhs::SpectrallyRich, SEED);
}

/// The run controls, identical in both programs.
[[nodiscard]] inline solvers::SolverOptions options() {
    solvers::SolverOptions run;
    run.mode = solvers::RunMode::FixedIterations;
    run.max_iterations = ITERATIONS;
    // Large enough that no residual evaluation falls inside the run. The
    // residual does not change the iterate for any solver here, so this is
    // about speed and about the file describing the sweeps and nothing else.
    run.check_interval = 1000000;
    // Zero means "ask the solver", which is the only default that is right for
    // all twelve: SOR wants Young's optimum, SSOR wants one and Richardson
    // wants the reciprocal of the Gershgorin bound.
    run.relaxation = 0.0;
    run.block_count = 0;
    return run;
}

/// The backend, identical in both programs: serial, one worker, deterministic
/// reduction. Serial because a golden file is about the arithmetic and not
/// about the execution model, and the equivalence suite is what ties every
/// other backend to this one.
[[nodiscard]] inline backend::Config backend_config() {
    backend::Config config;
    config.workers = 1;
    config.reduction = backend::ReductionMode::Deterministic;
    config.schedule = backend::Schedule::Static;
    return config;
}

/// A sentence naming the configuration, written into each file's header line
/// and checked by the test, so a file cannot be silently regenerated under a
/// different configuration and still read as the same record.
[[nodiscard]] inline std::string description() {
    return "Poisson2D(" + std::to_string(SIDE) + ",rich,seed=" + std::to_string(SEED) +
           ") backend=serial reduction=deterministic iterations=" + std::to_string(ITERATIONS);
}

/// Run one solver over the fixed configuration and return the iterate.
[[nodiscard]] inline Vector iterate(const std::string& solver_name) {
    problems::Poisson2D poisson = problem();
    auto execution = backend::make_backend("serial", backend_config());
    auto solver = solvers::make_solver(solver_name);
    return solver->solve(poisson, *execution, options()).solution;
}

}  // namespace pnl::golden
