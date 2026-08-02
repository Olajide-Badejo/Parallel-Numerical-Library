#pragma once

/// \file registry.hpp
/// Lookup of the solver zoo by name.
///
/// One list, used by the CLI, the sweep driver, the equivalence tests and the
/// generated documentation, so none of them can drift out of step with the
/// others or quietly omit a method.

#include <pnl/solvers/block_solvers.hpp>
#include <pnl/solvers/cg.hpp>
#include <pnl/solvers/gauss_seidel.hpp>
#include <pnl/solvers/jacobi.hpp>
#include <pnl/solvers/richardson.hpp>
#include <pnl/solvers/sor.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace pnl::solvers {

/// Construct every solver in the zoo, in the order the report presents them:
/// the splitting family from simplest to most refined, then the Krylov method.
[[nodiscard]] inline std::vector<std::unique_ptr<Solver>> all_solvers() {
    std::vector<std::unique_ptr<Solver>> solvers;
    solvers.push_back(std::make_unique<Richardson>());
    solvers.push_back(std::make_unique<Jacobi>());
    solvers.push_back(std::make_unique<GaussSeidelForward>());
    solvers.push_back(std::make_unique<GaussSeidelBackward>());
    solvers.push_back(std::make_unique<GaussSeidelSymmetric>());
    solvers.push_back(std::make_unique<GaussSeidelRedBlack>());
    solvers.push_back(std::make_unique<Sor>());
    solvers.push_back(std::make_unique<SymmetricSor>());
    solvers.push_back(std::make_unique<SorRedBlack>());
    solvers.push_back(std::make_unique<BlockJacobi>());
    solvers.push_back(std::make_unique<BlockGaussSeidel>());
    solvers.push_back(std::make_unique<ConjugateGradient>());
    return solvers;
}

/// The nine methods Objective 1 names, as a subset of the full list. The three
/// extras in all_solvers are the red black variants, which exist because the
/// GPU comparison of Section 8.3 needs a parallel Gauss Seidel, and symmetric
/// SOR, which completes the symmetric preconditioner story.
[[nodiscard]] inline std::vector<std::string> core_solver_names() {
    return {"richardson", "jacobi",       "gauss_seidel_f",     "gauss_seidel_b",
            "gauss_seidel_s", "sor",      "block_jacobi",       "block_gauss_seidel",
            "cg"};
}

/// Names of every solver, in registry order.
[[nodiscard]] inline std::vector<std::string> all_solver_names() {
    std::vector<std::string> names;
    for (const auto& solver : all_solvers()) names.emplace_back(solver->name());
    return names;
}

/// Construct one solver by name.
///
/// \throws InvalidArgument if the name is not in the registry, listing the
///         names that are.
[[nodiscard]] inline std::unique_ptr<Solver> make_solver(std::string_view name) {
    for (auto& solver : all_solvers()) {
        if (solver->name() == name) return std::move(solver);
    }
    std::string known;
    for (const auto& candidate : all_solver_names()) {
        if (!known.empty()) known += ", ";
        known += candidate;
    }
    throw InvalidArgument("unknown solver '" + std::string(name) + "'; known solvers are " +
                          known);
}

}  // namespace pnl::solvers
