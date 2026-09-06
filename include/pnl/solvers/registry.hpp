// SPDX-License-Identifier: MIT
#pragma once

/// \file registry.hpp
/// Lookup of the solver zoo by name, and the solver extension point.
///
/// One list, used by the CLI, the sweep driver, the equivalence tests and the
/// generated documentation, so none of them can drift out of step with the
/// others or quietly omit a method. Until release 1.1.0 that list was a fixed
/// sequence of push_back calls with no way in from outside, which is the solver
/// half of finding 4.8. It is a registry now, and a third party writes one
/// namespace scope object:
///
///     namespace {
///     const pnl::solvers::SolverRegistration my_solver(
///         "mine", [] { return std::make_unique<MySolver>(); });
///     }  // namespace
///
/// after which make_solver("mine") builds it and `pnl --list` names it.
///
/// **Order is part of the contract.** all_solver_names() returns registration
/// order, the built ins are registered in the order the report presents them,
/// and benchmarks/run_sweep.py and the `--list` output both read that order. A
/// registration a third party adds lands after all of them.
///
/// Unlike the backend registry this one is header only, because the solvers
/// are. The built ins are registered by the accessor on first use rather than
/// by namespace scope objects, so there is no static initialisation order to
/// reason about: a consumer's own registration always finds a populated
/// registry, whenever it runs.

#include <pnl/core/error.hpp>
#include <pnl/solvers/block_solvers.hpp>
#include <pnl/solvers/cg.hpp>
#include <pnl/solvers/gauss_seidel.hpp>
#include <pnl/solvers/jacobi.hpp>
#include <pnl/solvers/richardson.hpp>
#include <pnl/solvers/sor.hpp>

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pnl::solvers {

/// What a solver factory is: a call that returns a fresh solver.
///
/// Solvers are stateless and hold no configuration, so the factory takes no
/// arguments. What varies between runs is SolverOptions, which is passed to
/// solve() and not to construction.
using SolverFactory = std::function<std::unique_ptr<Solver>()>;

/// Name to factory, in registration order.
class SolverRegistry {
 public:
    /// Add \p factory under \p name.
    ///
    /// \throws InvalidArgument if \p name is empty or already registered. A
    ///         duplicate is refused rather than replaced, for the same reason
    ///         the backend registry refuses one: two solvers answering to a
    ///         single name would make every result row that carries it
    ///         ambiguous.
    void register_solver(std::string name, SolverFactory factory) {
        require(!name.empty(), "a solver name must not be empty");
        require(static_cast<bool>(factory), "a solver factory must not be empty");
        if (contains(name)) {
            require(false, "a solver named '" + name + "' is already registered");
        }
        entries_.push_back(Entry{std::move(name), std::move(factory)});
    }

    /// Whether \p name is registered.
    [[nodiscard]] bool contains(std::string_view name) const noexcept {
        for (const auto& entry : entries_) {
            if (entry.name == name) return true;
        }
        return false;
    }

    /// Every registered name, in registration order.
    [[nodiscard]] std::vector<std::string> names() const {
        std::vector<std::string> result;
        result.reserve(entries_.size());
        for (const auto& entry : entries_) result.push_back(entry.name);
        return result;
    }

    /// The registered names as one comma separated string, for a message.
    [[nodiscard]] std::string name_list() const {
        std::string known;
        for (const auto& entry : entries_) {
            if (!known.empty()) known += ", ";
            known += entry.name;
        }
        return known;
    }

    /// One of each, in registration order.
    [[nodiscard]] std::vector<std::unique_ptr<Solver>> create_all() const {
        std::vector<std::unique_ptr<Solver>> solvers;
        solvers.reserve(entries_.size());
        for (const auto& entry : entries_) solvers.push_back(entry.factory());
        return solvers;
    }

    /// Construct the solver registered under \p name.
    ///
    /// \throws InvalidArgument if the name is not in the registry, listing the
    ///         names that are.
    [[nodiscard]] std::unique_ptr<Solver> create(std::string_view name) const {
        for (const auto& entry : entries_) {
            if (entry.name == name) return entry.factory();
        }
        throw InvalidArgument("unknown solver '" + std::string(name) + "'; known solvers are " +
                              name_list());
    }

 private:
    struct Entry {
        std::string name;
        SolverFactory factory;
    };

    std::vector<Entry> entries_;
};

namespace detail {

/// The zoo, in the order the report presents it: the splitting family from
/// simplest to most refined, then the Krylov method. This is the order the
/// fixed list held before the registry existed. Do not sort it.
///
/// The name each solver registers under is the name its own name() returns.
/// tests/unit/test_registry.cpp asserts that pairing element by element, so a
/// literal here that drifts from the class fails a test rather than producing a
/// registry whose keys disagree with the result rows.
inline void register_builtin_solvers(SolverRegistry& registry) {
    registry.register_solver("richardson", [] { return std::make_unique<Richardson>(); });
    registry.register_solver("jacobi", [] { return std::make_unique<Jacobi>(); });
    registry.register_solver("gauss_seidel_f",
                             [] { return std::make_unique<GaussSeidelForward>(); });
    registry.register_solver("gauss_seidel_b",
                             [] { return std::make_unique<GaussSeidelBackward>(); });
    registry.register_solver("gauss_seidel_s",
                             [] { return std::make_unique<GaussSeidelSymmetric>(); });
    registry.register_solver("gauss_seidel_rb",
                             [] { return std::make_unique<GaussSeidelRedBlack>(); });
    registry.register_solver("sor", [] { return std::make_unique<Sor>(); });
    registry.register_solver("ssor", [] { return std::make_unique<SymmetricSor>(); });
    registry.register_solver("sor_rb", [] { return std::make_unique<SorRedBlack>(); });
    registry.register_solver("block_jacobi", [] { return std::make_unique<BlockJacobi>(); });
    registry.register_solver("block_gauss_seidel",
                             [] { return std::make_unique<BlockGaussSeidel>(); });
    registry.register_solver("cg", [] { return std::make_unique<ConjugateGradient>(); });
}

}  // namespace detail

/// The process wide solver registry, with the built ins already in it.
[[nodiscard]] inline SolverRegistry& solver_registry() {
    static SolverRegistry registry = [] {
        SolverRegistry created;
        detail::register_builtin_solvers(created);
        return created;
    }();
    return registry;
}

/// One line self registration, for a namespace scope object.
struct SolverRegistration {
    SolverRegistration(std::string name, SolverFactory factory) {
        solver_registry().register_solver(std::move(name), std::move(factory));
    }
};

/// Construct every solver in the registry, in registration order.
[[nodiscard]] inline std::vector<std::unique_ptr<Solver>> all_solvers() {
    return solver_registry().create_all();
}

/// The nine methods Objective 1 names, as a subset of the full list. The three
/// extras in all_solvers are the red black variants, which exist because the
/// GPU comparison of Section 8.3 needs a parallel Gauss Seidel, and symmetric
/// SOR, which completes the symmetric preconditioner story.
[[nodiscard]] inline std::vector<std::string> core_solver_names() {
    return {"richardson",
            "jacobi",
            "gauss_seidel_f",
            "gauss_seidel_b",
            "gauss_seidel_s",
            "sor",
            "block_jacobi",
            "block_gauss_seidel",
            "cg"};
}

/// Names of every solver, in registry order.
[[nodiscard]] inline std::vector<std::string> all_solver_names() {
    return solver_registry().names();
}

/// Construct one solver by name.
///
/// \throws InvalidArgument if the name is not in the registry, listing the
///         names that are.
[[nodiscard]] inline std::unique_ptr<Solver> make_solver(std::string_view name) {
    return solver_registry().create(name);
}

}  // namespace pnl::solvers
