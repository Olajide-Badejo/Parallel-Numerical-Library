// SPDX-License-Identifier: MIT
#pragma once

/// \file sor.hpp
/// Successive over relaxation and its symmetric and red black variants.

#include <pnl/solvers/splitting.hpp>

namespace pnl::solvers {

using problems::Colour;
using problems::Sweep;

/// Successive over relaxation.
///
/// Derivation. Take M = D / omega + L. Then N = M - A = (1 - omega) D / omega - U
/// and the iteration is, componentwise,
///
///     x_i^{k+1} = (1 - omega) x_i^k
///                 + (omega / a_ii) (b_i - sum_{j<i} a_ij x_j^{k+1}
///                                       - sum_{j>i} a_ij x_j^k),
///
/// that is, the Gauss Seidel update extrapolated by a factor omega. At omega = 1
/// it is exactly Gauss Seidel; omega greater than one over relaxes, which is
/// what accelerates the propagation of information across the grid.
///
/// Convergence. The theorem of Kahan gives the necessary condition
/// 0 < omega < 2, and the theorem of Ostrowski and Reich makes it sufficient
/// when A is symmetric positive definite. For a consistently ordered matrix
/// with property A, Young's theory gives the optimal factor in closed form,
///
///     omega* = 2 / (1 + sqrt(1 - rho_J^2)),   rho(G_{omega*}) = omega* - 1,
///
/// with rho_J the Jacobi spectral radius. On the model Poisson problem
/// rho_J = cos(pi h), so omega* = 2 / (1 + sin(pi h)) and the asymptotic rate
/// becomes 1 - O(h) instead of the 1 - O(h^2) of Jacobi and Gauss Seidel. That
/// is a change in the order of the iteration count, from O(n^2) to O(n) for an
/// n by n grid, and it is the single largest improvement available inside the
/// stationary family. The convergence tests check the measured optimum against
/// this closed form rather than assuming it.
///
/// Convergence order: linear, rate omega* - 1 at the optimum.
///
/// Reference: Young, "Iterative Solution of Large Linear Systems", Academic
/// Press 1971, chapters 4 to 6; Saad, "Iterative Methods for Sparse Linear
/// Systems", 2nd ed., SIAM 2003, section 4.2.
class Sor final : public Solver {
 public:
    [[nodiscard]] std::string_view name() const noexcept override { return "sor"; }

    [[nodiscard]] std::string_view splitting() const noexcept override {
        return "M = D / omega + L";
    }

    [[nodiscard]] bool applicable_to(const Problem&) const override { return true; }

    /// One extrapolated update per unknown in one traversal, exactly as Gauss
    /// Seidel, which SOR is at omega = 1.
    [[nodiscard]] WorkUnit work_unit() const noexcept override { return {1, 1}; }

    [[nodiscard]] Real relaxation_factor(const Problem& problem,
                                         const SolverOptions& options) const override {
        return resolve_relaxation(problem, options);
    }

    /// \param options relaxation is omega. A non positive value asks for the
    ///        closed form optimum when the problem is the model Poisson problem
    ///        and falls back to one otherwise.
    /// \throws InvalidArgument if omega is outside (0, 2), where the iteration
    ///         cannot converge by Kahan's theorem.
    using Solver::solve;

    [[nodiscard]] SolveReport solve(Problem& problem,
                                    Backend& backend,
                                    const SolverOptions& options,
                                    SolverWorkspace& workspace) const override {
        const Real omega = relaxation_factor(problem, options);
        require(omega > 0.0 && omega < 2.0,
                "SOR needs omega in (0, 2); outside that interval the spectral radius of the "
                "iteration matrix is at least one by Kahan's theorem");
        auto sweep = [&](VectorView x, VectorView) {
            problem.relaxation_sweep(backend, x, omega, Sweep::Forward);
            // In place, as every relaxation sweep is.
            return x;
        };
        return detail::run_stationary(
            problem, backend, options, "sor", work_unit(), workspace, sweep);
    }

    /// The relaxation factor this solver would use, exposed so the sweep driver
    /// can record it in the result row.
    ///
    /// It used to ask a dynamic_cast to Poisson2D and fall back to one for
    /// anything else, so a third party's own SPD stencil was given omega = 1
    /// with no diagnostic and no way to say otherwise, whatever its author knew
    /// about its spectrum. Problem::suggested_relaxation() is the same number
    /// for Poisson2D, the same fallback for a problem that does not override it,
    /// and a question a third party can now answer. Finding 4.8.
    [[nodiscard]] static Real resolve_relaxation(const Problem& problem,
                                                 const SolverOptions& options) {
        if (options.relaxation > 0.0) return options.relaxation;
        return problem.suggested_relaxation();
    }
};

/// Symmetric SOR: a forward sweep followed by a backward sweep, both at omega.
///
/// As with symmetric Gauss Seidel the point is that the resulting
/// preconditioner is symmetric when A is, which is what a conjugate gradient
/// preconditioner requires. Its optimal omega is not the SOR optimum and is
/// generally closer to one; the sweep records whatever factor was used.
class SymmetricSor final : public Solver {
 public:
    [[nodiscard]] std::string_view name() const noexcept override { return "ssor"; }

    [[nodiscard]] std::string_view splitting() const noexcept override {
        return "M = (D / omega + L) (D / omega)^-1 (D / omega + U) / (2 - omega)";
    }

    [[nodiscard]] bool applicable_to(const Problem&) const override { return true; }

    /// Two updates per unknown in two traversals, as symmetric Gauss Seidel,
    /// which SSOR is at omega = 1.
    [[nodiscard]] WorkUnit work_unit() const noexcept override { return {2, 2}; }

    /// Its own default, which is one rather than the SOR optimum. The result
    /// row used to record the SOR optimum on an SSOR run that never used it.
    [[nodiscard]] Real relaxation_factor(const Problem&,
                                         const SolverOptions& options) const override {
        return options.relaxation > 0.0 ? options.relaxation : 1.0;
    }

    using Solver::solve;

    [[nodiscard]] SolveReport solve(Problem& problem,
                                    Backend& backend,
                                    const SolverOptions& options,
                                    SolverWorkspace& workspace) const override {
        const Real omega = relaxation_factor(problem, options);
        require(omega > 0.0 && omega < 2.0, "SSOR needs omega in (0, 2)");
        auto sweep = [&](VectorView x, VectorView) {
            problem.relaxation_sweep(backend, x, omega, Sweep::Forward);
            problem.relaxation_sweep(backend, x, omega, Sweep::Backward);
            return x;
        };
        return detail::run_stationary(
            problem, backend, options, "ssor", work_unit(), workspace, sweep);
    }
};

/// Red black SOR: the over relaxed update applied to each colour in turn.
///
/// This is the fully parallel form of SOR and the one a GPU can run. Because
/// the red black ordering is still consistently ordered in Young's sense, the
/// closed form optimal omega carries over unchanged, which is the reason this
/// variant is the standard choice for parallel stencil codes: it buys the full
/// O(n) to O(n) improvement of SOR without giving up parallelism.
class SorRedBlack final : public Solver {
 public:
    [[nodiscard]] std::string_view name() const noexcept override { return "sor_rb"; }

    [[nodiscard]] std::string_view splitting() const noexcept override {
        return "M = D / omega + L in red black ordering";
    }

    [[nodiscard]] bool applicable_to(const Problem& problem) const override {
        return problem.supports_colouring();
    }

    [[nodiscard]] std::string inapplicable_reason(const Problem&) const override {
        return "red black ordering needs a bipartite coupling graph, which the five point "
               "stencil has and a general dense system does not";
    }

    /// One update per unknown in two colour passes, as red black Gauss Seidel.
    [[nodiscard]] WorkUnit work_unit() const noexcept override { return {1, 2}; }

    /// The closed form optimum carries over to the red black ordering
    /// unchanged, so this is the same factor plain SOR would use.
    [[nodiscard]] Real relaxation_factor(const Problem& problem,
                                         const SolverOptions& options) const override {
        return Sor::resolve_relaxation(problem, options);
    }

    using Solver::solve;

    [[nodiscard]] SolveReport solve(Problem& problem,
                                    Backend& backend,
                                    const SolverOptions& options,
                                    SolverWorkspace& workspace) const override {
        // Built only when it is needed; see the note in gauss_seidel.hpp.
        if (!problem.supports_colouring()) require(false, inapplicable_reason(problem));
        const Real omega = relaxation_factor(problem, options);
        require(omega > 0.0 && omega < 2.0, "red black SOR needs omega in (0, 2)");
        auto sweep = [&](VectorView x, VectorView) {
            problem.coloured_sweep(backend, x, omega, Colour::Red);
            problem.coloured_sweep(backend, x, omega, Colour::Black);
            return x;
        };
        return detail::run_stationary(
            problem, backend, options, "sor_rb", work_unit(), workspace, sweep);
    }
};

}  // namespace pnl::solvers
