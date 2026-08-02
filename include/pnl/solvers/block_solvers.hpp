#pragma once

/// \file block_solvers.hpp
/// Block Jacobi and block Gauss Seidel.

#include <pnl/solvers/splitting.hpp>

namespace pnl::solvers {

/// Block Jacobi.
///
/// Derivation. Partition the unknowns into contiguous blocks and write A in the
/// corresponding block form. Take M to be the block diagonal of A, so N is
/// everything off the block diagonal, and the iteration solves, for each block
/// p independently,
///
///     A_pp x_p^{k+1} = b_p - sum_{q != p} A_pq x_q^k.
///
/// Each block system is solved exactly: by LU with partial pivoting for a dense
/// problem, and by the Thomas algorithm for the 2D Poisson problem, whose
/// natural block is a grid line and whose diagonal block is therefore
/// tridiagonal. Point Jacobi is the special case of one unknown per block.
///
/// Why it helps. Solving the diagonal blocks exactly removes their contribution
/// to the error entirely, so the iteration matrix only has to contend with the
/// coupling between blocks. On the model Poisson problem with one grid line per
/// block this is line Jacobi, whose spectral radius is cos(pi h) / (2 - cos(pi h)),
/// smaller than the point Jacobi value of cos(pi h) by roughly a factor of two
/// in iteration count. The measured counts confirm that.
///
/// Parallel note. The blocks are independent by construction, so the sweep is
/// fully parallel. The block count is a solver parameter, never the worker
/// count: that is what keeps the iterates identical whatever the backend, and
/// it is why the equivalence suite can compare block methods bit for bit.
///
/// Reference: Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed.,
/// SIAM 2003, section 4.1.1; Golub and Van Loan, "Matrix Computations", 4th
/// ed., Johns Hopkins 2013, section 11.2.
class BlockJacobi final : public Solver {
   public:
    [[nodiscard]] std::string_view name() const noexcept override { return "block_jacobi"; }

    [[nodiscard]] std::string_view splitting() const noexcept override {
        return "M = block diagonal of A";
    }

    [[nodiscard]] bool applicable_to(const Problem&) const override { return true; }

    [[nodiscard]] SolveResult solve(Problem& problem, Backend& backend,
                                    const SolverOptions& options) const override {
        const Index blocks = detail::resolve_block_count(problem, options);
        auto sweep = [&](VectorView x, VectorView) {
            problem.block_sweep(backend, x, blocks, true);
        };
        return detail::run_stationary(problem, backend, options, "block_jacobi", sweep);
    }
};

/// Block Gauss Seidel.
///
/// Derivation. The same block partition, but M is the block lower triangle of
/// A, so blocks are visited in ascending order and each one reads the already
/// updated blocks before it:
///
///     A_pp x_p^{k+1} = b_p - sum_{q < p} A_pq x_q^{k+1} - sum_{q > p} A_pq x_q^k.
///
/// As in the point case, using the newest available values halves the iteration
/// count relative to the block Jacobi variant on the model problem, and as in
/// the point case it costs the parallelism between blocks: the blocks must be
/// visited in order. On the 2D Poisson problem with one line per block this is
/// line Gauss Seidel, and the measured ratio to line Jacobi is the block level
/// analogue of the point level ratio of two that Young's theory predicts.
///
/// Reference: Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed.,
/// SIAM 2003, section 4.1.1.
class BlockGaussSeidel final : public Solver {
   public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "block_gauss_seidel";
    }

    [[nodiscard]] std::string_view splitting() const noexcept override {
        return "M = block lower triangle of A";
    }

    [[nodiscard]] bool applicable_to(const Problem&) const override { return true; }

    [[nodiscard]] SolveResult solve(Problem& problem, Backend& backend,
                                    const SolverOptions& options) const override {
        const Index blocks = detail::resolve_block_count(problem, options);
        auto sweep = [&](VectorView x, VectorView) {
            problem.block_sweep(backend, x, blocks, false);
        };
        return detail::run_stationary(problem, backend, options, "block_gauss_seidel", sweep);
    }
};

}  // namespace pnl::solvers
