#pragma once

/// \file dense_generator.hpp
/// Seeded dense linear systems.
///
/// The Poisson stencil is sparse, matrix free, and has one grid line as its
/// natural block. A dense system exercises the parts of the solver zoo that
/// the stencil cannot: block variants whose diagonal blocks are genuinely dense
/// and are factorised by the LU routine from the numerics module, exactly as
/// Section 8.2 requires.
///
/// Two families are generated, both from a recorded seed so the sweep is
/// reproducible:
///
///   DiagonallyDominant  strictly row diagonally dominant, which guarantees
///                       Jacobi and Gauss Seidel converge for any starting
///                       vector (Saad 2003, theorem 4.9) but is not symmetric,
///                       so conjugate gradient does not apply.
///
///   SymmetricPositiveDefinite  symmetric with a dominant positive diagonal,
///                       which makes it positive definite by Gershgorin and so
///                       admissible for conjugate gradient and for the SOR
///                       convergence theorem of Ostrowski and Reich.

#include <pnl/backend/backend.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>
#include <pnl/numerics/lu.hpp>
#include <pnl/problems/problem.hpp>

#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace pnl::problems {

/// Which family of dense system to generate.
enum class DenseKind {
    DiagonallyDominant,
    SymmetricPositiveDefinite,
};

[[nodiscard]] constexpr std::string_view to_string(DenseKind kind) noexcept {
    return kind == DenseKind::DiagonallyDominant ? "dd" : "spd";
}

/// A dense system A x = b with a known exact solution.
class DenseProblem final : public Problem {
   public:
    /// \param n order of the system.
    /// \param seed recorded in every result row so the problem can be rebuilt.
    /// \param kind which family to generate.
    /// \param block_count number of diagonal blocks the block solvers use.
    /// \throws InvalidArgument if n is not positive or block_count does not
    ///         divide the order into non empty blocks.
    DenseProblem(Index n, std::uint64_t seed, DenseKind kind, Index block_count = 8)
        : n_(n), seed_(seed), kind_(kind), block_count_(block_count) {
        require(n >= 1, "DenseProblem needs a positive order");
        require(block_count >= 1 && block_count <= n,
                "block_count must lie between one and the order");

        matrix_ = numerics::DenseMatrix(n);
        std::mt19937_64 engine(seed);
        std::uniform_real_distribution<Real> off_diagonal(-1.0, 1.0);

        if (kind == DenseKind::SymmetricPositiveDefinite) {
            for (Index i = 0; i < n; ++i) {
                for (Index j = i + 1; j < n; ++j) {
                    const Real value = off_diagonal(engine);
                    matrix_(i, j) = value;
                    matrix_(j, i) = value;
                }
            }
        } else {
            for (Index i = 0; i < n; ++i) {
                for (Index j = 0; j < n; ++j) {
                    if (i != j) matrix_(i, j) = off_diagonal(engine);
                }
            }
        }

        // Make every row strictly diagonally dominant. For the symmetric case
        // this also makes the matrix positive definite, because a symmetric
        // matrix with a strictly dominant positive diagonal has all its
        // Gershgorin discs in the right half plane.
        for (Index i = 0; i < n; ++i) {
            Real row_sum = 0.0;
            for (Index j = 0; j < n; ++j) {
                if (i != j) row_sum += std::abs(matrix_(i, j));
            }
            matrix_(i, i) = row_sum + DOMINANCE_MARGIN;
            // Gershgorin: every eigenvalue lies within row_sum of the diagonal,
            // so the largest possible magnitude is diagonal plus row sum.
            gershgorin_ = std::max(gershgorin_, matrix_(i, i) + row_sum);
        }

        // Choose the exact solution first, then form b, so the unit tests have
        // something to compare against that is not another solver's output.
        exact_.assign(static_cast<std::size_t>(n), 0.0);
        std::uniform_real_distribution<Real> solution_values(-1.0, 1.0);
        for (Index i = 0; i < n; ++i) {
            exact_[static_cast<std::size_t>(i)] = solution_values(engine);
        }

        rhs_.assign(static_cast<std::size_t>(n), 0.0);
        for (Index i = 0; i < n; ++i) {
            Real value = 0.0;
            const Real* row = matrix_.row(i);
            for (Index j = 0; j < n; ++j) value += row[j] * exact_[static_cast<std::size_t>(j)];
            rhs_[static_cast<std::size_t>(i)] = value;
        }

        factorise_diagonal_blocks();
    }

    [[nodiscard]] std::string name() const override {
        return std::string("dense_") + std::string(to_string(kind_)) + "_" + std::to_string(n_);
    }

    [[nodiscard]] Index unknown_count() const noexcept override { return n_; }

    [[nodiscard]] Index state_size() const noexcept override { return n_; }

    [[nodiscard]] Index natural_block_count() const noexcept override { return block_count_; }

    [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }

    [[nodiscard]] const numerics::DenseMatrix& matrix() const noexcept { return matrix_; }

    [[nodiscard]] ConstVectorView exact_solution() const noexcept { return exact_; }

    [[nodiscard]] Vector make_state() const override {
        return Vector(static_cast<std::size_t>(n_), 0.0);
    }

    [[nodiscard]] ConstVectorView rhs() const noexcept override { return rhs_; }

    [[nodiscard]] bool is_symmetric_positive_definite() const noexcept override {
        return kind_ == DenseKind::SymmetricPositiveDefinite;
    }

    [[nodiscard]] Real gershgorin_bound() const noexcept override { return gershgorin_; }

    /// A dense row sweep reads a whole matrix row per unknown, so the byte
    /// count per unknown per sweep is dominated by the n matrix entries rather
    /// than by the vectors. Stated so the roofline discussion can distinguish
    /// the dense case, which is compute bound at small n and bandwidth bound at
    /// large n, from the stencil case, which is always bandwidth bound.
    [[nodiscard]] Real bytes_per_unknown_per_sweep() const noexcept override {
        return static_cast<Real>(n_ + 3) * static_cast<Real>(sizeof(Real));
    }

    void apply(backend::Backend& backend, VectorView x, VectorView y) const override {
        const Range rows = backend.local_rows(n_);
        backend.parallel_for(rows.size(), [&](Range chunk) {
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                const Index i = rows.begin + k;
                const Real* row = matrix_.row(i);
                Real value = 0.0;
                for (Index j = 0; j < n_; ++j) value += row[j] * x[static_cast<std::size_t>(j)];
                y[static_cast<std::size_t>(i)] = value;
            }
        });
    }

    void jacobi_sweep(backend::Backend& backend, VectorView x, VectorView out) const override {
        const Range rows = backend.local_rows(n_);
        backend.parallel_for(rows.size(), [&](Range chunk) {
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                const Index i = rows.begin + k;
                const Real* row = matrix_.row(i);
                Real value = rhs_[static_cast<std::size_t>(i)];
                for (Index j = 0; j < n_; ++j) {
                    if (j != i) value -= row[j] * x[static_cast<std::size_t>(j)];
                }
                out[static_cast<std::size_t>(i)] = value / row[i];
            }
        });
    }

    void relaxation_sweep(backend::Backend& backend, VectorView x, Real relaxation,
                          Sweep direction) const override {
        require(relaxation > 0.0 && relaxation < 2.0,
                "relaxation factor must lie in (0, 2) for convergence on an SPD system");
        const Range rows = backend.local_rows(n_);
        const bool forward = direction == Sweep::Forward;

        auto update = [&](Index i) {
            const Real* row = matrix_.row(i);
            Real value = rhs_[static_cast<std::size_t>(i)];
            for (Index j = 0; j < n_; ++j) {
                if (j != i) value -= row[j] * x[static_cast<std::size_t>(j)];
            }
            const auto ii = static_cast<std::size_t>(i);
            x[ii] = (1.0 - relaxation) * x[ii] + relaxation * value / row[i];
        };

        backend.run_ordered(
            [&] {
                if (forward) {
                    for (Index i = rows.begin; i < rows.end; ++i) update(i);
                } else {
                    for (Index i = rows.end - 1; i >= rows.begin; --i) update(i);
                }
            },
            forward);
    }

    /// A general dense matrix has no two colouring, so this is not available.
    /// Reported rather than silently approximated.
    ///
    /// \throws InvalidArgument always.
    void coloured_sweep(backend::Backend&, VectorView, Real, Colour) const override {
        throw InvalidArgument(
            "a general dense system has no red black colouring; colouring exists for the five "
            "point stencil because its graph is bipartite");
    }

    /// Block relaxation with diagonal blocks solved exactly by LU.
    ///
    /// The factorisations are computed once in the constructor and reused every
    /// sweep, since the diagonal blocks never change. That is what makes the
    /// block methods competitive rather than merely correct.
    void block_sweep(backend::Backend& backend, VectorView x, Index block_count,
                     bool jacobi_coupling) const override {
        require(block_count == block_count_,
                "DenseProblem factorised its diagonal blocks for block_count = " +
                    std::to_string(block_count_) +
                    "; pass natural_block_count() so the cached factorisations apply");

        if (jacobi_coupling) {
            Vector previous(x.begin(), x.end());
            backend.parallel_for(block_count_, [&](Range chunk) {
                Vector local;
                for (Index b = chunk.begin; b < chunk.end; ++b) {
                    solve_block(previous, x, b, local);
                }
            });
        } else {
            backend.run_ordered(
                [&] {
                    Vector local;
                    for (Index b = 0; b < block_count_; ++b) solve_block(x, x, b, local);
                },
                true);
        }
    }

    Real residual(backend::Backend& backend, VectorView x, VectorView r) const override {
        const Range rows = backend.local_rows(n_);
        backend.parallel_for(rows.size(), [&](Range chunk) {
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                const Index i = rows.begin + k;
                const Real* row = matrix_.row(i);
                Real value = 0.0;
                for (Index j = 0; j < n_; ++j) value += row[j] * x[static_cast<std::size_t>(j)];
                r[static_cast<std::size_t>(i)] = rhs_[static_cast<std::size_t>(i)] - value;
            }
        });
        return norm(backend, r);
    }

    [[nodiscard]] Real dot(backend::Backend& backend, ConstVectorView x,
                           ConstVectorView y) const override {
        const Range rows = backend.local_rows(n_);
        return backend.reduce(rows.size(), 0.0, [&](Range chunk) -> Real {
            Real partial = 0.0;
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                const auto i = static_cast<std::size_t>(rows.begin + k);
                partial += x[i] * y[i];
            }
            return partial;
        });
    }

    void axpy(backend::Backend& backend, Real alpha, ConstVectorView x,
              VectorView y) const override {
        const Range rows = backend.local_rows(n_);
        backend.parallel_for(rows.size(), [&](Range chunk) {
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                const auto i = static_cast<std::size_t>(rows.begin + k);
                y[i] += alpha * x[i];
            }
        });
    }

    void xpby(backend::Backend& backend, ConstVectorView x, Real beta,
              VectorView y) const override {
        const Range rows = backend.local_rows(n_);
        backend.parallel_for(rows.size(), [&](Range chunk) {
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                const auto i = static_cast<std::size_t>(rows.begin + k);
                y[i] = x[i] + beta * y[i];
            }
        });
    }

    [[nodiscard]] Real rhs_norm(backend::Backend& backend) const override {
        return norm(backend, rhs_);
    }

   private:
    /// Margin added to the row sum to make the diagonal strictly dominant.
    /// Large enough that the generated systems are well conditioned and the
    /// splitting methods converge in a countable number of iterations, small
    /// enough that the problem is not trivial.
    static constexpr Real DOMINANCE_MARGIN = 1.0;

    void factorise_diagonal_blocks() {
        blocks_.clear();
        blocks_.reserve(static_cast<std::size_t>(block_count_));
        for (Index b = 0; b < block_count_; ++b) {
            const Range span = block_partition(n_, block_count_, b);
            numerics::DenseMatrix diagonal_block(span.size());
            for (Index i = 0; i < span.size(); ++i) {
                for (Index j = 0; j < span.size(); ++j) {
                    diagonal_block(i, j) = matrix_(span.begin + i, span.begin + j);
                }
            }
            blocks_.push_back(
                std::make_shared<numerics::LuFactorisation>(std::move(diagonal_block)));
        }
    }

    /// Solve block \p b, reading coupling terms from \p source and writing the
    /// updated unknowns into \p destination.
    void solve_block(ConstVectorView source, VectorView destination, Index b,
                     Vector& local) const {
        const Range span = block_partition(n_, block_count_, b);
        local.assign(static_cast<std::size_t>(span.size()), 0.0);
        for (Index i = 0; i < span.size(); ++i) {
            const Index row_index = span.begin + i;
            const Real* row = matrix_.row(row_index);
            Real value = rhs_[static_cast<std::size_t>(row_index)];
            // Everything outside this block is lagged onto the right hand side.
            for (Index j = 0; j < span.begin; ++j) {
                value -= row[j] * source[static_cast<std::size_t>(j)];
            }
            for (Index j = span.end; j < n_; ++j) {
                value -= row[j] * source[static_cast<std::size_t>(j)];
            }
            local[static_cast<std::size_t>(i)] = value;
        }
        blocks_[static_cast<std::size_t>(b)]->solve_in_place(local);
        for (Index i = 0; i < span.size(); ++i) {
            destination[static_cast<std::size_t>(span.begin + i)] =
                local[static_cast<std::size_t>(i)];
        }
    }

    Index n_;
    std::uint64_t seed_;
    DenseKind kind_;
    Index block_count_;
    Real gershgorin_ = 0.0;
    numerics::DenseMatrix matrix_;
    Vector rhs_;
    Vector exact_;
    std::vector<std::shared_ptr<numerics::LuFactorisation>> blocks_;
};

}  // namespace pnl::problems
