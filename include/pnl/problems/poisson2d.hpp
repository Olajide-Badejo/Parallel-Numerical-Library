#pragma once

/// \file poisson2d.hpp
/// The 2D Poisson equation on the unit square, discretised by the five point
/// stencil. This is the canonical model problem: its eigenvalues are known in
/// closed form, so Jacobi, Gauss Seidel and optimal SOR all have predicted
/// convergence rates that the tests check the implementation against rather
/// than taking on trust.
///
/// Discretisation. On an (n+1) by (n+1) uniform mesh of the unit square with
/// spacing h = 1/(n+1), interior unknowns u(i,j) for i and j in [1, n] satisfy
///
///     4 u(i,j) - u(i-1,j) - u(i+1,j) - u(i,j-1) - u(i,j+1) = h^2 f(i,j)
///
/// after multiplying through by h^2. The resulting matrix is symmetric positive
/// definite with diagonal 4 and off diagonals -1, which is what conjugate
/// gradient requires and what the closed form spectral radii below describe.
///
/// Storage. The grid is stored with a one cell boundary ring, so the array is
/// (n+2) by (n+2) row major with stride n+2 and the interior begins at (1,1).
/// The ring holds the homogeneous Dirichlet values and is never updated, which
/// removes every boundary branch from the innermost loop. On a distributed
/// backend the same ring doubles as the halo.
///
/// References: Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed.,
/// SIAM 2003, chapters 4 and 12; Young, "Iterative Solution of Large Linear
/// Systems", Academic Press 1971, for the optimal relaxation factor; Golub and
/// Van Loan, "Matrix Computations", 4th ed., Johns Hopkins 2013, section 11.2.

#include <pnl/backend/backend.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>
#include <pnl/problems/problem.hpp>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace pnl::problems {

/// Closed form spectral quantities for the model problem, used by the
/// convergence tests. Grouped here so the theory lives beside the operator it
/// describes and no test re-derives it.
struct PoissonTheory {
    /// Spectral radius of the Jacobi iteration matrix, cos(pi h).
    Real jacobi_spectral_radius = 0.0;
    /// Spectral radius of natural ordering Gauss Seidel, which for this
    /// consistently ordered matrix is exactly the square of the Jacobi value
    /// (Young 1971, theorem 4.3).
    Real gauss_seidel_spectral_radius = 0.0;
    /// Optimal SOR factor, 2 / (1 + sin(pi h)).
    Real optimal_relaxation = 0.0;
    /// Spectral radius of SOR at the optimal factor, which is that factor
    /// minus one.
    Real optimal_sor_spectral_radius = 0.0;

    /// Ratio of Jacobi to Gauss Seidel iteration counts predicted by the two
    /// spectral radii. Asymptotically this is two.
    [[nodiscard]] Real jacobi_to_gauss_seidel_ratio() const noexcept {
        return std::log(gauss_seidel_spectral_radius) / std::log(jacobi_spectral_radius);
    }
};

/// Compute the closed form theory for an n by n interior grid.
[[nodiscard]] inline PoissonTheory poisson_theory(Index n) {
    require(n >= 1, "poisson_theory needs at least one interior point per side");
    const Real h_pi = std::numbers::pi_v<Real> / static_cast<Real>(n + 1);
    PoissonTheory theory;
    theory.jacobi_spectral_radius = std::cos(h_pi);
    theory.gauss_seidel_spectral_radius =
        theory.jacobi_spectral_radius * theory.jacobi_spectral_radius;
    theory.optimal_relaxation = 2.0 / (1.0 + std::sin(h_pi));
    theory.optimal_sor_spectral_radius = theory.optimal_relaxation - 1.0;
    return theory;
}

/// Which right hand side to build.
///
/// This choice matters more than it looks, and getting it wrong invalidated a
/// convergence test before it was caught.
///
/// The eigenvectors of the five point stencil are sin(p pi x) sin(q pi y). The
/// obvious manufactured solution u = sin(pi x) sin(pi y) is therefore exactly
/// one eigenvector, so with a zero initial guess the whole Krylov subspace that
/// conjugate gradient builds is one dimensional and it converges in a single
/// iteration at every grid size. That is not conjugate gradient being fast, it
/// is the problem being degenerate.
///
/// The stationary methods are unaffected, because the slowest decaying mode of
/// the Jacobi iteration matrix is that same lowest mode, so a single mode right
/// hand side isolates the asymptotic rate cleanly and gives exactly the closed
/// form spectral radius. Both kinds are therefore kept, and each is used where
/// it is honest.
enum class PoissonRhs {
    /// u = sin(pi x) sin(pi y) with the matching source. Has a closed form
    /// solution, so the discretisation error test can measure the truncation
    /// error against theory. Degenerate for Krylov methods.
    ManufacturedSine,
    /// A seeded pseudo random source, which excites the whole spectrum. No
    /// closed form solution, but it is the right choice for any comparison
    /// involving conjugate gradient and for the benchmark sweep, where a
    /// degenerate spectrum would flatter one method over the others.
    SpectrallyRich,
};

[[nodiscard]] constexpr std::string_view to_string(PoissonRhs kind) noexcept {
    return kind == PoissonRhs::ManufacturedSine ? "sine" : "rich";
}

/// The five point Poisson operator.
class Poisson2D final : public Problem {
   public:
    /// \param n interior points per side; the system has n^2 unknowns.
    /// \param kind which right hand side to build; see PoissonRhs.
    /// \param seed used only by PoissonRhs::SpectrallyRich, and recorded in
    ///        every result row so the problem can be rebuilt exactly.
    /// \throws InvalidArgument if n is less than one.
    explicit Poisson2D(Index n, PoissonRhs kind = PoissonRhs::ManufacturedSine,
                       std::uint64_t seed = 20260802)
        : n_(n), stride_(n + 2), kind_(kind), seed_(seed) {
        require(n >= 1, "Poisson2D needs at least one interior point per side");
        const Index cells = stride_ * stride_;
        rhs_.assign(static_cast<std::size_t>(cells), 0.0);
        exact_.assign(static_cast<std::size_t>(cells), 0.0);

        const Real h = 1.0 / static_cast<Real>(n + 1);
        const Real pi = std::numbers::pi_v<Real>;

        if (kind == PoissonRhs::ManufacturedSine) {
            const Real scale = h * h * 2.0 * pi * pi;
            for (Index i = 1; i <= n_; ++i) {
                const Real y = static_cast<Real>(i) * h;
                for (Index j = 1; j <= n_; ++j) {
                    const Real x = static_cast<Real>(j) * h;
                    const Real u = std::sin(pi * x) * std::sin(pi * y);
                    rhs_[static_cast<std::size_t>(at(i, j))] = scale * u;
                    exact_[static_cast<std::size_t>(at(i, j))] = u;
                }
            }
        } else {
            // A seeded source with no smoothness, so every eigenmode is
            // present. Scaled by h^2 to match the scaling of the operator, so
            // the solution stays O(1) as the grid refines.
            std::mt19937_64 engine(seed);
            std::uniform_real_distribution<Real> values(-1.0, 1.0);
            const Real scale = h * h;
            for (Index i = 1; i <= n_; ++i) {
                for (Index j = 1; j <= n_; ++j) {
                    rhs_[static_cast<std::size_t>(at(i, j))] = scale * values(engine);
                }
            }
            // No closed form solution exists for this source, and exact_ stays
            // zero. exact_solution() reports that through has_exact_solution().
        }
        theory_ = poisson_theory(n_);
    }

    [[nodiscard]] std::string name() const override {
        return "poisson2d_" + std::string(to_string(kind_)) + "_" + std::to_string(n_);
    }

    /// True when exact_solution() carries a genuine closed form solution.
    [[nodiscard]] bool has_exact_solution() const noexcept {
        return kind_ == PoissonRhs::ManufacturedSine;
    }

    [[nodiscard]] PoissonRhs rhs_kind() const noexcept { return kind_; }

    [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }

    [[nodiscard]] Index unknown_count() const noexcept override { return n_ * n_; }

    [[nodiscard]] Index state_size() const noexcept override { return stride_ * stride_; }

    [[nodiscard]] Index side() const noexcept { return n_; }

    [[nodiscard]] Index stride() const noexcept { return stride_; }

    [[nodiscard]] const PoissonTheory& theory() const noexcept { return theory_; }

    /// The manufactured continuous solution sampled on the grid, in grid
    /// layout. Used by the discretisation error test.
    [[nodiscard]] ConstVectorView exact_solution() const noexcept { return exact_; }

    [[nodiscard]] Vector make_state() const override {
        // Zero initial guess, and the boundary ring is already the homogeneous
        // Dirichlet data, so nothing further is needed.
        return Vector(static_cast<std::size_t>(state_size()), 0.0);
    }

    [[nodiscard]] ConstVectorView rhs() const noexcept override { return rhs_; }

    [[nodiscard]] Index natural_block_count() const noexcept override { return n_; }

    [[nodiscard]] bool supports_colouring() const noexcept override { return true; }

    [[nodiscard]] bool is_symmetric_positive_definite() const noexcept override { return true; }

    /// Every row of the five point stencil has diagonal 4 and at most four off
    /// diagonal entries of magnitude 1, so Gershgorin gives 8. The true largest
    /// eigenvalue is 4 + 4 cos(pi h), which approaches 8 from below, so the
    /// bound is tight as the grid refines.
    [[nodiscard]] Real gershgorin_bound() const noexcept override { return 8.0; }

    /// A Jacobi sweep reads five values and writes one per unknown. The read of
    /// the right hand side and the four neighbour reads are counted; the
    /// neighbour reads are what a perfectly cached implementation would still
    /// have to move from memory once per unknown in the streaming limit, which
    /// is the regime the 1024 squared and larger grids sit in.
    ///
    /// Counted, not estimated: one write of x_new (8 bytes), one read of b
    /// (8 bytes), and one read of x_old (8 bytes) per unknown, since the four
    /// neighbours of consecutive unknowns overlap and a streaming sweep touches
    /// each x_old value a constant number of times. Section 8.3 requires this
    /// number to be stated rather than assumed, and the roofline discussion in
    /// the report uses exactly this value.
    [[nodiscard]] Real bytes_per_unknown_per_sweep() const noexcept override {
        return 3.0 * static_cast<Real>(sizeof(Real));
    }

    void apply(backend::Backend& backend, VectorView x, VectorView y) const override {
        backend.exchange_halo(x, stride_, n_);
        const Range rows = backend.local_rows(n_);
        backend.parallel_for(rows.size(), [&](Range chunk) {
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                const Index i = rows.begin + k + 1;
                const Real* xr = x.data() + i * stride_;
                const Real* up = xr - stride_;
                const Real* dn = xr + stride_;
                Real* yr = y.data() + i * stride_;
                for (Index j = 1; j <= n_; ++j) {
                    yr[j] = 4.0 * xr[j] - xr[j - 1] - xr[j + 1] - up[j] - dn[j];
                }
            }
        });
    }

    void jacobi_sweep(backend::Backend& backend, VectorView x, VectorView out) const override {
        backend.exchange_halo(x, stride_, n_);
        const Range rows = backend.local_rows(n_);
        const Real* b = rhs_.data();
        backend.parallel_for(rows.size(), [&](Range chunk) {
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                const Index i = rows.begin + k + 1;
                const Real* xr = x.data() + i * stride_;
                const Real* up = xr - stride_;
                const Real* dn = xr + stride_;
                const Real* br = b + i * stride_;
                Real* o = out.data() + i * stride_;
                for (Index j = 1; j <= n_; ++j) {
                    o[j] = 0.25 * (br[j] + xr[j - 1] + xr[j + 1] + up[j] + dn[j]);
                }
            }
        });
    }

    /// Natural ordering relaxation, exact on every backend.
    ///
    /// The recurrence is sequentially dependent, so there is no partition of
    /// the unknowns that preserves it. Rather than silently substitute a
    /// reordering, this runs the sweep in strict index order: on a shared memory
    /// backend that means a plain serial loop, and on a distributed backend it
    /// means each rank sweeps its own rows only after receiving the boundary
    /// row of its predecessor, which is exactly the natural ordering and
    /// exactly as unparallel as the mathematics says it is. Measuring that cost
    /// is one of the results the report reports.
    void relaxation_sweep(backend::Backend& backend, VectorView x, Real relaxation,
                          Sweep direction) const override {
        require(relaxation > 0.0 && relaxation < 2.0,
                "relaxation factor must lie in (0, 2) for convergence on an SPD system");
        backend.exchange_halo(x, stride_, n_);
        const Range rows = backend.local_rows(n_);
        const Real* b = rhs_.data();
        const Real one_minus = 1.0 - relaxation;
        const Real quarter = 0.25 * relaxation;

        auto sweep_row = [&](Index i) {
            Real* xr = x.data() + i * stride_;
            const Real* up = xr - stride_;
            const Real* dn = xr + stride_;
            const Real* br = b + i * stride_;
            if (direction == Sweep::Forward) {
                for (Index j = 1; j <= n_; ++j) {
                    xr[j] = one_minus * xr[j] +
                            quarter * (br[j] + xr[j - 1] + xr[j + 1] + up[j] + dn[j]);
                }
            } else {
                for (Index j = n_; j >= 1; --j) {
                    xr[j] = one_minus * xr[j] +
                            quarter * (br[j] + xr[j - 1] + xr[j + 1] + up[j] + dn[j]);
                }
            }
        };

        const bool forward = direction == Sweep::Forward;
        backend.run_ordered(
            [&] {
                if (forward) {
                    for (Index i = rows.begin + 1; i <= rows.end; ++i) sweep_row(i);
                } else {
                    for (Index i = rows.end; i >= rows.begin + 1; --i) sweep_row(i);
                }
            },
            forward, x, stride_, n_);
    }

    /// Red black half sweep: fully parallel, and independent of the worker
    /// count because a cell's colour depends only on its coordinates.
    void coloured_sweep(backend::Backend& backend, VectorView x, Real relaxation,
                        Colour colour) const override {
        require(relaxation > 0.0 && relaxation < 2.0,
                "relaxation factor must lie in (0, 2) for convergence on an SPD system");
        backend.exchange_halo(x, stride_, n_);
        const Range rows = backend.local_rows(n_);
        const Real* b = rhs_.data();
        const Real one_minus = 1.0 - relaxation;
        const Real quarter = 0.25 * relaxation;
        const Index parity = (colour == Colour::Red) ? 0 : 1;

        backend.parallel_for(rows.size(), [&](Range chunk) {
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                const Index i = rows.begin + k + 1;
                Real* xr = x.data() + i * stride_;
                const Real* up = xr - stride_;
                const Real* dn = xr + stride_;
                const Real* br = b + i * stride_;
                // Start at the first column of this colour in this row.
                const Index first = 1 + ((i + 1 + parity) & 1);
                for (Index j = first; j <= n_; j += 2) {
                    xr[j] = one_minus * xr[j] +
                            quarter * (br[j] + xr[j - 1] + xr[j + 1] + up[j] + dn[j]);
                }
            }
        });
    }

    /// Line relaxation. The natural block of the five point stencil is a grid
    /// line, whose diagonal block is tridiagonal and is solved exactly by the
    /// Thomas algorithm. With Jacobi coupling every line reads the previous
    /// iterate and all lines are independent, so the sweep is fully parallel;
    /// with Gauss Seidel coupling lines are visited in ascending order.
    ///
    /// \throws InvalidArgument if \p block_count is not the grid line count.
    void block_sweep(backend::Backend& backend, VectorView x, Index block_count,
                     bool jacobi_coupling) const override {
        require(block_count == n_,
                "Poisson2D solves one grid line per block, so block_count must equal the "
                "number of grid lines returned by natural_block_count()");
        backend.exchange_halo(x, stride_, n_);
        const Range rows = backend.local_rows(n_);
        const Real* b = rhs_.data();

        if (jacobi_coupling) {
            // Lines are independent: each reads the neighbouring lines of the
            // previous iterate, so they can be solved concurrently. The
            // previous iterate must be preserved, hence the snapshot.
            Vector previous(x.begin(), x.end());
            backend.parallel_for(rows.size(), [&](Range chunk) {
                Vector scratch(static_cast<std::size_t>(3 * n_));
                for (Index k = chunk.begin; k < chunk.end; ++k) {
                    const Index i = rows.begin + k + 1;
                    solve_line(previous.data(), b, x.data(), i, scratch);
                }
            });
        } else {
            backend.run_ordered(
                [&] {
                    Vector scratch(static_cast<std::size_t>(3 * n_));
                    for (Index i = rows.begin + 1; i <= rows.end; ++i) {
                        solve_line(x.data(), b, x.data(), i, scratch);
                    }
                },
                true, x, stride_, n_);
        }
    }

    Real residual(backend::Backend& backend, VectorView x, VectorView r) const override {
        backend.exchange_halo(x, stride_, n_);
        const Range rows = backend.local_rows(n_);
        const Real* b = rhs_.data();
        backend.parallel_for(rows.size(), [&](Range chunk) {
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                const Index i = rows.begin + k + 1;
                const Real* xr = x.data() + i * stride_;
                const Real* up = xr - stride_;
                const Real* dn = xr + stride_;
                const Real* br = b + i * stride_;
                Real* rr = r.data() + i * stride_;
                for (Index j = 1; j <= n_; ++j) {
                    rr[j] = br[j] - (4.0 * xr[j] - xr[j - 1] - xr[j + 1] - up[j] - dn[j]);
                }
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
                const Index i = rows.begin + k + 1;
                const Real* xr = x.data() + i * stride_;
                const Real* yr = y.data() + i * stride_;
                Real row_sum = 0.0;
                for (Index j = 1; j <= n_; ++j) row_sum += xr[j] * yr[j];
                partial += row_sum;
            }
            return partial;
        });
    }

    void axpy(backend::Backend& backend, Real alpha, ConstVectorView x,
              VectorView y) const override {
        const Range rows = backend.local_rows(n_);
        backend.parallel_for(rows.size(), [&](Range chunk) {
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                const Index i = rows.begin + k + 1;
                const Real* xr = x.data() + i * stride_;
                Real* yr = y.data() + i * stride_;
                for (Index j = 1; j <= n_; ++j) yr[j] += alpha * xr[j];
            }
        });
    }

    void xpby(backend::Backend& backend, ConstVectorView x, Real beta,
              VectorView y) const override {
        const Range rows = backend.local_rows(n_);
        backend.parallel_for(rows.size(), [&](Range chunk) {
            for (Index k = chunk.begin; k < chunk.end; ++k) {
                const Index i = rows.begin + k + 1;
                const Real* xr = x.data() + i * stride_;
                Real* yr = y.data() + i * stride_;
                for (Index j = 1; j <= n_; ++j) yr[j] = xr[j] + beta * yr[j];
            }
        });
    }

    [[nodiscard]] Real rhs_norm(backend::Backend& backend) const override {
        return norm(backend, rhs_);
    }

    /// The interior rows a rank owns occupy one contiguous run of the padded
    /// array, from row rows.begin + 1 to row rows.end inclusive, so the gather
    /// is a single flat range and needs no knowledge of the padding.
    void synchronise(backend::Backend& backend, VectorView x) const override {
        const Range rows = backend.local_rows(n_);
        backend.gather_rows(x, Range{(rows.begin + 1) * stride_, (rows.end + 1) * stride_});
    }

    /// Linear index of interior or boundary cell (i, j).
    [[nodiscard]] Index at(Index i, Index j) const noexcept { return i * stride_ + j; }

   private:
    /// Solve the tridiagonal system for grid line \p i by the Thomas algorithm.
    ///
    /// The line's diagonal block is tridiagonal with 4 on the diagonal and -1
    /// off it; the coupling to lines i-1 and i+1 is moved to the right hand
    /// side, read from \p source so that the caller controls whether that is
    /// the previous iterate (Jacobi coupling) or the current one (Gauss Seidel
    /// coupling). The matrix is diagonally dominant, so no pivoting is needed
    /// and the recurrence is stable.
    void solve_line(const Real* source, const Real* b, Real* destination, Index i,
                    Vector& scratch) const {
        Real* c_prime = scratch.data();
        Real* d_prime = scratch.data() + n_;
        const Real* sr = source + i * stride_;
        const Real* up = source + (i - 1) * stride_;
        const Real* dn = source + (i + 1) * stride_;
        const Real* br = b + i * stride_;
        (void)sr;

        // Forward elimination. a = c = -1, diagonal = 4.
        c_prime[0] = -1.0 / 4.0;
        d_prime[0] = (br[1] + up[1] + dn[1]) / 4.0;
        for (Index j = 1; j < n_; ++j) {
            const Real denominator = 4.0 - (-1.0) * c_prime[j - 1];
            c_prime[j] = -1.0 / denominator;
            const Real rhs_j = br[j + 1] + up[j + 1] + dn[j + 1];
            d_prime[j] = (rhs_j - (-1.0) * d_prime[j - 1]) / denominator;
        }

        // Back substitution straight into the destination row.
        Real* dr = destination + i * stride_;
        dr[n_] = d_prime[n_ - 1];
        for (Index j = n_ - 2; j >= 0; --j) {
            dr[j + 1] = d_prime[j] - c_prime[j] * dr[j + 2];
        }
    }

    Index n_;
    Index stride_;
    PoissonRhs kind_;
    std::uint64_t seed_;
    Vector rhs_;
    Vector exact_;
    PoissonTheory theory_;
};

}  // namespace pnl::problems
