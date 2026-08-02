#pragma once

/// \file lu.hpp
/// Dense LU factorisation with partial pivoting.
///
/// Method: Doolittle LU with row interchanges, PA = LU, computed in place.
/// Partial pivoting bounds the multipliers by one, which keeps the growth
/// factor at its practical O(n) rather than the 2^{n-1} worst case and makes
/// the factorisation backward stable for every matrix met here.
///
/// Reference: Golub and Van Loan, "Matrix Computations", 4th ed., Johns Hopkins
/// 2013, sections 3.2 and 3.4.

#include <pnl/core/diagnostics.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>

#include <cmath>
#include <span>
#include <vector>

namespace pnl::numerics {

/// Row major dense square matrix, owning its storage.
class DenseMatrix {
   public:
    DenseMatrix() = default;

    explicit DenseMatrix(Index n) : n_(n), data_(static_cast<std::size_t>(n * n), 0.0) {
        require(n >= 0, "DenseMatrix order must be non negative");
    }

    [[nodiscard]] Index order() const noexcept { return n_; }

    [[nodiscard]] Real& operator()(Index i, Index j) noexcept {
        return data_[static_cast<std::size_t>(i * n_ + j)];
    }

    [[nodiscard]] Real operator()(Index i, Index j) const noexcept {
        return data_[static_cast<std::size_t>(i * n_ + j)];
    }

    [[nodiscard]] Real* row(Index i) noexcept { return data_.data() + i * n_; }

    [[nodiscard]] const Real* row(Index i) const noexcept { return data_.data() + i * n_; }

    [[nodiscard]] Vector& storage() noexcept { return data_; }

    [[nodiscard]] const Vector& storage() const noexcept { return data_; }

   private:
    Index n_ = 0;
    Vector data_;
};

/// An LU factorisation with its pivot sequence.
class LuFactorisation {
   public:
    LuFactorisation() = default;

    /// Factorise \p matrix in place into L and U with partial pivoting.
    ///
    /// \throws NumericalFailure if a pivot is numerically zero, which means the
    ///         matrix is singular to working precision.
    explicit LuFactorisation(DenseMatrix matrix) : lu_(std::move(matrix)) {
        const Index n = lu_.order();
        pivots_.resize(static_cast<std::size_t>(n));
        sign_ = 1.0;

        for (Index k = 0; k < n; ++k) {
            // Select the pivot row.
            Index pivot = k;
            Real best = std::abs(lu_(k, k));
            for (Index i = k + 1; i < n; ++i) {
                const Real candidate = std::abs(lu_(i, k));
                if (candidate > best) {
                    best = candidate;
                    pivot = i;
                }
            }
            pivots_[static_cast<std::size_t>(k)] = pivot;

            if (best <= TINY_PIVOT) {
                throw NumericalFailure("singular matrix: pivot " + std::to_string(k) +
                                       " is " + std::to_string(best));
            }

            if (pivot != k) {
                Real* a = lu_.row(k);
                Real* b = lu_.row(pivot);
                for (Index j = 0; j < n; ++j) std::swap(a[j], b[j]);
                sign_ = -sign_;
            }

            const Real inverse_pivot = 1.0 / lu_(k, k);
            for (Index i = k + 1; i < n; ++i) {
                const Real multiplier = lu_(i, k) * inverse_pivot;
                lu_(i, k) = multiplier;
                if (multiplier == 0.0) continue;
                Real* target = lu_.row(i);
                const Real* source = lu_.row(k);
                for (Index j = k + 1; j < n; ++j) target[j] -= multiplier * source[j];
            }
        }
    }

    /// Solve A x = b in place on \p b.
    ///
    /// \throws InvalidArgument if the length of \p b does not match the order.
    void solve_in_place(VectorView b) const {
        const Index n = lu_.order();
        require(static_cast<Index>(b.size()) == n, "right hand side length must match the order");

        // Apply the pivot sequence, then forward substitute through L.
        for (Index k = 0; k < n; ++k) {
            const Index pivot = pivots_[static_cast<std::size_t>(k)];
            if (pivot != k) std::swap(b[static_cast<std::size_t>(k)],
                                      b[static_cast<std::size_t>(pivot)]);
            const Real* row = lu_.row(k);
            Real sum = b[static_cast<std::size_t>(k)];
            for (Index j = 0; j < k; ++j) sum -= row[j] * b[static_cast<std::size_t>(j)];
            b[static_cast<std::size_t>(k)] = sum;
        }

        // Back substitute through U.
        for (Index k = n - 1; k >= 0; --k) {
            const Real* row = lu_.row(k);
            Real sum = b[static_cast<std::size_t>(k)];
            for (Index j = k + 1; j < n; ++j) sum -= row[j] * b[static_cast<std::size_t>(j)];
            b[static_cast<std::size_t>(k)] = sum / row[k];
        }
    }

    /// Determinant, from the product of the diagonal of U and the pivot sign.
    [[nodiscard]] Real determinant() const noexcept {
        Real value = sign_;
        for (Index k = 0; k < lu_.order(); ++k) value *= lu_(k, k);
        return value;
    }

    [[nodiscard]] const DenseMatrix& factors() const noexcept { return lu_; }

    [[nodiscard]] const std::vector<Index>& pivots() const noexcept { return pivots_; }

   private:
    /// Below this a pivot counts as zero. Absolute rather than relative because
    /// the callers here work with matrices scaled to O(1) entries.
    static constexpr Real TINY_PIVOT = 1.0e-300;

    DenseMatrix lu_;
    std::vector<Index> pivots_;
    Real sign_ = 1.0;
};

/// Solve A x = b by LU with partial pivoting.
///
/// \returns the solution together with diagnostics whose error_estimate is the
///          relative residual ||b - A x|| / ||b|| measured after the solve, so
///          the caller always sees how well the factorisation actually did.
/// \throws NumericalFailure if the matrix is singular to working precision.
/// \throws InvalidArgument if the dimensions disagree.
[[nodiscard]] inline Result<Vector> lu_solve(const DenseMatrix& matrix, ConstVectorView b) {
    const Index n = matrix.order();
    require(static_cast<Index>(b.size()) == n, "right hand side length must match the order");

    LuFactorisation factorisation{DenseMatrix(matrix)};
    Vector x(b.begin(), b.end());
    factorisation.solve_in_place(x);

    // Measure the residual rather than assert success.
    Real residual_norm = 0.0;
    Real rhs_norm = 0.0;
    for (Index i = 0; i < n; ++i) {
        Real row_value = 0.0;
        const Real* row = matrix.row(i);
        for (Index j = 0; j < n; ++j) row_value += row[j] * x[static_cast<std::size_t>(j)];
        const Real difference = b[static_cast<std::size_t>(i)] - row_value;
        residual_norm += difference * difference;
        rhs_norm += b[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
    }
    residual_norm = std::sqrt(residual_norm);
    rhs_norm = std::sqrt(rhs_norm);

    Diagnostics diagnostics;
    diagnostics.iterations = 1;
    diagnostics.evaluations = n;
    diagnostics.error_estimate = rhs_norm > 0.0 ? residual_norm / rhs_norm : residual_norm;
    // A direct solve either factorises or throws; convergence here means the
    // computed solution is backward stable at the level a caller can rely on.
    diagnostics.converged = diagnostics.error_estimate < 1.0e-10;
    diagnostics.reason = diagnostics.converged ? StopReason::Converged : StopReason::Stagnated;
    return make_result(std::move(x), diagnostics);
}

/// Solve a tridiagonal system by the Thomas algorithm.
///
/// Method: Gaussian elimination specialised to a tridiagonal matrix, O(n) work
/// and O(n) storage. No pivoting, which is safe precisely when the matrix is
/// diagonally dominant; that is checked rather than assumed.
///
/// \param lower sub diagonal, entries 1 to n-1 used.
/// \param diagonal main diagonal.
/// \param upper super diagonal, entries 0 to n-2 used.
/// \param rhs right hand side, overwritten with the solution.
/// \throws InvalidArgument if the lengths disagree or the matrix is not
///         diagonally dominant, since without that the unpivoted recurrence is
///         not stable.
inline void thomas_solve(ConstVectorView lower, ConstVectorView diagonal, ConstVectorView upper,
                         VectorView rhs) {
    const Index n = static_cast<Index>(diagonal.size());
    require(static_cast<Index>(lower.size()) == n && static_cast<Index>(upper.size()) == n &&
                static_cast<Index>(rhs.size()) == n,
            "thomas_solve needs four spans of equal length");

    for (Index i = 0; i < n; ++i) {
        const Real off = (i > 0 ? std::abs(lower[static_cast<std::size_t>(i)]) : 0.0) +
                         (i + 1 < n ? std::abs(upper[static_cast<std::size_t>(i)]) : 0.0);
        require(std::abs(diagonal[static_cast<std::size_t>(i)]) >= off,
                "thomas_solve requires a diagonally dominant matrix; row " + std::to_string(i) +
                    " is not");
    }

    Vector c_prime(static_cast<std::size_t>(n), 0.0);
    c_prime[0] = upper[0] / diagonal[0];
    rhs[0] = rhs[0] / diagonal[0];
    for (Index i = 1; i < n; ++i) {
        const auto k = static_cast<std::size_t>(i);
        const Real denominator = diagonal[k] - lower[k] * c_prime[k - 1];
        c_prime[k] = (i + 1 < n ? upper[k] : 0.0) / denominator;
        rhs[k] = (rhs[k] - lower[k] * rhs[k - 1]) / denominator;
    }
    for (Index i = n - 2; i >= 0; --i) {
        const auto k = static_cast<std::size_t>(i);
        rhs[k] -= c_prime[k] * rhs[k + 1];
    }
}

}  // namespace pnl::numerics
