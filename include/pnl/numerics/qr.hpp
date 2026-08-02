#pragma once

/// \file qr.hpp
/// Householder QR factorisation.
///
/// Method: successive Householder reflections annihilate the sub diagonal of
/// each column in turn, giving A = QR with Q orthogonal. Reflections are used
/// rather than Gram Schmidt because they are unconditionally backward stable;
/// classical Gram Schmidt loses orthogonality in proportion to the condition
/// number, and even the modified variant loses it linearly.
///
/// The reflectors are stored in the annihilated part of the matrix in the usual
/// compact form, so the factorisation needs no extra storage beyond the vector
/// of scaling factors.
///
/// Reference: Golub and Van Loan, "Matrix Computations", 4th ed., Johns Hopkins
/// 2013, section 5.2; Trefethen and Bau, "Numerical Linear Algebra", SIAM 1997,
/// lectures 10 and 16.

#include <pnl/core/diagnostics.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>
#include <pnl/numerics/lu.hpp>

#include <cmath>
#include <vector>

namespace pnl::numerics {

/// Row major dense rectangular matrix.
class Matrix {
   public:
    Matrix(Index rows, Index cols)
        : rows_(rows), cols_(cols), data_(static_cast<std::size_t>(rows * cols), 0.0) {
        require(rows >= 0 && cols >= 0, "Matrix dimensions must be non negative");
    }

    [[nodiscard]] Index rows() const noexcept { return rows_; }

    [[nodiscard]] Index cols() const noexcept { return cols_; }

    [[nodiscard]] Real& operator()(Index i, Index j) noexcept {
        return data_[static_cast<std::size_t>(i * cols_ + j)];
    }

    [[nodiscard]] Real operator()(Index i, Index j) const noexcept {
        return data_[static_cast<std::size_t>(i * cols_ + j)];
    }

   private:
    Index rows_;
    Index cols_;
    Vector data_;
};

/// Householder QR of an m by n matrix with m at least n.
class QrFactorisation {
   public:
    /// \throws InvalidArgument if the matrix has fewer rows than columns.
    explicit QrFactorisation(Matrix matrix) : qr_(std::move(matrix)) {
        const Index m = qr_.rows();
        const Index n = qr_.cols();
        require(m >= n, "Householder QR needs at least as many rows as columns");
        beta_.assign(static_cast<std::size_t>(n), 0.0);
        // Indexed by column, not appended to: a column that needs no reflector
        // still occupies its slot, so the two arrays stay in step.
        v0_.assign(static_cast<std::size_t>(n), 0.0);

        for (Index k = 0; k < n; ++k) {
            // Norm of the column below and including the diagonal.
            Real norm = 0.0;
            for (Index i = k; i < m; ++i) norm += qr_(i, k) * qr_(i, k);
            norm = std::sqrt(norm);
            if (norm == 0.0) {
                beta_[static_cast<std::size_t>(k)] = 0.0;
                continue;
            }

            // Choose the sign that avoids cancellation in the leading entry.
            const Real alpha = qr_(k, k) >= 0.0 ? -norm : norm;
            const Real v0 = qr_(k, k) - alpha;

            // Store the reflector, normalised so its leading entry is one, and
            // keep the resulting scale factor in beta.
            Real vnorm2 = v0 * v0;
            for (Index i = k + 1; i < m; ++i) vnorm2 += qr_(i, k) * qr_(i, k);
            if (vnorm2 == 0.0) {
                beta_[static_cast<std::size_t>(k)] = 0.0;
                continue;
            }
            beta_[static_cast<std::size_t>(k)] = 2.0 / vnorm2;

            // Apply the reflector to the trailing columns before overwriting.
            for (Index j = k + 1; j < n; ++j) {
                Real dot = v0 * qr_(k, j);
                for (Index i = k + 1; i < m; ++i) dot += qr_(i, k) * qr_(i, j);
                dot *= beta_[static_cast<std::size_t>(k)];
                qr_(k, j) -= dot * v0;
                for (Index i = k + 1; i < m; ++i) qr_(i, j) -= dot * qr_(i, k);
            }

            // The diagonal of R, then the reflector tail in the annihilated part.
            v0_[static_cast<std::size_t>(k)] = v0;
            qr_(k, k) = alpha;
        }
    }

    /// Apply Q transpose to \p b in place.
    void apply_q_transpose(VectorView b) const {
        const Index m = qr_.rows();
        const Index n = qr_.cols();
        require(static_cast<Index>(b.size()) == m, "vector length must match the row count");
        for (Index k = 0; k < n; ++k) {
            const Real beta = beta_[static_cast<std::size_t>(k)];
            if (beta == 0.0) continue;
            const Real v0 = v0_[static_cast<std::size_t>(k)];
            Real dot = v0 * b[static_cast<std::size_t>(k)];
            for (Index i = k + 1; i < m; ++i) dot += qr_(i, k) * b[static_cast<std::size_t>(i)];
            dot *= beta;
            b[static_cast<std::size_t>(k)] -= dot * v0;
            for (Index i = k + 1; i < m; ++i) {
                b[static_cast<std::size_t>(i)] -= dot * qr_(i, k);
            }
        }
    }

    /// Solve the least squares problem min ||A x - b|| in the two norm.
    ///
    /// \throws NumericalFailure if R is singular, meaning A is rank deficient.
    [[nodiscard]] Vector solve_least_squares(ConstVectorView b) const {
        const Index n = qr_.cols();
        Vector work(b.begin(), b.end());
        apply_q_transpose(work);

        Vector x(static_cast<std::size_t>(n), 0.0);
        for (Index k = n - 1; k >= 0; --k) {
            Real sum = work[static_cast<std::size_t>(k)];
            for (Index j = k + 1; j < n; ++j) sum -= qr_(k, j) * x[static_cast<std::size_t>(j)];
            const Real diagonal = qr_(k, k);
            if (std::abs(diagonal) <= 1.0e-300) {
                throw NumericalFailure("rank deficient matrix: R has a zero diagonal at " +
                                       std::to_string(k));
            }
            x[static_cast<std::size_t>(k)] = sum / diagonal;
        }
        return x;
    }

    /// The upper triangular factor R, as an n by n matrix.
    [[nodiscard]] Matrix upper_triangular() const {
        const Index n = qr_.cols();
        Matrix r(n, n);
        for (Index i = 0; i < n; ++i) {
            for (Index j = i; j < n; ++j) r(i, j) = qr_(i, j);
        }
        return r;
    }

   private:
    Matrix qr_;
    Vector beta_;
    Vector v0_;
};

/// Solve min ||A x - b|| by Householder QR.
///
/// \returns the solution with diagnostics whose error_estimate is the relative
///          residual norm, which for an overdetermined system is the genuine
///          least squares residual and not an error.
/// \throws InvalidArgument if the dimensions disagree.
/// \throws NumericalFailure if A is rank deficient.
[[nodiscard]] inline Result<Vector> qr_solve(const Matrix& matrix, ConstVectorView b) {
    require(static_cast<Index>(b.size()) == matrix.rows(),
            "right hand side length must match the row count");
    QrFactorisation factorisation{Matrix(matrix)};
    Vector x = factorisation.solve_least_squares(b);

    Real residual_norm = 0.0;
    Real rhs_norm = 0.0;
    for (Index i = 0; i < matrix.rows(); ++i) {
        Real value = 0.0;
        for (Index j = 0; j < matrix.cols(); ++j) {
            value += matrix(i, j) * x[static_cast<std::size_t>(j)];
        }
        const Real difference = b[static_cast<std::size_t>(i)] - value;
        residual_norm += difference * difference;
        rhs_norm += b[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
    }
    residual_norm = std::sqrt(residual_norm);
    rhs_norm = std::sqrt(rhs_norm);

    Diagnostics diagnostics;
    diagnostics.iterations = 1;
    diagnostics.evaluations = matrix.cols();
    diagnostics.error_estimate = rhs_norm > 0.0 ? residual_norm / rhs_norm : residual_norm;
    diagnostics.converged = true;
    diagnostics.reason = StopReason::Converged;
    return make_result(std::move(x), diagnostics);
}

}  // namespace pnl::numerics
