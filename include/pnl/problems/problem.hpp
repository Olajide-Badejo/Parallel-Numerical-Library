#pragma once

/// \file problem.hpp
/// The interface every solver in the zoo is written against.
///
/// The nine solvers of Section 8.2 are implemented exactly once, over these
/// primitives, and therefore run over every backend without a line of solver
/// code knowing which backend it is on. A problem supplies the sweeps; a
/// backend supplies the parallelism; a solver supplies the algebra.
///
/// A problem owns its storage layout. The 2D Poisson problem keeps a padded
/// grid with a boundary ring so that stencil access is branch free, while the
/// dense problem uses a plain vector. Solvers never index into state directly:
/// they allocate through make_state() and reduce through dot() and norms, so
/// the halo never leaks into an inner product.

#include <pnl/backend/backend.hpp>
#include <pnl/core/diagnostics.hpp>
#include <pnl/core/types.hpp>

#include <cmath>
#include <string>

namespace pnl::problems {

using backend::Backend;

/// Direction of a Gauss Seidel or SOR sweep.
enum class Sweep {
    /// Ascending index order: the splitting M = D + L.
    Forward,
    /// Descending index order: the splitting M = D + U.
    Backward,
};

/// Colour of a red black half sweep.
///
/// Red black ordering splits the unknowns of the five point stencil into two
/// sets, each of which has no intra set coupling, so a half sweep over one
/// colour is fully parallel. See Saad, "Iterative Methods for Sparse Linear
/// Systems", 2nd ed., SIAM 2003, chapter 12.
enum class Colour {
    Red,
    Black,
};

/// A linear system A x = b together with the sweeps the splitting solvers need.
class Problem {
 public:
    Problem() = default;
    Problem(const Problem&) = delete;
    Problem& operator=(const Problem&) = delete;
    Problem(Problem&&) = delete;
    Problem& operator=(Problem&&) = delete;
    virtual ~Problem() = default;

    /// Stable identifier for result rows.
    [[nodiscard]] virtual std::string name() const = 0;

    /// Number of actual unknowns. Norms and inner products count these only.
    [[nodiscard]] virtual Index unknown_count() const noexcept = 0;

    /// Length of the storage a solver must allocate, which for a padded grid
    /// exceeds unknown_count() by the boundary ring.
    [[nodiscard]] virtual Index state_size() const noexcept = 0;

    /// Allocate correctly sized and correctly initialised state, including any
    /// boundary values the problem requires.
    [[nodiscard]] virtual Vector make_state() const = 0;

    /// The right hand side, in the same layout as the state.
    [[nodiscard]] virtual ConstVectorView rhs() const noexcept = 0;

    /// y = A x.
    ///
    /// \p x is mutable because on a distributed backend the operator must
    /// refresh the halo rows of \p x before it can be applied. The owned
    /// entries of \p x are never modified.
    virtual void apply(Backend& backend, VectorView x, VectorView y) const = 0;

    /// One Jacobi update: out = D^{-1} (b - (A - D) x).
    ///
    /// Reads only \p x and writes only \p out, so it is fully parallel and its
    /// result does not depend on how the range was partitioned.
    ///
    /// Distributed contract. A rank writes only the entries of \p out it owns,
    /// so on return \p out is stale everywhere else. The solver driver
    /// alternates the two buffers rather than copying one over the other, which
    /// means \p out will be the next call's \p x with those stale entries still
    /// in it. An implementation must therefore make \p x consistent for
    /// everything it is about to read, at entry, on every call, and not rely on
    /// the caller having left the buffer complete. Every implementation here
    /// opens with the exchange that does it, and residual() and apply() do the
    /// same, so no consumer of the iterate ever sees a stale entry.
    virtual void jacobi_sweep(Backend& backend, VectorView x, VectorView out) const = 0;

    /// One in place relaxation sweep in natural ordering.
    ///
    /// With \p relaxation equal to one this is Gauss Seidel; otherwise it is
    /// SOR with that factor. The sweep is sequentially dependent by definition,
    /// which is the whole point of the method, so implementations must preserve
    /// exact natural ordering semantics on every backend even where that costs
    /// all of the parallelism. The report measures that cost rather than hiding
    /// it behind a reordering the caller did not ask for.
    virtual void relaxation_sweep(Backend& backend,
                                  VectorView x,
                                  Real relaxation,
                                  Sweep direction) const = 0;

    /// One half sweep over a single colour, in place and fully parallel.
    ///
    /// \throws InvalidArgument if the problem has no red black colouring.
    virtual void coloured_sweep(Backend& backend,
                                VectorView x,
                                Real relaxation,
                                Colour colour) const = 0;

    /// True when coloured_sweep is available.
    [[nodiscard]] virtual bool supports_colouring() const noexcept { return false; }

    /// One block relaxation sweep over \p block_count blocks.
    ///
    /// Each block solves its diagonal sub system exactly by a direct method
    /// matched to that block's structure: dense LU with partial pivoting for
    /// the dense problem, and the Thomas algorithm for the 2D Poisson problem,
    /// whose natural block is a single grid line and whose diagonal block is
    /// therefore tridiagonal. Coupling between blocks is lagged. When
    /// \p jacobi_coupling is true every block reads the previous iterate (block
    /// Jacobi); otherwise blocks are visited in ascending order and read
    /// already updated neighbours (block Gauss Seidel).
    ///
    /// The block count is a solver parameter, never the worker count. That is
    /// what keeps the iterates of the block methods independent of how many
    /// workers happen to be running, so the equivalence suite can compare them
    /// bit for bit across backends.
    ///
    /// \throws InvalidArgument if \p block_count is not a block count this
    ///         problem can decompose into exactly solvable diagonal blocks.
    virtual void block_sweep(Backend& backend,
                             VectorView x,
                             Index block_count,
                             bool jacobi_coupling) const = 0;

    /// The block count whose diagonal blocks this problem can solve exactly.
    /// The solver zoo passes this to block_sweep by default: the number of grid
    /// lines for 2D Poisson, and a configured group count for dense systems.
    [[nodiscard]] virtual Index natural_block_count() const noexcept = 0;

    /// r = b - A x, returning the Euclidean norm of r. See apply() for why
    /// \p x is mutable.
    virtual Real residual(Backend& backend, VectorView x, VectorView r) const = 0;

    /// Euclidean inner product over the unknowns only.
    [[nodiscard]] virtual Real dot(Backend& backend,
                                   ConstVectorView x,
                                   ConstVectorView y) const = 0;

    /// Euclidean norm over the unknowns only.
    [[nodiscard]] Real norm(Backend& backend, ConstVectorView x) const {
        return std::sqrt(dot(backend, x, x));
    }

    /// y = y + alpha * x over the unknowns.
    virtual void axpy(Backend& backend, Real alpha, ConstVectorView x, VectorView y) const = 0;

    /// y = x + beta * y over the unknowns.
    virtual void xpby(Backend& backend, ConstVectorView x, Real beta, VectorView y) const = 0;

    /// Norm of the right hand side, cached by implementations because every
    /// relative residual test divides by it.
    [[nodiscard]] virtual Real rhs_norm(Backend& backend) const = 0;

    /// Make a distributed iterate globally complete.
    ///
    /// During the iteration a rank only needs its own rows plus a halo, so
    /// nothing gathers the whole vector and nothing should: gathering per sweep
    /// would dominate the communication cost and would measure the wrong thing.
    /// The returned solution, on the other hand, has to be complete on every
    /// rank, so solvers call this once at the end.
    ///
    /// A no operation for shared memory backends.
    virtual void synchronise(Backend& backend, VectorView x) const = 0;

    /// Bytes moved per unknown per **pass** over the arrays of a Jacobi sweep,
    /// counted from the implementation rather than estimated.
    ///
    /// A pass, not an iteration. Diagnostics::passes says how many passes over
    /// memory one iteration of a given method makes, so the traffic of an
    /// iteration is this figure times that count. The two differ for the red
    /// black methods, which do one sweep of work in two passes.
    ///
    /// This is the conservative count: every array the pass touches is charged
    /// once, a read for a read and a write for a write, and nothing is charged
    /// for the line a store has to fetch before it can modify it. The count
    /// with that charge included is dram_bytes_per_unknown_per_sweep().
    ///
    /// Section 8.3 requires this number to be stated, because the device
    /// normalised efficiency comparison divides achieved bandwidth by it.
    [[nodiscard]] virtual Real bytes_per_unknown_per_sweep() const noexcept = 0;

    /// The same pass counted with read for ownership charged, which is the
    /// traffic that actually crosses the memory bus on a write allocate cache.
    ///
    /// A store to a line the cache does not hold fetches that line from memory
    /// before modifying it, so an array a pass writes without having read it
    /// first costs a read as well as a write. An array the pass reads and then
    /// writes in place pays nothing extra, because the read brought the line in
    /// already. Each implementation writes out which of its arrays is which.
    ///
    /// Neither figure replaces the other and both appear in every result row,
    /// which is ground rule 9: no published number changes underneath a reader.
    /// Section 4.2 leaves open which of the two the report should divide by.
    /// The non temporal triad of backend/stream_probe.hpp is the instrument
    /// that settles it, the rule that reads the instrument is pre registered in
    /// benchmarks/sweep_matrix.yaml, and phase A8b applies that rule to the
    /// publication session's measurement.
    [[nodiscard]] virtual Real dram_bytes_per_unknown_per_sweep() const noexcept = 0;

    /// True when the operator is symmetric positive definite, which conjugate
    /// gradient requires.
    [[nodiscard]] virtual bool is_symmetric_positive_definite() const noexcept = 0;

    /// An upper bound on the spectral radius, from Gershgorin's theorem: every
    /// eigenvalue lies in some disc centred on a diagonal entry with radius the
    /// sum of the other magnitudes in that row.
    ///
    /// Richardson iteration needs this. A power iteration estimate would
    /// approach the largest eigenvalue from below and so could return a step
    /// outside the convergence interval, whereas a Gershgorin bound is an
    /// overestimate by construction and therefore always safe.
    [[nodiscard]] virtual Real gershgorin_bound() const noexcept = 0;
};

}  // namespace pnl::problems
