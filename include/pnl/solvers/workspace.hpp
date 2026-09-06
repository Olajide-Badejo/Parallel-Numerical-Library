// SPDX-License-Identifier: MIT
#pragma once

/// \file workspace.hpp
/// The full size vectors a solve needs, owned by the caller instead of by the
/// call.
///
/// Why this exists. The driver times `solve()`, and `solve` used to allocate
/// every vector it needed on entry and free them on return. On the 4095 squared
/// grid a state vector is 128 MiB, so a Jacobi repetition mapped and unmapped
/// 384 MiB and a conjugate gradient repetition 512 MiB. Fresh anonymous
/// mappings arrive untouched, so every timed repetition paid a first touch page
/// fault on every page of them, and the untimed warm up before the loop bought
/// nothing at all because the next repetition's memory was not the memory the
/// warm up had faulted in. Hoisting the allocation out of the timed region is
/// the whole of the repair; it is measurement finding MEAS-10 and phase A5 of
/// the version 2 specification.
///
/// What it is not. It is not a placement policy. The vectors are first touched
/// by whichever thread constructs the workspace, and on the single socket
/// target machine that is the right answer because there is no NUMA distance to
/// get wrong. Decision 11 in docs/DESIGN_DECISIONS.md records why parallel
/// first touch is future work rather than part of this change.

#include <pnl/core/diagnostics.hpp>
#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>
#include <pnl/problems/problem.hpp>

#include <cstddef>
#include <vector>

namespace pnl::solvers {

using problems::Problem;

/// The full size vectors of one solve, allocated once and reused.
///
/// Slot 0 is always the iterate, on every solver, and is what a solve reports
/// its solution as a view of. The rest are the solver's own business: the
/// shared stationary driver uses slot 1 as the second buffer it alternates with
/// and slot 2 as the residual, while Richardson and conjugate gradient document
/// their own layouts. Nothing here knows which is which, which is what keeps
/// the solvers ignorant of each other.
///
/// The vectors are stored as one flat allocation rather than a vector of
/// vectors so that the whole workspace is one mapping and the slots are a fixed
/// stride apart.
class SolverWorkspace {
 public:
    SolverWorkspace() = default;

    /// Allocate \p vectors buffers of \p problem's state size, each holding the
    /// problem's initial state.
    ///
    /// \throws InvalidArgument if \p vectors is not positive.
    [[nodiscard]] static SolverWorkspace for_problem(const Problem& problem, Index vectors) {
        require(vectors > 0, "a solver workspace needs at least one vector");
        SolverWorkspace workspace;
        workspace.state_size_ = problem.state_size();
        workspace.count_ = vectors;
        workspace.storage_.assign(static_cast<std::size_t>(vectors * workspace.state_size_), 0.0);
        workspace.reset(problem);
        return workspace;
    }

    /// Vectors this workspace holds.
    [[nodiscard]] Index vector_count() const noexcept { return count_; }

    /// Length of each of them.
    [[nodiscard]] Index state_size() const noexcept { return state_size_; }

    /// Slot \p index.
    ///
    /// \throws InvalidArgument if \p index is outside [0, vector_count()).
    [[nodiscard]] VectorView vector(Index index) {
        require(index >= 0 && index < count_, "solver workspace slot is out of range");
        return VectorView(storage_.data() + index * state_size_,
                          static_cast<std::size_t>(state_size_));
    }

    /// The iterate, which is slot 0 for every solver.
    [[nodiscard]] VectorView iterate() { return vector(0); }

    /// Put every slot back into \p problem's initial state.
    ///
    /// This is what a repetition loop calls between repetitions, outside the
    /// timed region. It is a write over memory this process already owns, not a
    /// mapping, so it faults nothing in after the first time.
    ///
    /// Every slot, not only the iterate. A sweep writes the interior of its
    /// output and leaves the boundary ring alone, so a buffer that has held an
    /// iterate is only safe to reuse because its ring is still the problem's;
    /// resetting all of them makes that true by construction rather than by
    /// argument.
    void reset(const Problem& problem) {
        for (Index index = 0; index < count_; ++index) problem.initial_state(vector(index));
    }

 private:
    Index state_size_ = 0;
    Index count_ = 0;
    Vector storage_;
};

/// What a solve driven through a workspace reports.
///
/// The iterate is not in here. It stays in the workspace, as a view, because
/// handing back an owning vector would mean allocating one per call and that is
/// the whole thing this phase removes. Solver::solve's three argument overload
/// is the one that returns an owning SolveResult, for callers that are not
/// timing anything.
struct SolveReport {
    /// The iterate the run finished with: a view of the workspace's slot 0.
    VectorView solution;

    Diagnostics diagnostics;

    /// Relative residual after each iteration, index 0 being the initial guess.
    /// Empty unless SolverOptions::record_history asked for it, which no timed
    /// path does.
    std::vector<Real> residual_history;
};

}  // namespace pnl::solvers
