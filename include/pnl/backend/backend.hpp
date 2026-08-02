#pragma once

/// \file backend.hpp
/// The one interface every execution model implements.
///
/// Design note, because this is the decision the whole comparison rests on.
///
/// The callbacks are chunk level, not element level: a body receives a Range
/// and loops over it itself. Two consequences follow, both deliberate. First,
/// the std::function indirection is paid once per chunk rather than once per
/// element, so it is O(workers) per sweep and does not contaminate the timings
/// the study exists to measure. Second, the innermost loop stays a plain loop
/// over contiguous memory that the compiler can still vectorise, so each
/// backend measures its own dispatch and synchronisation cost rather than a
/// penalty this abstraction imposed.
///
/// The interface is the minimal common denominator of the models it spans, as
/// argued in McCool, Robison and Reinders, "Structured Parallel Programming",
/// Morgan Kaufmann 2012, chapters 3 and 5: a data parallel map, a reduction,
/// and a barrier. Anything richer would privilege one model. Each
/// implementation stays idiomatic underneath: OpenMP uses its own scheduler,
/// pthreads uses a condition variable pool with explicit affinity, the jthread
/// pool uses std::barrier and std::stop_token, MPI uses remainder aware block
/// decomposition.
///
/// Backends never leak their model's types through this interface: no
/// MPI_Comm, no omp_ types, no cudaStream_t appears in any signature here.

#include <pnl/core/error.hpp>
#include <pnl/core/types.hpp>

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace pnl::backend {

/// How a parallel_for distributes its chunks.
enum class Schedule {
    /// One contiguous block per worker, decided up front. The right choice for
    /// the uniform stencil sweeps that dominate this library.
    Static,
    /// Chunks handed out on demand. Measured once and reported so the cost of
    /// dynamic scheduling on uniform work is a number rather than folklore.
    Dynamic,
};

/// How a reduction combines partial results.
///
/// Floating point addition is not associative, so a reduction's answer depends
/// on the order in which partials are combined. Rather than hide that behind a
/// tolerance, the library makes it a choice.
enum class ReductionMode {
    /// Partials are combined in a fixed chunk order that depends only on the
    /// problem size, never on the worker count or the arrival order of threads.
    /// Every backend therefore produces bit identical reductions, which is what
    /// lets the equivalence suite assert exact equality instead of "close
    /// enough". This is the default.
    Deterministic,
    /// The model's native reduction: an OpenMP reduction clause, an atomic
    /// accumulation, MPI_Allreduce. Faster, and not reproducible across worker
    /// counts. Available so the sweep can price determinism honestly.
    Native,
};

/// Thread to core binding policy, swept as a variable in the scaling study
/// because this is a heterogeneous CPU and the answer differs per policy.
enum class Pinning {
    /// No affinity set; the scheduler places threads.
    None,
    /// Workers bound to logical CPUs 0, 1, 2, ... in order.
    Compact,
    /// Workers bound one per physical core, skipping the sibling hyperthread.
    Scatter,
    /// Workers bound to performance cores only, as classified at runtime.
    PerformanceCores,
    /// Workers bound to efficiency cores only, as classified at runtime.
    EfficiencyCores,
};

[[nodiscard]] constexpr std::string_view to_string(Pinning pinning) noexcept {
    switch (pinning) {
        case Pinning::None: return "none";
        case Pinning::Compact: return "compact";
        case Pinning::Scatter: return "scatter";
        case Pinning::PerformanceCores: return "pcore";
        case Pinning::EfficiencyCores: return "ecore";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(ReductionMode mode) noexcept {
    return mode == ReductionMode::Deterministic ? "deterministic" : "native";
}

/// Body of a parallel_for: called once per chunk with the chunk's range.
using RangeBody = std::function<void(Range)>;

/// Body of a reduce: returns this chunk's partial result.
using RangeReducer = std::function<Real(Range)>;

/// Configuration handed to a backend factory.
struct Config {
    /// Number of workers: threads for the thread backends, ranks for MPI,
    /// ranks times threads for hybrid. Zero means "ask the system".
    int workers = 0;
    /// Threads per rank; only the hybrid backend reads this.
    int threads_per_rank = 1;
    Pinning pinning = Pinning::None;
    ReductionMode reduction = ReductionMode::Deterministic;
    Schedule schedule = Schedule::Static;
    /// Chunks per worker for Schedule::Dynamic. Ignored when Static.
    int chunks_per_worker = 8;
};

/// The execution backend interface.
class Backend {
   public:
    Backend() = default;
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    Backend(Backend&&) = delete;
    Backend& operator=(Backend&&) = delete;
    virtual ~Backend() = default;

    /// Stable identifier used in result rows and figures.
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// Workers actually in use, which may differ from what was requested if the
    /// system refused. Result rows record this value, not the request.
    [[nodiscard]] virtual int worker_count() const noexcept = 0;

    /// Apply \p body to a partition of [0, n).
    ///
    /// The partition covers the range exactly once with contiguous chunks. The
    /// body must be safe to run concurrently on disjoint chunks; the interface
    /// makes no other ordering promise.
    ///
    /// \throws BackendFailure if the execution model reports an error.
    virtual void parallel_for(Index n, const RangeBody& body) = 0;

    /// Reduce \p reducer over a partition of [0, n) starting from \p init.
    ///
    /// Combination is by addition. Under ReductionMode::Deterministic the
    /// partials are summed in ascending chunk index regardless of which worker
    /// produced them or when, so repeated runs and different worker counts
    /// agree bit for bit.
    ///
    /// \throws BackendFailure if the execution model reports an error.
    [[nodiscard]] virtual Real reduce(Index n, Real init, const RangeReducer& reducer) = 0;

    /// Synchronise all workers. A no operation for the serial backend; a real
    /// barrier for the thread pools; MPI_Barrier for the distributed backends.
    virtual void barrier() = 0;

    /// True on the rank that owns the terminal. Only this rank prints, per
    /// Section 9 of the specification.
    [[nodiscard]] virtual bool is_root() const noexcept { return true; }

    /// Number of distributed ranks. One for every shared memory backend.
    [[nodiscard]] virtual int rank_count() const noexcept { return 1; }

    /// This process's rank. Zero for every shared memory backend.
    [[nodiscard]] virtual int rank() const noexcept { return 0; }

    /// The rows of a distributed problem this process owns. For shared memory
    /// backends this is the whole range, which is what makes solver code
    /// identical across backends.
    [[nodiscard]] virtual Range local_rows(Index total_rows) const noexcept {
        return Range{0, total_rows};
    }

    /// Exchange halo rows with neighbours for a row decomposed grid.
    ///
    /// A no operation for shared memory backends, where neighbours are simply
    /// readable. Implemented by the MPI backends. \p row_stride is the number of
    /// values per grid row.
    virtual void exchange_halo(VectorView /*grid*/, Index /*row_stride*/,
                               Index /*total_rows*/) {}

    /// Run \p local_work under global sequential ordering across ranks.
    ///
    /// Natural ordering Gauss Seidel and SOR are sequentially dependent by
    /// definition: unknown k reads the already updated unknown k-1. Preserving
    /// that across a row decomposition means rank r may not start until rank
    /// r-1 has finished and handed over its boundary row, which is the
    /// classical pipelined Gauss Seidel. Shared memory backends simply call
    /// \p local_work, since their rows are already in one address space and
    /// already in order.
    ///
    /// Doing this rather than quietly substituting a red black reordering is
    /// what lets every backend produce bit identical iterates for these
    /// methods, and it makes the resulting lack of parallel speedup an honest
    /// measurement instead of a hidden change of algorithm.
    ///
    /// \param forward true to order ranks 0, 1, 2, ...; false to reverse them,
    ///        as a backward sweep requires.
    virtual void run_ordered(const std::function<void()>& local_work, bool /*forward*/) {
        local_work();
    }

    /// Configuration this backend was built with, for the result row.
    [[nodiscard]] virtual const Config& config() const noexcept = 0;
};

/// Which backends this build actually contains. Populated by CMake through
/// compile definitions, so the CLI can refuse an unavailable backend with a
/// clear message rather than a link error.
[[nodiscard]] std::vector<std::string> available_backends();

/// Construct a backend by name.
///
/// Recognised names: "serial", "openmp", "pthreads", "jthread", "mpi",
/// "hybrid", "cuda".
///
/// \throws InvalidArgument if the name is unknown or the backend was not
///         compiled into this build.
/// \throws BackendFailure if construction fails, for example a thread that will
///         not start or a pinning request the operating system refused.
[[nodiscard]] std::unique_ptr<Backend> make_backend(std::string_view name, const Config& config);

/// Number of logical CPUs visible to this process, respecting any affinity mask
/// already applied. Used to size default worker counts and to bound the sweep.
[[nodiscard]] int available_logical_cpus();

}  // namespace pnl::backend
