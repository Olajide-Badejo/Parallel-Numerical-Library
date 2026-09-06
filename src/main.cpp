// SPDX-License-Identifier: MIT
/// \file main.cpp
/// Command line driver.
///
/// One invocation runs one configuration: a solver, on a backend, on a problem,
/// at a worker count and pinning policy, repeated a few times. It prints a
/// single result row carrying everything Section 7 requires to reproduce it,
/// including the seed and the commit hash, so no number in the report can exist
/// without the run that produced it.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/stream_probe.hpp>
#include <pnl/backend/topology.hpp>
#include <pnl/bench/timed_solve.hpp>
#include <pnl/core/error.hpp>
#include <pnl/problems/dense_generator.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>
#include <pnl/version.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(PNL_WITH_MPI)
#include <mpi.h>
#endif

#if defined(PNL_WITH_CUDA)
#include <pnl/backend/cuda.hpp>
#endif

#ifndef PNL_GIT_COMMIT
#define PNL_GIT_COMMIT "unknown"
#endif

namespace {

using namespace pnl;

struct Options {
    std::string solver = "jacobi";
    std::string backend = "serial";
    std::string problem = "poisson";
    /// Right hand side for the Poisson problem. "rich" is the default because
    /// the single mode source makes conjugate gradient converge in one
    /// iteration at any grid size, which would flatter it against every other
    /// method in the sweep. "sine" is the one with a closed form solution and
    /// is what the discretisation error checks use.
    std::string rhs = "rich";
    Index size = 255;
    int workers = 0;
    int threads_per_rank = 1;
    std::string pinning = "none";
    std::string reduction = "deterministic";
    std::string schedule = "static";
    std::string mode = "solve";
    Index iterations = DEFAULT_MAX_ITERATIONS;
    Real tolerance = DEFAULT_TOLERANCE;
    Real relaxation = 0.0;
    Index blocks = 0;
    Index check_interval = 1;
    int repetitions = 3;
    std::uint64_t seed = 20260802;
    bool progress = false;
    bool header = false;
    bool list = false;
    bool topology = false;
    bool bandwidth = false;
    bool version = false;
    std::string label;
};

[[noreturn]] void usage(int status) {
    std::fprintf(status == 0 ? stdout : stderr,
                 "pnl: run one solver on one backend and print a result row\n"
                 "\n"
                 "  --solver NAME        solver from the registry (default jacobi)\n"
                 "  --backend NAME       serial, openmp, pthreads, jthread, mpi, hybrid\n"
                 "  --problem NAME       poisson, dense_dd, dense_spd\n"
                 "  --rhs NAME           poisson source: rich (default) or sine\n"
                 "  --size N             interior points per side, or dense order\n"
                 "  --workers N          threads or ranks; 0 asks the system\n"
                 "  --threads-per-rank N threads inside each rank, hybrid backend only\n"
                 "  --pinning NAME       none, compact, scatter, pcore, ecore\n"
                 "  --reduction NAME     deterministic, native\n"
                 "  --schedule NAME      static, dynamic\n"
                 "  --mode NAME          solve (to tolerance) or fixed (fixed iterations)\n"
                 "  --iterations N       iteration cap, or the exact count in fixed mode\n"
                 "  --tolerance X        relative residual target (default 1e-8)\n"
                 "  --omega X            relaxation factor; 0 asks for the closed form optimum\n"
                 "  --blocks N           block count; 0 uses the problem's natural choice\n"
                 "  --check-interval N   iterations between residual evaluations\n"
                 "  --reps N             timed repetitions; the median is reported\n"
                 "  --seed N             problem seed, recorded in the row\n"
                 "  --label TEXT         free text copied into the row\n"
                 "  --progress           show an in run progress bar\n"
                 "  --header             print the CSV header and exit\n"
                 "  --list               list solvers and backends and exit\n"
                 "  --topology           probe and describe the CPU topology and exit\n"
                 "  --bandwidth          measure host and device STREAM triad and exit\n"
                 "  --version            print the library version and commit and exit\n"
                 "  --help\n");
    std::exit(status);
}

[[nodiscard]] std::string_view argument_value(int argc,
                                              char** argv,
                                              int& index,
                                              std::string_view flag) {
    if (index + 1 >= argc) {
        std::fprintf(stderr, "pnl: %s needs a value\n", std::string(flag).c_str());
        std::exit(2);
    }
    return argv[++index];
}

/// Parse the value of \p flag as an integer, naming the flag when it is not one.
///
/// std::stoll was here, and it reports a bad value by throwing
/// std::invalid_argument whose message is the single word "stoll". That was one
/// half of the fault Section 4.7 records against this file: the other half was
/// that parse() ran outside the try in main, so `pnl --size abc` reached no
/// handler at all and terminated on an unhandled exception. This reports what
/// was wrong with which flag, and main catches it.
///
/// std::from_chars rather than std::stoll, because it also refuses trailing
/// text: `--size 12abc` is a typo, not the number twelve.
///
/// \throws InvalidArgument if the text is not an integer, or has anything after
///         one.
template<typename Integer>
[[nodiscard]] Integer integer_argument(std::string_view flag, std::string_view text) {
    Integer value{};
    const char* const first = text.data();
    const char* const last = first + text.size();
    const auto [stop, code] = std::from_chars(first, last, value);
    if (code != std::errc{} || stop != last) {
        throw InvalidArgument(std::string(flag) + " needs a whole number, and '" +
                              std::string(text) + "' is not one");
    }
    return value;
}

/// The same for a floating point value.
///
/// \throws InvalidArgument if the text is not a number, or has anything after
///         one.
[[nodiscard]] Real real_argument(std::string_view flag, std::string_view text) {
    Real value{};
    const char* const first = text.data();
    const char* const last = first + text.size();
    const auto [stop, code] = std::from_chars(first, last, value);
    if (code != std::errc{} || stop != last) {
        throw InvalidArgument(std::string(flag) + " needs a number, and '" + std::string(text) +
                              "' is not one");
    }
    return value;
}

[[nodiscard]] Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h")
            usage(0);
        else if (flag == "--solver")
            options.solver = argument_value(argc, argv, i, flag);
        else if (flag == "--backend")
            options.backend = argument_value(argc, argv, i, flag);
        else if (flag == "--problem")
            options.problem = argument_value(argc, argv, i, flag);
        else if (flag == "--rhs")
            options.rhs = argument_value(argc, argv, i, flag);
        else if (flag == "--size")
            options.size = integer_argument<Index>(flag, argument_value(argc, argv, i, flag));
        else if (flag == "--workers")
            options.workers = integer_argument<int>(flag, argument_value(argc, argv, i, flag));
        else if (flag == "--threads-per-rank")
            options.threads_per_rank =
                integer_argument<int>(flag, argument_value(argc, argv, i, flag));
        else if (flag == "--pinning")
            options.pinning = argument_value(argc, argv, i, flag);
        else if (flag == "--reduction")
            options.reduction = argument_value(argc, argv, i, flag);
        else if (flag == "--schedule")
            options.schedule = argument_value(argc, argv, i, flag);
        else if (flag == "--mode")
            options.mode = argument_value(argc, argv, i, flag);
        else if (flag == "--iterations")
            options.iterations = integer_argument<Index>(flag, argument_value(argc, argv, i, flag));
        else if (flag == "--tolerance")
            options.tolerance = real_argument(flag, argument_value(argc, argv, i, flag));
        else if (flag == "--omega")
            options.relaxation = real_argument(flag, argument_value(argc, argv, i, flag));
        else if (flag == "--blocks")
            options.blocks = integer_argument<Index>(flag, argument_value(argc, argv, i, flag));
        else if (flag == "--check-interval")
            options.check_interval =
                integer_argument<Index>(flag, argument_value(argc, argv, i, flag));
        else if (flag == "--reps")
            options.repetitions = integer_argument<int>(flag, argument_value(argc, argv, i, flag));
        else if (flag == "--seed")
            options.seed =
                integer_argument<std::uint64_t>(flag, argument_value(argc, argv, i, flag));
        else if (flag == "--label")
            options.label = argument_value(argc, argv, i, flag);
        else if (flag == "--progress")
            options.progress = true;
        else if (flag == "--header")
            options.header = true;
        else if (flag == "--list")
            options.list = true;
        else if (flag == "--topology")
            options.topology = true;
        else if (flag == "--bandwidth")
            options.bandwidth = true;
        else if (flag == "--version")
            options.version = true;
        else {
            std::fprintf(stderr, "pnl: unknown option %s\n", argv[i]);
            usage(2);
        }
    }

    // Refused here rather than discovered later. Every timed path collects one
    // duration per repetition and then takes the median, the minimum and the
    // maximum of the vector it collected, so zero repetitions indexes an empty
    // vector three times over and reads whatever is at the front of nothing.
    // The row it would print carries timings that were never measured. Section
    // 4.7 lists the indexing; refusing the flag is what makes it unreachable.
    if (options.repetitions < 1) {
        throw InvalidArgument("--reps needs at least one repetition, and " +
                              std::to_string(options.repetitions) +
                              " would leave the timings vector empty");
    }

    return options;
}

[[nodiscard]] backend::Pinning parse_pinning(std::string_view text) {
    if (text == "none") return backend::Pinning::None;
    if (text == "compact") return backend::Pinning::Compact;
    if (text == "scatter") return backend::Pinning::Scatter;
    if (text == "pcore") return backend::Pinning::PerformanceCores;
    if (text == "ecore") return backend::Pinning::EfficiencyCores;
    throw InvalidArgument("unknown pinning policy '" + std::string(text) +
                          "'; expected none, compact, scatter, pcore or ecore");
}

[[nodiscard]] std::unique_ptr<problems::Problem> make_problem(const Options& options) {
    if (options.problem == "poisson") {
        const auto kind = options.rhs == "sine" ? problems::PoissonRhs::ManufacturedSine
                                                : problems::PoissonRhs::SpectrallyRich;
        return std::make_unique<problems::Poisson2D>(options.size, kind, options.seed);
    }
    if (options.problem == "dense_dd") {
        return std::make_unique<problems::DenseProblem>(options.size,
                                                        options.seed,
                                                        problems::DenseKind::DiagonallyDominant,
                                                        options.blocks > 0 ? options.blocks : 8);
    }
    if (options.problem == "dense_spd") {
        return std::make_unique<problems::DenseProblem>(
            options.size,
            options.seed,
            problems::DenseKind::SymmetricPositiveDefinite,
            options.blocks > 0 ? options.blocks : 8);
    }
    throw InvalidArgument("unknown problem '" + options.problem +
                          "'; expected poisson, dense_dd or dense_spd");
}

// The result row schema, defined once. `--header` prints it, `run_sweep.py`
// compares it against the summary it is merging into, and `migrate_summary.py`
// adds to a stored summary whatever this line has gained.
//
// Everything after `label` was appended in one schema change, so that a stored
// summary is migrated once rather than once per phase. A column whose value
// arrives with a later phase is printed empty, which pandas reads as NaN,
// because a placeholder that looks like a number is a measurement nobody made.
//
//   sweeps       updates per unknown per iteration for this method. It is the
//                multiplier in `updates = unknowns * iterations * sweeps`, and
//                so in `updates_per_second` and `gib_per_second` too. Filled
//                here from the solver.
//   passes       streams over the state array per iteration, which is what a
//                traffic model divides by. It differs from `sweeps` for exactly
//                the red black methods, which do one sweep of work in two
//                passes over memory. Filled here from the solver.
//   dram_bytes_per_unknown_per_sweep
//                the second traffic model of Section 4.2: the same pass counted
//                with read for ownership charged on every array the pass writes
//                without reading first. `bytes_per_unknown` beside it is the
//                conservative count. Both are published and neither replaces
//                the other, which is ground rule 9; which of the two the report
//                should divide by is settled by the non temporal triad probe
//                and the rule pre registered in `benchmarks/sweep_matrix.yaml`.
//                Filled here from the problem.
//   pinning_status
//                what the requested pinning achieved, as one of `not_requested`,
//                `bound`, `not_applicable` or `refused`, with the count of
//                workers the operating system refused appended after a colon
//                when it is not zero. Only the first two can appear: a policy
//                that did not bind on every worker fails the run instead of
//                writing a row. The device row carries `none`, as its `pinning`
//                column already does. Filled here from the backend.
//   measured_at  when this row was printed, ISO 8601 UTC to the second. Filled here.
//   seconds_reps every timed repetition in run order. Filled here.
//   kernels      which kernel table ran: `cxx` on a host row, `device` on a device
//                row. Fortran joins the values in release 1.2.0, part C.
//   kernel_variant
//                which implementation of that table ran: `cpp` on a host row,
//                `device` on a device row. Assembly joins it in 1.2.0, part D.
//
// `seconds_reps` is a semicolon separated list inside one CSV field, which is
// safe because no other field uses a semicolon. It is printed because the three
// order statistics beside it cannot be resampled: phase A7 bootstraps the knee
// fit, and a bootstrap needs the repetitions rather than their median, minimum
// and maximum.
//
// `omega` is empty on a row whose method has no relaxation factor. Four of the
// twelve have one: Richardson and the three members of the SOR family. It used
// to carry the SOR optimum on every row, computed by asking Sor
// unconditionally, so the other eight recorded a parameter the run never read,
// and two of the four that do use one, Richardson and SSOR, recorded a factor
// other than the one they took. The solver is asked now; see
// Solver::relaxation_factor.
constexpr const char* CSV_HEADER =
    "problem,unknowns,solver,backend,workers,ranks,threads_per_rank,pinning,reduction,"
    "schedule,mode,iterations,converged,stop_reason,relative_residual,omega,blocks,"
    "check_interval,seconds_median,seconds_min,seconds_max,reps,updates_per_second,"
    "gib_per_second,bytes_per_unknown,seed,commit,label,sweeps,passes,"
    "dram_bytes_per_unknown_per_sweep,pinning_status,measured_at,seconds_reps,kernels,"
    "kernel_variant";

/// The clock the timed region reads, defined once in pnl/bench/timed_solve.hpp
/// so that the driver and the allocation gate time the same thing.
using bench::now_seconds;

/// When this row was measured, ISO 8601 UTC to the second.
///
/// Stamped by the binary rather than by the sweep driver, for the same reason
/// the commit hash and the seed are: a row must not be able to carry provenance
/// that the run which produced it did not have.
[[nodiscard]] std::string utc_timestamp() {
    const std::time_t stamp = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&stamp, &utc);
    char text[32];
    std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return text;
}

/// The `omega` field: the factor when the method takes one, an empty field when
/// it does not.
///
/// Empty rather than zero or one. An empty field reads as NaN in pandas, which
/// is what "this run had no relaxation factor" means, where a number there is a
/// parameter a reader can average, plot and compare against a row that really
/// did use one.
[[nodiscard]] std::string format_relaxation(Real relaxation) {
    if (!(relaxation > 0.0)) return {};
    char text[32];
    std::snprintf(text, sizeof(text), "%.6f", relaxation);
    return text;
}

/// Every timed repetition in run order, joined by semicolons.
///
/// Called before the timings are sorted, because run order is the part a
/// resampling method needs and sorting destroys it. Same precision as
/// `seconds_median`, so the median in the row is one of these values.
[[nodiscard]] std::string join_seconds(const std::vector<double>& values) {
    std::string joined;
    char entry[32];
    for (const double value : values) {
        std::snprintf(entry, sizeof(entry), "%.6f", value);
        if (!joined.empty()) joined += ';';
        joined += entry;
    }
    return joined;
}

#if defined(PNL_WITH_CUDA)

/// Run one configuration on the GPU.
///
/// The device path is separate rather than another Backend implementation, and
/// deliberately so: the whole point of running on a GPU is that the state stays
/// in device memory for the entire solve. Forcing it behind the same
/// parallel_for interface would mean a host callback per chunk, which would
/// measure PCIe latency and nothing else. The numerics are the same, verified
/// against the serial backend by the CUDA tests, and the result row has the
/// same columns.
int run_cuda(const Options& options) {
    // Before the device is even looked for, because this is a property of the
    // request rather than of the machine. The device kernels index a padded
    // grid with 32 bit arithmetic and the entry point takes an int, so a size
    // above the bound would be narrowed on the way in and then wrap inside;
    // Section 4.7 records both halves. PNL_CUDA_MAX_SIDE carries the derivation.
    if (options.size > PNL_CUDA_MAX_SIDE) {
        std::fprintf(stderr,
                     "pnl: --size %td is above the largest interior side the device path can "
                     "index, %d. The kernels address the padded grid with 32 bit integers and "
                     "the index wraps above that; a grid at the limit is already 17 GB per "
                     "array. Use a smaller size, or a host backend.\n",
                     options.size,
                     static_cast<int>(PNL_CUDA_MAX_SIDE));
        return 3;
    }
    if (pnl_cuda_device_count() <= 0) {
        std::fprintf(stderr, "pnl: no CUDA device is available, so the cuda backend cannot run\n");
        return 4;
    }
    if (options.problem != "poisson") {
        std::fprintf(stderr,
                     "pnl: the cuda backend implements the 2D Poisson stencil only; a dense "
                     "system has no stencil structure for it to exploit\n");
        return 3;
    }

    int method = PNL_CUDA_JACOBI;
    if (options.solver == "jacobi")
        method = PNL_CUDA_JACOBI;
    else if (options.solver == "gauss_seidel_rb")
        method = PNL_CUDA_GAUSS_SEIDEL_RB;
    else if (options.solver == "sor_rb")
        method = PNL_CUDA_SOR_RB;
    else if (options.solver == "cg")
        method = PNL_CUDA_CG;
    else {
        std::fprintf(stderr,
                     "pnl: the cuda backend implements jacobi, gauss_seidel_rb, sor_rb and cg. "
                     "Natural ordering Gauss Seidel is absent because it is sequentially "
                     "dependent and has no parallelism to offer a wide device, which is a "
                     "result rather than a gap.\n");
        return 3;
    }

    const auto kind = options.rhs == "sine" ? problems::PoissonRhs::ManufacturedSine
                                            : problems::PoissonRhs::SpectrallyRich;
    problems::Poisson2D problem(options.size, kind, options.seed);
    const Real omega =
        options.relaxation > 0.0 ? options.relaxation : problem.theory().optimal_relaxation;

    // The work unit and the relaxation factor come from the host solver of the
    // same name rather than from a second table here. The device runs the same
    // four methods through different kernels, and a table that had to be kept
    // in step with the host one by hand would not be.
    const std::unique_ptr<solvers::Solver> reference = solvers::make_solver(options.solver);
    const solvers::WorkUnit unit = reference->work_unit();
    solvers::SolverOptions relaxation_query;
    relaxation_query.relaxation = options.relaxation;
    const std::string omega_field =
        format_relaxation(reference->relaxation_factor(problem, relaxation_query));

    std::vector<double> timings;
    PnlCudaResult device_result{};
    Vector x;

    // One untimed warm up, so context creation and the first allocation do not
    // land in the measurement, then the timed repetitions.
    for (int rep = -1; rep < options.repetitions; ++rep) {
        x = problem.make_state();
        const double start = now_seconds();
        const int status = pnl_cuda_poisson_solve(static_cast<int>(options.size),
                                                  problem.rhs().data(),
                                                  x.data(),
                                                  method,
                                                  omega,
                                                  options.tolerance,
                                                  static_cast<long>(options.iterations),
                                                  static_cast<long>(options.check_interval),
                                                  options.mode == "fixed" ? 1 : 0,
                                                  &device_result);
        const double elapsed = now_seconds() - start;
        if (status != 0) {
            std::fprintf(stderr, "pnl: the device solve failed: %s\n", pnl_cuda_last_error());
            return 1;
        }
        if (rep >= 0) timings.push_back(elapsed);
    }

    const std::string seconds_reps = join_seconds(timings);
    const std::string measured_at = utc_timestamp();

    std::sort(timings.begin(), timings.end());
    const double median = timings[timings.size() / 2];

    const auto unknowns = static_cast<double>(problem.unknown_count());
    const auto iterations = static_cast<double>(device_result.iterations);
    const double updates = unknowns * iterations * static_cast<double>(unit.sweeps);
    // Kernel time, not wall time: the bandwidth figure describes the sweep, and
    // the transfer is reported in the label so it can be added back.
    const double kernel = device_result.kernel_seconds;
    const double updates_per_second = kernel > 0.0 ? updates / kernel : 0.0;
    const double gib_per_second = kernel > 0.0 ? updates * device_result.bytes_per_unknown /
                                                     kernel / (1024.0 * 1024.0 * 1024.0)
                                               : 0.0;

    char label[192];
    std::snprintf(label,
                  sizeof(label),
                  "%skernel=%.6f transfer=%.6f",
                  options.label.empty() ? "" : (options.label + " ").c_str(),
                  kernel,
                  device_result.transfer_seconds);

    // The device path declares one byte count of its own, `bytes_per_unknown`,
    // and has no second declaration to put in the read for ownership column.
    // The host figure goes there instead: the device kernels move the same
    // three arrays per unknown as the host ones, the right hand side, the
    // iterate and the output, so the count of arrays is the same count. What
    // that column adds on top is a write allocate cache fetching the line it is
    // about to overwrite, which is a property of this host's memory system;
    // the pre registered rule settles it on the host triad and claims nothing
    // about the device.
    const double dram_bytes = problem.dram_bytes_per_unknown_per_sweep();

    // `pinning_status` reads `none`, as the `pinning` column beside it already
    // does: the device path binds no host thread and has no policy to report on.
    // `kernels` and `kernel_variant` read `device` here for the same reason
    // `reduction` does: the device path runs neither the C++ kernel table nor a
    // host variant of it.
    std::printf(
        "%s,%td,%s,cuda,1,1,1,none,%s,static,%s,%ld,%d,%s,%.6e,%s,%td,%td,"
        "%.6f,%.6f,%.6f,%d,%.6e,%.4f,%.1f,%llu,%s,%s,%td,%td,%.1f,none,%s,%s,device,device\n",
        problem.name().c_str(),
        problem.unknown_count(),
        options.solver.c_str(),
        "device",
        options.mode.c_str(),
        device_result.iterations,
        device_result.converged,
        device_result.converged ? "converged" : "iteration_cap",
        device_result.relative_residual,
        omega_field.c_str(),
        problem.natural_block_count(),
        options.check_interval,
        median,
        timings.front(),
        timings.back(),
        options.repetitions,
        updates_per_second,
        gib_per_second,
        device_result.bytes_per_unknown,
        static_cast<unsigned long long>(options.seed),
        PNL_GIT_COMMIT,
        label,
        unit.sweeps,
        unit.passes,
        dram_bytes,
        measured_at.c_str(),
        seconds_reps.c_str());
    return 0;
}

#endif  // PNL_WITH_CUDA

}  // namespace

int main(int argc, char** argv) {
    // Inside a try, which it was not. `pnl --size abc` used to reach no handler
    // at all and end in std::terminate with the word "stoll" for a diagnostic.
    // It is a separate try from the one around the run below because it has to
    // finish before MPI is initialised: the flags decide whether this process
    // is part of a distributed job at all. Section 4.7.
    Options options;
    try {
        options = parse(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "pnl: %s\n", error.what());
        std::fprintf(stderr, "pnl: try --help\n");
        return 2;
    }

    if (options.version) {
        // The version and the commit are two different facts and both are
        // printed. The version says which published release this claims to be;
        // the commit says which tree it was actually built from, and ends in
        // .dirty when that tree had uncommitted changes in a file that
        // determines the binary's behaviour.
        std::printf("pnl %s\ncommit %s\n", pnl::VERSION_STRING, PNL_GIT_COMMIT);
        return 0;
    }

    if (options.header) {
        std::printf("%s\n", CSV_HEADER);
        return 0;
    }

    if (options.list) {
        std::printf("backends:");
        for (const auto& name : backend::available_backends()) std::printf(" %s", name.c_str());
        std::printf("\nsolvers:\n");
        for (const auto& solver : solvers::all_solvers()) {
            std::printf("  %-20s %s\n",
                        std::string(solver->name()).c_str(),
                        std::string(solver->splitting()).c_str());
        }
        return 0;
    }

    if (options.topology) {
        const auto& report = backend::shared_topology(true);
        std::printf("logical processors: %d\n", report.logical_cpus);
        std::printf("physical cores:     %d\n", report.physical_cores);
        std::printf("verdict: %s\n", report.verdict.c_str());
        std::printf("\n  cpu  relative_throughput  group\n");
        for (const auto& probe : report.probes) {
            std::printf("  %3d  %19.4f  %s\n",
                        probe.cpu,
                        probe.relative_throughput,
                        report.classification_succeeded ? (probe.fast_group ? "fast" : "slow")
                                                        : "unclassified");
        }
        return 0;
    }

    if (options.bandwidth) {
        // Both halves of the Section 8.3 denominator, measured on this machine.
        //
        // The host probe sweeps worker counts rather than taking a single one,
        // because the denominator has to be what this machine can actually
        // reach. Running it at the logical processor count understates that:
        // with every hyperthread engaged the siblings contend for the same load
        // and store ports and the achieved figure falls well below the peak at
        // one thread per physical core. Dividing by an understated peak would
        // inflate every efficiency number in the report, so the best over the
        // sweep is the value reported and the whole curve is printed beside it.
        std::printf("device,gib_per_second,detail\n");

        const int available = backend::available_logical_cpus();
        std::vector<int> counts;
        if (options.workers > 0) {
            counts.push_back(options.workers);
        } else {
            for (int candidate : {2, 4, 8, 12, 16, 20, 24, 28, 32}) {
                if (candidate <= available) counts.push_back(candidate);
            }
            if (counts.empty()) counts.push_back(available);
        }

        // Two host probes over the same worker counts, not one. The plain loop
        // declares 24 bytes per element and stores into an array it never
        // reads, so on a write allocate cache it would move 32: the store
        // misses and the line is fetched before it is overwritten. The non
        // temporal loop writes through _mm256_stream_pd and moves 24 for real.
        // Both are reported against the same declared 24, so the ratio of the
        // two is the ratio of the traffic they actually move, and it is the
        // instrument Section 4.2 asks for. Whether the plain loop really pays
        // that read is the open question; nothing here assumes an answer. The
        // second probe is an additional row, never a replacement.
        //
        // Five repetitions of each probe at every worker count, interleaved, and
        // every figure is reported rather than only the best.
        //
        // MEAS-07 measured the selection statistic three times on an idle
        // machine and got 1.0990, 1.0121 and 1.0601: a spread of 0.087 against
        // an undecided band 0.10 wide. Keeping the best of five timings inside
        // one probe suppresses variance within a run and says nothing about
        // variance between runs, and the statistic was being asked to decide
        // something finer than its own spread. The amendment of 2026-09-06 to
        // the pre registration in benchmarks/sweep_matrix.yaml takes the
        // statistic at a matched worker count over five repetitions and records
        // its interval; alternating the two probes at each count is what makes
        // a repetition's two arms comparable, since anything drifting under the
        // guest drifts under both of them together.
        constexpr int kProbeRepetitions = 5;
        backend::StreamResult best;
        backend::StreamResult best_nt;
        std::string curve;
        std::string curve_nt;
        std::string every;
        std::string every_nt;
        for (int workers : counts) {
            backend::Config config;
            config.workers = workers;
            auto execution = backend::make_backend(
                options.backend == "cuda" ? "openmp" : options.backend, config);
            backend::StreamResult count_best;
            backend::StreamResult count_best_nt;
            char entry[32];
            std::snprintf(entry, sizeof(entry), "%d:", workers);
            every += entry;
            every_nt += entry;
            for (int repeat = 0; repeat < kProbeRepetitions; ++repeat) {
                const auto measured = backend::measure_host_triad(*execution);
                const auto streamed = backend::measure_host_triad_nontemporal(*execution);
                std::snprintf(
                    entry, sizeof(entry), "%s%.3f", repeat ? "|" : "", measured.gib_per_second);
                every += entry;
                std::snprintf(
                    entry, sizeof(entry), "%s%.3f", repeat ? "|" : "", streamed.gib_per_second);
                every_nt += entry;
                if (measured.gib_per_second > count_best.gib_per_second) count_best = measured;
                if (streamed.gib_per_second > count_best_nt.gib_per_second) {
                    count_best_nt = streamed;
                }
            }
            every += " ";
            every_nt += " ";
            std::snprintf(entry, sizeof(entry), "%d:%.1f ", workers, count_best.gib_per_second);
            curve += entry;
            std::snprintf(entry, sizeof(entry), "%d:%.1f ", workers, count_best_nt.gib_per_second);
            curve_nt += entry;
            if (count_best.gib_per_second > best.gib_per_second) best = count_best;
            if (count_best_nt.gib_per_second > best_nt.gib_per_second) best_nt = count_best_nt;
        }

        std::printf(
            "host,%.3f,plain stores over the execution backend, best of workers %s over %td "
            "MiB arrays, %d repetitions each; declares 24 bytes per element\n",
            best.gib_per_second,
            curve.c_str(),
            best.bytes_per_array / (1024 * 1024),
            kProbeRepetitions);
        std::printf(
            "host_nontemporal,%.3f,%s, best of workers %s over %td MiB arrays, %d "
            "repetitions each; %s\n",
            best_nt.gib_per_second,
            best_nt.nontemporal ? "_mm256_stream_pd with one sfence"
                                : "scalar fallback, this build has no AVX",
            curve_nt.c_str(),
            best_nt.bytes_per_array / (1024 * 1024),
            kProbeRepetitions,
            best_nt.nontemporal ? "moves 24 bytes per element for real"
                                : "issued ordinary stores, so this figure settles nothing");

        // Every repetition, not the best of them, as workers:first|second|...
        // per worker count. Prefixed `repetitions_` so the sweep driver files
        // them apart from a single figure and reads the amended statistic off
        // them; the rule that turns them into an outcome lives in run_sweep.py
        // and in one place only.
        std::printf("repetitions_host,%d,%s\n", kProbeRepetitions, every.c_str());
        std::printf("repetitions_host_nontemporal,%d,%s\n", kProbeRepetitions, every_nt.c_str());

        // Derived, not measured. Printed here because the traffic model is read
        // off these three numbers and a reader should not have to recompute
        // them, and prefixed `derived_` so the sweep driver files them apart
        // from the two figures a probe actually returned.
        if (best_nt.nontemporal && best.gib_per_second > 0.0 && best_nt.gib_per_second > 0.0) {
            const double ratio = best_nt.gib_per_second / best.gib_per_second;
            std::printf(
                "derived_ratio_nontemporal_over_plain,%.4f,derived, not measured: "
                "host_nontemporal divided by host, each the best over the worker sweep. "
                "This is what the A3a pre registration selected on as registered, and the "
                "amendment of 2026-09-06 no longer does, because the two bests need not "
                "come from the same worker count; it is kept so the registered figure "
                "stays visible. The statistic that decides is traffic_model in the session "
                "manifest. Thresholds either way: above 1.20 charges read for ownership, "
                "below 1.10 does not, between is unresolved\n",
                ratio);
            std::printf(
                "derived_ratio_plain_over_nontemporal,%.4f,derived, not measured: the "
                "reciprocal, recorded so the rule cannot be read in the wrong direction. "
                "Above 0.909 charges nothing, below 0.833 charges read for ownership\n",
                1.0 / ratio);
        } else {
            std::printf(
                "derived_ratio_nontemporal_over_plain,,not available: the non temporal probe "
                "fell back to ordinary stores, so there is no ratio to take\n");
            std::printf(
                "derived_ratio_plain_over_nontemporal,,not available: the non temporal probe "
                "fell back to ordinary stores, so there is no ratio to take\n");
        }
        std::printf(
            "derived_host_plain_at_32_bytes,%.3f,derived, not measured: host times 32 over 24, "
            "the plain triad recounted with the read for ownership its own store pays and its "
            "declared figure does not charge. No memory speed is recorded anywhere in this "
            "repository, so this number is not corroborated against a theoretical peak\n",
            best.gib_per_second * 32.0 / 24.0);
#if defined(PNL_WITH_CUDA)
        if (pnl_cuda_device_count() > 0) {
            char name[256] = {0};
            int major = 0, minor = 0, multiprocessors = 0;
            std::size_t total = 0;
            pnl_cuda_device_info(0, name, sizeof(name), &major, &minor, &total, &multiprocessors);
            const double gpu = pnl_cuda_stream_triad(0, 512u * 1024u * 1024u, 5);
            if (gpu > 0.0) {
                std::printf("gpu,%.3f,%s sm_%d%d %d SMs %.1f GiB, 512 MiB arrays, best of 5\n",
                            gpu,
                            name,
                            major,
                            minor,
                            multiprocessors,
                            static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0));
            } else {
                std::fprintf(
                    stderr, "pnl: device bandwidth probe failed: %s\n", pnl_cuda_last_error());
            }
        } else {
            std::printf("gpu,,no CUDA device present, probe skipped\n");
        }
#else
        std::printf("gpu,,built without CUDA, probe skipped\n");
#endif
        return 0;
    }

#if defined(PNL_WITH_CUDA)
    if (options.backend == "cuda") {
        return run_cuda(options);
    }
#elif !defined(PNL_WITH_CUDA)
    if (options.backend == "cuda") {
        std::fprintf(stderr, "pnl: this build has no CUDA backend\n");
        return 4;
    }
#endif

#if defined(PNL_WITH_MPI)
    const bool distributed = options.backend == "mpi" || options.backend == "hybrid";
    if (distributed) {
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (provided < MPI_THREAD_FUNNELED) {
            std::fprintf(stderr,
                         "pnl: the MPI library provided thread level %d, below the "
                         "MPI_THREAD_FUNNELED this build needs\n",
                         provided);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
#endif

    int status = 0;
    try {
        backend::Config config;
        config.workers = options.workers;
        config.threads_per_rank = options.threads_per_rank;
        config.pinning = parse_pinning(options.pinning);
        config.reduction = options.reduction == "native" ? backend::ReductionMode::Native
                                                         : backend::ReductionMode::Deterministic;
        config.schedule =
            options.schedule == "dynamic" ? backend::Schedule::Dynamic : backend::Schedule::Static;

        auto problem = make_problem(options);
        auto execution = backend::make_backend(options.backend, config);
        auto solver = solvers::make_solver(options.solver);

        if (!solver->applicable_to(*problem)) {
            if (execution->is_root()) {
                std::fprintf(stderr,
                             "pnl: solver %s does not apply to problem %s: %s\n",
                             options.solver.c_str(),
                             problem->name().c_str(),
                             solver->inapplicable_reason(*problem).c_str());
            }
            status = 3;
            throw InvalidArgument("solver not applicable");
        }

        solvers::SolverOptions solver_options;
        solver_options.tolerance = options.tolerance;
        solver_options.max_iterations = options.iterations;
        solver_options.mode = options.mode == "fixed" ? solvers::RunMode::FixedIterations
                                                      : solvers::RunMode::ToTolerance;
        solver_options.check_interval = options.check_interval;
        solver_options.relaxation = options.relaxation;
        solver_options.block_count = options.blocks;
        solver_options.show_progress = options.progress;

        // Allocated once, here, outside the repetition loop. It used to be
        // allocated inside solve() on every call: three full state vectors for
        // most methods and four for conjugate gradient, which at 4095 squared
        // is 384 MiB of fresh mapping per repetition and a first touch page
        // fault on every page of it. The warm up below bought nothing, because
        // the memory it faulted in was freed before the first timed repetition
        // asked for its own. That is MEAS-10.
        solvers::SolverWorkspace workspace = solver->make_workspace(*problem);

        // One untimed warm up, through that same workspace, so page faults,
        // first touch and any first dispatch cost inside a backend land here
        // and not in the measurement. Then the timed repetitions, each of which
        // resets the workspace outside its own timed region.
        bench::TimedRepetition repetition =
            bench::timed_repetition(*solver, *problem, *execution, solver_options, workspace);

        std::vector<double> timings;
        timings.reserve(static_cast<std::size_t>(options.repetitions));
        for (int rep = 0; rep < options.repetitions; ++rep) {
            repetition =
                bench::timed_repetition(*solver, *problem, *execution, solver_options, workspace);
            timings.push_back(repetition.seconds);
        }
        const std::string seconds_reps = join_seconds(timings);
        const std::string measured_at = utc_timestamp();

        std::sort(timings.begin(), timings.end());
        const double median = timings[timings.size() / 2];

        if (execution->is_root()) {
            const auto unknowns = static_cast<double>(problem->unknown_count());
            const auto iterations = static_cast<double>(repetition.report.diagnostics.iterations);
            // The work unit, not the iteration count. A symmetric method writes
            // every unknown twice per iteration and used to be credited with
            // one, which halved its updates per second and its bandwidth
            // against methods that do half the work. See MEAS-03.
            const auto sweeps = static_cast<double>(repetition.report.diagnostics.sweeps);
            const double updates = unknowns * iterations * sweeps;
            const double updates_per_second = median > 0.0 ? updates / median : 0.0;
            const double bytes = problem->bytes_per_unknown_per_sweep();
            // `gib_per_second` keeps dividing by the conservative count it has
            // always divided by. Ground rule 9: the second count is published
            // beside it in its own column and the report derives the second
            // bandwidth from that, so no number a reader has already quoted
            // changes meaning underneath them.
            const double dram_bytes = problem->dram_bytes_per_unknown_per_sweep();
            const double gib_per_second =
                median > 0.0 ? updates * bytes / median / (1024.0 * 1024.0 * 1024.0) : 0.0;

            const std::string omega =
                format_relaxation(solver->relaxation_factor(*problem, solver_options));

            // A row that says it pinned must have pinned. With make_backend
            // refusing a policy this machine cannot classify for, and every
            // shared memory backend throwing when a worker was refused, this
            // cannot fire; the check is what makes that an invariant rather
            // than an accident. MEAS-08.
            const std::string pinning_status = execution->pinning_status();
            if (config.pinning != backend::Pinning::None && pinning_status != "bound") {
                throw BackendFailure("refusing to write a row that claims '" +
                                     std::string(backend::to_string(config.pinning)) +
                                     "' pinning with a status of '" + pinning_status + "'");
            }

            // `kernels` and `kernel_variant` are the C++ table and its C++
            // implementation, which is all this release has.
            std::printf(
                "%s,%td,%s,%s,%d,%d,%d,%s,%s,%s,%s,%td,%d,%s,%.6e,%s,%td,%td,"
                "%.6f,%.6f,%.6f,%d,%.6e,%.4f,%.1f,%llu,%s,%s,%td,%td,%.1f,%s,%s,%s,cxx,cpp\n",
                problem->name().c_str(),
                problem->unknown_count(),
                std::string(solver->name()).c_str(),
                std::string(execution->name()).c_str(),
                execution->worker_count(),
                execution->rank_count(),
                options.threads_per_rank,
                std::string(backend::to_string(config.pinning)).c_str(),
                std::string(backend::to_string(config.reduction)).c_str(),
                options.schedule.c_str(),
                options.mode.c_str(),
                repetition.report.diagnostics.iterations,
                repetition.report.diagnostics.converged ? 1 : 0,
                std::string(to_string(repetition.report.diagnostics.reason)).c_str(),
                repetition.report.diagnostics.error_estimate,
                omega.c_str(),
                solver_options.block_count > 0 ? solver_options.block_count
                                               : problem->natural_block_count(),
                options.check_interval,
                median,
                timings.front(),
                timings.back(),
                options.repetitions,
                updates_per_second,
                gib_per_second,
                bytes,
                static_cast<unsigned long long>(options.seed),
                PNL_GIT_COMMIT,
                options.label.c_str(),
                repetition.report.diagnostics.sweeps,
                repetition.report.diagnostics.passes,
                dram_bytes,
                pinning_status.c_str(),
                measured_at.c_str(),
                seconds_reps.c_str());
        }
    } catch (const Error& error) {
        std::fprintf(stderr, "pnl: %s\n", error.what());
        if (status == 0) status = 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "pnl: unexpected failure: %s\n", error.what());
        status = 1;
    }

#if defined(PNL_WITH_MPI)
    if (distributed) {
        // A distributed failure is a job failure, and it has to be spelled that
        // way rather than left to the exit status. The catch above is rank
        // local: the rank that threw walks out of the solve while every other
        // rank is still inside it, waiting in the halo exchange or the
        // allgather that this one was going to take part in. Returning here
        // would finalise on one rank and leave the rest of the job resident
        // until something killed it, which is the hang Section 4.7 records
        // against mpi.hpp. MPI_Abort takes the whole communicator down with a
        // status a caller can read.
        if (status != 0) {
            std::fflush(stderr);
            std::fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, status);
        }
        MPI_Finalize();
    }
#endif
    return status;
}
