/// \file main.cpp
/// Command line driver.
///
/// One invocation runs one configuration: a solver, on a backend, on a problem,
/// at a worker count and pinning policy, repeated a few times. It prints a
/// single result row carrying everything Section 7 requires to reproduce it,
/// including the seed and the commit hash, so no number in the report can exist
/// without the run that produced it.

#include <pnl/backend/backend.hpp>
#include <pnl/backend/topology.hpp>
#include <pnl/core/error.hpp>
#include <pnl/problems/dense_generator.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include <pnl/backend/stream_probe.hpp>

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
    std::string label;
};

[[noreturn]] void usage(int status) {
    std::fprintf(
        status == 0 ? stdout : stderr,
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
        "  --help\n");
    std::exit(status);
}

[[nodiscard]] std::string_view argument_value(int argc, char** argv, int& index,
                                              std::string_view flag) {
    if (index + 1 >= argc) {
        std::fprintf(stderr, "pnl: %s needs a value\n", std::string(flag).c_str());
        std::exit(2);
    }
    return argv[++index];
}

[[nodiscard]] Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") usage(0);
        else if (flag == "--solver") options.solver = argument_value(argc, argv, i, flag);
        else if (flag == "--backend") options.backend = argument_value(argc, argv, i, flag);
        else if (flag == "--problem") options.problem = argument_value(argc, argv, i, flag);
        else if (flag == "--rhs") options.rhs = argument_value(argc, argv, i, flag);
        else if (flag == "--size") options.size = std::stoll(std::string(argument_value(argc, argv, i, flag)));
        else if (flag == "--workers") options.workers = std::stoi(std::string(argument_value(argc, argv, i, flag)));
        else if (flag == "--threads-per-rank") options.threads_per_rank = std::stoi(std::string(argument_value(argc, argv, i, flag)));
        else if (flag == "--pinning") options.pinning = argument_value(argc, argv, i, flag);
        else if (flag == "--reduction") options.reduction = argument_value(argc, argv, i, flag);
        else if (flag == "--schedule") options.schedule = argument_value(argc, argv, i, flag);
        else if (flag == "--mode") options.mode = argument_value(argc, argv, i, flag);
        else if (flag == "--iterations") options.iterations = std::stoll(std::string(argument_value(argc, argv, i, flag)));
        else if (flag == "--tolerance") options.tolerance = std::stod(std::string(argument_value(argc, argv, i, flag)));
        else if (flag == "--omega") options.relaxation = std::stod(std::string(argument_value(argc, argv, i, flag)));
        else if (flag == "--blocks") options.blocks = std::stoll(std::string(argument_value(argc, argv, i, flag)));
        else if (flag == "--check-interval") options.check_interval = std::stoll(std::string(argument_value(argc, argv, i, flag)));
        else if (flag == "--reps") options.repetitions = std::stoi(std::string(argument_value(argc, argv, i, flag)));
        else if (flag == "--seed") options.seed = std::stoull(std::string(argument_value(argc, argv, i, flag)));
        else if (flag == "--label") options.label = argument_value(argc, argv, i, flag);
        else if (flag == "--progress") options.progress = true;
        else if (flag == "--header") options.header = true;
        else if (flag == "--list") options.list = true;
        else if (flag == "--topology") options.topology = true;
        else if (flag == "--bandwidth") options.bandwidth = true;
        else {
            std::fprintf(stderr, "pnl: unknown option %s\n", argv[i]);
            usage(2);
        }
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
        return std::make_unique<problems::DenseProblem>(
            options.size, options.seed, problems::DenseKind::DiagonallyDominant,
            options.blocks > 0 ? options.blocks : 8);
    }
    if (options.problem == "dense_spd") {
        return std::make_unique<problems::DenseProblem>(
            options.size, options.seed, problems::DenseKind::SymmetricPositiveDefinite,
            options.blocks > 0 ? options.blocks : 8);
    }
    throw InvalidArgument("unknown problem '" + options.problem +
                          "'; expected poisson, dense_dd or dense_spd");
}

constexpr const char* CSV_HEADER =
    "problem,unknowns,solver,backend,workers,ranks,threads_per_rank,pinning,reduction,"
    "schedule,mode,iterations,converged,stop_reason,relative_residual,omega,blocks,"
    "check_interval,seconds_median,seconds_min,seconds_max,reps,updates_per_second,"
    "gib_per_second,bytes_per_unknown,seed,commit,label";

[[nodiscard]] double now_seconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
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
    if (pnl_cuda_device_count() <= 0) {
        std::fprintf(stderr,
                     "pnl: no CUDA device is available, so the cuda backend cannot run\n");
        return 4;
    }
    if (options.problem != "poisson") {
        std::fprintf(stderr,
                     "pnl: the cuda backend implements the 2D Poisson stencil only; a dense "
                     "system has no stencil structure for it to exploit\n");
        return 3;
    }

    int method = PNL_CUDA_JACOBI;
    if (options.solver == "jacobi") method = PNL_CUDA_JACOBI;
    else if (options.solver == "gauss_seidel_rb") method = PNL_CUDA_GAUSS_SEIDEL_RB;
    else if (options.solver == "sor_rb") method = PNL_CUDA_SOR_RB;
    else if (options.solver == "cg") method = PNL_CUDA_CG;
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
    const Real omega = options.relaxation > 0.0 ? options.relaxation
                                                : problem.theory().optimal_relaxation;

    std::vector<double> timings;
    PnlCudaResult device_result{};
    Vector x;

    // One untimed warm up, so context creation and the first allocation do not
    // land in the measurement, then the timed repetitions.
    for (int rep = -1; rep < options.repetitions; ++rep) {
        x = problem.make_state();
        const double start = now_seconds();
        const int status = pnl_cuda_poisson_solve(
            static_cast<int>(options.size), problem.rhs().data(), x.data(), method, omega,
            options.tolerance, static_cast<long>(options.iterations),
            static_cast<long>(options.check_interval),
            options.mode == "fixed" ? 1 : 0, &device_result);
        const double elapsed = now_seconds() - start;
        if (status != 0) {
            std::fprintf(stderr, "pnl: the device solve failed: %s\n", pnl_cuda_last_error());
            return 1;
        }
        if (rep >= 0) timings.push_back(elapsed);
    }

    std::sort(timings.begin(), timings.end());
    const double median = timings[timings.size() / 2];

    const auto unknowns = static_cast<double>(problem.unknown_count());
    const auto iterations = static_cast<double>(device_result.iterations);
    const double updates = unknowns * iterations;
    // Kernel time, not wall time: the bandwidth figure describes the sweep, and
    // the transfer is reported in the label so it can be added back.
    const double kernel = device_result.kernel_seconds;
    const double updates_per_second = kernel > 0.0 ? updates / kernel : 0.0;
    const double gib_per_second =
        kernel > 0.0
            ? updates * device_result.bytes_per_unknown / kernel / (1024.0 * 1024.0 * 1024.0)
            : 0.0;

    char label[192];
    std::snprintf(label, sizeof(label), "%skernel=%.6f transfer=%.6f",
                  options.label.empty() ? "" : (options.label + " ").c_str(), kernel,
                  device_result.transfer_seconds);

    std::printf("%s,%td,%s,cuda,1,1,1,none,%s,static,%s,%ld,%d,%s,%.6e,%.6f,%td,%td,"
                "%.6f,%.6f,%.6f,%d,%.6e,%.4f,%.1f,%llu,%s,%s\n",
                problem.name().c_str(), problem.unknown_count(), options.solver.c_str(),
                "device", options.mode.c_str(), device_result.iterations,
                device_result.converged,
                device_result.converged ? "converged" : "iteration_cap",
                device_result.relative_residual, omega, problem.natural_block_count(),
                options.check_interval, median, timings.front(), timings.back(),
                options.repetitions, updates_per_second, gib_per_second,
                device_result.bytes_per_unknown,
                static_cast<unsigned long long>(options.seed), PNL_GIT_COMMIT, label);
    return 0;
}

#endif  // PNL_WITH_CUDA

}  // namespace

int main(int argc, char** argv) {
    Options options = parse(argc, argv);

    if (options.header) {
        std::printf("%s\n", CSV_HEADER);
        return 0;
    }

    if (options.list) {
        std::printf("backends:");
        for (const auto& name : backend::available_backends()) std::printf(" %s", name.c_str());
        std::printf("\nsolvers:\n");
        for (const auto& solver : solvers::all_solvers()) {
            std::printf("  %-20s %s\n", std::string(solver->name()).c_str(),
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
            std::printf("  %3d  %19.4f  %s\n", probe.cpu, probe.relative_throughput,
                        report.classification_succeeded ? (probe.fast_group ? "fast" : "slow")
                                                        : "unclassified");
        }
        return 0;
    }

    if (options.bandwidth) {
        // Both halves of the Section 8.3 denominator, measured on this machine.
        backend::Config config;
        config.workers = options.workers;
        auto execution = backend::make_backend(
            options.backend == "cuda" ? "openmp" : options.backend, config);
        const auto host = backend::measure_host_triad(*execution);
        std::printf("device,gib_per_second,detail\n");
        std::printf("host,%.3f,%d workers over %td MiB arrays, best of %d\n",
                    host.gib_per_second, host.workers,
                    host.bytes_per_array / (1024 * 1024), host.repeats);
#if defined(PNL_WITH_CUDA)
        if (pnl_cuda_device_count() > 0) {
            char name[256] = {0};
            int major = 0, minor = 0, multiprocessors = 0;
            std::size_t total = 0;
            pnl_cuda_device_info(0, name, sizeof(name), &major, &minor, &total,
                                 &multiprocessors);
            const double gpu = pnl_cuda_stream_triad(0, 512u * 1024u * 1024u, 5);
            if (gpu > 0.0) {
                std::printf("gpu,%.3f,%s sm_%d%d %d SMs %.1f GiB, 512 MiB arrays, best of 5\n",
                            gpu, name, major, minor, multiprocessors,
                            static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0));
            } else {
                std::fprintf(stderr, "pnl: device bandwidth probe failed: %s\n",
                             pnl_cuda_last_error());
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
                std::fprintf(stderr, "pnl: solver %s does not apply to problem %s: %s\n",
                             options.solver.c_str(), problem->name().c_str(),
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

        // One untimed warm up so page faults and first touch do not land in the
        // measurement, then the timed repetitions.
        SolveResult result = solver->solve(*problem, *execution, solver_options);

        std::vector<double> timings;
        timings.reserve(static_cast<std::size_t>(options.repetitions));
        for (int rep = 0; rep < options.repetitions; ++rep) {
            execution->barrier();
            const double start = now_seconds();
            result = solver->solve(*problem, *execution, solver_options);
            execution->barrier();
            timings.push_back(now_seconds() - start);
        }
        std::sort(timings.begin(), timings.end());
        const double median = timings[timings.size() / 2];

        if (execution->is_root()) {
            const auto unknowns = static_cast<double>(problem->unknown_count());
            const auto iterations = static_cast<double>(result.diagnostics.iterations);
            const double updates = unknowns * iterations;
            const double updates_per_second = median > 0.0 ? updates / median : 0.0;
            const double bytes = problem->bytes_per_unknown_per_sweep();
            const double gib_per_second =
                median > 0.0 ? updates * bytes / median / (1024.0 * 1024.0 * 1024.0) : 0.0;

            const Real omega =
                options.relaxation > 0.0
                    ? options.relaxation
                    : solvers::Sor::resolve_relaxation(*problem, solver_options);

            std::printf(
                "%s,%td,%s,%s,%d,%d,%d,%s,%s,%s,%s,%td,%d,%s,%.6e,%.6f,%td,%td,"
                "%.6f,%.6f,%.6f,%d,%.6e,%.4f,%.1f,%llu,%s,%s\n",
                problem->name().c_str(), problem->unknown_count(),
                std::string(solver->name()).c_str(), std::string(execution->name()).c_str(),
                execution->worker_count(), execution->rank_count(), options.threads_per_rank,
                std::string(backend::to_string(config.pinning)).c_str(),
                std::string(backend::to_string(config.reduction)).c_str(),
                options.schedule.c_str(), options.mode.c_str(), result.diagnostics.iterations,
                result.diagnostics.converged ? 1 : 0,
                std::string(to_string(result.diagnostics.reason)).c_str(),
                result.diagnostics.error_estimate, omega,
                solver_options.block_count > 0 ? solver_options.block_count
                                               : problem->natural_block_count(),
                options.check_interval, median, timings.front(), timings.back(),
                options.repetitions, updates_per_second, gib_per_second, bytes,
                static_cast<unsigned long long>(options.seed), PNL_GIT_COMMIT,
                options.label.c_str());
        }
    } catch (const Error& error) {
        std::fprintf(stderr, "pnl: %s\n", error.what());
        if (status == 0) status = 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "pnl: unexpected failure: %s\n", error.what());
        status = 1;
    }

#if defined(PNL_WITH_MPI)
    if (distributed) MPI_Finalize();
#endif
    return status;
}
