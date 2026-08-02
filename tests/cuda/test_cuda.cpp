/// \file test_cuda.cpp
/// GPU sweeps against the CPU serial reference.
///
/// The Section 10 gate is that GPU sweeps match CPU serial to tolerance. This
/// suite asserts something stronger where it is achievable and says plainly
/// where it is not.
///
/// The sweep kernels are compiled with fused multiply add contraction disabled
/// and evaluate their arithmetic in the same order as the host, so a Jacobi or
/// red black sweep is asserted **bit identical** to the serial backend. The
/// reductions are not: a tree reduction inside a thread block associates
/// differently from the host's ordered chunk sum, and no compiler flag changes
/// that. Anything that goes through a reduction is therefore compared to
/// tolerance, and the difference is a documented property of the port.
///
/// When no GPU is present every case skips and says so, rather than failing.
/// CI builds the CUDA code on a machine with no device, so this matters.

#include <pnl_test.hpp>

#include <pnl/backend/cuda.hpp>
#include <pnl/backend/serial.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/registry.hpp>

#include <cstdio>

using namespace pnl;
using namespace pnl::solvers;

namespace {

bool gpu_present() {
    static const bool present = pnl_cuda_device_count() > 0;
    return present;
}

/// Print a skip notice once, so a run without a GPU reads as deliberate.
bool skip_without_gpu(const char* what) {
    if (gpu_present()) return false;
    std::printf("  skip  %s (no CUDA device present)\n", what);
    return true;
}

backend::SerialBackend make_serial() { return backend::SerialBackend{backend::Config{}}; }

/// Run the device solver over a fixed number of sweeps.
PnlCudaResult device_fixed(problems::Poisson2D& problem, Vector& x, int method, double omega,
                           long sweeps) {
    PnlCudaResult result{};
    const int status = pnl_cuda_poisson_solve(static_cast<int>(problem.side()),
                                              problem.rhs().data(), x.data(), method, omega,
                                              1.0e-12, sweeps, 1000000, 1, &result);
    PNL_REQUIRE_MESSAGE(status == 0, std::string("the device solve failed: ") +
                                         pnl_cuda_last_error());
    return result;
}

Real worst_interior_difference(const problems::Poisson2D& problem, const Vector& a,
                               const Vector& b) {
    Real worst = 0.0;
    for (Index i = 1; i <= problem.side(); ++i) {
        for (Index j = 1; j <= problem.side(); ++j) {
            const auto k = static_cast<std::size_t>(problem.at(i, j));
            worst = std::max(worst, std::abs(a[k] - b[k]));
        }
    }
    return worst;
}

SolverOptions fixed_options(Index sweeps) {
    SolverOptions options;
    options.mode = RunMode::FixedIterations;
    options.max_iterations = sweeps;
    options.check_interval = 1000000;
    return options;
}

}  // namespace

PNL_TEST("cuda/a device is present and describes itself") {
    if (skip_without_gpu("device description")) return;
    char name[256] = {0};
    int major = 0;
    int minor = 0;
    int multiprocessors = 0;
    std::size_t total = 0;
    PNL_REQUIRE(pnl_cuda_device_info(0, name, sizeof(name), &major, &minor, &total,
                                     &multiprocessors) == 0);
    std::printf("        device: %s sm_%d%d, %d SMs, %.1f GiB\n", name, major, minor,
                multiprocessors, static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0));
    PNL_REQUIRE(major >= 5);
    PNL_REQUIRE(total > 0);
}

PNL_TEST("cuda/the Jacobi sweep is bit identical to the CPU") {
    if (skip_without_gpu("Jacobi sweep equality")) return;
    for (Index n : {63, 127, 255}) {
        problems::Poisson2D problem(n, problems::PoissonRhs::SpectrallyRich);
        auto serial = make_serial();

        auto solver = make_solver("jacobi");
        const Vector expected = solver->solve(problem, serial, fixed_options(30)).solution;

        Vector actual = problem.make_state();
        (void)device_fixed(problem, actual, PNL_CUDA_JACOBI, 1.0, 30);

        const Real difference = worst_interior_difference(problem, expected, actual);
        PNL_REQUIRE_MESSAGE(
            difference == 0.0,
            "at n = " + std::to_string(n) + " the device Jacobi sweep differs from the host by " +
                test::format(difference) +
                "; with fused multiply add disabled these should be bit identical");
    }
}

PNL_TEST("cuda/the red black Gauss Seidel sweep is bit identical to the CPU") {
    if (skip_without_gpu("red black sweep equality")) return;
    for (Index n : {63, 128, 255}) {
        problems::Poisson2D problem(n, problems::PoissonRhs::SpectrallyRich);
        auto serial = make_serial();

        auto solver = make_solver("gauss_seidel_rb");
        const Vector expected = solver->solve(problem, serial, fixed_options(30)).solution;

        Vector actual = problem.make_state();
        (void)device_fixed(problem, actual, PNL_CUDA_GAUSS_SEIDEL_RB, 1.0, 30);

        const Real difference = worst_interior_difference(problem, expected, actual);
        PNL_REQUIRE_MESSAGE(
            difference == 0.0,
            "at n = " + std::to_string(n) +
                " the device red black sweep differs from the host by " +
                test::format(difference));
    }
}

PNL_TEST("cuda/red black SOR matches the CPU at the optimal factor") {
    if (skip_without_gpu("red black SOR equality")) return;
    const Index n = 127;
    problems::Poisson2D problem(n, problems::PoissonRhs::SpectrallyRich);
    auto serial = make_serial();
    const Real omega = problem.theory().optimal_relaxation;

    auto solver = make_solver("sor_rb");
    SolverOptions options = fixed_options(25);
    options.relaxation = omega;
    const Vector expected = solver->solve(problem, serial, options).solution;

    Vector actual = problem.make_state();
    (void)device_fixed(problem, actual, PNL_CUDA_SOR_RB, omega, 25);

    const Real difference = worst_interior_difference(problem, expected, actual);
    PNL_REQUIRE_MESSAGE(difference == 0.0,
                        "the device red black SOR sweep differs from the host by " +
                            test::format(difference));
}

PNL_TEST("cuda/conjugate gradient agrees with the CPU to reduction tolerance") {
    // Not bit identical, and not claimed to be: CG goes through two reductions
    // per iteration, and a device tree reduction associates differently from an
    // ordered host sum. The iterates diverge in the last bits and the
    // difference compounds, so the assertion is on the converged answer.
    if (skip_without_gpu("conjugate gradient agreement")) return;
    const Index n = 255;
    problems::Poisson2D problem(n, problems::PoissonRhs::SpectrallyRich);
    auto serial = make_serial();

    auto solver = make_solver("cg");
    SolverOptions options;
    options.tolerance = 1.0e-10;
    options.max_iterations = 20000;
    const auto expected = solver->solve(problem, serial, options);
    PNL_REQUIRE(expected.converged());

    Vector actual = problem.make_state();
    PnlCudaResult device{};
    const int status = pnl_cuda_poisson_solve(static_cast<int>(n), problem.rhs().data(),
                                              actual.data(), PNL_CUDA_CG, 1.0, 1.0e-10, 20000,
                                              1, 0, &device);
    PNL_REQUIRE_MESSAGE(status == 0, std::string("device CG failed: ") + pnl_cuda_last_error());
    PNL_REQUIRE_MESSAGE(device.converged != 0, "device CG did not converge");

    const Real difference = worst_interior_difference(problem, expected.solution, actual);
    PNL_REQUIRE_MESSAGE(difference <= 1.0e-8,
                        "device CG differs from host CG by " + test::format(difference));

    // Both should need a comparable number of iterations; a large gap would
    // mean the recurrences have genuinely diverged rather than merely rounded
    // differently.
    const Real ratio = static_cast<Real>(device.iterations) /
                       static_cast<Real>(expected.diagnostics.iterations);
    PNL_REQUIRE_MESSAGE(ratio > 0.8 && ratio < 1.25,
                        "device CG took " + std::to_string(device.iterations) +
                            " iterations against the host's " +
                            std::to_string(expected.diagnostics.iterations));
}

PNL_TEST("cuda/the red black ordering penalty is measured") {
    // Section 5 predicts red black converges slightly more slowly per iteration
    // than natural ordering. Both counts are recorded here so the report quotes
    // a measurement.
    if (skip_without_gpu("red black iteration penalty")) return;
    const Index n = 127;
    problems::Poisson2D problem(n, problems::PoissonRhs::SpectrallyRich);
    auto serial = make_serial();

    SolverOptions options;
    options.tolerance = 1.0e-8;
    options.max_iterations = 500000;
    options.check_interval = 1;

    const auto natural = make_solver("gauss_seidel_f")->solve(problem, serial, options);
    const auto red_black = make_solver("gauss_seidel_rb")->solve(problem, serial, options);
    PNL_REQUIRE(natural.converged() && red_black.converged());

    Vector x = problem.make_state();
    PnlCudaResult device{};
    const int status =
        pnl_cuda_poisson_solve(static_cast<int>(n), problem.rhs().data(), x.data(),
                               PNL_CUDA_GAUSS_SEIDEL_RB, 1.0, 1.0e-8, 500000, 1, 0, &device);
    PNL_REQUIRE(status == 0);
    PNL_REQUIRE(device.converged != 0);

    std::printf("        natural %td, red black host %td, red black device %ld iterations\n",
                natural.diagnostics.iterations, red_black.diagnostics.iterations,
                device.iterations);

    // The device and host red black runs solve the identical recurrence, so
    // their counts must agree exactly.
    PNL_REQUIRE_MESSAGE(device.iterations == red_black.diagnostics.iterations,
                        "device red black took " + std::to_string(device.iterations) +
                            " iterations against the host's " +
                            std::to_string(red_black.diagnostics.iterations) +
                            ", so the two are not running the same method");
}

PNL_TEST("cuda/the bandwidth probe returns a plausible figure") {
    if (skip_without_gpu("device bandwidth probe")) return;
    const double gib = pnl_cuda_stream_triad(0, 256u * 1024u * 1024u, 3);
    PNL_REQUIRE_MESSAGE(gib > 0.0, std::string("the probe failed: ") + pnl_cuda_last_error());
    std::printf("        device triad: %.1f GiB/s\n", gib);
    // Wide enough to catch a broken probe without encoding this specific card:
    // no discrete GPU of this era is below 50 or above 10000 GiB per second.
    PNL_REQUIRE_MESSAGE(gib > 50.0 && gib < 10000.0,
                        "the device triad measured " + test::format(gib) +
                            " GiB/s, which is not a plausible figure");
}
