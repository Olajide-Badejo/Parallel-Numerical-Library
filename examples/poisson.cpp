// SPDX-License-Identifier: MIT
/// \file poisson.cpp
/// Solve a 2D Poisson problem with conjugate gradient, against an installed pnl.
///
/// This is the install test as much as it is an example. The project around it
/// declares LANGUAGES CXX and nothing else, and it is built from a staging
/// prefix through find_package, so it fails on the day the exported package
/// starts demanding a language, a flag or a dependency that a plain C++
/// consumer does not have. Every header it includes is public.

#include <pnl/backend/backend.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/cg.hpp>
#include <pnl/version.hpp>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

int main() {
    std::printf("pnl %s\n", pnl::VERSION_STRING);

    // Ask the library what it was built with rather than assuming. An install
    // configured without OpenMP still runs this example, on the serial backend,
    // and says so.
    const std::vector<std::string> backends = pnl::backend::available_backends();
    const bool parallel = std::find(backends.begin(), backends.end(), "openmp") != backends.end();
    const std::string name = parallel ? "openmp" : "serial";

    pnl::backend::Config config;
    config.workers = parallel ? 4 : 1;
    const std::unique_ptr<pnl::backend::Backend> backend = pnl::backend::make_backend(name, config);

    // 127 interior points per side, so 16129 unknowns: large enough that the
    // iteration count means something, small enough to finish in a moment.
    // SpectrallyRich rather than the manufactured sine, because that source is
    // an eigenvector of the operator and conjugate gradient solves it in one
    // step, which would flatter the method rather than exercise it.
    pnl::problems::Poisson2D problem(127, pnl::problems::PoissonRhs::SpectrallyRich);

    pnl::solvers::SolverOptions options;
    options.tolerance = 1.0e-10;
    options.max_iterations = 5000;

    const pnl::solvers::ConjugateGradient solver;
    const pnl::SolveResult result = solver.solve(problem, *backend, options);

    std::printf("backend           %s\n", name.c_str());
    std::printf("workers           %d\n", config.workers);
    std::printf("unknowns          %td\n", problem.unknown_count());
    std::printf("iterations        %td\n", result.diagnostics.iterations);
    std::printf("relative residual %.3e\n", result.diagnostics.error_estimate);
    return result.converged() ? 0 : 1;
}
