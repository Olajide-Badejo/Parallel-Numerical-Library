# Parallel Numerical Library

**Twelve iterative linear solvers, written once, running unchanged over seven
execution backends: serial, OpenMP, POSIX threads, `std::jthread`, MPI, hybrid
MPI with threads, and CUDA.**

[![ci](https://github.com/Olajide-Badejo/Parallel-Numerical-Library/actions/workflows/ci.yml/badge.svg)](https://github.com/Olajide-Badejo/Parallel-Numerical-Library/actions/workflows/ci.yml)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-13.3-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this project answers

The interesting question about parallel programming is not "is OpenMP fast." It
is **what does each programming model actually cost on identical work.** A study
that reimplements an algorithm per model measures the care that went into each
reimplementation. This library removes that variable: a solver never knows its
backend, and a backend never knows its solver.

> **Every shared memory backend, at every worker count, produces bit identical
> iterates. The GPU sweeps are bit identical to the CPU as well.**

Not "close," not "within tolerance," but identical, asserted as exact equality by
the test suite. Floating point addition is not associative, so a reduction's
answer normally depends on how many threads computed it; here the chunk grid is
fixed by the problem size alone. Two limits are documented rather than hidden:
across MPI **rank** counts results agree only to reduction tolerance, and GPU
**reductions** use a tree, which is not an ordered sum. What that invariant
costs is measured rather than excused.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/figures/scaling_speedup-dark.png">
  <img alt="Speedup against worker count for three thread backends" src="assets/figures/scaling_speedup-light.png" width="720">
</picture>

Every figure and every number in the reports is generated from
[`experiments/results/summary.csv`](experiments/results/summary.csv) by script.
Read [the comparison methodology](docs/comparison_methodology.md) before quoting
any of it: it is canonical for how the comparison is built and for what each
measurement licenses, and it is where the honest framing lives.

## Supported platforms

Linux on x86-64 is the supported platform and the only measured one; the target
machine runs Ubuntu under WSL2, which is also the only way Windows is supported.
macOS builds and runs without CUDA and without pinning: it has neither the POSIX
affinity call nor the sysfs topology tree, so every policy except `none` reports
`not_applicable` rather than binding, and no number here was measured there.
MSVC configures and gets the equivalent contract and warning flags; it is untested.

The toolchain is GCC 15.2.0 with CUDA 13.3 behind GCC 14, CMake and Ninja,
OpenMPI, and Python 3.11 or newer for the scripts and the Python gates; the
version of every tool is in [PROGRESS.md](PROGRESS.md).

## Install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cmake --install build --prefix /your/prefix
```

```cmake
find_package(pnl REQUIRED CONFIG)          # against that prefix
target_link_libraries(your_target PRIVATE pnl::core)
```

```cmake
include(FetchContent)                      # or from source, with no install
FetchContent_Declare(pnl
  GIT_REPOSITORY https://github.com/Olajide-Badejo/Parallel-Numerical-Library.git
  GIT_TAG        v1.1.0)
FetchContent_MakeAvailable(pnl)
target_link_libraries(your_target PRIVATE pnl::core)
```

`pnl::core` carries `-ffp-contract=off` and C++20, which a consumer has to
compile with for the bit identity claim to hold of their own translation units,
and carries no architecture flag and no warnings as errors policy.

## Quick start

```cpp
#include <pnl/backend/backend.hpp>
#include <pnl/problems/poisson2d.hpp>
#include <pnl/solvers/cg.hpp>

int main() {
    pnl::backend::Config config;
    config.workers = 4;
    const auto backend = pnl::backend::make_backend("openmp", config);
    // SpectrallyRich, not the manufactured sine: that source is an eigenvector
    // of the operator and conjugate gradient would solve it in one step.
    pnl::problems::Poisson2D problem(127, pnl::problems::PoissonRhs::SpectrallyRich);
    pnl::solvers::SolverOptions options;
    options.tolerance = 1.0e-10;
    const pnl::SolveResult result =
        pnl::solvers::ConjugateGradient().solve(problem, *backend, options);
    return result.converged() ? 0 : 1;
}
```

[`examples/poisson.cpp`](examples/poisson.cpp) is the full program and asks
`available_backends()` rather than assuming OpenMP is there;
[`examples/custom_backend.cpp`](examples/custom_backend.cpp) registers a backend
of your own and shows its iterate is bit identical to the serial one. From a
clone, `make setup` checks the toolchain, `make build` compiles, `make test` runs
every gate, `make install-test` stages an install and builds `examples/` against
it, and `make all` does everything including the sweep and the reports.

## What continuous integration proves

Every push and every pull request builds the library on a pinned `ubuntu-24.04`
image with gcc-14, gcc-15 and clang-18, in Debug and in Release, with warnings as
errors, and runs the unit, convergence, equivalence and style gates in all six
legs. MPI agreement is checked at one, two and four ranks, and a deliberately
failing rank has to end the job rather than hang it. Separate jobs run the
address and undefined behaviour sanitizers, the thread sanitizer over the
equivalence label at one, two, four and eight workers with OpenMP and MPI off, a
build with OpenMP and MPI switched off, a staged install consumed by
`examples/`, and a regeneration of every report asset from the committed summary
whose rebuilt PDFs are compared against the tracked ones under `assets/reports/`
numeric token by numeric token.

**There is no GPU runner.** The CUDA job compiles the device code and runs the
device test binary, which finds no device and skips its device cases with a
printed reason. Every device number in the reports was measured on the target
machine, and the last local run of `test_cuda` with a device present was
2026-09-06. A green badge therefore says that the device code compiles, not that
it was executed.

**End to end wall clock for `make clean && make all`: 2329 seconds**, of which
the sweep session and its bandwidth refresh are 2223. Both were measured on the
target machine in the publication session; how the figure is composed is in
[PROGRESS.md](PROGRESS.md).

## Where everything is

| | |
| --- | --- |
| [Main report](assets/reports/main_report.pdf), [engineering report](assets/reports/debug_report.pdf) | The measurements and the conclusions; every fault behind them |
| [comparison_methodology.md](docs/comparison_methodology.md) | Canonical: how the comparison is built and how to read it |
| [RELATED_WORK.md](docs/RELATED_WORK.md) | The prior art, and what this project adds beyond it |
| [solvers.md](docs/solvers.md), [backends.md](docs/backends.md) | The solver family tree; the backend interface contract |
| [DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md), [ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md) | The choices, and the faults behind them |
| [PROGRESS.md](PROGRESS.md), [CONTRIBUTING.md](CONTRIBUTING.md) | Phase by phase with each gate's output; the public API and how to build |

## Licence

MIT. Copyright 2026 Olajide Badejo. See [LICENSE](LICENSE).
