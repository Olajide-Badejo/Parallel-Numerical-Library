#pragma once

/// \file cuda.hpp
/// The CUDA boundary.
///
/// Everything here is `extern "C"` taking plain pointers and scalars. That is
/// not stylistic: nvcc 13.3 refuses GCC newer than 15 and cannot parse GCC 15's
/// libstdc++ headers either, so the `.cu` files are compiled by nvcc driving
/// g++-14 while the rest of the project is built by g++-16. The two never have
/// to agree on a C++ ABI, only on the platform C ABI. See ENV-01 in the
/// engineering log.
///
/// It also satisfies the Section 11 rule directly: no cudaStream_t, no
/// device pointer type, and no CUDA header appears in any interface the rest of
/// the library can see.
///
/// Scope. The GPU implements what Section 8.1 asks for and no more: Jacobi and
/// red black Gauss Seidel sweeps on the 2D Poisson problem, red black SOR since
/// it is the same kernel with a factor, conjugate gradient because it needs the
/// dot and norm reductions, and a STREAM triad probe for the device's own
/// achieved bandwidth. Natural ordering Gauss Seidel is absent because it has
/// no parallelism to offer a wide device, which is the point the comparison
/// makes rather than a gap in the implementation.

#include <cstddef>

extern "C" {

/// Methods the device solver implements. Values are part of the C boundary and
/// must not be renumbered.
enum PnlCudaMethod {
    PNL_CUDA_JACOBI = 0,
    PNL_CUDA_GAUSS_SEIDEL_RB = 1,
    PNL_CUDA_SOR_RB = 2,
    PNL_CUDA_CG = 3,
};

/// Outcome of a device solve, filled by pnl_cuda_poisson_solve.
struct PnlCudaResult {
    long iterations;
    int converged;
    double relative_residual;
    /// Seconds spent in kernels, excluding host to device transfer, so the
    /// bandwidth number describes the sweep and not the PCIe link.
    double kernel_seconds;
    /// Seconds spent transferring the problem in and the answer out. Reported
    /// separately and discussed in the comparison, because for a single solve
    /// it can dominate and for a production run it would not.
    double transfer_seconds;
    /// Bytes moved per unknown per sweep, counted by the kernel that ran.
    double bytes_per_unknown;
};

/// Number of CUDA capable devices, or 0 when there is none or the runtime is
/// unavailable. Never fails, so a build without a GPU can still run and report
/// that it skipped.
int pnl_cuda_device_count(void);

/// Describe a device. \p name receives at most \p name_capacity bytes.
/// \returns 0 on success, non zero on failure.
int pnl_cuda_device_info(int device, char* name, int name_capacity, int* compute_major,
                         int* compute_minor, size_t* total_bytes, int* multiprocessors);

/// The most recent error message from this translation unit, or an empty
/// string. Valid until the next call into the CUDA boundary.
const char* pnl_cuda_last_error(void);

/// Measure the device's achieved STREAM triad bandwidth, a[i] = b[i] + q c[i].
///
/// This is the denominator of every device normalised efficiency number in the
/// report. Section 7 requires it to be measured on this hardware rather than
/// quoted from a specification sheet, so this runs once per sweep session and
/// lands in the session manifest.
///
/// \param bytes_per_array size of each of the three arrays.
/// \param repeats timed repetitions; the best is returned, since the best is
///        the run least contaminated by anything else on the device.
/// \returns achieved GiB per second, or a negative value on failure.
double pnl_cuda_stream_triad(int device, size_t bytes_per_array, int repeats);

/// Solve the 2D Poisson problem on the device.
///
/// \param n interior points per side. The arrays are (n+2) by (n+2) row major
///        with a boundary ring, exactly the layout Poisson2D uses on the host,
///        so no repacking happens at the boundary.
/// \param rhs right hand side in that layout, host memory.
/// \param x initial guess in, solution out, host memory.
/// \param method one of PnlCudaMethod.
/// \param omega relaxation factor, used by PNL_CUDA_SOR_RB only.
/// \param tolerance relative residual target.
/// \param max_iterations iteration cap, or the exact count when
///        \p fixed_iterations is non zero.
/// \param check_interval iterations between residual evaluations.
/// \param fixed_iterations non zero to run exactly \p max_iterations sweeps
///        with no convergence test.
/// \param result filled on success.
/// \returns 0 on success, non zero on failure; call pnl_cuda_last_error.
int pnl_cuda_poisson_solve(int n, const double* rhs, double* x, int method, double omega,
                           double tolerance, long max_iterations, long check_interval,
                           int fixed_iterations, struct PnlCudaResult* result);

}  // extern "C"
