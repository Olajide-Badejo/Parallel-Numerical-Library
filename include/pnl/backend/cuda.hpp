// SPDX-License-Identifier: MIT
#pragma once

/// \file cuda.hpp
/// The CUDA boundary.
///
/// Everything here is `extern "C"` taking plain pointers and scalars. That is
/// not stylistic: nvcc 13.3 refuses GCC newer than 15 and cannot parse GCC 15's
/// libstdc++ headers either, so the `.cu` files are compiled by nvcc driving
/// g++-14 while the rest of the project is built by g++-15. The two never have
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

/// Limits of the device solver, part of the C boundary.
enum PnlCudaLimits {
    /// Largest interior side per dimension the device solver accepts.
    ///
    /// The kernels address a padded (n+2) by (n+2) grid as `i * stride + j`
    /// with 32 bit integers, and the largest index that arithmetic produces is
    /// `n * (n + 2) + n`. The largest n for which that stays inside a signed 32
    /// bit integer is 46338. Above it the index wraps, which is undefined
    /// behaviour rather than a large number, and the interior count `n * n` in
    /// the solve driver wraps a little later for the same reason. Section 4.7
    /// names both.
    ///
    /// The bound is checked at the entry point rather than worked around by
    /// widening the kernels. Widening would put a 64 bit division into the
    /// reduction's per element loop, which is measured, to reach sizes no
    /// device can hold: a grid at this bound is 17 GB for each of the five
    /// arrays a solve allocates. The check costs one comparison and makes every
    /// index below it provably in range.
    PNL_CUDA_MAX_SIDE = 46338,
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
int pnl_cuda_device_info(int device,
                         char* name,
                         int name_capacity,
                         int* compute_major,
                         int* compute_minor,
                         size_t* total_bytes,
                         int* multiprocessors);

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
/// \param n interior points per side, at least 1 and at most PNL_CUDA_MAX_SIDE.
///        The arrays are (n+2) by (n+2) row major with a boundary ring, exactly
///        the layout Poisson2D uses on the host, so no repacking happens at the
///        boundary.
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
int pnl_cuda_poisson_solve(int n,
                           const double* rhs,
                           double* x,
                           int method,
                           double omega,
                           double tolerance,
                           long max_iterations,
                           long check_interval,
                           int fixed_iterations,
                           struct PnlCudaResult* result);

/// Launch the red black half sweeps with the block geometry given, so that a
/// test can hand the driver one it will reject.
///
/// This exists because a launch configuration error and an execution fault
/// arrive at different times and used to read the same way, and the only honest
/// way to test the check that tells them apart is to cause one. It allocates a
/// small grid, launches, frees, and reports what the launch itself said.
///
/// \param block_x threads per block in x.
/// \param block_y threads per block in y. Their product above the device limit
///        is what makes the driver refuse the launch.
/// \returns 0 when the launch was accepted, which for a deliberately bad
///          geometry is the failure this probe exists to catch, and non zero
///          when it was rejected; call pnl_cuda_last_error for the wording.
int pnl_cuda_probe_launch_geometry(int block_x, int block_y);

}  // extern "C"
