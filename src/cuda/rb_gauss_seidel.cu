// SPDX-License-Identifier: MIT
/// \file rb_gauss_seidel.cu
/// Red black Gauss Seidel and red black SOR half sweeps.
///
/// Natural ordering Gauss Seidel cannot run here at all: unknown k reads
/// unknown k-1 of the same sweep, so there is no partition of the grid that a
/// wide device could exploit. Red black colouring is what makes a Gauss Seidel
/// style method available to a GPU, and the price is a different iteration
/// matrix and therefore a different, generally slightly larger, iteration count.
/// The comparison in Section 8.3 pairs GPU red black against CPU red black so
/// that the ordering penalty is charged to both sides equally, and reports the
/// penalty against natural ordering separately.
///
/// Reference: Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed.,
/// SIAM 2003, section 12.4.

#include <pnl/backend/cuda.hpp>

#include "cuda_common.cuh"

namespace {

/// One half sweep over the cells whose colour matches \p parity.
///
/// Thread (tx, ty) handles the tx-th cell of that colour in row ty + 1. Mapping
/// threads to cells of one colour rather than to all cells and masking means no
/// thread in the launch is idle, which matters because a masked version would
/// waste half of every warp.
///
/// The stride two access along a row costs coalescing: consecutive threads
/// touch alternate doubles, so a 32 thread warp pulls twice the cache lines it
/// strictly needs. That is inherent to red black on a row major grid and is one
/// of the reasons the measured red black bandwidth sits below the Jacobi
/// figure, which the report quantifies rather than glosses.
__global__ void coloured_kernel(double* __restrict__ x,
                                const double* __restrict__ b,
                                int side,
                                int stride,
                                double relaxation,
                                int parity) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > side) return;

    // First column of this colour in this row, matching the host's
    // 1 + ((i + 1 + parity) & 1).
    const int first = 1 + ((i + 1 + parity) & 1);
    const int count = (side - first) / 2 + 1;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;

    const int j = first + 2 * k;
    const int index = i * stride + j;
    const double neighbours =
        b[index] + x[index - 1] + x[index + 1] + x[index - stride] + x[index + stride];
    x[index] = (1.0 - relaxation) * x[index] + 0.25 * relaxation * neighbours;
}

}  // namespace

namespace pnl_cuda::detail {

cudaError_t launch_coloured(
    double* x, const double* b, int side, int stride, double relaxation, dim3 grid, dim3 block) {
    // Half as many columns per row, so half the blocks in x.
    const int columns = (side + 1) / 2;
    const dim3 coloured_grid((columns + static_cast<int>(block.x) - 1) / static_cast<int>(block.x),
                             grid.y);

    // Red first, then black. The launch boundary between them is the device
    // wide synchronisation the method requires; there is no cheaper way to get
    // it and no correct way to avoid it.
    //
    // Each launch is checked immediately, which is the only moment a launch
    // configuration error exists: cudaGetLastError returns it before any thread
    // has run, and if nothing asks, the next call to return anything reports it
    // instead, as though a kernel had faulted. Checking here also names which
    // of the two half sweeps was refused, which the single check at the call
    // site could not.
    coloured_kernel<<<coloured_grid, block>>>(x, b, side, stride, relaxation, 0);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        record_launch_error("the red half sweep of launch_coloured", status, __FILE__, __LINE__);
        return status;
    }

    coloured_kernel<<<coloured_grid, block>>>(x, b, side, stride, relaxation, 1);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        record_launch_error("the black half sweep of launch_coloured", status, __FILE__, __LINE__);
    }
    return status;
}

}  // namespace pnl_cuda::detail

extern "C" {

int pnl_cuda_probe_launch_geometry(int block_x, int block_y) {
    // A grid small enough to be free and large enough to be a real launch.
    constexpr int SIDE = 8;
    constexpr int STRIDE = SIDE + 2;
    constexpr std::size_t CELLS = static_cast<std::size_t>(STRIDE) * STRIDE;

    // A failed allocation returns the same non zero this reports a refused
    // launch with, and records its own message. The caller checks the wording,
    // so a probe that could not run reads as a probe that could not run rather
    // than as a launch the driver accepted.
    double* x = nullptr;
    double* b = nullptr;
    PNL_CUDA_CHECK(cudaMalloc(&x, CELLS * sizeof(double)), 1);
    if (cudaMalloc(&b, CELLS * sizeof(double)) != cudaSuccess) {
        pnl_cuda::record_error("cudaMalloc", cudaGetLastError(), __FILE__, __LINE__);
        cudaFree(x);
        return 1;
    }
    cudaMemset(x, 0, CELLS * sizeof(double));
    cudaMemset(b, 0, CELLS * sizeof(double));

    const dim3 block(static_cast<unsigned>(block_x), static_cast<unsigned>(block_y));
    const dim3 grid(1, 1);
    const cudaError_t status =
        pnl_cuda::detail::launch_coloured(x, b, SIDE, STRIDE, 1.0, grid, block);

    cudaFree(x);
    cudaFree(b);
    // Whatever the launch left behind, so a rejected launch here cannot be
    // reported against the next kernel in the process.
    (void)cudaGetLastError();
    return status == cudaSuccess ? 0 : 1;
}

}  // extern "C"
