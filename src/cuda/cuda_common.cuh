// SPDX-License-Identifier: MIT
#pragma once

/// \file cuda_common.cuh
/// Error handling, launch geometry, and the cross file kernel declarations.
///
/// Kernels themselves are deliberately not defined here. A `__global__`
/// function in a header is emitted into every translation unit that includes
/// it, and the resulting duplicate device stubs fail the link with a multiple
/// definition error that names a mangled `__device_stub__` symbol and gives no
/// hint that a header is the cause. Each kernel therefore lives in exactly one
/// .cu file, and anything needed across files is reached through a plain
/// launcher function declared here.

#include <cstdio>
#include <string>

#include <cuda_runtime.h>

namespace pnl_cuda {

/// Last error message, exposed to the host through pnl_cuda_last_error.
inline std::string& last_error() {
    static std::string message;
    return message;
}

inline void record_error(const char* call, cudaError_t status, const char* file, int line) {
    last_error() = std::string(call) + " failed at " + file + ":" + std::to_string(line) + ": " +
                   cudaGetErrorString(status);
}

/// Check a CUDA call and return \p failure_value from the enclosing function on
/// error, after recording a message the host side can retrieve.
///
/// Prefixed for the reason the MPI one is: a macro is not scoped by a
/// namespace, and the unprefixed spelling is one of the most widely defined
/// names in CUDA code. Section 4.7.
#define PNL_CUDA_CHECK(call, failure_value)                                       \
    do {                                                                          \
        const cudaError_t pnl_cuda_status = (call);                               \
        if (pnl_cuda_status != cudaSuccess) {                                     \
            ::pnl_cuda::record_error(#call, pnl_cuda_status, __FILE__, __LINE__); \
            return (failure_value);                                               \
        }                                                                         \
    } while (0)

/// Threads per block for the two dimensional stencil kernels. 32 by 8 gives
/// fully coalesced 256 byte loads along a row, which is what a bandwidth bound
/// stencil needs, and 256 threads per block leaves plenty of blocks resident.
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

/// Threads per block for the reduction.
constexpr int REDUCE_BLOCK = 256;

/// Blocks used by the reduction. Fixed rather than derived from the problem
/// size, so the number of partials, and therefore the order in which the host
/// combines them, does not change with n and a device reduction is reproducible
/// across runs.
constexpr int REDUCE_BLOCKS = 512;

}  // namespace pnl_cuda

namespace pnl_cuda::detail {

/// Launch a full red black relaxation step: the red half sweep, then the black
/// one. Defined in rb_gauss_seidel.cu.
///
/// The two half sweeps are separate kernel launches on purpose. Each launch is
/// a device wide synchronisation point, and that is exactly what the method
/// needs: every red cell must be updated before any black cell reads it.
/// Fusing them would need a grid wide barrier and would change the method.
///
/// In a namespace rather than at global scope, because it is an internal helper
/// with external linkage: jacobi_sweep.cu calls it, so it cannot be static, and
/// a plain global name in a translation unit that a consumer's build links is
/// exactly the kind of collision Section 4.7 asks this phase to remove. The
/// namespace is pnl_cuda rather than pnl::cuda because these files are compiled
/// by a different compiler across a C ABI boundary and share no type with the
/// host library; the extern "C" entry points of pnl/backend/cuda.hpp are the
/// only names that cross, and decision 8 says they stay exactly as they are.
void launch_coloured(
    double* x, const double* b, int side, int stride, double relaxation, dim3 grid, dim3 block);

}  // namespace pnl_cuda::detail
