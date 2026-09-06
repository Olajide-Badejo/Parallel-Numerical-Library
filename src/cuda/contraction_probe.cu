// SPDX-License-Identifier: MIT
/// \file contraction_probe.cu
/// The device half of ground rule 8: a probe that fails when this build fused a
/// multiply and an add on the GPU.
///
/// `pnl_cuda` is compiled with `--fmad=false`, and that flag is the reason the
/// device sweeps can be asserted **bit identical** to the host sweeps rather
/// than merely close: the SOR update has the shape `a*b + c*d`, the host has
/// `-ffp-contract=off` and does not fuse it, and without `--fmad=false` the
/// device would. Until this file existed, nothing in the tree could tell whether
/// the flag was there. Remove `--fmad=false` from `CMakeLists.txt` and the whole
/// suite stays green, because the device tests that compare against the host all
/// run on kernels simple enough that nvcc happens not to contract, and the one
/// that would notice was never written. That is NUM-05 on the device side, and
/// it is the same hole `assert_no_contraction` closed on the host.
///
/// **The operands arrive from the host and are not compile time constants, and
/// that is load bearing rather than stylistic.** A probe whose inputs were
/// literals in this file would be folded by nvcc at compile time with correct
/// rounding, both builds would agree on the unfused answer, and the fused build
/// would fail its own assertion for a reason that has nothing to do with
/// contraction. The host passes `CONTRACTION_PROBE_A`, `_B` and `_C` from
/// `pnl/core/contract.hpp`, which is also what keeps one definition of those
/// four numbers in the tree rather than a second spelling here.
///
/// The kernel is one thread. There is nothing to parallelise: the question is
/// what instruction the compiler emitted, not how fast the device runs it.
///
/// **This file is compiled twice.** Once into `pnl_cuda` with the project's
/// `--fmad=false`, and once into `pnl_cuda_fused` with `--fmad=true`, which is a
/// separate small target that nothing else links. The second is what proves the
/// first can tell: a probe that always returned the unfused value would pass
/// just as happily with an empty body. The two copies are told apart by the two
/// macros below, which the fused target overrides on its compile line, because
/// the alternative is two symbols with the same name in one link.

#ifndef PNL_PROBE_ENTRY
#define PNL_PROBE_ENTRY pnl_cuda_contraction_probe
#endif

#ifndef PNL_PROBE_KERNEL
#define PNL_PROBE_KERNEL pnl_cuda_contraction_probe_kernel
#endif

#include "cuda_common.cuh"

/// Compute `a * b + c` on the device, in one thread, and write it to \p out.
///
/// Not in an anonymous namespace: a `__global__` function with internal linkage
/// is legal but its device stub is easier to reason about with a name, and the
/// name is macro generated precisely so the two builds of this file do not
/// collide.
__global__ void PNL_PROBE_KERNEL(double a, double b, double c, double* out) {
    *out = a * b + c;
}

/// Run the probe on the current device and return what it computed.
///
/// \param out receives `a * b + c` as the device evaluated it. Untouched on
///        failure.
/// \returns 0 on success, or a non zero status after recording a message that
///          `pnl_cuda_last_error()` returns.
extern "C" int PNL_PROBE_ENTRY(double a, double b, double c, double* out) {
    if (out == nullptr) {
        pnl_cuda::last_error() = "the contraction probe was given a null result pointer";
        return 1;
    }

    double* device_out = nullptr;
    PNL_CUDA_CHECK(cudaMalloc(&device_out, sizeof(double)), 2);

    PNL_PROBE_KERNEL<<<1, 1>>>(a, b, c, device_out);
    const cudaError_t launched = cudaGetLastError();
    if (launched != cudaSuccess) {
        pnl_cuda::record_launch_error("the contraction probe kernel", launched, __FILE__, __LINE__);
        (void)cudaFree(device_out);
        return 3;
    }

    const cudaError_t ran = cudaDeviceSynchronize();
    if (ran != cudaSuccess) {
        pnl_cuda::record_error("cudaDeviceSynchronize", ran, __FILE__, __LINE__);
        (void)cudaFree(device_out);
        return 4;
    }

    double host_out = 0.0;
    const cudaError_t copied =
        cudaMemcpy(&host_out, device_out, sizeof(double), cudaMemcpyDeviceToHost);
    (void)cudaFree(device_out);
    if (copied != cudaSuccess) {
        pnl_cuda::record_error("cudaMemcpy", copied, __FILE__, __LINE__);
        return 5;
    }

    *out = host_out;
    return 0;
}
