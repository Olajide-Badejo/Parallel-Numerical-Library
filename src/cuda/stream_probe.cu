/// \file stream_probe.cu
/// The device's own achieved memory bandwidth.
///
/// Section 8.3 makes every efficiency number in the report a ratio of achieved
/// bandwidth to the device's measured peak, never to a figure from a
/// specification sheet. A specification number is a bus width times a clock and
/// no real kernel reaches it; dividing by it would make both devices look
/// equally inefficient and would tell the reader nothing about which one is
/// being used well.
///
/// The kernel is the triad of McCalpin's STREAM benchmark, a[i] = b[i] + q c[i],
/// which moves two reads and one write per element and has no reuse, so it
/// measures the memory system and nothing else.
///
/// Reference: McCalpin, "Memory Bandwidth and Machine Balance in Current High
/// Performance Computers", IEEE TCCA Newsletter, December 1995.

#include <pnl/backend/cuda.hpp>

#include "cuda_common.cuh"

#include <vector>

namespace {

__global__ void triad_kernel(double* __restrict__ a, const double* __restrict__ b,
                             const double* __restrict__ c, double q, size_t n) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        a[i] = b[i] + q * c[i];
    }
}

__global__ void fill_kernel(double* __restrict__ a, double value, size_t n) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        a[i] = value;
    }
}

}  // namespace

extern "C" double pnl_cuda_stream_triad(int device, size_t bytes_per_array, int repeats) {
    if (repeats < 1) repeats = 1;
    if (cudaSetDevice(device) != cudaSuccess) {
        pnl_cuda::last_error() = "cudaSetDevice failed in the bandwidth probe";
        return -1.0;
    }

    const size_t n = bytes_per_array / sizeof(double);
    if (n == 0) {
        pnl_cuda::last_error() = "the bandwidth probe was asked for a zero sized array";
        return -1.0;
    }

    double* a = nullptr;
    double* b = nullptr;
    double* c = nullptr;
    auto release = [&] {
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
    };

    if (cudaMalloc(&a, n * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&b, n * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&c, n * sizeof(double)) != cudaSuccess) {
        pnl_cuda::last_error() = "the bandwidth probe could not allocate its arrays";
        release();
        return -1.0;
    }

    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        release();
        return -1.0;
    }
    // Enough blocks to fill the device several times over, so the measurement
    // is not limited by occupancy.
    const int threads = 256;
    const int blocks = properties.multiProcessorCount * 32;

    fill_kernel<<<blocks, threads>>>(b, 1.0, n);
    fill_kernel<<<blocks, threads>>>(c, 2.0, n);
    fill_kernel<<<blocks, threads>>>(a, 0.0, n);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        pnl_cuda::last_error() = "the bandwidth probe failed while initialising";
        release();
        return -1.0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // One untimed pass so the first timed one is not paying for cold caches or
    // a clock still ramping up.
    triad_kernel<<<blocks, threads>>>(a, b, c, 3.0, n);
    cudaDeviceSynchronize();

    double best_gib = 0.0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        cudaEventRecord(start);
        triad_kernel<<<blocks, threads>>>(a, b, c, 3.0, n);
        cudaEventRecord(stop);
        if (cudaEventSynchronize(stop) != cudaSuccess) {
            pnl_cuda::last_error() = "the bandwidth probe failed during timing";
            release();
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return -1.0;
        }
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (milliseconds > 0.0f) {
            // Two reads and one write per element.
            const double moved = 3.0 * static_cast<double>(n) * sizeof(double);
            const double gib =
                moved / (static_cast<double>(milliseconds) / 1000.0) / (1024.0 * 1024.0 * 1024.0);
            if (gib > best_gib) best_gib = gib;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    release();
    return best_gib;
}
