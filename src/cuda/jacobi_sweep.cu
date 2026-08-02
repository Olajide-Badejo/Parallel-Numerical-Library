/// \file jacobi_sweep.cu
/// The device Jacobi sweep, the device solve driver, and the device query
/// entry points.
///
/// The whole iteration runs on the device. The grid goes across once at the
/// start and the answer comes back once at the end; nothing crosses PCIe per
/// sweep. That is what makes the bandwidth comparison of Section 8.3 a
/// statement about the two memory systems rather than about the interconnect,
/// and the transfer time is reported separately so a reader can add it back if
/// their use case is a single solve.

#include <pnl/backend/cuda.hpp>

#include "cuda_common.cuh"

#include <cmath>
#include <vector>

namespace {

using namespace pnl_cuda;

/// Sum \p n interior values of the elementwise product into \p partials, one
/// per block, with a shared memory tree inside each block.
///
/// The host then sums the partials in block order. The grouping is fixed by
/// REDUCE_BLOCKS rather than by the problem size, so the result is reproducible
/// across runs. It is still not bit identical to the host reduction, because a
/// tree inside a block associates differently from the host's ordered chunk
/// sum, and no compiler flag changes that. The CUDA tests therefore compare
/// sweeps exactly and reductions to tolerance, and say which is which.
__global__ void reduce_dot_kernel(const double* __restrict__ a, const double* __restrict__ b,
                                  int n, int stride, int side, double* __restrict__ partials) {
    __shared__ double scratch[REDUCE_BLOCK];
    const int tid = threadIdx.x;
    double sum = 0.0;

    for (int k = blockIdx.x * blockDim.x + tid; k < n; k += blockDim.x * gridDim.x) {
        const int i = k / side + 1;
        const int j = k % side + 1;
        const int index = i * stride + j;
        sum += a[index] * b[index];
    }
    scratch[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) scratch[tid] += scratch[tid + offset];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = scratch[0];
}

/// r = b - A x over the interior, five point stencil.
__global__ void residual_kernel(const double* __restrict__ x, const double* __restrict__ b,
                                double* __restrict__ r, int side, int stride) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > side || j > side) return;
    const int index = i * stride + j;
    r[index] = b[index] - (4.0 * x[index] - x[index - 1] - x[index + 1] - x[index - stride] -
                           x[index + stride]);
}

/// y = y + alpha x over the interior.
__global__ void axpy_kernel(double alpha, const double* __restrict__ x, double* __restrict__ y,
                            int side, int stride) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > side || j > side) return;
    const int index = i * stride + j;
    y[index] += alpha * x[index];
}

/// y = x + beta y over the interior.
__global__ void xpby_kernel(const double* __restrict__ x, double beta, double* __restrict__ y,
                            int side, int stride) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > side || j > side) return;
    const int index = i * stride + j;
    y[index] = x[index] + beta * y[index];
}

/// y = A x over the interior.
__global__ void apply_kernel(const double* __restrict__ x, double* __restrict__ y, int side,
                             int stride) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > side || j > side) return;
    const int index = i * stride + j;
    y[index] = 4.0 * x[index] - x[index - 1] - x[index + 1] - x[index - stride] -
               x[index + stride];
}

/// One Jacobi update over the interior.
///
/// The arithmetic is written in exactly the order the host sweep uses, and the
/// translation unit is compiled with fused multiply add contraction disabled,
/// so the device produces bit identical values to the CPU for this kernel. That
/// turns "the GPU agrees with the CPU" from a tolerance into an equality, which
/// is a much stronger statement about the port being faithful. The reductions
/// cannot make the same promise and do not claim it.
__global__ void jacobi_kernel(const double* __restrict__ x, const double* __restrict__ b,
                              double* __restrict__ out, int side, int stride) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > side || j > side) return;
    const int index = i * stride + j;
    out[index] = 0.25 * (b[index] + x[index - 1] + x[index + 1] + x[index - stride] +
                         x[index + stride]);
}

/// Sum the reduction partials on the host, in block order.
double combine(const std::vector<double>& partials) {
    double total = 0.0;
    for (double value : partials) total += value;
    return total;
}

}  // namespace

extern "C" {

int pnl_cuda_device_count(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
}

const char* pnl_cuda_last_error(void) { return pnl_cuda::last_error().c_str(); }

int pnl_cuda_device_info(int device, char* name, int name_capacity, int* compute_major,
                         int* compute_minor, size_t* total_bytes, int* multiprocessors) {
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device), 1);
    if (name != nullptr && name_capacity > 0) {
        std::snprintf(name, static_cast<size_t>(name_capacity), "%s", properties.name);
    }
    if (compute_major != nullptr) *compute_major = properties.major;
    if (compute_minor != nullptr) *compute_minor = properties.minor;
    if (total_bytes != nullptr) *total_bytes = properties.totalGlobalMem;
    if (multiprocessors != nullptr) *multiprocessors = properties.multiProcessorCount;
    return 0;
}

int pnl_cuda_poisson_solve(int n, const double* rhs, double* x, int method, double omega,
                           double tolerance, long max_iterations, long check_interval,
                           int fixed_iterations, struct PnlCudaResult* result) {
    if (n < 1 || rhs == nullptr || x == nullptr || result == nullptr) {
        pnl_cuda::last_error() = "pnl_cuda_poisson_solve received an invalid argument";
        return 1;
    }
    if (check_interval < 1) check_interval = 1;

    const int stride = n + 2;
    const size_t cells = static_cast<size_t>(stride) * static_cast<size_t>(stride);
    const size_t bytes = cells * sizeof(double);

    double* d_x = nullptr;
    double* d_b = nullptr;
    double* d_work = nullptr;
    double* d_r = nullptr;
    double* d_p = nullptr;
    double* d_ap = nullptr;
    double* d_partials = nullptr;

    auto release = [&] {
        cudaFree(d_x);
        cudaFree(d_b);
        cudaFree(d_work);
        cudaFree(d_r);
        cudaFree(d_p);
        cudaFree(d_ap);
        cudaFree(d_partials);
    };

    cudaEvent_t transfer_start, transfer_stop, kernel_start, kernel_stop;
    if (cudaEventCreate(&transfer_start) != cudaSuccess ||
        cudaEventCreate(&transfer_stop) != cudaSuccess ||
        cudaEventCreate(&kernel_start) != cudaSuccess ||
        cudaEventCreate(&kernel_stop) != cudaSuccess) {
        pnl_cuda::last_error() = "could not create CUDA events";
        return 1;
    }

    auto fail = [&](int code) {
        release();
        cudaEventDestroy(transfer_start);
        cudaEventDestroy(transfer_stop);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_stop);
        return code;
    };

#define CUDA_OR_FAIL(call)                                                       \
    do {                                                                         \
        const cudaError_t status = (call);                                       \
        if (status != cudaSuccess) {                                             \
            ::pnl_cuda::record_error(#call, status, __FILE__, __LINE__);          \
            return fail(1);                                                      \
        }                                                                        \
    } while (0)

    CUDA_OR_FAIL(cudaMalloc(&d_x, bytes));
    CUDA_OR_FAIL(cudaMalloc(&d_b, bytes));
    CUDA_OR_FAIL(cudaMalloc(&d_work, bytes));
    CUDA_OR_FAIL(cudaMalloc(&d_r, bytes));
    CUDA_OR_FAIL(cudaMalloc(&d_partials, REDUCE_BLOCKS * sizeof(double)));
    if (method == PNL_CUDA_CG) {
        CUDA_OR_FAIL(cudaMalloc(&d_p, bytes));
        CUDA_OR_FAIL(cudaMalloc(&d_ap, bytes));
    }

    CUDA_OR_FAIL(cudaEventRecord(transfer_start));
    CUDA_OR_FAIL(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
    CUDA_OR_FAIL(cudaMemcpy(d_b, rhs, bytes, cudaMemcpyHostToDevice));
    // The work buffer must start as a copy so its boundary ring carries the
    // Dirichlet values; the sweep never writes the ring. The source is the
    // device copy, not the host array.
    CUDA_OR_FAIL(cudaMemcpy(d_work, d_x, bytes, cudaMemcpyDeviceToDevice));
    CUDA_OR_FAIL(cudaMemset(d_r, 0, bytes));
    if (method == PNL_CUDA_CG) {
        CUDA_OR_FAIL(cudaMemset(d_p, 0, bytes));
        CUDA_OR_FAIL(cudaMemset(d_ap, 0, bytes));
    }
    CUDA_OR_FAIL(cudaEventRecord(transfer_stop));
    CUDA_OR_FAIL(cudaEventSynchronize(transfer_stop));

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((n + BLOCK_X - 1) / BLOCK_X, (n + BLOCK_Y - 1) / BLOCK_Y);
    const int interior = n * n;
    std::vector<double> partials(REDUCE_BLOCKS);

    auto dot = [&](const double* a, const double* b_vector, double* value) -> bool {
        reduce_dot_kernel<<<REDUCE_BLOCKS, REDUCE_BLOCK>>>(a, b_vector, interior, stride, n,
                                                           d_partials);
        if (cudaGetLastError() != cudaSuccess) return false;
        if (cudaMemcpy(partials.data(), d_partials, REDUCE_BLOCKS * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        *value = combine(partials);
        return true;
    };

    double rhs_norm = 0.0;
    if (!dot(d_b, d_b, &rhs_norm)) {
        pnl_cuda::last_error() = "reduction failed while measuring the right hand side norm";
        return fail(1);
    }
    rhs_norm = std::sqrt(rhs_norm);
    const double scale = rhs_norm > 0.0 ? rhs_norm : 1.0;

    long iterations = 0;
    double relative_residual = 0.0;
    int converged = 0;
    const bool to_tolerance = fixed_iterations == 0;

    auto measure_residual = [&](double* value) -> bool {
        residual_kernel<<<grid, block>>>(d_x, d_b, d_r, n, stride);
        if (cudaGetLastError() != cudaSuccess) return false;
        double square = 0.0;
        if (!dot(d_r, d_r, &square)) return false;
        *value = std::sqrt(square) / scale;
        return true;
    };

    CUDA_OR_FAIL(cudaEventRecord(kernel_start));

    if (method == PNL_CUDA_CG) {
        // r = b - A x, p = r.
        residual_kernel<<<grid, block>>>(d_x, d_b, d_r, n, stride);
        CUDA_OR_FAIL(cudaGetLastError());
        CUDA_OR_FAIL(cudaMemcpy(d_p, d_r, bytes, cudaMemcpyDeviceToDevice));

        double rr = 0.0;
        if (!dot(d_r, d_r, &rr)) return fail(1);
        relative_residual = std::sqrt(rr) / scale;

        for (; iterations < max_iterations; ++iterations) {
            if (to_tolerance && relative_residual <= tolerance) {
                converged = 1;
                break;
            }
            apply_kernel<<<grid, block>>>(d_p, d_ap, n, stride);
            CUDA_OR_FAIL(cudaGetLastError());

            double curvature = 0.0;
            if (!dot(d_p, d_ap, &curvature)) return fail(1);
            if (!(curvature > 0.0)) {
                pnl_cuda::last_error() =
                    "conjugate gradient found a non positive curvature on the device";
                return fail(2);
            }
            const double alpha = rr / curvature;

            axpy_kernel<<<grid, block>>>(alpha, d_p, d_x, n, stride);
            CUDA_OR_FAIL(cudaGetLastError());
            axpy_kernel<<<grid, block>>>(-alpha, d_ap, d_r, n, stride);
            CUDA_OR_FAIL(cudaGetLastError());

            double rr_next = 0.0;
            if (!dot(d_r, d_r, &rr_next)) return fail(1);
            relative_residual = std::sqrt(rr_next) / scale;

            const double beta = rr_next / rr;
            rr = rr_next;
            xpby_kernel<<<grid, block>>>(d_r, beta, d_p, n, stride);
            CUDA_OR_FAIL(cudaGetLastError());
        }
        if (to_tolerance && relative_residual <= tolerance) converged = 1;
    } else {
        if (!measure_residual(&relative_residual)) return fail(1);
        for (; iterations < max_iterations; ++iterations) {
            if (method == PNL_CUDA_JACOBI) {
                jacobi_kernel<<<grid, block>>>(d_x, d_b, d_work, n, stride);
                CUDA_OR_FAIL(cudaGetLastError());
                double* swap = d_x;
                d_x = d_work;
                d_work = swap;
            } else {
                const double factor = method == PNL_CUDA_SOR_RB ? omega : 1.0;
                pnl_cuda_launch_coloured(d_x, d_b, n, stride, factor, grid, block);
                CUDA_OR_FAIL(cudaGetLastError());
            }

            if ((iterations + 1) % check_interval == 0 ||
                iterations + 1 == max_iterations) {
                if (!measure_residual(&relative_residual)) return fail(1);
                if (to_tolerance && relative_residual <= tolerance) {
                    ++iterations;
                    converged = 1;
                    break;
                }
            }
        }
    }

    CUDA_OR_FAIL(cudaEventRecord(kernel_stop));
    CUDA_OR_FAIL(cudaEventSynchronize(kernel_stop));

    float transfer_ms = 0.0f;
    float kernel_ms = 0.0f;
    CUDA_OR_FAIL(cudaEventElapsedTime(&transfer_ms, transfer_start, transfer_stop));
    CUDA_OR_FAIL(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));

    cudaEvent_t back_start, back_stop;
    CUDA_OR_FAIL(cudaEventCreate(&back_start));
    CUDA_OR_FAIL(cudaEventCreate(&back_stop));
    CUDA_OR_FAIL(cudaEventRecord(back_start));
    CUDA_OR_FAIL(cudaMemcpy(x, d_x, bytes, cudaMemcpyDeviceToHost));
    CUDA_OR_FAIL(cudaEventRecord(back_stop));
    CUDA_OR_FAIL(cudaEventSynchronize(back_stop));
    float back_ms = 0.0f;
    CUDA_OR_FAIL(cudaEventElapsedTime(&back_ms, back_start, back_stop));
    cudaEventDestroy(back_start);
    cudaEventDestroy(back_stop);

    result->iterations = iterations;
    result->converged = converged;
    result->relative_residual = relative_residual;
    result->kernel_seconds = static_cast<double>(kernel_ms) / 1000.0;
    result->transfer_seconds = static_cast<double>(transfer_ms + back_ms) / 1000.0;
    // Counted, not estimated, and matching the host count in Poisson2D: one
    // write of the new iterate, one read of the right hand side, and one read
    // of the old iterate per unknown in the streaming limit.
    result->bytes_per_unknown = 3.0 * sizeof(double);

    // fail() also performs the cleanup, so the success path uses it with a zero
    // status rather than duplicating the frees.
    return fail(0);

#undef CUDA_OR_FAIL
}

}  // extern "C"
