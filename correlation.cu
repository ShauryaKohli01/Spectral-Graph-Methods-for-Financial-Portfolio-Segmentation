// Tiled Gram-matrix kernel: C = Z * Z^T for a row-major, standardised Z (n x t).
// With rows scaled to zero mean / unit norm, C is the Pearson correlation matrix.
#include "cuda_kernels.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace hs::detail {
namespace {

constexpr int TILE = 16;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Each block computes a TILE x TILE patch of C. Threads cooperatively stage a
// TILE-wide slice of the time axis for TILE rows of Z ("A" tile) and for TILE other
// rows of Z ("B" tile) in shared memory, so each global element is loaded once per
// block instead of once per output. The +1 padding avoids shared-memory bank conflicts
// when reading b_tile[threadIdx.x][j] (stride TILE would map a half-warp to one bank).
// Out-of-range loads are zero-filled, which contributes nothing to the dot product.
__global__ void gram_kernel(const double* __restrict__ z, double* __restrict__ c, int n, int t) {
    __shared__ double a_tile[TILE][TILE + 1];
    __shared__ double b_tile[TILE][TILE + 1];

    const int row = blockIdx.y * TILE + threadIdx.y;    // asset i
    const int col = blockIdx.x * TILE + threadIdx.x;    // asset j
    const int b_row = blockIdx.x * TILE + threadIdx.y;  // asset staged for the B tile

    double acc = 0.0;
    for (int t0 = 0; t0 < t; t0 += TILE) {
        const int k = t0 + threadIdx.x;  // time index this thread loads (coalesced)
        a_tile[threadIdx.y][threadIdx.x] =
            (row < n && k < t) ? z[static_cast<size_t>(row) * t + k] : 0.0;
        b_tile[threadIdx.y][threadIdx.x] =
            (b_row < n && k < t) ? z[static_cast<size_t>(b_row) * t + k] : 0.0;
        __syncthreads();

        // a_tile[ty][j] = Z[row][t0+j];  b_tile[tx][j] = Z[blockIdx.x*TILE+tx][t0+j] = Z[col][t0+j]
        #pragma unroll
        for (int j = 0; j < TILE; ++j) acc += a_tile[threadIdx.y][j] * b_tile[threadIdx.x][j];
        __syncthreads();  // do not overwrite the tiles while others are still reading
    }
    if (row < n && col < n) c[static_cast<size_t>(row) * n + col] = acc;
}

}  // namespace

int cuda_device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();  // clear the error so it does not leak into later checks
        return 0;
    }
    return count;
}

void gram_cuda(const double* z, int n, int t, double* c, double* kernel_ms) {
    const size_t z_bytes = sizeof(double) * static_cast<size_t>(n) * t;
    const size_t c_bytes = sizeof(double) * static_cast<size_t>(n) * n;
    double *d_z = nullptr, *d_c = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    auto release = [&] {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        cudaFree(d_z);  // cudaFree(nullptr) is a no-op
        cudaFree(d_c);
    };
    try {
        check(cudaMalloc(&d_z, z_bytes), "cudaMalloc(Z)");
        check(cudaMalloc(&d_c, c_bytes), "cudaMalloc(C)");
        check(cudaEventCreate(&start), "cudaEventCreate");
        check(cudaEventCreate(&stop), "cudaEventCreate");
        check(cudaMemcpy(d_z, z, z_bytes, cudaMemcpyHostToDevice), "H2D copy");

        const dim3 block(TILE, TILE);
        const dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        check(cudaEventRecord(start), "cudaEventRecord");
        gram_kernel<<<grid, block>>>(d_z, d_c, n, t);
        check(cudaGetLastError(), "kernel launch");       // launch-configuration errors
        check(cudaEventRecord(stop), "cudaEventRecord");
        check(cudaEventSynchronize(stop), "kernel execution");  // asynchronous execution errors
        if (kernel_ms) {
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
            *kernel_ms = static_cast<double>(ms);
        }
        check(cudaMemcpy(c, d_c, c_bytes, cudaMemcpyDeviceToHost), "D2H copy");
    } catch (...) {
        release();
        throw;
    }
    release();
}

}  // namespace hs::detail
