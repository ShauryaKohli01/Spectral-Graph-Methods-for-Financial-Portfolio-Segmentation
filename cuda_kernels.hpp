#pragma once
// Raw-pointer interface to the CUDA translation unit (kept free of Eigen so nvcc
// never has to parse it). Only linked when the project is built with ENABLE_CUDA=ON.
namespace hs::detail {

int cuda_device_count();  // 0 if no usable device

// C = Z * Z^T where Z is n x t, row-major, already standardised. C is n x n row-major.
// Throws std::runtime_error on any CUDA error.
// If kernel_ms is non-null it receives the kernel time measured with CUDA events.
void gram_cuda(const double* z, int n, int t, double* c, double* kernel_ms);

}  // namespace hs::detail
