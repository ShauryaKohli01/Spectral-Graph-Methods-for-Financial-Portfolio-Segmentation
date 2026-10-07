#pragma once
#include "data.hpp"

namespace hs {

// Centre each row (two-pass mean) and scale it to unit Euclidean norm, so that
// Z * Z^T is exactly the Pearson correlation matrix. Rows with (numerically) zero
// variance are set to zero. Input: assets x observations. Throws std::invalid_argument
// on fewer than 2 observations or any non-finite value.
Matrix standardize_rows(const Matrix& returns);

// Pearson correlation of the rows of `returns` (assets x observations). The result is
// symmetric, clamped to [-1, 1] and has a unit diagonal. A constant series has no
// defined correlation; by convention it gets 0 against every other asset.
Matrix correlation_cpu(const Matrix& returns);

// Same maths, with the O(N^2 T) product computed by a hand-written tiled CUDA kernel.
// Throws std::runtime_error if built without CUDA or if no device is present.
// If kernel_ms is non-null it receives the device-side kernel time (CUDA events), which
// excludes host standardisation and PCIe transfers; the call itself is end-to-end.
Matrix correlation_gpu(const Matrix& returns, double* kernel_ms = nullptr);

bool cuda_available();  // false if compiled without CUDA or no device found

}  // namespace hs
