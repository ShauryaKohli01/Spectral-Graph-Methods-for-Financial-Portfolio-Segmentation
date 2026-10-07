#include "correlation.hpp"

#include <cmath>
#include <stdexcept>

#ifdef HS_WITH_CUDA
#include "cuda_kernels.hpp"
#endif

namespace hs {

Matrix standardize_rows(const Matrix& r) {
    const Eigen::Index n = r.rows(), t = r.cols();
    if (t < 2) throw std::invalid_argument("correlation needs at least 2 observations");
    if (!r.allFinite()) throw std::invalid_argument("returns contain NaN or Inf");

    Matrix z(n, t);
    for (Eigen::Index i = 0; i < n; ++i) {
        // Two-pass: subtract the mean first, then sum squares of the centred data.
        // The one-pass E[x^2] - E[x]^2 formula cancels catastrophically when |mean| >> std.
        const double mean = r.row(i).mean();
        z.row(i) = r.row(i).array() - mean;
        const double norm = z.row(i).norm();
        const double scale = r.row(i).cwiseAbs().maxCoeff();
        if (norm <= 1e-12 * scale * std::sqrt(static_cast<double>(t)) || norm == 0.0)
            z.row(i).setZero();  // constant series
        else
            z.row(i) /= norm;
    }
    return z;
}

namespace {
// Shared post-processing so CPU and GPU paths return identical conventions.
void finalize_correlation(Matrix& c) {
    c = (0.5 * (c + c.transpose())).eval();                   // remove rounding asymmetry
    c = c.cwiseMax(-1.0).cwiseMin(1.0).eval();                // |rho| <= 1 despite rounding
    c.diagonal().setOnes();
}
}  // namespace

Matrix correlation_cpu(const Matrix& returns) {
    const Matrix z = standardize_rows(returns);
    Matrix c = z * z.transpose();
    finalize_correlation(c);
    return c;
}

#ifdef HS_WITH_CUDA
bool cuda_available() { return detail::cuda_device_count() > 0; }

Matrix correlation_gpu(const Matrix& returns, double* kernel_ms) {
    if (!cuda_available()) throw std::runtime_error("no CUDA device available");
    using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const RowMajor z = standardize_rows(returns);  // kernel wants each asset's series contiguous
    RowMajor c(z.rows(), z.rows());
    detail::gram_cuda(z.data(), static_cast<int>(z.rows()), static_cast<int>(z.cols()), c.data(), kernel_ms);
    Matrix out = c;
    finalize_correlation(out);
    return out;
}
#else
bool cuda_available() { return false; }

Matrix correlation_gpu(const Matrix&, double*) {
    throw std::runtime_error("built without CUDA; reconfigure with -DENABLE_CUDA=ON");
}
#endif

}  // namespace hs
