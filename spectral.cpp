#include "spectral.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace hs {

Eigendecomposition eigh(const Matrix& a) {
    if (a.rows() != a.cols()) throw std::invalid_argument("eigh: matrix must be square");
    Eigen::SelfAdjointEigenSolver<Matrix> solver(a);
    if (solver.info() != Eigen::Success) throw std::runtime_error("eigh: eigensolver did not converge");
    return {solver.eigenvalues(), solver.eigenvectors()};
}

Matrix spectral_embedding(const Eigendecomposition& e, int k) {
    if (k < 1 || k > e.vectors.cols()) throw std::invalid_argument("spectral_embedding: bad k");
    Matrix x = e.vectors.leftCols(k);
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
        const double norm = x.row(i).norm();
        if (norm > 1e-12) x.row(i) /= norm;
    }
    return x;
}

Matrix low_rank_approximation(const Eigendecomposition& e, int k) {
    const Eigen::Index n = e.values.size();
    if (k < 0 || k > n) throw std::invalid_argument("low_rank_approximation: bad k");

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return std::abs(e.values(a)) > std::abs(e.values(b)); });

    Matrix vk(n, k);       // V_k, columns scaled by lambda so that C_k = (V_k Lambda) V^T
    Matrix scaled(n, k);
    for (int j = 0; j < k; ++j) {
        vk.col(j) = e.vectors.col(order[j]);
        scaled.col(j) = e.values(order[j]) * vk.col(j);
    }
    return scaled * vk.transpose();
}

std::vector<RankError> reconstruction_errors(const Matrix& corr, const std::vector<int>& ks) {
    const Eigendecomposition e = eigh(corr);
    const double denom = corr.norm();  // Frobenius
    std::vector<RankError> out;
    for (int k : ks) {
        const double num = (corr - low_rank_approximation(e, k)).norm();
        out.push_back({k, denom > 0.0 ? num / denom : 0.0});
    }
    return out;
}

}  // namespace hs
