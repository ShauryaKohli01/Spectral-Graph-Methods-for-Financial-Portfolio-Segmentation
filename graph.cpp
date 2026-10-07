#include "graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hs {

Matrix build_adjacency(const Matrix& corr, const GraphOptions& opt) {
    if (!(opt.threshold >= 0.0 && opt.threshold <= 1.0))
        throw std::invalid_argument("threshold must lie in [0, 1]");
    const Eigen::Index n = corr.rows();
    Matrix w = Matrix::Zero(n, n);
    for (Eigen::Index i = 0; i < n; ++i)
        for (Eigen::Index j = i + 1; j < n; ++j) {
            const double rho = corr(i, j);
            if (std::abs(rho) >= opt.threshold) w(i, j) = w(j, i) = opt.signed_weights ? rho : std::abs(rho);
        }
    return w;
}

Vector degrees(const Matrix& w) { return w.cwiseAbs().rowwise().sum(); }

Matrix laplacian(const Matrix& w) {
    Matrix l = -w;
    l.diagonal() += degrees(w);  // diagonal of w is zero by construction
    return l;
}

Matrix normalized_laplacian(const Matrix& w) {
    const Vector d = degrees(w);
    Vector inv_sqrt(d.size());
    for (Eigen::Index i = 0; i < d.size(); ++i) inv_sqrt(i) = d(i) > 1e-12 ? 1.0 / std::sqrt(d(i)) : 0.0;
    Matrix l = -(inv_sqrt.asDiagonal() * w * inv_sqrt.asDiagonal());
    l.diagonal().array() += 1.0;
    return l;
}

int count_isolated(const Matrix& w) { return static_cast<int>((degrees(w).array() <= 1e-12).count()); }

std::vector<Edge> minimum_spanning_tree(const Matrix& corr) {
    const int n = static_cast<int>(corr.rows());
    std::vector<Edge> tree;
    if (n < 2) return tree;

    auto dist = [&](int i, int j) { return std::sqrt(std::max(0.0, 2.0 * (1.0 - corr(i, j)))); };
    std::vector<double> best(n, std::numeric_limits<double>::infinity());
    std::vector<int> parent(n, -1);
    std::vector<bool> in_tree(n, false);
    best[0] = 0.0;
    for (int step = 0; step < n; ++step) {
        int u = -1;
        for (int i = 0; i < n; ++i)
            if (!in_tree[i] && (u < 0 || best[i] < best[u])) u = i;
        in_tree[u] = true;
        if (parent[u] >= 0) tree.push_back({parent[u], u, best[u]});
        for (int v = 0; v < n; ++v)
            if (!in_tree[v] && dist(u, v) < best[v]) { best[v] = dist(u, v); parent[v] = u; }
    }
    return tree;
}

}  // namespace hs
