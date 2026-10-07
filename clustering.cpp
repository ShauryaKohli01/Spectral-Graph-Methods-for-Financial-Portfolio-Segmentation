#include "clustering.hpp"

#include "graph.hpp"
#include "spectral.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <random>
#include <stdexcept>

namespace hs {

namespace {

double sq_dist(const Matrix& x, Eigen::Index i, const Matrix& c, Eigen::Index j) {
    return (x.row(i) - c.row(j)).squaredNorm();
}

Matrix kmeanspp_init(const Matrix& x, int k, std::mt19937_64& rng) {
    const Eigen::Index n = x.rows();
    Matrix c(k, x.cols());
    c.row(0) = x.row(static_cast<Eigen::Index>(uniform01(rng) * n));
    Vector d2 = Vector::Constant(n, std::numeric_limits<double>::infinity());
    for (int j = 1; j < k; ++j) {
        double total = 0.0;
        for (Eigen::Index i = 0; i < n; ++i) {
            d2(i) = std::min(d2(i), sq_dist(x, i, c, j - 1));
            total += d2(i);
        }
        // Sample the next centroid with probability proportional to D(x)^2.
        double target = uniform01(rng) * total, acc = 0.0;
        Eigen::Index pick = n - 1;
        for (Eigen::Index i = 0; i < n; ++i) {
            acc += d2(i);
            if (acc >= target && d2(i) > 0.0) { pick = i; break; }
        }
        c.row(j) = x.row(pick);
    }
    return c;
}

KMeansResult lloyd(const Matrix& x, int k, std::mt19937_64& rng, int max_iter) {
    const Eigen::Index n = x.rows();
    Matrix c = kmeanspp_init(x, k, rng);
    std::vector<int> labels(n, -1);
    double inertia = 0.0;
    for (int it = 0; it < max_iter; ++it) {
        bool changed = false;
        inertia = 0.0;
        for (Eigen::Index i = 0; i < n; ++i) {
            int best = 0;
            double bd = sq_dist(x, i, c, 0);
            for (int j = 1; j < k; ++j) {
                const double d = sq_dist(x, i, c, j);
                if (d < bd) { bd = d; best = j; }
            }
            if (labels[i] != best) { labels[i] = best; changed = true; }
            inertia += bd;
        }
        if (!changed) break;  // labels and inertia are consistent with the current centroids

        Matrix sum = Matrix::Zero(k, x.cols());
        std::vector<int> count(k, 0);
        for (Eigen::Index i = 0; i < n; ++i) { sum.row(labels[i]) += x.row(i); ++count[labels[i]]; }
        for (int j = 0; j < k; ++j) {
            if (count[j] > 0) { c.row(j) = sum.row(j) / count[j]; continue; }
            // Empty cluster: re-seed at the point currently farthest from its centroid.
            Eigen::Index far = 0;
            double fd = -1.0;
            for (Eigen::Index i = 0; i < n; ++i) {
                const double d = sq_dist(x, i, c, labels[i]);
                if (d > fd) { fd = d; far = i; }
            }
            c.row(j) = x.row(far);
        }
    }
    // If max_iter was hit, the centroids moved after the last assignment: recompute inertia
    // against the final centroids so restarts are compared on equal terms.
    inertia = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) inertia += sq_dist(x, i, c, labels[i]);
    return {labels, inertia};
}

}  // namespace

KMeansResult kmeans(const Matrix& x, int k, std::uint64_t seed, int n_init, int max_iter) {
    if (k < 1 || k > x.rows()) throw std::invalid_argument("kmeans: need 1 <= k <= n_points");
    std::mt19937_64 rng(seed);
    KMeansResult best{{}, std::numeric_limits<double>::infinity()};
    for (int r = 0; r < n_init; ++r) {
        KMeansResult cur = lloyd(x, k, rng, max_iter);
        if (cur.inertia < best.inertia) best = std::move(cur);
    }
    return best;
}

SpectralResult spectral_clustering(const Matrix& w, int k, std::uint64_t seed) {
    const Eigendecomposition e = eigh(normalized_laplacian(w));
    const Matrix embedding = spectral_embedding(e, k);
    return {kmeans(embedding, k, seed).labels, e.values};
}

double adjusted_rand_index(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size() || a.size() < 2) throw std::invalid_argument("ARI: need equal sizes >= 2");
    std::map<std::pair<int, int>, double> joint;
    std::map<int, double> ca, cb;
    for (std::size_t i = 0; i < a.size(); ++i) { ++joint[{a[i], b[i]}]; ++ca[a[i]]; ++cb[b[i]]; }

    auto c2 = [](double x) { return x * (x - 1.0) / 2.0; };
    double s_joint = 0.0, s_a = 0.0, s_b = 0.0;
    for (auto& kv : joint) s_joint += c2(kv.second);
    for (auto& kv : ca) s_a += c2(kv.second);
    for (auto& kv : cb) s_b += c2(kv.second);

    const double expected = s_a * s_b / c2(static_cast<double>(a.size()));
    const double denom = 0.5 * (s_a + s_b) - expected;
    return std::abs(denom) < 1e-15 ? 1.0 : (s_joint - expected) / denom;  // both partitions trivial
}

}  // namespace hs
