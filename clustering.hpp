#pragma once
#include "data.hpp"

#include <cstdint>
#include <vector>

namespace hs {

struct KMeansResult {
    std::vector<int> labels;
    double inertia;  // sum of squared distances to assigned centroids
};

// Lloyd's algorithm with k-means++ seeding, `n_init` restarts, best inertia wins.
// Rows of x are points. Deterministic for a given seed.
KMeansResult kmeans(const Matrix& x, int k, std::uint64_t seed, int n_init = 10, int max_iter = 100);

struct SpectralResult {
    std::vector<int> labels;
    Vector eigenvalues;  // full spectrum of L_norm, ascending (useful for eigengap inspection)
};

// Normalised spectral clustering (Ng-Jordan-Weiss) on adjacency matrix W.
SpectralResult spectral_clustering(const Matrix& w, int k, std::uint64_t seed = 42);

// Adjusted Rand Index in [-1, 1]; 1 = identical partitions up to relabelling, ~0 = chance.
double adjusted_rand_index(const std::vector<int>& a, const std::vector<int>& b);

}  // namespace hs
