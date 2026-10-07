#pragma once
#include "data.hpp"

#include <vector>

namespace hs {

struct GraphOptions {
    double threshold = 0.30;    // keep edge (i,j) iff |rho_ij| >= threshold
    bool signed_weights = false;  // false: w = |rho|; true: w = rho (signed graph)
};

// Weighted adjacency matrix W from a correlation matrix. Symmetric, zero diagonal
// (a self-correlation of 1 carries no information about structure).
// Throws std::invalid_argument unless 0 <= threshold <= 1.
Matrix build_adjacency(const Matrix& corr, const GraphOptions& opt = {});

// Degrees are d_i = sum_j |W_ij| (equal to the usual degree for non-negative W; for
// signed W this gives the "signed Laplacian", which stays positive semi-definite).
Vector degrees(const Matrix& w);

// L = D - W.
Matrix laplacian(const Matrix& w);

// L_norm = I - D^{-1/2} W D^{-1/2}. Isolated vertices (d_i = 0) use D^{-1/2}_ii := 0, so
// their row/column of the normalised adjacency is zero and L_norm_ii = 1: they sit at
// eigenvalue 1 rather than polluting the near-zero part of the spectrum.
Matrix normalized_laplacian(const Matrix& w);

int count_isolated(const Matrix& w);

struct Edge {
    int u, v;
    double weight;  // Mantegna distance sqrt(2 (1 - rho))
};

// Minimum spanning tree of the complete graph with distance d_ij = sqrt(2 (1 - rho_ij)),
// via Prim's algorithm, O(N^2). Returns N-1 edges (empty if N < 2).
std::vector<Edge> minimum_spanning_tree(const Matrix& corr);

}  // namespace hs
