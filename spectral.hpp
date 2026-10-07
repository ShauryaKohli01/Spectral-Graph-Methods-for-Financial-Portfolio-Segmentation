#pragma once
#include "data.hpp"

#include <vector>

namespace hs {

struct Eigendecomposition {
    Vector values;   // ascending
    Matrix vectors;  // column j is the unit eigenvector for values(j)
};

// Symmetric eigendecomposition A = V diag(values) V^T (reads the lower triangle only).
Eigendecomposition eigh(const Matrix& a);

// Rows are the first k eigenvectors (smallest eigenvalues) of a Laplacian, each row
// scaled to unit length (Ng-Jordan-Weiss). All-zero rows stay zero. Result is N x k.
Matrix spectral_embedding(const Eigendecomposition& laplacian_eig, int k);

// C_k = V_k Lambda_k V_k^T using the k eigenpairs of largest |lambda|.
Matrix low_rank_approximation(const Eigendecomposition& eig, int k);

struct RankError {
    int k;
    double error;  // ||C - C_k||_F / ||C||_F
};
std::vector<RankError> reconstruction_errors(const Matrix& corr, const std::vector<int>& ks);

}  // namespace hs
