#pragma once
// Synthetic factor-model returns and a minimal CSV loader.
#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace hs {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Portable RNG helpers. std::normal_distribution / uniform_real_distribution are
// allowed to produce different streams on different standard libraries; these
// are not, so "same seed" means "same numbers" on every platform.
inline double uniform01(std::mt19937_64& g) {  // uniform on [0, 1)
    return static_cast<double>(g() >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
}
inline double standard_normal(std::mt19937_64& g) {  // Box-Muller (one draw per call)
    constexpr double kTwoPi = 6.28318530717958647692;
    const double u1 = 1.0 - uniform01(g);  // in (0, 1] so log() is finite
    const double u2 = uniform01(g);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
}

struct SyntheticConfig {
    int n_assets = 100;
    int n_obs = 500;
    int n_sectors = 5;
    double sector_vol = 0.010;  // std-dev of each sector factor per period
    double idio_vol = 0.010;    // std-dev of idiosyncratic noise per period
    std::uint64_t seed = 42;    // fixed default => reproducible
};

struct SyntheticData {
    Matrix returns;           // n_assets x n_obs  (rows = assets, columns = observations)
    std::vector<int> labels;  // planted sector of each asset, in [0, n_sectors)
};

// r_{i,t} = beta_i * f_{s(i),t} + e_{i,t},  f ~ N(0, sector_vol^2), e ~ N(0, idio_vol^2),
// beta_i ~ U(0.5, 1.5), s(i) = i mod n_sectors. Throws std::invalid_argument on bad input.
SyntheticData generate_synthetic(const SyntheticConfig& cfg);

// CSV with one row per observation and one column per asset (optional header line).
// Returns assets x observations. Throws std::runtime_error on I/O errors, ragged rows,
// or non-numeric / missing cells (missing data is deliberately not imputed).
Matrix load_returns_csv(const std::string& path);

}  // namespace hs
