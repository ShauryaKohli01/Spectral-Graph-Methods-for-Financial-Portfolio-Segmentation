#pragma once
// Header-only timing helpers for the correlation benchmark.
#include "correlation.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hs {

inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

template <class F>
double elapsed_ms(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct BenchRow {
    int n_assets = 0, n_obs = 0;
    double cpu_ms = 0.0;  // end-to-end: standardise + GEMM + finalise
    double nan = std::numeric_limits<double>::quiet_NaN();
    double gpu_total_ms = nan;   // end-to-end: standardise + copies + kernel + finalise (NaN => not measured)
    double gpu_kernel_ms = nan;  // kernel only (CUDA events)
    double max_abs_diff = nan;   // max |C_gpu - C_cpu|
};

// Median over `reps` timed runs, each preceded by one untimed warm-up of the same path
// (absorbs first-touch page faults and CUDA context creation).
inline BenchRow bench_correlation(int n_assets, int n_obs, int reps, bool use_gpu) {
    if (reps < 1) throw std::invalid_argument("reps must be >= 1");
    SyntheticConfig cfg;
    cfg.n_assets = n_assets;
    cfg.n_obs = n_obs;
    cfg.n_sectors = std::min(10, n_assets);
    const Matrix r = generate_synthetic(cfg).returns;

    BenchRow row;
    row.n_assets = n_assets;
    row.n_obs = n_obs;

    Matrix c_cpu = correlation_cpu(r);  // warm-up
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) t.push_back(elapsed_ms([&] { c_cpu = correlation_cpu(r); }));
    row.cpu_ms = median(t);

    if (use_gpu) {
        Matrix c_gpu = correlation_gpu(r);  // warm-up
        std::vector<double> total, kernel;
        for (int i = 0; i < reps; ++i) {
            double k = 0.0;
            total.push_back(elapsed_ms([&] { c_gpu = correlation_gpu(r, &k); }));
            kernel.push_back(k);
        }
        row.gpu_total_ms = median(total);
        row.gpu_kernel_ms = median(kernel);
        row.max_abs_diff = (c_gpu - c_cpu).cwiseAbs().maxCoeff();
    }
    return row;
}

}  // namespace hs
