// Correlation benchmark. Prints whatever it measures; nothing is hard-coded.
#include "benchmark.hpp"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) try {
    int n_obs = 2000, reps = 5;
    std::vector<int> sizes = {100, 250, 500, 1000};
    for (int i = 1; i < argc; i += 2) {
        const std::string a = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
        if (a == "--obs") n_obs = std::atoi(argv[i + 1]);
        else if (a == "--reps") reps = std::atoi(argv[i + 1]);
        else if (a == "--sizes") {
            sizes.clear();
            std::stringstream ss(argv[i + 1]);
            for (std::string s; std::getline(ss, s, ',');) sizes.push_back(std::atoi(s.c_str()));
        } else throw std::invalid_argument("unknown option " + a + " (use --sizes a,b,c --obs T --reps R)");
    }

    const bool gpu = hs::cuda_available();
    std::printf("correlation benchmark: median of %d runs (each path warmed up once), %d observations\n", reps, n_obs);
    std::printf("cpu_ms        = standardise + Eigen product + finalise, single thread unless Eigen uses OpenMP\n");
    if (gpu) {
        std::printf("gpu_total_ms  = standardise (host) + row-major copy + cudaMalloc + H2D + kernel + D2H + finalise, FP64\n");
        std::printf("gpu_kernel_ms = kernel only (CUDA events); speedup = cpu_ms / gpu_total_ms\n");
    } else {
        std::printf("GPU: not available in this build/machine; GPU columns omitted\n");
    }

    std::printf("\n%8s %8s %12s", "assets", "obs", "cpu_ms");
    if (gpu) std::printf(" %13s %14s %9s %11s", "gpu_total_ms", "gpu_kernel_ms", "speedup", "max|diff|");
    std::printf("\n");
    for (int n : sizes) {
        const hs::BenchRow r = hs::bench_correlation(n, n_obs, reps, gpu);
        std::printf("%8d %8d %12.3f", r.n_assets, r.n_obs, r.cpu_ms);
        if (gpu)
            std::printf(" %13.3f %14.3f %8.2fx %11.2e", r.gpu_total_ms, r.gpu_kernel_ms,
                        r.cpu_ms / r.gpu_total_ms, r.max_abs_diff);
        std::printf("\n");
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
