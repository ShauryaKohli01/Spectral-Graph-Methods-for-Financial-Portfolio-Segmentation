// Pipeline: returns -> correlation -> thresholded graph -> normalised Laplacian ->
// eigendecomposition -> spectral clustering, plus low-rank reconstruction of C.
#include "clustering.hpp"
#include "correlation.hpp"
#include "data.hpp"
#include "graph.hpp"
#include "spectral.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {

struct Options {
    hs::SyntheticConfig data;
    hs::GraphOptions graph;
    std::string csv, out;
    int clusters = 0;  // 0 => use data.n_sectors
    bool gpu = false, mst = false;
};

void usage() {
    std::puts(
        "usage: hs_demo [options]\n"
        "  --assets N --obs T --sectors K --sector-vol X --idio-vol X --seed S   synthetic data\n"
        "  --csv FILE            load returns instead (rows = observations, columns = assets)\n"
        "  --clusters K          number of clusters (default: --sectors, i.e. 5)\n"
        "  --threshold X         keep edges with |rho| >= X (default 0.30)\n"
        "  --signed              use signed weights w = rho instead of |rho|\n"
        "  --mst                 also report the minimum spanning tree\n"
        "  --gpu                 compute the correlation matrix with the CUDA kernel\n"
        "  --out FILE            write asset,cluster[,planted] CSV");
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--assets") o.data.n_assets = std::atoi(next());
        else if (a == "--obs") o.data.n_obs = std::atoi(next());
        else if (a == "--sectors") o.data.n_sectors = std::atoi(next());
        else if (a == "--sector-vol") o.data.sector_vol = std::atof(next());
        else if (a == "--idio-vol") o.data.idio_vol = std::atof(next());
        else if (a == "--seed") o.data.seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--csv") o.csv = next();
        else if (a == "--clusters") o.clusters = std::atoi(next());
        else if (a == "--threshold") o.graph.threshold = std::atof(next());
        else if (a == "--signed") o.graph.signed_weights = true;
        else if (a == "--mst") o.mst = true;
        else if (a == "--gpu") o.gpu = true;
        else if (a == "--out") o.out = next();
        else if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        else throw std::invalid_argument("unknown option " + a);
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) try {
    const Options o = parse(argc, argv);

    hs::Matrix returns;
    std::vector<int> planted;
    if (o.csv.empty()) {
        auto d = hs::generate_synthetic(o.data);
        returns = std::move(d.returns);
        planted = std::move(d.labels);
    } else {
        returns = hs::load_returns_csv(o.csv);
    }
    const int n = static_cast<int>(returns.rows()), t = static_cast<int>(returns.cols());
    const int k = o.clusters > 0 ? o.clusters : o.data.n_sectors;

    std::printf("data: %d assets x %d observations (%s), correlation on %s\n", n, t,
                o.csv.empty() ? "synthetic" : o.csv.c_str(), o.gpu ? "GPU" : "CPU");
    const hs::Matrix corr = o.gpu ? hs::correlation_gpu(returns) : hs::correlation_cpu(returns);

    const hs::Matrix w = hs::build_adjacency(corr, o.graph);
    const double edges = (w.array() != 0.0).count() / 2.0;
    std::printf("graph: threshold %.2f, %s weights, %.0f edges (density %.3f), %d isolated vertices\n",
                o.graph.threshold, o.graph.signed_weights ? "signed" : "|rho|", edges,
                n > 1 ? 2.0 * edges / (static_cast<double>(n) * (n - 1)) : 0.0, hs::count_isolated(w));

    const auto sc = hs::spectral_clustering(w, k, o.data.seed);
    std::printf("clusters requested: %d\nsmallest L_norm eigenvalues:", k);
    for (int i = 0; i < std::min(n, k + 3); ++i) {
        const double ev = sc.eigenvalues(i);
        std::printf(" %.4f", std::abs(ev) < 5e-5 ? 0.0 : ev);  // avoid printing "-0.0000"
    }
    std::puts("");
    // Zero eigenvalues of L_norm = connected components (isolated vertices sit at 1, not 0).
    // If there are more components than clusters requested, the bottom-k eigenspace is an
    // arbitrary rotation inside the null space and the clustering is not meaningful.
    const int components = static_cast<int>((sc.eigenvalues.array() < 1e-8).count());
    std::printf("connected components among non-isolated vertices: %d\n", components);
    if (hs::count_isolated(w) > 0)
        std::printf("warning: %d isolated vertices get zero embedding rows and are clustered arbitrarily; lower --threshold\n",
                    hs::count_isolated(w));
    if (components > k)
        std::printf("warning: %d components > %d clusters requested; raise --clusters or lower --threshold\n",
                    components, k);

    std::map<int, int> sizes;
    for (int l : sc.labels) ++sizes[l];
    std::printf("cluster sizes:");
    for (auto& kv : sizes) std::printf(" %d", kv.second);
    std::puts("");
    if (!planted.empty())
        std::printf("adjusted Rand index vs planted sectors: %.4f\n",
                    hs::adjusted_rand_index(sc.labels, planted));

    std::printf("\nasset -> cluster%s (first %d)\n", planted.empty() ? "" : " (planted)", std::min(n, 12));
    for (int i = 0; i < std::min(n, 12); ++i) {
        if (planted.empty()) std::printf("  %4d -> %d\n", i, sc.labels[i]);
        else std::printf("  %4d -> %d (%d)\n", i, sc.labels[i], planted[i]);
    }

    if (!o.out.empty()) {
        std::ofstream f(o.out);
        f << (planted.empty() ? "asset,cluster\n" : "asset,cluster,planted\n");
        for (int i = 0; i < n; ++i) {
            f << i << ',' << sc.labels[i];
            if (!planted.empty()) f << ',' << planted[i];
            f << '\n';
        }
    }

    if (o.mst) {
        const auto tree = hs::minimum_spanning_tree(corr);
        double total = 0.0;
        for (const auto& e : tree) total += e.weight;
        std::printf("\nMST: %zu edges, total Mantegna distance %.4f\n", tree.size(), total);
    }

    std::vector<int> ks;
    for (int kk : {1, 2, 3, 5, 10, 20, 50, 100, 200, 500})
        if (kk < n) ks.push_back(kk);
    ks.push_back(n);
    std::printf("\n%6s  %s\n", "k", "reconstruction_error");
    for (const auto& r : hs::reconstruction_errors(corr, ks)) std::printf("%6d  %.6f\n", r.k, r.error);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
