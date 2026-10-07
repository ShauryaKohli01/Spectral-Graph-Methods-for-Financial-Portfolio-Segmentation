// Dependency-free test runner. Every test is deterministic (fixed seeds, no wall-clock).
#include "clustering.hpp"
#include "correlation.hpp"
#include "data.hpp"
#include "graph.hpp"
#include "spectral.hpp"

#include <cmath>
#include <limits>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace hs;

struct TestCase { const char* name; void (*fn)(); };
static std::vector<TestCase>& registry() { static std::vector<TestCase> r; return r; }

#define TEST(name)                                                              \
    static void name();                                                         \
    [[maybe_unused]] static const bool reg_##name = (registry().push_back({#name, &name}), true); \
    static void name()
#define CHECK(cond)                                                             \
    do { if (!(cond)) { std::ostringstream os_; os_ << __FILE__ << ":" << __LINE__ << ": CHECK(" #cond ")"; \
                        throw std::runtime_error(os_.str()); } } while (0)
#define CHECK_NEAR(a, b, tol)                                                   \
    do { const double a_ = (a), b_ = (b); if (!(std::abs(a_ - b_) <= (tol))) { std::ostringstream os_; \
         os_ << __FILE__ << ":" << __LINE__ << ": " #a " = " << a_ << " vs " #b " = " << b_; \
         throw std::runtime_error(os_.str()); } } while (0)

static Matrix rows(std::initializer_list<std::initializer_list<double>> init) {
    Matrix m(init.size(), init.begin()->size());
    Eigen::Index i = 0;
    for (auto& r : init) { Eigen::Index j = 0; for (double v : r) m(i, j++) = v; ++i; }
    return m;
}

TEST(correlation_known_values) {
    // x, y = 2x+1 (rho = 1), z = -x (rho = -1), w = [2,1,4,3,5] (rho = 0.8 by hand: 8/sqrt(10*10)).
    const Matrix r = rows({{1, 2, 3, 4, 5}, {3, 5, 7, 9, 11}, {-1, -2, -3, -4, -5}, {2, 1, 4, 3, 5}});
    const Matrix c = correlation_cpu(r);
    CHECK_NEAR(c(0, 1), 1.0, 1e-14);
    CHECK_NEAR(c(0, 2), -1.0, 1e-14);
    CHECK_NEAR(c(0, 3), 0.8, 1e-14);
    CHECK_NEAR(c(1, 3), 0.8, 1e-14);
}

TEST(correlation_matches_numpy_reference) {
    // Values produced by tests/gen_reference.py (np.corrcoef, seed 7).
    const Matrix r = rows({{0.0000, 0.0060, -0.0055, -0.0178, -0.0091, -0.0198, 0.0012, 0.0268},
                           {-0.0098, -0.0124, 0.0098, 0.0071, 0.0021, -0.0186, -0.0006, 0.0139},
                           {-0.0269, -0.0092, -0.0380, -0.0258, -0.0368, -0.0047, -0.0253, 0.0054},
                           {0.0031, -0.0037, -0.0503, -0.0108, -0.0010, 0.0023, -0.0306, -0.0096}});
    const double ref[6] = {0.373700351047479, 0.486242744419612, -0.058169264427284,
                           -0.197161663674503, -0.544674987433164, 0.421214601168749};
    const Matrix c = correlation_cpu(r);
    int idx = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j) CHECK_NEAR(c(i, j), ref[idx++], 1e-12);
}

TEST(standardize_rows_has_zero_mean_unit_norm) {
    // Diagonal "=1" of correlation_cpu is forced by construction, so test the quantity that
    // actually produces it: each standardised row has mean 0 and Euclidean norm 1.
    Matrix r = generate_synthetic({}).returns;
    r.row(3).setConstant(0.01);  // a degenerate row must become exactly zero
    const Matrix z = standardize_rows(r);
    for (int i = 0; i < z.rows(); ++i) {
        CHECK_NEAR(z.row(i).mean(), 0.0, 1e-15);
        CHECK_NEAR(z.row(i).norm(), i == 3 ? 0.0 : 1.0, 1e-14);
    }
    const Matrix raw = z * z.transpose();  // before symmetrise/clamp: still symmetric to rounding
    CHECK((raw - raw.transpose()).cwiseAbs().maxCoeff() < 1e-14);
}

TEST(correlation_is_valid_psd_matrix) {
    const Matrix c = correlation_cpu(generate_synthetic({}).returns);
    CHECK((c - c.transpose()).cwiseAbs().maxCoeff() == 0.0);
    CHECK(c.diagonal().isApproxToConstant(1.0, 1e-15));
    CHECK(c.cwiseAbs().maxCoeff() <= 1.0);
    CHECK(eigh(c).values(0) > -1e-10);  // positive semi-definite up to rounding
}

TEST(correlation_rank_deficient_when_assets_exceed_observations) {
    SyntheticConfig cfg;
    cfg.n_assets = 20;
    cfg.n_obs = 10;  // centred data have rank <= T-1 = 9
    cfg.n_sectors = 4;
    const Vector ev = eigh(correlation_cpu(generate_synthetic(cfg).returns)).values;
    CHECK(ev(0) > -1e-10);
    CHECK((ev.array() > 1e-10).count() == 9);
}

TEST(invalid_input_is_rejected) {
    auto throws_invalid = [](auto&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
    Matrix nan_data = generate_synthetic({}).returns;
    nan_data(2, 5) = std::numeric_limits<double>::quiet_NaN();
    CHECK(throws_invalid([&] { correlation_cpu(nan_data); }));
    CHECK(throws_invalid([&] { correlation_cpu(Matrix::Ones(3, 1)); }));  // one observation
    CHECK(throws_invalid([&] { build_adjacency(Matrix::Identity(3, 3), {1.5, false}); }));
    SyntheticConfig bad;
    bad.n_sectors = 200;  // more sectors than assets
    CHECK(throws_invalid([&] { generate_synthetic(bad); }));
}

TEST(correlation_stable_with_large_mean_and_constant_row) {
    // Mean 1e8, std ~1: a one-pass E[x^2]-E[x]^2 formula loses all precision here.
    Matrix r = rows({{1, 2, 3, 4, 5}, {2, 1, 4, 3, 5}, {7, 7, 7, 7, 7}});
    r.row(0).array() += 1e8;
    r.row(1).array() += 1e8;
    const Matrix c = correlation_cpu(r);
    CHECK_NEAR(c(0, 1), 0.8, 1e-7);
    CHECK(c(0, 2) == 0.0 && c(2, 2) == 1.0);  // constant series: documented convention
}

TEST(synthetic_is_reproducible) {
    SyntheticConfig cfg;
    const auto a = generate_synthetic(cfg), b = generate_synthetic(cfg);
    CHECK(a.returns == b.returns && a.labels == b.labels);
    cfg.seed = 43;
    CHECK(generate_synthetic(cfg).returns != a.returns);
    CHECK(a.returns.rows() == 100 && a.returns.cols() == 500);
    CHECK(a.labels[7] == 7 % 5);
}

TEST(laplacian_construction) {
    // Triangle with weights w01=1, w02=2, w12=3  =>  degrees 3, 4, 5.
    const Matrix w = rows({{0, 1, 2}, {1, 0, 3}, {2, 3, 0}});
    const Matrix l = laplacian(w);
    CHECK_NEAR(l(0, 0), 3.0, 1e-15);
    CHECK_NEAR(l(1, 1), 4.0, 1e-15);
    CHECK_NEAR(l(0, 1), -1.0, 1e-15);
    CHECK(l.rowwise().sum().cwiseAbs().maxCoeff() < 1e-15);  // rows sum to zero
    const Matrix ln = normalized_laplacian(w);
    CHECK_NEAR(ln(0, 0), 1.0, 1e-15);
    CHECK_NEAR(ln(0, 1), -1.0 / std::sqrt(12.0), 1e-15);   // -w01 / sqrt(d0 d1)
    CHECK((ln - ln.transpose()).cwiseAbs().maxCoeff() < 1e-15);
    CHECK_NEAR(eigh(ln).values(0), 0.0, 1e-12);
}

TEST(laplacian_isolated_vertex_and_components) {
    // Two disjoint edges + one isolated vertex => 2 zero eigenvalues of L_norm over the
    // connected part, isolated vertex pinned at eigenvalue 1, no NaNs anywhere.
    Matrix w = Matrix::Zero(5, 5);
    w(0, 1) = w(1, 0) = 1.0;
    w(2, 3) = w(3, 2) = 2.0;
    CHECK(count_isolated(w) == 1);
    const Matrix ln = normalized_laplacian(w);
    CHECK(ln.allFinite());
    CHECK_NEAR(ln(4, 4), 1.0, 1e-15);
    const Vector ev = eigh(ln).values;
    CHECK_NEAR(ev(0), 0.0, 1e-12);
    CHECK_NEAR(ev(1), 0.0, 1e-12);
    CHECK(ev(2) > 0.5);
    const Matrix l = laplacian(w);
    CHECK(l.row(4).cwiseAbs().maxCoeff() == 0.0);
}

TEST(adjacency_threshold_and_signed_option) {
    const Matrix c = rows({{1, 0.5, -0.4, 0.1}, {0.5, 1, 0.0, 0.2}, {-0.4, 0.0, 1, 0.3}, {0.1, 0.2, 0.3, 1}});
    const Matrix w = build_adjacency(c, {0.3, false});
    CHECK(w(0, 1) == 0.5 && w(0, 2) == 0.4 && w(2, 3) == 0.3);  // >= threshold kept, |rho| used
    CHECK(w(0, 3) == 0.0 && w.diagonal().isZero(0.0));
    const Matrix ws = build_adjacency(c, {0.3, true});
    CHECK(ws(0, 2) == -0.4);
    CHECK(laplacian(ws).diagonal()(0) == 0.9);  // signed Laplacian uses |w| degrees
}

TEST(mst_hand_computed_case) {
    // rho01=0.9, rho12=0.5, rho02=0.1: the tree must be {0-1, 1-2}, distances sqrt(2(1-rho)).
    const Matrix c = rows({{1, 0.9, 0.1}, {0.9, 1, 0.5}, {0.1, 0.5, 1}});
    const auto tree = minimum_spanning_tree(c);
    CHECK(tree.size() == 2);
    double total = 0.0;
    for (auto& e : tree) {
        CHECK((e.u == 0 && e.v == 1) || (e.u == 1 && e.v == 2));
        total += e.weight;
    }
    CHECK_NEAR(total, std::sqrt(0.2) + std::sqrt(1.0), 1e-14);
}

TEST(mst_is_spanning_tree) {
    const Matrix c = correlation_cpu(generate_synthetic({}).returns);
    const auto tree = minimum_spanning_tree(c);
    CHECK(tree.size() == 99);
    // Connectivity check via union-find.
    std::vector<int> p(100);
    for (int i = 0; i < 100; ++i) p[i] = i;
    auto find = [&](int x) { while (p[x] != x) x = p[x] = p[p[x]]; return x; };
    for (auto& e : tree) p[find(e.u)] = find(e.v);
    for (int i = 1; i < 100; ++i) CHECK(find(i) == find(0));
}

TEST(ari_properties) {
    const std::vector<int> a = {0, 0, 1, 1, 2, 2}, relabelled = {2, 2, 0, 0, 1, 1};
    CHECK_NEAR(adjusted_rand_index(a, relabelled), 1.0, 1e-15);
    CHECK(adjusted_rand_index(a, {0, 1, 0, 1, 0, 1}) < 0.1);  // unrelated partition
}

TEST(spectral_embedding_rows_are_unit_norm_and_component_aligned) {
    // Two components with very different degrees + one isolated vertex. After NJW row
    // normalisation: unit rows; same-component rows coincide (the sqrt(d_i) scaling of the
    // null-space eigenvectors is removed); different components are orthogonal; the isolated
    // vertex gets an all-zero row instead of NaN.
    Matrix w = Matrix::Zero(5, 5);
    w(0, 1) = w(1, 0) = 1.0;
    w(2, 3) = w(3, 2) = 50.0;
    const Matrix x = spectral_embedding(eigh(normalized_laplacian(w)), 2);
    for (int i = 0; i < 4; ++i) CHECK_NEAR(x.row(i).norm(), 1.0, 1e-12);
    CHECK(x.row(4).norm() == 0.0);
    CHECK((x.row(0) - x.row(1)).norm() < 1e-9);
    CHECK((x.row(2) - x.row(3)).norm() < 1e-9);
    CHECK_NEAR(x.row(0).dot(x.row(2)), 0.0, 1e-9);
}

TEST(spectral_clustering_recovers_planted_sectors) {
    const SyntheticData d = generate_synthetic({});
    const Matrix w = build_adjacency(correlation_cpu(d.returns), {0.25, false});
    const SpectralResult r = spectral_clustering(w, 5, 42);
    CHECK(adjusted_rand_index(r.labels, d.labels) > 0.95);
    CHECK(r.eigenvalues(4) < 0.5 * r.eigenvalues(5));  // eigengap after the 5th eigenvalue
    CHECK(r.labels == spectral_clustering(w, 5, 42).labels);  // deterministic
}

TEST(spectral_clustering_on_connected_graph_and_negative_control) {
    // Threshold 0.10 keeps cross-sector noise edges, so the graph is ONE component: recovery
    // now depends on the near-zero eigenvectors and the eigengap, not on trivial components.
    const SyntheticData d = generate_synthetic({});
    const Matrix w = build_adjacency(correlation_cpu(d.returns), {0.10, false});
    const SpectralResult r = spectral_clustering(w, 5, 42);
    CHECK(r.eigenvalues(1) > 1e-3);                 // connected: only one exact zero
    CHECK(r.eigenvalues(4) < 0.2 && r.eigenvalues(5) > 0.8);  // 5 small, then a clear gap
    CHECK(adjusted_rand_index(r.labels, d.labels) > 0.95);

    // Negative control: same pipeline with NO sector factor must not "recover" the labels,
    // otherwise the ARI assertion above would be vacuous.
    SyntheticConfig noise;
    noise.sector_vol = 0.0;
    const SyntheticData n = generate_synthetic(noise);
    const Matrix wn = build_adjacency(correlation_cpu(n.returns), {0.05, false});
    CHECK(adjusted_rand_index(spectral_clustering(wn, 5, 42).labels, n.labels) < 0.2);
}

TEST(reconstruction_error_decreases_and_matches_eckart_young) {
    const Matrix c = correlation_cpu(generate_synthetic({}).returns);
    const std::vector<int> ks = {1, 2, 5, 10, 25, 50, 100};
    const auto errs = reconstruction_errors(c, ks);
    for (std::size_t i = 1; i < errs.size(); ++i) CHECK(errs[i].error <= errs[i - 1].error + 1e-12);
    CHECK(errs.front().error < 1.0 && errs.front().error > errs[3].error);
    CHECK(errs.back().error < 1e-10);  // full rank reproduces C
    // Eckart-Young: ||C - C_k||_F^2 = sum of squares of the discarded eigenvalues.
    const Vector ev = eigh(c).values;  // ascending => the 100-k smallest are discarded
    const int k = 10;
    CHECK_NEAR(errs[3].error, std::sqrt(ev.head(100 - k).squaredNorm()) / c.norm(), 1e-9);
}

TEST(csv_loader_roundtrip_and_errors) {
    const std::string path = (std::filesystem::temp_directory_path() / "hs_test_returns.csv").string();
    { std::ofstream f(path); f << "a,b\n0.01,0.02\n-0.01,0.03\n0.02,-0.01\n"; }
    const Matrix r = load_returns_csv(path);
    CHECK(r.rows() == 2 && r.cols() == 3);
    CHECK(r(1, 1) == 0.03 && r(0, 1) == -0.01);
    auto fails = [&](const char* content) {
        { std::ofstream f(path); f << content; }
        try { load_returns_csv(path); } catch (const std::runtime_error&) { return true; }
        return false;
    };
    CHECK(fails("0.01,0.02\n0.01\n"));         // ragged
    CHECK(fails("0.01,0.02\n0.01,\n"));        // missing cell
    CHECK(fails("0.01,0.02\n0.01,abc\n"));     // non-numeric
    CHECK(fails("0.01,0.02\n"));                // only one observation
    CHECK(fails("a,b\n0.01,nan\n0.02,0.01\n"));  // NaN rejected
    std::filesystem::remove(path);
}

#ifdef HS_WITH_CUDA
TEST(gpu_matches_cpu_when_device_present) {
    if (!cuda_available()) { std::puts("    (no CUDA device: skipped)"); return; }
    SyntheticConfig cfg;
    cfg.n_assets = 131;  // deliberately not a multiple of the 16x16 tile
    cfg.n_obs = 203;
    const Matrix r = generate_synthetic(cfg).returns;
    CHECK((correlation_gpu(r) - correlation_cpu(r)).cwiseAbs().maxCoeff() < 1e-10);
}
#endif

int main() {
    int failed = 0;
    for (const auto& t : registry()) {
        try { t.fn(); std::printf("[ OK ] %s\n", t.name); }
        catch (const std::exception& e) { ++failed; std::printf("[FAIL] %s\n       %s\n", t.name, e.what()); }
    }
    std::printf("%zu tests, %d failed\n", registry().size(), failed);
    return failed == 0 ? 0 : 1;
}
