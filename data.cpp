#include "data.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace hs {

SyntheticData generate_synthetic(const SyntheticConfig& cfg) {
    if (cfg.n_sectors < 1 || cfg.n_assets < cfg.n_sectors || cfg.n_obs < 2)
        throw std::invalid_argument("need n_sectors >= 1, n_assets >= n_sectors, n_obs >= 2");
    if (cfg.sector_vol < 0.0 || cfg.idio_vol < 0.0)
        throw std::invalid_argument("volatilities must be non-negative");

    std::mt19937_64 rng(cfg.seed);
    const int n = cfg.n_assets, t = cfg.n_obs, k = cfg.n_sectors;

    // The draw order below is part of the reproducibility contract: factors, betas, noise.
    Matrix factors(k, t);
    for (int s = 0; s < k; ++s)
        for (int j = 0; j < t; ++j) factors(s, j) = cfg.sector_vol * standard_normal(rng);

    Vector beta(n);
    for (int i = 0; i < n; ++i) beta(i) = 0.5 + uniform01(rng);

    SyntheticData out;
    out.returns.resize(n, t);
    out.labels.resize(n);
    for (int i = 0; i < n; ++i) {
        const int s = i % k;
        out.labels[i] = s;
        for (int j = 0; j < t; ++j)
            out.returns(i, j) = beta(i) * factors(s, j) + cfg.idio_vol * standard_normal(rng);
    }
    return out;
}

Matrix load_returns_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);

    std::vector<std::vector<double>> rows;
    std::string line;
    std::size_t lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;  // blank line

        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        bool numeric = true;
        while (std::getline(ss, cell, ',')) {
            char* end = nullptr;
            const double v = std::strtod(cell.c_str(), &end);
            const bool empty = cell.find_first_not_of(" \t\r") == std::string::npos;
            if (empty || end == cell.c_str() || std::isnan(v)) { numeric = false; break; }
            while (*end == ' ' || *end == '\t' || *end == '\r') ++end;
            if (*end != '\0') { numeric = false; break; }
            row.push_back(v);
        }
        if (!numeric) {
            if (rows.empty() && lineno == 1) continue;  // treat first line as header
            throw std::runtime_error(path + ":" + std::to_string(lineno) +
                                     ": non-numeric or missing value");
        }
        if (!rows.empty() && row.size() != rows.front().size())
            throw std::runtime_error(path + ":" + std::to_string(lineno) + ": ragged row");
        rows.push_back(std::move(row));
    }
    if (rows.size() < 2 || rows.front().empty())
        throw std::runtime_error(path + ": need at least 2 observations and 1 asset");

    const int t = static_cast<int>(rows.size());
    const int n = static_cast<int>(rows.front().size());
    Matrix r(n, t);
    for (int j = 0; j < t; ++j)
        for (int i = 0; i < n; ++i) r(i, j) = rows[j][i];
    return r;
}

}  // namespace hs
