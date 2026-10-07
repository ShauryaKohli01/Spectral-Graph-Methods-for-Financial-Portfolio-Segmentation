# Spectral Graph Methods for Financial Portfolio Segmentation

A small C++17 research codebase that builds a financial **correlation graph**, applies **spectral
methods** to it (clustering and low-rank compression), and includes an optional hand-written
**CUDA** kernel for the correlation computation.

> **Scope and attribution.** This is an *independent educational implementation* inspired by the
> general idea of applying spectral methods to financial correlation graphs, a topic on which
> HedgeSPA has published (*Computers* 14(8):310). It is **not** a reproduction of that paper: it
> uses none of its code or data and does not implement its method or evaluation. It contains no
> trading signals and makes no performance or profitability claims.

## Motivation

Equity returns co-move through common drivers (market, sector, style). A correlation matrix encodes
this, but an $N\times N$ matrix is hard to inspect. Viewing it as a weighted graph exposes groups of
assets that move together, and the spectrum of the graph or of the matrix itself gives (i) a
principled way to cluster and (ii) a measure of how much of the matrix a few modes explain. This
repo implements both end-to-end on reproducible data.

## Methodology

**Data.** Asset $i$ in sector $s(i)=i \bmod K$ has $r_{i,t}=\beta_i f_{s(i),t}+\varepsilon_{i,t}$ with
$f_{s,t}\sim\mathcal N(0,\sigma_f^2)$, $\varepsilon_{i,t}\sim\mathcal N(0,\sigma_\varepsilon^2)$,
$\beta_i\sim U(0.5,1.5)$. A seeded `mt19937_64` with a hand-written Box-Muller transform makes the
underlying uniform stream identical on every platform; the Gaussian transform calls libm
(`log`, `cos`), so results can differ at the last-bit level between platforms.

**Correlation.** Centre each row (two-pass mean) and scale to unit norm to get $Z$; then

$$C_{ij}=\frac{\mathrm{Cov}(r_i,r_j)}{\sigma_i\sigma_j}=\big(ZZ^\top\big)_{ij}.$$

Two-pass centring avoids the cancellation in $E[x^2]-E[x]^2$ when the mean dwarfs the volatility.
The result is symmetrised, clamped to $[-1,1]$, with unit diagonal. A constant series has undefined
correlation; by convention it correlates 0 with everything. NaN/Inf input is rejected.

**Graph.** $w_{ij}=|C_{ij}|$ if $|C_{ij}|\ge\tau$, else $0$; zero diagonal; $\tau\in[0,1]$.
`--signed` uses $w_{ij}=C_{ij}$ with degrees $d_i=\sum_j|w_{ij}|$ (the signed Laplacian, which is
PSD). Signed mode is provided for experimentation only: negative edges push vertices *apart* in
the low eigenvectors, so its clustering semantics differ from the default and are not tested for
sector recovery. An optional MST uses the Mantegna distance $d_{ij}=\sqrt{2(1-C_{ij})}$
(Prim, $O(N^2)$).

**Laplacians.**

$$L=D-W,\qquad L_{\mathrm{norm}}=I-D^{-1/2}WD^{-1/2}.$$

Isolated vertices ($d_i=0$) use $D^{-1/2}_{ii}:=0$, giving $(L_{\mathrm{norm}})_{ii}=1$: they sit at
eigenvalue 1 and do not contaminate the near-zero part of the spectrum.

**Spectral clustering (Ng-Jordan-Weiss).** Take the $k$ eigenvectors of $L_{\mathrm{norm}}$ with the
*smallest* eigenvalues, stack them as columns, normalise each row to unit length, and run k-means
(k-means++ seeding, 10 restarts, seeded). For a graph with exactly $k$ connected components the $k$
smallest eigenvalues are all zero and their eigenvectors are what carries the cluster information,
so zero eigenvalues are kept ("smallest $k$", not "smallest non-zero"). If there are *more* than $k$
components, the bottom-$k$ eigenspace is an arbitrary rotation inside the null space and the result
is not meaningful; `hs_demo` prints a warning in that case. Recovery is scored with the Adjusted
Rand Index (ARI).

**Spectral compression.** With $C=V\Lambda V^\top$, keep the $k$ eigenpairs of largest $|\lambda|$:

$$C_k=V_k\Lambda_kV_k^\top,\qquad E_k=\frac{\lVert C-C_k\rVert_F}{\lVert C\rVert_F}.$$

By Eckart-Young this is the best rank-$k$ approximation in Frobenius norm and
$\lVert C-C_k\rVert_F^2=\sum_{j>k}\lambda_j^2$ (checked in the tests).

## Architecture

```
SyntheticConfig ──generate_synthetic──► returns (N x T) ◄── load_returns_csv (optional)
        │
        ▼
correlation_cpu / correlation_gpu ──► C (N x N)
        │                                   │
        ▼                                   ▼
build_adjacency(τ) ─► W              eigh(C) ─► low_rank_approximation ─► E_k table
        │
        ├─► minimum_spanning_tree (optional)
        ▼
normalized_laplacian ─► eigh ─► spectral_embedding ─► kmeans ─► labels ─► ARI vs planted
```

| File | Role |
|---|---|
| `include/data.hpp`, `src/data.cpp` | synthetic generator, portable RNG helpers, CSV loader |
| `include/correlation.hpp`, `src/correlation.cpp` | standardisation, CPU correlation, GPU dispatch |
| `include/graph.hpp`, `src/graph.cpp` | adjacency, Laplacians, MST |
| `include/spectral.hpp`, `src/spectral.cpp` | `eigh`, spectral embedding, low-rank reconstruction |
| `include/clustering.hpp`, `src/clustering.cpp` | k-means, spectral clustering, ARI |
| `include/benchmark.hpp`, `src/bench_main.cpp` | timing helpers and benchmark executable |
| `include/cuda_kernels.hpp`, `cuda/correlation.cu` | Eigen-free interface and the tiled CUDA kernel |
| `src/main.cpp` | `hs_demo` command-line pipeline |
| `tests/` | dependency-free test runner; `gen_reference.py` produces the NumPy reference values |

## Build

Requires a C++17 compiler, CMake ≥ 3.18 and Eigen 3 (`apt install libeigen3-dev`, `brew install
eigen`, or vcpkg). No other dependencies. Developed and tested with GCC 13, CMake 4.4 and Eigen
3.4.0 on Linux; other compilers and Eigen 3.3 are expected to work but have not been tested.

```bash
cmake -S . -B build            # defaults to Release, CUDA off
cmake --build build
ctest --test-dir build --output-on-failure
```

## Usage (CPU)

```bash
./build/hs_demo                                     # synthetic data, default seed 42
./build/hs_demo --assets 200 --obs 750 --sectors 8 --idio-vol 0.015 --threshold 0.2 --mst
./build/hs_demo --csv examples/sample_returns.csv --clusters 3 --threshold 0.3 --out clusters.csv
./build/hs_demo --help
```

CSV format: one row per observation, one column per asset, optional header line, numeric cells
only (no date column, no quoted fields, no missing values; a bad cell is a hard error with its line
number). `examples/sample_returns.csv` is **synthetic** (see `examples/make_sample.py`), not market
data. Nothing is downloaded.

### Example output (default synthetic data; deterministic)

```
data: 100 assets x 500 observations (synthetic), correlation on CPU
graph: threshold 0.30, |rho| weights, 887 edges (density 0.179), 0 isolated vertices
clusters requested: 5
smallest L_norm eigenvalues: 0.0000 0.0000 0.0000 0.0000 0.0000 0.8929 0.9125 0.9295
connected components among non-isolated vertices: 5
cluster sizes: 20 20 20 20 20
adjusted Rand index vs planted sectors: 1.0000
...
     k  reconstruction_error
     1  0.877276
     2  0.741517
     3  0.605823
     5  0.248945
    10  0.225296
    20  0.185745
    50  0.101132
   100  0.000000
```

**Read this result with care.** At $\tau=0.30$ every cross-sector edge is removed, so the graph is
five disconnected components and recovery is nearly trivial. A more informative setting is
`--threshold 0.10`, where noise edges connect the sectors into a single component: the smallest
eigenvalues become 0.0000, 0.0134, 0.0193, 0.0285, 0.0526 followed by 0.9994, and the clustering
must rely on the eigengap and on the eigenvectors (the unit tests cover this case).

The reconstruction error falls from 0.88 at $k=1$ to 0.25 at $k=5$ and then decays slowly. For the
default seed the five largest eigenvalues of $C$ are about 8.7-11.1 (the five planted sector
factors) and the sixth is about 1.15, the start of a noise bulk contributed by idiosyncratic
returns. The rows $k=1,2,3$ in the table (0.88, 0.74, 0.61) show that each sector factor explains a
similar share, since there is no common market factor in this generator.

## CUDA

```bash
cmake -S . -B build -DENABLE_CUDA=ON                                  # CMake >= 3.24 and a visible GPU ('native' arch)
cmake -S . -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86    # or name your GPU's SM explicitly
cmake --build build
./build/hs_demo --gpu
```

The kernel (`cuda/correlation.cu`) computes $C=ZZ^\top$ with 16×16 shared-memory tiles over the
observation axis (FP64; the `+1` tile padding avoids bank conflicts when reading the transposed
operand). Standardisation stays on the host so CPU and GPU paths share it and conventions are
identical. Errors from launch and from asynchronous execution are checked separately. With
`ENABLE_CUDA=OFF` (default) the project builds and runs completely on CPU, and `--gpu` /
`correlation_gpu` fail with a clear error instead of silently falling back.

**Status:** the CUDA path is included, but GPU performance has **not** been benchmarked on the
development machine, and no GPU timings are reported anywhere in this repository.
<!-- If you build and run on a GPU, replace this paragraph with your hardware, CUDA version and the measured table. -->

## Testing

`ctest` runs a deterministic unit-test executable (fixed seeds, no timing) plus smoke runs of
`hs_demo`. The unit tests cover: correlation against hand-computed values and against `np.corrcoef`
reference values (`tests/gen_reference.py`); standardised rows having zero mean and unit norm;
PSD and rank deficiency when $N>T$; stability with a $10^8$ mean and a constant row; rejection of
NaN, one-observation and invalid-threshold inputs; generator reproducibility; Laplacian entries,
zero row sums, component counting and isolated vertices; thresholding and signed weights; MST on a
hand-computed case and spanning-tree property; ARI invariances; NJW embedding properties; planted-sector
recovery on both a disconnected and a **connected** graph, with a **negative control** (no sector
factor) that must not be recovered; monotone reconstruction error with the Eckart-Young identity;
and CSV parsing and rejection of ragged, missing, non-numeric and NaN cells. With
`-DENABLE_CUDA=ON` an extra test compares GPU against CPU on tile-unaligned sizes (skipped if no
device is present).

## Benchmark methodology

```bash
cmake -S . -B build-native -DENABLE_NATIVE_ARCH=ON && cmake --build build-native
./build-native/hs_bench                                  # sizes 100,250,500,1000 at 2000 observations
./build-native/hs_bench --sizes 100,500 --obs 1000 --reps 9
```

Each path is warmed up once, then timed `--reps` times with `std::chrono::steady_clock`; the median
is reported. Columns:

- `cpu_ms`: end-to-end, standardise + Eigen matrix product + finalise. Single-threaded unless Eigen
  is built with OpenMP.
- `gpu_total_ms`: end-to-end, host standardise + row-major copy + `cudaMalloc` + H2D copy + kernel +
  D2H copy + finalise (FP64). `speedup = cpu_ms / gpu_total_ms`.
- `gpu_kernel_ms`: the kernel alone, from CUDA events, to separate compute from transfer cost.
- `max|diff|`: `max |C_gpu - C_cpu|`.

**Use `-DENABLE_NATIVE_ARCH=ON` for benchmarking.** Without it Eigen is compiled for baseline x86-64
and cannot use AVX/FMA, which materially slows the CPU baseline and would inflate any GPU speedup
(on one test machine the difference was over 2x at N=1000, so treat unflagged comparisons with
suspicion). Numbers depend on hardware, compiler flags and FP64 throughput (consumer GPUs have weak
FP64), so none are quoted here. Small matrices often do not benefit from the GPU because launch,
allocation and transfer overhead dominate.

## Limitations

- Synthetic planted sectors with Gaussian noise are an easy, idealised target. With a high
  threshold, edges vanish and recovery collapses; e.g. with `--idio-vol 0.02`, ARI is 1.00 at
  `--threshold 0.15` but 0.12 at `--threshold 0.30` (61 isolated vertices; the demo prints a
  warning). Threshold choice matters and is left to the user.
- Dense $O(N^3)$ eigendecomposition; fine for ~$10^3$ assets, not for $10^5$.
- Pearson correlation only: no shrinkage (Ledoit-Wolf), no random-matrix filtering, no handling of
  fat tails, non-stationarity, or asynchronous/missing data (the CSV loader rejects missing cells).
- The number of clusters $k$ is supplied by the user rather than estimated from the eigengap.
- Isolated vertices end up with a zero embedding row and are grouped arbitrarily.
- The CUDA kernel is a teaching-quality tiled kernel: FP64, computes the full matrix rather than one
  triangle, no streams, no comparison with cuBLAS.

## Relationship to the HedgeSPA research topic

HedgeSPA has published on spectral treatments of financial correlation graphs. This repo explores
the same family of ideas at textbook level: correlation graph → Laplacian spectrum → clustering and
low-rank structure. It does not implement the published compression scheme, evaluation protocol or
datasets, and should be read as a study of the underlying concepts, not as a replication.

## License

MIT, see `LICENSE`.
