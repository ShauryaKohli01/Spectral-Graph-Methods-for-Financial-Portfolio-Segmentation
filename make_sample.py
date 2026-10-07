"""Writes examples/sample_returns.csv: 60 synthetic daily-style returns for 9 assets in
3 planted groups (columns A0..A8, group = index mod 3). NOT real market data."""
import numpy as np

rng = np.random.default_rng(2024)
f = rng.normal(0, 0.010, size=(60, 3))
x = np.stack([f[:, i % 3] * (0.8 + 0.1 * (i // 3)) + rng.normal(0, 0.006, 60) for i in range(9)], axis=1)
with open("examples/sample_returns.csv", "w") as fh:
    fh.write(",".join(f"A{i}" for i in range(9)) + "\n")
    for row in x:
        fh.write(",".join(f"{v:.6f}" for v in row) + "\n")
