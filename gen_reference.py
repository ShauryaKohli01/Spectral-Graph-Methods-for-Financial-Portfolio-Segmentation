"""Generates the NumPy reference values hard-coded in tests/test_main.cpp.
The data are rounded to 4 decimals *and re-parsed from the printed text*, so the numbers
NumPy sees are bit-identical to the literals compiled into the C++ test.
Run: python3 tests/gen_reference.py   (requires numpy)"""
import numpy as np

rng = np.random.default_rng(7)
text = [[f"{x:.4f}" for x in row] for row in rng.normal(0.0, 0.02, size=(4, 8))]
r = np.array([[float(s) for s in row] for row in text])
print("returns (rows = assets):")
for row in text:
    print("    {" + ", ".join(row) + "},")
print("np.corrcoef upper triangle (0,1) (0,2) (0,3) (1,2) (1,3) (2,3):")
c = np.corrcoef(r)
print(", ".join(f"{c[i, j]:.15f}" for i in range(4) for j in range(i + 1, 4)))
