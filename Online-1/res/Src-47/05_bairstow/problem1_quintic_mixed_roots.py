"""
QUESTION  (a classic textbook polynomial with a mix of real and
complex-conjugate roots -- a standard Bairstow's method example)
------------------------------------------------------------------------
    f(x) = x^5 - 3.5*x^4 + 2.75*x^3 + 2.125*x^2 - 3.875*x + 1.25

(1) Use Bairstow's method (starting guess r0 = s0 = -1.0) to extract
    quadratic factors one at a time until the whole polynomial is
    deflated down to degree <= 2.
(2) From each quadratic factor x^2 - r*x - s, recover its two roots with
    the quadratic formula (they may be complex).
(3) Print every root found, separating real roots from complex-conjugate
    pairs.
(4) Cross-check the answer against numpy.roots(coeffs) -- they should
    match (numpy uses a different algorithm, eigenvalues of the companion
    matrix, so agreement is a good correctness check on our Bairstow code).
(5) Plot f(x) over a real range and mark the REAL roots only (complex
    roots don't show up on a real-x-axis plot).
"""

import os
import cmath
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "bairstow")
os.makedirs(PLOTS_DIR, exist_ok=True)

COEFFS = [1, -3.5, 2.75, 2.125, -3.875, 1.25]   # highest degree first


def f(x):
    return np.polyval(COEFFS, x)


# ---------------------------------------------------------------------------
# Bairstow's method building blocks
# ---------------------------------------------------------------------------
def bairstow_quadratic_factor(a, r0=-1.0, s0=-1.0, tol=1e-9, max_iter=200):
    n = len(a) - 1
    r, s = r0, s0
    for _ in range(max_iter):
        b = [0.0] * (n + 1)
        b[0] = a[0]
        b[1] = a[1] + r * b[0]
        for i in range(2, n + 1):
            b[i] = a[i] + r * b[i - 1] + s * b[i - 2]

        c = [0.0] * (n + 1)
        c[0] = b[0]
        c[1] = b[1] + r * c[0]
        for i in range(2, n):
            c[i] = b[i] + r * c[i - 1] + s * c[i - 2]

        A = np.array([[c[n - 2], c[n - 3]], [c[n - 1], c[n - 2]]])
        rhs = np.array([-b[n - 1], -b[n]])
        dr, ds = np.linalg.solve(A, rhs)
        r, s = r + dr, s + ds
        if abs(dr) < tol and abs(ds) < tol:
            break

    b = [0.0] * (n + 1)
    b[0] = a[0]
    b[1] = a[1] + r * b[0]
    for i in range(2, n + 1):
        b[i] = a[i] + r * b[i - 1] + s * b[i - 2]
    return r, s, b[0:n - 1]


def quadratic_roots(r, s):
    disc = r ** 2 + 4 * s
    sq = cmath.sqrt(disc)
    return (r + sq) / 2, (r - sq) / 2


# ---------------------------------------------------------------------------
# (1) - (3) deflate the whole quintic
# ---------------------------------------------------------------------------
a = [float(c) for c in COEFFS]
all_roots = []
print("Deflation steps:")
while len(a) - 1 > 2:
    r, s, a = bairstow_quadratic_factor(a)
    x1, x2 = quadratic_roots(r, s)
    all_roots.extend([x1, x2])
    print(f"  quadratic factor x^2 - ({r:.6f})x - ({s:.6f})  ->  roots {x1:.6f}, {x2:.6f}")
    print(f"  remaining (deflated) coefficients: {a}")

if len(a) - 1 == 2:
    r, s = -a[1] / a[0], -a[2] / a[0]
    x1, x2 = quadratic_roots(r, s)
    all_roots.extend([x1, x2])
elif len(a) - 1 == 1:
    all_roots.append(-a[1] / a[0])

print("\nAll roots found by Bairstow's method:")
real_roots, complex_roots = [], []
for root in all_roots:
    if abs(root.imag) < 1e-6:
        real_roots.append(root.real)
    else:
        complex_roots.append(root)
print("  Real roots     :", [round(r, 6) for r in real_roots])
print("  Complex roots   :", [complex(round(r.real, 6), round(r.imag, 6)) for r in complex_roots])

# ---------------------------------------------------------------------------
# (4) cross-check with numpy
# ---------------------------------------------------------------------------
print("\nCross-check with numpy.roots():")
print(" ", np.roots(COEFFS))

# ---------------------------------------------------------------------------
# (5) plot with real roots marked
# ---------------------------------------------------------------------------
x = np.linspace(-2, 3, 500)
y = f(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color="tab:blue", label="f(x) (quintic)")
ax.axhline(0, color="black", linewidth=0.8)
if real_roots:
    ax.scatter(real_roots, [f(r) for r in real_roots], color="red", zorder=5,
               label="real roots")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Bairstow's method: quintic with mixed real/complex roots")
ax.grid(True, alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem1_quintic_mixed_roots.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved to: {out_path}")
