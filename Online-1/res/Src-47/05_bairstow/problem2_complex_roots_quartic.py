"""
QUESTION  (a clean example where the exact roots are known in advance,
useful for verifying your Bairstow code is correct)
------------------------------------------------------------------------
    f(x) = x^4 - 1

By factoring, x^4 - 1 = (x-1)(x+1)(x^2+1), so the exact roots are
1, -1, i, -i -- two real roots and one complex-conjugate pair.

(1) Run Bairstow's method on coeffs = [1, 0, 0, 0, -1] to extract all
    four roots.
(2) Print the roots, clearly separating real roots from the
    complex-conjugate pair.
(3) Verify each root by plugging it back into f(x) and confirming the
    result is (numerically) zero.
(4) Plot f(x) over a real range and mark only the two real roots.
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

COEFFS = [1, 0, 0, 0, -1]   # x^4 - 1


def f(x):
    return np.polyval(COEFFS, x)


def bairstow_quadratic_factor(a, r0=-1.0, s0=-1.0, tol=1e-10, max_iter=200):
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
# (1) deflate down to two quadratics
# ---------------------------------------------------------------------------
a = [float(c) for c in COEFFS]
all_roots = []
while len(a) - 1 > 2:
    r, s, a = bairstow_quadratic_factor(a)
    x1, x2 = quadratic_roots(r, s)
    all_roots.extend([x1, x2])

if len(a) - 1 == 2:
    r, s = -a[1] / a[0], -a[2] / a[0]
    x1, x2 = quadratic_roots(r, s)
    all_roots.extend([x1, x2])

# ---------------------------------------------------------------------------
# (2) + (3) print and verify
# ---------------------------------------------------------------------------
print("Roots of x^4 - 1 found by Bairstow's method:")
real_roots = []
for root in all_roots:
    check = f(root)
    if abs(root.imag) < 1e-6:
        real_roots.append(root.real)
        print(f"  x = {root.real: .6f}            f(x) = {check:.2e}  (real root)")
    else:
        print(f"  x = {root.real: .6f} {root.imag:+.6f}i   f(x) = {check:.2e}  (complex root)")

print("\nExpected exact roots: 1, -1, i, -i")

# ---------------------------------------------------------------------------
# (4) plot with real roots marked
# ---------------------------------------------------------------------------
x = np.linspace(-2, 2, 500)
y = f(x)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, y, color="tab:blue", label="f(x) = x^4 - 1")
ax.axhline(0, color="black", linewidth=0.8)
ax.scatter(real_roots, [f(r) for r in real_roots], color="red", zorder=5,
           label="real roots (+/-1)")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Bairstow's method: x^4 - 1 (2 real + 1 complex-conjugate pair)")
ax.grid(True, alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem2_complex_roots_quartic.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved to: {out_path}")
