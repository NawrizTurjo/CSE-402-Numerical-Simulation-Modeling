"""
QUESTION  (comparing Bairstow's method against Newton-Raphson on the same
polynomial -- good for a "which method would you use and why" question)
------------------------------------------------------------------------
    f(x) = x^3 - 6*x^2 + 11*x - 6      (known roots: 1, 2, 3)

(1) Find all three roots using Bairstow's method (works directly on the
    polynomial coefficients, no derivative needed, no initial guess per
    root).
(2) Find all three roots using Newton-Raphson, trying several different
    initial guesses and collecting the distinct roots (as in the
    Newton-Raphson topic folder).
(3) Print both sets of roots side by side and confirm they agree.
(4) Discuss (short printed note): Bairstow needs only ONE starting guess
    (r0, s0) and automatically finds every root by deflation, whereas
    Newton-Raphson needs a separate good starting guess for every
    individual root and provides no built-in way to know how many roots
    to look for.
(5) Plot f(x) with all roots marked.
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

COEFFS = [1, -6, 11, -6]


def f(x):
    return np.polyval(COEFFS, x)


def fprime(x):
    return 3 * x ** 2 - 12 * x + 11


# ---------------------------------------------------------------------------
# (1) Bairstow's method
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


a = [float(c) for c in COEFFS]
bairstow_roots = []
while len(a) - 1 > 2:
    r, s, a = bairstow_quadratic_factor(a)
    x1, x2 = quadratic_roots(r, s)
    bairstow_roots.extend([x1.real, x2.real])
if len(a) - 1 == 2:
    r, s = -a[1] / a[0], -a[2] / a[0]
    x1, x2 = quadratic_roots(r, s)
    bairstow_roots.extend([x1.real, x2.real])
elif len(a) - 1 == 1:
    bairstow_roots.append(-a[1] / a[0])
bairstow_roots = sorted(round(r, 6) for r in bairstow_roots)

# ---------------------------------------------------------------------------
# (2) Newton-Raphson from several starting guesses
# ---------------------------------------------------------------------------
def newton_raphson(x0, tol=1e-8, max_iter=100):
    x = x0
    for _ in range(max_iter):
        dfx = fprime(x)
        if abs(dfx) < 1e-12:
            return None
        x_new = x - f(x) / dfx
        if abs(x_new) > 1e6:
            return None
        if abs((x_new - x) / x_new) * 100 < tol:
            return x_new
        x = x_new
    return x


newton_roots = set()
for x0 in range(-2, 7):
    root = newton_raphson(x0)
    if root is not None:
        newton_roots.add(round(root, 4))
newton_roots = sorted(newton_roots)

# ---------------------------------------------------------------------------
# (3) + (4) compare
# ---------------------------------------------------------------------------
print(f"Bairstow's method roots      : {bairstow_roots}")
print(f"Newton-Raphson (multi-guess) : {newton_roots}")
print("\nDiscussion:")
print(" - Bairstow's method needed only ONE starting guess (r0=s0=-1) and")
print("   automatically produced all 3 roots through repeated deflation.")
print(" - Newton-Raphson needed 9 different starting guesses (x0 = -2..6)")
print("   and we had to collect + de-duplicate the results ourselves --")
print("   it has no built-in notion of 'how many roots does this")
print("   polynomial have'.")

# ---------------------------------------------------------------------------
# (5) plot
# ---------------------------------------------------------------------------
x = np.linspace(-1, 5, 500)
y = f(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color="tab:blue", label="f(x) = x^3 - 6x^2 + 11x - 6")
ax.axhline(0, color="black", linewidth=0.8)
ax.scatter(bairstow_roots, [f(r) for r in bairstow_roots], color="red",
           zorder=5, label="roots (Bairstow & Newton-Raphson agree)")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Bairstow's method vs Newton-Raphson: same cubic, same roots")
ax.grid(True, alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem3_cubic_bairstow_vs_newton.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved to: {out_path}")
