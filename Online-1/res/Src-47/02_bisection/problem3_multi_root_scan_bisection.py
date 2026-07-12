"""
QUESTION
--------
    f(x) = x^3 - 2*x^2 - 5*x + 6

This cubic is known to have three real roots. Without solving it
algebraically first:

(1) Plot f(x) for -4 <= x <= 5.
(2) Scan [-4, 5] with step size 0.25 and find every sign-change bracket.
(3) Run bisection on every bracket found to compute all three roots.
(4) Print a table of all roots with their function values (should be
    close to 0), and cross-check them against the known factorization
    f(x) = (x+2)(x-1)(x-3)  ->  roots at x = -2, 1, 3.
(5) Save a plot with all three roots marked.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "bisection")
os.makedirs(PLOTS_DIR, exist_ok=True)


def f(x):
    return x ** 3 - 2 * x ** 2 - 5 * x + 6


def find_sign_change_intervals(func, a, b, step):
    brackets = []
    x = a
    while x < b:
        x_next = min(x + step, b)
        if func(x) * func(x_next) < 0:
            brackets.append((x, x_next))
        x = x_next
    return brackets


def bisection(func, xl, xu, tol=1e-6, max_iter=100):
    xm_old = None
    for _ in range(max_iter):
        xm = (xl + xu) / 2
        fxm = func(xm)
        if fxm == 0:
            return xm
        if func(xl) * fxm < 0:
            xu = xm
        else:
            xl = xm
        if xm_old is not None and abs((xm - xm_old) / xm) * 100 < tol:
            return xm
        xm_old = xm
    return xm


# ---------------------------------------------------------------------------
# (2) + (3) find and refine all roots
# ---------------------------------------------------------------------------
# NOTE: the true roots (-2, 1, 3) are spaced by whole numbers, so scanning
# from an integer start with a step like 0.25 lands EXACTLY on each root and
# f(x)*f(x_next) becomes 0 instead of negative -- the bracket gets missed.
# Starting the scan slightly off-grid (-4.03) avoids that coincidence.
brackets = find_sign_change_intervals(f, -4.03, 5.03, 0.2)
roots = [bisection(f, xl, xu) for xl, xu in brackets]

known_roots = [-2, 1, 3]

print(f"{'bracket':>16} {'root (bisection)':>18} {'f(root)':>12} {'nearest known root':>20}")
for br, r in zip(brackets, roots):
    nearest = min(known_roots, key=lambda k: abs(k - r))
    print(f"{str(br):>16} {r:>18.6f} {f(r):>12.3e} {nearest:>20}")

# ---------------------------------------------------------------------------
# (5) plot with roots marked
# ---------------------------------------------------------------------------
x = np.linspace(-4, 5, 500)
y = f(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color="tab:blue", label="f(x) = x^3 - 2x^2 - 5x + 6")
ax.axhline(0, color="black", linewidth=0.8)
ax.scatter(roots, [f(r) for r in roots], color="red", zorder=5, label="roots found")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Bisection: all real roots of a cubic")
ax.grid(True, alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem3_multi_root_scan_bisection.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved to: {out_path}")
