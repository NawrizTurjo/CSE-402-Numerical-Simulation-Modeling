"""
QUESTION
--------
    f(x)  = x^3 - 6*x^2 + 11*x - 6
    f'(x) = 3*x^2 - 12*x + 11

This cubic has three real roots (1, 2, 3), but Newton-Raphson only ever
finds ONE root per starting guess. To find ALL roots:

(1) Try a range of different initial guesses x0 = -2, -1, ..., 6 (step 1).
(2) Run Newton-Raphson from each x0 and collect the root it converges to
    (skip any guess where it diverges or fails, e.g. f'(x0) = 0).
(3) Round and remove duplicate roots to get the distinct set of roots.
(4) Print a table showing which starting guess led to which root.
(5) Plot f(x) with all distinct roots marked.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "newton_raphson")
os.makedirs(PLOTS_DIR, exist_ok=True)


def f(x):
    return x ** 3 - 6 * x ** 2 + 11 * x - 6


def fprime(x):
    return 3 * x ** 2 - 12 * x + 11


def newton_raphson(x0, tol=1e-8, max_iter=100):
    x = x0
    for _ in range(max_iter):
        dfx = fprime(x)
        if abs(dfx) < 1e-12:
            return None   # derivative too small -- treat as failure
        x_new = x - f(x) / dfx
        if abs(x_new) > 1e6:
            return None   # diverged
        if abs((x_new - x) / x_new) * 100 < tol:
            return x_new
        x = x_new
    return x


# ---------------------------------------------------------------------------
# (1) - (3) scan several initial guesses
# ---------------------------------------------------------------------------
initial_guesses = list(range(-2, 7))   # -2, -1, 0, ..., 6
found_roots = []

print(f"{'x0':>6} {'converged root':>16}")
for x0 in initial_guesses:
    root = newton_raphson(x0)
    root_str = "diverged" if root is None else f"{root:.6f}"
    print(f"{x0:>6} {root_str:>16}")
    if root is not None:
        found_roots.append(root)

# keep only distinct roots (rounded to 4 decimals to merge near-duplicates)
distinct_roots = sorted(set(round(r, 4) for r in found_roots))
print(f"\nDistinct roots found: {distinct_roots}")
print("Known exact roots: [1, 2, 3]")

# ---------------------------------------------------------------------------
# (5) plot with all distinct roots marked
# ---------------------------------------------------------------------------
x = np.linspace(-2, 6, 500)
y = f(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color="tab:blue", label="f(x) = x^3 - 6x^2 + 11x - 6")
ax.axhline(0, color="black", linewidth=0.8)
ax.scatter(distinct_roots, [f(r) for r in distinct_roots], color="red",
           zorder=5, label="roots found")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Newton-Raphson: finding all roots via multiple starting guesses")
ax.grid(True, alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem3_multi_root_scan_newton_raphson.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to: {out_path}")
