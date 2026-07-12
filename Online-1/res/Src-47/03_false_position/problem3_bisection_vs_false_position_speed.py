"""
QUESTION  (the "is false position always faster than bisection?" experiment
shown in class)
------------------------------------------------------------------------
    f(x) = ln(x),   exact root x = 1,   bracket [1e-4, 1e4]
    stopping tolerance |eps_a| <= 0.0001%

Because this bracket is extremely lopsided (ln(x) is very steep and very
negative near x = 1e-4, but only mildly positive out near x = 1e4), one
endpoint of the false-position bracket stagnates for a long time.

(1) Run bisection on this bracket and count the iterations needed.
(2) Run false position on the same bracket and count the iterations
    needed.
(3) Print both iteration counts and state which method was faster here
    -- contrary to the usual intuition that false position "should"
    always beat bisection.
(4) Plot both methods' |eps_a| history on the same semilogy plot, so the
    difference in convergence speed is visible.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "false_position")
os.makedirs(PLOTS_DIR, exist_ok=True)


def f(x):
    return math.log(x)


TOL = 0.0001   # percent


def bisection(func, xl, xu, tol, max_iter=500):
    xm_old = None
    eps_history = []
    for i in range(1, max_iter + 1):
        xm = (xl + xu) / 2
        fxm = func(xm)
        if xm_old is not None:
            eps_history.append(abs((xm - xm_old) / xm) * 100)
        if func(xl) * fxm < 0:
            xu = xm
        else:
            xl = xm
        if eps_history and eps_history[-1] < tol:
            break
        xm_old = xm
    return xm, i, eps_history


def false_position(func, xl, xu, tol, max_iter=500):
    xr_old = None
    eps_history = []
    for i in range(1, max_iter + 1):
        xr = xu - func(xu) * (xl - xu) / (func(xl) - func(xu))
        fxr = func(xr)
        if xr_old is not None:
            eps_history.append(abs((xr - xr_old) / xr) * 100)
        if func(xl) * fxr < 0:
            xu = xr
        else:
            xl = xr
        if eps_history and eps_history[-1] < tol:
            break
        xr_old = xr
    return xr, i, eps_history


# ---------------------------------------------------------------------------
# (1) - (3)
# ---------------------------------------------------------------------------
xl, xu = 1e-4, 1e4

root_b, iters_b, eps_b = bisection(f, xl, xu, TOL)
root_fp, iters_fp, eps_fp = false_position(f, xl, xu, TOL)

print(f"Bisection:      root = {root_b:.8f}   iterations = {iters_b}")
print(f"False Position: root = {root_fp:.8f}   iterations = {iters_fp}")

if iters_b < iters_fp:
    print(f"\n=> Bisection was faster ({iters_b} vs {iters_fp} iterations) on this "
          f"lopsided bracket -- false position is NOT always faster!")
else:
    print(f"\n=> False Position was faster here ({iters_fp} vs {iters_b} iterations).")

# ---------------------------------------------------------------------------
# (4) plot convergence comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(range(1, len(eps_b) + 1), eps_b, "o-", color="tab:blue",
            label=f"bisection ({iters_b} iters)")
ax.semilogy(range(1, len(eps_fp) + 1), eps_fp, "s-", color="tab:red",
            label=f"false position ({iters_fp} iters)")
ax.set_xlabel("iteration")
ax.set_ylabel("|eps_a| (%)  [log scale]")
ax.set_title("Bisection vs False Position on f(x)=ln(x), bracket [1e-4, 1e4]")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem3_bisection_vs_false_position_speed.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to: {out_path}")
