"""
QUESTION
--------
f(x) = x^2,   exact integral over [1, 4]:

    integral_1^4 x^2 dx = [x^3/3]_1^4 = (64 - 1)/3 = 21.0

(1) Approximate this integral using the LEFT-endpoint rectangle rule with
    n = 2, 4, 8, 16, 32, 64, 128 subintervals.
(2) For each n, compute the true error (exact - approximate).
(3) Using consecutive estimates (n and 2n), compute the absolute relative
    approximate error |eps_a| and determine how many significant digits
    are guaranteed at each doubling, using
        |eps_a| <= 0.5 * 10^(2-m)  =>  m = floor(2 - log10(2*|eps_a|))
(4) Print a table: n, approx integral, true error, |eps_a| %, sig figs.
(5) Plot the true error against n on a log-log scale to show that the
    error shrinks roughly like 1/n as the rectangles get thinner (matches
    the "more rectangles -> smaller truncation error" idea from class).
    Save the plot without displaying it.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "errors")
os.makedirs(PLOTS_DIR, exist_ok=True)

A, B = 1.0, 4.0


def f(x):
    return x ** 2


def exact_integral():
    return (B ** 3 - A ** 3) / 3


def left_rectangle_rule(n):
    x = np.linspace(A, B, n, endpoint=False)   # left edges of each strip
    width = (B - A) / n
    return np.sum(f(x)) * width


def sig_figs_guaranteed(eps_a_percent):
    eps_a_percent = abs(eps_a_percent)
    if eps_a_percent == 0:
        return math.inf
    return math.floor(2 - math.log10(2 * eps_a_percent))


# ---------------------------------------------------------------------------
# (1) - (4)
# ---------------------------------------------------------------------------
exact = exact_integral()
ns = [2, 4, 8, 16, 32, 64, 128]

approx_prev = None
true_errors = []
print(f"{'n':>5} {'approx':>12} {'true error':>12} {'|eps_a| %':>12} {'sig figs':>9}")
for n in ns:
    approx = left_rectangle_rule(n)
    true_err = exact - approx
    true_errors.append(abs(true_err))

    if approx_prev is None:
        eps_a_str, digits_str = "-", "-"
    else:
        eps_a = abs((approx - approx_prev) / approx) * 100
        eps_a_str = f"{eps_a:.6f}"
        digits_str = str(sig_figs_guaranteed(eps_a))

    print(f"{n:>5} {approx:>12.6f} {true_err:>12.6f} {eps_a_str:>12} {digits_str:>9}")
    approx_prev = approx

print(f"\nExact integral = {exact}")

# ---------------------------------------------------------------------------
# (5) log-log plot of true error vs n
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(ns, true_errors, "o-", color="tab:purple")
ax.set_xlabel("number of subintervals, n  [log scale]")
ax.set_ylabel("|true error|  [log scale]")
ax.set_title("Left-rectangle-rule truncation error for integral of x^2 on [1,4]")
ax.grid(True, which="both", alpha=0.3)

out_path = os.path.join(PLOTS_DIR, "problem3_integration_truncation_sigfigs.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to: {out_path}")
