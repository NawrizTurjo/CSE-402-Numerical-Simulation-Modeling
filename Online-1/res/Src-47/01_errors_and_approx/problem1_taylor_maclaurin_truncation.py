"""
QUESTION
--------
The function f(x) = cos(x) can be approximated near x = 0 by its Maclaurin
series:

    cos(x) = sum_{n=0}^inf  (-1)^n * x^(2n) / (2n)!

(1) Estimate cos(0.5) by adding one term at a time.
(2) After each new term, compute the absolute relative approximate error
        |eps_a| = |(S_new - S_old) / S_new| * 100 %
(3) Stop as soon as |eps_a| <= 0.5 * 10^(2-m) % for m = 4 significant
    digits (i.e. keep adding terms until at least 4 digits are trustworthy).
(4) Print a table: n, term added, partial sum, |eps_a| %, digits guaranteed
    -- exactly like the e^1.2 table shown in class.
(5) Compare the final partial sum against the exact value math.cos(0.5).
(6) Plot (i) the partial sum converging to the true value, and
    (ii) |eps_a| vs n on a log scale (semilogy). Save both as one figure
    with two subplots. Do not show the plot on screen.
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

X0 = 0.5          # point at which we evaluate cos(x)
SIG_FIGS = 4      # how many significant digits we want to guarantee


def term(n):
    """n-th term of the Maclaurin series for cos(x) at x = X0."""
    return (-1) ** n * X0 ** (2 * n) / math.factorial(2 * n)


def sig_figs_guaranteed(eps_a_percent):
    eps_a_percent = abs(eps_a_percent)
    if eps_a_percent == 0:
        return math.inf
    return math.floor(2 - math.log10(2 * eps_a_percent))


def stopping_threshold(m):
    return 0.5 * 10 ** (2 - m)


# ---------------------------------------------------------------------------
# (1) - (4): sum the series while tracking the error
# ---------------------------------------------------------------------------
threshold = stopping_threshold(SIG_FIGS)

ns, sums, term_vals, eps_as = [], [], [], []
total = 0.0
print(f"{'n':>3} {'term':>14} {'partial sum':>16} {'|eps_a| %':>14} {'sig figs':>9}")
for n in range(30):
    t = term(n)
    total_old = total
    total += t
    eps_a = None if n == 0 else abs((total - total_old) / total) * 100
    digits = "-" if eps_a is None else sig_figs_guaranteed(eps_a)

    ns.append(n)
    sums.append(total)
    term_vals.append(t)
    eps_as.append(eps_a if eps_a is not None else np.nan)

    eps_print = "-" if eps_a is None else f"{eps_a:.6f}"
    print(f"{n:>3} {t:>14.8f} {total:>16.8f} {eps_print:>14} {str(digits):>9}")

    if eps_a is not None and eps_a <= threshold:
        break

exact = math.cos(X0)
print(f"\nExact cos({X0}) = {exact:.10f}")
print(f"Series estimate = {total:.10f}  after {n+1} terms")
print(f"Stopped once |eps_a| <= {threshold:.4f}% "
      f"(needed for {SIG_FIGS} significant digits)")

# ---------------------------------------------------------------------------
# (6) Plot: convergence of the partial sum + error decay
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(ns, sums, "o-", color="tab:blue", label="partial sum")
ax1.axhline(exact, color="black", linestyle="--", linewidth=1, label="exact value")
ax1.set_xlabel("n (number of terms - 1)")
ax1.set_ylabel("approximate cos(0.5)")
ax1.set_title("Partial sum convergence")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.semilogy(ns[1:], eps_as[1:], "o-", color="tab:red")
ax2.set_xlabel("n")
ax2.set_ylabel("|eps_a| (%)  [log scale]")
ax2.set_title("Approximate error decay")
ax2.grid(True, which="both", alpha=0.3)

fig.tight_layout()
out_path = os.path.join(PLOTS_DIR, "problem1_taylor_maclaurin_truncation.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to: {out_path}")
