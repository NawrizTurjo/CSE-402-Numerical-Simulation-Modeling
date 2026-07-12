"""
TEMPLATE: Approximations & Errors (true error, approximate error,
significant figures, truncation error, round-off error)
--------------------------------------------------------------------
Formulas covered (exactly as given in class):

  True error:                 E_t = true value - approximate value
  Relative true error:        eps_t = E_t / true value          (x100 for %)

  Approximate error:          E_a = current approx - previous approx
  Relative approximate error: eps_a = E_a / current approx       (x100 for %)

  Stopping criterion (guarantee at least m significant digits):
        |eps_a| (%) <= 0.5 * 10^(2 - m)

  Number of significant digits guaranteed by a given |eps_a| (%):
        m = floor( 2 - log10(2 * |eps_a|) )

Copy this file and reuse the helper functions below for any error-analysis
question in the lab test.
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


def save_fig(fig, filename):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {path}")


# ---------------------------------------------------------------------------
# Basic error formulas
# ---------------------------------------------------------------------------
def true_error(true_value, approx_value):
    return true_value - approx_value


def rel_true_error_percent(true_value, approx_value):
    return true_error(true_value, approx_value) / true_value * 100


def approx_error(new_value, old_value):
    return new_value - old_value


def rel_approx_error_percent(new_value, old_value):
    return approx_error(new_value, old_value) / new_value * 100


def sig_figs_guaranteed(eps_a_percent):
    """Largest m such that |eps_a| <= 0.5 * 10^(2-m) is satisfied."""
    eps_a_percent = abs(eps_a_percent)
    if eps_a_percent == 0:
        return math.inf
    return math.floor(2 - math.log10(2 * eps_a_percent))


def stopping_threshold(m):
    """The |eps_a| (%) threshold needed to guarantee m significant digits."""
    return 0.5 * 10 ** (2 - m)


# ---------------------------------------------------------------------------
# Iterative series summation until a significant-digit criterion is met
# (used for Maclaurin / Taylor series style questions)
# ---------------------------------------------------------------------------
def sum_series_until_tolerance(term_func, sig_figs_needed=3, max_terms=200):
    """
    term_func(n) must return the n-th term of the series (n = 0, 1, 2, ...).
    Returns a table (list of dict rows) and the final partial sum.
    """
    threshold = stopping_threshold(sig_figs_needed)
    total = 0.0
    table = []
    for n in range(max_terms):
        total_old = total
        total += term_func(n)
        eps_a = None if n == 0 else rel_approx_error_percent(total, total_old)
        table.append({"n": n, "term": term_func(n), "sum": total, "eps_a_%": eps_a})
        if eps_a is not None and abs(eps_a) <= threshold:
            break
    return total, table


# ---------------------------------------------------------------------------
# Finite-difference derivative (truncation error demo)
# ---------------------------------------------------------------------------
def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h


def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


# ---------------------------------------------------------------------------
# Simple numerical integration (rectangle rule) - truncation error demo
# ---------------------------------------------------------------------------
def left_rectangle_rule(f, a, b, n):
    x = np.linspace(a, b, n, endpoint=False)
    width = (b - a) / n
    return np.sum(f(x)) * width


if __name__ == "__main__":
    # quick self-test using e^x Maclaurin series at x = 1 (true value = e)
    x0 = 1.0
    term = lambda n: x0 ** n / math.factorial(n)
    total, table = sum_series_until_tolerance(term, sig_figs_needed=4)
    print(f"Approximated e^{x0} = {total:.8f}  (true e = {math.e:.8f})")
    for row in table:
        print(row)
