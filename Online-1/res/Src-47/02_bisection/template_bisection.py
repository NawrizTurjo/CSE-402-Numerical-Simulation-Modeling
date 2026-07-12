"""
TEMPLATE: Bisection Method
--------------------------
General-purpose bisection template. Copy this file, change FUNCTION and the
interval, run it.

Algorithm (as taught in class):
  1. Choose xl, xu such that f(xl)*f(xu) < 0 (a sign change).
  2. Estimate the root: xm = (xl + xu) / 2.
  3. Decide the new bracket:
       if f(xl)*f(xm) < 0 : root is in [xl, xm]  -> set xu = xm
       if f(xl)*f(xm) > 0 : root is in [xm, xu]  -> set xl = xm
       if f(xl)*f(xm) == 0: xm is the root -> stop
  4. Compute |eps_a| = |(xm_new - xm_old) / xm_new| * 100 %
  5. Repeat from step 2 until |eps_a| <= tol (or max_iter reached).
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "bisection")
os.makedirs(PLOTS_DIR, exist_ok=True)


def bisection(f, xl, xu, tol=1e-6, max_iter=100, verbose=False):
    """
    Classic bisection method.
    Returns: root, table (list of dict rows, one per iteration)
    """
    if f(xl) * f(xu) > 0:
        raise ValueError("f(xl) and f(xu) must have opposite signs.")

    table = []
    xm_old = None
    for i in range(1, max_iter + 1):
        xm = (xl + xu) / 2
        fxm = f(xm)
        eps_a = None if xm_old is None else abs((xm - xm_old) / xm) * 100

        table.append({"iter": i, "xl": xl, "xu": xu, "xm": xm,
                       "f(xm)": fxm, "eps_a_%": eps_a})
        if verbose:
            print(table[-1])

        if fxm == 0:
            break
        if f(xl) * fxm < 0:
            xu = xm
        else:
            xl = xm

        if eps_a is not None and eps_a < tol:
            break
        xm_old = xm

    return xm, table


def find_sign_change_intervals(f, a, b, step):
    """Scan [a, b] with the given step size and return every [x, x+step]
    bracket where f changes sign (f(x)*f(x+step) < 0)."""
    brackets = []
    x = a
    while x < b:
        x_next = min(x + step, b)
        if f(x) * f(x_next) < 0:
            brackets.append((x, x_next))
        x = x_next
    return brackets


if __name__ == "__main__":
    # quick self-test: f(x) = x^2 - 4  ->  root at x = 2
    f = lambda x: x ** 2 - 4
    root, table = bisection(f, 0, 5, tol=1e-6, verbose=True)
    print(f"\nRoot ~= {root:.6f}  (exact = 2.0)")
