"""
TEMPLATE: False Position (Regula Falsi) Method
-----------------------------------------------
Same overall structure as bisection, EXCEPT the new estimate is the point
where the straight line (secant) joining (xl, f(xl)) and (xu, f(xu))
crosses the x-axis, instead of the plain midpoint.

Formula:
    xr = xu - f(xu) * (xl - xu) / (f(xl) - f(xu))

Algorithm:
  1. Choose xl, xu with f(xl)*f(xu) < 0.
  2. Estimate the root: xr = xu - f(xu)*(xl - xu) / (f(xl) - f(xu))
  3. Decide the new bracket:
       if f(xl)*f(xr) < 0 : root in [xl, xr]  -> set xu = xr
       if f(xl)*f(xr) > 0 : root in [xr, xu]  -> set xl = xr
       if f(xl)*f(xr) == 0: xr is the root -> stop
  4. |eps_a| = |(xr_new - xr_old) / xr_new| * 100 %
  5. Repeat from step 2 until |eps_a| <= tol (or max_iter reached).

Known quirk (seen in class): one endpoint can get "stuck" for many
iterations if the function is much more curved on that side -- this can
occasionally make false position converge SLOWER than plain bisection.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "false_position")
os.makedirs(PLOTS_DIR, exist_ok=True)


def false_position(f, xl, xu, tol=1e-6, max_iter=100, verbose=False):
    if f(xl) * f(xu) > 0:
        raise ValueError("f(xl) and f(xu) must have opposite signs.")

    table = []
    xr_old = None
    for i in range(1, max_iter + 1):
        xr = xu - f(xu) * (xl - xu) / (f(xl) - f(xu))
        fxr = f(xr)
        eps_a = None if xr_old is None else abs((xr - xr_old) / xr) * 100

        table.append({"iter": i, "xl": xl, "xu": xu, "xr": xr,
                       "f(xr)": fxr, "eps_a_%": eps_a})
        if verbose:
            print(table[-1])

        if fxr == 0:
            break
        if f(xl) * fxr < 0:
            xu = xr
        else:
            xl = xr

        if eps_a is not None and eps_a < tol:
            break
        xr_old = xr

    return xr, table


if __name__ == "__main__":
    # quick self-test: f(x) = x^2 - 4  ->  root at x = 2
    f = lambda x: x ** 2 - 4
    root, table = false_position(f, 0, 5, tol=1e-6, verbose=True)
    print(f"\nRoot ~= {root:.6f}  (exact = 2.0)  in {len(table)} iterations")
