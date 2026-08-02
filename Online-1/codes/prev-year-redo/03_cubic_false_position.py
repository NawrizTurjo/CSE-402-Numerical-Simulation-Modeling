"""
Problem A1: Cubic Equation Solver (online_A1.pdf)
Objective:
  Find real root for f(x) = x^3 - x - 1 = 0 over interval [1.0, 2.0].

Parameters:
  - Bracket: xl = 1.0, xu = 2.0
  - Stopping Criterion: |ea| <= 0.05%
  - Method: False Position (Regula Falsi)
"""

import math
import numpy as np
from common import P, use, banner

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
XL = 1.0
XU = 2.0
TOL = 0.05       # Stopping tolerance (%)


def f(x):
    return x**3 - x - 1.0


def df(x):
    return 3.0 * x**2 - 1.0


use(f, df)

banner("PROBLEM A1: CUBIC EQUATION SOLVER (FALSE POSITION)")
print(f"  Initial bracket: [{XL}, {XU}]     Tolerance = {TOL}%\n")

res = P.false_position_opt(XL, XU, tol=TOL, max_iter=50, verbose=False)

if res is not None:
    root, iters, ea, lu, uu, history = res

    print(f"{'Iter':<6}{'x_L':<12}{'x_U':<12}{'x_r':<14}{'Error ea (%)':<16}{'f(x_r)':<14}")
    print("-" * 74)
    for h in history:
        ea_str = "Initial" if h['iter'] == 1 else f"{h['ea']:.4f}%"
        print(f"{h['iter']:<6}{h['xl']:<12.4f}{h['xu']:<12.4f}{h['xr']:<14.6f}{ea_str:<16}{h['fxr']:<14.4e}")
    print("-" * 74)

    print(f"\n  Converged to real root x ~ {root:.6f} in {iters} iterations.")
    print(f"  xl updates: {lu}, xu updates: {uu}")
    print("  Residual check:", P.classify_and_verify(root, 'false_position', residual_tol=1e-4))

    f_np = np.vectorize(f)
    P.plot_with_root(f_np, 0.0, 2.5, root=root, xl=XL, xu=XU, title="Problem A1: Cubic Equation Root", fname="A1_cubic_plot.png")
