"""
Problem B2: Chemical Firm Daily Break-Even Point (online_B2.pdf)
Objective:
  Find production quantity x (grams) where Revenue R(x) equals Cost C(x).
  Cost C(x) = 1000 + 2x + 3x^(2/3)
  Revenue R(x) = 300x
  Non-linear Eq: f(x) = 298x - 3x^(2/3) - 1000 = 0
  Derivative: df(x) = 298 - 2x^(-1/3)

Parameters:
  - Initial Guess (x0): 4.0 grams
  - Stopping Criterion: |ea| <= 0.05%
  - Method: Newton-Raphson
"""

import math
import numpy as np
from common import P, use, banner

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
X0 = 4.0         # Initial guess (grams)
TOL = 0.05       # Stopping tolerance (%)


def f(x):
    return 298.0 * x - 3.0 * (x ** (2.0 / 3.0)) - 1000.0


def df(x):
    return 298.0 - 2.0 * (x ** (-1.0 / 3.0))


use(f, df)

banner("PROBLEM B2: CHEMICAL FIRM BREAK-EVEN POINT (NEWTON-RAPHSON)")
print(f"  Initial guess x0 = {X0} grams     Tolerance = {TOL}%\n")

root, history = P._newton_raphson_core(X0, m=1, tol=TOL, max_iter=20, verbose=False)

print(f"{'Iter':<6}{'Estimate (x_i)':<18}{'f(x_i)':<15}{'f_prime(x_i)':<15}{'Error ea (%)':<15}")
print("-" * 70)
for h in history:
    ea_str = "Initial" if h['iter'] == 1 else f"{h['ea']:.4f}%"
    print(f"{h['iter']:<6}{h['xi1']:<18.6f}{h['fxi']:<15.6f}{h['dfxi']:<15.6f}{ea_str:<15}")
print("-" * 70)

print(f"\n  Converged to break-even point x ~ {root:.6f} grams in {len(history)} iterations.")
print("  Residual check:", P.classify_and_verify(root, 'newton_raphson', residual_tol=1e-4))

f_np = np.vectorize(f)
P.plot_with_root(f_np, 1.0, 10.0, root=root, title="Problem B2: Chemical Firm Break-Even Point", fname="B2_breakeven_plot.png")
