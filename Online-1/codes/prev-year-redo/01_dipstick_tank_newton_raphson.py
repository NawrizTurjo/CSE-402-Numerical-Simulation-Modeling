"""
Problem A2: Spherical Oil Tank Dipstick Design (online_A2.pdf)
Objective:
  Find wet height h for target volume V = 5 ft^3 in a spherical tank of diameter 8 ft (radius r = 4 ft).
  Formula: V = pi * h^2 * (3*r - h) / 3
  Non-linear Eq: f(h) = pi * h^3 - 12 * pi * h^2 + 15 = 0
  Derivative: df(h) = 3 * pi * h^2 - 24 * pi * h

Parameters:
  - Initial Guess (h0): 0.5 ft
  - Stopping Criterion: |ea| <= 0.05%
  - Method: Newton-Raphson
"""

import math
import numpy as np
from common import P, use, banner

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
H0 = 0.5         # Initial guess (ft)
TOL = 0.05       # Stopping tolerance (%)


def f(h):
    return math.pi * h**3 - 12.0 * math.pi * h**2 + 15.0


def df(h):
    return 3.0 * math.pi * h**2 - 24.0 * math.pi * h


use(f, df)

banner("PROBLEM A2: SPHERICAL TANK DIPSTICK DESIGN (NEWTON-RAPHSON)")
print(f"  Initial guess h0 = {H0} ft     Tolerance = {TOL}%\n")

root, history = P._newton_raphson_core(H0, m=1, tol=TOL, max_iter=20, verbose=False)

print(f"{'Iter':<6}{'Estimate (h_i)':<18}{'f(h_i)':<15}{'f_prime(h_i)':<15}{'Error ea (%)':<15}")
print("-" * 70)
for h in history:
    ea_str = "Initial" if h['iter'] == 1 else f"{h['ea']:.4f}%"
    print(f"{h['iter']:<6}{h['xi1']:<18.6f}{h['fxi']:<15.6f}{h['dfxi']:<15.6f}{ea_str:<15}")
print("-" * 70)

print(f"\n  Converged to wet height h ~ {root:.6f} ft in {len(history)} iterations.")
print("  Residual check:", P.classify_and_verify(root, 'newton_raphson', residual_tol=1e-4))

f_np = np.vectorize(f)
P.plot_with_root(f_np, 0.0, 2.0, root=root, title="Problem A2: Spherical Tank Dipstick Height", fname="A2_dipstick_plot.png")
