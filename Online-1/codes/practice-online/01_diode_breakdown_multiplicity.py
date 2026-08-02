"""
Problem 1: Zener Diode Breakdown (Multiplicity m=2 Trap)
Objective:
  Solve f(V) = (V - 0.7)^2 * e^(0.5*V) - 0.05*V + 0.035 = 0 near V0 = 1.0 V.
  V = 0.7 V is a root of multiplicity m=2.

Demonstrate:
  1. Standard Newton-Raphson (m=1) degrades to slow linear convergence.
  2. Modified Newton-Raphson (m=2) restores fast 3-iteration quadratic convergence.
"""

import math
import numpy as np
from common import P, use, banner

# -----------------------------------------------------------------------------
# Problem Definition
# -----------------------------------------------------------------------------
V0 = 1.0         # Initial guess (V)
TOL = 0.005      # 0.005% Scarborough tolerance


def f(V):
    return (V - 0.7)**2 * math.exp(0.5 * V) - 0.05 * V + 0.035


def df(V):
    # Product rule: d/dV[(V-0.7)^2 * e^(0.5V)] - 0.05
    # = 2*(V-0.7)*e^(0.5V) + 0.5*(V-0.7)^2*e^(0.5V) - 0.05
    return (2.0 * (V - 0.7) + 0.5 * (V - 0.7)**2) * math.exp(0.5 * V) - 0.05


use(f, df)

banner("PROBLEM 1: ZENER DIODE MULTIPLICITY TRAP (m=1 VS m=2)")

print("\n--- Standard Newton-Raphson (m = 1) ---")
r1, h1 = P._newton_raphson_core(V0, m=1, tol=TOL, max_iter=30, verbose=False)
print(f"{'Iter':<6}{'Estimate (V_i)':<18}{'Error ea (%)':<18}")
print("-" * 45)
for h in h1:
    ea_str = "Initial" if h['iter'] == 1 else f"{h['ea']:.4f}%"
    print(f"{h['iter']:<6}{h['xi1']:<18.6f}{ea_str:<18}")
print(f"Total iterations (m=1): {len(h1)}")

print("\n--- Modified Newton-Raphson (m = 2) ---")
r2, h2 = P._newton_raphson_core(V0, m=2, tol=TOL, max_iter=30, verbose=False)
print(f"{'Iter':<6}{'Estimate (V_i)':<18}{'Error ea (%)':<18}")
print("-" * 45)
for h in h2:
    ea_str = "Initial" if h['iter'] == 1 else f"{h['ea']:.4f}%"
    print(f"{h['iter']:<6}{h['xi1']:<18.6f}{ea_str:<18}")
print(f"Total iterations (m=2): {len(h2)}")

print(f"\n[CONCLUSION]: m=1 took {len(h1)} iterations vs m=2 took {len(h2)} iterations!")
print("Residual check (m=2):", P.classify_and_verify(r2, 'newton_raphson', residual_tol=1e-4))

f_np = np.vectorize(f)
P.plot_with_root(f_np, 0.4, 1.2, root=r2, title="Problem 1: Multiplicity m=2 Double Root at V=0.7V", fname="01_diode_multiplicity_plot.png")
