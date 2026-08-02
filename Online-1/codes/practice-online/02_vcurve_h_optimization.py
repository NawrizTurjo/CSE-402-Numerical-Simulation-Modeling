"""
Problem 2: Thermal Sensor Differentiation (V-Curve h* Optimization)
Objective:
  Numerically differentiate f(x) = cos(x) * e^(-x) at x = 1.5.
  Exact derivative: f'(x) = -sin(x)*e^(-x) - cos(x)*e^(-x).

Demonstrate:
  1. Truncation error dominates for large h (O(h)).
  2. Round-off error dominates for tiny h (subtractive cancellation).
  3. Plot V-curve and find optimal step size h*.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from common import P, use, banner

# -----------------------------------------------------------------------------
# Problem Definition
# -----------------------------------------------------------------------------
X0 = 1.5


def f(x):
    return math.cos(x) * math.exp(-x)


def fprime_exact(x):
    return -math.sin(x) * math.exp(-x) - math.cos(x) * math.exp(-x)


use(f)

banner("PROBLEM 2: THERMAL SENSOR V-CURVE STEP SIZE OPTIMIZATION")

exact_val = fprime_exact(X0)
print(f"  Target x0 = {X0}     Exact f'(1.5) = {exact_val:.10f}\n")

# Logarithmic sweep of h
h_values = np.logspace(0, -16, 80)
fwd_errors = []
cnt_errors = []

for h in h_values:
    fwd = P.forward_diff(f, X0, h)
    cnt = P.central_diff(f, X0, h)
    fwd_errors.append(abs(exact_val - fwd))
    cnt_errors.append(abs(exact_val - cnt))

best_fwd_idx = int(np.argmin(fwd_errors))
best_cnt_idx = int(np.argmin(cnt_errors))

opt_h_fwd = h_values[best_fwd_idx]
opt_err_fwd = fwd_errors[best_fwd_idx]

opt_h_cnt = h_values[best_cnt_idx]
opt_err_cnt = cnt_errors[best_cnt_idx]

print(f"Forward Difference : Optimal h* ~ {opt_h_fwd:.2e}   Min Error = {opt_err_fwd:.4e}")
print(f"Central Difference : Optimal h* ~ {opt_h_cnt:.2e}   Min Error = {opt_err_cnt:.4e}")

# Generate comparison table
P.diff_truncation_table(f, fprime_exact, X0, h_list=[1.0, 0.1, 0.01, 1e-4, 1e-8, 1e-12, 1e-16])

# Plot V-curve Tradeoff
plt.figure(figsize=(8, 5))
plt.loglog(h_values, fwd_errors, 'b-', label='Forward Diff O(h)')
plt.loglog(h_values, cnt_errors, 'g--', label='Central Diff O(h^2)')
plt.axvline(opt_h_fwd, color='blue', linestyle=':', label=f'Fwd h* ~ {opt_h_fwd:.1e}')
plt.axvline(opt_h_cnt, color='green', linestyle=':', label=f'Cnt h* ~ {opt_h_cnt:.1e}')
plt.gca().invert_xaxis()
plt.xlabel('Step Size h (log scale)'); plt.ylabel('Absolute Error (log scale)')
plt.title('Problem 2: Truncation vs Round-off V-Curve Tradeoff')
plt.grid(True, which='both', ls='--', alpha=0.5); plt.legend()
plt.tight_layout()
plt.savefig('fig/02_vcurve_tradeoff.png', dpi=150)
print("\nSaved V-curve plot to fig/02_vcurve_tradeoff.png")
