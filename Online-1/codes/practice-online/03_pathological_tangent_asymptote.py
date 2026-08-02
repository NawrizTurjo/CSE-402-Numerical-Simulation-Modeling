"""
Problem 3: Chemical Reactor Singularity Trap (Asymptote Filtering)
Objective:
  Find real roots of f(x) = (tan(x) - x) / (x - 1.5708) over [0, 3.0].

Demonstrate:
  1. Vertical pole / asymptote at x = 1.5708 causes a fake sign-change bracket.
  2. Bisection alone converges onto the asymptote (garbage root).
  3. Using P.classify_and_verify with residual tolerance rejects the fake pole.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from common import P, use, banner

# -----------------------------------------------------------------------------
# Problem Definition
# -----------------------------------------------------------------------------
POLE = 1.5707963267948966  # pi / 2


def f(x):
    # Avoid exact division by zero at x = pi/2
    denom = x - POLE
    if abs(denom) < 1e-12:
        return 1e12
    return (math.tan(x) - x) / denom


use(f)

banner("PROBLEM 3: CHEMICAL REACTOR SINGULARITY TRAP (ASYMPTOTE FILTERING)")

# Scan domain [0, 3.0]
direct, intervals = P.coarse_scan(0.0, 3.0, step=0.1)

print(f"Direct root candidates: {direct}")
print(f"Sign-change intervals found: {intervals}")

print("\n--- Solving all brackets without residual filter ---")
raw_roots = []
for xl, xu in intervals:
    res = P.bisection(xl, xu, tol=0.0001, verbose=False)
    if res is not None:
        raw_roots.append(res[0])

print(f"Raw bisection output roots: {[round(r, 6) for r in raw_roots]}")

print("\n--- Verifying roots with P.classify_and_verify ---")
verified_roots = []
for r in raw_roots:
    verdict = P.classify_and_verify(r, 'bisection', residual_tol=1e-3)
    print(f"Candidate r = {r:.6f} -> Verdict: {'ACCEPTED' if verdict['accepted'] else 'REJECTED (Pole Singularity!)'} (Residual f(r) = {verdict['f_x']})")
    if verdict['accepted']:
        verified_roots.append(r)

print(f"\nFinal Validated Physical Roots: {verified_roots}")

# Plot function with asymptote and valid roots
xs = np.linspace(0, 3.0, 1000)
ys = []
for x in xs:
    if abs(x - POLE) < 0.05:
        ys.append(np.nan)
    else:
        ys.append(f(x))

plt.figure(figsize=(8, 5))
plt.plot(xs, ys, color='steelblue', linewidth=2, label='f(x)')
plt.axvline(POLE, color='red', linestyle='--', label='Asymptote x = pi/2')
plt.axhline(0, color='black', linewidth=1)
if verified_roots:
    plt.scatter(verified_roots, [0]*len(verified_roots), color='green', marker='x', s=120, zorder=5, label='Valid Root')
plt.ylim(-20, 20)
plt.xlabel('x'); plt.ylabel('f(x)'); plt.title('Problem 3: Singularity Trap vs True Root')
plt.grid(alpha=0.4); plt.legend()
plt.tight_layout()
plt.savefig('fig/03_asymptote_filter_plot.png', dpi=150)
print("\nSaved plot to fig/03_asymptote_filter_plot.png")
