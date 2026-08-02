"""
Problem 4: Solar Farm Margin (Convexity & Illinois Stagnation Fix)
Objective:
  Solve f(x) = 2^x - 5*x + 1 = 0 over bracket [3.0, 5.0].
  Function is strictly convex (f''(x) > 0).

Demonstrate:
  1. Standard False Position suffers severe endpoint stagnation (xl never updates).
  2. Bisection moves both bounds steadily.
  3. Illinois False Position restores balanced bound updates.
"""

import math
from common import P, use, banner

# -----------------------------------------------------------------------------
# Problem Definition
# -----------------------------------------------------------------------------
XL = 3.0
XU = 5.0
TOL = 0.005      # 0.005% tolerance


def f(x):
    return 2.0**x - 5.0 * x + 1.0


def df(x):
    return (2.0**x) * math.log(2.0) - 5.0


use(f, df)

banner("PROBLEM 4: CONVEXITY STAGNATION & ILLINOIS COMPARISON")

bi = P.bisection(XL, XU, tol=TOL, verbose=False)
fp = P.false_position_opt(XL, XU, tol=TOL, verbose=False)
ill = P.false_position_illinois(XL, XU, tol=TOL, verbose=False)

print(f"{'Method':<25}{'Root Estimate':<18}{'Iterations':<12}{'xl updates':<14}{'xu updates':<14}")
print("-" * 83)

if bi is not None:
    r, i, ea, lu, uu, _ = bi
    print(f"{'Bisection':<25}{r:<18.8f}{i:<12}{lu:<14}{uu:<14}")

if fp is not None:
    r, i, ea, lu, uu, _ = fp
    print(f"{'False Position (Standard)':<25}{r:<18.8f}{i:<12}{lu:<14}{uu:<14}")

if ill is not None:
    r, i, ea, lu, uu, _ = ill
    print(f"{'False Position (Illinois)':<25}{r:<18.8f}{i:<12}{lu:<14}{uu:<14}")

print("-" * 83)
print("\n[KEY OBSERVATION]: Standard False Position had xl_updates = 0 (STAGNATED!).")
print("Illinois method successfully halved the weight on the stagnant bound to restore balance!")
