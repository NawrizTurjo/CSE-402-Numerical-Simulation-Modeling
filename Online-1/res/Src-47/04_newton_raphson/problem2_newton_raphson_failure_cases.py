"""
QUESTION  (three classic "explain why Newton-Raphson fails here" cases,
shown in class as drawbacks of the method)
------------------------------------------------------------------------
Case A - Root jumping (multiple roots):
    f(x) = sin(x),   x0 = 2.4*pi  (trying to reach the root at 2*pi)
    Show that Newton-Raphson can leap across several roots and converge
    to a completely different one (here it lands on x = 0).

Case B - No real root at all (oscillation forever):
    f(x) = x^2 + 2      (never crosses zero for real x)
    x0 = -1.0
    Show that the iterates bounce around near the minimum and never
    settle down.

Case C - Near-zero derivative (divergence):
    f(x)  = x^3 - 0.03*x^2 + 2.4e-6
    f'(x) = 3*x^2 - 0.06*x     (vanishes at x = 0 and x = 0.02)
    x0 = 0.01999   (deliberately close to the derivative's zero at 0.02)
    Show that the method launches far away instead of converging.

For each case: print the iteration table (10 iterations) and a short
printed explanation of WHY it fails. Also save one plot per case showing
the function and, where relevant, why the tangent line misbehaves.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "newton_raphson")
os.makedirs(PLOTS_DIR, exist_ok=True)


def run_newton(f, fprime, x0, n_iter=10):
    x = x0
    table = []
    for i in range(n_iter):
        fx = f(x)
        dfx = fprime(x)
        eps_a = None
        if dfx == 0:
            print(f"  Stopped at i={i}: f'(x) = 0 exactly, cannot continue.")
            break
        x_new = x - fx / dfx
        if x_new != 0:
            eps_a = abs((x_new - x) / x_new) * 100
        table.append({"i": i, "x_i": x, "f(x_i)": fx, "eps_a_%": eps_a})
        x = x_new
    return x, table


def print_table(table):
    print(f"{'i':>3} {'x_i':>16} {'f(x_i)':>16} {'eps_a %':>14}")
    for row in table:
        eps_str = "-" if row["eps_a_%"] is None else f"{row['eps_a_%']:.4f}"
        print(f"{row['i']:>3} {row['x_i']:>16.6f} {row['f(x_i)']:>16.6f} {eps_str:>14}")


# ===========================================================================
# CASE A: root jumping with sin(x)
# ===========================================================================
print("=" * 70)
print("CASE A: f(x) = sin(x)  --  root jumping")
print("=" * 70)
fA = math.sin
fA_prime = math.cos
x0_A = 2.4 * math.pi
final_A, table_A = run_newton(fA, fA_prime, x0_A)
print_table(table_A)
print(f"\nStarted near x0 = 2.4*pi = {x0_A:.4f} hoping to reach 2*pi = {2*math.pi:.4f}")
print(f"Instead converged to x = {final_A:.6f} (a different root of sin(x)).")
print("Why: the tangent line at a poorly-chosen point can overshoot across "
      "several neighbouring roots of a periodic function like sin(x).\n")

xA = np.linspace(-1, 9, 500)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(xA, np.sin(xA), color="tab:blue", label="f(x) = sin(x)")
ax.axhline(0, color="black", linewidth=0.8)
ax.scatter([x0_A], [fA(x0_A)], color="orange", zorder=5, label="starting guess x0")
ax.scatter([final_A], [fA(final_A)], color="red", zorder=5, label="converged root")
ax.set_title("Case A: Newton-Raphson root jumping on sin(x)")
ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.grid(True, alpha=0.3); ax.legend()
fig.savefig(os.path.join(PLOTS_DIR, "problem2_caseA_root_jumping.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# CASE B: no real root, x^2 + 2
# ===========================================================================
print("=" * 70)
print("CASE B: f(x) = x^2 + 2  --  no real root exists")
print("=" * 70)
fB = lambda x: x ** 2 + 2
fB_prime = lambda x: 2 * x
x0_B = -1.0
final_B, table_B = run_newton(fB, fB_prime, x0_B)
print_table(table_B)
print("\nThe iterates never settle down -- they keep bouncing because there "
      "is no real x with x^2 + 2 = 0. Newton-Raphson has no way to detect "
      "this in advance; it simply keeps chasing a root that doesn't exist.\n")

xB = np.linspace(-6, 6, 500)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(xB, fB(xB), color="tab:blue", label="f(x) = x^2 + 2")
ax.axhline(0, color="black", linewidth=0.8)
ax.scatter([x0_B], [fB(x0_B)], color="orange", zorder=5, label="starting guess x0")
ax.set_title("Case B: no real root -- curve never touches y=0")
ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.grid(True, alpha=0.3); ax.legend()
fig.savefig(os.path.join(PLOTS_DIR, "problem2_caseB_no_real_root.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# CASE C: near-zero derivative, x^3 - 0.03x^2 + 2.4e-6
# ===========================================================================
print("=" * 70)
print("CASE C: f(x) = x^3 - 0.03x^2 + 2.4e-6  --  derivative near zero")
print("=" * 70)
fC = lambda x: x ** 3 - 0.03 * x ** 2 + 2.4e-6
fC_prime = lambda x: 3 * x ** 2 - 0.06 * x
x0_C = 0.01999
final_C, table_C = run_newton(fC, fC_prime, x0_C)
print_table(table_C)
print(f"\nf'(x) = 3x^2 - 0.06x is exactly 0 at x = 0 and x = 0.02.")
print(f"Starting at x0 = {x0_C} (very close to 0.02) makes the tangent "
      "line almost flat, so dividing by the tiny derivative sends the "
      "next guess flying far away instead of converging.\n")

xC = np.linspace(-0.05, 0.05, 500)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(xC, fC(xC), color="tab:blue", label="f(x) = x^3 - 0.03x^2 + 2.4e-6")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0.02, color="gray", linestyle="--", linewidth=1, label="f'(x)=0 at x=0.02")
ax.scatter([x0_C], [fC(x0_C)], color="orange", zorder=5, label="starting guess x0")
ax.set_title("Case C: starting guess near a zero of f'(x)")
ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.grid(True, alpha=0.3); ax.legend()
fig.savefig(os.path.join(PLOTS_DIR, "problem2_caseC_near_zero_derivative.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nAll three plots saved inside: {PLOTS_DIR}")
