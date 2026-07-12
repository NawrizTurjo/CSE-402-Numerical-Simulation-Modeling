"""
QUESTION  (same rubber-ball model, solved a third way for comparison)
------------------------------------------------------------------------
    f(x)  = x^3 - 0.165*x^2 + 3.993e-4
    f'(x) = 3*x^2 - 0.33*x

(1) Explain (in a comment/printed note) why x0 = 0 or x0 = 0.11 would be
    bad initial guesses here (hint: check f'(x) at those points).
(2) Starting from a sensible guess x0 = 0.05, run Newton-Raphson and
    print the iteration table: i, x_i, f(x_i), x_(i+1), |eps_a|%.
(3) Report the converged root and compare with the bisection/false
    position answers from the other topic folders (should all agree).
(4) Plot f(x) with the root marked; save without showing on screen.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "newton_raphson")
os.makedirs(PLOTS_DIR, exist_ok=True)


def f(x):
    return x ** 3 - 0.165 * x ** 2 + 3.993e-4


def fprime(x):
    return 3 * x ** 2 - 0.33 * x


# ---------------------------------------------------------------------------
# (1) why the interval endpoints are bad initial guesses
# ---------------------------------------------------------------------------
print(f"f'(0)    = {fprime(0)}")
print(f"f'(0.11) = {fprime(0.11)}")
print("Both are 0 -> the tangent line would be horizontal there and never "
      "meet the x-axis (division by zero). We must start somewhere in "
      "between, e.g. x0 = 0.05.\n")

# ---------------------------------------------------------------------------
# (2) Newton-Raphson iteration table
# ---------------------------------------------------------------------------
x = 0.05
tol = 1e-6
print(f"{'i':>3} {'x_i':>12} {'f(x_i)':>14} {'x_(i+1)':>12} {'eps_a %':>12}")
for i in range(15):
    fx = f(x)
    dfx = fprime(x)
    x_new = x - fx / dfx
    eps_a = abs((x_new - x) / x_new) * 100
    print(f"{i:>3} {x:>12.6f} {fx:>14.6e} {x_new:>12.7f} {eps_a:>12.6f}")
    if eps_a < tol:
        x = x_new
        break
    x = x_new

root = x
print(f"\nRoot found: x = {root:.6f} m")

# ---------------------------------------------------------------------------
# (4) plot
# ---------------------------------------------------------------------------
xs = np.linspace(0, 0.11, 400)
ys = f(xs)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(xs, ys, color="tab:blue", label="f(x) = x^3 - 0.165x^2 + 3.993e-4")
ax.axhline(0, color="black", linewidth=0.8)
ax.scatter([root], [f(root)], color="red", zorder=5, label=f"root ~ {root:.5f}")
ax.set_xlabel("x (m)")
ax.set_ylabel("f(x)")
ax.set_title("Newton-Raphson: rubber ball submerged depth")
ax.grid(True, alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem1_rubber_ball_newton_raphson.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to: {out_path}")
