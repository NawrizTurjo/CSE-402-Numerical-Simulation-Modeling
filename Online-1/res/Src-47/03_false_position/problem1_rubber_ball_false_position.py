"""
QUESTION  (same rubber-ball model as the bisection topic, solved by a
different method so results can be compared)
------------------------------------------------------------------------
    f(x) = x^3 - 0.165*x^2 + 3.993e-4 = 0,   bracket [0, 0.11]

(1) Confirm the bracket [0, 0.11] is valid (sign change).
(2) Run the False Position (Regula Falsi) method and print an iteration
    table: xl, xu, xr, f(xr), |eps_a|%.
(3) Report the converged root.
(4) Notice (and print a note about) whether one endpoint of the bracket
    stays frozen across iterations -- a known quirk of this method.
(5) Plot f(x) with the root marked; save without showing on screen.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "false_position")
os.makedirs(PLOTS_DIR, exist_ok=True)


def f(x):
    return x ** 3 - 0.165 * x ** 2 + 3.993e-4


# ---------------------------------------------------------------------------
# (1) check bracket
# ---------------------------------------------------------------------------
xl, xu = 0.0, 0.11
assert f(xl) * f(xu) < 0, "No sign change -- not a valid bracket!"

# ---------------------------------------------------------------------------
# (2) false position iteration table
# ---------------------------------------------------------------------------
tol = 0.01   # stop once |eps_a| <= 0.01%
xr_old = None
xl_history, xu_history = [], []
print(f"{'iter':>4} {'xl':>10} {'xu':>10} {'xr':>12} {'f(xr)':>14} {'eps_a %':>10}")
for i in range(1, 20):
    xr = xu - f(xu) * (xl - xu) / (f(xl) - f(xu))
    fxr = f(xr)
    eps_a = None if xr_old is None else abs((xr - xr_old) / xr) * 100
    eps_str = "-" if eps_a is None else f"{eps_a:.4f}"
    print(f"{i:>4} {xl:>10.5f} {xu:>10.5f} {xr:>12.7f} {fxr:>14.6e} {eps_str:>10}")

    xl_history.append(xl)
    xu_history.append(xu)

    if fxr == 0:
        break
    if f(xl) * fxr < 0:
        xu = xr
    else:
        xl = xr

    if eps_a is not None and eps_a < tol:
        break
    xr_old = xr

root = xr
print(f"\nRoot found: x = {root:.6f} m")

# ---------------------------------------------------------------------------
# (4) check which endpoint stayed frozen
# ---------------------------------------------------------------------------
if len(set(xl_history)) < len(set(xu_history)):
    print("Note: xl stayed frozen for most iterations -- classic false-position "
          "behaviour when the curve is much steeper on the xu side.")
elif len(set(xu_history)) < len(set(xl_history)):
    print("Note: xu stayed frozen for most iterations -- classic false-position "
          "behaviour when the curve is much steeper on the xl side.")

# ---------------------------------------------------------------------------
# (5) plot
# ---------------------------------------------------------------------------
x = np.linspace(0, 0.11, 400)
y = f(x)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, y, color="tab:blue", label="f(x) = x^3 - 0.165x^2 + 3.993e-4")
ax.axhline(0, color="black", linewidth=0.8)
ax.scatter([root], [f(root)], color="red", zorder=5, label=f"root ~ {root:.5f}")
ax.set_xlabel("x (m)")
ax.set_ylabel("f(x)")
ax.set_title("False Position: rubber ball submerged depth")
ax.grid(True, alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem1_rubber_ball_false_position.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to: {out_path}")
