"""
QUESTION  (classic "rubber ball in water" problem, done in class)
------------------------------------------------------------------
A rubber ball of specific gravity 0.6 and radius R = 5.5 cm = 0.055 m is
dropped into a tank of water. The depth x (in metres) to which it sinks
satisfies:

    f(x) = x^3 - 0.165*x^2 + 3.993e-4 = 0

The physically meaningful root must lie between the bottom and top of the
ball, i.e. 0 <= x <= 2R = 0.11.

(1) Plot f(x) for 0 <= x <= 0.11.
(2) Confirm f(0) and f(0.11) have opposite signs (a valid bracket).
(3) Run the bisection method on [0, 0.11] and print an iteration table
    (xl, xu, xm, f(xm), |eps_a|%) just like the one shown in class.
(4) Report the final root and how many significant digits are guaranteed.
(5) Mark the root on the plot and save it (no on-screen popup).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "bisection")
os.makedirs(PLOTS_DIR, exist_ok=True)


def f(x):
    return x ** 3 - 0.165 * x ** 2 + 3.993e-4


# ---------------------------------------------------------------------------
# (2) check bracket validity
# ---------------------------------------------------------------------------
xl, xu = 0.0, 0.11
print(f"f(xl) = f({xl}) = {f(xl):.6e}")
print(f"f(xu) = f({xu}) = {f(xu):.6e}")
assert f(xl) * f(xu) < 0, "No sign change -- not a valid bracket!"

# ---------------------------------------------------------------------------
# (3) bisection with iteration table
# ---------------------------------------------------------------------------
tol = 0.5   # stop once |eps_a| <= 0.5% (about 2 significant digits)
xm_old = None
print(f"\n{'iter':>4} {'xl':>10} {'xu':>10} {'xm':>10} {'f(xm)':>14} {'eps_a %':>10}")
for i in range(1, 21):
    xm = (xl + xu) / 2
    fxm = f(xm)
    eps_a = None if xm_old is None else abs((xm - xm_old) / xm) * 100
    eps_str = "-" if eps_a is None else f"{eps_a:.4f}"
    print(f"{i:>4} {xl:>10.5f} {xu:>10.5f} {xm:>10.5f} {fxm:>14.6e} {eps_str:>10}")

    if f(xl) * fxm < 0:
        xu = xm
    else:
        xl = xm

    if eps_a is not None and eps_a < tol:
        break
    xm_old = xm

root = xm
print(f"\nRoot found: x = {root:.6f} m  (~{root*100:.2f} cm submerged depth)")

# ---------------------------------------------------------------------------
# (5) plot with root marked
# ---------------------------------------------------------------------------
x = np.linspace(0, 0.11, 400)
y = f(x)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, y, color="tab:blue", label="f(x) = x^3 - 0.165x^2 + 3.993e-4")
ax.axhline(0, color="black", linewidth=0.8)
ax.scatter([root], [f(root)], color="red", zorder=5, label=f"root ~ {root:.5f}")
ax.set_xlabel("x (m)")
ax.set_ylabel("f(x)")
ax.set_title("Bisection: rubber ball submerged depth")
ax.grid(True, alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem1_rubber_ball_bisection.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to: {out_path}")
