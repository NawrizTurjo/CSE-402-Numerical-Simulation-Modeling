"""
QUESTION  (classic "one endpoint freezes" demonstration, done in class)
------------------------------------------------------------------------
    f(x) = (x - 4)^2 * (x + 2)

Note x = 4 is a DOUBLE root (the curve touches zero but does not cross --
no sign change there) while x = -2 is a simple root (the curve does cross
zero there).

(1) Use bracket xl = -2.5, xu = -1.0 (which brackets only the simple root
    at x = -2; the sign change theorem applies here).
(2) Run False Position with stopping tolerance |eps_a| <= 0.1%.
(3) Print the iteration table and note that xl barely moves while xu does
    almost all the work -- because the function is far more curved near
    x = 4 (even though that root isn't in this bracket, the *cubic shape*
    of the whole function still biases the secant line).
(4) Plot f(x) over a wide enough range to show both the double root
    (touching, not crossing, at x=4) and the simple root (crossing at
    x=-2), with the converged root marked.
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
    return (x - 4) ** 2 * (x + 2)


# ---------------------------------------------------------------------------
# (1) + (2) + (3)
# ---------------------------------------------------------------------------
xl, xu = -2.5, -1.0
assert f(xl) * f(xu) < 0

tol = 0.1
xr_old = None
xl_seen, xu_seen = set(), set()
print(f"{'iter':>4} {'xl':>10} {'xu':>10} {'xr':>12} {'f(xr)':>14} {'eps_a %':>10}")
for i in range(1, 15):
    xr = xu - f(xu) * (xl - xu) / (f(xl) - f(xu))
    fxr = f(xr)
    eps_a = None if xr_old is None else abs((xr - xr_old) / xr) * 100
    eps_str = "-" if eps_a is None else f"{eps_a:.4f}"
    print(f"{i:>4} {xl:>10.5f} {xu:>10.5f} {xr:>12.7f} {fxr:>14.6e} {eps_str:>10}")

    xl_seen.add(round(xl, 8))
    xu_seen.add(round(xu, 8))

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
print(f"\nRoot found: x = {root:.6f}   (true root is exactly -2)")
print(f"Distinct xl values seen: {len(xl_seen)}   Distinct xu values seen: {len(xu_seen)}")
if len(xl_seen) < len(xu_seen):
    print("-> xl stayed essentially frozen: classic false-position stagnation.")

# ---------------------------------------------------------------------------
# (4) plot showing both roots
# ---------------------------------------------------------------------------
x = np.linspace(-4, 6, 500)
y = f(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color="tab:blue", label="f(x) = (x-4)^2 (x+2)")
ax.axhline(0, color="black", linewidth=0.8)
ax.scatter([root], [f(root)], color="red", zorder=5, label=f"root found ~ {root:.4f}")
ax.scatter([4], [0], color="green", marker="x", s=80, zorder=5,
           label="double root at x=4 (touches, no sign change)")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("False Position with a nearby double root")
ax.grid(True, alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem2_double_root_stagnation.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to: {out_path}")
