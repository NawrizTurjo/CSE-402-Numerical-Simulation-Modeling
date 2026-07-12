"""
QUESTION
--------
Two functions are given:

    g(x) = e^(-x)
    h(x) = sin(x)

(a) Plot g(x) and h(x) together on the same axes for 0 <= x <= 6 (use a
    legend to tell them apart).
(b) The two curves intersect wherever g(x) = h(x), i.e. wherever
    f(x) = g(x) - h(x) = 0. Using a simple bracket scan with step 0.1,
    find every interval where f(x) changes sign, then use bisection on
    each interval to find the intersection point(s).
(c) Mark all intersection points on the g/h plot with red dots.
(d) Save the plot; it must not pop up on screen.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "visualization")
os.makedirs(PLOTS_DIR, exist_ok=True)


def g(x):
    return np.exp(-x)


def h(x):
    return np.sin(x)


def f(x):          # difference function whose roots are the intersections
    return g(x) - h(x)


# ---------------------------------------------------------------------------
# (b) scan for sign changes, then bisect each bracket
# ---------------------------------------------------------------------------
def find_brackets(func, a, b, step):
    brackets = []
    x_left = a
    while x_left < b:
        x_right = min(x_left + step, b)
        if func(x_left) * func(x_right) < 0:
            brackets.append((x_left, x_right))
        x_left = x_right
    return brackets


def bisection(func, xl, xu, tol=1e-8, max_iter=100):
    xm_old = xl
    for _ in range(max_iter):
        xm = (xl + xu) / 2
        if func(xl) * func(xm) < 0:
            xu = xm
        else:
            xl = xm
        if abs((xm - xm_old) / xm) * 100 < tol:
            break
        xm_old = xm
    return xm


brackets = find_brackets(f, 0, 6, 0.1)
roots = [bisection(f, xl, xu) for xl, xu in brackets]

print("Sign-change brackets found:", brackets)
print("Intersection points (x where g(x) = h(x)):")
for r in roots:
    print(f"  x = {r:.6f}   g(x) = {g(r):.6f}   h(x) = {h(r):.6f}")

# ---------------------------------------------------------------------------
# (a) + (c) + (d)
# ---------------------------------------------------------------------------
x = np.linspace(0, 6, 500)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, g(x), color="tab:blue", label="g(x) = e^(-x)")
ax.plot(x, h(x), color="tab:orange", label="h(x) = sin(x)")
ax.scatter(roots, [g(r) for r in roots], color="red", zorder=5,
           label="intersections")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("g(x) and h(x) with intersection points marked")
ax.grid(True, alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem2_multi_function_and_roots_plot.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Plot saved to: {out_path}")
