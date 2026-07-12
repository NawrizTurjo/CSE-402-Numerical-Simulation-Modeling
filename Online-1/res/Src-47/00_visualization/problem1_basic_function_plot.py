"""
QUESTION
--------
A function is given:

    f(x) = e^(-0.5x) * sin(3x) - 0.2x + 1

(a) Plot f(x) for 0 <= x <= 8 with a proper title, axis labels and grid.
(b) On the same axes, draw the line y = 0 so sign changes are easy to see
    just by looking at the plot.
(c) Save the plot as "problem1_basic_function_plot.png" inside the plots
    folder. The plot must NOT pop up on screen when the script is run.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "visualization")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# The given function
# ---------------------------------------------------------------------------
def f(x):
    return np.exp(-0.5 * x) * np.sin(3 * x) - 0.2 * x + 1


# ---------------------------------------------------------------------------
# (a) + (b) + (c)
# ---------------------------------------------------------------------------
x = np.linspace(0, 8, 1000)
y = f(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color="tab:blue", linewidth=1.8, label="f(x)")
ax.axhline(0, color="black", linewidth=0.9)   # the y = 0 reference line
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("f(x) = e^(-0.5x) sin(3x) - 0.2x + 1   for 0 <= x <= 8")
ax.grid(True, alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem1_basic_function_plot.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Plot saved to: {out_path}")

# Just by eyeballing the plot you can already count how many times the
# curve crosses y = 0 in [0, 8] -- that is exactly what root-finding
# methods later in this course will pin down numerically.
