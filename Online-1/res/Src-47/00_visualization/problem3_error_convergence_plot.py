"""
QUESTION
--------
sqrt(2) can be approximated with the iterative (Newton's method) formula

    x_(i+1) = 0.5 * ( x_i + 2 / x_i )

starting from x_0 = 1.

(a) Run the iteration for 8 steps.
(b) At every step compute the absolute relative approximate error
        |eps_a| = | (x_new - x_old) / x_new | * 100 %
(c) Print a table of iteration number, x_i, and |eps_a|.
(d) Plot |eps_a| against iteration number using a log scale on the y-axis
    (semilogy) so the very fast convergence is easy to see.
(e) Save the plot; do not let it pop up on screen.
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
# (a) + (b) + (c)
# ---------------------------------------------------------------------------
x = 1.0
iterations = []
values = []
errors = []

print(f"{'i':>3} {'x_i':>14} {'|eps_a| %':>16}")
for i in range(1, 9):
    x_new = 0.5 * (x + 2 / x)
    eps_a = abs((x_new - x) / x_new) * 100
    iterations.append(i)
    values.append(x_new)
    errors.append(eps_a)
    print(f"{i:>3} {x_new:>14.10f} {eps_a:>16.10f}")
    x = x_new

print(f"\nTrue value sqrt(2) = {np.sqrt(2):.10f}")
print(f"Final estimate      = {x:.10f}")

# ---------------------------------------------------------------------------
# (d) + (e)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
# errors[0] is undefined in a strict sense (no "old" x before x0), so we
# only plot from iteration 2 onward where the drop is visible on log scale.
ax.semilogy(iterations[1:], errors[1:], "o-", color="tab:red")
ax.set_xlabel("iteration, i")
ax.set_ylabel("|eps_a|  (%)   [log scale]")
ax.set_title("Convergence of Newton's iteration for sqrt(2)")
ax.grid(True, which="both", alpha=0.3)

out_path = os.path.join(PLOTS_DIR, "problem3_error_convergence_plot.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Plot saved to: {out_path}")
