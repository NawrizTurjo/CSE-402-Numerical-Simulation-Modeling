"""
TEMPLATE: Visualization and Plotting
------------------------------------
General-purpose template for plotting a function (or several) during a lab
test. Copy this file, change FUNCTION(s) and the range, run it.

Rules followed here (matches lab-test expectations):
  - No plt.show() anywhere -> the plot window never pops up.
  - Every figure is saved as a .png file inside the "plots/<topic>" folder.
  - Paths are built from this script's own location, so it works no matter
    which folder you run "python xxx.py" from.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend -> no popup window
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Folder where plots for this topic get saved
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "visualization")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_fig(fig, filename):
    """Save a figure into the plots/visualization folder and close it."""
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {path}")


# ---------------------------------------------------------------------------
# Example 1: plot a single function over a range
# ---------------------------------------------------------------------------
def f(x):
    return x**3 - 6 * x**2 + 11 * x - 6


def plot_single_function():
    x = np.linspace(-1, 5, 500)
    y = f(x)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, color="tab:blue", label="f(x)")
    ax.axhline(0, color="black", linewidth=0.8)   # x-axis reference line
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("f(x) = x^3 - 6x^2 + 11x - 6")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, "template_single_function.png")


# ---------------------------------------------------------------------------
# Example 2: mark specific points (e.g. roots) on the curve
# ---------------------------------------------------------------------------
def plot_with_marked_roots(roots):
    x = np.linspace(-1, 5, 500)
    y = f(x)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, color="tab:blue", label="f(x)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.scatter(roots, [0] * len(roots), color="red", zorder=5, label="roots")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("f(x) with roots marked")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, "template_marked_roots.png")


if __name__ == "__main__":
    plot_single_function()
    plot_with_marked_roots(roots=[1, 2, 3])
