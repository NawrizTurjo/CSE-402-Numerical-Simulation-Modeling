"""
Visual diagnostics for a sequence of values expected to be i.i.d.
Uniform(0,1) -- e.g. output from rng/lcg.py or rng/middle_square.py.
These are exactly the checks Res/Src-14/random_number/visualize_random_numbers.py
did by hand with three separate plt.figure() calls; here they're one
reusable function producing one figure with three panels.

Run standalone:
    python -m viz.rng_plots
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import matplotlib.pyplot as plt


def plot_rng_diagnostics(values, title="Random Numbers", bins=10, show=True, save_path=None):
    """
    One figure, three panels, each catching a different kind of defect:

      1. Sequence plot (value vs. generation order) -- spot drift, or an
         obvious short cycle repeating visually.
      2. Histogram (bins over [0,1)) -- spot non-uniformity; this is the
         same binning chi_square_test.py's test statistic is built from.
      3. Scatter of R_i vs R_{i+1} -- spot lattice/pattern structure
         between CONSECUTIVE values. This is the 2-D version of the
         RANDU hyperplane check in rng/lcg.py's docstring: a generator
         can look perfectly uniform in panel 2 while every point in
         panel 3 visibly falls on a small number of diagonal stripes.

    Parameters
    ----------
    values : sequence of floats in [0, 1).
    show   : call plt.show() (interactive). Set False for a non-blocking
             run (e.g. scripted/headless) and pass save_path instead.
    save_path : optional file path to also save the figure to.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    x_axis = list(range(1, len(values) + 1))
    axes[0].plot(x_axis, values, marker="o", markersize=2, linewidth=0.8)
    axes[0].set_title(f"{title}\nSequence")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Value")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True)

    axes[1].hist(values, bins=bins, range=(0, 1), edgecolor="black")
    axes[1].set_title(f"{title}\nHistogram ({bins} bins)")
    axes[1].set_xlabel("Interval")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(axis="y")

    axes[2].scatter(values[:-1], values[1:], s=10)
    axes[2].set_title(f"{title}\nR_i vs R_(i+1)")
    axes[2].set_xlabel("R_i")
    axes[2].set_ylabel("R_(i+1)")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].grid(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    from rng.lcg import lcg_uniforms
    from rng.middle_square import middle_square_uniforms

    plot_rng_diagnostics(
        lcg_uniforms(seed=1, a=1103515245, c=12345, m=2**31, n=1000),
        title="ANSI-C LCG",
        show=False, save_path="rng_plot_demo_lcg.png",
    )
    plot_rng_diagnostics(
        middle_square_uniforms(seed=5731, n=1000),
        title="Middle-Square (seed 5731)",
        show=False, save_path="rng_plot_demo_middle_square.png",
    )
    print("Saved rng_plot_demo_lcg.png and rng_plot_demo_middle_square.png")
    print("(pass show=True, save_path=None for the normal interactive exam usage)")
