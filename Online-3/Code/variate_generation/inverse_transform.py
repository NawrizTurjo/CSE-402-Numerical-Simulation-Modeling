"""
Inverse-Transform Method: turn a Uniform(0,1) draw R into a draw from any
distribution whose CDF F you can invert by hand.

General recipe: if U ~ Uniform(0,1) and X = F^{-1}(U), then X has CDF F.
    F(x) = u  =>  solve for x  =>  x = F^{-1}(u)

Each function below takes `r` (a value already drawn from Uniform(0,1) --
e.g. from rng/lcg.py or rng/middle_square.py) and returns one variate.
IMPORTANT: inverse-transform is a deterministic RESHAPING of randomness,
not a source of it -- if the underlying uniform stream is short-period or
correlated (see rng/lcg.py's RANDU note), every distribution built from
it inherits the exact same flaw.

Run standalone:
    python -m variate_generation.inverse_transform
"""

import math
import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


def uniform(r, a, b):
    """F(x) = (x-a)/(b-a)  =>  x = a + (b-a)*r"""
    return a + (b - a) * r


def exponential(r, mean):
    """F(x) = 1 - e^(-x/mean)  =>  x = -mean * ln(1 - r)"""
    return -mean * math.log(1 - r)


def weibull(r, shape_k, scale_c):
    """F(x) = 1 - e^(-(x/c)^k)  =>  x = c * (-ln(1 - r))^(1/k)"""
    return scale_c * (-math.log(1 - r)) ** (1 / shape_k)


def discrete(r, values, probs):
    """Staircase CDF: walk the cumulative probability until r fits under it."""
    cumulative = 0.0
    for v, p in zip(values, probs):
        cumulative += p
        if r <= cumulative:
            return v
    return values[-1]  # floating-point safety net


def triangular(r, low, high, mode):
    """
    Triangular(low, high, mode). CDF has two pieces (split at the mode),
    so solve F(x)=r separately on each side.
    """
    split = (mode - low) / (high - low)
    if r <= split: ## Left Side
        return low + math.sqrt(r * (high - low) * (mode - low))
    # else Right Side
    return high - math.sqrt((1 - r) * (high - low) * (high - mode))


def plot_inverse_transform(samples, dist_name="Exponential", theoretical_pdf=None, bins=30,
                           title=None, show=True, save_path=None):
    """
    Plot histogram of generated variates alongside theoretical continuous density curve.

    Parameters
    ----------
    samples : list of float
        Generated variates.
    dist_name : str, default='Exponential'
        Distribution name.
    theoretical_pdf : callable, optional
        Theoretical probability density function.
    bins : int, default=30
        Number of histogram bins.
    title : str, optional
        Figure title.
    show : bool, default=True
        Whether to call plt.show().
    save_path : str, optional
        If provided, save the figure to this file path.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(samples, bins=bins, density=True, color="skyblue", edgecolor="black", alpha=0.7, label=f"Generated (N={len(samples)})")

    if theoretical_pdf is not None:
        min_x, max_x = min(samples), max(samples)
        grid = [min_x + (max_x - min_x) * i / 300.0 for i in range(301)]
        pdf_vals = [theoretical_pdf(x) for x in grid]
        ax.plot(grid, pdf_vals, "r-", linewidth=2, label="Theoretical PDF")

    plot_title = title or f"Inverse Transform Generation: {dist_name} Distribution"
    ax.set_title(plot_title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    from rng.lcg import lcg_uniforms

    uniforms = lcg_uniforms(seed=27, a=17, c=43, m=100, n=5)
    print("Uniforms:", uniforms)
    print("-> Uniform[5,10]      :", [round(uniform(r, 5, 10), 4) for r in uniforms])
    print("-> Exponential(mean=2):", [round(exponential(r, 2.0), 4) for r in uniforms])
