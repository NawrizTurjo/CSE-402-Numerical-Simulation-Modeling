"""
Thin, named wrappers around monte_carlo.core.monte_carlo_estimate for the
two most common exam shapes: integrating a 1-D function, and hit-or-miss
estimation of pi. Read monte_carlo/core.py first -- everything here is
just "define trial_fn, call the core estimator".

Run standalone:
    python -m monte_carlo.integration
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import math
import random

from monte_carlo.core import monte_carlo_estimate


def integrate_1d(f, a, b, n, seed=None):
    """
    Estimate integral_a^b f(x) dx via the sample-mean method:
        X ~ Uniform(a, b),  Y = (b - a) * f(X),  estimate = mean(Y).
    Works for ANY f, even ones with no closed-form antiderivative
    (e.g. sin(x)*cos(x^2)) -- that's the whole point of Monte Carlo
    integration.
    """
    width = b - a

    def trial():
        x = random.uniform(a, b)
        return width * f(x)

    return monte_carlo_estimate(trial, n, seed=seed)


def estimate_pi_hit_or_miss(n, seed=None):
    """
    Classic pi estimation: throw darts at the unit square [0,1]x[0,1],
    check whether they land inside the quarter unit circle (x^2+y^2<=1).
    P(inside) = (pi/4 * 1^2) / (1*1) = pi/4   =>   pi = 4 * P(inside).
    This is a hit-or-miss volume estimate in disguise: box area = 1,
    trial returns 4 * indicator (the "4" folds the box-area scaling in).
    """
    def trial():
        x, y = random.random(), random.random()
        return 4.0 if x * x + y * y <= 1.0 else 0.0

    return monte_carlo_estimate(trial, n, seed=seed)


def plot_integration_1d(f, a, b, n_samples=500, title="1-D Monte Carlo Integration", show=True, save_path=None):
    """
    Plot integrand f(x) over [a, b] with sampled random evaluation points.

    Parameters
    ----------
    f : callable
        Function to integrate.
    a : float
        Lower bound.
    b : float
        Upper bound.
    n_samples : int, default=500
        Number of scatter points.
    title : str, default='1-D Monte Carlo Integration'
        Figure title.
    show : bool, default=True
        Whether to call plt.show().
    save_path : str, optional
        If provided, save the figure to this file path.
    """
    import matplotlib.pyplot as plt

    xs = [a + (b - a) * random.random() for _ in range(n_samples)]
    ys = [f(x) for x in xs]

    # Grid for smooth curve
    x_grid = [a + (b - a) * i / 400.0 for i in range(401)]
    y_grid = [f(x) for x in x_grid]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_grid, y_grid, "r-", linewidth=2, label="Integrand $f(x)$")
    ax.scatter(xs, ys, color="royalblue", alpha=0.5, s=15, label=f"Samples (N={n_samples})")
    ax.fill_between(x_grid, y_grid, alpha=0.15, color="red")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("f(x)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_hit_or_miss_pi(n_points=2000, seed=None, title="Monte Carlo Pi: Quarter-Circle Hit-or-Miss", show=True, save_path=None):
    """
    Plot 2-D unit square with quarter-circle boundary and colored hit/miss points.

    Parameters
    ----------
    n_points : int, default=2000
        Number of points.
    seed : int, optional
        Random seed.
    title : str, default='Monte Carlo Pi: Quarter-Circle Hit-or-Miss'
        Figure title.
    show : bool, default=True
        Whether to call plt.show().
    save_path : str, optional
        If provided, save the figure to this file path.
    """
    import matplotlib.pyplot as plt

    if seed is not None:
        random.seed(seed)

    inside_x, inside_y = [], []
    outside_x, outside_y = [], []

    for _ in range(n_points):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1.0:
            inside_x.append(x)
            inside_y.append(y)
        else:
            outside_x.append(x)
            outside_y.append(y)

    pi_est = 4.0 * len(inside_x) / n_points

    fig, ax = plt.subplots(figsize=(6, 6))
    theta = [i * math.pi / 200.0 for i in range(101)]
    cx = [math.cos(t) for t in theta]
    cy = [math.sin(t) for t in theta]

    ax.scatter(inside_x, inside_y, color="forestgreen", s=8, alpha=0.6, label=f"Inside (Hits={len(inside_x)})")
    ax.scatter(outside_x, outside_y, color="crimson", s=8, alpha=0.6, label=f"Outside (Misses={len(outside_x)})")
    ax.plot(cx, cy, "k-", linewidth=2, label="Quarter-Circle Boundary")

    ax.set_title(f"{title}\nEstimated $\pi = {pi_est:.4f}$ (Error: {abs(pi_est - math.pi):.4f})", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    # sin(x) from 0 to pi. Exact answer: [-cos(x)] from 0 to pi = 2.
    result = integrate_1d(math.sin, 0, math.pi, n=100000, seed=1)
    print(f"integral of sin(x), 0..pi  = {result['estimate']:.4f}  "
          f"(exact = 2.0000)  CI95={tuple(round(v,4) for v in result['ci95'])}")

    result = estimate_pi_hit_or_miss(n=1000000, seed=1)
    print(f"pi estimate = {result['estimate']:.4f}  (exact = {math.pi:.4f})")
