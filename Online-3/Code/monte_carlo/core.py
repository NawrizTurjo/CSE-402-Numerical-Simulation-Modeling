"""
ONE generic Monte Carlo estimator that every Monte Carlo problem in this
course reduces to. This is the single most important file to understand
for the exam -- once you get this, every practice problem (pi estimation,
Buffon's needle, sphere volume, dice-game expected value, random-walk
probability, project-completion probability, integrating a weird
function) is just "what do I put in `trial_fn`?".

Why one function covers all of it:

    A Monte Carlo estimate is ALWAYS the average of many independent
    trials of some random quantity Y:

        estimate = (1/n) * sum(Y_1, Y_2, ..., Y_n)

    - Estimating a PROBABILITY P(event)?
        Y = 1 if the event happens in this trial, else 0.
        Average of 0/1 values = fraction of trials where it happened = P(event).
        (pi via hit-or-miss, Buffon's needle crossing prob., random-walk
         >15 prob., project >11 days prob. -- all this shape.)

    - Estimating an EXPECTED VALUE / payoff?
        Y = the payoff/outcome of one trial (e.g. dice-game winnings).
        Average of outcomes = sample estimate of E[Y]. (dice game)

    - Estimating an INTEGRAL  integral_a^b f(x) dx  ?
        Y = (b - a) * f(X),  X ~ Uniform(a, b).
        E[Y] = (b-a) * E[f(X)] = (b-a) * integral_a^b f(x) * (1/(b-a)) dx
             = integral_a^b f(x) dx.               (weird-curve area, sin(x) integral)

    - Estimating a VOLUME (hit-or-miss in n dimensions)?
        Y = box_volume  if point is inside the region, else 0.
        Average of Y = box_volume * P(inside) = region's volume.
                                                  (sphere volume, circle/pi)

So every wrapper in this package (integration.py, and the solutions/
files) is a few lines that build a `trial_fn` closure and hand it to
`monte_carlo_estimate`. Nothing else changes.
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


def monte_carlo_estimate(trial_fn, n, seed=None):
    """
    Run `trial_fn()` n times (each call = one independent random trial
    returning a single float) and return the sample-mean estimate with
    its standard error and a 95% confidence interval.

    Parameters
    ----------
    trial_fn : callable
        Zero-argument function. Each call performs ONE random trial and
        returns a numeric outcome (0/1 for probabilities, a payoff for
        expected value, box_volume*indicator for hit-or-miss volume,
        width*f(x) for integration -- see module docstring above).
    n : number of independent trials to run.
    seed : optional seed passed to `random.seed()` for reproducibility.
           trial_fn must use the shared `random` module (not its own
           generator) for this to have any effect.

    Returns
    -------
    dict with:
        estimate   : sample mean = the Monte Carlo answer.
        std_dev    : sample standard deviation of the n outcomes.
        std_error  : std_dev / sqrt(n)  (this is why error shrinks as 1/sqrt(n),
                     REGARDLESS of dimension -- the key advantage of MC
                     over grid methods like Simpson's rule in high-D).
        ci95       : (low, high) 95% confidence interval, mean +/- 1.96*std_error.
    """
    if seed is not None:
        random.seed(seed)

    total = 0.0
    total_sq = 0.0
    for _ in range(n):
        y = trial_fn()
        total += y
        total_sq += y * y

    mean = total / n
    variance = max(0.0, (total_sq - n * mean * mean) / (n - 1)) if n > 1 else 0.0
    std_dev = math.sqrt(variance)
    std_error = std_dev / math.sqrt(n)

    return {
        "estimate": mean,
        "std_dev": std_dev,
        "std_error": std_error,
        "ci95": (mean - 1.96 * std_error, mean + 1.96 * std_error),
    }


def plot_monte_carlo_convergence(trial_fn, n=10000, true_value=None, seed=None,
                                 title="Monte Carlo Estimate Convergence vs N", show=True, save_path=None):
    """
    Plot running Monte Carlo estimate and 95% confidence interval band as sample size N grows.

    Parameters
    ----------
    trial_fn : callable
        Function returning one random sample y.
    n : int, default=10000
        Total sample count.
    true_value : float, optional
        Exact theoretical value for reference line.
    seed : int, optional
        Random seed.
    title : str, default='Monte Carlo Estimate Convergence vs N'
        Figure title.
    show : bool, default=True
        Whether to call plt.show().
    save_path : str, optional
        If provided, save the figure to this file path.
    """
    import matplotlib.pyplot as plt

    if seed is not None:
        random.seed(seed)

    running_estimates = []
    ci_lowers = []
    ci_uppers = []
    n_points = []

    total = 0.0
    total_sq = 0.0

    step = max(1, n // 200)
    for i in range(1, n + 1):
        y = trial_fn()
        total += y
        total_sq += y * y

        if i % step == 0 or i == n:
            mean = total / i
            var = max(0.0, (total_sq - i * mean * mean) / (i - 1)) if i > 1 else 0.0
            se = math.sqrt(var / i)
            n_points.append(i)
            running_estimates.append(mean)
            ci_lowers.append(mean - 1.96 * se)
            ci_uppers.append(mean + 1.96 * se)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n_points, running_estimates, color="royalblue", linewidth=2, label="MC Estimate $\hat{\mu}_N$")
    ax.fill_between(n_points, ci_lowers, ci_uppers, color="royalblue", alpha=0.2, label="95% Confidence Band")
    if true_value is not None:
        ax.axhline(true_value, color="crimson", linestyle="--", linewidth=2, label=f"True Value ({true_value:.4f})")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample Size (N)", fontsize=11)
    ax.set_ylabel("Estimate", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    # Sanity check: probability that two dice sum to 7 (true answer = 1/6).
    def trial():
        d1 = random.randint(1, 6)
        d2 = random.randint(1, 6)
        return 1.0 if d1 + d2 == 7 else 0.0

    result = monte_carlo_estimate(trial, n=200000, seed=1)
    print(f"P(sum=7) estimate = {result['estimate']:.4f}  "
          f"(true = {1/6:.4f})  CI95={result['ci95']}")
    plot_monte_carlo_convergence(trial, n=200000, seed=1)
