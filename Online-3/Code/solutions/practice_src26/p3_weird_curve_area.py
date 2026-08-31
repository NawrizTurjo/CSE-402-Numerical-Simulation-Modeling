"""
Practice 3 (Res/Src-26/practice-problems.md): Area under y = sin(x)*cos(x^2)
from x=0 to x=pi. No closed-form antiderivative -- exactly the case Monte
Carlo integration is for. Reuses monte_carlo.integration.integrate_1d
unchanged (same sample-mean method as the sin(x) example in that module).

Run standalone:
    python -m solutions.practice_src26.p3_weird_curve_area
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

def f(x):
    return math.sin(x) * math.cos(x**2)


def monte_carlo_area(n=200000, seed=1):
    random.seed(seed)

    a = 0
    b = math.pi
    width = b - a

    total = 0
    total_sq = 0

    for i in range(n):
        # Random x between 0 and pi
        x = a + (b-a)*(random.random())

        # y = sin(x) * cos(x^2)
        y = math.sin(x) * math.cos(x**2)

        # Monte Carlo integration
        value = width * y

        total += value
        total_sq += value * value

    # Mean = estimated area
    mean = total / n

    # Standard deviation
    variance = (total_sq - n * mean**2) / (n - 1)
    std_dev = math.sqrt(max(0, variance))

    # Standard error
    std_error = std_dev / math.sqrt(n)

    # 95% confidence interval
    low = mean - 1.96 * std_error
    high = mean + 1.96 * std_error

    print("Area estimate =", round(mean, 5))
    print("Standard error =", round(std_error, 5))
    print("95% CI =", (round(low, 5), round(high, 5)))


monte_carlo_area()


# if __name__ == "__main__":
#     result = integrate_1d(f, 0, math.pi, n=200000, seed=1)
#     print(f"Area estimate = {result['estimate']:.5f}  std_error={result['std_error']:.5f}  "
#           f"CI95={tuple(round(v, 5) for v in result['ci95'])}")
