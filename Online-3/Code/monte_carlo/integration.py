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


if __name__ == "__main__":
    # sin(x) from 0 to pi. Exact answer: [-cos(x)] from 0 to pi = 2.
    result = integrate_1d(math.sin, 0, math.pi, n=100000, seed=1)
    print(f"integral of sin(x), 0..pi  = {result['estimate']:.4f}  "
          f"(exact = 2.0000)  CI95={tuple(round(v,4) for v in result['ci95'])}")

    result = estimate_pi_hit_or_miss(n=1000000, seed=1)
    print(f"pi estimate = {result['estimate']:.4f}  (exact = {math.pi:.4f})")
