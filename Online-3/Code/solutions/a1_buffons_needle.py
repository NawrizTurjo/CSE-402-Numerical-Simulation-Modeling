"""
A1 (CSE 402 Online-3 Official Exam Question): Buffon's Needle Simulation.

Problem:
Estimate the mathematical constant pi using Monte Carlo simulation of
the classical Buffon's Needle Experiment.
A needle of length L = 1.0 is dropped onto a surface with parallel lines
spaced D = 2.0 apart (L <= D).

Mathematical Modeling:
1. Needle Center Position: X ~ U(0, D/2) -> x = (D / 2.0) * random.random()
2. Needle Orientation Angle: theta ~ U(0, pi/2) -> theta = (math.pi / 2.0) * random.random()
3. Hit Condition: x <= (L / 2.0) * math.sin(theta)
4. Theoretical Crossing Probability: P = 2L / (pi * D)
   For L=1.0, D=2.0 -> P = 1 / pi approx 0.31831
5. Estimator:
   p_hat = H / N
   pi_estimate = (2 * L) / (D * p_hat) = (2 * L * N) / (D * H)
   For L=1.0, D=2.0 -> pi_estimate = N / H = 1 / p_hat

Design:
Reuses `monte_carlo.core.monte_carlo_estimate` by providing a single-drop
`trial_fn` that returns the 0/1 hit indicator.

Run standalone:
    python -m solutions.a1_buffons_needle
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


def buffon_needle_pi(n_drops=None, L=1.0, D=2.0, seed=None, **kwargs):
    """
    Simulate Buffon's Needle Experiment to estimate pi.
    Reuses monte_carlo_estimate from monte_carlo.core.

    Parameters
    ----------
    n_drops : int
        Total number of drops (N).
    L : float, default=1.0
        Length of the needle (L <= D).
    D : float, default=2.0
        Distance between parallel lines.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - 'n_drops': Total drops N
        - 'hits': Total crossings H
        - 'p_hat': Empirical crossing probability (H / N)
        - 'pi_estimate': Estimated value of pi
        - 'abs_error': |pi_estimate - pi|
        - 'ci95': 95% confidence interval on crossing probability
        - 'p_cross': Alias for p_hat (backwards compatibility)
    """
    if n_drops is None:
        n_drops = kwargs.get("n", 100000)

    half_D = D / 2.0
    half_L = L / 2.0
    half_pi = math.pi / 2.0

    def drop_trial():
        # 1. Midpoint distance to nearest line: X ~ U(0, D/2)
        x = half_D * random.random()
        # 2. Angle with lines: theta ~ U(0, pi/2)
        theta = half_pi * random.random()
        # 3. Hit condition
        return 1.0 if x <= half_L * math.sin(theta) else 0.0

    # Re-use the shared Monte Carlo engine
    mc_result = monte_carlo_estimate(drop_trial, n_drops, seed=seed)
    p_hat = mc_result["estimate"]
    hits = int(round(p_hat * n_drops))
    pi_est = (2.0 * L) / (D * p_hat) if p_hat > 0 else float("inf")
    abs_err = abs(pi_est - math.pi)

    return {
        "n_drops": n_drops,
        "hits": hits,
        "p_hat": p_hat,
        "pi_estimate": pi_est,
        "abs_error": abs_err,
        "ci95": mc_result["ci95"],
        "p_cross": p_hat,
    }


# Alias matching previous helper naming
buffons_needle_pi = buffon_needle_pi


if __name__ == "__main__":
    n_values = [100, 1000, 10000, 100000, 1000000]

    print("Buffon's Needle Simulation (L=1.0, D=2.0) [reusing monte_carlo.core]")
    print("-" * 75)
    print(f"| {'N (Drops)':<11} | {'Hits (H)':<10} | {'p_hat (H/N)':<11} | {'Pi Estimate':<11} | {'Absolute Error':<14} |")
    print(f"|{'-'*13}|{'-'*12}|{'-'*13}|{'-'*13}|{'-'*16}|")

    # Fixed seed sequence for reproducible sample output
    for i, N in enumerate(n_values):
        res = buffon_needle_pi(N, L=1.0, D=2.0, seed=42 + i)
        n_str = f"{res['n_drops']:,}"
        h_str = f"{res['hits']:,}"
        print(
            f"| {n_str:<11} | {h_str:<10} | {res['p_hat']:<11.5f} | "
            f"{res['pi_estimate']:<11.6f} | {res['abs_error']:<14.6f} |"
        )
    print("-" * 75)
