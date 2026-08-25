"""
A1 (online exam, section A1 -- reconstructed from a friend's memory, see
Res/questions.txt): Buffon's Needle.

Drop a needle of length L=0.5 onto a floor of parallel planks, each of
width 1. Estimate P(needle crosses a line) over 100,000 drops, then use
that probability to estimate pi.

Setup per drop:
    distance = distance from needle's CENTER to the nearest line,
               uniformly distributed in [0, width/2].
    angle    = angle the needle makes with the lines,
               uniformly distributed in [0, pi/2].
    crosses  = distance <= (L/2) * sin(angle)

Theory: P(cross) = 2L / (pi * width)   =>   pi = 2L / (P(cross) * width)

This is a plain probability estimate, so it's just one call into the
shared monte_carlo.core.monte_carlo_estimate (trial_fn returns the 0/1
crossing indicator) -- see that module's docstring for why every Monte
Carlo problem in this course reduces to the same estimator.

Run standalone:
    python -m solutions.a1_buffons_needle
"""

import math
import random

from monte_carlo.core import monte_carlo_estimate


def buffons_needle_pi(n, needle_length=0.5, plank_width=1.0, seed=None):
    half_width = plank_width / 2

    def trial():
        distance = random.uniform(0, half_width)
        angle = random.uniform(0, math.pi / 2)
        crosses = distance <= (needle_length / 2) * math.sin(angle)
        return 1.0 if crosses else 0.0

    result = monte_carlo_estimate(trial, n, seed=seed)
    p_cross = result["estimate"]
    result["p_cross"] = p_cross
    result["pi_estimate"] = (2 * needle_length) / (p_cross * plank_width) if p_cross > 0 else math.inf
    return result


if __name__ == "__main__":
    result = buffons_needle_pi(n=100000, seed=1)
    print(f"P(needle crosses a line) = {result['p_cross']:.4f}")
    print(f"pi estimate              = {result['pi_estimate']:.4f}  (true = {math.pi:.4f})")
    print(f"95% CI on P(cross)       = {tuple(round(v, 4) for v in result['ci95'])}")
