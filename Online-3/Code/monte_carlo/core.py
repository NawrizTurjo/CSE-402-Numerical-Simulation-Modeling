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


if __name__ == "__main__":
    # Sanity check: probability that two dice sum to 7 (true answer = 1/6).
    def trial():
        d1 = random.randint(1, 6)
        d2 = random.randint(1, 6)
        return 1.0 if d1 + d2 == 7 else 0.0

    result = monte_carlo_estimate(trial, n=200000, seed=1)
    print(f"P(sum=7) estimate = {result['estimate']:.4f}  "
          f"(true = {1/6:.4f})  CI95={result['ci95']}")
