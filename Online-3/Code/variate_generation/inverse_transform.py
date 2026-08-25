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
    if r <= split:
        return low + math.sqrt(r * (high - low) * (mode - low))
    return high - math.sqrt((1 - r) * (high - low) * (high - mode))


if __name__ == "__main__":
    from rng.lcg import lcg_uniforms

    uniforms = lcg_uniforms(seed=27, a=17, c=43, m=100, n=5)
    print("Uniforms:", uniforms)
    print("-> Uniform[5,10]      :", [round(uniform(r, 5, 10), 4) for r in uniforms])
    print("-> Exponential(mean=2):", [round(exponential(r, 2.0), 4) for r in uniforms])
