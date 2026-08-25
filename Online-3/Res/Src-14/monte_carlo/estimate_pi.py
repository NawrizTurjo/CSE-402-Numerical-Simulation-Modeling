import random

def monte_carlo_pi(n, seed=None):
    """
    Estimate pi using hit-or-miss Monte Carlo.
    Throw random points into unit square,
    count fraction that land inside quarter circle.
    """
    if seed is not None:
        random.seed(seed)
    hits = 0
    for _ in range(n):
        x = random.random()   # U[0,1]
        y = random.random()   # U[0,1]
        if x**2 + y**2 <= 1:
            hits += 1
    return 4 * hits / n

print(monte_carlo_pi(1000000, seed=1))   # ≈ 3.1416