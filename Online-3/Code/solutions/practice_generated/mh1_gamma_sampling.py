"""
Practice MH1: Metropolis-Hastings Sampling for Gamma(3, 1).
Target density: p(x) proportional to x^2 * e^(-x) for x > 0.
"""

import math
import random


def log_target(x):
    """Log of unnormalized target density p(x) = x^2 * e^(-x)."""
    if x <= 0:
        return float("-inf")
    return 2.0 * math.log(x) - x


def sample_gamma_mh(n=5000, step_size=1.5, x0=1.0, seed=42):
    random.seed(seed)

    samples = []
    x = x0
    accepted = 0

    for _ in range(n):
        # 1. Propose candidate using symmetric Normal random walk
        x_prop = x + random.gauss(0, step_size)
        log_alpha = log_target(x_prop) - log_target(x)

        # 2. Accept / Reject
        if log_alpha >= 0 or math.log(random.random()) < log_alpha:
            x = x_prop
            accepted += 1

        samples.append(x)

    # Discard burn-in (first 500 samples)
    kept = samples[500:]
    mean = sum(kept) / len(kept)
    variance = sum((s - mean) ** 2 for s in kept) / (len(kept) - 1)

    print("--- Metropolis-Hastings Sampling for Gamma(3, 1) ---")
    print(f"Total Samples   = {n:,} (Burn-in discarded = 500)")
    print(f"Acceptance Rate = {accepted / float(n):.4f}")
    print(f"Sample Mean     = {mean:.4f}  (True Mean = 3.0000)")
    print(f"Sample Variance = {variance:.4f}  (True Variance = 3.0000)")


sample_gamma_mh()
