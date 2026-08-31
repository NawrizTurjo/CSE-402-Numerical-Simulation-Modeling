"""
Practice M1: Estimating integral_0^1 e^(-x^2) dx.
Simple, self-contained Monte Carlo integration.
"""

import math
import random


def estimate_gaussian_integral(n=100000, seed=1):
    random.seed(seed)

    total = 0.0
    total_sq = 0.0

    for _ in range(n):
        x = random.random()  # draw X ~ Uniform(0, 1)
        y = math.exp(-x * x)  # evaluate f(X) = e^(-X^2)
        total += y
        total_sq += y * y

    estimate = total / n
    variance = (total_sq - n * estimate**2) / (n - 1)
    std_error = math.sqrt(variance / n)

    # 95% confidence interval
    ci_low = estimate - 1.96 * std_error
    ci_high = estimate + 1.96 * std_error

    # Exact value via error function: (sqrt(pi)/2) * erf(1)
    exact = math.sqrt(math.pi) / 2 * math.erf(1)

    print("--- Monte Carlo Integration of e^(-x^2) over [0, 1] ---")
    print(f"Estimate (N={n:,}) = {estimate:.5f}")
    print(f"Exact Value        = {exact:.5f}")
    print(f"Absolute Error     = {abs(estimate - exact):.5f}")
    print(f"Standard Error     = {std_error:.5f}")
    print(f"95% CI             = ({ci_low:.5f}, {ci_high:.5f})")


estimate_gaussian_integral()
