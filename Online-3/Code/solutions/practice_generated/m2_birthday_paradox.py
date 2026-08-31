"""
Practice M2: Birthday Paradox.
Simple, self-contained Monte Carlo discrete event probability.
"""

import math
import random


def birthday_paradox(num_people=23, n=100000, seed=1):
    random.seed(seed)

    success = 0

    for _ in range(n):
        seen = set()
        for _ in range(num_people):
            bday = random.randint(1, 365)
            if bday in seen:
                success += 1
                break
            seen.add(bday)

    p_hat = success / float(n)
    std_error = math.sqrt(p_hat * (1.0 - p_hat) / n)
    ci_low = p_hat - 1.96 * std_error
    ci_high = p_hat + 1.96 * std_error

    # Exact closed-form formula: 1 - product_{j=0}^{k-1} (365 - j)/365
    p_distinct = 1.0
    for j in range(num_people):
        p_distinct *= (365 - j) / 365.0
    exact = 1.0 - p_distinct

    print(f"--- Birthday Paradox Simulation (Group Size = {num_people}) ---")
    print(f"P(at least 2 people share birthday) = {p_hat:.4f}")
    print(f"Exact Theoretical Probability       = {exact:.4f}")
    print(f"Standard Error                      = {std_error:.4f}")
    print(f"95% CI                              = ({ci_low:.4f}, {ci_high:.4f})")


birthday_paradox()
