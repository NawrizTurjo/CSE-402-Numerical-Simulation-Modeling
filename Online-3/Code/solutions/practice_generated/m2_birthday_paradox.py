"""
Practice M2: The Birthday Paradox (see ../../../Practice/PRACTICE_QUESTIONS.md).
A discrete-probability Monte Carlo estimate -- same 0/1-indicator shape
as Buffon's Needle, just with a combinatorial event instead of a
geometric one, to show the estimator doesn't care which.

Run standalone:
    python -m solutions.practice_generated.m2_birthday_paradox
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import random

from monte_carlo.core import monte_carlo_estimate


def shared_birthday_trial(num_people=23, days_in_year=365):
    seen = set()
    for _ in range(num_people):
        day = random.randint(1, days_in_year)
        if day in seen:
            return 1.0
        seen.add(day)
    return 0.0


def task1_2_estimate():
    print("TASK 1-2: P(at least 2 of 23 people share a birthday)")
    result = monte_carlo_estimate(shared_birthday_trial, n=100000, seed=1)
    print(f"  estimate={result['estimate']:.4f}  "
          f"CI95={tuple(round(v, 4) for v in result['ci95'])}")
    print("  (true value, from the closed-form formula, is approximately 0.5073)")
    print()


def task3_group_size_trend():
    print("TASK 3: how the probability grows with group size")
    for n_people in (5, 10, 23, 40, 60):
        result = monte_carlo_estimate(lambda: shared_birthday_trial(n_people), n=50000, seed=1)
        print(f"  {n_people:>3} people -> P(shared) ~= {result['estimate']:.4f}")
    print("  Probability rises much faster than intuition suggests -- driven by")
    print("  the number of PAIRS of people, which grows like n^2, not n itself.")


if __name__ == "__main__":
    task1_2_estimate()
    task3_group_size_trend()
