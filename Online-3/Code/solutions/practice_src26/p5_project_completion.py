"""
Practice 5 (Res/Src-26/practice-problems.md): Project completion time (PERT-style).

Three sequential tasks, each uniformly distributed:
    A ~ U(2,4), B ~ U(3,6), C ~ U(1,5)
Estimate P(A+B+C > 11 days). Sequential tasks just mean we sum three
independent uniform draws per trial -- still a single 0/1 indicator
averaged by the shared estimator.

Run standalone:
    python -m solutions.practice_src26.p5_project_completion
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


def project_over_deadline(deadline=11.0):
    duration = random.uniform(2, 4) + random.uniform(3, 6) + random.uniform(1, 5)
    return 1.0 if duration > deadline else 0.0


if __name__ == "__main__":
    result = monte_carlo_estimate(project_over_deadline, n=100000, seed=1)
    print(f"P(total project time > 11 days) = {result['estimate']:.4f}  "
          f"CI95={tuple(round(v, 4) for v in result['ci95'])}")
