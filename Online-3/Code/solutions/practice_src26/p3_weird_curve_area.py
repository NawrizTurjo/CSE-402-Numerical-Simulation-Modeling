"""
Practice 3 (Res/Src-26/practice-problems.md): Area under y = sin(x)*cos(x^2)
from x=0 to x=pi. No closed-form antiderivative -- exactly the case Monte
Carlo integration is for. Reuses monte_carlo.integration.integrate_1d
unchanged (same sample-mean method as the sin(x) example in that module).

Run standalone:
    python -m solutions.practice_src26.p3_weird_curve_area
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

from monte_carlo.integration import integrate_1d


def f(x):
    return math.sin(x) * math.cos(x**2)


if __name__ == "__main__":
    result = integrate_1d(f, 0, math.pi, n=200000, seed=1)
    print(f"Area estimate = {result['estimate']:.5f}  std_error={result['std_error']:.5f}  "
          f"CI95={tuple(round(v, 5) for v in result['ci95'])}")
