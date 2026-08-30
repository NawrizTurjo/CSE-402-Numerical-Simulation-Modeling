"""
Practice 1 (Res/Src-26/practice-problems.md): Volume of a 3D sphere.

Sphere of radius 1 inside a cube of side 2 (x,y,z in [-1,1], cube volume 8).
Hit-or-miss: trial_fn returns cube_volume if the random point falls inside
the sphere (x^2+y^2+z^2<=1), else 0 -- average = cube_volume * P(inside)
= sphere volume. Same pattern as monte_carlo.integration.estimate_pi_hit_or_miss,
just in 3 dimensions instead of 2.

Run standalone:
    python -m solutions.practice_src26.p1_sphere_volume
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
import random

from monte_carlo.core import monte_carlo_estimate


def sphere_volume(n, seed=None):
    cube_volume = 2 * 2 * 2

    def trial():
        x, y, z = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
        return cube_volume if x * x + y * y + z * z <= 1 else 0.0

    return monte_carlo_estimate(trial, n, seed=seed)


if __name__ == "__main__":
    result = sphere_volume(n=200000, seed=1)
    exact = 4 / 3 * math.pi
    print(f"Volume estimate = {result['estimate']:.4f}  (exact = {exact:.4f})")
