"""
Practice 2 (Res/Src-26/practice-problems.md): Expected value of a dice game.

Roll 3 dice: all same -> +$20, exactly two same -> +$5, all different -> -$2.
trial_fn returns the payoff of ONE game; the Monte Carlo mean over many
games is the sample estimate of the game's expected value.

Run standalone:
    python -m solutions.practice_src26.p2_dice_game_ev
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


def dice_game_payoff():
    # dice = [random.randint(1, 6) for _ in range(3)]
    # distinct = len(set(dice))
    # if distinct == 1:
    #     return 20.0
    # if distinct == 2:
    #     return 5.0
    # return -2.0
    d1 = random.randint(1,6)
    d2 = random.randint(1,6)
    d3 = random.randint(1,6)
    if d1==d2==d3:
        return 20.0
    if d1==d2 or d2==d3 or d3==d1:
        return 5.0
    return -2.0


if __name__ == "__main__":
    result = monte_carlo_estimate(dice_game_payoff, n=100000, seed=1)
    print(f"Expected value per game = {result['estimate']:.4f}  "
          f"CI95={tuple(round(v, 4) for v in result['ci95'])}")
