"""
Python's `random` module -- syntax reference. NOT used inside rng/lcg.py
or rng/middle_square.py (those must not use it, C1's question says so
explicitly), but everything ELSE in this course's problems -- Monte
Carlo trial_fn closures, DES interarrival/service sampling, Metropolis-
Hastings proposals -- leans on it constantly.

Run standalone (prints every section):
    python -m cheatsheets.random_module_syntax
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


def basics():
    random.seed(42)   # reproducibility: same seed -> same sequence, every run

    print("random module basics")
    print("random.random()             ->", random.random())          # U[0,1)
    print("random.uniform(5, 10)       ->", random.uniform(5, 10))     # U[5,10)
    print("random.randint(1, 6)        ->", random.randint(1, 6))      # inclusive both ends (dice roll)
    print("random.randrange(1, 7)      ->", random.randrange(1, 7))    # exclusive upper bound (7 excluded)
    print("random.choice(['H', 'T'])   ->", random.choice(["H", "T"]))
    print("random.sample(range(10), 3) ->", random.sample(range(10), 3))  # k distinct items, no repeats

    items = [1, 2, 3, 4, 5]
    random.shuffle(items)   # shuffles IN PLACE, returns None
    print("random.shuffle([1..5])      ->", items)


def continuous_distributions():
    """Handy when a problem says 'draw from an Exponential/Normal
    distribution' and doesn't require you to hand-roll inverse-transform
    yourself (compare variate_generation/inverse_transform.py, which
    IS the hand-rolled version, needed when the problem gives you raw
    uniforms/an LCG stream instead)."""
    random.seed(1)
    print("\nContinuous distributions")
    print("random.expovariate(1/mean=1/2) ->", random.expovariate(1 / 2))  # mean=2
    print("random.gauss(mu=0, sigma=1)    ->", random.gauss(0, 1))
    print("random.triangular(0, 10, 3)    ->", random.triangular(0, 10, 3))  # low, high, mode


def reproducibility_check():
    print("\nReproducibility with a fixed seed")
    random.seed(7)
    first_run = [random.random() for _ in range(5)]
    random.seed(7)
    second_run = [random.random() for _ in range(5)]
    print("same seed -> identical sequence:", first_run == second_run)


def main():
    basics()
    continuous_distributions()
    reproducibility_check()


if __name__ == "__main__":
    main()
