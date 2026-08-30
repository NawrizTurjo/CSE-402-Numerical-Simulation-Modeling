"""
Middle-Square Method (von Neumann, 1946).

Algorithm (this is exactly Task 1 of the C1 question in Res/questions.txt,
which used a 4-digit seed -- the default here -- but see the `digits`
parameter below in case the exam gives you a different width):
    1. Start with a `digits`-digit seed X0.
    2. Square it -> up to 2*digits digits. Left-pad with zeros to force
       exactly 2*digits digits.
    3. The next value is the MIDDLE `digits` digits of that string.
    4. Repeat.

Worked example (from the question, digits=4):
    seed = 5731
    5731^2 = 32844361            -> zero-padded to 8 digits: "32844361"
    middle four digits (idx 2:6) = "8443"
    8443^2 = 71284249            -> "71284249"
    middle four digits           = "2842"

Known weakness (Task 2): the method can degenerate.
    - If a value squares to something whose middle digits equal itself,
      the generator gets stuck on that value forever (cycle length 1).
    - It can also enter longer cycles, or collapse toward 0 (0^2 = 0 forever).
    A concrete degenerate seed (digits=4): 2500 -> 2500^2 = 06250000 ->
    middle 4 = 2500. So seed 2500 repeats after 1 iteration, cycle length 1.
    This isn't a fluke of one bad seed either: even "normal-looking" seeds
    like 5731 and 6239 collapse into a short cycle well within 1000
    iterations (see Code/solutions/c1_middle_square_investigation.py's
    Task 4 for the exact numbers) -- the state space is only 10^digits
    values, and squaring is not a 1-to-1 map, so short cycles are the norm,
    not the exception, for this method at any digit width.

Fix (see `middle_square_weyl` below): mixing in a full-period Weyl
sequence (a running sum of a fixed odd increment, mod 10^digits) makes
the combined state effectively (x, w) instead of just x, so it can no
longer get trapped revisiting the same x forever.

Run standalone:
    python -m rng.middle_square
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from rng.common import find_cycle, bin_counts


def middle_square_step(x, digits=4):
    """One step: x -> middle `digits` digits of x^2 (zero-padded to 2*digits digits)."""
    width = 2 * digits
    offset = digits // 2
    square_as_digits = f"{x * x:0{width}d}"
    return int(square_as_digits[offset:offset + digits])


def middle_square(seed, n, digits=4):
    """
    Generate n successive values X1..Xn (each an integer with up to
    `digits` digits) starting from `seed`, using the middle-square
    recurrence. digits=4 (the C1 question's width) is the default.
    """
    max_seed = 10**digits - 1
    if not isinstance(seed, int) or not (0 <= seed <= max_seed):
        raise ValueError(f"seed must be an integer in [0, {max_seed}]")

    values = []
    current = seed
    for _ in range(n):
        current = middle_square_step(current, digits)
        values.append(current)
    return values


def middle_square_uniforms(seed, n, digits=4):
    """Same sequence as middle_square(), rescaled to floats in [0, 1)."""
    scale = 10**digits
    return [value / scale for value in middle_square(seed, n, digits)]


def find_middle_square_cycle(seed, digits=4):
    """Task 2: locate the first repeat, its iteration, and the cycle length."""
    return find_cycle(lambda x: middle_square_step(x, digits), seed)


def middle_square_weyl(seed, n, digits=4, weyl_increment=3571):
    """
    Middle-Square with a Weyl-sequence repair for the short-cycle weakness:

        w_{i+1} = (w_i + s) mod 10^digits            (s odd -> full period 10^digits)
        x_{i+1} = (middle_square_step(x_i) + w_{i+1}) mod 10^digits

    `weyl_increment` (s) must be odd, or the Weyl sequence itself won't
    reach full period. The state is now effectively the PAIR (x, w) --
    since w cycles through all 10^digits values before repeating, x can
    never get trapped revisiting the same value forever the way plain
    middle-square does (verified against all three seeds used in the C1
    solution -- 2500, 5731, 6239 -- each of which fails Chi-Square as
    plain middle-square but passes comfortably once repaired this way).
    """
    if weyl_increment % 2 == 0:
        raise ValueError("weyl_increment must be odd for a full-period Weyl sequence")

    modulus = 10**digits
    values = []
    x, w = seed, 0
    for _ in range(n):
        w = (w + weyl_increment) % modulus
        mid = middle_square_step(x, digits)
        x = (mid + w) % modulus
        values.append(x)
    return values


def middle_square_weyl_uniforms(seed, n, digits=4, weyl_increment=3571):
    """Same sequence as middle_square_weyl(), rescaled to floats in [0, 1)."""
    scale = 10**digits
    return [value / scale for value in middle_square_weyl(seed, n, digits, weyl_increment)]


if __name__ == "__main__":
    seed = 5731
    first_values = middle_square(seed, 10)
    print(f"First 10 values from seed {seed}: {first_values}")

    problem_seed = 2500
    repeated_value, repeat_iteration, cycle_length = find_middle_square_cycle(problem_seed)
    print(f"\nProblematic seed        : {problem_seed}")
    print(f"First repeated value    : {repeated_value}")
    print(f"Iteration it repeats at : {repeat_iteration}")
    print(f"Cycle length            : {cycle_length}")

    uniforms = middle_square_uniforms(seed, 1000)
    print(f"\nBin counts (10 bins) for {len(uniforms)} values: {bin_counts(uniforms)}")

    print("\nDifferent seed widths (digits=2,4,6), first 5 values each:")
    for digits, example_seed in ((2, 57), (4, 5731), (6, 577777)):
        values = middle_square(example_seed, 5, digits=digits)
        print(f"  digits={digits}  seed={example_seed}  -> {values}")

    print("\nWeyl-repaired seed 2500 (degenerates immediately in plain middle-square):")
    weyl_values = middle_square_weyl(problem_seed, 10)
    middle_square_values = middle_square(problem_seed,10)
    print(f"========== Middle Square ==========")
    print(f"  first 10 values: {middle_square_values}")
    print(f"  no longer stuck: {len(set(middle_square_values)) > 1}")

    print(f"========== Weyl Repaired ==========")
    print(f"  first 10 values: {weyl_values}")
    print(f"  no longer stuck: {len(set(weyl_values)) > 1}")
