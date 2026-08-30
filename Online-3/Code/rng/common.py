"""
Shared helpers used by BOTH random-number generators in this folder
(rng/lcg.py and rng/middle_square.py).

The point of this file: cycle-detection is the exact same algorithm no
matter which generator produces the sequence, because any generator of
the form  X_{i+1} = step(X_i)  over a FINITE state space MUST eventually
repeat a state (pigeonhole principle). Once a state repeats, everything
after it repeats too -> that's a cycle. So we only need to implement
"detect the first repeat" once, and hand it a `step` function.

If a future exam question introduces a NEW generator (e.g. a Fibonacci
generator, or a different digit-extraction rule), you don't write new
cycle-detection code -- you just write its `step(x) -> next_x` function
and pass it in here.
"""



import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

def find_cycle(step, seed):
    """
    Run the generator until a value repeats, tracking when each value
    was first seen (dict lookup = O(1), so this is O(cycle length)).

    Parameters
    ----------
    step : callable
        step(x) -> next value in the sequence.
    seed : the starting value X0.

    Returns
    -------
    repeated_value   : the first value that gets generated twice.
    repeat_iteration : the iteration count (1-based number of steps taken)
                        at which the repeat was observed.
    cycle_length      : length of the repeating cycle (period).

    Example (Middle-Square, seed=2500):
        2500 -> 2500^2 = 06250000 -> middle 4 = 2500   (repeats immediately!)
        find_cycle returns (2500, 1, 1)
    """
    seen_at = {}
    current = seed
    iteration = 0

    while current not in seen_at:
        seen_at[current] = iteration
        current = step(current)
        iteration += 1

    first_seen_iteration = seen_at[current]
    cycle_length = iteration - first_seen_iteration
    return current, iteration, cycle_length


def bin_counts(uniforms, bins=10):
    """
    Count how many values in `uniforms` (each expected in [0, 1)) fall
    into each of `bins` equal-width buckets. Shared by every RNG/testing
    module that needs a histogram before running a Chi-Square test.
    """
    counts = [0] * bins
    for value in uniforms:
        index = min(bins - 1, int(value * bins))  # clamp value==1.0 edge case into last bin
        counts[index] += 1
    return counts
