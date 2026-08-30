"""
C1 (online exam, section C1, see Res/questions.txt): Investigating the
Middle-Square Random Number Generator.

All four tasks are answered here purely by calling the reusable
rng/middle_square.py and testing/chi_square_test.py modules -- this file
just wires them together and prints the required tables.

Run standalone:
    python -m solutions.c1_middle_square_investigation
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from rng.middle_square import middle_square, middle_square_uniforms, find_middle_square_cycle
from testing.chi_square_test import chi_square_uniform_test

PROBLEMATIC_SEED = 2500  # 2500^2 = 06250000 -> middle four digits = 2500 (stuck immediately)


def task1_generate():
    seed = 5731
    values = middle_square(seed, 100)
    print(f"TASK 1: 100 values generated from seed {seed}")
    print(values)
    print()
    return values


def task2_edge_case():
    repeated_value, repeat_iteration, cycle_length = find_middle_square_cycle(PROBLEMATIC_SEED)
    print("TASK 2: Edge case")
    print(f"  Seed:                     {PROBLEMATIC_SEED}")
    print(f"  First repeated value:     {repeated_value}")
    print(f"  Iteration at which it repeats: {repeat_iteration}")
    print(f"  Cycle length:             {cycle_length}")
    print()


def task3_chi_square():
    print("TASK 3: Chi-Square test, N=1000, 10 bins, alpha=0.05")
    print("| Seed             | Chi^2    | p-value   | Decision          |")
    print("|------------------|----------|-----------|-------------------|")
    for label, seed in [("5731", 5731), ("6239", 6239), ("Problematic Seed", PROBLEMATIC_SEED)]:
        uniforms = middle_square_uniforms(seed, 1000)
        result = chi_square_uniform_test(uniforms)
        print(f"| {label:<16} | {result['chi2']:8.4f} | {result['p_value']:.6g} | {result['decision']:<17} |")
    print()


def task4_reflect():
    print("TASK 4: Reflection")
    print(
        "No. Passing the Chi-Square test only shows the ONE-DIMENSIONAL bin\n"
        "counts look uniform in aggregate -- it says nothing about how\n"
        "consecutive values relate to each other, or how short the generator's\n"
        "cycle really is. This dataset makes that concrete: seeds 5731 and 6239\n"
        "look like 'normal' seeds (no obvious problem in the first 10 values),\n"
        "yet find_middle_square_cycle shows BOTH collapse into the same short\n"
        "4-value cycle (6100 -> 2100 -> 4100 -> 8100 -> 6100 -> ...) well before\n"
        "1000 iterations (at iteration 75 and 111 respectively). Once a stream\n"
        "is dominated by a repeating cycle, its Chi-Square statistic explodes\n"
        "and it correctly REJECTS H0 -- so here Chi-Square actually catches the\n"
        "problem. But it would NOT catch every problem: a generator can have\n"
        "strong autocorrelation/lattice structure that is invisible to a 1-D\n"
        "bin count while still passing Chi-Square (see rng/lcg.py's RANDU note:\n"
        "passes 1-D uniformity, fails badly once you look at consecutive\n"
        "TRIPLES). A trustworthy RNG needs to pass uniformity tests (Chi-Square,\n"
        "K-S) AND independence tests (autocorrelation, runs) AND have an\n"
        "adequately long period for the intended use -- Chi-Square alone checks\n"
        "only the first of these."
    )


if __name__ == "__main__":
    task1_generate()
    task2_edge_case()
    task3_chi_square()
    task4_reflect()
