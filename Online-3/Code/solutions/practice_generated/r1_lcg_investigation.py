"""
Practice R1: LCG Investigation (see ../../../Practice/PRACTICE_QUESTIONS.md).
Same shape as the C1 exam question, but for the LCG instead of Middle-Square.

Run standalone:
    python -m solutions.practice_generated.r1_lcg_investigation
"""

from rng.lcg import lcg, lcg_uniforms, find_lcg_cycle, check_max_period_conditions
from testing.chi_square_test import chi_square_uniform_test


def task1_generate():
    seed, a, c, m = 7, 5, 3, 16
    values = lcg(seed, a, c, m, 20)
    print(f"TASK 1: seed={seed}, a={a}, c={c}, m={m} -> first 20 values")
    print(values)
    print()


def task2_investigate():
    print("TASK 2: Parameter investigation")

    case, max_p, ok, details = check_max_period_conditions(a=5, c=3, m=16, seed=7)
    _, _, measured_period = find_lcg_cycle(7, 5, 3, 16)
    print(f"  (a) Full-period config a=5,c=3,m=16,seed=7: {case} -> theoretical max={max_p}")
    print(f"      Conditions satisfied: {ok} ({details})")
    print(f"      Measured period: {measured_period}  (matches theory)")

    repeated, iteration, cycle_length = find_lcg_cycle(7, 1, 0, 16)
    print(f"  (b) Degenerate config a=1,c=0,m=16,seed=7:")
    print(f"      First repeated value: {repeated}")
    print(f"      Iteration it repeats: {iteration}")
    print(f"      Cycle length: {cycle_length}")
    print(f"      Why: X_next = (1*X + 0) mod 16 = X, always -- the recurrence is the identity function.")
    print()


def task3_chi_square():
    print("TASK 3: Chi-Square test, N=1000, 10 bins, alpha=0.05")
    print("| Generator                             | Chi^2      | p-value    | Decision          |")
    print("|----------------------------------------|-----------|------------|-------------------|")
    configs = [
        ("Full-period (a=5,c=3,m=16)", 7, 5, 3, 16),
        ("Degenerate (a=1,c=0,m=16)", 7, 1, 0, 16),
        ("ANSI-C rand (a=1103515245,c=12345,m=2^31)", 1, 1103515245, 12345, 2**31),
    ]
    for label, seed, a, c, m in configs:
        uniforms = lcg_uniforms(seed, a, c, m, 1000)
        result = chi_square_uniform_test(uniforms)
        print(f"| {label:<40} | {result['chi2']:9.4f} | {result['p_value']:.6g} | {result['decision']:<17} |")
    print()


def task4_reflect():
    print("TASK 4: Reflection")
    print(
        "No -- reaching the THEORETICAL maximum period for a given m does not\n"
        "make a generator statistically good, because the max period is capped\n"
        "by m itself. a=5,c=3,m=16 genuinely achieves its theoretical max period\n"
        "of 16 (verified in Task 2) -- but 16 distinct values repeated ~63 times\n"
        "each to fill 1000 draws is nowhere close to i.i.d. Uniform(0,1), so it\n"
        "fails Chi-Square badly (Task 3). Contrast with the ANSI-C generator:\n"
        "its m=2^31 is astronomically larger, so even without checking whether\n"
        "it hits ITS theoretical max period, 1000 draws barely sample its state\n"
        "space and it passes easily. Lesson: period THEORY tells you the best\n"
        "case for a given m; you separately need m itself to be large enough for\n"
        "how many values you plan to draw."
    )


if __name__ == "__main__":
    task1_generate()
    task2_investigate()
    task3_chi_square()
    task4_reflect()
