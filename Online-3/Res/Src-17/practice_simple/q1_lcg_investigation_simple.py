"""
Practice Q1: Investigating the Linear Congruential Generator (LCG)
Simple, Student-Friendly Implementation
"""

from scipy.stats import chisquare
from math import gcd

# ==============================================================================
# Task 1: LCG Implementation
# ==============================================================================
def lcg(seed, a, c, m, n):
    """
    Generates n random numbers in [0, 1) using LCG:
        X_{i+1} = (a * X_i + c) mod m,   R_i = X_i / m
    """
    values = []
    x = seed
    for _ in range(n):
        x = (a * x + c) % m
        values.append(x / m)
    return values


# ==============================================================================
# Task 2: Finding Periods and Degeneracies
# ==============================================================================
def find_period(seed, a, c, m):
    """
    Returns (first_repeated_val, iteration_index, cycle_length)
    """
    seen = {}
    x = seed
    iteration = 0
    while x not in seen:
        seen[x] = iteration
        x = (a * x + c) % m
        iteration += 1

    first_seen = seen[x]
    cycle_len = iteration - first_seen
    return x, iteration, cycle_len


# ==============================================================================
# Task 3: Chi-Square Test
# ==============================================================================
def chi_square_test_10bins(numbers, alpha=0.05):
    k = 10
    observed = [0] * k
    for r in numbers:
        bin_idx = min(int(r * k), k - 1)
        observed[bin_idx] += 1

    chi2_stat, p_val = chisquare(observed)
    decision = "Reject H0" if p_val < alpha else "Do not reject H0"
    return chi2_stat, p_val, decision, observed


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TASK 1: Generate 100 values (a=17, c=43, m=100, X0=27)")
    print("=" * 70)
    nums_100 = lcg(seed=27, a=17, c=43, m=100, n=100)
    print("First 10 values:", [round(v, 2) for v in nums_100[:10]])

    print("\n" + "=" * 70)
    print("TASK 2: Period Investigation")
    print("=" * 70)
    val, it, cycle = find_period(27, 17, 43, 100)
    print(f"(a) Generator (17*X + 43) mod 100: Period = {cycle} (Only 4 distinct values: 02, 77, 52, 27!)")

    print("\n(b) Multiplicative LCG: X_{i+1} = 13*X_i mod 64")
    for s in (1, 2, 3, 4):
        _, _, p = find_period(s, 13, 0, 64)
        print(f"    Seed {s} -> Period: {p:2d} ({'Max period m/4=16 reached' if p==16 else 'Subgroup due to even seed'})")

    print("\n(c) Degenerate parameter sets:")
    print("    • c = 0, X0 = 0 -> Period 1 (Stuck at 0 forever)")
    print("    • a = 1, c = 0 -> Period 1 (X_{i+1} = X_i forever)")
    print("    • a = 5, c = 0, m = 25, X0 = 5 -> Collapses to 0 immediately")

    print("\n" + "=" * 70)
    print("TASK 3: Chi-Square Test (N = 1000, 10 Bins)")
    print("=" * 70)
    configs = [
        ("a=17, c=43, m=100, X0=27", 27, 17, 43, 100),
        ("a=13, c=0, m=64, X0=1", 1, 13, 0, 64),
        ("a=16807, c=0, m=2^31-1, X0=123457", 123457, 16807, 0, 2**31 - 1)
    ]
    print(f"{'Generator':<35} | {'Chi^2':<10} | {'p-value':<12} | {'Decision'}")
    print("-" * 70)
    for label, seed, a, c, m in configs:
        sample = lcg(seed, a, c, m, 1000)
        chi2, p_val, dec, _ = chi_square_test_10bins(sample)
        print(f"{label:<35} | {chi2:<10.2f} | {p_val:<12.4e} | {dec}")

    print("\n" + "=" * 70)
    print("TASK 4: Reflection & RANDU 3D Hyperplane Defect")
    print("=" * 70)
    print("Does passing Chi-Square test mean the generator is good? NO!")
    print("Example: IBM's RANDU (a = 65539, c = 0, m = 2^31).")
    print("It passes 1-D Chi-Square tests, but consecutive triples fall into planes:")
    print("    9 * X_i - 6 * X_{i+1} + X_{i+2} = 0 (mod 2^31)")

    # Verify RANDU identity
    randu_vals = []
    x = 1
    for _ in range(6):
        x = (65539 * x) % (2**31)
        randu_vals.append(x)
    residual = (9 * randu_vals[0] - 6 * randu_vals[1] + randu_vals[2]) % (2**31)
    print(f"RANDU residual test on actual outputs: {residual} (Exact 0 -> points fall on 15 2D planes in 3D space!)")
