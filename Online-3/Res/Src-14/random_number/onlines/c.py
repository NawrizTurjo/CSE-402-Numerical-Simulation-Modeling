import math

def middle_square(seed, n):
    """
    Middle-Square pseudo-random number generator.
    seed: 4-digit starting integer
    n:    number of values to generate
    Returns raw 4-digit integers and normalized [0,1) floats.
    """
    values = []
    X = seed
    for _ in range(n):
        squared = X ** 2
        # Represent as exactly 8 digits (pad with leading zeros if needed)
        s = str(squared).zfill(8)
        # Extract middle 4 digits: positions 2,3,4,5 of the 8-char string
        X = int(s[2:6])
        values.append(X)
    # Normalize to [0,1) by dividing by 10000
    normalized = [v / 10000 for v in values]
    return values, normalized

def find_cycle(seed, max_iter=10000):
    """Detect when sequence first repeats."""
    seen = {seed: 0}
    X = seed
    
    for i in range(1, max_iter + 1):
        squared = X ** 2
        s = str(squared).zfill(8)
        X = int(s[2:6])
        if X in seen:
            return X, i, i - seen[X]
        seen[X] = i
    return None, None, None

def chi_square_p_value(chi2, df):
    """Upper-tail p-value. For this task df = 9 because there are 10 bins."""
    if df != 9:
        raise ValueError("This fallback p-value function is for df = 9 only")

    z = chi2 / 2
    p = math.erfc(math.sqrt(z))
    for k in range(4):
        a = k + 0.5
        p += (z**a * math.exp(-z)) / math.gamma(a + 1)
    return p

def chi_square_test(seed, n=1000, bins=10):
    """
    Chi-square uniformity test — no library functions.
    Decision rule: reject H0 if p < 0.05.
    """
    _, normalized = middle_square(seed, n)

    # Step 1: Count observed values per bin
    E = n / bins        # expected per bin (uniform assumption)
    counts = [0] * bins
    for r in normalized:
        idx = min(bins - 1, int(r * bins))
        counts[idx] += 1

    # Step 2: Compute chi-square statistic
    # formula: chi2 = sum of (Oi - E)^2 / E  for each bin
    chi2 = 0
    for i in range(bins):
        chi2 += ((counts[i] - E) ** 2) / E

    # Step 3: Calculate p-value and make decision
    p = chi_square_p_value(chi2, bins - 1)
    decision = "Reject H0" if p < 0.05 else "Fail to Reject H0"

    return chi2, p, decision, counts


def print_chi_square_table(rows):
    print("| Seed | Chi^2 | p-value | Decision |")
    print("|---:|---:|---:|:---|")
    for seed, chi2, p, decision in rows:
        print(f"| {seed} | {chi2:.4f} | {p:.6g} | {decision} |")


def main():
    # Task 1
    seed = 5731
    values, _ = middle_square(seed, 100)
    print(f"Task 1: First 100 values for seed {seed}")
    print(values)
    print()

    # Task 2
    problematic_seed = 2500
    repeated_value, repeat_iteration, cycle_length = find_cycle(problematic_seed)
    print("Task 2: Problematic seed")
    print(f"Seed: {problematic_seed}")
    print(f"First repeated value: {repeated_value}")
    print(f"Iteration at which it repeats: {repeat_iteration}")
    print(f"Cycle length: {cycle_length}")
    print()

    # Task 3
    print("Task 3: Chi-Square uniformity test")
    rows = []
    for test_seed in [5731, 6239, problematic_seed]:
        chi2, p, decision, counts = chi_square_test(test_seed)
        print(f"Bin counts for seed {test_seed}: {counts}")
        rows.append((test_seed, chi2, p, decision))
    print()
    print_chi_square_table(rows)
    print()

    # Task 4
    print("Task 4: Reflection")
    print(
        "Passing the Chi-Square test does not necessarily mean that the generator "
        "is a good random-number generator. The test only checks whether the "
        "counts in the bins look uniform. A generator can pass this test but "
        "still have short cycles, predictable patterns, or dependence between "
        "successive values."
    )


if __name__ == "__main__":
    main()
