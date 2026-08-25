import math


def middle_square(seed, n):
    """Generate n values with the 4-digit middle-square method."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed > 9999:
        raise ValueError("seed must be between 0 and 9999")
    if n < 0:
        raise ValueError("n must be non-negative")

    values = []
    current = seed
    for _ in range(n):
        square_as_8_digits = f"{current * current:08d}"
        current = int(square_as_8_digits[2:6])
        values.append(current)
    return values


def middle_square_uniforms(seed, n):
    """Convert generated 4-digit integers to values in [0, 1)."""
    return [value / 10000 for value in middle_square(seed, n)]


def find_cycle(seed):
    """Return repeated value, repeat iteration, and cycle length."""
    seen_at = {}
    current = seed
    iteration = 0

    while current not in seen_at:
        seen_at[current] = iteration
        current = middle_square(current, 1)[0]
        iteration += 1

    first_seen_iteration = seen_at[current]
    cycle_length = iteration - first_seen_iteration
    return current, iteration, cycle_length


def bin_counts(values, bins=10):
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int(value * bins))
        counts[index] += 1
    return counts


def chi_square_statistic(counts):
    total = sum(counts)
    expected = total / len(counts)
    return sum(((observed - expected) ** 2) / expected for observed in counts)


def chi_square_p_value(chi_square, degrees_of_freedom):
    try:
        from scipy.stats import chi2

        return chi2.sf(chi_square, degrees_of_freedom)
    except ImportError:
        if degrees_of_freedom != 9:
            raise RuntimeError("Install scipy for degrees of freedom other than 9")

        # For this task there are 10 bins, so df = 9. The chi-square upper-tail
        # probability is Q(9/2, chi_square/2), computed exactly for half-integers.
        z = chi_square / 2
        p_value = math.erfc(math.sqrt(z))
        for k in range(4):
            a = k + 0.5
            p_value += (z**a * math.exp(-z)) / math.gamma(a + 1)
        return p_value


def chi_square_uniform_test(seed, n=1000, bins=10, alpha=0.05):
    values = middle_square_uniforms(seed, n)
    counts = bin_counts(values, bins)
    chi_square = chi_square_statistic(counts)
    p_value = chi_square_p_value(chi_square, bins - 1)
    decision = "Reject H0" if p_value < alpha else "Do not reject H0"
    return counts, chi_square, p_value, decision


def print_table(rows):
    print("| Seed | Chi^2 | p-value | Decision |")
    print("|---:|---:|---:|:---|")
    for seed, chi_square, p_value, decision in rows:
        print(f"| {seed} | {chi_square:.4f} | {p_value:.6g} | {decision} |")


def main():
    seed = 5731
    first_100 = middle_square(seed, 100)
    print(f"First 100 values for seed {seed}:")
    print(first_100)
    print()

    problematic_seed = 2500
    repeated_value, repeat_iteration, cycle_length = find_cycle(problematic_seed)
    print("Problematic seed analysis:")
    print(f"Seed: {problematic_seed}")
    print(f"First repeated value: {repeated_value}")
    print(f"Iteration at which it repeats: {repeat_iteration}")
    print(f"Cycle length: {cycle_length}")
    print()

    rows = []
    for test_seed in [5731, 6239, problematic_seed]:
        counts, chi_square, p_value, decision = chi_square_uniform_test(test_seed)
        print(f"Bin counts for seed {test_seed}: {counts}")
        rows.append((test_seed, chi_square, p_value, decision))
    print()
    print_table(rows)

    print()
    print("Reflection:")
    print(
        "Passing the Chi-Square test does not prove that a generator is good. "
        "It only checks whether the one-dimensional bin counts look uniform. "
        "A generator may pass this test but still have short cycles, predictable "
        "patterns, dependence between consecutive values, or poor behavior in "
        "other statistical tests."
    )


if __name__ == "__main__":
    main()
