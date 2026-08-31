"""Middle-Square Random Number Generator assignment."""

from scipy.stats import chi2


def next_value(number):
    """Square a number and take the middle four digits."""
    square = number * number
    middle_four_digits = (square // 100) % 10000
    return middle_four_digits


def middle_square(seed, n):
    """Generate n values between 0 and 1 using Middle-Square."""
    values = []
    current = seed

    for i in range(n):
        current = next_value(current)
        values.append(current / 10000)

    return values


def find_repeat(seed):
    """Find the first repeated value, repeat iteration, and cycle length."""
    generated = [seed]
    current = seed
    iteration = 0

    while True:
        current = next_value(current)
        iteration += 1

        if current in generated:
            # Find where this value appeared for the first time.
            for i in range(len(generated)):
                if generated[i] == current:
                    first_position = i
                    break

            cycle_length = iteration - first_position
            return current, iteration, cycle_length

        generated.append(current)


def chi_square_test(seed):
    """Perform a Chi-Square test using 1,000 generated values."""
    values = middle_square(seed, 1000)
    bins = 10
    counts = [0] * bins

    for value in values:
        bin_number = int(value * bins)
        counts[bin_number] += 1

    expected = 100
    chi_square_value = 0

    for observed in counts:
        chi_square_value += (observed - expected) ** 2 / expected

    # 10 bins give 10 - 1 = 9 degrees of freedom.
    p_value = chi2.sf(chi_square_value, df=bins - 1)
    return counts, chi_square_value, p_value


# ------------------------------------------------------------
# Task 1: Generate at least 100 values
# ------------------------------------------------------------

values = middle_square(5731, 100)

print("TASK 1")
print("Generated values:", len(values))
print("First 10 values:", values[:10])


# ------------------------------------------------------------
# Task 2: Investigate a problematic seed
# ------------------------------------------------------------

# 1000 -> 0000 -> 0000 -> 0000, so its cycle length is 1.
problematic_seed = 1000
repeated_value, repeat_iteration, cycle_length = find_repeat(problematic_seed)

print("\nTASK 2")
print("Seed:", problematic_seed)
print("First repeated value:", repeated_value)
print("Iteration at which it repeats:", repeat_iteration)
print("Cycle length:", cycle_length)


# ------------------------------------------------------------
# Task 3: Chi-Square goodness-of-fit test
# ------------------------------------------------------------

print("\nTASK 3")
print(f"{'Seed':<18} {'Chi^2':<12} {'p-value':<12} Decision")
print("-" * 62)

for seed in [5731, 6239, problematic_seed]:
    counts, chi_square_value, p_value = chi_square_test(seed)

    if p_value < 0.05:
        decision = "Reject H0"
    else:
        decision = "Do not reject H0"

    seed_name = str(seed)
    if seed == problematic_seed:
        seed_name = f"{seed} (problematic)"

    print(
        f"{seed_name:<18} {chi_square_value:<12.2f} "
        f"{p_value:<12.6f} {decision}"
    )
    print("Bin counts:", counts)


# ------------------------------------------------------------
# Task 4: Reflection
# ------------------------------------------------------------

print("\nTASK 4")
print(
    "No. Passing the Chi-Square test only suggests that the values are "
    "uniform across the 10 bins. A generator may still have a short cycle "
    "or predictable patterns, so other tests are also needed."
)
