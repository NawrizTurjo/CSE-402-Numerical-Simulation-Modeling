"""
Python random module cheat sheet + comparison with own RNG implementations.

Run:
    python3 random_module_cheatsheet.py
"""

import math
import random
import statistics


def lcg(seed, a, c, m, n):
    """Linear congruential generator: X_n = (a X_{n-1} + c) mod m."""
    values = []
    x = seed
    for _ in range(n):
        x = (a * x + c) % m
        values.append(x / m)
    return values


def middle_square(seed, n):
    """4-digit middle-square generator, returned as U[0, 1) values."""
    values = []
    x = seed
    for _ in range(n):
        x = int(f"{x * x:08d}"[2:6])
        values.append(x / 10000)
    return values


def python_random_uniforms(seed, n):
    random.seed(seed)
    return [random.random() for _ in range(n)]


def bin_counts(values, bins=10):
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int(value * bins))
        counts[index] += 1
    return counts


def chi_square_statistic(counts):
    expected = sum(counts) / len(counts)
    return sum(((observed - expected) ** 2) / expected for observed in counts)


def chi_square_p_value_df9(chi_square):
    """Upper-tail p-value for df=9, enough for 10-bin uniformity tests."""
    z = chi_square / 2
    p_value = math.erfc(math.sqrt(z))
    for k in range(4):
        a = k + 0.5
        p_value += (z**a * math.exp(-z)) / math.gamma(a + 1)
    return p_value


def chi_square_test_uniform(values, bins=10, alpha=0.05):
    counts = bin_counts(values, bins)
    chi_square = chi_square_statistic(counts)

    try:
        from scipy.stats import chi2

        p_value = chi2.sf(chi_square, bins - 1)
    except ImportError:
        if bins != 10:
            raise RuntimeError("Install scipy for non-10-bin p-values")
        p_value = chi_square_p_value_df9(chi_square)

    decision = "Reject H0" if p_value < alpha else "Do not reject H0"
    return counts, chi_square, p_value, decision


def inverse_transform_exponential(uniform_values, mean):
    """If U is uniform on [0, 1), then X = -mean * ln(1-U)."""
    return [-mean * math.log(1 - u) for u in uniform_values]


def manual_mean(values):
    return sum(values) / len(values)


def manual_population_variance(values):
    mean = manual_mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def demo_random_module_basics():
    random.seed(42)

    print("Python random module basics")
    print("random.random()              ->", random.random())
    print("random.uniform(5, 10)        ->", random.uniform(5, 10))
    print("random.randint(1, 6)         ->", random.randint(1, 6), "(inclusive)")
    print("random.randrange(1, 7)       ->", random.randrange(1, 7), "(7 excluded)")
    print("random.choice(['H', 'T'])    ->", random.choice(["H", "T"]))
    print("random.sample(range(10), 3)  ->", random.sample(range(10), 3))

    items = [1, 2, 3, 4, 5]
    random.shuffle(items)
    print("random.shuffle([1..5])       ->", items)

    print("random.expovariate(1/2)      ->", random.expovariate(1 / 2))
    print("random.gauss(0, 1)           ->", random.gauss(0, 1))


def demo_reproducibility():
    print("\nReproducibility with seed")
    random.seed(7)
    first_run = [random.random() for _ in range(5)]

    random.seed(7)
    second_run = [random.random() for _ in range(5)]

    print("First run :", first_run)
    print("Second run:", second_run)
    print("Same result after same seed:", first_run == second_run)


def demo_manual_vs_library_statistics():
    values = python_random_uniforms(seed=123, n=1000)

    print("\nManual calculations vs Python statistics library")
    print("Manual mean              :", manual_mean(values))
    print("statistics.mean         :", statistics.mean(values))
    print("Manual population var    :", manual_population_variance(values))
    print("statistics.pvariance    :", statistics.pvariance(values))


def demo_compare_generators():
    n = 1000
    generators = [
        ("Python random", python_random_uniforms(seed=5731, n=n)),
        ("LCG", lcg(seed=27, a=17, c=43, m=100, n=n)),
        ("Middle-square 5731", middle_square(seed=5731, n=n)),
        ("Middle-square 6239", middle_square(seed=6239, n=n)),
        ("Middle-square 2500", middle_square(seed=2500, n=n)),
    ]

    print("\n10-bin Chi-Square comparison")
    print("| Generator | Chi^2 | p-value | Decision |")
    print("|:---|---:|---:|:---|")
    for name, values in generators:
        _, chi_square, p_value, decision = chi_square_test_uniform(values)
        print(f"| {name} | {chi_square:.4f} | {p_value:.6g} | {decision} |")


def demo_transform_random_numbers():
    uniforms = python_random_uniforms(seed=99, n=5)
    exponential = inverse_transform_exponential(uniforms, mean=2)
    uniform_5_to_10 = [5 + (10 - 5) * u for u in uniforms]

    print("\nTransforming U[0, 1) values")
    print("Uniforms:", uniforms)
    print("Uniform[5, 10]:", uniform_5_to_10)
    print("Exponential mean 2:", exponential)


if __name__ == "__main__":
    demo_random_module_basics()
    demo_reproducibility()
    demo_manual_vs_library_statistics()
    demo_compare_generators()
    demo_transform_random_numbers()
