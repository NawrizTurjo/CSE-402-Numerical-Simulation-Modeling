"""Easy practice codes for RNG and Monte Carlo methods from the slides."""

import math
from scipy.stats import chi2, kstest


# 1. LINEAR CONGRUENTIAL GENERATOR (LCG)
# Formula: X(next) = (a * X(current) + c) % m
def lcg(seed, a, c, m, n):
    values = []
    x = seed

    for i in range(n):
        x = (a * x + c) % m
        values.append(x / m)

    return values


# Mixed LCG has c != 0.
mixed_values = lcg(seed=7, a=5, c=3, m=16, n=10)
print("Mixed LCG:", mixed_values)

# Multiplicative LCG has c = 0.
multiplicative_values = lcg(seed=7, a=5, c=0, m=16, n=10)
print("Multiplicative LCG:", multiplicative_values)


# 2. FIND THE PERIOD OF AN LCG
def find_lcg_period(seed, a, c, m):
    seen = {}
    x = seed
    iteration = 0

    while x not in seen:
        seen[x] = iteration
        x = (a * x + c) % m
        iteration += 1

    period = iteration - seen[x]
    return x, period


repeated_value, period = find_lcg_period(7, 5, 3, 16)
print("\nRepeated value:", repeated_value)
print("LCG period:", period)


# Generate values for the remaining tests and simulations.
numbers = lcg(seed=123, a=1103515245, c=12345, m=2**31, n=1000)


# 3. CHI-SQUARE UNIFORMITY TEST
def chi_square_test(values):
    counts = [0] * 10

    for value in values:
        bin_number = int(value * 10)
        counts[bin_number] += 1

    expected = len(values) / 10
    chi_square_value = 0

    for observed in counts:
        chi_square_value += (observed - expected) ** 2 / expected

    p_value = chi2.sf(chi_square_value, 9)
    return counts, chi_square_value, p_value


counts, chi_value, chi_p = chi_square_test(numbers)
print("\nChi-Square bin counts:", counts)
print("Chi-Square value:", round(chi_value, 4))
print("Chi-Square p-value:", round(chi_p, 4))
print("Decision:", "Reject H0" if chi_p < 0.05 else "Do not reject H0")


# 4. KOLMOGOROV-SMIRNOV (K-S) TEST
# This compares the generated values with a uniform distribution.
ks_value, ks_p = kstest(numbers, "uniform")
print("\nK-S value:", round(ks_value, 4))
print("K-S p-value:", round(ks_p, 4))
print("Decision:", "Reject H0" if ks_p < 0.05 else "Do not reject H0")


# 5. AUTOCORRELATION TEST
# A value near zero suggests little linear relation between neighbours.
def autocorrelation(values):
    mean = sum(values) / len(values)
    top = 0
    bottom = 0

    for i in range(len(values) - 1):
        top += (values[i] - mean) * (values[i + 1] - mean)

    for value in values:
        bottom += (value - mean) ** 2

    return top / bottom


print("\nAutocorrelation:", round(autocorrelation(numbers), 4))


# 6. MONTE CARLO ESTIMATION OF PI
def estimate_pi(n):
    values = lcg(123, 1103515245, 12345, 2**31, 2 * n)
    inside_circle = 0

    for i in range(n):
        x = values[2 * i]
        y = values[2 * i + 1]

        if x * x + y * y <= 1:
            inside_circle += 1

    return 4 * inside_circle / n


print("\nEstimated pi:", estimate_pi(10000))


# 7. MONTE CARLO INTEGRATION
# Example: integral of x^2 from a to b.
def f(x):
    return x * x


def monte_carlo_integral(a, b, n):
    values = lcg(321, 1103515245, 12345, 2**31, n)
    total = 0

    for value in values:
        x = a + (b - a) * value
        total += f(x)

    return (b - a) * total / n


answer = monte_carlo_integral(0, 1, 10000)
print("Integral of x^2 from 0 to 1:", round(answer, 4))


# 8. INVERSE TRANSFORM METHOD FOR EXPONENTIAL VALUES
# Formula: X = -mean * ln(1 - U)
def exponential_values(mean, n):
    uniform_values = lcg(456, 1103515245, 12345, 2**31, n)
    results = []

    for u in uniform_values:
        x = -mean * math.log(1 - u)
        results.append(x)

    return results


print("\nExponential values:", exponential_values(mean=2, n=5))
