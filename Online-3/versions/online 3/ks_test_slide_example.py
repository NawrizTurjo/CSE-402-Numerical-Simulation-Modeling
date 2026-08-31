"""Manual Kolmogorov-Smirnov test following the slide example."""

import math


def ks_test(values, alpha):
    # Step 1: Sort the random numbers.
    sorted_values = sorted(values)
    n = len(sorted_values)

    d_plus_values = []
    d_minus_values = []

    print("Original values:", values)
    print("Sorted values:", sorted_values)
    print()

    print(" i    R(i)    i/N    D+       (i-1)/N    D-")
    print("------------------------------------------------")

    # Steps 2 and 3: Calculate D+ and D- for every value.
    for i in range(1, n + 1):
        r = sorted_values[i - 1]

        i_by_n = i / n
        previous_by_n = (i - 1) / n

        d_plus = i_by_n - r
        d_minus = r - previous_by_n

        d_plus_values.append(d_plus)
        d_minus_values.append(d_minus)

        print(
            f"{i:2}   {r:5.2f}   {i_by_n:4.2f}   {d_plus:6.2f}   "
            f"{previous_by_n:7.2f}   {d_minus:6.2f}"
        )

    # Step 4: Take the largest D+ and D-.
    d_plus_maximum = max(d_plus_values)
    d_minus_maximum = max(d_minus_values)
    d = max(d_plus_maximum, d_minus_maximum)

    print("\nD+ =", round(d_plus_maximum, 2))
    print("D- =", round(d_minus_maximum, 2))
    print("D  = max(D+, D-) =", round(d, 2))

    # Step 5: Calculate the approximate K-S critical value.
    # This formula works for different values of alpha.
    constant = math.sqrt(-0.5 * math.log(alpha / 2))
    critical_value = constant / math.sqrt(n)

    # Step 6: Compare D with the critical value.
    print("Alpha =", alpha)
    print("Critical value =", round(critical_value, 4))

    if d > critical_value:
        print("Decision: Reject H0")
        print("The values are not uniformly distributed.")
    else:
        print("Decision: Do not reject H0")
        print("There is no significant evidence of non-uniformity.")

    return d_plus_maximum, d_minus_maximum, d


# Data from the slide: N = 5 and alpha = 0.05.
random_values = [0.44, 0.81, 0.14, 0.05, 0.93]

alpha = 0.05

print("KOLMOGOROV-SMIRNOV TEST")
print("H0: The numbers are uniformly distributed over [0, 1).")
print("N = 5, alpha = 0.05\n")

ks_test(random_values, alpha)
