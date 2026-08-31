"""Autocorrelation test with M calculation, following the slide."""

import math


def autocorrelation_test(numbers, starting_position, lag):
    n = len(numbers)

    # M is the largest integer satisfying i + (M + 1) * lag <= N.
    m = (n - starting_position) // lag - 1

    print("N =", n)
    print("Starting position i =", starting_position)
    print("Lag =", lag)
    print("M =", m)
    print(
        "Check:", starting_position, "+ (", m, "+ 1 ) *",
        lag, "<=", n
    )

    # Select Ri, Ri+lag, ..., Ri+(M+1)lag.
    selected = []

    for k in range(m + 2):
        position = starting_position + k * lag
        selected.append(numbers[position - 1])  # -1 because Python starts at 0

    print("Selected subsequence:", selected)

    # Multiply each neighboring pair in the selected subsequence.
    product_sum = 0

    print("\nProducts:")
    for k in range(m + 1):
        product = selected[k] * selected[k + 1]
        product_sum += product
        print(selected[k], "*", selected[k + 1], "=", round(product, 4))

    average_product = product_sum / (m + 1)
    rho = average_product - 0.25

    # Standard deviation from the slide.
    standard_deviation = math.sqrt(13 * m + 7) / (12 * (m + 1))
    z_value = rho / standard_deviation

    print("\nSum of products:", round(product_sum, 4))
    print("Average product:", round(average_product, 4))
    print("Estimated autocorrelation:", round(rho, 4))
    print("Standard deviation:", round(standard_deviation, 4))
    print("Z0:", round(z_value, 2))

    # For alpha = 0.05, this is a two-tailed test.
    if abs(z_value) > 1.96:
        print("Decision: Reject H0")
        print("There is evidence of autocorrelation.")
    else:
        print("Decision: Do not reject H0")
        print("There is no evidence of autocorrelation.")


# N = 30. The slide tests R3, R8, R13, R18, R23, and R28.
random_numbers = [
    0.12, 0.45, 0.23, 0.67, 0.91,
    0.14, 0.52, 0.28, 0.74, 0.19,
    0.62, 0.41, 0.33, 0.87, 0.16,
    0.49, 0.71, 0.27, 0.38, 0.82,
    0.11, 0.58, 0.05, 0.94, 0.31,
    0.76, 0.22, 0.36, 0.69, 0.43
]

autocorrelation_test(random_numbers, starting_position=3, lag=5)
