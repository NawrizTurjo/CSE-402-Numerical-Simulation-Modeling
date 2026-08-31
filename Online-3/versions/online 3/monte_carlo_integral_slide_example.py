"""Monte Carlo integration following the slide example."""

import math


def g(x):
    return math.sin(x)


def monte_carlo_integral(random_points, a, b):
    total = 0

    print(" i     Xi      g(Xi)")
    print("---------------------")
    # for val in random_points:
    #     print(f"{random_points.index(val) + 1:2}    {val:.2f}     {g(val):.3f}")
    for i in range(len(random_points)):
        x = random_points[i]
        function_value = g(x)
        total += function_value
        print(f"{i + 1:2}    {x:.2f}     {function_value:.3f}")

    n = len(random_points)
    average_height = total / n
    integral = (b - a) * average_height

    return average_height, integral


# The 10 sample points given in the slide.
x_values = [0.52, 1.19, 2.83, 0.21, 1.77,
            2.50, 0.89, 1.53, 2.97, 2.11]

average, estimate = monte_carlo_integral(x_values, 0, math.pi)
true_value = 2
error_percentage = abs(estimate - true_value) / true_value * 100

print("\nAverage of g(Xi):", round(average, 3))
print("Interval length (b-a):", round(math.pi, 4))
print("Integral estimate:", round(estimate, 3))
print("True value:", true_value)
print("Percentage error:", round(error_percentage, 2), "%")
