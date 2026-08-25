"""
Monte Carlo Integration using Random Numbers sampled from U(0,1)

Idea:
    To estimate  I = integral of f(x) dx  over [a, b]:

        I = (b - a) * E[ f(X) ],   where X ~ Uniform(a, b)

    We approximate the expectation E[f(X)] by the sample mean of f
    evaluated at n random points drawn from U(a, b):

        I_hat = (b - a) * (1/n) * sum_{i=1}^{n} f(X_i)

    where X_i = a + (b - a) * R_i  and  R_i ~ Uniform(0, 1)

    As a bonus, we also report the standard error of the estimate and
    an approximate 95% confidence interval, since Monte Carlo estimates
    are random and their precision depends on the sample variance of
    f(X) and the number of samples n.
"""

import random
import math

# ---------------------------------------------------------------------
# INPUT: the function to integrate, the interval [a, b], and sample size
# ---------------------------------------------------------------------
def f(x):
    return x ** 2          # example: integrate f(x) = x^2 over [0, 1] -> exact = 1/3

a = 0.0        # lower limit of integration
b = 1.0        # upper limit of integration
n = 10000      # number of random samples

random.seed(42)   # fixed seed so results are reproducible (remove for true randomness)

# ---------------------------------------------------------------------
# GENERATE SAMPLES FROM U(0,1) AND MAP TO U(a,b)
# ---------------------------------------------------------------------
R = [random.random() for _ in range(n)]      # R_i ~ Uniform(0,1)
X = [a + (b - a) * r for r in R]             # X_i ~ Uniform(a,b)
f_values = [f(x) for x in X]                 # f(X_i)

# ---------------------------------------------------------------------
# MONTE CARLO ESTIMATE OF THE INTEGRAL
# ---------------------------------------------------------------------
mean_f = sum(f_values) / n
I_hat = (b - a) * mean_f

# ---------------------------------------------------------------------
# SAMPLE VARIANCE, STANDARD ERROR, AND 95% CONFIDENCE INTERVAL
# ---------------------------------------------------------------------
variance_f = sum((fx - mean_f) ** 2 for fx in f_values) / (n - 1)
std_error_I = (b - a) * math.sqrt(variance_f / n)

z_975 = 1.96   # standard normal critical value for a 95% CI
ci_lower = I_hat - z_975 * std_error_I
ci_upper = I_hat + z_975 * std_error_I

# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------
print(f"Integrating f(x) over [{a}, {b}] using {n} random samples from U(0,1)")
print("-" * 60)
print(f"Sample mean of f(X)      : {mean_f:.6f}")
print(f"Estimated integral I_hat : {I_hat:.6f}")
print(f"Sample variance of f(X)  : {variance_f:.6f}")
print(f"Standard error of I_hat  : {std_error_I:.6f}")
print(f"95% Confidence Interval  : [{ci_lower:.6f}, {ci_upper:.6f}]")

# For reference, print a few of the sampled points and their f values
print("\nFirst 5 sampled points:")
print(f"{'i':>3} | {'R_i':>8} | {'X_i':>8} | {'f(X_i)':>8}")
print("-" * 36)
for i in range(5):
    print(f"{i + 1:3d} | {R[i]:8.4f} | {X[i]:8.4f} | {f_values[i]:8.4f}")