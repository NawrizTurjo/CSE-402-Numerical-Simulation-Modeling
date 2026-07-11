import numpy as np
import math

# ---------- Round-off error ----------
# Arises from representing a number with a finite number of digits.
x = 1/3
x_rounded = np.round(x, 5)          # round 1/3 to 5 decimal digits
roundoff_error = abs(x - x_rounded)

print("Round-off error demo")
print(f"True value (1/3)        : {x}")
print(f"Rounded to 5 digits      : {x_rounded}")
print(f"Round-off error          : {roundoff_error:.10f}\n")

# ---------- Truncation error ----------
# Arises from approximating an infinite process (like a Taylor series)
# with a finite number of terms.
def exp_taylor(x, n_terms):
    """Approximate e^x using n_terms of its Taylor series."""
    n = np.arange(n_terms)
    terms = x**n / np.array([math.factorial(k) for k in n])
    return np.sum(terms)

x_val = 2.0
n_terms = 5
approx_exp = exp_taylor(x_val, n_terms)
true_exp = np.exp(x_val)
truncation_error = abs(true_exp - approx_exp)

print("Truncation error demo")
print(f"True value  (e^{x_val})  : {true_exp}")
print(f"Taylor approx ({n_terms} terms): {approx_exp}")
print(f"Truncation error         : {truncation_error:.10f}")