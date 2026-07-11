import numpy as np

f = np.sin
f_prime_exact = np.cos   # exact derivative of sin(x) is cos(x)

# ---------- Truncation error in differentiation ----------
# Forward-difference formula: f'(x) ≈ [f(x+h) - f(x)] / h
# As h -> 0 this approaches the true derivative, but for finite h
# there's a truncation error from the higher-order terms dropped
# in the Taylor expansion of f(x+h).

def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h

x0 = 1.0  # point at which we evaluate the derivative
true_derivative = f_prime_exact(x0)

print("Truncation error in differentiation (forward difference)")
print(f"True derivative f'(x) at x={x0} : {true_derivative:.10f}\n")

for h in [1, 0.1, 0.01, 0.001, 0.0001]:
    approx = forward_difference(f, x0, h)
    error = abs(true_derivative - approx)
    print(f"h = {h:<10} approx = {approx:.10f}   error = {error:.10f}")

# ---------- Truncation error in integration ----------
# Riemann sum: approximate the integral of f(x) over [a, b]
# by summing the areas of n rectangles under the curve.
# As n -> infinity, the sum converges to the true integral,
# but for finite n there's a truncation error.

def riemann_sum(f, a, b, n):
    x = np.linspace(a, b, n, endpoint=False)  # left endpoints of each rectangle
    width = (b - a) / n
    return np.sum(f(x) * width)

a, b = 0, np.pi   # integrate sin(x) from 0 to pi
true_integral = -np.cos(b) + np.cos(a)   # exact value = 2

print("\nTruncation error in integration (Riemann sum)")
print(f"True integral of sin(x) from {a} to pi : {true_integral:.10f}\n")

for n in [5, 10, 50, 100, 1000]:
    approx = riemann_sum(f, a, b, n)
    error = abs(true_integral - approx)
    print(f"n = {n:<6} approx = {approx:.10f}   error = {error:.10f}")