import numpy as np

# ---------- Bisection Method ----------
# Approximates a root of f(x) = 0 by repeatedly halving an interval [a, b]
# known to contain a root (i.e. f(a) and f(b) have opposite signs).
# The number of iterations controls how "truncated" the approximation is:
# more iterations -> interval shrinks further -> closer to the true root.

def bisection(f, a, b, n_iterations):
    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    history = []  # keep track of the midpoint at each iteration

    for i in range(n_iterations):
        midpoint = (a + b) / 2
        history.append(midpoint)

        if f(a) * f(midpoint) < 0:
            b = midpoint   # root lies in [a, midpoint]
        else:
            a = midpoint   # root lies in [midpoint, b]

    return midpoint, history

# ---------- Example: find root of f(x) = x^2 - 2  (true root = sqrt(2)) ----------
f = lambda x: x**2 - 2
true_root = np.sqrt(2)

a, b = 0, 2   # initial bracket, f(0) = -2, f(2) = 2 -> opposite signs

print(f"True root (sqrt(2)) : {true_root:.10f}\n")

for n_iterations in [1, 5, 10, 20, 40]:
    approx_root, history = bisection(f, a, b, n_iterations)
    error = abs(true_root - approx_root)
    print(f"n_iterations = {n_iterations:<3} approx = {approx_root:.10f}   error = {error:.10f}")

# ---------- Show how the interval and error shrink step by step ----------
print("\nStep-by-step convergence for n_iterations = 10:")
_, history = bisection(f, a, b, 10)
for i, midpoint in enumerate(history, start=1):
    error = abs(true_root - midpoint)
    print(f"iteration {i:<2} midpoint = {midpoint:.10f}   error = {error:.10f}")