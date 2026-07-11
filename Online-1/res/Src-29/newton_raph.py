import numpy as np

# ---------- Newton-Raphson Method ----------
# Approximates a root of f(x) = 0 using the update rule:
#   x_{n+1} = x_n - f(x_n) / f'(x_n)
# Geometrically, this follows the tangent line at x_n down to the x-axis
# and uses that crossing point as the next guess.
# Unlike bisection, it doesn't need a bracketing interval, and it
# typically converges much faster (quadratically, near the root).

def newton_raphson(f, f_prime, x0, n_iterations):
    x = x0
    history = [x0]

    for i in range(n_iterations):
        x = x - f(x) / f_prime(x)
        history.append(x)

    return x, history

# ---------- Example: find root of f(x) = x^2 - 2  (true root = sqrt(2)) ----------
f = lambda x: x**2 - 2
f_prime = lambda x: 2 * x
true_root = np.sqrt(2)

x0 = 1.0   # initial guess

print(f"True root (sqrt(2)) : {true_root:.15f}\n")

for n_iterations in [1, 2, 3, 4, 5]:
    approx_root, history = newton_raphson(f, f_prime, x0, n_iterations)
    error = abs(true_root - approx_root)
    print(f"n_iterations = {n_iterations:<3} approx = {approx_root:.15f}   error = {error:.15f}")

# ---------- Step-by-step convergence ----------
print("\nStep-by-step convergence (starting guess x0 = 1.0):")
_, history = newton_raphson(f, f_prime, x0, 5)
for i, x_i in enumerate(history):
    error = abs(true_root - x_i)
    print(f"iteration {i:<2} x = {x_i:.15f}   error = {error:.15f}")