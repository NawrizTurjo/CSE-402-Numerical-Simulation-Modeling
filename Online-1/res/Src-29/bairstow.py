import numpy as np

# ---------- Bairstow's Method ----------
# Finds the roots of a polynomial by extracting quadratic factors
# (x^2 - u*x - v) one at a time, which lets it find complex root pairs
# using only real arithmetic. It refines guesses for u and v using
# synthetic division (twice) and a 2x2 Newton step, so like Newton-Raphson,
# the number of iterations is the tunable truncation parameter.

def synthetic_division_twice(coeffs, u, v):
    """
    Divide the polynomial (given by coeffs, highest degree first) by
    (x^2 - u*x - v) twice. Returns b (quotient/remainder coeffs from the
    first division) and c (from dividing b by the same quadratic again,
    needed for the Newton step).
    """
    n = len(coeffs) - 1
    b = np.zeros(n + 1)
    b[0] = coeffs[0]
    b[1] = coeffs[1] + u * b[0]
    for i in range(2, n + 1):
        b[i] = coeffs[i] + u * b[i - 1] + v * b[i - 2]

    c = np.zeros(n + 1)
    c[0] = b[0]
    c[1] = b[1] + u * c[0]
    for i in range(2, n):
        c[i] = b[i] + u * c[i - 1] + v * c[i - 2]

    return b, c


def bairstow_quadratic(coeffs, u0, v0, n_iterations):
    """Refine one quadratic factor x^2 - u*x - v of the polynomial."""
    u, v = u0, v0
    history = [(u, v)]

    n = len(coeffs) - 1

    for _ in range(n_iterations):
        b, c = synthetic_division_twice(coeffs, u, v)

        r, s = b[n - 1], b[n]          # remainder terms (want these -> 0)

        # Solve the 2x2 linear system for the Newton update (du, dv):
        # [c[n-2]   c[n-3]] [du]   [-r]
        # [c[n-1]   c[n-2]] [dv] = [-s]
        J = np.array([[c[n - 2], c[n - 3]],
                       [c[n - 1], c[n - 2]]])
        rhs = np.array([-r, -s])

        du, dv = np.linalg.solve(J, rhs)
        u += du
        v += dv
        history.append((u, v))

    b, _ = synthetic_division_twice(coeffs, u, v)
    deflated_coeffs = b[:n - 1]  # quotient polynomial (drop the two remainder terms)
    return u, v, deflated_coeffs, history


def bairstow_all_roots(coeffs, u0=0.0, v0=1.0, n_iterations=20):
    """Repeatedly extract quadratic factors until the polynomial is fully reduced."""
    coeffs = np.array(coeffs, dtype=float)
    all_roots = []

    while len(coeffs) - 1 > 2:
        u, v, coeffs, _ = bairstow_quadratic(coeffs, u0, v0, n_iterations)
        roots = np.roots([1, -u, -v])   # roots of x^2 - u*x - v = 0
        all_roots.extend(roots)

    if len(coeffs) - 1 == 2:
        all_roots.extend(np.roots(coeffs))
    elif len(coeffs) - 1 == 1:
        all_roots.append(-coeffs[1] / coeffs[0])

    return all_roots


# ---------- Example: x^4 - 10x^3 + 35x^2 - 50x + 24 ----------
# True roots are 1, 2, 3, 4 (a nice clean case to check against)
coeffs = [1, -10, 35, -50, 24]
true_roots = sorted([1, 2, 3, 4])

print("Convergence of (u, v) for the first quadratic factor:")
u, v, deflated, history = bairstow_quadratic(coeffs, u0=0.0, v0=1.0, n_iterations=6)
for i, (ui, vi) in enumerate(history):
    print(f"iteration {i:<2} u = {ui:.10f}   v = {vi:.10f}")

print(f"\nExtracted quadratic factor: x^2 - ({u:.5f})x - ({v:.5f})")
print(f"Roots of this factor: {np.roots([1, -u, -v])}\n")

print("Full root-finding across several iteration counts:")
for n_iterations in [1, 3, 6, 10]:
    roots = bairstow_all_roots(coeffs, u0=0.0, v0=1.0, n_iterations=n_iterations)
    roots_sorted = sorted(np.real(roots))
    error = np.max(np.abs(np.array(roots_sorted) - np.array(true_roots)))
    print(f"n_iterations = {n_iterations:<3} roots = {np.round(roots_sorted, 6)}   max error = {error:.2e}")