"""
TEMPLATE: Newton-Raphson Method
--------------------------------
Needs only ONE initial guess (no bracket). Derived from the tangent line
at the current point:

    x_(i+1) = x_i - f(x_i) / f'(x_i)

Algorithm:
  1. Have f(x) and f'(x) (analytically, or approximate f' numerically).
  2. From a guess x_i, compute x_(i+1) = x_i - f(x_i)/f'(x_i).
  3. |eps_a| = |(x_(i+1) - x_i) / x_(i+1)| * 100 %
  4. Repeat from step 2 until |eps_a| <= tol (or max_iter reached).

Converges MUCH faster than bisection/false-position when it converges,
but convergence is NOT guaranteed:
  - f'(x) = 0 (or very small) at some point along the way -> blows up.
  - Starting near an inflection point can cause wild oscillation.
  - A poor guess can "jump" past the intended root to a completely
    different one.
  - If f(x) has no real root nearby, the iteration can wander forever.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "newton_raphson")
os.makedirs(PLOTS_DIR, exist_ok=True)


def numerical_derivative(f, x, h=1e-6):
    """Central-difference derivative, handy when f'(x) is hard to derive by hand."""
    return (f(x + h) - f(x - h)) / (2 * h)


def newton_raphson(f, x0, fprime=None, tol=1e-6, max_iter=100, verbose=False):
    if fprime is None:
        fprime = lambda x: numerical_derivative(f, x)

    table = []
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = fprime(x)
        if dfx == 0:
            raise ZeroDivisionError(f"f'(x) = 0 at x = {x} -- Newton-Raphson breaks down.")
        x_new = x - fx / dfx
        eps_a = abs((x_new - x) / x_new) * 100 if x_new != 0 else float("inf")

        table.append({"iter": i, "x_i": x, "f(x_i)": fx, "x_(i+1)": x_new, "eps_a_%": eps_a})
        if verbose:
            print(table[-1])

        if eps_a < tol:
            x = x_new
            break
        x = x_new

    return x, table


if __name__ == "__main__":
    # quick self-test: f(x) = x^2 - 4, f'(x) = 2x -> root at x = 2
    f = lambda x: x ** 2 - 4
    fprime = lambda x: 2 * x
    root, table = newton_raphson(f, x0=5.0, fprime=fprime, tol=1e-6, verbose=True)
    print(f"\nRoot ~= {root:.6f}  (exact = 2.0)  in {len(table)} iterations")
