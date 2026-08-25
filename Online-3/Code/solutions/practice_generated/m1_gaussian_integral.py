"""
Practice M1: Estimating integral_0^1 e^(-x^2) dx (see ../../../Practice/PRACTICE_QUESTIONS.md).
Same shape as monte_carlo/integration.py's own sin(x) example -- a
function with no elementary closed-form antiderivative, which is
precisely when Monte Carlo integration earns its keep.

Run standalone:
    python -m solutions.practice_generated.m1_gaussian_integral
"""

import math

from monte_carlo.integration import integrate_1d

EXACT = math.sqrt(math.pi) / 2 * math.erf(1)  # closed form via the error function, for checking only


def f(x):
    return math.exp(-x * x)


def task1_estimate():
    result = integrate_1d(f, 0, 1, n=100000, seed=1)
    print("TASK 1: sample-mean Monte Carlo estimate")
    print(f"  estimate={result['estimate']:.5f}  std_error={result['std_error']:.5f}  "
          f"exact={EXACT:.5f}")
    print()


def task2_convergence():
    print("TASK 2: convergence as n grows")
    for n in (100, 1000, 10000, 100000):
        result = integrate_1d(f, 0, 1, n=n, seed=1)
        error = abs(result["estimate"] - EXACT)
        print(f"  n={n:>7,}  estimate={result['estimate']:.5f}  "
              f"error={error:.5f}  std_error={result['std_error']:.5f}")
    print("  std_error shrinks by ~1/sqrt(10) each time n grows 10x, matching")
    print("  the theoretical 1/sqrt(n) Monte Carlo convergence rate.")


if __name__ == "__main__":
    task1_estimate()
    task2_convergence()
