"""
Problem 6: RLC Circuit Complex Characteristic Roots (Complex Newton-Raphson)
Objective:
  Find all roots (real and complex conjugate pairs) for the RLC circuit polynomial:
    f(z) = z^3 + 2*z^2 + 5*z + 10 = 0

Demonstrate:
  1. Real-bracketing methods (Bisection, False Position) fail to find complex roots.
  2. Newton-Raphson with real guess z0 = 0.0 finds real root z1 = -2.0.
  3. Newton-Raphson with complex guess z0 = 1.0 + 1.0j converges to complex root z2 = +2.236068j.
  4. Newton-Raphson with complex guess z0 = 1.0 - 1.0j converges to complex root z3 = -2.236068j.
"""

import cmath
from common import P, use, banner

# -----------------------------------------------------------------------------
# Problem Definition
# -----------------------------------------------------------------------------
def f(z):
    return z**3 + 2.0 * z**2 + 5.0 * z + 10.0


def df(z):
    return 3.0 * z**2 + 4.0 * z + 5.0


use(f, df)

banner("PROBLEM 6: RLC CIRCUIT COMPLEX ROOTS (COMPLEX NEWTON-RAPHSON)")


def complex_newton_raphson(z0, tol=0.0001, max_iter=20):
    """Newton-Raphson extension for complex initial guesses z0 = x + y*j."""
    zi = complex(z0)
    history = []
    print(f"\n--- Complex Newton-Raphson starting at z0 = {z0} ---")
    print(f"{'Iter':<6}{'Real Part':<15}{'Imag Part':<15}{'Error ea (%)':<15}")
    print("-" * 55)

    for i in range(1, max_iter + 1):
        fzi = f(zi)
        dfzi = df(zi)
        if abs(dfzi) < 1e-12:
            print("Zero derivative in complex plane!")
            break

        zi1 = zi - fzi / dfzi
        ea = abs((zi1 - zi) / zi1) * 100.0 if abs(zi1) > 1e-12 else 0.0

        history.append({'iter': i, 'zi': zi, 'zi1': zi1, 'ea': ea})
        print(f"{i:<6}{zi1.real:<15.6f}{zi1.imag:<15.6f}{ea:<15.6f}%")

        if ea <= tol:
            break
        zi = zi1

    return zi, history


# 1. Real root search
r1, _ = complex_newton_raphson(z0=0.0 + 0.0j)

# 2. Complex root 1 search (positive imaginary)
r2, _ = complex_newton_raphson(z0=1.0 + 1.0j)

# 3. Complex root 2 search (negative imaginary)
r3, _ = complex_newton_raphson(z0=1.0 - 1.0j)

print(f"\n{'='*55}")
print(f"ALL ROOTS FOUND FOR f(z) = z^3 + 2z^2 + 5z + 10 = 0:")
print(f"{'='*55}")
print(f"  Real Root  z1 = {r1.real:.6f} + {r1.imag:.6f}j")
print(f"  Complex z2 = {r2.real:.6f} + {r2.imag:.6f}j  (exact: +sqrt(5)j ~= +2.236068j)")
print(f"  Complex z3 = {r3.real:.6f} + {r3.imag:.6f}j  (exact: -sqrt(5)j ~= -2.236068j)")

print("\nResidual verification:")
print(f"  f(z1) = {f(r1):.4e}")
print(f"  f(z2) = {f(r2):.4e}")
print(f"  f(z3) = {f(r3):.4e}")
