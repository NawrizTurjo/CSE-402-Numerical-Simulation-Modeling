"""
TEMPLATE: Bairstow's Method
----------------------------
Bairstow's method finds ALL roots (real and complex) of a polynomial by
repeatedly extracting a quadratic factor  x^2 - r*x - s  using a
2-variable version of Newton-Raphson, then "deflating" (dividing it out)
and repeating on the smaller polynomial.

Coefficients are stored HIGHEST degree first, e.g. for
    a0*x^n + a1*x^(n-1) + ... + a_(n-1)*x + a_n
use coeffs = [a0, a1, ..., a_n]   (same convention as numpy.roots / numpy.polyval)

--- The algorithm for ONE quadratic factor ---
Given a guess (r, s), synthetic-divide the polynomial by (x^2 - r*x - s):
    b0 = a0
    b1 = a1 + r*b0
    bi = ai + r*b_(i-1) + s*b_(i-2)      for i = 2..n
The remainder is b_(n-1)*x + b_n; we want both to be (numerically) zero.

Divide the b's by (x^2 - r*x - s) AGAIN to get the c's (needed for the
partial derivatives of b_(n-1), b_n with respect to r and s):
    c0 = b0
    c1 = b1 + r*c0
    ci = bi + r*c_(i-1) + s*c_(i-2)      for i = 2..n-1

Then solve the 2x2 linear system for the correction (dr, ds):
    c_(n-2)*dr + c_(n-3)*ds = -b_(n-1)
    c_(n-1)*dr + c_(n-2)*ds = -b_n

Update r += dr, s += ds, and repeat until (dr, ds) are tiny.

Once (r, s) has converged, the quadratic factor x^2 - r*x - s gives two
roots via the quadratic formula:
    x = ( r +- sqrt(r^2 + 4s) ) / 2
(complex if r^2 + 4s < 0).

Deflate: the quotient's coefficients are b[0 .. n-2] (drop the last two),
and repeat on that smaller polynomial. Once the remaining polynomial has
degree 2, solve it directly with the quadratic formula; degree 1, solve
directly; degree 0, done.
"""

import os
import cmath
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "bairstow")
os.makedirs(PLOTS_DIR, exist_ok=True)


def bairstow_quadratic_factor(a, r0=-1.0, s0=-1.0, tol=1e-8, max_iter=200):
    """
    Find one quadratic factor x^2 - r*x - s of the polynomial with
    coefficients `a` (highest degree first, len(a) = n+1, degree n >= 3).
    Returns (r, s, deflated_coeffs).
    """
    n = len(a) - 1
    r, s = r0, s0

    for _ in range(max_iter):
        b = [0.0] * (n + 1)
        b[0] = a[0]
        b[1] = a[1] + r * b[0]
        for i in range(2, n + 1):
            b[i] = a[i] + r * b[i - 1] + s * b[i - 2]

        c = [0.0] * (n + 1)
        c[0] = b[0]
        c[1] = b[1] + r * c[0]
        for i in range(2, n):
            c[i] = b[i] + r * c[i - 1] + s * c[i - 2]

        A = np.array([[c[n - 2], c[n - 3]],
                      [c[n - 1], c[n - 2]]])
        rhs = np.array([-b[n - 1], -b[n]])
        dr, ds = np.linalg.solve(A, rhs)

        r += dr
        s += ds

        if abs(dr) < tol and abs(ds) < tol:
            break

    # final deflation with the converged (r, s)
    b = [0.0] * (n + 1)
    b[0] = a[0]
    b[1] = a[1] + r * b[0]
    for i in range(2, n + 1):
        b[i] = a[i] + r * b[i - 1] + s * b[i - 2]

    deflated = b[0:n - 1]   # quotient coefficients (degree n-2)
    return r, s, deflated


def quadratic_roots(r, s):
    """Roots of x^2 - r*x - s = 0 (may be complex)."""
    disc = r ** 2 + 4 * s
    sq = cmath.sqrt(disc)
    return (r + sq) / 2, (r - sq) / 2


def bairstow_all_roots(coeffs, tol=1e-8):
    """
    Find ALL roots (real and complex) of the polynomial with coefficients
    `coeffs` (highest degree first) using repeated Bairstow deflation.
    """
    a = [float(c) for c in coeffs]
    roots = []

    while len(a) - 1 > 2:
        r, s, a = bairstow_quadratic_factor(a, tol=tol)
        x1, x2 = quadratic_roots(r, s)
        roots.extend([x1, x2])

    if len(a) - 1 == 2:
        r = -a[1] / a[0]
        s = -a[2] / a[0]
        x1, x2 = quadratic_roots(r, s)
        roots.extend([x1, x2])
    elif len(a) - 1 == 1:
        roots.append(-a[1] / a[0])

    return roots


if __name__ == "__main__":
    # quick self-test: x^3 - 6x^2 + 11x - 6 -> roots 1, 2, 3
    coeffs = [1, -6, 11, -6]
    roots = bairstow_all_roots(coeffs)
    print("Bairstow roots:", roots)
    print("numpy.roots    :", np.roots(coeffs))
