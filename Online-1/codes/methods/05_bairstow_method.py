"""
=============================================================================
  TOPIC 5: Bairstow's Method - Finding ALL Roots of a Polynomial
           (including complex-conjugate root pairs)
=============================================================================

WHY BAIRSTOW'S METHOD:
  - Bisection / False Position / Newton-Raphson only find REAL roots one at a time.
  - Bairstow's extracts QUADRATIC factors (x^2 - r*x - s) using real arithmetic,
    which automatically handles complex conjugate root pairs.
  - Works entirely in real numbers even when roots are complex!

ALGORITHM:
  Given polynomial P(x) with coefficients a[0..n] (highest power first):

  STEP 1: Initial guess r0, s0 for the quadratic factor x^2 - r*x - s

  STEP 2: Synthetic division TWICE
    First pass (b-array):
      b0 = a0
      b1 = a1 + r*b0
      b_i = a_i + r*b_i-1 + s*b_i-2   for i = 2..n

    Second pass (c-array, divide b by same quadratic):
      c0 = b0
      c1 = b1 + r*c0
      c_i = b_i + r*c_i-1 + s*c_i-2   for i = 2..n-1   <- only up to n-1!

  STEP 3: Solve 2x2 system for corrections (dr, ds):
    determinant D = c[n-2]^2 - c[n-3]*c[n-1]
    dr = (-b[n-1]*c[n-2] + b[n]*c[n-3]) / D
    ds = (-b[n]*c[n-2] + b[n-1]*c[n-1]) / D

  STEP 4: Update  r <- r + dr,  s <- s + ds
    ea(r) = |dr/r| x 100%,   ea(s) = |ds/s| x 100%
    Stop when BOTH ea(r) <= tol AND ea(s) <= tol

  STEP 5: Extract roots of x^2 - r*x - s = 0:
    discriminant = r^2 + 4s
    If disc >= 0: real roots (r +/- sqrtdisc) / 2
    If disc < 0: complex pair  r/2 +/- i*sqrt|disc|/2

  STEP 6: DEFLATION - divide the polynomial by x^2 - r*x - s
    The quotient coefficients are b[0..n-2] (drop last two residuals)
    Repeat with the deflated polynomial until degree <= 2

  STEP 7: Solve the remaining linear or quadratic directly

COEFFICIENT ORDER CONVENTION:
  Bairstow uses HIGHEST power first:
  x^4 - 5x^3 + 7x^2 - 5x + 6  ->  [1, -5, 7, -5, 6]  (index 0 = leading term)
=============================================================================
"""

import math
import cmath
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# SYNTHETIC DIVISION (single pass)
# -----------------------------------------------------------------------------

def synthetic_division(a, r, s):
    """
    Divides polynomial `a` by (x^2 - r*x - s) using synthetic division.

    Parameters
    ----------
    a : list of coefficients, highest power FIRST, e.g. x^3-6x^2+11x-6 -> [1,-6,11,-6]
    r, s : coefficients of trial quadratic factor x^2 - r*x - s

    Returns
    -------
    b : list, same length as a
        b[0..n-2] = quotient coefficients
        b[n-1], b[n] = remainders (should -> 0 at convergence)

    RECURRENCE:
      b[0] = a[0]
      b[1] = a[1] + r*b[0]
      b[i] = a[i] + r*b[i-1] + s*b[i-2]   for i = 2..n
    """
    n = len(a) - 1
    b = [0.0] * (n + 1)
    b[0] = a[0]
    b[1] = a[1] + r * b[0]
    for i in range(2, n + 1):
        b[i] = a[i] + r * b[i - 1] + s * b[i - 2]
    return b


# -----------------------------------------------------------------------------
# BAIRSTOW CORE - refine ONE quadratic factor
# -----------------------------------------------------------------------------

def bairstow_one_factor(a, r0=1.0, s0=1.0, tol=0.001, max_iter=100, verbose=True):
    """
    Refines ONE quadratic factor x^2 - r*x - s for polynomial `a`.

    Returns
    -------
    r, s        : converged quadratic factor coefficients
    quotient    : deflated polynomial (a divided by x^2 - r*x - s)
    history     : list of dicts {iter, r, s, dr, ds, ea_r, ea_s}
    """
    n = len(a) - 1
    r, s = float(r0), float(s0)
    history = []

    if verbose:
        print(f"\n  Degree {n} polynomial: refining x^2 - r*x - s")
        print(f"  {'Iter':<5} {'r':>12} {'s':>12} {'dr':>12} {'ds':>12} {'ea(r)%':>10} {'ea(s)%':>10}")
        print(f"  {'-'*75}")

    for i in range(1, max_iter + 1):
        # -- First synthetic division: get b-array -----------------------------
        b = synthetic_division(a, r, s)

        # -- Second synthetic division on b (up to n-1 only!) -----------------
        # Create b' = b[0..n-1] (drop b[n]), then divide by same quadratic
        b_short = b[:n]                 # length n (indices 0..n-1)
        c = synthetic_division(b_short, r, s)
        # c has length n, indexed 0..n-1
        # We need c[n-2], c[n-3], c[n-1] in 0-indexed terms of the length-n array
        # In the b_short array of length n (degree n-1):
        #   c[n-2] is at index n-2, c[n-3] at n-3, c[n-1] at n-1
        cn2 = c[n - 2]                  # c_{n-2}
        cn3 = c[n - 3] if n >= 3 else 0.0  # c_{n-3}
        cn1 = c[n - 1]                  # c_{n-1}

        # Residuals we want to drive to zero
        bn1 = b[n - 1]
        bn  = b[n]

        # -- Compute 2x2 system determinant ------------------------------------
        det = cn2 * cn2 - cn3 * cn1

        if abs(det) < 1e-14:
            # singular - perturb and try again
            r += 0.5;  s += 0.5
            if verbose:
                print(f"  {i:<5} Singular determinant - perturbing r, s")
            continue

        # -- Corrections dr, ds ------------------------------------------------
        dr = (-bn1 * cn2 + bn  * cn3) / det
        ds = (-bn  * cn2 + bn1 * cn1) / det

        r += dr
        s += ds

        # -- Approximate relative errors ---------------------------------------
        ea_r = abs(dr / r) * 100.0 if r != 0 else abs(dr) * 1e10
        ea_s = abs(ds / s) * 100.0 if s != 0 else abs(ds) * 1e10

        history.append({'iter': i, 'r': r, 's': s, 'dr': dr, 'ds': ds,
                        'ea_r': ea_r, 'ea_s': ea_s})

        if verbose:
            print(f"  {i:<5} {r:>12.6f} {s:>12.6f} {dr:>12.6f} {ds:>12.6f} "
                  f"{ea_r:>10.4f} {ea_s:>10.4f}")

        # -- Stopping check: both errors must be below tolerance ---------------
        if ea_r <= tol and ea_s <= tol:
            break

    # -- Deflation: quotient = b[0..n-2] (drop the two remainder entries) -----
    b_final   = synthetic_division(a, r, s)
    quotient  = b_final[:n - 1]        # coefficients of the deflated polynomial

    if verbose:
        print(f"\n  Converged: r = {r:.8f},  s = {s:.8f}")
        print(f"  Quadratic factor: x^2 - ({r:.6f})*x - ({s:.6f})")

    return r, s, quotient, history


# -----------------------------------------------------------------------------
# BAIRSTOW ALL ROOTS - full deflation loop
# -----------------------------------------------------------------------------

def bairstow_all_roots(coeffs, r0=1.0, s0=1.0, tol=0.001, max_iter=100, verbose=True):
    """
    Finds ALL roots of a polynomial by repeatedly extracting quadratic factors.

    Parameters
    ----------
    coeffs : list, highest power FIRST.
             e.g. x^4-5x^3+7x^2-5x+6 -> [1, -5, 7, -5, 6]
    r0, s0 : initial guess for quadratic factor coefficients

    Returns
    -------
    roots : list of floats or complex numbers
    """
    a     = [float(c) for c in coeffs]
    roots = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"{'Bairstow\'s Method - Full Root Extraction':^70}")
        print(f"  Polynomial degree: {len(a)-1}")
        print(f"  Initial guesses: r0 = {r0},  s0 = {s0},  tol = {tol}%")
        print(f"{'='*70}")

    # -- Extraction loop: while degree >= 3, extract a quadratic factor ---------
    while len(a) - 1 >= 3:
        r, s, a, _ = bairstow_one_factor(a, r0, s0, tol, max_iter, verbose)

        # -- Roots of x^2 - r*x - s = 0 ----------------------------------------
        discriminant = r * r + 4 * s
        if discriminant >= 0:
            sq = math.sqrt(discriminant)
            root1 = (r + sq) / 2.0
            root2 = (r - sq) / 2.0
        else:
            sq = cmath.sqrt(discriminant)    # complex square root
            root1 = (r + sq) / 2.0
            root2 = (r - sq) / 2.0

        roots.extend([root1, root2])
        if verbose:
            print(f"  Roots from this factor: {root1},  {root2}")

    # -- Handle remaining polynomial (degree 2 or 1) ---------------------------
    if len(a) - 1 == 2:
        # quadratic: a[0]x^2 + a[1]x + a[2] = 0
        a0, a1, a2 = a
        disc = a1*a1 - 4*a0*a2
        if disc >= 0:
            sq = math.sqrt(disc)
            roots.extend([(-a1 + sq) / (2*a0), (-a1 - sq) / (2*a0)])
        else:
            sq = cmath.sqrt(disc)
            roots.extend([(-a1 + sq) / (2*a0), (-a1 - sq) / (2*a0)])
        if verbose:
            print(f"  Final quadratic solved directly.")

    elif len(a) - 1 == 1:
        # linear: a[0]x + a[1] = 0
        roots.append(-a[1] / a[0])
        if verbose:
            print(f"  Final linear solved directly.")

    # -- Clean up: remove negligible imaginary parts from real roots -----------
    cleaned = []
    for rt in roots:
        if isinstance(rt, complex) and abs(rt.imag) < 1e-10:
            cleaned.append(float(rt.real))
        else:
            cleaned.append(rt)

    if verbose:
        print(f"\n{'='*70}")
        print(f"{'ALL ROOTS':^70}")
        print(f"{'-'*70}")
        for k, rt in enumerate(cleaned):
            print(f"  Root {k+1}: {rt}")
        print(f"{'='*70}")

    return cleaned


# -----------------------------------------------------------------------------
# POLYNOMIAL EVALUATOR - verify roots by plugging back in
# -----------------------------------------------------------------------------

def poly_eval(coeffs, x):
    """Evaluate polynomial at x using Horner's method (coeffs highest first)."""
    val = 0.0 + 0j if isinstance(x, complex) else 0.0
    for c in coeffs:
        val = val * x + c
    return val


def verify_roots(coeffs, roots):
    """Print |f(root)| for each root - should be approx. 0."""
    print(f"\n  {'Root':>25}  {'|f(root)|':>14}")
    print(f"  {'-'*42}")
    for rt in roots:
        fval = poly_eval(coeffs, rt)
        print(f"  {str(rt):>25}  {abs(fval):>14.2e}")


# -----------------------------------------------------------------------------
# == EXAMPLES ==
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -- EXAMPLE 1: Standard textbook polynomial --------------------------------
    # f(x) = x^4 - 5x^3 + 7x^2 - 5x + 6
    # True roots: 2, 3, and complex conjugate pair (1+/-i)
    print("\n" + "="*70)
    print("EXAMPLE 1: x^4 - 5x^3 + 7x^2 - 5x + 6")
    print("  True roots: 2, 3, 1+i, 1-i")
    print("="*70)

    coeffs1 = [1, -5, 7, -5, 6]
    roots1  = bairstow_all_roots(coeffs1, r0=1.0, s0=1.0, tol=0.001)
    verify_roots(coeffs1, roots1)

    # -- EXAMPLE 2: Chapra 5th edition x^4 - 10x^3 + 35x^2 - 50x + 24 ----------
    # True roots: 1, 2, 3, 4
    print("\n" + "="*70)
    print("EXAMPLE 2: x^4 - 10x^3 + 35x^2 - 50x + 24")
    print("  True roots: 1, 2, 3, 4")
    print("="*70)

    coeffs2 = [1, -10, 35, -50, 24]
    roots2  = bairstow_all_roots(coeffs2, r0=0.0, s0=1.0, tol=0.001)
    verify_roots(coeffs2, roots2)

    # Convergence of (r, s) - show how they evolve iteration by iteration
    print("\n  Detailed convergence for Example 2 (first quadratic factor):")
    r, s, quot, hist = bairstow_one_factor(coeffs2, r0=0.0, s0=1.0, tol=0.001, verbose=True)

    # Plot convergence of r and s
    if hist:
        iters = [h['iter'] for h in hist]
        r_vals = [h['r']    for h in hist]
        s_vals = [h['s']    for h in hist]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(iters, r_vals, 'o-', color='royalblue')
        ax1.set_xlabel('Iteration'); ax1.set_ylabel('r')
        ax1.set_title('Convergence of r'); ax1.grid(alpha=0.4)

        ax2.plot(iters, s_vals, 'o-', color='darkorange')
        ax2.set_xlabel('Iteration'); ax2.set_ylabel('s')
        ax2.set_title('Convergence of s'); ax2.grid(alpha=0.4)

        fig.suptitle("Bairstow's Method: r and s convergence", fontsize=13)
        plt.tight_layout()
        plt.savefig('bairstow_convergence.png', dpi=150)
        plt.show()

    # -- EXAMPLE 3: Cubic with one real + one complex pair --------------------
    # f(x) = x^3 - 6x^2 + 11x - 6  -> roots: 1, 2, 3
    print("\n" + "="*70)
    print("EXAMPLE 3: x^3 - 6x^2 + 11x - 6")
    print("  True roots: 1, 2, 3")
    print("="*70)

    coeffs3 = [1, -6, 11, -6]
    roots3  = bairstow_all_roots(coeffs3, r0=1.0, s0=-1.0, tol=0.001)
    verify_roots(coeffs3, roots3)

    # -- EXAMPLE 4: Degree-5 with mixed roots ----------------------------------
    # f(x) = x^5 - 3.5x^4 + 2.75x^3 + 2.125x^2 - 3.875x + 1.25
    print("\n" + "="*70)
    print("EXAMPLE 4: x^5 - 3.5x^4 + 2.75x^3 + 2.125x^2 - 3.875x + 1.25")
    print("="*70)

    coeffs4 = [1, -3.5, 2.75, 2.125, -3.875, 1.25]
    roots4  = bairstow_all_roots(coeffs4, r0=1.0, s0=0.5, tol=0.001, verbose=False)
    print(f"\n  Roots found:")
    for rt in roots4:
        print(f"    {rt}")
    verify_roots(coeffs4, roots4)
