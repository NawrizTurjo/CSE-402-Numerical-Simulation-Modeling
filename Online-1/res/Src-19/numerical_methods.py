"""
================================================================================
 Numerical Methods -- Reference Implementation
 Topics: Approximation/Error Analysis, Visualization, and Root-Finding Methods
          (Bisection, False Position, Newton-Raphson, Bairstow's Method)
================================================================================

This module is a self-contained, exam/lab-ready reference implementation.
Every function is written from first principles (no scipy.optimize shortcuts)
so the underlying algorithm is fully visible -- matching what is expected in
a numerical methods coding exam.

Run this file directly to see a demo of every method on example problems,
including plots (requires matplotlib).

    python3 numerical_methods.py

Dependencies: numpy, matplotlib  (both are standard for this course)
"""

import math
import cmath
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ==============================================================================
# SECTION 1: APPROXIMATION & ERROR ANALYSIS
# ==============================================================================

def true_error(true_value, approx_value):
    """E_t = true - approx"""
    return true_value - approx_value


def true_relative_error_percent(true_value, approx_value):
    """eps_t (%) = (true - approx) / true * 100.  Requires true_value != 0."""
    return (true_value - approx_value) / true_value * 100.0


def approx_relative_error_percent(x_new, x_old):
    """
    eps_a (%) = |(x_new - x_old) / x_new| * 100

    This is THE practical stopping criterion for iterative methods -- it does
    NOT require knowledge of the true value, only the last two iterates.
    """
    if x_new == 0:
        return float('inf')
    return abs((x_new - x_old) / x_new) * 100.0


def required_tolerance_for_sig_figs(n_sig_figs):
    """
    Scarborough criterion: returns the tolerance eps_s (%) needed to guarantee
    at least n_sig_figs correct significant figures.
        eps_s = (0.5 * 10^(2-n)) %
    """
    return 0.5 * 10 ** (2 - n_sig_figs)


def machine_epsilon():
    """
    Empirically determine machine epsilon: the smallest eps such that
    1.0 + eps != 1.0 in floating point arithmetic.
    """
    eps = 1.0
    while (1.0 + eps / 2.0) != 1.0:
        eps /= 2.0
    return eps


def taylor_exp(x, n_terms):
    """
    Truncated Taylor series approximation of e^x about x0 = 0, using the
    first `n_terms` terms:  1 + x + x^2/2! + ... + x^(n-1)/(n-1)!
    Demonstrates truncation error shrinking as n_terms grows.
    """
    total = 0.0
    term = 1.0
    for k in range(n_terms):
        total += term
        term *= x / (k + 1)
    return total


def demo_error_analysis():
    print("\n" + "=" * 70)
    print("SECTION 1: Approximation & Error Analysis")
    print("=" * 70)

    print(f"Machine epsilon (measured): {machine_epsilon():.3e}")
    print(f"Machine epsilon (numpy):    {np.finfo(float).eps:.3e}")

    true_val = math.e  # e^1
    print(f"\nApproximating e^1 = {true_val:.6f} via truncated Taylor series:")
    print(f"{'n_terms':>8} {'approx':>12} {'E_t':>12} {'eps_t (%)':>12}")
    for n in range(1, 10):
        approx = taylor_exp(1.0, n)
        et = true_error(true_val, approx)
        rel = true_relative_error_percent(true_val, approx)
        print(f"{n:>8} {approx:>12.6f} {et:>12.6f} {rel:>12.4f}")

    print("\nStopping tolerance to guarantee 5 significant figures: "
          f"eps_s = {required_tolerance_for_sig_figs(5):.5f} %")

    # Subtractive cancellation illustration
    print("\nSubtractive cancellation demo: f(x) = sqrt(x+1) - sqrt(x)")
    for x in [1e0, 1e6, 1e12]:
        naive = math.sqrt(x + 1) - math.sqrt(x)
        stable = 1.0 / (math.sqrt(x + 1) + math.sqrt(x))
        print(f"  x={x:<10.0e} naive={naive:.10f}  stable={stable:.10f}  "
              f"diff={abs(naive - stable):.2e}")


# ==============================================================================
# SECTION 2: VISUALIZATION AND PLOTTING HELPERS
# ==============================================================================

def plot_function_with_roots(f, x_range, roots=None, iterates=None, title="f(x)"):
    """
    Standard 'graphical method' plot: draws f(x) over x_range, marks the
    x-axis, and optionally overlays converged roots and/or intermediate
    iterate positions (useful for visualizing bisection/false-position/NR
    convergence on top of the curve).
    """
    if plt is None:
        print(f"  [Plotting Skipped] '{title}' - matplotlib is not installed.")
        return

    x = np.linspace(x_range[0], x_range[1], 500)
    y = np.array([f(xi) for xi in x])

    plt.figure(figsize=(7, 4.5))
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.plot(x, y, label='f(x)', color='steelblue')

    if iterates:
        plt.scatter(iterates, [f(xi) for xi in iterates],
                    color='orange', zorder=4, s=25, label='iterates')
    if roots:
        plt.scatter(roots, [0] * len(roots),
                    color='red', zorder=5, s=50, marker='x', label='converged root')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plot_{title.replace(' ', '_').replace('/', '_')}.png", dpi=120)
    plt.close()


def plot_convergence(errors, title="Convergence"):
    """
    Plots approximate error vs. iteration number on a semi-log y-axis.
    A straight-line decline => linear convergence (bisection/false position).
    A rapidly steepening decline => quadratic convergence (Newton-Raphson).
    """
    if plt is None:
        print(f"  [Plotting Skipped] '{title}' - matplotlib is not installed.")
        return

    plt.figure(figsize=(7, 4.5))
    iters = list(range(1, len(errors) + 1))
    plt.semilogy(iters, errors, marker='o', color='darkred')
    plt.xlabel('Iteration')
    plt.ylabel('Approximate relative error (%) [log scale]')
    plt.title(title)
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f"plot_{title.replace(' ', '_').replace('/', '_')}.png", dpi=120)
    plt.close()


# ==============================================================================
# SECTION 3: BRACKETING METHODS
# ==============================================================================

def bisection(f, xl, xu, tol=1e-6, max_iter=200, verbose=False):
    """
    Bisection method.

    Parameters
    ----------
    f        : callable, f(x)
    xl, xu   : initial bracket such that f(xl)*f(xu) < 0
    tol      : stopping tolerance on approximate relative error (%)
    max_iter : safety cap on iterations

    Returns
    -------
    root, history
        root    : final root estimate
        history : list of dicts with iteration data (xl, xu, xr, ea)
    """
    fl = f(xl)
    fu = f(xu)
    if fl == 0.0:
        return xl, [{'iter': 1, 'xl': xl, 'xu': xu, 'xr': xl, 'ea': 0.0}]
    if fu == 0.0:
        return xu, [{'iter': 1, 'xl': xl, 'xu': xu, 'xr': xu, 'ea': 0.0}]

    sign_l = math.copysign(1.0, fl)
    sign_u = math.copysign(1.0, fu)
    if sign_l == sign_u:
        raise ValueError("f(xl) and f(xu) must have opposite signs (no sign change in bracket).")

    xr_old = xl
    history = []

    for i in range(1, max_iter + 1):
        xr = (xl + xu) / 2.0
        ea = approx_relative_error_percent(xr, xr_old) if i > 1 else float('inf')

        history.append({'iter': i, 'xl': xl, 'xu': xu, 'xr': xr, 'ea': ea})
        if verbose:
            print(f"  iter {i:3d}: xl={xl:.6f} xu={xu:.6f} xr={xr:.6f} ea={ea:.4f}%")

        fr = f(xr)
        if fr == 0.0:
            return xr, history  # exact root found

        sign_r = math.copysign(1.0, fr)
        if sign_l != sign_r:
            xu = xr
        else:
            xl = xr
            fl = fr
            sign_l = sign_r

        if ea < tol:
            return xr, history

        xr_old = xr

    return xr, history  # did not converge within max_iter -- return best estimate


def false_position(f, xl, xu, tol=1e-6, max_iter=200, verbose=False):
    """
    False Position (Regula Falsi) method.
    """
    fl, fu = f(xl), f(xu)
    if fl == 0.0:
        return xl, [{'iter': 1, 'xl': xl, 'xu': xu, 'xr': xl, 'ea': 0.0}]
    if fu == 0.0:
        return xu, [{'iter': 1, 'xl': xl, 'xu': xu, 'xr': xu, 'ea': 0.0}]

    sign_l = math.copysign(1.0, fl)
    sign_u = math.copysign(1.0, fu)
    if sign_l == sign_u:
        raise ValueError("f(xl) and f(xu) must have opposite signs (no sign change in bracket).")

    xr_old = xl
    history = []

    for i in range(1, max_iter + 1):
        if abs(fl - fu) < 1e-15:
            break
        xr = xu - fu * (xl - xu) / (fl - fu)
        fr = f(xr)
        ea = approx_relative_error_percent(xr, xr_old) if i > 1 else float('inf')

        history.append({'iter': i, 'xl': xl, 'xu': xu, 'xr': xr, 'ea': ea})
        if verbose:
            print(f"  iter {i:3d}: xl={xl:.6f} xu={xu:.6f} xr={xr:.6f} ea={ea:.4f}%")

        if fr == 0.0:
            return xr, history

        sign_r = math.copysign(1.0, fr)
        if sign_l != sign_r:
            xu, fu = xr, fr
            sign_u = sign_r
        else:
            xl, fl = xr, fr
            sign_l = sign_r

        if ea < tol:
            return xr, history

        xr_old = xr

    return xr, history


# ==============================================================================
# SECTION 4: OPEN METHODS
# ==============================================================================

def newton_raphson(f, fprime, x0, tol=1e-6, max_iter=100, verbose=False):
    """
    Newton-Raphson method.

    Parameters
    ----------
    f, fprime : callables, f(x) and its derivative f'(x)
    x0        : initial guess

    Returns
    -------
    root, history  (history holds x_i, f(x_i), ea at each iteration)

    Notes
    -----
    Raises ZeroDivisionError-safe handling: if |f'(x)| is extremely small,
    the iteration is aborted with a warning printed, since the correction
    term would blow up (a known pitfall of this method).
    """
    x_old = x0
    history = [{'iter': 0, 'x': x0, 'fx': f(x0), 'ea': float('inf')}]

    for i in range(1, max_iter + 1):
        fpx = fprime(x_old)
        if abs(fpx) < 1e-14:
            print(f"  WARNING: |f'(x)| too small at x={x_old:.6f} -- aborting "
                  "(possible inflection point / zero derivative pitfall).")
            break

        x_new = x_old - f(x_old) / fpx
        ea = approx_relative_error_percent(x_new, x_old)

        history.append({'iter': i, 'x': x_new, 'fx': f(x_new), 'ea': ea})
        if verbose:
            print(f"  iter {i:3d}: x={x_new:.8f} f(x)={f(x_new):.2e} ea={ea:.6f}%")

        if ea < tol or abs(x_new - x_old) < 1e-12:
            return x_new, history

        x_old = x_new

    return x_old, history


def newton_raphson_numeric_derivative(f, x0, h=1e-6, tol=1e-6, max_iter=100, verbose=False):
    """
    Newton-Raphson variant using a central-difference numerical derivative,
    for cases where f'(x) is inconvenient to derive analytically.
        f'(x) ~= (f(x+h) - f(x-h)) / (2h)
    """
    fprime_numeric = lambda x: (f(x + h) - f(x - h)) / (2 * h)
    return newton_raphson(f, fprime_numeric, x0, tol, max_iter, verbose)


# ==============================================================================
# SECTION 5: BAIRSTOW'S METHOD (polynomial root finding, real + complex roots)
# ==============================================================================

def _synthetic_division(a, r, s):
    """
    Single synthetic-division pass used by Bairstow's method.
    Given polynomial coefficients a[0..n] (highest degree first) and a trial
    quadratic divisor x^2 - r*x - s, computes the b[] coefficients:
        b0 = a0
        b1 = a1 + r*b0
        bi = ai + r*b[i-1] + s*b[i-2]   for i = 2..n
    Returns the list b (same length as a).
    """
    n = len(a) - 1
    b = [0.0] * (n + 1)
    b[0] = a[0]
    b[1] = a[1] + r * b[0]
    for i in range(2, n + 1):
        b[i] = a[i] + r * b[i - 1] + s * b[i - 2]
    return b


def bairstow_quadratic_factor(a, r0=1.0, s0=1.0, tol=1e-8, max_iter=200, verbose=False):
    """
    Extracts ONE quadratic factor (x^2 - r*x - s) from polynomial `a`
    (coefficients, highest degree first, a[0] != 0) using Bairstow's method.

    Returns
    -------
    r, s, b   : converged r, s and the final quotient coefficients b[0..n-2]
                (b[n-1], b[n] are the residuals, ~0 at convergence)
    """
    n = len(a) - 1
    if n < 2:
        raise ValueError("Polynomial degree must be >= 2 for Bairstow's method.")

    r, s = r0, s0

    for it in range(1, max_iter + 1):
        b = _synthetic_division(a, r, s)
        # second synthetic division of b (dropping the last two entries) -> c
        c = _synthetic_division(b[:-1], r, s)  # length n (indices 0..n-1)

        bn1, bn = b[n - 1], b[n]          # the two residuals to drive to 0
        cn2, cn3 = c[n - 2], c[n - 3] if n - 3 >= 0 else 0.0
        cn1 = c[n - 1]

        det = cn2 * cn2 - cn3 * cn1  # a common way to write the 2x2 determinant
        # Solve:  cn2*dr + cn3*ds = -bn1
        #         cn1*dr + cn2*ds = -bn
        if abs(det) < 1e-14:
            # Perturb to escape a singular system
            r += 1.0
            s += 1.0
            continue

        dr = (-bn1 * cn2 + bn * cn3) / det
        ds = (-bn * cn2 + bn1 * cn1) / det

        r += dr
        s += ds

        ea_r = approx_relative_error_percent(r, r - dr)
        ea_s = approx_relative_error_percent(s, s - ds)

        if verbose:
            print(f"  iter {it:3d}: r={r:.8f} s={s:.8f} ea_r={ea_r:.4e}% ea_s={ea_s:.4e}% "
                  f"resid=({bn1:.2e},{bn:.2e})")

        if ea_r < tol and ea_s < tol:
            b_final = _synthetic_division(a, r, s)
            return r, s, b_final[:-2]  # quotient coefficients (degree n-2)

    b_final = _synthetic_division(a, r, s)
    return r, s, b_final[:-2]


def poly_eval_and_deriv(a, x):
    """
    Evaluates polynomial and its derivative using Horner's method.
    """
    val = a[0]
    deriv = 0.0 + 0.0j if isinstance(x, complex) else 0.0
    for i in range(1, len(a)):
        deriv = deriv * x + val
        val = val * x + a[i]
    return val, deriv


def bairstow_all_roots(a, tol=1e-8, max_iter=200, verbose=False):
    """
    Repeatedly applies Bairstow's method + deflation to find ALL roots
    (real and complex-conjugate) of a polynomial with real coefficients.

    Parameters
    ----------
    a : list of coefficients, highest degree first, e.g. x^3-6x^2+11x-6 ->
        [1, -6, 11, -6]

    Returns
    -------
    roots : list of complex numbers (real roots have ~0 imaginary part)
    """
    coeffs = [float(c) for c in a]
    roots = []

    while len(coeffs) - 1 > 2:
        r, s, quotient = bairstow_quadratic_factor(coeffs, tol=tol, max_iter=max_iter, verbose=verbose)
        # Roots of x^2 - r*x - s = 0
        disc = r * r + 4 * s
        if disc >= 0:
            sq = math.sqrt(disc)
            roots.append((r + sq) / 2.0)
            roots.append((r - sq) / 2.0)
        else:
            sq = cmath.sqrt(disc)
            roots.append((r + sq) / 2.0)
            roots.append((r - sq) / 2.0)
        coeffs = quotient

    # Final degree-2 (or 1) polynomial: solve directly
    if len(coeffs) - 1 == 2:
        a0, a1, a2 = coeffs
        disc = a1 * a1 - 4 * a0 * a2
        if disc >= 0:
            sq = math.sqrt(disc)
        else:
            sq = cmath.sqrt(disc)
        roots.append((-a1 + sq) / (2 * a0))
        roots.append((-a1 - sq) / (2 * a0))
    elif len(coeffs) - 1 == 1:
        a0, a1 = coeffs
        roots.append(-a1 / a0)

    # Refine the roots to eliminate deflation error accumulation
    f = lambda x: poly_eval_and_deriv(a, x)[0]
    fprime = lambda x: poly_eval_and_deriv(a, x)[1]
    refined_roots = []
    for rt in roots:
        refined_roots.append(refine_root_newton(f, fprime, rt, tol=tol))

    return refined_roots


def refine_root_newton(f, fprime, root_guess, tol=1e-10, max_iter=50):
    """
    Refines a (possibly complex) root found via Bairstow's deflation by
    running a few Newton-Raphson iterations on the ORIGINAL (non-deflated)
    polynomial, to counteract round-off error accumulated during deflation.
    Works for complex root_guess too since Python complex arithmetic is native.
    """
    x = root_guess
    for _ in range(max_iter):
        fpx = fprime(x)
        if abs(fpx) < 1e-14:
            break
        x_new = x - f(x) / fpx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x


# ==============================================================================
# SECTION 6: DEMOS  (run this file directly to execute all of these)
# ==============================================================================

def demo_visualization_and_bisection():
    print("\n" + "=" * 70)
    print("SECTION 3: Bisection Method  (f(x) = x^3 - 6x^2 + 11x - 6.1)")
    print("=" * 70)

    f = lambda x: x**3 - 6*x**2 + 11*x - 6.1

    root, hist = bisection(f, 0.0, 1.5, tol=1e-5, verbose=True)
    print(f"Root found: {root:.8f}  (f(root) = {f(root):.2e}), "
          f"iterations = {len(hist)}")

    plot_function_with_roots(f, (-1, 4), roots=[root],
                              iterates=[h['xr'] for h in hist],
                              title="Bisection Demo")
    plot_convergence([h['ea'] for h in hist if h['ea'] != float('inf')],
                      title="Bisection Convergence")


def demo_false_position():
    print("\n" + "=" * 70)
    print("SECTION 3: False Position Method  (falling object velocity)")
    print("=" * 70)

    g, m, c = 9.81, 68.1, 12.5
    v = lambda t: (g * m / c) * (1 - math.exp(-(c / m) * t))
    f = lambda t: v(t) - 40.0

    root, hist = false_position(f, 0.0, 20.0, tol=1e-6, verbose=True)
    print(f"Time to reach v=40 m/s: t = {root:.6f} s, iterations = {len(hist)}")

    root_b, hist_b = bisection(f, 0.0, 20.0, tol=1e-6)
    print(f"(Bisection comparison: {len(hist_b)} iterations for same tolerance)")


def demo_newton_raphson():
    print("\n" + "=" * 70)
    print("SECTION 4: Newton-Raphson Method  (f(x) = x^2 - 2, target sqrt(2))")
    print("=" * 70)

    f = lambda x: x**2 - 2
    fprime = lambda x: 2 * x

    root, hist = newton_raphson(f, fprime, x0=1.0, tol=1e-10, verbose=True)
    print(f"Root found: {root:.10f}  (true value sqrt(2) = {math.sqrt(2):.10f})")

    plot_convergence([h['ea'] for h in hist if h['ea'] != float('inf')],
                      title="Newton-Raphson Convergence")


def demo_bairstow():
    print("\n" + "=" * 70)
    print("SECTION 5: Bairstow's Method  (f(x) = x^3 - 6x^2 + 11x - 6, roots 1,2,3)")
    print("=" * 70)

    a = [1, -6, 11, -6]
    roots = bairstow_all_roots(a, verbose=True)
    print("All roots found:")
    for rt in roots:
        print(f"  {rt}")

    print("\n--- Degree-5 example with complex roots ---")
    a5 = [1, -3.5, 2.75, 2.125, -3.875, 1.25]
    roots5 = bairstow_all_roots(a5)
    print("Roots of x^5 - 3.5x^4 + 2.75x^3 + 2.125x^2 - 3.875x + 1.25:")
    for rt in roots5:
        print(f"  {rt}")


if __name__ == "__main__":
    demo_error_analysis()
    demo_visualization_and_bisection()
    demo_false_position()
    demo_newton_raphson()
    demo_bairstow()
    print("\nAll demos complete. PNG plots saved in the current directory.")
