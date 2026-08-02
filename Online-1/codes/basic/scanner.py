# -*- coding: utf-8 -*-
"""
=============================================================================
  CSE-402 Numerical Methods: Incremental Scanner Utility
=============================================================================

This module provides a standalone, decoupled utility for scanning continuous
functions to locate sub-brackets where roots exist (sign-change intervals).
Exposing this as a utility allows quick reuse across different methods during
timed root-finding tests.
=============================================================================
"""

import sys
import io
import numpy as np

# Ensure UTF-8 output to avoid Windows console errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def find_sign_change_intervals(f, start, end, step=0.1):
    """
    Scans a domain sequentially to identify all sub-brackets containing a root
    (where f(x_left) * f(x_right) < 0).
    
    Parameters:
    -----------
    f     : callable, the objective function f(x)
    start : float, the beginning of the search interval
    end   : float, the end of the search interval
    step  : float, the increment step size (default is 0.1)
    
    Returns:
    --------
    list of tuples: [(xl_1, xu_1), (xl_2, xu_2), ...]
    """
    # Round domain bounds to float precision to avoid floating-point issues
    x_domain = np.arange(start, end + step, step)
    intervals = []
    
    for i in range(len(x_domain) - 1):
        x_left = x_domain[i]
        x_right = x_domain[i+1]
        
        # Avoid exact zeros if function is undefined or already hit a root
        try:
            val_l = f(x_left)
            val_r = f(x_right)
            if val_l * val_r < 0:
                intervals.append((round(x_left, 10), round(x_right, 10)))
        except (ValueError, ZeroDivisionError, OverflowError):
            # Skip points out of the domain of f
            continue
            
    return intervals


def safe_eval(f, x):
    """
    Evaluate f(x), catching domain errors (division by zero, log of a
    non-positive number, overflow) instead of letting them crash the caller.
    Returns (value, None) on success, (None, error_message) on failure.
    """
    try:
        return f(x), None
    except (ZeroDivisionError, ValueError, OverflowError) as e:
        return None, str(e)


def coarse_scan(f, a, b, step=0.1, direct_tol=1e-8):
    """
    Scan [a, b] with a fixed step and classify every grid point into two
    disjoint candidate buckets. Never run bisection/false-position once over
    a whole wide domain - scan first, then refine each candidate separately
    (a wide interval can hide an even number of roots, or an asymptote can
    fake a sign change).
      - direct_candidates   : x where |f(x)| is already <= direct_tol
      - interval_candidates : consecutive (x, x+step) pairs where f changes sign
    Domain errors mid-scan (e.g. an asymptote) are skipped, not fatal.
    """
    xs = list(np.arange(a, b + step, step))
    vals = [safe_eval(f, x)[0] for x in xs]

    direct_candidates = [xs[i] for i, v in enumerate(vals)
                          if v is not None and abs(v) <= direct_tol]

    interval_candidates = []
    for i in range(len(xs) - 1):
        v0, v1 = vals[i], vals[i + 1]
        if v0 is None or v1 is None:
            continue
        if v0 * v1 < 0:
            interval_candidates.append((round(xs[i], 10), round(xs[i + 1], 10)))

    return direct_candidates, interval_candidates


def classify_and_verify(f, candidate, kind, extra_filter=None, residual_tol=1e-6):
    """
    Turn a raw candidate x-value into a reportable verdict row: type
    ('direct' or 'interval'), the residual f(candidate), and an explicit
    ACCEPT/REJECT decision. Convergence alone is never enough - the residual
    must actually be near zero (guards against bisection/FP "converging" onto
    a sign-flipping singularity, e.g. f(x)=1/x), and any problem-specific
    extra filter (e.g. "accept only if |xr| <= 1e-6") must also pass.
    """
    fx, err = safe_eval(f, candidate)
    accepted = fx is not None and abs(fx) <= residual_tol
    if accepted and extra_filter is not None:
        accepted = extra_filter(candidate)
    return {'type': kind, 'x': candidate, 'f_x': fx, 'error': err, 'accepted': accepted}


if __name__ == '__main__':
    # Test function with known roots: f(x) = sin(x) on [0, 10]
    # Roots should be at pi (~3.14), 2*pi (~6.28), 3*pi (~9.42)
    test_f = lambda x: np.sin(x)
    found = find_sign_change_intervals(test_f, 0, 10, step=0.1)

    print("=== Testing Incremental Scanner Utility ===")
    print(f"Function: sin(x) on [0, 10] with step = 0.1")
    print(f"Discovered sign-change intervals: {found}")

    # Test coarse_scan + classify_and_verify on a pathological function
    # (double root that never sign-changes + an asymptote that fakes one)
    print("\n=== Testing coarse_scan + classify_and_verify ===")
    path_f = lambda x: ((x - 2.5) ** 2 * (x + 1.052)) / (x - 3.551)
    direct, intervals = coarse_scan(path_f, -2, 5, step=0.1, direct_tol=1e-8)
    print(f"Function: (x-2.5)^2*(x+1.052)/(x-3.551) on [-2, 5] with step = 0.1")
    print(f"Direct candidates:   {direct}")
    print(f"Interval candidates: {intervals}")
    for xl, xu in intervals:
        verdict = classify_and_verify(path_f, (xl + xu) / 2.0, 'interval',
                                      extra_filter=lambda x: abs(x) <= 1e-6)
        print(f"  midpoint of [{xl:.2f},{xu:.2f}] -> {verdict}")
