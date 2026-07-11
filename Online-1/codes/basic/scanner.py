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


if __name__ == '__main__':
    # Test function with known roots: f(x) = sin(x) on [0, 10]
    # Roots should be at pi (~3.14), 2*pi (~6.28), 3*pi (~9.42)
    test_f = lambda x: np.sin(x)
    found = find_sign_change_intervals(test_f, 0, 10, step=0.1)
    
    print("=== Testing Incremental Scanner Utility ===")
    print(f"Function: sin(x) on [0, 10] with step = 0.1")
    print(f"Discovered sign-change intervals: {found}")
