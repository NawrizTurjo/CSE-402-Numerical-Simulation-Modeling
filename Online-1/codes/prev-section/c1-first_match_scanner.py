"""
Section-C1: First-Match Multi-Root Scanner (False Position)
Objective:
  Scan the domain of a multi-root function to identify all sign-change intervals.
  Automatically isolate the FIRST interval with a sign change and solve it
  using the False Position Method.
  
Tasks:
  1. Plot the function across the domain, highlighting the target interval.
  2. Perform incremental search to locate all sign-change intervals.
  3. Execute False Position only on the first interval.
  4. Output the complete iteration table targeting an error tolerance of |ea| <= 0.0001%.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
DOMAIN_START = 0.0
DOMAIN_END = 4.0
STEP = 0.1
TOL = 0.0001     # Stopping tolerance (%)
MAX_ITER = 100

# -----------------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------------
def f(x):
    """Cubic multi-root equation template."""
    return 2.0 * x**3 - 11.7 * x**2 + 17.7 * x - 5.0

def f_np(x):
    """Numpy vectorized cubic equation."""
    return 2.0 * x**3 - 11.7 * x**2 + 17.7 * x - 5.0

# -----------------------------------------------------------------------------
# Incremental Search
# -----------------------------------------------------------------------------
def scan_intervals(start, end, step_size):
    """Scan the domain to locate all sub-intervals containing sign changes."""
    xs = np.arange(start, end + step_size, step_size)
    intervals = []
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i+1]
        f1_val = f(x1)
        f2_val = f(x2)
        if f1_val * f2_val < 0:
            intervals.append((x1, x2))
    return intervals

# -----------------------------------------------------------------------------
# False Position Solver
# -----------------------------------------------------------------------------
def run_false_position(xl, xu):
    """Run False Position on a single interval and print its iteration table."""
    print(f"\nRunning False Position on First Detected Interval: [{xl:.2f}, {xu:.2f}]")
    print("-" * 72)
    print(f"{'Iter':<6} {'xl':>12} {'xu':>12} {'xr':>12} {'ea (%)':>12} {'f(xr)':>14}")
    print("-" * 72)
    
    xr_old = None
    xr = None
    converged = False
    
    for i in range(1, MAX_ITER + 1):
        fl = f(xl)
        fu = f(xu)
        
        # False Position formula
        xr = xu - (fu * (xl - xu)) / (fl - fu)
        fxr = f(xr)
        
        # Approximate relative error
        if xr_old is not None:
            if abs(xr) >= 1e-15:
                ea = abs((xr - xr_old) / xr) * 100.0
                ea_str = f"{ea:.6f}"
            else:
                # ea = float('inf'); ea_str = "---"  # Alternative exact-zero infinity sentinel
                ea = abs(xr - xr_old) * 100.0
                ea_str = f"{ea:.6f}"
        else:
            ea = 100.0
            ea_str = "---"
            
        print(f"{i:<6} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>14.6e}")
        
        if xr_old is not None and ea <= TOL:
            converged = True
            break
            
        # Update bounds
        if fl * fxr < 0:
            xu = xr
        else:
            xl = xr
            
        xr_old = xr
        
    print("-" * 72)
    if converged:
        print(f"[SUCCESS] Converged to root x approx. {xr:.8f} in {i} iterations.")
    else:
        print(f"[FAILURE] Did not converge within {MAX_ITER} iterations.")
        
    return xr

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("   SECTION-C1: FIRST-MATCH MULTI-ROOT SCANNER (FALSE POSITION)")
    print("=" * 80)
    
    # Task 2: Scan for all intervals
    intervals = scan_intervals(DOMAIN_START, DOMAIN_END, STEP)
    print(f"Incremental scan results (step size = {STEP}):")
    for idx, (xl, xu) in enumerate(intervals, 1):
        print(f"  Interval #{idx}: [{xl:.2f}, {xu:.2f}]")
        
    # Task 3: Execute False Position only on the first interval
    root = None
    if intervals:
        xl_first, xu_first = intervals[0]
        root = run_false_position(xl_first, xu_first)
    else:
        print("No sign-change intervals detected in the domain.")
        
    # Task 1: Plot the function and highlight the first interval
    x_plot = np.linspace(DOMAIN_START, DOMAIN_END, 500)
    y_plot = f_np(x_plot)
    
    plt.figure(figsize=(9, 6))
    plt.plot(x_plot, y_plot, label=r"$f(x) = 2x^3 - 11.7x^2 + 17.7x - 5$", color="forestgreen", linewidth=2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    
    # Mark sign change intervals
    for idx, (xl, xu) in enumerate(intervals, 1):
        if idx == 1:
            plt.axvspan(xl, xu, color="yellow", alpha=0.3, label="Target First Interval")
        else:
            plt.axvspan(xl, xu, color="gray", alpha=0.1, label="Other Sign Changes" if idx == 2 else "")
            
    if root is not None:
        plt.plot(root, f(root), "ro", markersize=8, label=f"Solved Root (x ≈ {root:.5f})")
        
    plt.title("Section-C1: First-Match Multi-Root Scanner", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("x", fontsize=11)
    plt.ylabel("f(x)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plot_path = "c1_first_match_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved visualization graph to '{plot_path}'")
    print("=" * 80)

if __name__ == "__main__":
    main()
