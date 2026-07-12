"""
Section-B2: Pathological/Discontinuous Functions (Bisection)
Objective:
  Analyze and process the root behavior of the rational function:
  f(x) = ((x-2.5)^2 * (x+1.5)) / (x-3.56)

Tasks:
  1. Perform an incremental interval scan over [-2, 5] and run Bisection on valid bounding regions.
  2. Plot the curve, paying close attention to points near x = 2.5 and x = 3.56.
  3. Provide written conceptual analysis on missed double roots and false sign changes at asymptotes.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
X_MIN = -2.0
X_MAX = 5.0
STEP = 0.1
TOL = 0.0001     # Stopping tolerance (%)
MAX_ITER = 100

# -----------------------------------------------------------------------------
# Function Definition
# -----------------------------------------------------------------------------
def f(x):
    """Rational pathological function."""
    denom = x - 3.56
    if abs(denom) < 1e-15:
        # Avoid explicit division by zero in point evaluation
        return float('inf') if (x - 2.5)**2 * (x + 1.5) >= 0 else float('-inf')
    return ((x - 2.5)**2 * (x + 1.5)) / denom

def f_np(x):
    """Numpy vectorized version for plotting, with masking for the asymptote."""
    denom = x - 3.56
    # Avoid zero division warnings in numpy
    with np.errstate(divide='ignore', invalid='ignore'):
        y = ((x - 2.5)**2 * (x + 1.5)) / denom
    # Replace extremely large values or division by zero with nan to keep plot clean
    y[np.abs(denom) < 0.02] = np.nan
    y[np.abs(y) > 100] = np.nan
    return y

# -----------------------------------------------------------------------------
# Incremental Search
# -----------------------------------------------------------------------------
def scan_intervals(x_start, x_end, step_size):
    """Scan domain to find sign changes. Gracefully handles asymptotes by checking bounds."""
    xs = np.arange(x_start, x_end + step_size, step_size)
    intervals = []
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i+1]
        f1_val = f(x1)
        f2_val = f(x2)
        
        # Guard against inf/nan sign checks
        if not math.isfinite(f1_val) or not math.isfinite(f2_val):
            continue
            
        if f1_val * f2_val < 0:
            intervals.append((x1, x2))
    return intervals

# -----------------------------------------------------------------------------
# Bisection Solver
# -----------------------------------------------------------------------------
def run_bisection(xl, xu, root_index):
    """Run Bisection on bracket [xl, xu] and print iteration table."""
    print(f"\n[SCAN RESULT #{root_index}] Running Bisection on bracket [{xl:.4f}, {xu:.4f}]")
    print("-" * 72)
    print(f"{'Iter':<6} {'xl':>12} {'xu':>12} {'xr':>12} {'ea (%)':>12} {'f(xr)':>14}")
    print("-" * 72)
    
    xr_old = None
    xr = None
    converged = False
    
    for i in range(1, MAX_ITER + 1):
        xr = (xl + xu) / 2.0
        fxr = f(xr)
        
        if xr_old is not None:
            if abs(xr) > 1e-15:
                ea = abs((xr - xr_old) / xr) * 100.0
                ea_str = f"{ea:.6f}"
            else:
                ea = float('inf')
                ea_str = "---"
        else:
            ea = 100.0
            ea_str = "---"
            
        # Format function value for output (safeguard infinite values)
        fxr_str = f"{fxr:>14.6f}" if math.isfinite(fxr) else f"{fxr:>14}"
            
        print(f"{i:<6} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr_str}")
        
        if xr_old is not None and ea <= TOL:
            converged = True
            break
            
        if not math.isfinite(fxr):
            # If hit vertical asymptote exactly, stop to avoid arithmetic failure
            print(f"  [STOP] Hit a non-finite function value at x = {xr:.6f}.")
            break
            
        # Update bounds
        # Check sign change
        fl = f(xl)
        if not math.isfinite(fl):
            # Stop if lower bound has non-finite value
            break
            
        if fl * fxr < 0:
            xu = xr
        else:
            xl = xr
            
        xr_old = xr
        
    print("-" * 72)
    if converged:
        print(f"Result: Converged to x approx. {xr:.8f} (iterations: {i}, ea = {ea:.6f}%)")
    else:
        print(f"Result: Ended at x approx. {xr:.8f} (no convergence or hit discontinuity)")
    return xr

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("      SECTION-B2: PATHOLOGICAL / DISCONTINUOUS FUNCTIONS")
    print("=" * 80)
    
    # Task 1: Perform incremental scan
    intervals = scan_intervals(X_MIN, X_MAX, STEP)
    print(f"Discovered sign-change intervals in [{X_MIN}, {X_MAX}] (step size = {STEP}):")
    for idx, (xl, xu) in enumerate(intervals, 1):
        print(f"  Interval #{idx}: [{xl:.2f}, {xu:.2f}]")
        
    # Run Bisection on discovered intervals
    results = []
    for idx, (xl, xu) in enumerate(intervals, 1):
        root_approx = run_bisection(xl, xu, idx)
        results.append(root_approx)
        
    # Task 2: Plot the curve, paying close attention to x = 2.5 and x = 3.56
    x_plot = np.linspace(X_MIN, X_MAX, 2000)
    y_plot = f_np(x_plot)
    
    plt.figure(figsize=(10, 6.5))
    plt.plot(x_plot, y_plot, label=r"$f(x) = \frac{(x-2.5)^2(x+1.5)}{x-3.56}$", color="teal", linewidth=2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
    
    # Mark vertical asymptote at x = 3.56
    plt.axvline(3.56, color="red", linestyle=":", linewidth=2, label="Vertical Asymptote (x = 3.56)")
    
    # Highlight the roots and asymptotes
    # True Roots
    plt.plot(-1.5, 0, "go", markersize=8, label="True Root (x = -1.5, simple)")
    plt.plot(2.5, 0, "bo", markersize=8, label="True Double Root (x = 2.5, even multiplicity)")
    
    # Draw convergence points
    for idx, r in enumerate(results, 1):
        plt.plot(r, f_np(np.array([r]))[0], "ro", markersize=8, 
                 label="Algorithm Convergence Point" if idx == 1 else "")
                 
    plt.title("Section-B2: Rational & Pathological Root Behavior", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("x", fontsize=11)
    plt.ylabel("f(x)", fontsize=11)
    plt.ylim(-30, 30)  # Bound y-axis to see the roots clearly despite asymptote infinity
    plt.xlim(X_MIN, X_MAX)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    
    plot_path = "b2_pathological_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved plot to '{plot_path}'")
    
    # Task 3: Printed Conceptual Analysis
    print("""
--------------------------------------------------------------------------------
CONCEPTUAL ANALYSIS: Analysis of Pathological Behavior
--------------------------------------------------------------------------------
1. Why the True Root at x = 2.5 is completely missed by sign scanning:
   - The root at x = 2.5 is a double root (multiplicity 2) due to the factor (x - 2.5)^2.
   - The factor (x - 2.5)^2 is strictly non-negative (>= 0) for all real x.
   - Thus, as x crosses 2.5, the function does not cross the x-axis; it merely touches
     it and remains on the negative side (since x + 1.5 > 0 and x - 3.56 < 0).
   - Because f(x) does not change sign across x = 2.5, the sign-change test
     f(xl) * f(xu) < 0 is never satisfied for any interval surrounding 2.5.
     Hence, the scanner completely misses it.

2. Why the interval surrounding x = 3.56 yields a false/garbage root:
   - x = 3.56 is a vertical asymptote (discontinuity) where the denominator is 0.
   - As x approaches 3.56 from the left (x < 3.56), the denominator is negative,
     so f(x) -> -infinity.
   - As x approaches 3.56 from the right (x > 3.56), the denominator is positive,
     so f(x) -> +infinity.
   - This causes the function to jump signs from negative to positive.
   - The scanner detects this sign change (e.g. f(3.5) < 0 and f(3.6) > 0) and running
     Bisection converges to x = 3.56.
   - However, 3.56 is NOT a root since the function is undefined there (does not equal 0).
     It is a mathematical discontinuity (singularity), representing a "garbage root".
--------------------------------------------------------------------------------
""")

if __name__ == "__main__":
    main()
