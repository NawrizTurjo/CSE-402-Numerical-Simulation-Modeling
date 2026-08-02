"""
Section-B1: Multi-Root Scanner (Bisection)
Objective:
  Find all real roots for the transcendental equation:
  f(x) = 0.6 * ln(x+1) - C * sin(1.7x) - 0.08x^2 - 0.08 = 0

Parameters:
  - Domain: 0 <= x <= 10
  - Incremental search step size: dx = 0.1
  - Coefficient (C): 1.0 (default, easily adjustable)
  - Stopping Criterion: |ea| <= 0.0001%
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
C = 1.0          # Equation coefficient
TOL = 0.0001     # Stopping tolerance (%)
STEP = 0.1       # Incremental search step size
X_MIN = 0.0      # Domain start
X_MAX = 10.0     # Domain end
MAX_ITER = 100   # Guard rail for bisection

# -----------------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------------
def f(x):
    """The transcendental equation for point evaluation (using math module)."""
    return 0.6 * math.log(x + 1.0) - C * math.sin(1.7 * x) - 0.08 * x**2 - 0.08

def f_np(x):
    """Vectorized version of the function for numpy plotting."""
    return 0.6 * np.log(x + 1.0) - C * np.sin(1.7 * x) - 0.08 * x**2 - 0.08

# -----------------------------------------------------------------------------
# Incremental Search for Bounding Intervals
# -----------------------------------------------------------------------------
def scan_intervals(x_start, x_end, step_size):
    """Scan the domain to find all intervals where a sign change occurs."""
    xs = np.arange(x_start, x_end + step_size, step_size)
    intervals = []
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i+1]
        try:
            f1_val = f(x1)
            f2_val = f(x2)
            if f1_val * f2_val < 0:
                intervals.append((x1, x2))
        except ValueError:
            # Handle potential log or domain errors gracefully
            continue
    return intervals

# -----------------------------------------------------------------------------
# Bisection Solver
# -----------------------------------------------------------------------------
def run_bisection(xl, xu, root_index):
    """Run Bisection on a specific bracket [xl, xu] and print the iteration table."""
    print(f"\n[ROOT #{root_index}] Running Bisection on bracket [{xl:.4f}, {xu:.4f}]")
    print("-" * 72)
    print(f"{'Iter':<6} {'xl':>12} {'xu':>12} {'xr':>12} {'ea (%)':>12} {'f(xr)':>14}")
    print("-" * 72)
    
    xr_old = None
    xr = None
    converged = False
    
    for i in range(1, MAX_ITER + 1):
        xr = (xl + xu) / 2.0
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
            
        print(f"{i:<6} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>14.6f}")
        
        # Stopping criteria check
        if xr_old is not None and ea <= TOL:
            converged = True
            break
            
        # Update bracket
        if f(xl) * fxr < 0:
            xu = xr
        else:
            xl = xr
            
        xr_old = xr
        
    print("-" * 72)
    if converged:
        print(f"Result: Root found at x approx. {xr:.8f} (iterations: {i}, ea = {ea:.6f}%)")
    else:
        print(f"Result: Ended at x approx. {xr:.8f} without satisfying strict tolerance.")
    return xr

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("          SECTION-B1: MULTI-ROOT SCANNER (BISECTION)")
    print("=" * 80)
    
    # Task 1: Plot the function across the domain
    x_plot = np.linspace(X_MIN, X_MAX, 1000)
    plt.figure(figsize=(9, 6))
    plt.plot(x_plot, f_np(x_plot), label=f"f(x), C={C}", color="purple", linewidth=2.0)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
    plt.title("Section-B1: Visualization of $f(x)$ over $[0, 10]$", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("x", fontsize=11)
    plt.ylabel("f(x)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Task 2: Programmatically isolate intervals
    intervals = scan_intervals(X_MIN, X_MAX, STEP)
    print(f"Isolated sign-change intervals (step size = {STEP}):")
    for idx, (xl, xu) in enumerate(intervals, 1):
        print(f"  Interval #{idx}: [{xl:.1f}, {xu:.1f}]")
        plt.axvspan(xl, xu, color="yellow", alpha=0.3, label="Sign-change Interval" if idx == 1 else "")
        
    # Task 3 & 4: Run Bisection on each interval
    discovered_roots = []
    for idx, (xl, xu) in enumerate(intervals, 1):
        root = run_bisection(xl, xu, idx)
        discovered_roots.append(root)
        plt.plot(root, f_np(root), "ro", markersize=7, label="Discovered Root" if idx == 1 else "")
        
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot_path = "b1_multi_root_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved visualization plot to '{plot_path}'")
    
    print("\n" + "=" * 80)
    print("Discovered Roots Summary:")
    for idx, root in enumerate(discovered_roots, 1):
        print(f"  Root #{idx}: x approx. {root:.8f}")
    print("=" * 80)
    
    # Conceptual Analysis Explanation Output
    print("""
--------------------------------------------------------------------------------
CONCEPTUAL ANALYSIS: Why a single global Bisection run over [0, 10] fails:
--------------------------------------------------------------------------------
1. Multiple Roots Cancellation:
   The Bisection method relies on the Intermediate Value Theorem. It requires
   the function to have opposite signs at the interval boundaries (f(xl)*f(xu) < 0).
   If there are multiple roots within [0, 10], their sign-crossings can cancel
   each other out. In this case, f(0) is negative and f(10) is also negative,
   so f(0) * f(10) > 0.
   
2. Initialization Failure:
   Because f(0) * f(10) > 0, Bisection cannot even initialize on [0, 10] as
   there is no guaranteed sign change.
   
3. Singularity of Root Discovery:
   Even if the sign change test passed (e.g., if there were an odd number of roots),
   Bisection is mathematically designed to converge to only ONE root.
   The remaining roots would be completely missed.
--------------------------------------------------------------------------------
""")

if __name__ == "__main__":
    main()
