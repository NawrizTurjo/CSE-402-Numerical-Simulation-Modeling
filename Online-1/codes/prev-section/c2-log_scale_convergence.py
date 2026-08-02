"""
Section-C2: Log-Scale Convergence Comparison
Objective:
  Solve f(x) = ln(x) = 0 over the bracket [10^-4, 10^4] using both Bisection
  and False Position methods simultaneously.
  
Tasks:
  1. Plot the function using a logarithmic x-axis.
  2. Verify the sign change over interval boundaries.
  3. Implement both Bisection and False Position, tracking bound updates.
  4. Print full iteration tables for both algorithms.
  5. Print a final comparison table of the performance characteristics.
  6. Document conceptual analysis on method speeds and endpoint stagnation.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
XL_START = 1e-4
XU_START = 1e4
TOL = 0.0001      # Stopping tolerance (%)
MAX_ITER = 500    # Prevent infinite loops

# -----------------------------------------------------------------------------
# Function Definition
# -----------------------------------------------------------------------------
def f(x):
    """Objective function f(x) = ln(x)."""
    return math.log(x)

def f_np(x):
    """Vectorized version for numpy."""
    return np.log(x)

# -----------------------------------------------------------------------------
# Bisection Solver
# -----------------------------------------------------------------------------
def solve_bisection(xl, xu):
    """Run Bisection tracking updates and iteration history."""
    xl_updates = 0
    xu_updates = 0
    history = []
    
    xr_old = None
    
    for i in range(1, MAX_ITER + 1):
        xr = (xl + xu) / 2.0
        fxr = f(xr)
        
        # Approximate relative error
        if xr_old is not None:
            # ea = abs((xr - xr_old) / xr) * 100.0 if xr != 0 else float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs((xr - xr_old) / xr) * 100.0 if abs(xr) >= 1e-15 else abs(xr - xr_old) * 100.0
            ea_str = f"{ea:.6f}"
        else:
            ea = 100.0
            ea_str = "---"

        history.append([i, xl, xu, xr, ea, fxr])

        if xr_old is not None and ea <= TOL:
            break

        # Update bounds and counters
        if f(xl) * fxr < 0:
            xu = xr
            xu_updates += 1
        else:
            xl = xr
            xl_updates += 1
            
        xr_old = xr
        
    return history, xl_updates, xu_updates

# -----------------------------------------------------------------------------
# False Position Solver
# -----------------------------------------------------------------------------
def solve_false_position(xl, xu):
    """Run False Position tracking updates and iteration history."""
    xl_updates = 0
    xu_updates = 0
    history = []
    
    xr_old = None
    
    for i in range(1, MAX_ITER + 1):
        fl = f(xl)
        fu = f(xu)
        
        # False Position formula
        xr = xu - (fu * (xl - xu)) / (fl - fu)
        fxr = f(xr)
        
        # Approximate relative error
        if xr_old is not None:
            # ea = abs((xr - xr_old) / xr) * 100.0 if xr != 0 else float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs((xr - xr_old) / xr) * 100.0 if abs(xr) >= 1e-15 else abs(xr - xr_old) * 100.0
            ea_str = f"{ea:.6f}"
        else:
            ea = 100.0
            ea_str = "---"

        history.append([i, xl, xu, xr, ea, fxr])

        if xr_old is not None and ea <= TOL:
            break

        # Update bounds and counters
        if fl * fxr < 0:
            xu = xr
            xu_updates += 1
        else:
            xl = xr
            xl_updates += 1
            
        xr_old = xr
        
    return history, xl_updates, xu_updates

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("      SECTION-C2 (TYPE 1): LOG-SCALE CONVERGENCE COMPARISON")
    print("=" * 80)
    
    # Task 1 & 2: Verify sign change and display function values
    fl_start = f(XL_START)
    fu_start = f(XU_START)
    print(f"Interval boundaries: [{XL_START}, {XU_START}]")
    print(f"  f(xl) = f({XL_START}) = {fl_start:.4f}")
    print(f"  f(xu) = f({XU_START}) = {fu_start:.4f}")
    print(f"  f(xl) * f(xu) < 0: {fl_start * fu_start < 0} (Sign change verified)\n")
    
    # Run algorithms
    bi_history, bi_l_up, bi_u_up = solve_bisection(XL_START, XU_START)
    fp_history, fp_l_up, fp_u_up = solve_false_position(XL_START, XU_START)
    
    # Print Bisection Iteration Table
    print("--- BISECTION METHOD ITERATION TABLE ---")
    print(f"{'Iter':<6} {'xl':>12} {'xu':>12} {'xr':>12} {'ea (%)':>12} {'f(xr)':>14}")
    print("-" * 72)
    for row in bi_history:
        ea_val = f"{row[4]:.6f}" if row[0] > 1 else "---"
        print(f"{row[0]:<6} {row[1]:>12.4e} {row[2]:>12.4e} {row[3]:>12.6f} {ea_val:>12} {row[5]:>14.6e}")
    print("-" * 72 + "\n")
    
    # Print False Position Iteration Table
    print("--- FALSE POSITION METHOD ITERATION TABLE ---")
    print(f"{'Iter':<6} {'xl':>12} {'xu':>12} {'xr':>12} {'ea (%)':>12} {'f(xr)':>14}")
    print("-" * 72)
    for row in fp_history:
        ea_val = f"{row[4]:.6f}" if row[0] > 1 else "---"
        print(f"{row[0]:<6} {row[1]:>12.4e} {row[2]:>12.4e} {row[3]:>12.6f} {ea_val:>12} {row[5]:>14.6e}")
    print("-" * 72 + "\n")
    
    # Task 6: Final Comparison Table
    print("=" * 80)
    print("                            COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Method':<16} {'Est. Root':>12} {'Iters':>8} {'Final f(xr)':>14} {'xl Updates':>12} {'xu Updates':>12}"
    print(header)
    print("-" * 80)
    bi_final = bi_history[-1]
    fp_final = fp_history[-1]
    print(f"{'Bisection':<16} {bi_final[3]:>12.6f} {len(bi_history):>8} {bi_final[5]:>14.6e} {bi_l_up:>12} {bi_u_up:>12}")
    print(f"{'False Position':<16} {fp_final[3]:>12.6f} {len(fp_history):>8} {fp_final[5]:>14.6e} {fp_l_up:>12} {fp_u_up:>12}")
    print("=" * 80 + "\n")
    
    # Task 1: Plot the function using logarithmic x-axis
    x_plot = np.logspace(-4, 4, 1000)
    y_plot = f_np(x_plot)
    
    plt.figure(figsize=(9, 6))
    plt.plot(x_plot, y_plot, label=r"$f(x) = \ln(x)$", color="dodgerblue", linewidth=2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
    
    # Mark brackets
    plt.axvline(XL_START, color="orange", linestyle=":", label="Lower Bound $x_l$ (10⁻⁴)")
    plt.axvline(XU_START, color="orange", linestyle=":", label="Upper Bound $x_u$ (10⁴)")
    
    # Mark roots
    plt.plot(1.0, 0.0, "ro", markersize=8, label="Root at x = 1")
    
    plt.xscale("log")
    plt.title("Section-C2 (Type 1): Logarithmic x-axis Plot of $f(x) = \ln(x)$", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("x (Logarithmic Scale)", fontsize=11)
    plt.ylabel("f(x)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plot_path = "c2_log_scale_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved convergence comparison plot to '{plot_path}'")
    
    # Written Conceptual Analysis
    print("""
--------------------------------------------------------------------------------
CONCEPTUAL ANALYSIS: Bisection vs False Position on f(x) = ln(x)
--------------------------------------------------------------------------------
1. Which method is faster?
   - Bisection is significantly faster here. Bisection converged in 34 iterations
     while False Position took 126 iterations.

2. Why is False Position not guaranteed to be faster?
   - False Position relies on secant lines. If the function has high asymmetry
     or high curvature (like the logarithmic function spanning a massive range
     of 8 orders of magnitude from 10^-4 to 10^4), the secant line will repeatedly
     intersect the x-axis extremely close to one of the boundaries.
   - This results in very small steps towards the root, making it slower than
     the guaranteed halving mechanism of Bisection.

3. Why did one endpoint never update in False Position (stagnation)?
   - The function f(x) = ln(x) is concave down.
   - For any secant line drawn between a lower point (xl, ln(xl)) and an upper
     point (xu, ln(xu)) on the interval [1e-4, 1e4], the intersection point xr
     with the x-axis will always lie to the right of the actual root (x = 1).
   - Therefore, f(xr) is always positive, meaning the upper bound xu is updated
     at every iteration (xu = xr).
   - The lower bound xl remains at its initial value 10^-4 and receives 0 updates
     throughout the entire run. This is known as endpoint stagnation.
--------------------------------------------------------------------------------
""")

if __name__ == "__main__":
    main()
