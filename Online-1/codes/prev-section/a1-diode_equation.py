"""
Section-A1: Diode Equation Solver (Newton-Raphson)
Objective:
  Find the operating voltage V that satisfies the diode characteristic equation:
  f(V) = 1e-12 * (e^(V / (n * VT)) - 1) + V / R - IL = 0

Parameters:
  - Initial Guess (V_0): 0.65 V
  - Ideality Factor (n): 1.8
  - Resistance (R): 0.5 kOhm = 500 Ohm
  - Thermal Voltage (VT): 0.02585 V
  - Light Current (IL): 0.2 mA = 0.0002 A
  - Stopping Criterion: |ea| <= 0.0001%
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
IS = 1e-12        # Reverse saturation current (A)
N = 1.8          # Ideality factor
VT = 0.02585     # Thermal voltage (V)
R = 500.0        # Resistance (Ohms)
IL = 0.0002      # Light current (A)
V_INIT = 0.65    # Initial guess (V)
TOL = 0.0001     # Stopping tolerance (%)
MAX_ITER = 100   # Guard rail for iterations

# -----------------------------------------------------------------------------
# Diode Function and Derivative Definition
# -----------------------------------------------------------------------------
def f(V):
    """Diode characteristic equation function."""
    return IS * (math.exp(V / (N * VT)) - 1.0) + V / R - IL

def df(V):
    """Derivative of the diode equation with respect to V."""
    return (IS / (N * VT)) * math.exp(V / (N * VT)) + 1.0 / R

# -----------------------------------------------------------------------------
# Newton-Raphson Execution
# -----------------------------------------------------------------------------
def solve_diode():
    print("=" * 80)
    print("          SECTION-A1: DIODE EQUATION (NEWTON-RAPHSON)")
    print("=" * 80)
    
    # Header format matches exact requirement
    header = f"{'Iter':<6} {'V_i':>12} {'f(V_i)':>14} {'f_prime(V_i)':>14} {'V_{i+1}':>12} {'ea (%)':>12}"
    print(header)
    print("-" * 80)
    
    V_i = V_INIT
    converged = False
    
    for i in range(1, MAX_ITER + 1):
        fval = f(V_i)
        dfval = df(V_i)
        
        if abs(dfval) < 1e-15:
            print(f"Error: Derivative too close to zero (df = {dfval}) at iteration {i}.")
            break
            
        V_next = V_i - fval / dfval
        
        # Approximate relative error calculation
        if abs(V_next) > 1e-15:
            ea = abs((V_next - V_i) / V_next) * 100.0
            ea_str = f"{ea:.6f}"
        else:
            ea = float('inf')
            ea_str = "---"
            
        # Print tabular output line
        print(f"{i:<6} {V_i:>12.6f} {fval:>14.6e} {dfval:>14.6e} {V_next:>12.6f} {ea_str:>12}")
        
        if ea <= TOL:
            V_i = V_next
            converged = True
            break
            
        V_i = V_next
        
    print("-" * 80)
    if converged:
        print(f"[SUCCESS] Converged to root V = {V_i:.8f} V in {i} iterations.")
        print(f"Stopping criterion |ea| <= {TOL}% met.")
    else:
        print("[FAILURE] Method failed to converge within maximum iteration limit.")
    print("=" * 80)
    
    # -----------------------------------------------------------------------------
    # Visualization & Plotting
    # -----------------------------------------------------------------------------
    V_vals = np.linspace(0.0, 0.9, 500)
    f_vals = [IS * (math.exp(v / (N * VT)) - 1.0) + v / R - IL for v in V_vals]
    
    plt.figure(figsize=(9, 6))
    plt.plot(V_vals, f_vals, label=r"$f(V) = I_s(e^{V/n V_T} - 1) + \frac{V}{R} - I_L$", color="darkorange", linewidth=2.5)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    
    if converged:
        plt.plot(V_i, f(V_i), "ro", markersize=8, label=f"Operating Point (V ≈ {V_i:.5f} V)")
        
    plt.title("Diode Characteristic Equation $f(V)$ vs Voltage $V$", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Voltage $V$ (Volts)", fontsize=11)
    plt.ylabel("$f(V)$", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = "a1_diode_equation.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved successfully as '{plot_path}'.\n")

if __name__ == "__main__":
    solve_diode()
