# -*- coding: utf-8 -*-
"""
=============================================================================
  CSE-402 Numerical Methods: Previous Batch Online Exam Solutions
  Source: prev-batch-all.md
=============================================================================

This file structures the three previous batch online exam questions (A2, B2, A1)
with full annotations, detailed formatting, and production-ready matplotlib 
plotting functions.

PROBLEMS INCLUDED:
  1. Problem A2: Spherical Oil Tank Dipstick Design (Newton-Raphson)
  2. Problem B2: Chemical Firm Daily Break-Even Point (Newton-Raphson)
  3. Problem A1: Cubic Equation Solver (False Position Method)
=============================================================================
"""

import sys
import io
import numpy as np
import matplotlib.pyplot as plt

# Ensure UTF-8 output to avoid Windows console errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 1: Spherical Oil Tank Dipstick Design (online_A2.pdf)
# ─────────────────────────────────────────────────────────────────────────────

def run_problem_A2_dipstick():
    """
    Spherical tank, diameter = 8 ft (radius r = 4 ft).
    Find wet height h for target volume V = 5 ft^3.
    
    Formula: V = pi * h^2 * (3*r - h) / 3
    Non-linear Eq: f(h) = pi * h^3 - 12 * pi * h^2 + 15 = 0
    Derivative: df(h) = 3 * pi * h^2 - 24 * pi * h
    """
    print("\n" + "═"*80)
    print(" [PROBLEM 1]: Spherical Oil Tank Dipstick Design (online_A2.pdf)")
    print(" Method: Newton-Raphson | Tolerance: \u03b5_s \u2264 0.05%")
    print("═"*80)

    # 1. Define objective function and analytical derivative
    def f(h):
        return np.pi * h**3 - 12 * np.pi * h**2 + 15

    def df(h):
        return 3 * np.pi * h**2 - 24 * np.pi * h

    # 2. Newton-Raphson configuration
    h_old = 0.5  # Initial guess based on sign change in [0, 1]
    tol = 0.05   # 0.05%
    max_iter = 20

    print(f"\nInitial Guess h_0 = {h_old} ft")
    print(f"{'Iter':<6}{'Estimate (h_i)':<18}{'f(h_i)':<15}{'f\u2032(h_i)':<15}{'Error \u03b5_a (%)':<15}")
    print("─"*70)

    for i in range(1, max_iter + 1):
        fval = f(h_old)
        dfval = df(h_old)
        
        # Guard against division by zero (horizontal tangent)
        if abs(dfval) < 1e-12:
            print(f"Error: Derivative too close to zero at h = {h_old}. Aborting.")
            break
            
        h_new = h_old - fval / dfval
        ea = abs((h_new - h_old) / h_new) * 100
        
        print(f"{i:<6}{h_new:<18.6f}{fval:<15.6f}{dfval:<15.6f}{ea:<15.4f}%")
        
        if ea <= tol:
            print("─"*70)
            print(f"\u2714 Converged to root h = {h_new:.6f} ft in {i} iterations.\n")
            break
        h_old = h_new

    # 3. Visualization
    h_vals = np.linspace(0, 2, 500)
    plt.figure(figsize=(7, 4.5))
    plt.plot(h_vals, f(h_vals), label='$f(h) = \\pi h^3 - 12\\pi h^2 + 15$', color='blue', linewidth=2)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    
    # Mark the converged root and initial guess
    plt.scatter([h_new], [0], color='darkred', marker='x', s=100, zorder=5, label=f'Root h \u2248 {h_new:.4f} ft')
    plt.scatter([0.5], [f(0.5)], color='orange', marker='o', s=80, zorder=5, label='Initial Guess h₀ = 0.5')
    
    plt.title('online_A2: Spherical Tank Dipstick Design', fontsize=12)
    plt.xlabel('Height $h$ (ft)', fontsize=10)
    plt.ylabel('$f(h)$', fontsize=10)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('online_A2_dipstick_plot.png', dpi=150)
    plt.close()
    print("Saved plot as 'online_A2_dipstick_plot.png'")


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 2: Chemical Firm Daily Break-Even Point (online_B2.pdf)
# ─────────────────────────────────────────────────────────────────────────────

def run_problem_B2_breakeven():
    """
    Cost C(q) = 1000 + 2q + 3q^(2/3)
    Revenue R(q) = 300q
    Break-even: R(x) - C(x) = 0
    Non-linear Eq: f(x) = 298x - 3x^(2/3) - 1000 = 0
    Derivative: df(x) = 298 - 2x^(-1/3)
    """
    print("\n" + "═"*80)
    print(" [PROBLEM 2]: Chemical Firm Daily Break-Even Point (online_B2.pdf)")
    print(" Method: Newton-Raphson | Tolerance: \u03b5_s \u2264 0.05%")
    print("═"*80)

    # 1. Define objective function and derivative
    def f(x):
        return 298 * x - 3 * (x**(2/3)) - 1000

    def df(x):
        return 298 - 2 * (x**(-1/3))

    # 2. Newton-Raphson Configuration
    x_old = 4.0   # Initial guess based on bracket search [1.0, 4.0]
    tol = 0.05    # 0.05%
    max_iter = 20

    print(f"\nInitial Guess x_0 = {x_old} grams")
    print(f"{'Iter':<6}{'Estimate (x_i)':<18}{'f(x_i)':<15}{'f\u2032(x_i)':<15}{'Error \u03b5_a (%)':<15}")
    print("─"*70)

    for i in range(1, max_iter + 1):
        fval = f(x_old)
        dfval = df(x_old)
        
        if abs(dfval) < 1e-12:
            print(f"Error: Derivative too close to zero at x = {x_old}. Aborting.")
            break
            
        x_new = x_old - fval / dfval
        ea = abs((x_new - x_old) / x_new) * 100
        
        print(f"{i:<6}{x_new:<18.6f}{fval:<15.6f}{dfval:<15.6f}{ea:<15.4f}%")
        
        if ea <= tol:
            print("─"*70)
            print(f"\u2714 Converged to break-even point x = {x_new:.6f} grams in {i} iterations.\n")
            break
        x_old = x_new

    # 3. Visualization
    x_vals = np.linspace(1, 10, 500)
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_vals, f(x_vals), label='$f(x) = 298x - 3x^{2/3} - 1000$', color='green', linewidth=2)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    
    # Mark points
    plt.scatter([x_new], [0], color='darkred', marker='x', s=100, zorder=5, label=f'Root x \u2248 {x_new:.4f} g')
    plt.scatter([4.0], [f(4.0)], color='orange', marker='o', s=80, zorder=5, label='Initial Guess x₀ = 4.0')
    
    plt.title('online_B2: Chemical Firm Break-Even Point', fontsize=12)
    plt.xlabel('Quantity $x$ (grams)', fontsize=10)
    plt.ylabel('$f(x)$', fontsize=10)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('online_B2_breakeven_plot.png', dpi=150)
    plt.close()
    print("Saved plot as 'online_B2_breakeven_plot.png'")


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 3: Cubic Equation Solver (online_A1.pdf)
# ─────────────────────────────────────────────────────────────────────────────

def run_problem_A1_cubic():
    """
    Cubic: f(x) = x^3 - x - 1 = 0
    Method: False Position (Regula Falsi)
    Interval: [1.0, 2.0]
    """
    print("\n" + "═"*80)
    print(" [PROBLEM 3]: Cubic Equation Solver (online_A1.pdf)")
    print(" Method: False Position | Tolerance: \u03b5_s \u2264 0.05%")
    print("═"*80)

    # 1. Define objective function
    def f(x):
        return x**3 - x - 1

    # 2. Configuration
    xl = 1.0  # f(1) = -1 < 0
    xu = 2.0  # f(2) = 5 > 0
    tol = 0.05  # 0.05%
    max_iter = 50

    print(f"\nInitial Bracket: [{xl}, {xu}]")
    print(f"{'Iter':<6}{'x_L':<10}{'x_U':<10}{'x_r':<12}{'Error \u03b5_a (%)':<15}{'f(x_r)':<12}")
    print("─"*70)

    xr_old = None

    for idx in range(1, max_iter + 1):
        fl, fu = f(xl), f(xu)
        
        # False Position formula
        xr = xu - (fu * (xl - xu)) / (fl - fu)
        fxr = f(xr)
        
        # Calculate error (skip first iteration)
        if xr_old is not None:
            ea = abs((xr - xr_old) / xr) * 100
            error_str = f"{ea:.4f}%"
        else:
            ea = float('inf')
            error_str = "Initial"
            
        print(f"{idx:<6}{xl:<10.4f}{xu:<10.4f}{xr:<12.6f}{error_str:<15}{fxr:<12.4e}")
        
        if ea <= tol:
            print("─"*70)
            print(f"\u2714 Converged to real root x = {xr:.6f} in {idx} iterations.\n")
            break
            
        # Update bounds using sign check
        if fl * fxr < 0:
            xu = xr
        else:
            xl = xr
            
        xr_old = xr

    # 3. Visualization
    x_vals = np.linspace(0, 2.5, 500)
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_vals, f(x_vals), label='$f(x) = x^3 - x - 1$', color='crimson', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvspan(1.0, 2.0, color='yellow', alpha=0.2, label='Initial Bracket [1, 2]')
    
    plt.scatter([xr], [0], color='darkred', marker='x', s=100, zorder=5, label=f'Root x \u2248 {xr:.4f}')
    
    plt.title('online_A1: Cubic Equation Solver', fontsize=12)
    plt.xlabel('x', fontsize=10)
    plt.ylabel('f(x)', fontsize=10)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('online_A1_cubic_plot.png', dpi=150)
    plt.close()
    print("Saved plot as 'online_A1_cubic_plot.png'")


# ─────────────────────────────────────────────────────────────────────────────
# Main execution gate
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run_problem_A2_dipstick()
    run_problem_B2_breakeven()
    run_problem_A1_cubic()
    print("All tasks completed.")
