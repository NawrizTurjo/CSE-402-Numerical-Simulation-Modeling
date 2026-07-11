# -*- coding: utf-8 -*-
"""
=============================================================================
  TOPIC 1 Helper: Truncation Error vs. Round-off Error Tradeoff Simulation
=============================================================================

This script demonstrates the classic numerical error tradeoff when estimating 
derivatives using finite differences.

THEORY:
  1. Truncation Error: Arises from dropping higher-order terms in the Taylor
     series expansion. For forward difference, this error is O(h) which decreases
     as step size h shrinks.
  2. Round-off Error: Arises from the limited 64-bit floating point precision 
     representation of numbers. When h is extremely small, subtracting f(x+h) - f(x)
     leads to severe subtractive cancellation, and dividing by h amplifies this noise.
  3. Total Error: Sum of Truncation + Round-off. It creates a classic V-shaped curve
     on a log-log plot. The optimal step size h occurs at the minimum of the V.

TEST SETUP:
  f(x) = exp(x), evaluated at x_target = 1.0. 
  True derivative is exp(1.0) ~= 2.718281828459045.
=============================================================================
"""

import sys
import io
import numpy as np
import matplotlib.pyplot as plt

# Ensure UTF-8 output to avoid Windows console errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def f_test(x): 
    return np.exp(x)

# True derivative value at x = 1.0
true_deriv = np.exp(1.0)
x_target = 1.0

# Generate a wide range of step sizes from 10^0 down to 10^-20
h_values = np.logspace(0, -20, 100)
errors = []

print("=== Running Error Tradeoff Simulation ===")
print(f"Estimating derivative of e^x at x = {x_target}")
print(f"True Derivative: {true_deriv:.16f}")
print("─"*75)
print(f"{'Step Size h':<15}{'Approximation':<25}{'Absolute Error':<20}")
print("─"*75)

# Run simulation and print representative values
for h in h_values:
    # Forward difference approximation: f'(x) ~= (f(x+h) - f(x)) / h
    approx_deriv = (f_test(x_target + h) - f_test(x_target)) / h
    abs_error = abs(true_deriv - approx_deriv)
    errors.append(abs_error)
    
    # Print representative steps (powers of 10)
    exponent = np.log10(h)
    if abs(exponent - round(exponent)) < 1e-9 and round(exponent) % 2 == 0 and round(exponent) >= -20:
        print(f"10^{int(round(exponent)):<11.0f}{approx_deriv:<25.16f}{abs_error:<20.4e}")

# Identify optimal step size h
min_error_idx = np.argmin(errors)
optimal_h = h_values[min_error_idx]
optimal_error = errors[min_error_idx]

print("─"*75)
print(f"Optimal Step Size (h)  : {optimal_h:.2e}")
print(f"Minimum Absolute Error : {optimal_error:.4e}")
print("─"*75)

# Generate the classic V-shaped Error Tradeoff Curve
plt.figure(figsize=(8, 5))
plt.loglog(h_values, errors, color='darkblue', linewidth=2.5, label='Total Numerical Error')
plt.axvline(optimal_h, color='green', linestyle=':', label=f'Optimal h \u2248 {optimal_h:.1e}')

plt.xlabel('Step Size $h$ (Log Scale)')
plt.ylabel('Absolute Error (Log Scale)')
plt.title('Truncation vs. Round-off Error Tradeoff Curve')
plt.gca().invert_xaxis()  # Invert x-axis to show large h down to tiny h

# Text annotations for the exam paper explanation
plt.text(1e-2, 1e-2, '\u2190 Truncation Error Dominated\n    (h is too large, Taylor series cut too early)', color='darkred', fontsize=9)
plt.text(1e-18, 1e-2, 'Round-off Error Dominated \u2192\n(h is too small, subtractive cancellation noise)', color='purple', fontsize=9)

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc='lower left')
plt.tight_layout()

# Save the plot relative to script directory
import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error_tradeoff_curve.png')
plt.savefig(save_path, dpi=150)
print(f"[SUCCESS] Saved error tradeoff plot as '{os.path.basename(save_path)}'.")
plt.close()
