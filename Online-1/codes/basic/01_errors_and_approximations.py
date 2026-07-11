# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
=============================================================================
  TOPIC 1: Approximations, Round-off Errors & Truncation Errors
=============================================================================

KEY CONCEPTS:
  - Round-off error   : finite precision storage of numbers in computer memory
  - Truncation error  : cutting off an infinite process (Taylor series) early
  - Machine epsilon   : smallest number epsilon such that 1.0 + epsilon != 1.0 in float64
  - Significant figures & Scarborough criterion for stopping tolerance

FORMULAS TO MEMORIZE:
  True Error:         E_t  = true_value - approx_value
  True Rel. Error:    epsilon_t% = |true - approx| / |true| x 100
  Approx Rel. Error:  ea% = |x_new - x_old| / |x_new| x 100   ← used in exams!
  Scarborough:        es% = (0.5 x 10^(2-n))%  for n significant figures
=============================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# PART A: ERROR FORMULAS
# ─────────────────────────────────────────────────────────────────────────────

def true_error(true_val, approx_val):
    """Absolute true error. Sign matters (positive = over-estimated)."""
    return true_val - approx_val

def true_relative_error_pct(true_val, approx_val):
    """True relative error as a percentage. Needs the real answer (not available in practice)."""
    return abs(true_val - approx_val) / abs(true_val) * 100.0

def approx_relative_error_pct(x_new, x_old):
    """
    Approximate relative error - the PRACTICAL stopping criterion.
    We don't know the true value, so we compare consecutive iterates instead.
    Stop iterating when this falls below the tolerance.
    """
    if x_new == 0:
        return float('inf')              # avoid division by zero
    return abs((x_new - x_old) / x_new) * 100.0

def scarborough_tolerance(n_sig_figs):
    """
    Scarborough criterion: the tolerance (%) that guarantees at least
    n_sig_figs correct significant figures in the result.
    Formula: es = 0.5 x 10^(2-n)  (%)
    Example: n=3 → es = 0.05%,  n=4 → es = 0.005%
    """
    return 0.5 * 10 ** (2 - n_sig_figs)


# ─────────────────────────────────────────────────────────────────────────────
# PART B: MACHINE EPSILON
# ─────────────────────────────────────────────────────────────────────────────

def compute_machine_epsilon():
    """
    Keep halving until (1 + epsilon/2) == 1 in floating-point.
    This is the fundamental precision limit of a 64-bit float (~2.22e-16).
    """
    eps = 1.0
    while (1.0 + eps / 2.0) != 1.0:
        eps /= 2.0
    return eps


# ─────────────────────────────────────────────────────────────────────────────
# PART C: ROUND-OFF ERROR DEMO
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("PART A & B: Scarborough Tolerance & Machine Epsilon")
print("=" * 65)

print(f"\nMachine epsilon (computed) : {compute_machine_epsilon():.3e}")
print(f"Machine epsilon (numpy)    : {np.finfo(float).eps:.3e}")

print("\nScarborough tolerance table:")
print(f"  {'Sig. Figs':>10}  {'es (%)':>12}")
for n in range(1, 7):
    print(f"  {n:>10}  {scarborough_tolerance(n):>12.6f}%")

print("\n" + "=" * 65)
print("PART C: Round-off Error")
print("=" * 65)

# 1/3 cannot be represented exactly in binary floating-point
exact_third = 1 / 3
rounded     = round(exact_third, 5)
ro_error    = abs(exact_third - rounded)

print(f"\nStoring 1/3 with 5 decimal digits:")
print(f"  Exact  (1/3)    = {exact_third}")
print(f"  Stored (rounded)= {rounded}")
print(f"  Round-off error = {ro_error:.2e}")

# Subtractive cancellation: a classic round-off hazard
print("\nSubtractive cancellation example:  sqrt(x+1) - sqrt(x)")
print("  This loses significant digits for large x. Rearrange to 1/(sqrt(x+1)+sqrt(x)).")
print(f"  {'x':>12}  {'naive':>18}  {'stable':>18}  {'diff':>12}")
for x in [1e0, 1e4, 1e8, 1e12]:
    naive  = math.sqrt(x + 1) - math.sqrt(x)
    stable = 1.0 / (math.sqrt(x + 1) + math.sqrt(x))
    print(f"  {x:>12.0e}  {naive:>18.12f}  {stable:>18.12f}  {abs(naive-stable):>12.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# PART D: TRUNCATION ERROR - Taylor Series for e^x
# ─────────────────────────────────────────────────────────────────────────────

def taylor_exp(x, n_terms):
    """
    Approximates e^x with `n_terms` terms of the Taylor series:
        e^x = 1 + x + x^2/2! + x^3/3! + ...
    The remainder (truncation error) gets smaller as n_terms increases.
    """
    total = 0.0
    term  = 1.0              # first term = x^0 / 0! = 1
    for k in range(n_terms):
        total += term
        term  *= x / (k + 1) # recurrence: term_{k+1} = term_k * x / (k+1)
    return total

print("\n" + "=" * 65)
print("PART D: Truncation Error (Taylor series for e^x at x=1)")
print("=" * 65)
print(f"\nTrue value e^1 = {math.e:.10f}")
print(f"\n  {'n_terms':>8}  {'approx':>14}  {'E_t':>14}  {'epsilon_t (%)':>12}")
for n in range(1, 11):
    approx = taylor_exp(1.0, n)
    et     = true_error(math.e, approx)
    eps_t  = true_relative_error_pct(math.e, approx)
    print(f"  {n:>8}  {approx:>14.8f}  {et:>14.8f}  {eps_t:>12.6f}")

print("\n▶ Truncation error shrinks with each additional term.")
print("  Round-off error is different - it's the fixed hardware limit (~1e-16).")


# ─────────────────────────────────────────────────────────────────────────────
# PART E: TRUNCATION ERROR IN DIFFERENTIATION & INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("PART E: Truncation Error in Numerical Differentiation")
print("  Forward-difference: f'(x) approx. [f(x+h) - f(x)] / h")
print("  True: f(x) = sin(x), f'(x) = cos(x) at x = 1.0")
print("=" * 65)

x0 = 1.0
true_deriv = math.cos(x0)
print(f"\n  True f'({x0}) = cos({x0}) = {true_deriv:.10f}\n")
print(f"  {'h':>10}  {'approx':>14}  {'trunc. error':>14}")
for h in [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]:
    approx = (math.sin(x0 + h) - math.sin(x0)) / h
    err    = abs(true_deriv - approx)
    print(f"  {h:>10}  {approx:>14.10f}  {err:>14.10f}")
print("\n▶ Smaller h → smaller truncation error (but eventually round-off dominates)")


# ─────────────────────────────────────────────────────────────────────────────
# PART F: VISUALIZATION - Taylor Series Convergence
# ─────────────────────────────────────────────────────────────────────────────

x_range = np.linspace(-3, 3, 400)
true_curve = np.exp(x_range)

plt.figure(figsize=(9, 5))
plt.plot(x_range, true_curve, 'k-', linewidth=2, label='True $e^x$')

colors = ['red', 'orange', 'green', 'blue', 'purple']
for i, n_terms in enumerate([1, 2, 3, 4, 10]):
    approx_curve = np.array([taylor_exp(xi, n_terms) for xi in x_range])
    plt.plot(x_range, approx_curve, '--', color=colors[i],
             label=f'{n_terms} terms', alpha=0.8)

plt.ylim(-2, 20)
plt.axhline(0, color='gray', linewidth=0.8)
plt.axvline(0, color='gray', linewidth=0.8)
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Taylor Series Approximation of $e^x$ - Truncation Error')
plt.legend(loc='upper left')
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('errors_taylor_convergence.png', dpi=150)
plt.show()
print("\nPlot saved: errors_taylor_convergence.png")
