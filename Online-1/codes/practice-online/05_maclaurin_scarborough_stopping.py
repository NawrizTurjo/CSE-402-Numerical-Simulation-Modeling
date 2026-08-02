"""
Problem 5: Damped Wave Maclaurin Expansion (Scarborough Stopping)
Objective:
  Approximate sin(x)/x at x = 1.2 using Maclaurin series expansion:
    sin(x)/x = sum_{k=0}^inf (-1)^k * x^(2k) / (2k + 1)!

Demonstrate:
  1. Target n = 5 significant figures -> Scarborough tolerance es = 0.5 * 10^(2-5) = 0.0005%.
  2. Recurrence term T_k = -T_{k-1} * x^2 / ((2k) * (2k+1)).
  3. Per-iteration tracking of guaranteed significant digits using P.calc_sig_digit(ea).
"""

import math
from common import P, banner

# -----------------------------------------------------------------------------
# Problem Definition
# -----------------------------------------------------------------------------
X0 = 1.2
SIG_TARGET = 5


def term_recurrence(prev_term, k, x):
    # Recurrence from term (k-1) to term k:
    # T_k = T_{k-1} * (-x^2) / ((2*k) * (2*k + 1))
    return -prev_term * (x**2) / ((2 * k) * (2 * k + 1))


banner("PROBLEM 5: MACLAURIN EXPANSION WITH SCARBOROUGH STOPPING")

exact_val = math.sin(X0) / X0
es = P.scarborough_tolerance(SIG_TARGET)

print(f"  Target x = {X0}     Exact sin(1.2)/1.2 = {exact_val:.10f}")
print(f"  Target Significant Figures: {SIG_TARGET} -> Scarborough es = {es:.6f}%\n")

S_approx, history = P.maclaurin_series(
    term_recurrence,
    x=X0,
    first_term=1.0,
    sig_digits=SIG_TARGET,
    max_terms=20
)

print(f"{'Term (k)':<10}{'Partial Sum S_k':<20}{'Error ea (%)':<18}{'Sig Digits Guaranteed':<22}")
print("-" * 70)
for h in history:
    ea_str = "Initial" if h['term'] == 0 else f"{h['ea']:.6f}%"
    sig_str = "---" if h['term'] == 0 else f"{h['sig']} digits"
    print(f"{h['term']:<10}{h['S']:<20.10f}{ea_str:<18}{sig_str:<22}")
print("-" * 70)

true_err = P.true_error(exact_val, S_approx)
true_rel_err = P.true_relative_error_pct(exact_val, S_approx)

print(f"\n  Converged in {len(history)-1} terms.")
print(f"  Approximation = {S_approx:.10f}")
print(f"  True Error Et = {true_err:.10e}")
print(f"  True Relative Error eps_t = {true_rel_err:.6f}%")
