"""
QUESTION
--------
f(x) = ln(x)

The derivative at x = 2 can be estimated numerically using:

  Forward difference:  f'(x) ~= [ f(x+h) - f(x) ]     / h
  Central difference:  f'(x) ~= [ f(x+h) - f(x-h) ]   / (2h)

(1) The exact derivative is f'(x) = 1/x, so f'(2) = 0.5 exactly.
(2) For h = 0.5, 0.25, 0.125, 0.0625, ... (halved 8 times), compute both
    the forward-difference and central-difference estimate of f'(2).
(3) For each h, compute the true error (exact - approximate) for both
    methods.
(4) Print a table of h, forward estimate, forward true error, central
    estimate, central true error.
(5) Plot |true error| vs h on a log-log scale for both methods on the same
    axes, so the different truncation-error orders (O(h) vs O(h^2)) are
    visible as straight lines with different slopes. Save the plot, do not
    show it.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "errors")
os.makedirs(PLOTS_DIR, exist_ok=True)

X_POINT = 2.0


def f(x):
    return np.log(x)


def exact_derivative(x):
    return 1 / x


def forward_diff(h):
    return (f(X_POINT + h) - f(X_POINT)) / h


def central_diff(h):
    return (f(X_POINT + h) - f(X_POINT - h)) / (2 * h)


# ---------------------------------------------------------------------------
# (2) - (4)
# ---------------------------------------------------------------------------
exact = exact_derivative(X_POINT)
hs = [0.5 / (2 ** k) for k in range(9)]   # 0.5, 0.25, 0.125, ...

fwd_errors, ctr_errors = [], []
print(f"{'h':>10} {'forward':>12} {'fwd err':>12} {'central':>12} {'ctr err':>12}")
for h in hs:
    fwd = forward_diff(h)
    ctr = central_diff(h)
    fwd_err = abs(exact - fwd)
    ctr_err = abs(exact - ctr)
    fwd_errors.append(fwd_err)
    ctr_errors.append(ctr_err)
    print(f"{h:>10.6f} {fwd:>12.8f} {fwd_err:>12.8f} {ctr:>12.8f} {ctr_err:>12.8f}")

print(f"\nExact f'(2) = {exact}")

# ---------------------------------------------------------------------------
# (5) log-log plot of truncation error vs step size
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(hs, fwd_errors, "o-", color="tab:red", label="forward difference (O(h))")
ax.loglog(hs, ctr_errors, "s-", color="tab:green", label="central difference (O(h^2))")
ax.set_xlabel("step size, h  [log scale]")
ax.set_ylabel("|true error|  [log scale]")
ax.set_title("Truncation error of numerical derivative of ln(x) at x=2")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

out_path = os.path.join(PLOTS_DIR, "problem2_derivative_truncation_error.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to: {out_path}")
