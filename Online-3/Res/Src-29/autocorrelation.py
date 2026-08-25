"""
Autocorrelation Test for Independence (Z-value comparison)

Purpose:
    Given an array of random numbers R_1, R_2, ..., R_N that are supposed
    to be independent, the autocorrelation test checks whether numbers
    spaced l apart (lag l), starting from index i, are correlated with
    each other. If they are, the "independence" assumption is violated.

Procedure (Banks / Law & Kelton formulation):
    Consider the numbers  R_i, R_(i+l), R_(i+2l), ..., R_(i+(M+1)l)
    where M = floor( (N - i) / l ) - 1   (number of consecutive pairs used)

    1. Estimated autocorrelation between numbers l apart:
           rho_hat_il = [ (1/(M+1)) * sum_{k=0}^{M} R_(i+kl) * R_(i+(k+1)l) ] - 0.25

    2. Standard deviation of the estimator (for large M):
           sigma_rho_il = sqrt(13*M + 7) / (12*(M+1))

    3. Test statistic:
           Z0 = rho_hat_il / sigma_rho_il

    4. Compare |Z0| against the critical value z_(1 - alpha/2) from the
       standard normal distribution.
       If |Z0| <= z_(1-alpha/2) -> do NOT reject H0 (no significant autocorrelation).
       If |Z0| >  z_(1-alpha/2) -> reject H0 (numbers are NOT independent).
"""

import math
from scipy.stats import norm

# ---------------------------------------------------------------------
# INPUT: array of generated random numbers to test
# (Replace this with the R_i sequence produced by your LCM generator,
#  or any other array of numbers you want to test for independence.)
# ---------------------------------------------------------------------
random_numbers = [
    0.12, 0.01, 0.23, 0.28, 0.89, 0.31, 0.64, 0.28, 0.83, 0.93,
    0.99, 0.15, 0.33, 0.35, 0.91, 0.41, 0.60, 0.27, 0.75, 0.88,
]

i = 1          # starting index (1-based, as in the standard formulation)
l = 5          # lag (test correlation between numbers l apart)
alpha = 0.05   # significance level

# ---------------------------------------------------------------------
# STEP 0: set up 1-based indexing helper and determine M
# ---------------------------------------------------------------------
N = len(random_numbers)
R = [None] + list(random_numbers)   # R[1..N] so indices match the formula directly

# i + (M+1).l <= N
M = math.floor((N - i) / l) - 1

if M < 1:
    raise ValueError(
        f"Not enough data for i={i}, l={l}, N={N}. "
        f"Need a larger sample or smaller lag/start index."
    )

# ---------------------------------------------------------------------
# STEP 1: estimated autocorrelation rho_hat_il
# ---------------------------------------------------------------------
pair_products = []
print(f"{'k':>3} | {'i+kl':>6} | {'i+(k+1)l':>9} | {'R_(i+kl)':>10} | {'R_(i+(k+1)l)':>13} | {'product':>8}")
print("-" * 66)

sum_products = 0.0
for k in range(M + 1):
    idx1 = i + k * l
    idx2 = i + (k + 1) * l
    val1 = R[idx1]
    val2 = R[idx2]
    product = val1 * val2
    sum_products += product
    pair_products.append(product)
    print(f"{k:3d} | {idx1:6d} | {idx2:9d} | {val1:10.4f} | {val2:13.4f} | {product:8.4f}")

rho_hat = (sum_products / (M + 1)) - 0.25

# ---------------------------------------------------------------------
# STEP 2: standard deviation of the estimator
# ---------------------------------------------------------------------
sigma_rho = math.sqrt(13 * M + 7) / (12 * (M + 1))

# ---------------------------------------------------------------------
# STEP 3: test statistic Z0
# ---------------------------------------------------------------------
Z0 = rho_hat / sigma_rho

# ---------------------------------------------------------------------
# STEP 4: critical value z_(1 - alpha/2)
# ---------------------------------------------------------------------
z_critical = norm.ppf(1 - alpha / 2)

# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------
print("-" * 66)
print(f"N                 = {N}")
print(f"i (start index)   = {i}")
print(f"l (lag)           = {l}")
print(f"M                 = {M}")
print(f"rho_hat_il        = {rho_hat:.5f}")
print(f"sigma_rho_il      = {sigma_rho:.5f}")
print(f"Z0 (statistic)    = {Z0:.5f}")
print(f"alpha             = {alpha}")
print(f"z_critical        = +/- {z_critical:.5f}")

if abs(Z0) <= z_critical:
    print(f"\nResult: |Z0| ({abs(Z0):.5f}) <= z_critical ({z_critical:.5f})")
    print("=> Do NOT reject H0: no significant autocorrelation detected (independent).")
else:
    print(f"\nResult: |Z0| ({abs(Z0):.5f}) > z_critical ({z_critical:.5f})")
    print("=> Reject H0: significant autocorrelation detected (NOT independent).")