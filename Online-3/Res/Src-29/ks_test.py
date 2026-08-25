"""
Kolmogorov-Smirnov (K-S) Test for Uniformity ("Frequency Test")

Purpose:
    Given an array of random numbers R_1, R_2, ..., R_n that are supposed
    to be drawn from a Uniform(0,1) distribution, the K-S test compares
    the empirical CDF of the sample against the theoretical CDF F(x) = x
    and checks whether the maximum deviation D is small enough to accept
    the hypothesis of uniformity.

Procedure (Law & Kelton / Banks et al. formulation):
    1. Sort the R_i values in ascending order:  R_(1) <= R_(2) <= ... <= R_(n)
    2. Compute:
           D+ = max_i ( i/n       - R_(i) )
           D- = max_i ( R_(i)     - (i-1)/n )
    3. D = max(D+, D-)
    4. Compare D against the critical value D_alpha,n.
       If D <= D_alpha,n  -> do NOT reject the hypothesis of uniformity.
       If D >  D_alpha,n  -> reject the hypothesis of uniformity.
"""

import math

# ---------------------------------------------------------------------
# INPUT: array of generated random numbers to test
# (Replace this with the R_i sequence produced by your LCM generator,
#  or any other array of numbers you want to test for uniformity.)
# ---------------------------------------------------------------------
random_numbers = [0.07, 0.62, 0.97, 0.92, 0.34, 0.51, 0.15, 0.88, 0.29, 0.66]

alpha = 0.05   # significance level (common choices: 0.01, 0.05, 0.10, 0.15, 0.20)

# ---------------------------------------------------------------------
# STEP 1: sort the sample
# ---------------------------------------------------------------------
n = len(random_numbers)
R = sorted(random_numbers)

# ---------------------------------------------------------------------
# STEP 2: compute D+ and D-
# ---------------------------------------------------------------------
D_plus = max((i + 1) / n - R[i] for i in range(n))
D_minus = max(R[i] - i / n for i in range(n))

# ---------------------------------------------------------------------
# STEP 3: overall statistic D
# ---------------------------------------------------------------------
D = max(D_plus, D_minus)

# ---------------------------------------------------------------------
# STEP 4: critical value D_alpha,n
# Asymptotic (large-sample) approximation, valid for n > ~35,
# from Law & Kelton "Simulation Modeling and Analysis":
#     D_alpha ≈ c(alpha) / ( sqrt(n) + 0.12 + 0.11/sqrt(n) )
# where c(alpha) depends on the significance level.
# ---------------------------------------------------------------------
c_alpha_table = {
    0.20: 1.07,
    0.15: 1.14,
    0.10: 1.22,
    0.05: 1.36,
    0.01: 1.63,
}

if alpha not in c_alpha_table:
    raise ValueError(f"alpha must be one of {list(c_alpha_table.keys())}")

c_alpha = c_alpha_table[alpha]
D_critical = c_alpha / (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n))

# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------
print(f"{'i':>3} | {'R_(i)':>8} | {'i/n':>8} | {'(i-1)/n':>8}")
print("-" * 38)
for i in range(n):
    print(f"{i + 1:3d} | {R[i]:8.4f} | {(i + 1) / n:8.4f} | {i / n:8.4f}")

print("-" * 38)
print(f"n            = {n}")
print(f"D+           = {D_plus:.5f}")
print(f"D-           = {D_minus:.5f}")
print(f"D (statistic)= {D:.5f}")
print(f"alpha        = {alpha}")
print(f"D_critical   = {D_critical:.5f}")

if D <= D_critical:
    print(f"\nResult: D ({D:.5f}) <= D_critical ({D_critical:.5f})")
    print("=> Do NOT reject H0: the numbers are consistent with Uniform(0,1).")
else:
    print(f"\nResult: D ({D:.5f}) > D_critical ({D_critical:.5f})")
    print("=> Reject H0: the numbers are NOT consistent with Uniform(0,1).")