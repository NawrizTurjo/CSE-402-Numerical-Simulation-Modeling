"""
Chi-Square Test for Uniformity ("Frequency Test")

Purpose:
    Given an array of random numbers R_1, R_2, ..., R_n that are supposed
    to be drawn from a Uniform(0,1) distribution, the chi-square test
    divides [0,1) into k equal-width sub-intervals, compares the OBSERVED
    frequency in each interval against the EXPECTED frequency, and checks
    whether the discrepancy is small enough to accept the hypothesis of
    uniformity.

Procedure (Banks / Law & Kelton formulation):
    1. Divide [0, 1) into k equal sub-intervals of width 1/k.
    2. Count O_i = observed number of R_i values falling in interval i.
    3. Expected count under H0 (uniform):  E_i = n / k   for every interval.
    4. Chi-square statistic:
           chi0^2 = sum_i ( (O_i - E_i)^2 / E_i )
    5. Compare chi0^2 against the critical value chi^2_(alpha, k-1)
       from the chi-square distribution with (k - 1) degrees of freedom.
       If chi0^2 <= chi^2_(alpha, k-1) -> do NOT reject uniformity.
       If chi0^2 >  chi^2_(alpha, k-1) -> reject uniformity.
"""

from scipy.stats import chi2

# ---------------------------------------------------------------------
# INPUT: array of generated random numbers to test
# (Replace this with the R_i sequence produced by your LCM generator,
#  or any other array of numbers you want to test for uniformity.)
# ---------------------------------------------------------------------
random_numbers = [0.07, 0.62, 0.97, 0.92, 0.34, 0.51, 0.15, 0.88, 0.29, 0.66]

#bin size
k = 5          # number of equal-width sub-intervals to divide [0,1) into
alpha = 0.05   # significance level

# ---------------------------------------------------------------------
# STEP 1 & 2: bucket the numbers into k equal-width intervals, count tai O_i
# ---------------------------------------------------------------------
n = len(random_numbers)
interval_width = 1.0 / k
observed = [0] * k

for r in random_numbers:
    # guard against r == 1.0 falling outside the last bucket
    idx = min(int(r / interval_width), k - 1)
    observed[idx] += 1

# ---------------------------------------------------------------------
# STEP 3: expected count per interval under H0
# ---------------------------------------------------------------------
expected = n / k

# ---------------------------------------------------------------------
# STEP 4: chi-square statistic
# ---------------------------------------------------------------------
chi_sq_stat = sum((O - expected) ** 2 / expected for O in observed)

# ---------------------------------------------------------------------
# STEP 5: critical value chi^2_(alpha, k-1)
# ---------------------------------------------------------------------
df = k - 1
chi_sq_critical = chi2.ppf(1 - alpha, df)

# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------
print(f"{'Interval':>10} | {'Range':>15} | {'O_i':>5} | {'E_i':>6} | {'(O_i-E_i)^2/E_i':>16}")
print("-" * 62)
for i in range(k):
    low = i * interval_width
    high = (i + 1) * interval_width
    contrib = (observed[i] - expected) ** 2 / expected
    print(f"{i + 1:10d} | [{low:.2f}, {high:.2f}) | {observed[i]:5d} | {expected:6.2f} | {contrib:16.4f}")

print("-" * 62)
print(f"n                 = {n}")
print(f"k (intervals)     = {k}")
print(f"Degrees of freedom= {df}")
print(f"chi0^2 (statistic)= {chi_sq_stat:.5f}")
print(f"alpha             = {alpha}")
print(f"chi^2_critical    = {chi_sq_critical:.5f}")

if chi_sq_stat <= chi_sq_critical:
    print(f"\nResult: chi0^2 ({chi_sq_stat:.5f}) <= chi^2_critical ({chi_sq_critical:.5f})")
    print("=> Do NOT reject H0: the numbers are consistent with Uniform(0,1).")
else:
    print(f"\nResult: chi0^2 ({chi_sq_stat:.5f}) > chi^2_critical ({chi_sq_critical:.5f})")
    print("=> Reject H0: the numbers are NOT consistent with Uniform(0,1).")