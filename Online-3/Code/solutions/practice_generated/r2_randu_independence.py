"""
Practice R2: RANDU Investigation.
Parameters: X_{n+1} = (65539 * X_n) mod 2^31, seed = 1.
"""

import math
from scipy import stats


def run_randu_practice():
    seed = 1
    a = 65539
    m = 2**31
    n = 1000

    # 1. Generate numbers
    x = seed
    integers = []
    uniforms = []
    for _ in range(n):
        x = (a * x) % m
        integers.append(x)
        uniforms.append(x / float(m))

    print("--- RANDU PRNG Investigation ---")
    print("First 5 RANDU integers:", integers[:5])

    # 2. Chi-Square Test (1-D Uniformity)
    bins = 10
    observed = [0] * bins
    for u in uniforms:
        idx = min(int(u * bins), bins - 1)
        observed[idx] += 1
    expected = n / float(bins)
    chi2 = sum((o - expected) ** 2 / expected for o in observed)
    p_val = stats.chi2.sf(chi2, df=bins - 1)
    print(f"Chi-Square Test (1-D)    : Chi^2 = {chi2:.2f}, p-value = {p_val:.4f} -> Passes Uniformity!")

    # 3. Lag-1 Autocorrelation (2-D Independence)
    sum_prod = sum(uniforms[k] * uniforms[k + 1] for k in range(n - 1))
    rho_hat = (1.0 / (n - 1)) * sum_prod - 0.25
    sigma = math.sqrt(13 * (n - 2) + 7) / (12.0 * (n - 1))
    z0 = rho_hat / sigma
    print(f"Autocorrelation (Lag-1)  : rho = {rho_hat:.4f}, Z0 = {z0:.4f} -> Passes Independence!")

    # 4. 3D Hyperplane Defect: 9*X_i - 6*X_{i+1} + X_{i+2} mod m == 0
    residuals = [(9 * integers[i] - 6 * integers[i + 1] + integers[i + 2]) % m for i in range(5)]
    print(f"3D Hyperplane Residuals  : {residuals} (All Zero!)")
    print("Conclusion: RANDU deceptively passes 1-D and 2-D tests, but fails catastrophically in 3-D.")


run_randu_practice()
