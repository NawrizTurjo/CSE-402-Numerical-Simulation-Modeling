"""
CSE 401: Statistical Testing of Random Numbers (Simple Implementation)
Includes:
  1. Kolmogorov-Smirnov (K-S) Test for Uniformity
  2. Chi-Square Frequency Test for Uniformity
  3. Autocorrelation Test for Independence
"""

import math

# ==============================================================================
# 1. Kolmogorov-Smirnov (K-S) Test
# ==============================================================================
def kolmogorov_smirnov_test(numbers, alpha=0.05):
    """
    Performs K-S test for H0: R_i ~ Uniform[0, 1].
    
    Formulas:
      D+ = max_i ( i/N - R_(i) )
      D- = max_i ( R_(i) - (i-1)/N )
      D  = max(D+, D-)
    """
    sorted_r = sorted(numbers)
    N = len(sorted_r)
    
    d_plus_max = -math.inf
    d_minus_max = -math.inf
    
    table_rows = []
    for i, r in enumerate(sorted_r, start=1):
        d_plus = (i / N) - r
        d_minus = r - ((i - 1) / N)
        
        d_plus_max = max(d_plus_max, d_plus)
        d_minus_max = max(d_minus_max, d_minus)
        
        table_rows.append((i, r, i / N, d_plus, (i - 1) / N, d_minus))
        
    D = max(d_plus_max, d_minus_max)
    
    # Critical value table for alpha = 0.05
    ks_table_005 = {1: 0.975, 2: 0.842, 3: 0.708, 4: 0.624, 5: 0.565, 10: 0.410, 20: 0.294}
    if N in ks_table_005:
        d_crit = ks_table_005[N]
    else:
        # Large sample approximation for alpha = 0.05: 1.36 / sqrt(N)
        d_crit = 1.36 / math.sqrt(N)
        
    decision = "Reject H0" if D > d_crit else "Do not reject H0"
    
    return {
        "D": D,
        "D+": d_plus_max,
        "D-": d_minus_max,
        "critical_value": d_crit,
        "decision": decision,
        "table": table_rows
    }


# ==============================================================================
# 2. Chi-Square Goodness-of-Fit Test
# ==============================================================================
def chi_square_test(numbers, k=10, alpha=0.05):
    """
    Performs Chi-Square test for H0: R_i ~ Uniform[0, 1].
    
    Formula:
      chi2 = sum ( (O_i - E_i)^2 / E_i )
      E_i = N / k
    """
    N = len(numbers)
    observed = [0] * k
    
    for r in numbers:
        bin_idx = int(r * k)
        if bin_idx >= k:
            bin_idx = k - 1
        observed[bin_idx] += 1
        
    expected = N / k
    chi2_stat = 0.0
    table_rows = []
    
    for i, o in enumerate(observed):
        diff_sq = (o - expected) ** 2
        term = diff_sq / expected
        chi2_stat += term
        table_rows.append((f"[{i/k:.1f}, {(i+1)/k:.1f})", o, expected, diff_sq, term))
        
    # Critical value for df = 9, alpha = 0.05 is approximately 16.9
    # For df = k - 1
    chi2_crit = 16.9 if k == 10 else 16.92
    decision = "Reject H0" if chi2_stat > chi2_crit else "Do not reject H0"
    
    return {
        "chi2": chi2_stat,
        "critical_value": chi2_crit,
        "expected": expected,
        "observed": observed,
        "decision": decision,
        "table": table_rows
    }


# ==============================================================================
# 3. Autocorrelation Test
# ==============================================================================
def autocorrelation_test(numbers, start=3, lag=5, alpha=0.05):
    """
    Performs Autocorrelation test for H0: rho_{i,l} = 0.
    
    Formulas:
      M = largest integer such that start + (M + 1) * lag <= N
      rho_hat = [ 1 / (M + 1) * sum_{k=0}^M ( R_{i+kl} * R_{i+(k+1)l} ) ] - 0.25
      sigma = sqrt(13M + 7) / (12 * (M + 1))
      Z0 = rho_hat / sigma
      Reject H0 if |Z0| > 1.96 (for alpha = 0.05)
    """
    N = len(numbers)
    # 0-indexed positions: index = (start - 1) + k * lag
    # Find M:
    M = -1
    while start + (M + 2) * lag <= N:
        M += 1
        
    indices = [start + k * lag for k in range(M + 2)]  # 1-based indices
    subsequence = [numbers[idx - 1] for idx in indices]
    
    products = [subsequence[k] * subsequence[k + 1] for k in range(M + 1)]
    sum_products = sum(products)
    
    rho_hat = (sum_products / (M + 1)) - 0.25
    sigma = math.sqrt(13 * M + 7) / (12 * (M + 1))
    z0 = rho_hat / sigma
    
    z_crit = 1.96  # for alpha = 0.05
    decision = "Reject H0" if abs(z0) > z_crit else "Do not reject H0"
    
    return {
        "M": M,
        "indices": indices,
        "subsequence": subsequence,
        "products": products,
        "rho_hat": rho_hat,
        "sigma": sigma,
        "Z0": z0,
        "critical_value": z_crit,
        "decision": decision
    }


# ==============================================================================
# Main: Verification with Slide Examples
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Example 6: Kolmogorov-Smirnov Test (Data: 0.44, 0.81, 0.14, 0.05, 0.93)")
    print("=" * 70)
    sample_ks = [0.44, 0.81, 0.14, 0.05, 0.93]
    res_ks = kolmogorov_smirnov_test(sample_ks, alpha=0.05)
    print(f"{'i':<3} | {'R_(i)':<8} | {'i/N':<8} | {'D+':<8} | {'(i-1)/N':<8} | {'D-'}")
    print("-" * 55)
    for row in res_ks["table"]:
        print(f"{row[0]:<3} | {row[1]:<8.2f} | {row[2]:<8.2f} | {row[3]:<8.2f} | {row[4]:<8.2f} | {row[5]:.2f}")
    print(f"\nD+ = {res_ks['D+']:.2f}, D- = {res_ks['D-']:.2f} -> D = {res_ks['D']:.2f}")
    print(f"Critical D_0.05 = {res_ks['critical_value']:.3f} -> Decision: {res_ks['decision']}")

    print("\n" + "=" * 70)
    print("Example 7: Chi-Square Frequency Test (100 numbers, 10 bins)")
    print("=" * 70)
    counts = [8, 8, 10, 9, 12, 8, 10, 14, 10, 11]
    # Reconstruct data matching exact bin frequencies
    dummy_data = []
    for bin_idx, cnt in enumerate(counts):
        dummy_data.extend([(bin_idx + 0.5) / 10.0] * cnt)
        
    res_chi = chi_square_test(dummy_data, k=10, alpha=0.05)
    print(f"{'Interval':<12} | {'O_i':<5} | {'E_i':<5} | {'(O-E)^2':<8} | {'(O-E)^2/E'}")
    print("-" * 55)
    for row in res_chi["table"]:
        print(f"{row[0]:<12} | {row[1]:<5} | {row[2]:<5.1f} | {row[3]:<8.1f} | {row[4]:.2f}")
    print(f"\nCalculated Chi^2 = {res_chi['chi2']:.1f}, Critical Chi^2_0.05,9 = {res_chi['critical_value']:.1f}")
    print(f"Decision: {res_chi['decision']}")

    print("\n" + "=" * 70)
    print("Example 8: Autocorrelation Test (i = 3, lag = 5, N = 30)")
    print("=" * 70)
    numbers_30 = [
        0.12, 0.01, 0.23, 0.28, 0.89, 0.31, 0.64, 0.28, 0.83, 0.93,
        0.99, 0.15, 0.33, 0.35, 0.91, 0.41, 0.60, 0.27, 0.75, 0.88,
        0.68, 0.49, 0.05, 0.43, 0.95, 0.58, 0.19, 0.36, 0.69, 0.87
    ]
    res_auto = autocorrelation_test(numbers_30, start=3, lag=5, alpha=0.05)
    print(f"M = {res_auto['M']}")
    print(f"Subsequence values (indices {res_auto['indices']}): {res_auto['subsequence']}")
    print(f"Products: {[round(p, 4) for p in res_auto['products']]}")
    print(f"rho_hat = {res_auto['rho_hat']:.4f}")
    print(f"sigma   = {res_auto['sigma']:.4f}")
    print(f"Z0      = {res_auto['Z0']:.2f} (Critical = +/-{res_auto['critical_value']})")
    print(f"Decision: {res_auto['decision']}")
