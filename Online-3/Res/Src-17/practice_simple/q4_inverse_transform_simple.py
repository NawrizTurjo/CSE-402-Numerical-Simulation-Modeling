"""
Practice Q4: Random Variate Generation by Inverse-Transform Method
Simple, Student-Friendly Implementation
"""

import math
from scipy.stats import chisquare

# Simple generator for Uniform [0, 1)
def get_uniforms(n, seed=123457):
    x = seed
    a, m = 16807, 2**31 - 1
    out = []
    for _ in range(n):
        x = (a * x) % m
        out.append(x / m)
    return out

# ==============================================================================
# Task 1: Inverse-Transform Formulas
# ==============================================================================
def inverse_uniform(r, a, b):
    # F(x) = (x - a) / (b - a)  =>  x = a + (b - a) * R
    return a + (b - a) * r

def inverse_exponential(r, mean):
    # F(x) = 1 - e^(-x/mean)  =>  x = -mean * ln(1 - R)
    return -mean * math.log(1.0 - r)

def inverse_weibull(r, shape_k, scale_c):
    # F(x) = 1 - e^(-(x/c)^k)  =>  x = c * (-ln(1 - R))^(1/k)
    return scale_c * (-math.log(1.0 - r)) ** (1.0 / shape_k)

def inverse_discrete(r, values, probs):
    # Staircase CDF
    cumulative = 0.0
    for v, p in zip(values, probs):
        cumulative += p
        if r <= cumulative:
            return v
    return values[-1]

def inverse_triangular(r, a, b, c):
    # Triangular with min a, max b, mode c
    split = (c - a) / (b - a)
    if r <= split:
        return a + math.sqrt(r * (b - a) * (c - a))
    else:
        return b - math.sqrt((1.0 - r) * (b - a) * (b - c))


# ==============================================================================
# Task 3: Chi-Square Goodness-of-Fit with Equiprobable Bins
# ==============================================================================
def test_exponential_equiprobable(sample, mean, k=10):
    """
    Equiprobable bins: Cutpoints at x_i = -mean * ln(1 - i/k)
    so each bin has expected probability 1/k.
    """
    cutpoints = [-mean * math.log(1.0 - i / k) for i in range(1, k)]
    observed = [0] * k
    
    for x in sample:
        idx = 0
        while idx < len(cutpoints) and x > cutpoints[idx]:
            idx += 1
        observed[idx] += 1
        
    chi2, p_val = chisquare(observed)
    decision = "Reject H0" if p_val < 0.05 else "Do not reject H0"
    return chi2, p_val, decision, observed, cutpoints


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
if __name__ == "__main__":
    n = 10000
    r_vals = get_uniforms(n)

    print("=" * 70)
    print("TASKS 1 & 2: Verification of Generated Variates (Moments)")
    print("=" * 70)
    print(f"{'Distribution':<25} | {'Sample Mean':<12} | {'True Mean':<10} | {'Sample Var':<12} | {'True Var'}")
    print("-" * 75)

    # 1. Uniform(5, 15)
    u_data = [inverse_uniform(r, 5, 15) for r in r_vals]
    u_mean = sum(u_data) / n
    u_var = sum((x - u_mean)**2 for x in u_data) / (n - 1)
    print(f"{'Uniform(5, 15)':<25} | {u_mean:<12.4f} | {10.0:<10.4f} | {u_var:<12.4f} | {100/12:.4f}")

    # 2. Exponential(mean 2)
    e_data = [inverse_exponential(r, 2.0) for r in r_vals]
    e_mean = sum(e_data) / n
    e_var = sum((x - e_mean)**2 for x in e_data) / (n - 1)
    print(f"{'Exponential(mean 2)':<25} | {e_mean:<12.4f} | {2.0:<10.4f} | {e_var:<12.4f} | {4.0:.4f}")

    # 3. Weibull(k=2, c=3)
    w_data = [inverse_weibull(r, 2.0, 3.0) for r in r_vals]
    w_mean = sum(w_data) / n
    w_var = sum((x - w_mean)**2 for x in w_data) / (n - 1)
    true_w_mean = 3.0 * math.gamma(1.5)
    true_w_var = (3.0**2) * (math.gamma(2.0) - math.gamma(1.5)**2)
    print(f"{'Weibull(k=2, c=3)':<25} | {w_mean:<12.4f} | {true_w_mean:<10.4f} | {w_var:<12.4f} | {true_w_var:.4f}")

    print("\n" + "=" * 70)
    print("TASK 3: Chi-Square Test for Exponential Variates (Equiprobable Bins)")
    print("=" * 70)
    chi2, p_val, dec, obs, cuts = test_exponential_equiprobable(e_data, mean=2.0, k=10)
    print(f"Equiprobable Cutpoints: {[round(c, 3) for c in cuts]}")
    print(f"Observed counts in 10 bins: {obs}")
    print(f"Chi^2 = {chi2:.2f}, p-value = {p_val:.4f} -> Decision: {dec}")
    print("\nWhy Equiprobable Bins? Equal-width bins on an exponential distribution leave")
    print("the tail bins with almost 0 expected observations (violating E_i >= 5).")

    print("\n" + "=" * 70)
    print("TASK 4: Reflection Summary")
    print("=" * 70)
    print("• Inverse transform is only a deterministic mapping; it does NOT create randomness.")
    print("• If the input uniform stream has a short period or correlation, the generated variates")
    print("  will inherit the exact same flaws (e.g. repeating cycles of service times).")
