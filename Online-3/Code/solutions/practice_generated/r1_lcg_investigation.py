"""
Practice R1: LCG Investigation & Chi-Square Uniformity Test.
Recurrence: X_{n+1} = (a * X_n + c) mod m,  U_n = X_n / m.
"""

from scipy import stats


def lcg_stream(seed, a, c, m, n):
    x = seed
    out = []
    for _ in range(n):
        x = (a * x + c) % m
        out.append(x)
    return out


def chi_square_test(uniforms, bins=10, alpha=0.05):
    n = len(uniforms)
    observed = [0] * bins
    for u in uniforms:
        idx = min(int(u * bins), bins - 1)
        observed[idx] += 1

    expected = n / float(bins)
    chi2 = sum((o - expected) ** 2 / expected for o in observed)
    df = bins - 1
    p_val = stats.chi2.sf(chi2, df)
    decision = "Reject H0" if p_val < alpha else "Do not reject H0"
    return chi2, p_val, decision


def run_lcg_practice():
    print("--- LCG Investigation & Chi-Square Test ---")

    # 1. Generate first 20 values
    vals = lcg_stream(seed=7, a=5, c=3, m=16, n=20)
    print("First 20 values (seed=7, a=5, c=3, m=16):", vals)

    # 2. Chi-Square uniformity tests (N = 1000)
    # Small full-period LCG (m = 16)
    u_small = [x / 16.0 for x in lcg_stream(seed=7, a=5, c=3, m=16, n=1000)]
    chi2_s, p_s, dec_s = chi_square_test(u_small)
    print(f"Small LCG (m=16)   -> Chi^2 = {chi2_s:.2f}, p-value = {p_s:.4e} -> {dec_s}")

    # ANSI-C LCG (m = 2^31)
    m_large = 2**31
    u_ansic = [x / float(m_large) for x in lcg_stream(seed=1, a=1103515245, c=12345, m=m_large, n=1000)]
    chi2_a, p_a, dec_a = chi_square_test(u_ansic)
    print(f"ANSI-C (m=2^31)    -> Chi^2 = {chi2_a:.2f}, p-value = {p_a:.4f} -> {dec_a}")

    print("\nNote: Full-period on a small m (e.g. m=16) repeats too fast and fails uniformity.")
    print("A large modulus m (like 2^31) is required for quality simulations.")


run_lcg_practice()
