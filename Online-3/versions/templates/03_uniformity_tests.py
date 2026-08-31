"""
03_uniformity_tests.py
======================
TEMPLATE: "Test whether a sequence of numbers is uniformly distributed."

Both uniformity tests from the slides, done by hand AND cross-checked with
SciPy, plus the full worked examples from the deck so you can verify the code
is right before trusting it on new data.

  A. Chi-Square goodness-of-fit  (slide 38-39)
  B. Kolmogorov-Smirnov          (slide 34-36)
  C. Which to use, and the multiple-testing trap (slides 30-33, 40)

Swap DATA at the bottom for whatever the exam gives you.
Run:  python3 03_uniformity_tests.py
"""

import math

try:
    from scipy.stats import chi2 as scipy_chi2, kstest, chisquare
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ===========================================================================
# A. CHI-SQUARE GOODNESS-OF-FIT
# ===========================================================================

def chi_square_test(values, n_bins=10, alpha=0.05, lo=0.0, hi=1.0, verbose=True):
    """H0: values ~ Uniform[lo, hi)

        chi2_0 = sum_{i=1..n} (O_i - E_i)^2 / E_i
        E_i    = N / n          (equal-probability bins)
        df     = n - 1          (no parameters estimated from the data)
        reject H0 if chi2_0 > chi2_{alpha, n-1}

    Rule of thumb from the slides: chi-square needs N >= 50, and every
    expected count E_i should be at least 5.
    """
    N = len(values)
    width = (hi - lo) / n_bins
    obs = [0] * n_bins
    for v in values:
        k = min(max(int((v - lo) / width), 0), n_bins - 1)
        obs[k] += 1

    exp = N / n_bins
    terms = [(o - exp) ** 2 / exp for o in obs]
    chi2_0 = sum(terms)
    df = n_bins - 1
    crit = chi2_critical(df, alpha)
    p = chi2_sf(chi2_0, df)

    if verbose:
        print(f"  H0: the numbers are uniformly distributed on "
              f"[{lo}, {hi})     alpha = {alpha}")
        print(f"  N = {N},  bins = {n_bins},  E_i = {exp:g},  df = {df}")
        if exp < 5:
            print(f"  !! WARNING: expected count {exp:g} < 5 -- "
                  f"chi-square approximation is unreliable. Use fewer bins.")
        print()
        print(f"  {'Interval':<16}{'O_i':>6}{'E_i':>8}"
              f"{'(O-E)^2':>12}{'(O-E)^2/E':>12}")
        print("  " + "-" * 54)
        for i in range(n_bins):
            a, b = lo + i * width, lo + (i + 1) * width
            print(f"  [{a:.2f}, {b:.2f}){'':<4}{obs[i]:>6}{exp:>8.1f}"
                  f"{(obs[i]-exp)**2:>12.2f}{terms[i]:>12.4f}")
        print("  " + "-" * 54)
        print(f"  {'Total':<16}{sum(obs):>6}{N:>8.1f}{'':>12}{chi2_0:>12.4f}")
        print()
        print(f"  chi2_0             = {chi2_0:.4f}")
        print(f"  chi2_({alpha},{df})       = {crit:.4f}")
        print(f"  p-value            = {p:.6f}")
        if chi2_0 > crit:
            print(f"  DECISION: chi2_0 > critical value  ->  REJECT H0 "
                  f"(evidence of non-uniformity)")
        else:
            print(f"  DECISION: chi2_0 <= critical value  ->  DO NOT REJECT H0 "
                  f"(no evidence against uniformity)")

    return {"observed": obs, "expected": exp, "chi2": chi2_0, "df": df,
            "crit": crit, "p": p, "reject": chi2_0 > crit}


# ===========================================================================
# B. KOLMOGOROV-SMIRNOV
# ===========================================================================

def ks_test(values, alpha=0.05, verbose=True, show_table_upto=20):
    """H0: values ~ Uniform(0,1), one-sample two-sided K-S.

    Theoretical CDF   F(x)  = x               on [0,1]
    Empirical CDF     S_N(x) = #{R_i <= x} / N

        D+ = max_{1<=i<=N} ( i/N - R_(i) )        S_N above F
        D- = max_{1<=i<=N} ( R_(i) - (i-1)/N )    F above S_N
        D  = max(D+, D-)

    Reject H0 if D > D_alpha.
    """
    R = sorted(values)
    N = len(R)

    rows = []
    d_plus, d_minus = -1.0, -1.0
    ip = im = 0
    for i in range(1, N + 1):
        a = i / N - R[i - 1]
        b = R[i - 1] - (i - 1) / N
        rows.append((i, R[i - 1], i / N, a, b))
        if a > d_plus:
            d_plus, ip = a, i
        if b > d_minus:
            d_minus, im = b, i
    D = max(d_plus, d_minus)
    crit = ks_critical(N, alpha)

    if verbose:
        print(f"  H0: the numbers are uniformly distributed on [0,1]"
              f"     alpha = {alpha}")
        print(f"  N = {N}")
        print(f"\n  Sorted sample: "
              f"{[round(r,4) for r in R[:12]]}{' ...' if N > 12 else ''}")
        if N <= show_table_upto:
            print(f"\n  {'i':>3}{'R_(i)':>10}{'i/N':>10}"
                  f"{'i/N - R_(i)':>14}{'R_(i)-(i-1)/N':>16}")
            print("  " + "-" * 53)
            for i, r, iN, a, b in rows:
                sa = f"{a:.4f}" if a > 0 else "  --  "
                sb = f"{b:.4f}" if b > 0 else "  --  "
                print(f"  {i:>3}{r:>10.4f}{iN:>10.4f}{sa:>14}{sb:>16}")
            print("  " + "-" * 53)
        print(f"\n  D+ = {d_plus:.6f}   (at i = {ip})")
        print(f"  D- = {d_minus:.6f}   (at i = {im})")
        print(f"  D  = max(D+, D-) = {D:.6f}")
        print(f"  D_({alpha}) for N = {N}: {crit:.6f}"
              f"{'  (exact table)' if N <= 40 else '  (large-sample 1.36/sqrt(N))'}")
        if D > crit:
            print(f"  DECISION: D > D_alpha  ->  REJECT H0 "
                  f"(evidence of non-uniformity)")
        else:
            print(f"  DECISION: D <= D_alpha  ->  DO NOT REJECT H0 "
                  f"(no evidence against uniformity)")

    return {"D": D, "d_plus": d_plus, "d_minus": d_minus, "N": N,
            "crit": crit, "reject": D > crit, "sorted": R}


# ===========================================================================
# CRITICAL VALUES
# ===========================================================================

KS_TABLE = {
    0.10: [.950, .776, .642, .564, .510, .470, .438, .411, .388, .368,
           .352, .338, .325, .314, .304, .295, .286, .278, .272, .264,
           .258, .252, .247, .242, .238, .233, .229, .225, .221, .218,
           .214, .211, .208, .205, .202, .199, .196, .194, .191, .189],
    0.05: [.975, .842, .708, .624, .565, .521, .486, .457, .432, .410,
           .391, .375, .361, .349, .338, .328, .318, .309, .301, .294,
           .287, .281, .275, .269, .264, .259, .254, .250, .246, .242,
           .238, .234, .231, .227, .224, .221, .218, .215, .213, .210],
    0.01: [.995, .929, .828, .733, .669, .618, .577, .543, .514, .490,
           .468, .450, .433, .418, .404, .392, .381, .371, .363, .356,
           .348, .341, .334, .327, .320, .313, .307, .301, .295, .290,
           .285, .281, .277, .273, .269, .265, .262, .258, .255, .252],
}


def ks_critical(N, alpha=0.05):
    if N <= 40 and alpha in KS_TABLE:
        return KS_TABLE[alpha][N - 1]
    return {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}.get(alpha, 1.36) / math.sqrt(N)


def chi2_critical(df, alpha=0.05):
    if HAVE_SCIPY:
        return float(scipy_chi2.ppf(1 - alpha, df))
    lo, hi = 0.0, 2000.0
    for _ in range(300):
        mid = (lo + hi) / 2
        if chi2_sf(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def chi2_sf(x, k):
    if HAVE_SCIPY:
        return float(scipy_chi2.sf(x, k))
    # regularized upper incomplete gamma Q(k/2, x/2)
    s, xx = k / 2.0, x / 2.0
    if xx == 0:
        return 1.0
    if xx < s + 1:
        term, total, n = 1.0 / s, 1.0 / s, 1
        while n < 1000:
            term *= xx / (s + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
            n += 1
        return 1.0 - total * math.exp(-xx + s * math.log(xx) - math.lgamma(s))
    tiny = 1e-300
    b, c, d = xx + 1 - s, 1 / tiny, 1 / (xx + 1 - s)
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h * math.exp(-xx + s * math.log(xx) - math.lgamma(s))


# ===========================================================================
# C. MULTIPLE TESTING (slides 30-33)
# ===========================================================================

def multiple_testing_table(alpha=0.05, ks=(1, 5, 10, 20, 50, 100)):
    """P(at least one false positive) = 1 - (1 - alpha)^k."""
    print(f"  For k independent tests at alpha = {alpha}, all H0 true:")
    print(f"      P(at least one significant result) = 1 - (1 - {alpha})^k\n")
    print(f"  {'k tests':>9}{'P(>=1 false positive)':>26}")
    print("  " + "-" * 35)
    for k in ks:
        print(f"  {k:>9}{1 - (1 - alpha) ** k:>26.4f}")
    print()
    print("  Consequence: an isolated failed test does NOT prove the generator")
    print("  is broken. Worry about (a) repeated failure of the SAME test on")
    print("  fresh sequences, or (b) failures across SEVERAL different tests.")
    print(f"  Expected chance rejections when testing 100 fresh sequences: "
          f"100 x {alpha} = {100*alpha:.0f}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 78)
    print("UNIFORMITY TESTS")
    print("=" * 78)

    # -- Verify against the worked example on slide 36 ---------------------
    print("\n### VERIFICATION vs slide 36 (K-S worked example) ###\n")
    res = ks_test([0.44, 0.81, 0.14, 0.05, 0.93], alpha=0.05)
    print(f"\n  Slide says: D+ = 0.26, D- = 0.21, D = 0.26, D_0.05 = 0.565,")
    print(f"              do not reject. Code agrees: "
          f"{abs(res['d_plus']-0.26) < 1e-9 and abs(res['d_minus']-0.21) < 1e-9}")

    # -- Verify against the worked example on slide 39 ---------------------
    print("\n\n### VERIFICATION vs slide 39 (chi-square worked example) ###\n")
    obs = [8, 8, 10, 9, 12, 8, 10, 14, 10, 11]
    fake = []
    for i, o in enumerate(obs):
        fake += [(i + 0.5) / 10] * o
    res = chi_square_test(fake, n_bins=10, alpha=0.05)
    print(f"\n  Slide says: chi2_0 = 3.4, critical = 16.9, do not reject.")
    print(f"  Code agrees: {abs(res['chi2'] - 3.4) < 1e-9}")

    # -- Apply both tests to a real generator ------------------------------
    print("\n\n### APPLYING BOTH TESTS TO AN LCG ###")

    def lcg(n, m=2**31 - 1, a=16807, c=0, x0=123457):
        out, x = [], x0
        for _ in range(n):
            x = (a * x + c) % m
            out.append(x / m)
        return out

    data = lcg(1000)
    print("\n  Generator: X = 16807*X mod (2^31 - 1), seed 123457, N = 1000\n")
    print("  --- Chi-square ---")
    chi_square_test(data, n_bins=10, alpha=0.05)
    print("\n  --- Kolmogorov-Smirnov ---")
    ks_test(data, alpha=0.05)

    if HAVE_SCIPY:
        print("\n  --- SciPy cross-check ---")
        st, p = kstest(data, 'uniform')
        print(f"  scipy.stats.kstest: D = {st:.6f}, p = {p:.6f}")

    # -- Comparison table --------------------------------------------------
    print("\n\n### K-S vs CHI-SQUARE (slide 40) ###\n")
    print("                        Kolmogorov-Smirnov      Chi-Square")
    print("  Works with small N?          Yes               No (needs N >= 50)")
    print("  Uses binning?                No (exact CDF)    Yes")
    print("  More powerful?               Generally yes     Sometimes")
    print("  Recommendation               Preferred         Good backup")
    print("\n  BUT: both only test UNIFORMITY. The sequence")
    print("  0.01, 0.02, ..., 0.99 passes both perfectly and is fully")
    print("  predictable. You must ALSO test independence -> 04_independence_tests.py")

    # -- Multiple testing --------------------------------------------------
    print("\n\n### THE MULTIPLE-TESTING TRAP (slides 30-33) ###\n")
    multiple_testing_table()


if __name__ == "__main__":
    main()
