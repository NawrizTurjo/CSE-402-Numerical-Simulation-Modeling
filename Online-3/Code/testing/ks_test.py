"""
Kolmogorov-Smirnov (K-S) test for uniformity on [0, 1).

H0: the sample comes from Uniform[0, 1).

Idea: compare the empirical CDF of the sorted sample against the
diagonal line y = x (the CDF of a true Uniform[0,1)). The test statistic
D is the single largest vertical gap between them, checked from both
above and below since the empirical CDF can overshoot or undershoot:

    sorted sample: R_(1) <= R_(2) <= ... <= R_(N)

    D+ = max_i ( i/N   - R_(i)   )     empirical CDF undershoots the diagonal
    D- = max_i ( R_(i) - (i-1)/N )     empirical CDF overshoots the diagonal
    D  = max(D+, D-)

Decision: reject H0 if D > D_critical(N, alpha).

scipy's `stats.kstest` (one-sample, against 'uniform') gives the same D
and an exact p-value for any N directly -- prefer it when you just need
an answer. The manual `ks_statistic` below is kept because some slide
questions want you to show the D+/D-/D table by hand.

Run standalone:
    python -m testing.ks_test
"""

from scipy import stats


def ks_statistic(sample):
    """Manual D+, D-, D computation (matches the slide's table layout)."""
    n = len(sample)
    s = sorted(sample)
    d_plus = max(i / n - s[i - 1] for i in range(1, n + 1))
    d_minus = max(s[i - 1] - (i - 1) / n for i in range(1, n + 1))
    return d_plus, d_minus, max(d_plus, d_minus)


def ks_uniform_test(sample, alpha=0.05):
    """
    Full K-S uniformity test. Uses scipy's `ksone` distribution for the
    critical value / p-value -- exact for any N and alpha (no table needed).
    """
    d_plus, d_minus, d = ks_statistic(sample)
    n = len(sample)
    d_critical = stats.ksone.ppf(1 - alpha, n)
    p_value = 1 - stats.ksone.cdf(d, n)
    decision = "Reject H0" if d > d_critical else "Do not reject H0"
    return {
        "D+": d_plus,
        "D-": d_minus,
        "D": d,
        "critical_value": d_critical,
        "p_value": p_value,
        "decision": decision,
    }


if __name__ == "__main__":
    # Slide's worked example.
    sample = [0.44, 0.81, 0.14, 0.05, 0.93]
    result = ks_uniform_test(sample)
    print(f"D+={result['D+']:.4f}  D-={result['D-']:.4f}  D={result['D']:.4f}")
    print(f"D_critical={result['critical_value']:.4f}  p={result['p_value']:.4f}  "
          f"-> {result['decision']}")
    # Expected (matches slide): D+=0.2600 D-=0.2100 D=0.2600 D_crit=0.5094 -> Do not reject H0
