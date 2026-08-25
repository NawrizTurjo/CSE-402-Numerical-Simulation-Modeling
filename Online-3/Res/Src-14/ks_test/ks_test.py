import math

def ks_test(sample, alpha=0.05):
    """
    Kolmogorov-Smirnov test for uniformity on [0,1].
    No library functions used.

    Returns D, D+, D-, D_critical, decision
    """
    N = len(sample)
    s = sorted(sample)

    # D+ : how far the empirical CDF overshoots the diagonal (above)
    D_plus  = max(i/N - s[i-1] for i in range(1, N+1))

    # D- : how far the diagonal overshoots the empirical CDF (below)
    D_minus = max(s[i-1] - (i-1)/N for i in range(1, N+1))

    # D : the single worst gap in either direction
    D = max(D_plus, D_minus)

    # Critical value from K-S table (hardcoded for common N and alpha)
    exact_table = {
        (5,  0.10): 0.447, (5,  0.05): 0.509, (5,  0.01): 0.669,
        (10, 0.10): 0.323, (10, 0.05): 0.369, (10, 0.01): 0.489,
        (15, 0.10): 0.266, (15, 0.05): 0.304, (15, 0.01): 0.404,
        (20, 0.10): 0.231, (20, 0.05): 0.264, (20, 0.01): 0.356,
        (25, 0.10): 0.210, (25, 0.05): 0.240, (25, 0.01): 0.323,
        (30, 0.10): 0.190, (30, 0.05): 0.218, (30, 0.01): 0.295,
        (40, 0.10): 0.165, (40, 0.05): 0.189, (40, 0.01): 0.252,
        (50, 0.10): 0.148, (50, 0.05): 0.170, (50, 0.01): 0.228,
    }

    if (N, alpha) in exact_table:
        D_critical = exact_table[(N, alpha)]
    else:
        # Large-N approximation: D_critical ≈ c(alpha) / sqrt(N)
        c = {0.10: 1.224, 0.05: 1.358, 0.01: 1.628}
        D_critical = c[alpha] / math.sqrt(N)

    decision = "Reject H0 (NOT uniform)" if D > D_critical else "Fail to Reject H0 (uniform)"

    return D, D_plus, D_minus, D_critical, decision





