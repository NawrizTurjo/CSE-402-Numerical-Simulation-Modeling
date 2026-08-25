def chi_square_test(seed, n=1000, bins=10):
    """
    Chi-square uniformity test — no library functions.
    Decision rule: reject H0 if chi2 > critical value from table.
    """
    _, normalized = middle_square(seed, n)

    # Step 1: Count observed values per bin
    E = n / bins        # expected per bin (uniform assumption)
    counts = [0] * bins
    for r in normalized:
        idx = min(bins - 1, int(r * bins))
        counts[idx] += 1

    # Step 2: Compute chi-square statistic
    # formula: chi2 = sum of (Oi - E)^2 / E  for each bin
    chi2 = 0
    for i in range(bins):
        chi2 += ((counts[i] - E) ** 2) / E

    # Step 3: Compare to critical value (hardcoded from chi-square table)
    # df = bins - 1 = 9,  alpha = 0.05  ->  critical value = 16.919
    critical_value = 16.919
    decision = "Reject H0" if chi2 > critical_value else "Fail to Reject H0"

    return chi2, critical_value, decision, counts


# using built in scipy.stats library for chi-square test
def chi_square_test(seed, n=1000, bins=10):
    """Run chi-square uniformity test on middle_square output."""
    _, normalized = middle_square(seed, n)
    E = n / bins   # expected = 100 per bin
    counts = [0] * bins
    for r in normalized:
        idx = min(bins - 1, int(r * bins))
        counts[idx] += 1
    chi2, p = stats.chisquare(counts, f_exp=[E] * bins)
    decision = "Reject H0" if p < 0.05 else "Fail to Reject H0"
    return chi2, p, decision, counts


def chi_square_p_value(chi2, df):
    """Upper-tail p-value. For this task df = 9 because there are 10 bins."""
    if df != 9:
        raise ValueError("This fallback p-value function is for df = 9 only")

    z = chi2 / 2
    p = math.erfc(math.sqrt(z))
    for k in range(4):
        a = k + 0.5
        p += (z**a * math.exp(-z)) / math.gamma(a + 1)
    return p