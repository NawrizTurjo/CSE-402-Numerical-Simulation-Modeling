"""
scipy.stats syntax reference -- the calls that replace hardcoded
critical-value tables in testing/chi_square_test.py, testing/ks_test.py,
and testing/independence_test.py. If you forget the exact scipy call
mid-exam, this is the file to check.

Run standalone (prints every section):
    python -m cheatsheets.scipy_syntax
"""

from scipy import stats


def chi_square_reference():
    alpha, df = 0.05, 9   # df = bins - 1, e.g. 9 for 10 bins
    chi2_stat = 15.51     # example statistic to look up

    print("Chi-Square (stats.chi2)")
    print("critical value = stats.chi2.ppf(1 - alpha, df) ->", stats.chi2.ppf(1 - alpha, df))
    print("p-value        = stats.chi2.sf(chi2_stat, df)  ->", stats.chi2.sf(chi2_stat, df))
    print("(sf = 'survival function' = 1 - cdf = upper-tail probability)")


def ks_reference():
    alpha, n = 0.05, 5
    D_stat = 0.26

    print("\nKolmogorov-Smirnov (stats.ksone)")
    print("critical value = stats.ksone.ppf(1 - alpha, n) ->", stats.ksone.ppf(1 - alpha, n))
    print("p-value        = 1 - stats.ksone.cdf(D, n)     ->", 1 - stats.ksone.cdf(D_stat, n))
    print("also available: scipy.stats.kstest(sample, 'uniform') does the whole test in one call")


def normal_reference():
    """Every Z-test in this course (autocorrelation, both runs tests)
    uses the same two-sided normal lookup."""
    alpha, Z0 = 0.05, 1.2

    print("\nNormal distribution (stats.norm) -- for Z-tests")
    print("critical value = stats.norm.ppf(1 - alpha/2)        ->", stats.norm.ppf(1 - alpha / 2))
    print("  (this is where the standard '1.96' for alpha=0.05 comes from)")
    print("p-value        = 2*(1 - stats.norm.cdf(abs(Z0)))    ->", 2 * (1 - stats.norm.cdf(abs(Z0))))


def chisquare_shortcut():
    """scipy.stats.chisquare is a one-call shortcut when you already
    have observed bin counts and just want the statistic + p-value
    directly, skipping the manual sum((O-E)^2/E) loop."""
    observed = [8, 8, 10, 9, 12, 8, 10, 14, 10, 11]

    print("\nOne-call shortcut: stats.chisquare(observed_counts)")
    chi2_stat, p_value = stats.chisquare(observed)
    print(f"chi2={chi2_stat:.4f}  p={p_value:.4f}")
    print("(assumes equal expected counts per bin by default -- pass f_exp= for unequal)")


def main():
    chi_square_reference()
    ks_reference()
    normal_reference()
    chisquare_shortcut()


if __name__ == "__main__":
    main()
