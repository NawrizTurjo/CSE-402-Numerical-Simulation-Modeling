from scipy import stats
alpha = 0.05
chi2_stat = 15.51  # example chi-square statistic
df = 9             # degrees of freedom for 10 bins


critical_value = stats.chi2.ppf(1 - alpha, df=bins-1)   # df = 10-1 = 9
p_value        = stats.chi2.sf(chi2, df=bins-1)          # df = 10-1 = 9


# For chi-square test
stats.chi2.ppf(1 - alpha, df)      # critical value
stats.chi2.sf(chi2_stat, df)       # p-value

# For K-S test
stats.ksone.ppf(1 - alpha, N)      # critical value
1 - stats.ksone.cdf(D, N)          # p-value

# For autocorrelation test (normal distribution)
stats.norm.ppf(1 - alpha/2)        # critical value (always 1.96 for alpha=0.05)
2 * (1 - stats.norm.cdf(abs(Z0)))  # p-value