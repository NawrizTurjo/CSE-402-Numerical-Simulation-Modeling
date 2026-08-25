import random
from scipy import stats

def python_random_generator(seed, n):
    random.seed(seed)
    return [random.random() for _ in range(n)]


def chi_square_test(seed, n=1000, bins=10, alpha=0.05):
    normalized = python_random_generator(seed, n)

    E = n / bins
    counts = [0] * bins
    for r in normalized:
        idx = min(bins - 1, int(r * bins))
        counts[idx] += 1

    chi2 = sum((counts[i] - E)**2 / E for i in range(bins))

    critical_value = stats.chi2.ppf(1 - alpha, df=bins-1)  # replaces table
    p_value        = stats.chi2.sf(chi2,        df=bins-1)  # exact p-value

    decision = "Reject H0" if chi2 > critical_value else "Fail to Reject H0"
    return chi2, critical_value, p_value, decision
