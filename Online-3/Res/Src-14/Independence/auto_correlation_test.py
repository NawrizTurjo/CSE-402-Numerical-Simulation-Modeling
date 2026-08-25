from scipy import stats
import math

def auto_correlation_test(R, i, l, N):
    """
    Tests if R[i], R[i+l], R[i+2l], ... are correlated.
    i: starting index (1-based)
    l: lag
    N: total sample size
    """
    M = (N - i) // l - 1
    idx = [i - 1 + k * l for k in range(M + 2)]
    summ = sum(R[idx[k]] * R[idx[k+1]] for k in range(len(idx)-1))
    rho_hat = (1 / (M + 1)) * summ - 0.25
    sigma = math.sqrt(13 * M + 7) / (12 * (M + 1))
    Z0 = rho_hat / sigma
    decision = "Reject H0 (DEPENDENT)" if abs(Z0) > 1.96 else "Fail to Reject H0"
    return rho_hat, Z0, decision


def main():
    sample = [
        0.23, 0.28, 0.33, 0.27, 0.05,
        0.36, 0.72, 0.81, 0.44, 0.91,
        0.12, 0.67, 0.53, 0.39, 0.88,
    ]

    i = 1
    l = 1
    N = len(sample)

    rho_hat, z0, decision = auto_correlation_test(sample, i, l, N)

    print("Autocorrelation Test for Independence")
    print("Sample values:", sample)
    print(f"Starting index i: {i}")
    print(f"Lag l: {l}")
    print(f"Sample size N: {N}")
    print(f"rho_hat: {rho_hat:.4f}")
    print(f"Z0: {z0:.4f}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()


## using scipy
alpha = 0.05
Z0 = 1.96  # for alpha=0.05, two-tailed
z_critical = stats.norm.ppf(1 - alpha/2)          # replaces "1.96"
p_value    = 2 * (1 - stats.norm.cdf(abs(Z0)))     # exact p-value, extra credit if asked



# run multiple lags and report how many reject independence
def test_multiple_lags(R, N, lags=[1,2,3,5,10], alpha=0.05):
    results = []
    for l in lags:
        rho, Z0, dec = auto_correlation_test(R, i=1, l=l, N=N)
        results.append((l, rho, Z0, dec))
        print(f"lag={l:>3}: rho={rho:.4f}  Z0={Z0:.4f}  {dec}")
    n_rejected = sum(1 for _, _, _, d in results if "Reject" in d)
    print(f"\n{n_rejected}/{len(lags)} lags rejected independence")
    if n_rejected / len(lags) > 0.15:  # well above the expected ~5%
        print("Pattern of repeated failures — likely genuine dependence")
    else:
        print("Occasional failures consistent with normal false-positive rate")
    return results