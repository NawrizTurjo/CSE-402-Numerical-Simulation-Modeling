from scipy import stats

def ks_test(sample, alpha=0.05):
    N = len(sample)
    s = sorted(sample)

    D_plus  = max(i/N - s[i-1] for i in range(1, N+1))
    D_minus = max(s[i-1] - (i-1)/N for i in range(1, N+1))
    D = max(D_plus, D_minus)

    # scipy replaces the entire hardcoded table — works for ANY N and alpha
    D_critical = stats.ksone.ppf(1 - alpha, N)

    # p-value: probability of seeing D this large or larger by chance
    p_value = 1 - stats.ksone.cdf(D, N)

    decision = "Reject H0 (NOT uniform)" if D > D_critical else "Fail to Reject H0 (uniform)"

    return D, D_plus, D_minus, D_critical, p_value, decision

# Slide's example
D, Dp, Dm, Dc, p, dec = ks_test([0.44, 0.81, 0.14, 0.05, 0.93])
print(f"D+={Dp:.4f}  D-={Dm:.4f}  D={D:.4f}  D_crit={Dc:.4f}  p={p:.4f}")
print(f"Decision: {dec}")
# D+=0.2600  D-=0.2100  D=0.2600  D_crit=0.5094  p=0.1877
# Decision: Fail to Reject H0 (uniform)  ✅ matches slide