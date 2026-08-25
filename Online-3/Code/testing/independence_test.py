"""
Three tests for INDEPENDENCE of a sequence of U(0,1) values (as opposed
to chi_square_test.py / ks_test.py, which test UNIFORMITY of individual
values -- a generator can pass those and still produce dependent,
predictable output; see rng/lcg.py's RANDU note).

1) Autocorrelation test
   Tests whether R_i, R_{i+l}, R_{i+2l}, ... (every l-th value, starting
   at position i) are correlated with each other.

       M         = largest integer such that i + (M+1)*l <= N
       rho_hat   = (1/(M+1)) * sum_{k=0}^{M} R_{i+k*l} * R_{i+(k+1)*l}  -  0.25
       sigma     = sqrt(13M + 7) / (12*(M+1))
       Z0        = rho_hat / sigma
       Decision  : reject H0 (independence) if |Z0| > z_{1-alpha/2}  (1.96 for alpha=0.05)

2) Runs test, above/below the mean (0.5)
   A "run" is a maximal streak of consecutive values all above 0.5, or
   all below 0.5. Too few runs -> long streaks -> positive autocorrelation.
   Too many runs -> constant alternation -> negative autocorrelation.

       n1, n2    = count of values above / below 0.5
       E[runs]   = 2*n1*n2/n + 1
       Var[runs] = 2*n1*n2*(2*n1*n2 - n) / (n^2 * (n - 1))
       Z0        = (runs - E[runs]) / sqrt(Var[runs])
       Decision  : same |Z0| > 1.96 rule as above.

3) Runs test, up and down
   A DIFFERENT runs test -- a "run" here is a maximal streak where
   consecutive values keep RISING, or keep FALLING (not about being
   above/below 0.5 at all). Catches a different kind of dependence than
   test 2 does, so use whichever the question specifically asks for --
   they are not interchangeable and can disagree on the same data (see
   the block-alternating example in this module's __main__ demo: a
   sequence that spends long stretches entirely above 0.5, then long
   stretches entirely below, but is locally patternless within each
   stretch, fails the above/below test badly while passing the up/down
   test cleanly -- the two tests are sensitive to different structure).

       runs      = count of maximal rising/falling streaks
       E[runs]   = (2n - 1) / 3
       Var[runs] = (16n - 29) / 90
       Z0        = (runs - E[runs]) / sqrt(Var[runs])
       Decision  : same |Z0| > 1.96 rule as above.

All three are two-sided normal tests, so scipy's `stats.norm.ppf`/`cdf`
replace the hardcoded 1.96 the same way they replace the Chi-Square and
K-S tables in the other two testing/ modules.

Run standalone:
    python -m testing.independence_test
"""

import math

from scipy import stats


def autocorrelation_test(R, i, lag, N, alpha=0.05):
    """
    R    : full sample (0-indexed list).
    i    : 1-based starting index into R.
    lag  : spacing `l` between the values being compared.
    N    : total sample size (usually len(R)).
    """
    M = (N - i) // lag - 1
    indices = [i - 1 + k * lag for k in range(M + 2)]  # convert to 0-based
    pair_sum = sum(R[indices[k]] * R[indices[k + 1]] for k in range(len(indices) - 1))

    rho_hat = (1 / (M + 1)) * pair_sum - 0.25
    sigma = math.sqrt(13 * M + 7) / (12 * (M + 1))
    z0 = rho_hat / sigma

    z_critical = stats.norm.ppf(1 - alpha / 2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z0)))
    decision = "Reject H0 (dependent)" if abs(z0) > z_critical else "Do not reject H0"

    return {"M": M, "rho_hat": rho_hat, "sigma": sigma, "Z0": z0,
            "critical_value": z_critical, "p_value": p_value, "decision": decision}


def runs_test(R, alpha=0.05):
    n = len(R)
    above = [1 if r > 0.5 else 0 for r in R]

    runs = 1
    for k in range(1, n):
        if above[k] != above[k - 1]:
            runs += 1

    n1 = sum(above)
    n2 = n - n1
    expected_runs = (2 * n1 * n2) / n + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
    z0 = (runs - expected_runs) / math.sqrt(var_runs)

    z_critical = stats.norm.ppf(1 - alpha / 2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z0)))
    decision = "Reject H0 (dependent)" if abs(z0) > z_critical else "Do not reject H0"

    return {"runs": runs, "expected_runs": expected_runs, "Z0": z0,
            "critical_value": z_critical, "p_value": p_value, "decision": decision}


def runs_up_down_test(R, alpha=0.05):
    """
    Runs test based on whether each value is higher or lower than the
    PREVIOUS value (rising/falling), not on being above/below 0.5. Catches
    dependence that runs_test() above can miss -- and misses some that it
    catches. See this module's __main__ for a verified example (a
    block-alternating sequence) where the two genuinely disagree.
    """
    n = len(R)
    rising = [R[k + 1] > R[k] for k in range(n - 1)]

    runs = 1
    for k in range(1, len(rising)):
        if rising[k] != rising[k - 1]:
            runs += 1

    expected_runs = (2 * n - 1) / 3
    var_runs = (16 * n - 29) / 90
    z0 = (runs - expected_runs) / math.sqrt(var_runs)

    z_critical = stats.norm.ppf(1 - alpha / 2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z0)))
    decision = "Reject H0 (dependent)" if abs(z0) > z_critical else "Do not reject H0"

    return {"runs": runs, "expected_runs": expected_runs, "Z0": z0,
            "critical_value": z_critical, "p_value": p_value, "decision": decision}


if __name__ == "__main__":
    sample = [0.23, 0.28, 0.33, 0.27, 0.05, 0.36, 0.72, 0.81, 0.44, 0.91,
              0.12, 0.67, 0.53, 0.39, 0.88]

    auto = autocorrelation_test(sample, i=1, lag=1, N=len(sample))
    print("Autocorrelation test (i=1, lag=1):")
    print(f"  rho_hat={auto['rho_hat']:.4f}  Z0={auto['Z0']:.4f}  -> {auto['decision']}")

    runs = runs_test(sample)
    print("\nRuns test (above/below 0.5):")
    print(f"  runs={runs['runs']}  expected={runs['expected_runs']:.4f}  "
          f"Z0={runs['Z0']:.4f}  -> {runs['decision']}")

    updown = runs_up_down_test(sample)
    print("\nRuns test (up/down):")
    print(f"  runs={updown['runs']}  expected={updown['expected_runs']:.4f}  "
          f"Z0={updown['Z0']:.4f}  -> {updown['decision']}")

    print("\n--- Why these two runs tests can disagree: a block-alternating sequence ---")
    import random as _random
    _random.seed(3)
    blocky = []
    for block in range(20):
        lo, hi = (0.5, 1.0) if block % 2 == 0 else (0.0, 0.5)
        blocky.extend(_random.uniform(lo, hi) for _ in range(10))
    # 20 blocks of 10 values, alternating which half of [0,1) each block is
    # drawn from -- but each value WITHIN a block is an independent draw
    # from that half, so locally it looks patternless.
    r1, r2 = runs_test(blocky), runs_up_down_test(blocky)
    print(f"  runs_test (above/below 0.5)  -> runs={r1['runs']} (expected {r1['expected_runs']:.0f})  -> {r1['decision']}")
    print(f"  runs_up_down_test            -> runs={r2['runs']} (expected {r2['expected_runs']:.0f})  -> {r2['decision']}")
    print("  Only 20 label-switches happen (one per block) vs ~101 expected under")
    print("  independence, so the above/below test rejects hard. But within each")
    print("  block the direction changes look statistically normal, so the up/down")
    print("  test sees nothing wrong. Same data, genuinely different verdicts --")
    print("  the two tests are sensitive to different kinds of dependence.")
