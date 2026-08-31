"""
04_independence_tests.py
========================
TEMPLATE: "Test whether a sequence of random numbers is INDEPENDENT."

Uniformity is only half the job. These tests look for patterns/order effects.

  A. Autocorrelation test  (slides 41-56)  <- the one derived in full in class
  B. Runs test, up-and-down                 (classic companion)
  C. Runs test, above/below the mean
  D. Serial test (2-D chi-square on pairs)
  E. Gap test
  F. Poker test
  G. Lag scatter / lattice check

The autocorrelation section reproduces the slide-56 worked example exactly,
so you can trust the code before applying it to new data.

Run:  python3 04_independence_tests.py
"""

import math

# ---------------------------------------------------------------------------
# z critical values (two-tailed at alpha => use alpha/2 in each tail)
# ---------------------------------------------------------------------------
Z = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}     # keyed by TOTAL alpha


def z_two_tailed(alpha=0.05):
    return Z.get(alpha, 1.96)


# ===========================================================================
# A. AUTOCORRELATION TEST  (RNG deck, slides 42-56)
# ===========================================================================

def autocorrelation_test(R, i=1, lag=1, alpha=0.05, verbose=True):
    """Test H0: rho_{i,lag} = 0 (no autocorrelation at this lag).

    R is 1-INDEXED in the slides: i = 3, lag = 5 means R3, R8, R13, ...

    Step 1  M = largest integer with  i + (M+1)*lag <= N
    Step 2  rho_hat = 1/(M+1) * sum_{k=0..M} R_{i+k*lag} * R_{i+(k+1)*lag} - 0.25
            (0.25 because E[R R'] = E[R]E[R'] = 1/2 * 1/2 under independence)
    Step 3  sigma = sqrt(13M + 7) / (12(M + 1))
            (from Var = (13M+7)/(144(M+1)^2); 7/144 per product, 1/48 per
             adjacent covariance -- adjacent products share one R)
    Step 4  Z0 = rho_hat / sigma
    Step 5  two-tailed: reject H0 if |Z0| > z_{alpha/2}
    """
    N = len(R)

    # largest M with i + (M+1)*lag <= N
    M = 0
    while i + (M + 2) * lag <= N:
        M += 1
    if i + lag > N:
        raise ValueError("lag/start too large for this sequence")

    def at(k):                      # 1-indexed access
        return R[k - 1]

    products, total = [], 0.0
    for k in range(M + 1):
        a, b = at(i + k * lag), at(i + (k + 1) * lag)
        products.append((i + k * lag, i + (k + 1) * lag, a, b, a * b))
        total += a * b

    avg = total / (M + 1)
    rho = avg - 0.25
    sigma = math.sqrt(13 * M + 7) / (12 * (M + 1))
    z0 = rho / sigma
    zc = z_two_tailed(alpha)

    if verbose:
        print(f"  H0: rho_({i},{lag}) = 0  (numbers {lag} apart are uncorrelated)")
        print(f"  H1: rho_({i},{lag}) != 0                       alpha = {alpha}")
        print(f"\n  N = {N}, start i = {i}, lag = {lag}")
        print(f"  M = largest integer with i + (M+1)*lag <= N  ->  "
              f"{i} + ({M}+1)*{lag} = {i + (M+1)*lag} <= {N}   =>  M = {M}")
        sub = [at(i + k * lag) for k in range(M + 2)]
        print(f"  Subsequence R_{i}, R_{i+lag}, ... = {sub}")
        print(f"\n  {'pair':>14}{'R_a':>10}{'R_b':>10}{'product':>12}")
        print("  " + "-" * 46)
        for ka, kb, a, b, pr in products:
            print(f"  (R_{ka:>2}, R_{kb:>2}){'':>2}{a:>10.4f}{b:>10.4f}{pr:>12.6f}")
        print("  " + "-" * 46)
        print(f"  {'sum':>14}{'':>20}{total:>12.6f}")
        print(f"\n  average product = {total:.6f} / {M+1} = {avg:.6f}")
        print(f"  rho_hat  = {avg:.6f} - 0.25 = {rho:.6f}")
        print(f"  sigma    = sqrt(13*{M} + 7) / (12*({M}+1)) "
              f"= sqrt({13*M+7}) / {12*(M+1)} = {sigma:.6f}")
        print(f"  Z0       = {rho:.6f} / {sigma:.6f} = {z0:.4f}")
        print(f"  z_({alpha}/2) = {zc}")
        if abs(z0) > zc:
            print(f"  DECISION: |Z0| = {abs(z0):.4f} > {zc}  ->  REJECT H0 "
                  f"({'positive' if z0 > 0 else 'negative'} autocorrelation)")
        else:
            print(f"  DECISION: |Z0| = {abs(z0):.4f} <= {zc}  ->  DO NOT REJECT H0 "
                  f"(no evidence of autocorrelation)")

    return {"M": M, "avg_product": avg, "rho": rho, "sigma": sigma,
            "z0": z0, "z_crit": zc, "reject": abs(z0) > zc}


def autocorrelation_scan(R, lags=range(1, 11), i=1, alpha=0.05):
    """Run the test at several lags and summarise -- watch for the
    multiple-testing trap when interpreting the result."""
    print(f"  {'lag':>5}{'M':>6}{'rho_hat':>12}{'sigma':>10}{'Z0':>10}"
          f"{'|Z0|>1.96?':>13}")
    print("  " + "-" * 56)
    n_reject = 0
    for lag in lags:
        try:
            r = autocorrelation_test(R, i=i, lag=lag, alpha=alpha, verbose=False)
        except (ValueError, ZeroDivisionError):
            continue
        flag = "REJECT" if r["reject"] else "ok"
        n_reject += r["reject"]
        print(f"  {lag:>5}{r['M']:>6}{r['rho']:>12.6f}{r['sigma']:>10.6f}"
              f"{r['z0']:>10.4f}{flag:>13}")
    k = len(list(lags))
    print("  " + "-" * 56)
    print(f"  {n_reject} rejection(s) out of {k} tests.")
    print(f"  Expected by chance if the generator is perfect: "
          f"{k * alpha:.1f}  (P(at least one) = {1-(1-alpha)**k:.2f})")
    return n_reject


# ===========================================================================
# B. RUNS TEST -- UP AND DOWN
# ===========================================================================

def runs_test_up_down(R, alpha=0.05, verbose=True):
    """Counts runs of consecutive increases/decreases.

        a      = number of runs
        E[a]   = (2N - 1) / 3
        Var(a) = (16N - 29) / 90
        Z0     = (a - E[a]) / sqrt(Var(a))    reject if |Z0| > z_{alpha/2}

    Too FEW runs  -> the sequence trends (positive dependence).
    Too MANY runs -> it oscillates (negative dependence).
    """
    N = len(R)
    signs = ''.join('+' if R[k] > R[k-1] else '-'
                    for k in range(1, N) if R[k] != R[k-1])
    a = 1 + sum(1 for k in range(1, len(signs)) if signs[k] != signs[k-1])
    mean = (2 * N - 1) / 3
    var = (16 * N - 29) / 90
    sd = math.sqrt(var)
    z0 = (a - mean) / sd
    zc = z_two_tailed(alpha)

    if verbose:
        print(f"  H0: the sequence is independent (runs occur at random)")
        print(f"  N = {N}")
        if N <= 60:
            print(f"  Sign pattern: {signs}")
        print(f"  observed runs a = {a}")
        print(f"  E[a]   = (2N - 1)/3   = (2*{N} - 1)/3 = {mean:.4f}")
        print(f"  Var(a) = (16N - 29)/90 = (16*{N} - 29)/90 = {var:.4f}")
        print(f"  sd     = {sd:.4f}")
        print(f"  Z0     = ({a} - {mean:.4f}) / {sd:.4f} = {z0:.4f}")
        print(f"  DECISION: |Z0| = {abs(z0):.4f} vs z = {zc}  ->  "
              f"{'REJECT H0' if abs(z0) > zc else 'DO NOT REJECT H0'}")
        if abs(z0) > zc:
            print(f"            ({'too few runs: trending/positively dependent'
                                  if a < mean else
                                  'too many runs: oscillating/negatively dependent'})")

    return {"runs": a, "mean": mean, "var": var, "z0": z0,
            "reject": abs(z0) > zc}


# ===========================================================================
# C. RUNS TEST -- ABOVE AND BELOW THE MEAN
# ===========================================================================

def runs_test_above_below(R, alpha=0.05, threshold=None, verbose=True):
    """
        n1 = # above threshold, n2 = # below,  b = number of runs
        E[b]   = 2*n1*n2/N + 1/2
        Var(b) = 2*n1*n2*(2*n1*n2 - N) / (N^2 (N - 1))
    """
    N = len(R)
    thr = 0.5 if threshold is None else threshold      # 0.5 for U(0,1)
    sym = ['A' if v > thr else 'B' for v in R]
    n1, n2 = sym.count('A'), sym.count('B')
    b = 1 + sum(1 for k in range(1, N) if sym[k] != sym[k-1])
    mean = 2 * n1 * n2 / N + 0.5
    var = 2 * n1 * n2 * (2 * n1 * n2 - N) / (N * N * (N - 1))
    sd = math.sqrt(var)
    z0 = (b - mean) / sd
    zc = z_two_tailed(alpha)

    if verbose:
        print(f"  H0: the sequence is independent")
        print(f"  threshold = {thr},  N = {N},  n1 (above) = {n1}, "
              f"n2 (below) = {n2}")
        if N <= 60:
            print(f"  Pattern: {''.join(sym)}")
        print(f"  observed runs b = {b}")
        print(f"  E[b]   = 2*{n1}*{n2}/{N} + 0.5 = {mean:.4f}")
        print(f"  Var(b) = {var:.6f},  sd = {sd:.4f}")
        print(f"  Z0     = {z0:.4f}")
        print(f"  DECISION: |Z0| = {abs(z0):.4f} vs z = {zc}  ->  "
              f"{'REJECT H0' if abs(z0) > zc else 'DO NOT REJECT H0'}")

    return {"runs": b, "n1": n1, "n2": n2, "mean": mean, "var": var,
            "z0": z0, "reject": abs(z0) > zc}


# ===========================================================================
# D. SERIAL TEST -- chi-square on non-overlapping PAIRS
# ===========================================================================

def serial_test(R, k=4, alpha=0.05, verbose=True):
    """Split [0,1)^2 into k x k cells; each non-overlapping pair (R_1,R_2),
    (R_3,R_4), ... should land uniformly.   df = k^2 - 1.

    Detects 2-D structure (the lattice defect) that 1-D tests cannot see.
    """
    pairs = [(R[j], R[j+1]) for j in range(0, len(R) - 1, 2)]
    counts = [[0]*k for _ in range(k)]
    for x, y in pairs:
        counts[min(int(x*k), k-1)][min(int(y*k), k-1)] += 1
    n = len(pairs)
    exp = n / (k*k)
    chi2 = sum((counts[a][b] - exp)**2 / exp for a in range(k) for b in range(k))
    df = k*k - 1
    crit = _chi2_crit(df, alpha)

    if verbose:
        print(f"  H0: consecutive pairs are uniform on the unit square")
        print(f"  {n} non-overlapping pairs, {k}x{k} = {k*k} cells, "
              f"E = {exp:.2f} per cell, df = {df}")
        if exp < 5:
            print(f"  !! WARNING: E = {exp:.2f} < 5, chi-square unreliable")
        print(f"\n  Cell counts (rows = first value, cols = second):")
        for row in counts:
            print("     " + " ".join(f"{c:5d}" for c in row))
        print(f"\n  chi2_0 = {chi2:.4f},  chi2_({alpha},{df}) = {crit:.4f}")
        print(f"  DECISION: {'REJECT H0' if chi2 > crit else 'DO NOT REJECT H0'}")

    return {"chi2": chi2, "df": df, "crit": crit, "reject": chi2 > crit,
            "counts": counts}


# ===========================================================================
# E. GAP TEST
# ===========================================================================

def gap_test(R, lo=0.0, hi=0.5, alpha=0.05, verbose=True):
    """Gaps between successive values landing in [lo, hi).

    P(in range) = p = hi - lo. Gap length g is geometric:
        P(gap = g) = (1-p)^g * p,  F(g) = 1 - (1-p)^(g+1)
    Compared with a K-S statistic over the cumulative distribution.
    """
    p = hi - lo
    gaps, count, started = [], 0, False
    for v in R:
        if lo <= v < hi:
            if started:
                gaps.append(count)
            started, count = True, 0
        elif started:
            count += 1
    if not gaps:
        print("  no gaps found")
        return None
    n = len(gaps)
    maxg = max(gaps)
    obs_cum, run = [], 0
    for g in range(maxg + 1):
        run += gaps.count(g)
        obs_cum.append(run / n)
    theo_cum = [1 - (1 - p) ** (g + 1) for g in range(maxg + 1)]
    D = max(abs(o - t) for o, t in zip(obs_cum, theo_cum))
    crit = ({0.10: 1.22, 0.05: 1.36, 0.01: 1.63}[alpha]) / math.sqrt(n)

    if verbose:
        print(f"  H0: gaps follow the geometric distribution implied by "
              f"independence")
        print(f"  Interval [{lo}, {hi}), p = {p}, number of gaps n = {n}, "
              f"max gap = {maxg}")
        print(f"\n  {'gap':>5}{'count':>8}{'F_obs':>10}{'F_theo':>10}{'|diff|':>10}")
        print("  " + "-" * 43)
        for g in range(min(maxg + 1, 15)):
            print(f"  {g:>5}{gaps.count(g):>8}{obs_cum[g]:>10.4f}"
                  f"{theo_cum[g]:>10.4f}{abs(obs_cum[g]-theo_cum[g]):>10.4f}")
        if maxg >= 15:
            print(f"  ... ({maxg - 14} more gap lengths)")
        print("  " + "-" * 43)
        print(f"\n  D = {D:.6f},  D_crit = {crit:.6f}")
        print(f"  DECISION: {'REJECT H0' if D > crit else 'DO NOT REJECT H0'}")

    return {"D": D, "crit": crit, "n_gaps": n, "reject": D > crit}


# ===========================================================================
# F. POKER TEST
# ===========================================================================

POKER_PROBS = {                    # 5 digits from {0..9} with replacement
    "all different":    0.3024,
    "one pair":         0.5040,
    "two pairs":        0.1080,
    "three of a kind":  0.0720,
    "full house":       0.0090,
    "four of a kind":   0.0045,
    "five of a kind":   0.0001,
}
_PATTERN = {(1,1,1,1,1): "all different", (2,1,1,1): "one pair",
            (2,2,1): "two pairs", (3,1,1): "three of a kind",
            (3,2): "full house", (4,1): "four of a kind",
            (5,): "five of a kind"}


def poker_test(R, alpha=0.05, verbose=True):
    """Classify each number's first 5 decimal digits by its repeat pattern."""
    from collections import Counter
    cats = []
    for v in R:
        s = f"{v:.5f}"[2:7]
        pat = tuple(sorted(Counter(s).values(), reverse=True))
        cats.append(_PATTERN[pat])
    obs = Counter(cats)
    N = len(R)
    chi2, rows = 0.0, []
    for name, prob in POKER_PROBS.items():
        o, e = obs.get(name, 0), N * prob
        term = (o - e) ** 2 / e if e > 0 else 0.0
        chi2 += term
        rows.append((name, o, e, term))
    df = len(POKER_PROBS) - 1
    crit = _chi2_crit(df, alpha)

    if verbose:
        print(f"  H0: the digits of each number are independent")
        print(f"  N = {N} five-digit groups\n")
        print(f"  {'category':<18}{'O':>7}{'E':>10}{'(O-E)^2/E':>12}")
        print("  " + "-" * 47)
        for name, o, e, term in rows:
            print(f"  {name:<18}{o:>7}{e:>10.2f}{term:>12.4f}")
        print("  " + "-" * 47)
        print(f"  {'chi2_0':<18}{'':>17}{chi2:>12.4f}")
        print(f"\n  chi2_({alpha},{df}) = {crit:.4f}")
        print(f"  DECISION: {'REJECT H0' if chi2 > crit else 'DO NOT REJECT H0'}")
        if any(e < 5 for _, _, e, _ in rows):
            print("  !! some expected counts < 5 -- consider pooling rare categories")

    return {"chi2": chi2, "df": df, "crit": crit, "reject": chi2 > crit}


# ===========================================================================
# G. LAG SCATTER / LATTICE CHECK
# ===========================================================================

def lattice_check(R, a=None, m=None, n_pairs=2000):
    """For an LCG, consecutive pairs lie on a few parallel lines:
        U_{n+1} = a*U_n - k   for integer k in {0, ..., a-1}
    Counting the distinct k values exposes the structure immediately."""
    if a is None:
        print("  (pass a= the multiplier to run the algebraic check)")
        return
    ks = sorted({round(a * R[i] - R[i+1]) for i in range(min(n_pairs, len(R)-1))})
    print(f"  U_(n+1) = {a}*U_n - k. Over {min(n_pairs, len(R)-1)} pairs, "
          f"k takes {len(ks)} distinct values: {ks if len(ks) <= 20 else ks[:20]}")
    print(f"  => all points lie on {len(ks)} parallel lines rather than "
          f"filling the square.")
    print(f"  A good generator would show no such algebraic relation.")


def _chi2_crit(df, alpha):
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(1 - alpha, df))
    except ImportError:
        table = {3: 7.81, 8: 15.51, 9: 16.92, 15: 25.00, 24: 36.42,
                 6: 12.59, 99: 123.23}
        return table.get(df, 3.84)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 78)
    print("INDEPENDENCE TESTS")
    print("=" * 78)

    # -- Slide 56 verification --------------------------------------------
    R30 = [0.12, 0.01, 0.23, 0.28, 0.89, 0.31, 0.64, 0.28, 0.83, 0.93,
           0.99, 0.15, 0.33, 0.35, 0.91, 0.41, 0.60, 0.27, 0.75, 0.88,
           0.68, 0.49, 0.05, 0.43, 0.95, 0.58, 0.19, 0.36, 0.69, 0.87]

    print("\n### A. AUTOCORRELATION -- verification vs slide 56 ###")
    print("  (test the 3rd, 8th, 13th, ... numbers: i = 3, lag = 5, N = 30)\n")
    r = autocorrelation_test(R30, i=3, lag=5, alpha=0.05)
    print(f"\n  Slide says: M = 4, rho = -0.1945, sigma = 0.1280, Z0 = -1.52,")
    print(f"              do not reject.  Code agrees: "
          f"{r['M']==4 and abs(r['rho']+0.1945)<1e-4 and abs(r['z0']+1.52)<0.01}")

    # -- The planted defect from slide 41 ----------------------------------
    print("\n\n### Catching the PLANTED defect from slide 41 ###")
    print("  Slide 41 notes that every 5th number starting at position 5 is")
    print("  large: 0.89, 0.93, 0.91, 0.88, 0.95, 0.87. The overall")
    print("  distribution still looks uniform, so a uniformity test sees")
    print("  nothing. Test i = 5, lag = 5:\n")
    r2 = autocorrelation_test(R30, i=5, lag=5, alpha=0.05)
    print(f"\n  -> The autocorrelation test DOES find it (|Z0| = "
          f"{abs(r2['z0']):.2f} > 1.96), while i = 3 on the same data found")
    print("     nothing. The defect is in a specific subsequence, which is")
    print("     exactly why we choose i and lag deliberately.")

    # -- Scan many lags ----------------------------------------------------
    print("\n\n### Autocorrelation scan over lags 1..8 (same data) ###\n")
    autocorrelation_scan(R30, lags=range(1, 9), i=1)

    # -- Runs tests --------------------------------------------------------
    print("\n\n### B. RUNS TEST (up and down) ###\n")
    runs_test_up_down(R30)

    print("\n\n### C. RUNS TEST (above/below 0.5) ###\n")
    runs_test_above_below(R30)

    # -- On a real generator -----------------------------------------------
    def lcg(n, m=2**31 - 1, a=16807, c=0, x0=123457):
        out, x = [], x0
        for _ in range(n):
            x = (a*x + c) % m
            out.append(x / m)
        return out

    good = lcg(5000)
    print("\n\n### D/E/F APPLIED TO A GOOD LCG (a=16807, m=2^31-1, N=5000) ###")
    print("\n  --- Serial test ---")
    serial_test(good, k=4)
    print("\n  --- Gap test ---")
    gap_test(good, 0.0, 0.5)
    print("\n  --- Poker test ---")
    poker_test(good)

    # -- Contrast with a bad generator -------------------------------------
    print("\n\n### G. THE SAME TESTS ON A BAD GENERATOR (a=5, m=2^16) ###")
    bad = lcg(5000, m=65536, a=5, c=0, x0=1)
    print("\n  --- Serial test (this is what catches it) ---")
    serial_test(bad, k=4)
    print("\n  --- Lattice check ---")
    lattice_check(bad, a=5)

    print("\n\n### SUMMARY: what each test detects ###\n")
    print("  Autocorrelation : linear dependence at a chosen lag")
    print("  Runs up/down    : trends and oscillation in the ORDER")
    print("  Runs above/below: clustering on one side of the mean")
    print("  Serial          : 2-D structure / lattice (1-D tests miss this)")
    print("  Gap             : whether hits on a sub-interval are spread properly")
    print("  Poker           : dependence between the DIGITS of each number")
    print("\n  Remember (slides 30-33): with k tests at alpha, P(>=1 false")
    print("  positive) = 1 - (1-alpha)^k. One isolated failure proves nothing;")
    print("  repeated failures of the same test on fresh sequences do.")


if __name__ == "__main__":
    main()
