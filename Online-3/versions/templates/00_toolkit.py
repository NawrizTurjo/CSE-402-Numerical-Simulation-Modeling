"""
00_toolkit.py  --  CSE401/402 Numerical Lab: one-stop reusable toolkit.

Everything here is pure-Python (no `random` module) except the optional SciPy
helpers, which are clearly marked. Import from here, or copy the 10 lines you
need straight into your answer file.

    from importlib import import_module
    tk = import_module("00_toolkit")        # module name starts with a digit
    # ...or just rename this file to toolkit.py and `import toolkit as tk`

CONTENTS
  1. Generators ................ LCG, MCG, middle-square, and friends
  2. Period / cycle detection .. brute force + Floyd, max-period conditions
  3. Uniformity tests .......... chi-square, Kolmogorov-Smirnov (manual + scipy)
  4. Independence tests ........ autocorrelation, runs, serial, gap, poker
  5. Critical value tables ..... KS table, chi-square table, z table
  6. Random variates ........... inverse transform: exp, uniform, discrete,
                                 normal (Box-Muller), geometric, poisson
  7. Monte Carlo ............... integration, pi, Buffon, batch means / CI
  8. Reporting ................. pretty ASCII tables
"""

import math

# ===========================================================================
# 1. GENERATORS
# ===========================================================================

class LCG:
    """Linear Congruential Generator:  X_{n+1} = (a*X_n + c) mod m

    c != 0 -> mixed congruential
    c == 0 -> multiplicative congruential (MCG)

    Usage:
        g = LCG(m=2**31 - 1, a=16807, c=0, seed=123457)
        u = g.u()                 # one U(0,1)
        us = g.uniforms(1000)     # list of 1000
    """

    def __init__(self, m, a, c, seed):
        self.m, self.a, self.c = m, a, c
        self.seed = seed
        self.x = seed

    def reset(self, seed=None):
        self.x = self.seed if seed is None else seed
        return self

    def next_int(self):
        """Advance the state and return the new integer X_{n+1}."""
        self.x = (self.a * self.x + self.c) % self.m
        return self.x

    def u(self):
        """Next U(0,1) value: U_n = X_n / m."""
        return self.next_int() / self.m

    def integers(self, n):
        return [self.next_int() for _ in range(n)]

    def uniforms(self, n):
        return [self.u() for _ in range(n)]


def lcg_generator(n_samples, m=65536, a=5, X0=1, c=0):
    """Bare-function form, the signature most assignments ask for.

    Returns a list of n_samples U(0,1) values.
    NOTE: the first value returned corresponds to X_1 (the seed X0 is *not*
    emitted). If your problem wants X0 included, prepend X0/m.
    """
    out, x = [], X0
    for _ in range(n_samples):
        x = (a * x + c) % m
        out.append(x / m)
    return out


def lcg_integers(n, m, a, c, X0, include_seed=False):
    """Same as above but returns the raw integer stream X_i."""
    out = [X0] if include_seed else []
    x = X0
    for _ in range(n):
        x = (a * x + c) % m
        out.append(x)
    return out


def middle_square(seed, n, digits=4):
    """Von Neumann middle-square method.

    Square the value, zero-pad to 2*digits, take the middle `digits` digits.
    Returns a list of n integers (the seed is NOT included).

        5731^2 = 32844361 -> middle 4 -> 8443
        8443^2 = 71284249 -> middle 4 -> 2842
    """
    out, x = [], seed
    width = 2 * digits
    for _ in range(n):
        s = str(x * x).zfill(width)          # pad to 8 digits for 4-digit seeds
        start = (width - digits) // 2        # = 2 when width=8, digits=4
        x = int(s[start:start + digits])
        out.append(x)
    return out


def middle_square_uniforms(seed, n, digits=4):
    """Middle-square values scaled to [0,1) by dividing by 10**digits."""
    return [v / (10 ** digits) for v in middle_square(seed, n, digits)]


# ---------------------------------------------------------------------------
# Extra generators that occasionally show up
# ---------------------------------------------------------------------------

def additive_congruential(seeds, n, m):
    """X_i = (X_{i-1} + X_{i-k}) mod m  (lagged Fibonacci, k = len(seeds))."""
    k = len(seeds)
    x = list(seeds)
    for i in range(k, k + n):
        x.append((x[i - 1] + x[i - k]) % m)
    return x[k:]


def combined_lcg(n, m1=2147483563, a1=40014, m2=2147483399, a2=40692,
                 s1=12345, s2=67890):
    """L'Ecuyer's combined LCG -- period ~ 2.3e18. Good 'better generator' demo."""
    out = []
    for _ in range(n):
        s1 = (a1 * s1) % m1
        s2 = (a2 * s2) % m2
        z = (s1 - s2) % (m1 - 1)
        out.append(z / m1 if z > 0 else (m1 - 1) / m1)
    return out


# ===========================================================================
# 2. PERIOD / CYCLE DETECTION
# ===========================================================================

def find_period(m, a, c, X0, max_steps=None):
    """Period of the LCG state sequence starting from X0.

    Returns (period, tail) where `tail` (mu) is the number of steps before the
    cycle is entered. For a full-period LCG, tail == 0 and period == m.

    Uses a dict of first-occurrence indices -> O(period + tail) time.
    """
    if max_steps is None:
        max_steps = m + 5
    seen = {X0: 0}
    x, i = X0, 0
    while i < max_steps:
        x = (a * x + c) % m
        i += 1
        if x in seen:
            return i - seen[x], seen[x]
        seen[x] = i
    return None, None       # no repeat found within max_steps


def find_period_floyd(step, x0, max_steps=10 ** 8):
    """Floyd's tortoise-and-hare -- O(1) memory. `step` is a function x -> x'.

    Returns (period lambda, tail mu).
    """
    tortoise, hare = step(x0), step(step(x0))
    n = 0
    while tortoise != hare and n < max_steps:
        tortoise, hare = step(tortoise), step(step(hare))
        n += 1
    # find tail mu
    mu = 0
    tortoise = x0
    while tortoise != hare:
        tortoise, hare = step(tortoise), step(hare)
        mu += 1
    # find period lam
    lam, hare = 1, step(tortoise)
    while tortoise != hare:
        hare = step(hare)
        lam += 1
    return lam, mu


def first_repeat(seq):
    """For a generated list, return (value, index_first_seen, index_repeat,
    cycle_length). Useful for the middle-square 'find the bad seed' task."""
    seen = {}
    for i, v in enumerate(seq):
        if v in seen:
            return v, seen[v], i, i - seen[v]
        seen[v] = i
    return None, None, None, None


def is_full_period_mixed(m, a, c):
    """Hull-Dobell theorem: mixed LCG (c != 0) has full period m iff
       (1) gcd(c, m) == 1
       (2) a - 1 divisible by every prime factor of m
       (3) if 4 | m then 4 | (a - 1)
    Returns (bool, list_of_reason_strings).
    """
    reasons = []
    ok = True
    if math.gcd(c, m) != 1:
        ok = False
        reasons.append(f"FAIL gcd(c,m) = {math.gcd(c, m)} != 1")
    else:
        reasons.append("OK   gcd(c,m) = 1")

    for p in prime_factors(m):
        if (a - 1) % p != 0:
            ok = False
            reasons.append(f"FAIL a-1 = {a-1} not divisible by prime factor {p}")
        else:
            reasons.append(f"OK   a-1 = {a-1} divisible by prime factor {p}")

    if m % 4 == 0:
        if (a - 1) % 4 != 0:
            ok = False
            reasons.append(f"FAIL 4 | m but 4 does not divide a-1 = {a-1}")
        else:
            reasons.append(f"OK   4 | m and 4 | a-1")
    return ok, reasons


def max_period_report(m, a, c, X0):
    """The three slide cases for maximum period (Slide 22 of the RNG deck)."""
    lines = []
    b = int(math.log2(m)) if m > 0 and (m & (m - 1)) == 0 else None

    if c != 0:
        lines.append("Case 1: mixed congruential (c != 0)")
        ok, reasons = is_full_period_mixed(m, a, c)
        lines += ["    " + r for r in reasons]
        lines.append(f"    => theoretical max period = m = {m}" if ok
                     else "    => full period m NOT guaranteed")
        if b is not None:
            lines.append(f"    (slide form: m = 2^{b}, need c,m coprime and a = 1+4k"
                         f"; a-1 = {a-1}, (a-1)%4 = {(a-1)%4})")
    else:
        if b is not None:
            lines.append(f"Case 2: multiplicative, m = 2^{b}, c = 0")
            lines.append(f"    max period = m/4 = {m // 4}")
            lines.append(f"    need X0 odd            : X0 = {X0} -> "
                         f"{'OK' if X0 % 2 == 1 else 'FAIL (even seed => shorter period)'}")
            lines.append(f"    need a = 3+8k or 5+8k  : a mod 8 = {a % 8} -> "
                         f"{'OK' if a % 8 in (3, 5) else 'FAIL'}")
        elif is_prime(m):
            lines.append(f"Case 3: multiplicative, m = {m} is prime, c = 0")
            lines.append(f"    max period = m - 1 = {m - 1}")
            lines.append(f"    requires a to be a primitive root mod m: "
                         f"{'YES' if is_primitive_root(a, m) else 'NO'}")
        else:
            lines.append("Multiplicative with m neither 2^b nor prime -- "
                         "no simple slide rule; compute the period numerically.")
    return "\n".join(lines)


def prime_factors(n):
    """Distinct prime factors of n."""
    fs, d = set(), 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d)
            n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return sorted(fs)


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def is_primitive_root(a, m):
    """True if a is a primitive root modulo prime m (i.e. order of a is m-1)."""
    if not is_prime(m):
        return False
    order = m - 1
    for p in prime_factors(order):
        if pow(a, order // p, m) == 1:
            return False
    return True


# ===========================================================================
# 3. UNIFORMITY TESTS
# ===========================================================================

def chi_square_uniform(values, n_bins=10, lo=0.0, hi=1.0):
    """Chi-square goodness-of-fit against Uniform[lo, hi).

        chi2_0 = sum_i (O_i - E_i)^2 / E_i,   E_i = N / n_bins
        df = n_bins - 1     (no parameters estimated)

    Returns dict with observed, expected, chi2, df, and the per-bin table.
    Pure Python -- no SciPy needed for the statistic itself.
    """
    N = len(values)
    width = (hi - lo) / n_bins
    obs = [0] * n_bins
    for v in values:
        k = int((v - lo) / width)
        k = min(max(k, 0), n_bins - 1)      # clamp: v == hi lands in last bin
        obs[k] += 1
    exp = N / n_bins
    terms = [(o - exp) ** 2 / exp for o in obs]
    return {
        "N": N, "n_bins": n_bins,
        "observed": obs, "expected": exp,
        "terms": terms,
        "chi2": sum(terms),
        "df": n_bins - 1,
        "edges": [lo + i * width for i in range(n_bins + 1)],
    }


def ks_test_uniform(values):
    """One-sample two-sided Kolmogorov-Smirnov against U(0,1), by hand.

        D+ = max_i ( i/N   - R_(i) )
        D- = max_i ( R_(i) - (i-1)/N )
        D  = max(D+, D-)

    Returns dict with d_plus, d_minus, d, and the argmax indices (1-based).
    """
    R = sorted(values)
    N = len(R)
    d_plus = d_minus = -1.0
    i_plus = i_minus = 0
    for i in range(1, N + 1):
        a = i / N - R[i - 1]
        b = R[i - 1] - (i - 1) / N
        if a > d_plus:
            d_plus, i_plus = a, i
        if b > d_minus:
            d_minus, i_minus = b, i
    return {
        "N": N,
        "d_plus": d_plus, "i_plus": i_plus,
        "d_minus": d_minus, "i_minus": i_minus,
        "d": max(d_plus, d_minus),
        "sorted": R,
    }


def ks_critical(N, alpha=0.05):
    """Critical value D_alpha for the one-sample K-S test.

    Small N: exact table (Miller). Large N (> 40): asymptotic c(alpha)/sqrt(N)
    with c(0.10)=1.22, c(0.05)=1.36, c(0.01)=1.63.
    """
    table = {
        0.10: [.950, .776, .642, .564, .510, .470, .438, .411, .388, .368,
               .352, .338, .325, .314, .304, .295, .286, .278, .272, .264,
               .258, .252, .247, .242, .238, .233, .229, .225, .221, .218,
               .214, .211, .208, .205, .202, .199, .196, .194, .191, .189],
        0.05: [.975, .842, .708, .624, .565, .521, .486, .457, .432, .410,
               .391, .375, .361, .349, .338, .328, .318, .309, .301, .294,
               .287, .281, .275, .269, .264, .259, .254, .250, .246, .242,
               .238, .234, .231, .227, .224, .221, .218, .215, .213, .210],
        0.01: [.995, .929, .828, .733, .669, .618, .577, .543, .514, .490,
               .468, .450, .433, .418, .404, .392, .381, .371, .363, .356,
               .348, .341, .334, .327, .320, .313, .307, .301, .295, .290,
               .285, .281, .277, .273, .269, .265, .262, .258, .255, .252],
    }
    c = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}
    if N <= 40 and alpha in table:
        return table[alpha][N - 1]
    return c.get(alpha, 1.36) / math.sqrt(N)


# --- optional SciPy wrappers (for p-values) --------------------------------

def chi_square_pvalue(chi2, df):
    """p-value via SciPy; falls back to a pure-Python survival function."""
    try:
        from scipy.stats import chi2 as _chi2
        return float(_chi2.sf(chi2, df))
    except ImportError:
        return _chi2_sf_fallback(chi2, df)


def _chi2_sf_fallback(x, k):
    """Upper tail of chi-square with k df, via the regularized incomplete gamma."""
    return _gammaincc(k / 2.0, x / 2.0)


def _gammaincc(s, x):
    """Regularized upper incomplete gamma Q(s,x). Good enough for tables."""
    if x < 0 or s <= 0:
        return float("nan")
    if x == 0:
        return 1.0
    if x < s + 1:                              # series for P(s,x)
        term = 1.0 / s
        total = term
        n = 1
        while n < 1000:
            term *= x / (s + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
            n += 1
        return 1.0 - total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    # continued fraction for Q(s,x)
    tiny = 1e-300
    b = x + 1 - s
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h * math.exp(-x + s * math.log(x) - math.lgamma(s))


def ks_pvalue(d, N):
    """Asymptotic K-S p-value: Q_KS(sqrt(N)*D)."""
    lam = (math.sqrt(N) + 0.12 + 0.11 / math.sqrt(N)) * d
    total, sign = 0.0, 1
    for j in range(1, 101):
        total += sign * math.exp(-2.0 * j * j * lam * lam)
        sign = -sign
    return max(0.0, min(1.0, 2.0 * total))


# ===========================================================================
# 4. INDEPENDENCE TESTS
# ===========================================================================

def autocorrelation_test(R, i, lag, alpha=0.05, one_indexed=True):
    """Slide-exact autocorrelation test.

        M   = largest integer with  i + (M+1)*lag <= N
        rho = 1/(M+1) * sum_{k=0..M} R_{i+k*lag} R_{i+(k+1)*lag}  -  0.25
        sigma = sqrt(13M + 7) / (12(M+1))
        Z0  = rho / sigma          reject H0 if |Z0| > z_{alpha/2}

    `R` is the full sequence. `i` and `lag` follow the slides' 1-based
    convention by default (i=3, lag=5 means R3, R8, R13, ...).
    """
    N = len(R)
    idx0 = (lambda k: k - 1) if one_indexed else (lambda k: k)

    M = 0
    while i + (M + 2) * lag <= N:
        M += 1
    # M is the largest int with i + (M+1)*lag <= N

    total = 0.0
    pairs = []
    for k in range(M + 1):
        a = R[idx0(i + k * lag)]
        b = R[idx0(i + (k + 1) * lag)]
        pairs.append((a, b, a * b))
        total += a * b

    rho = total / (M + 1) - 0.25
    sigma = math.sqrt(13 * M + 7) / (12 * (M + 1))
    z0 = rho / sigma
    zcrit = z_critical(alpha / 2)
    return {
        "M": M, "n_pairs": M + 1, "pairs": pairs,
        "avg_product": total / (M + 1),
        "rho": rho, "sigma": sigma, "z0": z0,
        "z_crit": zcrit,
        "reject": abs(z0) > zcrit,
        "subsequence": [R[idx0(i + k * lag)] for k in range(M + 2)],
    }


def runs_test_updown(R, alpha=0.05):
    """Runs test for independence (runs up and down).

        a       = observed number of runs
        E[a]    = (2N - 1) / 3
        Var(a)  = (16N - 29) / 90
        Z0      = (a - E[a]) / sd     reject if |Z0| > z_{alpha/2}
    """
    N = len(R)
    signs = []
    for k in range(1, N):
        if R[k] > R[k - 1]:
            signs.append('+')
        elif R[k] < R[k - 1]:
            signs.append('-')
        # exact ties are dropped (measure zero for continuous data)
    a = 1
    for k in range(1, len(signs)):
        if signs[k] != signs[k - 1]:
            a += 1
    mean = (2 * N - 1) / 3
    var = (16 * N - 29) / 90
    sd = math.sqrt(var)
    z0 = (a - mean) / sd
    zc = z_critical(alpha / 2)
    return {"N": N, "runs": a, "mean": mean, "var": var, "sd": sd,
            "z0": z0, "z_crit": zc, "reject": abs(z0) > zc, "signs": "".join(signs)}


def runs_test_above_below(R, alpha=0.05, threshold=None):
    """Runs above and below the mean (or a given threshold).

        n1 = # above, n2 = # below, b = number of runs
        E[b]   = 2*n1*n2/N + 1/2
        Var(b) = 2*n1*n2*(2*n1*n2 - N) / (N^2 * (N-1))
    """
    N = len(R)
    thr = (sum(R) / N) if threshold is None else threshold
    sym = ['A' if v > thr else 'B' for v in R]
    n1 = sym.count('A')
    n2 = sym.count('B')
    b = 1
    for k in range(1, N):
        if sym[k] != sym[k - 1]:
            b += 1
    mean = 2 * n1 * n2 / N + 0.5
    var = 2 * n1 * n2 * (2 * n1 * n2 - N) / (N * N * (N - 1))
    sd = math.sqrt(var) if var > 0 else float('nan')
    z0 = (b - mean) / sd
    zc = z_critical(alpha / 2)
    return {"N": N, "threshold": thr, "n1": n1, "n2": n2, "runs": b,
            "mean": mean, "var": var, "sd": sd, "z0": z0, "z_crit": zc,
            "reject": abs(z0) > zc}


def serial_test_2d(R, n_bins=10, alpha=0.05):
    """Serial test: chi-square on non-overlapping consecutive PAIRS.

    Splits [0,1)^2 into n_bins^2 cells; expects N/2 pairs spread evenly.
    df = n_bins^2 - 1
    """
    pairs = [(R[k], R[k + 1]) for k in range(0, len(R) - 1, 2)]
    k = n_bins
    counts = [[0] * k for _ in range(k)]
    for x, y in pairs:
        ix = min(int(x * k), k - 1)
        iy = min(int(y * k), k - 1)
        counts[ix][iy] += 1
    n_pairs = len(pairs)
    exp = n_pairs / (k * k)
    chi2 = sum((counts[a][b] - exp) ** 2 / exp for a in range(k) for b in range(k))
    df = k * k - 1
    return {"n_pairs": n_pairs, "expected": exp, "chi2": chi2, "df": df,
            "counts": counts, "p": chi_square_pvalue(chi2, df)}


def gap_test(R, lo=0.0, hi=0.5, max_gap=None, alpha=0.05):
    """Gap test: distribution of gaps between successive values in [lo, hi).

    P(hit) = p = hi - lo. Gap length g has P(g) = (1-p)^g * p, so the
    cumulative F(g) = 1 - (1-p)^(g+1). Compared with a K-S style statistic.
    """
    p = hi - lo
    gaps, count = [], 0
    started = False
    for v in R:
        if lo <= v < hi:
            if started:
                gaps.append(count)
            started = True
            count = 0
        elif started:
            count += 1
    if not gaps:
        return None
    n = len(gaps)
    if max_gap is None:
        max_gap = max(gaps)
    obs_cum, running = [], 0
    for g in range(max_gap + 1):
        running += gaps.count(g)
        obs_cum.append(running / n)
    theo_cum = [1 - (1 - p) ** (g + 1) for g in range(max_gap + 1)]
    d = max(abs(o - t) for o, t in zip(obs_cum, theo_cum))
    return {"n_gaps": n, "p": p, "D": d, "D_crit": ks_critical(n, alpha),
            "reject": d > ks_critical(n, alpha),
            "obs_cum": obs_cum, "theo_cum": theo_cum}


def poker_test(R, digits=5, alpha=0.05):
    """Poker test on `digits`-digit groups (default 5 digits, classic form).

    Classifies each group by its pattern of repeated digits and runs a
    chi-square against the theoretical pattern probabilities.
    """
    from collections import Counter
    groups = []
    for v in R:
        s = str(int(v * 10 ** digits)).zfill(digits)[:digits]
        groups.append(s)
    if digits != 5:
        raise ValueError("theoretical probabilities below are for digits=5")

    def category(s):
        c = sorted(Counter(s).values(), reverse=True)
        return tuple(c)

    theo = {                       # 5 digits drawn from 0-9 with replacement
        (1, 1, 1, 1, 1): 0.3024,   # all different
        (2, 1, 1, 1):    0.5040,   # one pair
        (2, 2, 1):       0.1080,   # two pairs
        (3, 1, 1):       0.0720,   # three of a kind
        (3, 2):          0.0090,   # full house
        (4, 1):          0.0045,   # four of a kind
        (5,):            0.0001,   # five of a kind
    }
    obs = Counter(category(g) for g in groups)
    N = len(groups)
    rows, chi2 = [], 0.0
    for cat, prob in theo.items():
        o = obs.get(cat, 0)
        e = N * prob
        term = (o - e) ** 2 / e if e > 0 else 0.0
        chi2 += term
        rows.append((cat, o, e, term))
    df = len(theo) - 1
    return {"N": N, "chi2": chi2, "df": df, "rows": rows,
            "p": chi_square_pvalue(chi2, df)}


# ===========================================================================
# 5. CRITICAL VALUE TABLES
# ===========================================================================

def z_critical(tail_prob):
    """Inverse standard normal CDF: returns z with P(Z > z) = tail_prob.
    z_critical(0.025) = 1.9600, z_critical(0.05) = 1.6449."""
    return _norm_ppf(1.0 - tail_prob)


def _norm_ppf(p):
    """Acklam's rational approximation to the standard normal quantile."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    # one Newton refinement
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def chi2_critical(df, alpha=0.05):
    """Upper-tail chi-square critical value. SciPy if available, else bisection."""
    try:
        from scipy.stats import chi2 as _c
        return float(_c.ppf(1 - alpha, df))
    except ImportError:
        lo, hi = 0.0, 1000.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if _chi2_sf_fallback(mid, df) > alpha:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


# Handy hard-coded values that show up in the slides / exams
CHI2_TABLE_005 = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07, 6: 12.59,
                  7: 14.07, 8: 15.51, 9: 16.92, 10: 18.31, 11: 19.68,
                  12: 21.03, 15: 25.00, 19: 30.14, 20: 31.41, 24: 36.42,
                  29: 42.56, 49: 66.34, 99: 123.23}
Z_TABLE = {0.10: 1.2816, 0.05: 1.6449, 0.025: 1.9600, 0.01: 2.3263, 0.005: 2.5758}


# ===========================================================================
# 6. RANDOM VARIATES (inverse transform etc. -- all driven by U(0,1))
# ===========================================================================

def exp_variate(u, mean):
    """Exponential with the given MEAN (= 1/lambda), by inverse transform.

        F(x) = 1 - e^{-x/mean}  ->  X = -mean * ln(1 - U)
    We use ln(U) instead of ln(1-U): identical in distribution and avoids
    log(0) when U == 0 is impossible but U == 1 is.
    """
    return -mean * math.log(1.0 - u) if u < 1.0 else float('inf')


def exp_variate_rate(u, lam):
    """Exponential with RATE lambda (mean = 1/lambda)."""
    return -math.log(1.0 - u) / lam


def uniform_variate(u, a, b):
    return a + (b - a) * u


def triangular_variate(u, a, c, b):
    """Triangular(a, mode c, b) by inverse transform."""
    fc = (c - a) / (b - a)
    if u < fc:
        return a + math.sqrt(u * (b - a) * (c - a))
    return b - math.sqrt((1 - u) * (b - a) * (b - c))


def weibull_variate(u, shape, scale):
    return scale * (-math.log(1.0 - u)) ** (1.0 / shape)


def discrete_variate(u, values, probs):
    """Inverse-transform sampling from a discrete distribution."""
    cum = 0.0
    for v, p in zip(values, probs):
        cum += p
        if u <= cum:
            return v
    return values[-1]


def geometric_variate(u, p):
    """Number of failures before the first success (support 0,1,2,...)."""
    return int(math.floor(math.log(1.0 - u) / math.log(1.0 - p)))


def bernoulli_variate(u, p):
    return 1 if u < p else 0


def box_muller(u1, u2):
    """Two independent standard normals from two U(0,1) values."""
    r = math.sqrt(-2.0 * math.log(u1))
    return r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)


def normal_variate(u1, u2, mu=0.0, sigma=1.0):
    z, _ = box_muller(u1, u2)
    return mu + sigma * z


def poisson_variate(rng, lam):
    """Knuth's method -- consumes a variable number of uniforms from `rng`
    (a callable returning U(0,1))."""
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng()
        if p <= L:
            return k
        k += 1


# ===========================================================================
# 7. MONTE CARLO
# ===========================================================================

def mc_integrate(g, a, b, n, uniforms):
    """Sample-mean Monte Carlo estimate of  I = int_a^b g(x) dx.

        Ybar(n) = (b - a) * (1/n) * sum g(X_i),  X_i ~ U(a,b)

    `uniforms` is an iterable/list of n U(0,1) values.
    Returns (estimate, std_error, half_width_95).
    """
    vals = []
    it = iter(uniforms)
    for _ in range(n):
        x = a + (b - a) * next(it)
        vals.append(g(x))
    mean = sum(vals) / n
    est = (b - a) * mean
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        se = (b - a) * math.sqrt(var / n)
    else:
        se = float('nan')
    return est, se, 1.96 * se


def mc_pi_quarter_circle(uniforms, n):
    """pi_hat = 4 * (points with x^2 + y^2 <= 1) / n. Consumes 2n uniforms."""
    it = iter(uniforms)
    hits = 0
    for _ in range(n):
        x, y = next(it), next(it)
        if x * x + y * y <= 1.0:
            hits += 1
    return 4.0 * hits / n, hits


def buffon_needle(uniforms, n, L=1.0, D=2.0):
    """Buffon's needle. Consumes 2n uniforms.

        x     ~ U(0, D/2)   distance from needle centre to nearest line
        theta ~ U(0, pi/2)  acute angle with the lines
        cross iff  x <= (L/2) * sin(theta)
        P = 2L / (pi D)     ->   pi_hat = 2L / (D * P_hat)
    """
    it = iter(uniforms)
    crossings = 0
    for _ in range(n):
        x = (D / 2.0) * next(it)
        theta = (math.pi / 2.0) * next(it)
        if x <= (L / 2.0) * math.sin(theta):
            crossings += 1
    p_hat = crossings / n
    pi_hat = (2.0 * L) / (D * p_hat) if p_hat > 0 else float('inf')
    return pi_hat, crossings, p_hat


def batch_means_ci(values, n_batches=10, alpha=0.05):
    """Confidence interval from batch means -- the standard way to put error
    bars on a single long simulation run."""
    n = len(values)
    bs = n // n_batches
    means = [sum(values[i*bs:(i+1)*bs]) / bs for i in range(n_batches)]
    gm = sum(means) / n_batches
    var = sum((m - gm) ** 2 for m in means) / (n_batches - 1)
    se = math.sqrt(var / n_batches)
    try:
        from scipy.stats import t
        tc = float(t.ppf(1 - alpha / 2, n_batches - 1))
    except ImportError:
        tc = 2.262        # t_{0.025, 9}
    return gm, gm - tc * se, gm + tc * se, se


def replication_ci(estimates, alpha=0.05):
    """CI across independent replications (different seeds)."""
    k = len(estimates)
    m = sum(estimates) / k
    var = sum((e - m) ** 2 for e in estimates) / (k - 1)
    se = math.sqrt(var / k)
    try:
        from scipy.stats import t
        tc = float(t.ppf(1 - alpha / 2, k - 1))
    except ImportError:
        tc = 1.96
    return m, m - tc * se, m + tc * se, se


# ===========================================================================
# 8. REPORTING
# ===========================================================================

def table(headers, rows, title=None, floatfmt="{:.6f}"):
    """Minimal ASCII table printer -- keeps report output tidy."""
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)

    srows = [[fmt(c) for c in r] for r in rows]
    widths = [max(len(headers[i]), *(len(r[i]) for r in srows)) if srows
              else len(headers[i]) for i in range(len(headers))]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = []
    if title:
        out.append(title)
    out.append(line)
    out.append("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    out.append(line.replace("-", "="))
    for r in srows:
        out.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
    out.append(line)
    return "\n".join(out)


def banner(text, char="=", width=78):
    return f"\n{char * width}\n{text}\n{char * width}"


def decision(stat, crit, name="statistic", crit_name="critical value",
             reject_msg="REJECT H0", accept_msg="DO NOT REJECT H0"):
    """Uniform phrasing for every hypothesis test in the course."""
    if stat > crit:
        return f"{name} = {stat:.6f} > {crit_name} = {crit:.6f}  ->  {reject_msg}"
    return f"{name} = {stat:.6f} <= {crit_name} = {crit:.6f}  ->  {accept_msg}"


if __name__ == "__main__":
    print(banner("00_toolkit.py self-test"))

    # --- Slide 18: X0=27, a=17, c=43, m=100 -> 2, 77, 52
    g = LCG(m=100, a=17, c=43, seed=27)
    print("Slide 18 LCG:", g.integers(3), "(expect [2, 77, 52])")

    # --- Slide 26: a=16807, c=0, m=2^31-1, X0=123457
    g = LCG(m=2**31 - 1, a=16807, c=0, seed=123457)
    print("Slide 26 LCG:", g.integers(3),
          "(expect [2074941799, 559872160, 1645535613])")

    # --- Middle square from the C1 handout
    print("Middle square 5731:", middle_square(5731, 2), "(expect [8443, 2842])")

    # --- Slide 21: a=13, m=64, c=0 periods
    for s in (1, 2, 3, 4):
        p, tail = find_period(64, 13, 0, s)
        print(f"  period(m=64,a=13,c=0,X0={s}) = {p} (slide: "
              f"{ {1:16, 2:8, 3:16, 4:4}[s] })")

    # --- Slide 36: K-S worked example
    ks = ks_test_uniform([0.44, 0.81, 0.14, 0.05, 0.93])
    print(f"Slide 36 K-S: D+={ks['d_plus']:.2f} D-={ks['d_minus']:.2f} "
          f"D={ks['d']:.2f} (expect 0.26 / 0.21 / 0.26), "
          f"D_crit={ks_critical(5, 0.05)} (expect 0.565)")

    # --- Slide 56: autocorrelation worked example
    R30 = [0.12, 0.01, 0.23, 0.28, 0.89, 0.31, 0.64, 0.28, 0.83, 0.93,
           0.99, 0.15, 0.33, 0.35, 0.91, 0.41, 0.60, 0.27, 0.75, 0.88,
           0.68, 0.49, 0.05, 0.43, 0.95, 0.58, 0.19, 0.36, 0.69, 0.87]
    ac = autocorrelation_test(R30, i=3, lag=5, alpha=0.05)
    print(f"Slide 56 autocorr: M={ac['M']} (expect 4), rho={ac['rho']:.4f} "
          f"(expect -0.1945), sigma={ac['sigma']:.4f} (expect 0.1280), "
          f"Z0={ac['z0']:.2f} (expect -1.52)")

    # --- Slide 39: chi-square worked example
    obs = [8, 8, 10, 9, 12, 8, 10, 14, 10, 11]
    chi2 = sum((o - 10) ** 2 / 10 for o in obs)
    print(f"Slide 39 chi2 = {chi2:.1f} (expect 3.4), "
          f"crit = {chi2_critical(9, 0.05):.2f} (expect 16.9)")

    print("\nz_critical(0.025) =", round(z_critical(0.025), 4), "(expect 1.96)")
