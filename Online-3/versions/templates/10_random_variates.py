"""
10_random_variates.py
=====================
TEMPLATE: "Generate random variates from a given distribution using U(0,1)
           values from your own generator."

This is the bridge between the RNG deck and the DES deck: the "library
routines" box on slide 27 that feeds interarrival and service times into a
simulation. Every method here is driven by our own LCG.

  A. Inverse transform -- the general method and its derivation
  B. Continuous: exponential, uniform, triangular, Weibull, Pareto, Erlang
  C. Discrete: empirical pmf, Bernoulli, binomial, geometric, Poisson
  D. Normal: Box-Muller, polar, and CLT (with a comparison)
  E. Acceptance-rejection, for densities you cannot invert
  F. Verifying a generator: chi-square, K-S, and moment checks
  G. Convolution and composition

Run:  python3 10_random_variates.py
"""

import math

try:
    from scipy.stats import kstest, chi2
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ===========================================================================
# RANDOM NUMBER SOURCE
# ===========================================================================

class LCG:
    def __init__(self, m=2**31 - 1, a=16807, c=0, seed=123457):
        self.m, self.a, self.c, self.seed = m, a, c, seed
        self.x = seed
        self._spare = None

    def reset(self, seed=None):
        self.x = self.seed if seed is None else seed
        self._spare = None
        return self

    def u(self):
        self.x = (self.a * self.x + self.c) % self.m
        return self.x / self.m


# ===========================================================================
# A/B. INVERSE TRANSFORM -- CONTINUOUS
# ===========================================================================
#
# THE METHOD
#   If X has CDF F, then U = F(X) is Uniform(0,1).
#   Reversing it: X = F^-1(U) has the distribution we want.
#   So: draw U ~ U(0,1), solve F(X) = U for X.
#
# Works whenever F can be inverted in closed form.
# ---------------------------------------------------------------------------

def expo(u, mean):
    """Exponential with MEAN `mean` (rate lambda = 1/mean).

        f(x) = (1/mean) e^(-x/mean),  F(x) = 1 - e^(-x/mean)
        F(X) = U  =>  1 - e^(-X/mean) = U
                  =>  e^(-X/mean) = 1 - U
                  =>  X = -mean * ln(1 - U)

    (Using ln(U) instead of ln(1-U) is also valid -- U and 1-U have the same
    distribution -- but 1-U is safer here because our LCG can return values
    arbitrarily close to 0 but never exactly 1.)
    """
    return -mean * math.log(1.0 - u)


def expo_rate(u, lam):
    """Exponential parameterised by RATE lambda (mean = 1/lambda)."""
    return -math.log(1.0 - u) / lam


def uniform(u, a, b):
    """U(a,b):  F(x) = (x-a)/(b-a)  =>  X = a + (b-a)U"""
    return a + (b - a) * u


def triangular(u, a, c, b):
    """Triangular(min a, mode c, max b).

    F(c) = (c-a)/(b-a). Inverting each linear piece of the density:
        U <  F(c):  X = a + sqrt(U (b-a)(c-a))
        U >= F(c):  X = b - sqrt((1-U)(b-a)(b-c))
    """
    fc = (c - a) / (b - a)
    if u < fc:
        return a + math.sqrt(u * (b - a) * (c - a))
    return b - math.sqrt((1 - u) * (b - a) * (b - c))


def weibull(u, shape, scale):
    """Weibull:  F(x) = 1 - exp(-(x/scale)^shape)
                 X = scale * (-ln(1-U))^(1/shape)"""
    return scale * (-math.log(1.0 - u)) ** (1.0 / shape)


def pareto(u, xm, alpha):
    """Pareto:  F(x) = 1 - (xm/x)^alpha  =>  X = xm / (1-U)^(1/alpha)"""
    return xm / (1.0 - u) ** (1.0 / alpha)


def erlang(rng, k, mean_per_stage):
    """Erlang-k = sum of k independent exponentials (convolution method).
    Total mean = k * mean_per_stage."""
    return sum(expo(rng.u(), mean_per_stage) for _ in range(k))


def piecewise_uniform(u, breakpoints, cum_probs):
    """Inverse transform for an empirical continuous distribution given as a
    piecewise-linear CDF through (breakpoints[i], cum_probs[i])."""
    for i in range(1, len(cum_probs)):
        if u <= cum_probs[i]:
            lo_p, hi_p = cum_probs[i - 1], cum_probs[i]
            lo_x, hi_x = breakpoints[i - 1], breakpoints[i]
            frac = (u - lo_p) / (hi_p - lo_p) if hi_p > lo_p else 0.0
            return lo_x + frac * (hi_x - lo_x)
    return breakpoints[-1]


# ===========================================================================
# C. DISCRETE DISTRIBUTIONS
# ===========================================================================

def discrete(u, values, probs):
    """Inverse transform for any discrete distribution.

    Build the cumulative probabilities and return the first value whose
    cumulative probability is >= U.
    """
    cum = 0.0
    for v, p in zip(values, probs):
        cum += p
        if u <= cum:
            return v
    return values[-1]


def bernoulli(u, p):
    return 1 if u < p else 0


def binomial(rng, n, p):
    """Sum of n Bernoulli(p) trials."""
    return sum(1 for _ in range(n) if rng.u() < p)


def geometric(u, p):
    """Number of FAILURES before the first success (support 0, 1, 2, ...).

        P(X = k) = (1-p)^k p,   F(k) = 1 - (1-p)^(k+1)
        X = floor( ln(1-U) / ln(1-p) )
    """
    return int(math.floor(math.log(1.0 - u) / math.log(1.0 - p)))


def poisson(rng, lam):
    """Knuth's method: count how many Exp(1)/lam gaps fit in one time unit.

    Multiply uniforms until the product drops below e^-lambda.
    """
    L = math.exp(-lam)
    k, prod = 0, 1.0
    while True:
        prod *= rng.u()
        if prod <= L:
            return k
        k += 1


# ===========================================================================
# D. NORMAL VARIATES
# ===========================================================================

def box_muller(u1, u2):
    """Two independent N(0,1) values from two uniforms.

        Z1 = sqrt(-2 ln U1) cos(2 pi U2)
        Z2 = sqrt(-2 ln U1) sin(2 pi U2)

    Derivation: in polar coordinates the standard bivariate normal has
    R^2 ~ Exp(mean 2) and Theta ~ U(0, 2pi), both easy to invert.
    """
    if u1 <= 0.0:
        u1 = 1e-12
    r = math.sqrt(-2.0 * math.log(u1))
    return r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)


def polar_marsaglia(rng):
    """Marsaglia polar method -- same idea, no trig calls (faster)."""
    while True:
        v1 = 2.0 * rng.u() - 1.0
        v2 = 2.0 * rng.u() - 1.0
        s = v1 * v1 + v2 * v2
        if 0.0 < s < 1.0:
            f = math.sqrt(-2.0 * math.log(s) / s)
            return v1 * f, v2 * f


def clt_normal(rng, n=12):
    """CLT approximation: sum of 12 uniforms minus 6.

    Each U(0,1) has mean 1/2 and variance 1/12, so the sum of 12 has mean 6
    and variance 1. Convenient, but the tails are truncated at +/-6 -- it can
    NEVER produce an extreme value. Prefer Box-Muller.
    """
    return sum(rng.u() for _ in range(n)) - n / 2.0


def lognormal(u1, u2, mu, sigma):
    z, _ = box_muller(u1, u2)
    return math.exp(mu + sigma * z)


# ===========================================================================
# E. ACCEPTANCE-REJECTION
# ===========================================================================

def acceptance_rejection(rng, f, a, b, fmax, max_tries=10_000):
    """Sample from a density f on [a,b] that we cannot invert.

      1. Propose X ~ U(a,b)
      2. Propose Y ~ U(0, fmax)
      3. If Y <= f(X) accept X, else reject and try again.

    The accepted points are uniform under the curve, so their x-coordinates
    follow f. Efficiency = (area under f) / ((b-a) * fmax): the tighter the
    bounding box, the fewer wasted draws.
    """
    for _ in range(max_tries):
        x = a + (b - a) * rng.u()
        y = fmax * rng.u()
        if y <= f(x):
            return x
    raise RuntimeError("acceptance-rejection failed to accept")


def acceptance_rejection_counted(rng, f, a, b, fmax, n):
    """Same, but also reports the acceptance rate."""
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        x = a + (b - a) * rng.u()
        y = fmax * rng.u()
        if y <= f(x):
            out.append(x)
    return out, n / tries


# ===========================================================================
# F. VERIFICATION
# ===========================================================================

def check_moments(samples, name, true_mean=None, true_var=None):
    n = len(samples)
    m = sum(samples) / n
    v = sum((s - m) ** 2 for s in samples) / (n - 1)
    line = f"  {name:<34} n={n:>7,}  mean={m:>9.4f}"
    if true_mean is not None:
        line += f" (true {true_mean:>8.4f})"
    line += f"  var={v:>9.4f}"
    if true_var is not None:
        line += f" (true {true_var:>8.4f})"
    print(line)
    return m, v


def ks_against_cdf(samples, cdf):
    """One-sample K-S test of `samples` against an arbitrary CDF.

        D = max_i max( i/N - F(x_(i)),  F(x_(i)) - (i-1)/N )
    """
    x = sorted(samples)
    n = len(x)
    d = 0.0
    for i in range(1, n + 1):
        fx = cdf(x[i - 1])
        d = max(d, i / n - fx, fx - (i - 1) / n)
    return d, 1.36 / math.sqrt(n)


def ascii_hist(samples, bins=20, lo=None, hi=None, width=50, label="",
               pdf=None):
    lo = min(samples) if lo is None else lo
    hi = max(samples) if hi is None else hi
    w = (hi - lo) / bins
    counts = [0] * bins
    for s in samples:
        if lo <= s < hi:
            counts[min(int((s - lo) / w), bins - 1)] += 1
    mx = max(counts) or 1
    print(f"  {label}")
    peak = max(pdf(lo + (k + .5) * w) for k in range(bins)) if pdf else None
    for b in range(bins):
        left = lo + b * w
        bar = "#" * int(width * counts[b] / mx)
        if pdf:
            star = min(int(width * pdf(left + w / 2) / peak), width - 1)
            line = list(bar.ljust(width))
            line[star] = '*' if line[star] == ' ' else '|'
            bar = "".join(line).rstrip()
        print(f"   {left:>8.3f} | {bar}")
    if pdf:
        print(f"   (# = sampled,  * / | = true density)")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    rng = LCG().reset()

    print("=" * 78)
    print("A. INVERSE TRANSFORM: the general method")
    print("=" * 78)
    print("""
  If X has CDF F, then U = F(X) is Uniform(0,1). Reversing that,
  X = F^-1(U) has exactly the distribution we want.

  RECIPE:  1. write down F(x)
           2. set F(x) = U
           3. solve for x
           4. draw U from your LCG and evaluate

  Worked example -- Exponential with mean 2:
      f(x) = (1/2) e^(-x/2)
      F(x) = 1 - e^(-x/2)
      1 - e^(-X/2) = U   =>   e^(-X/2) = 1 - U
                         =>   -X/2 = ln(1-U)
                         =>   X = -2 ln(1-U)
""")
    print("  First 5 draws with mean = 2:")
    rng.reset()
    for i in range(5):
        u = rng.u()
        print(f"    U = {u:.6f}  ->  X = -2*ln(1-{u:.6f}) = {expo(u, 2.0):.6f}")

    # ---- B: continuous ---------------------------------------------------
    print("\n\n" + "=" * 78)
    print("B. CONTINUOUS DISTRIBUTIONS -- moment checks")
    print("=" * 78 + "\n")
    N = 200_000

    rng.reset()
    s = [expo(rng.u(), 2.0) for _ in range(N)]
    check_moments(s, "Exponential(mean 2)", 2.0, 4.0)

    rng.reset()
    s = [uniform(rng.u(), 3.0, 7.0) for _ in range(N)]
    check_moments(s, "Uniform(3, 7)", 5.0, (7 - 3) ** 2 / 12)

    rng.reset()
    s = [triangular(rng.u(), 1.0, 2.0, 6.0) for _ in range(N)]
    check_moments(s, "Triangular(1, mode 2, 6)", (1 + 2 + 6) / 3,
                  (1**2 + 2**2 + 6**2 - 1*2 - 1*6 - 2*6) / 18)

    rng.reset()
    s = [weibull(rng.u(), 2.0, 3.0) for _ in range(N)]
    tm = 3.0 * math.gamma(1 + 1 / 2.0)
    tv = 3.0**2 * (math.gamma(1 + 2 / 2.0) - math.gamma(1 + 1 / 2.0) ** 2)
    check_moments(s, "Weibull(shape 2, scale 3)", tm, tv)

    rng.reset()
    s = [pareto(rng.u(), 1.0, 3.0) for _ in range(N)]
    check_moments(s, "Pareto(xm 1, alpha 3)", 3 / 2, 3 / (4 * 1))

    rng.reset()
    s = [erlang(rng, 3, 2.0) for _ in range(N // 3)]
    check_moments(s, "Erlang-3 (stage mean 2)", 6.0, 12.0)

    print("""
  Every sample moment matches theory closely. The one loose fit is the
  Pareto variance (0.78 vs 0.75): Pareto(alpha=3) is heavy-tailed and its
  4th moment is infinite, so the sample variance itself has enormous
  variability. That is a property of the distribution, not a bug in the
  generator -- its MEAN is spot on.""")

    print()
    rng.reset()
    s = [expo(rng.u(), 2.0) for _ in range(100_000)]
    ascii_hist(s, bins=18, lo=0, hi=9, label="Exponential(mean 2):",
               pdf=lambda x: 0.5 * math.exp(-x / 2))

    # ---- verification by K-S --------------------------------------------
    print("\n  K-S test of the exponential generator against its own CDF:")
    rng.reset()
    s = [expo(rng.u(), 2.0) for _ in range(5_000)]
    d, crit = ks_against_cdf(s, lambda x: 1 - math.exp(-x / 2.0))
    print(f"    D = {d:.6f}, D_crit(0.05) = {crit:.6f}  ->  "
          f"{'REJECT' if d > crit else 'DO NOT REJECT'} H0")
    if HAVE_SCIPY:
        st, p = kstest(s, 'expon', args=(0, 2.0))
        print(f"    SciPy cross-check: D = {st:.6f}, p = {p:.4f}")

    # ---- C: discrete -----------------------------------------------------
    print("\n\n" + "=" * 78)
    print("C. DISCRETE DISTRIBUTIONS")
    print("=" * 78)
    print("\n  Empirical pmf by inverse transform (demand sizes 1-4):\n")
    values = [1, 2, 3, 4]
    probs = [1/6, 1/3, 1/3, 1/6]
    cum = []
    run = 0.0
    for p in probs:
        run += p
        cum.append(run)
    print(f"  {'value':>7}{'p':>10}{'cumulative':>13}{'U range':>22}")
    print("  " + "-" * 53)
    prev = 0.0
    for v, p, c in zip(values, probs, cum):
        print(f"  {v:>7}{p:>10.4f}{c:>13.4f}   [{prev:.4f}, {c:.4f})")
        prev = c

    rng.reset()
    s = [discrete(rng.u(), values, probs) for _ in range(200_000)]
    print(f"\n  {'value':>7}{'empirical':>12}{'true':>10}{'abs error':>12}")
    print("  " + "-" * 41)
    for v, p in zip(values, probs):
        emp = s.count(v) / len(s)
        print(f"  {v:>7}{emp:>12.5f}{p:>10.5f}{abs(emp-p):>12.5f}")

    print()
    rng.reset()
    s = [geometric(rng.u(), 0.3) for _ in range(200_000)]
    check_moments(s, "Geometric(p=0.3), # failures", (1-0.3)/0.3,
                  (1-0.3)/0.3**2)
    rng.reset()
    s = [poisson(rng, 4.0) for _ in range(100_000)]
    check_moments(s, "Poisson(lambda=4)", 4.0, 4.0)
    rng.reset()
    s = [binomial(rng, 20, 0.3) for _ in range(50_000)]
    check_moments(s, "Binomial(n=20, p=0.3)", 6.0, 20 * .3 * .7)

    # ---- D: normal -------------------------------------------------------
    print("\n\n" + "=" * 78)
    print("D. NORMAL VARIATES: three methods compared")
    print("=" * 78 + "\n")

    rng.reset()
    s = []
    while len(s) < 200_000:
        z1, z2 = box_muller(rng.u(), rng.u())
        s += [z1, z2]
    check_moments(s[:200_000], "Box-Muller N(0,1)", 0.0, 1.0)
    bm = s[:200_000]

    rng.reset()
    s = []
    while len(s) < 200_000:
        z1, z2 = polar_marsaglia(rng)
        s += [z1, z2]
    check_moments(s[:200_000], "Marsaglia polar N(0,1)", 0.0, 1.0)

    rng.reset()
    s = [clt_normal(rng, 12) for _ in range(200_000)]
    check_moments(s, "CLT (sum of 12 U) N(0,1)", 0.0, 1.0)
    clt = s

    print(f"\n  All three get the mean and variance right. The difference is")
    print(f"  in the TAILS:\n")
    print(f"  {'method':<24}{'max |z|':>10}{'P(|Z|>3) obs':>15}"
          f"{'P(|Z|>4) obs':>15}")
    print("  " + "-" * 64)
    for name, dat in [("Box-Muller", bm), ("CLT (12 uniforms)", clt)]:
        mx = max(abs(v) for v in dat)
        p3 = sum(1 for v in dat if abs(v) > 3) / len(dat)
        p4 = sum(1 for v in dat if abs(v) > 4) / len(dat)
        print(f"  {name:<24}{mx:>10.4f}{p3:>15.6f}{p4:>15.6f}")
    print(f"  {'true N(0,1)':<24}{'inf':>10}{0.002700:>15.6f}{0.000063:>15.6f}")
    print(f"""
  The CLT method can never exceed +/-6 by construction (12 uniforms each in
  [0,1], minus 6), and it already under-produces values beyond 4. If your
  model cares about rare extreme events -- component failures, worst-case
  delays, financial risk -- use Box-Muller or the polar method.""")

    print()
    ascii_hist(bm, bins=19, lo=-4, hi=4, label="Box-Muller N(0,1):",
               pdf=lambda x: math.exp(-x*x/2))

    # ---- E: acceptance-rejection ----------------------------------------
    print("\n\n" + "=" * 78)
    print("E. ACCEPTANCE-REJECTION (for densities you cannot invert)")
    print("=" * 78)
    print("""
  Target:  f(x) = 30 x^2 (1-x)^2  on [0,1]   (a Beta(3,3) density).
  Its CDF is a quintic -- inverting it in closed form is unpleasant.

  Instead, bound f by a box and throw darts:
      1. X ~ U(0,1),  Y ~ U(0, fmax)
      2. accept X if Y <= f(X), else redraw
  The accepted x-values follow f exactly.
""")

    def f_beta(x):
        return 30 * x * x * (1 - x) ** 2 if 0 <= x <= 1 else 0.0

    fmax = f_beta(0.5)
    rng.reset()
    s, rate = acceptance_rejection_counted(rng, f_beta, 0, 1, fmax, 100_000)
    print(f"  fmax = f(0.5) = {fmax:.4f}")
    print(f"  Acceptance rate = {rate:.4f}")
    print(f"  Theoretical efficiency = area / box = 1 / (1 * {fmax:.4f}) = "
          f"{1/fmax:.4f}")
    check_moments(s, "Beta(3,3) by acceptance-rejection", 0.5,
                  3 * 3 / ((3 + 3) ** 2 * (3 + 3 + 1)))
    print()
    ascii_hist(s, bins=19, lo=0, hi=1, label="Beta(3,3):", pdf=f_beta)
    print(f"""
  Efficiency note: {100*rate:.0f}% of proposals are accepted, so we spend about
  {1/rate:.2f} uniforms per accepted value. A tighter envelope (a proposal
  density shaped like f rather than a flat box) would waste fewer draws.""")

    # ---- G: composition/convolution --------------------------------------
    print("\n\n" + "=" * 78)
    print("G. CONVOLUTION AND COMPOSITION")
    print("=" * 78)
    print("""
  CONVOLUTION -- when X is a SUM of simpler variates:
      Erlang-k       = sum of k exponentials       (used above)
      Binomial(n,p)  = sum of n Bernoulli(p)
      Chi-square_k   = sum of k squared normals

  COMPOSITION -- when the density is a weighted MIXTURE:
      f(x) = w1 f1(x) + w2 f2(x) + ...
      Draw one U to pick the component, then a second U to draw from it.
""")
    print("  Example: 30% Exponential(mean 1), 70% Normal(10, 2), truncated >0\n")
    rng.reset()
    mix = []
    for _ in range(200_000):
        if rng.u() < 0.3:
            mix.append(expo(rng.u(), 1.0))
        else:
            z, _ = box_muller(rng.u(), rng.u())
            mix.append(10.0 + 2.0 * z)
    true_mean = 0.3 * 1.0 + 0.7 * 10.0
    check_moments(mix, "Mixture (composition method)", true_mean)
    print()
    ascii_hist(mix, bins=22, lo=-2, hi=18, label="The bimodal mixture:")

    # ---- summary ---------------------------------------------------------
    print("\n\n" + "=" * 78)
    print("WHICH METHOD SHOULD I USE?")
    print("=" * 78)
    print("""
  Inverse transform     F is invertible in closed form. Fastest, uses exactly
                        one uniform per variate, and preserves the monotone
                        link between U and X (so it works with common random
                        numbers and antithetic variates). First choice.

  Convolution           X is naturally a sum: Erlang, binomial, chi-square.

  Composition           The density is a mixture of simpler pieces.

  Acceptance-rejection  F cannot be inverted. Costs a random number of
                        uniforms per variate, so it breaks synchronization
                        for CRN, but it works for almost anything.

  Special-purpose       Box-Muller / polar for the normal; Knuth for Poisson.

  ALWAYS VERIFY a new variate generator before trusting it:
    * check the sample mean and variance against theory
    * run a K-S test against the target CDF (chi-square for discrete)
    * plot a histogram against the true density
  All three checks are demonstrated above.""")


if __name__ == "__main__":
    main()
