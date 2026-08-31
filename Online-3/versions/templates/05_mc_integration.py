"""
05_mc_integration.py
====================
TEMPLATE: "Use Monte Carlo to evaluate an integral; show convergence."

RNG deck, slides 60-68. The sample-mean estimator, its error bars, the
convergence table, and the reasons the error is not monotone.

  A. Sample-mean estimator      I = int_a^b g(x) dx  ~  (b-a) * mean(g(X_i))
  B. Convergence table + 1/sqrt(n) behaviour
  C. Confidence intervals for the estimate
  D. Hit-or-miss estimator (the other standard method)
  E. Multi-dimensional integrals -- where MC actually wins
  F. Variance reduction: antithetic variates, importance sampling

Change G, A_LIM, B_LIM, EXACT at the top for a different integral.
Run:  python3 05_mc_integration.py
"""

import math

# ---------------------------------------------------------------------------
# THE INTEGRAL -- change these four lines for any problem
# ---------------------------------------------------------------------------
def G(x):
    return math.sin(x)

A_LIM, B_LIM = 0.0, math.pi
EXACT = 2.0                       # int_0^pi sin x dx = [-cos x]_0^pi = 2
LABEL = "int_0^pi sin(x) dx"


# ---------------------------------------------------------------------------
# Generator (no `random` module)
# ---------------------------------------------------------------------------
class LCG:
    def __init__(self, m=2**31 - 1, a=16807, c=0, seed=123457):
        self.m, self.a, self.c, self.seed = m, a, c, seed
        self.x = seed

    def reset(self, seed=None):
        self.x = self.seed if seed is None else seed
        return self

    def u(self):
        self.x = (self.a * self.x + self.c) % self.m
        return self.x / self.m


# ===========================================================================
# A. SAMPLE-MEAN ESTIMATOR
# ===========================================================================

def mc_integrate(g, a, b, n, rng, verbose=False):
    """Monte Carlo integration by the sample-mean method.

    Theory (slide 61): let X ~ U(a,b) and Y = (b-a) g(X). Then
        E[Y] = (b-a) * int_a^b g(x) * 1/(b-a) dx = int_a^b g(x) dx = I
    so the integral IS an expected value, and we estimate it by averaging:

        Ybar(n) = (b - a) * (1/n) * sum_{i=1..n} g(X_i),   X_i ~ U(a,b)

    Geometrically: (base) x (average height) = area of a rectangle that
    approximates the area under the curve.

    Returns (estimate, standard_error, half_width_95).
    """
    vals = []
    for _ in range(n):
        x = a + (b - a) * rng.u()
        vals.append(g(x))
    mean = sum(vals) / n
    est = (b - a) * mean
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        se = (b - a) * math.sqrt(var / n)
    else:
        se = float('nan')
    if verbose:
        print(f"    n = {n}: mean height = {mean:.6f}, "
              f"(b-a) = {b-a:.6f}, estimate = {est:.6f}")
    return est, se, 1.96 * se


# ===========================================================================
# D. HIT-OR-MISS ESTIMATOR
# ===========================================================================

def mc_hit_or_miss(g, a, b, n, rng, gmax=None):
    """Throw darts into the bounding box [a,b] x [0, gmax]; the fraction
    landing under the curve times the box area estimates the integral.

    Always has higher variance than the sample-mean method -- it throws away
    the actual value of g and keeps only a yes/no. Included because exams
    sometimes ask you to compare the two.
    """
    if gmax is None:
        gmax = max(g(a + (b - a) * i / 1000) for i in range(1001)) * 1.05
    hits = 0
    for _ in range(n):
        x = a + (b - a) * rng.u()
        y = gmax * rng.u()
        if y <= g(x):
            hits += 1
    area = (b - a) * gmax
    est = area * hits / n
    p = hits / n
    se = area * math.sqrt(p * (1 - p) / n) if 0 < p < 1 else float('nan')
    return est, se, hits, gmax


# ===========================================================================
# F. VARIANCE REDUCTION
# ===========================================================================

def mc_antithetic(g, a, b, n_pairs, rng):
    """Antithetic variates: for each U, also use 1-U. The two are negatively
    correlated, so their average has lower variance than 2n independent draws
    (when g is monotone)."""
    vals = []
    for _ in range(n_pairs):
        u = rng.u()
        x1 = a + (b - a) * u
        x2 = a + (b - a) * (1 - u)
        vals.append((g(x1) + g(x2)) / 2)
    mean = sum(vals) / n_pairs
    est = (b - a) * mean
    var = sum((v - mean) ** 2 for v in vals) / (n_pairs - 1)
    se = (b - a) * math.sqrt(var / n_pairs)
    return est, se


def mc_importance(g, pdf, sampler, n, rng):
    """Importance sampling: draw X ~ pdf (not uniform) and average g(X)/pdf(X).
    Cuts variance a lot when pdf is shaped like g."""
    vals = []
    for _ in range(n):
        x = sampler(rng)
        vals.append(g(x) / pdf(x))
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, math.sqrt(var / n)


# ===========================================================================
# E. MULTI-DIMENSIONAL
# ===========================================================================

def mc_integrate_nd(g, lows, highs, n, rng):
    """Integral of g over a d-dimensional box. This is where Monte Carlo
    beats grid methods: the error stays O(1/sqrt(n)) regardless of d, while
    a grid rule needs n^d points."""
    d = len(lows)
    vol = 1.0
    for lo, hi in zip(lows, highs):
        vol *= (hi - lo)
    vals = []
    for _ in range(n):
        pt = [lo + (hi - lo) * rng.u() for lo, hi in zip(lows, highs)]
        vals.append(g(pt))
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return vol * mean, vol * math.sqrt(var / n)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 78)
    print(f"MONTE CARLO INTEGRATION:  {LABEL}")
    print("=" * 78)
    print(f"\n  Exact value = {EXACT}")

    # ---- A: the estimator, small n, shown step by step -------------------
    print(f"\n--- A. The sample-mean estimator, step by step (n = 10) ---")
    print(f"\n  Ybar(n) = (b - a) * (1/n) * sum g(X_i),  X_i ~ U({A_LIM:.4f}, "
          f"{B_LIM:.4f})\n")
    rng = LCG().reset()
    xs, gs = [], []
    for _ in range(10):
        x = A_LIM + (B_LIM - A_LIM) * rng.u()
        xs.append(x)
        gs.append(G(x))
    print(f"  {'i':>3}{'X_i':>10}{'g(X_i)':>12}")
    print("  " + "-" * 25)
    for i, (x, gv) in enumerate(zip(xs, gs), 1):
        print(f"  {i:>3}{x:>10.4f}{gv:>12.6f}")
    print("  " + "-" * 25)
    mean = sum(gs) / 10
    print(f"  average height = {mean:.6f}")
    print(f"  x (b - a) = {B_LIM - A_LIM:.6f}")
    print(f"  Ybar(10) = {(B_LIM - A_LIM) * mean:.6f}   (exact {EXACT})")

    # ---- B: convergence table --------------------------------------------
    print(f"\n--- B. Convergence: does more sampling help? ---\n")
    print(f"  {'n':>9} | {'Ybar(n)':>10} | {'Error':>10} | {'% Error':>9} | "
          f"{'SE':>9} | {'95% CI half-width':>18}")
    print(f"  {'-'*9}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*18}")
    for n in [10, 20, 40, 80, 160, 1_000, 10_000, 100_000, 1_000_000]:
        rng = LCG().reset()
        est, se, hw = mc_integrate(G, A_LIM, B_LIM, n, rng)
        err = abs(est - EXACT)
        print(f"  {n:>9,} | {est:>10.6f} | {err:>10.6f} | "
              f"{100*err/EXACT:>8.3f}% | {se:>9.6f} | {hw:>18.6f}")

    print(f"""
  Two things to notice (slides 67-68):

  1. The estimate "orbits" the true value, tightening as n grows. This is
     the law of large numbers.

  2. The error is NOT monotone -- a larger n can give a worse answer than a
     smaller one. That is not a bug. Ybar(n) is itself a RANDOM VARIABLE:
     each run uses different X_i, so the estimate has its own distribution.
     What shrinks reliably is the STANDARD ERROR (the SE column), not the
     realised error of any single run.""")

    # ---- C: confidence interval ------------------------------------------
    print(f"\n--- C. Reporting the answer properly ---\n")
    rng = LCG().reset()
    est, se, hw = mc_integrate(G, A_LIM, B_LIM, 100_000, rng)
    print(f"  With n = 100,000:")
    print(f"    point estimate  = {est:.6f}")
    print(f"    standard error  = {se:.6f}")
    print(f"    95% CI          = [{est-hw:.6f}, {est+hw:.6f}]")
    print(f"    exact value {EXACT} inside the interval: "
          f"{est-hw <= EXACT <= est+hw}")
    print(f"\n  Never report a Monte Carlo result as a bare number. The output")
    print(f"  is an estimate; it needs an error bar.")

    # ---- how many samples for a target accuracy --------------------------
    target = 0.001
    n_needed = math.ceil((1.96 * se * math.sqrt(100_000) / target) ** 2)
    print(f"\n  To get a 95% half-width of {target}, you would need")
    print(f"    n ~ (1.96 * s / {target})^2 = {n_needed:,} samples.")
    print(f"  Cutting the error by 10x costs 100x the samples -- the 1/sqrt(n) law.")

    # ---- D: hit-or-miss comparison ---------------------------------------
    print(f"\n--- D. Sample-mean vs hit-or-miss (same n) ---\n")
    print(f"  {'n':>9} | {'sample-mean':>13} | {'SE':>9} | "
          f"{'hit-or-miss':>13} | {'SE':>9}")
    print(f"  {'-'*9}-+-{'-'*13}-+-{'-'*9}-+-{'-'*13}-+-{'-'*9}")
    for n in [1_000, 10_000, 100_000]:
        rng = LCG().reset()
        e1, s1, _ = mc_integrate(G, A_LIM, B_LIM, n, rng)
        rng = LCG().reset()
        e2, s2, hits, gmax = mc_hit_or_miss(G, A_LIM, B_LIM, n, rng)
        print(f"  {n:>9,} | {e1:>13.6f} | {s1:>9.6f} | "
              f"{e2:>13.6f} | {s2:>9.6f}")
    print(f"\n  The sample-mean method has the smaller standard error at every n.")
    print(f"  Hit-or-miss discards information: it records only whether the")
    print(f"  dart landed under the curve, not how far under.")

    # ---- F: variance reduction -------------------------------------------
    print(f"\n--- F. Variance reduction: antithetic variates ---\n")
    print(f"  For each U we also evaluate at 1-U. If g is MONOTONE the pair is")
    print(f"  negatively correlated, so the pair average fluctuates less than")
    print(f"  two independent draws and the standard error falls.")
    print(f"  Budget held fixed: n plain draws vs n/2 antithetic pairs.\n")

    def compare_antithetic(g, a, b, label, exact):
        print(f"  {label}   (exact = {exact:.6f})")
        print(f"  {'draws':>9} | {'plain MC':>12} | {'SE':>10} | "
              f"{'antithetic':>12} | {'SE':>10} | {'SE ratio':>9}")
        print(f"  {'-'*9}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*9}")
        for n in [1_000, 10_000, 100_000]:
            rng = LCG().reset()
            e1, s1, _ = mc_integrate(g, a, b, n, rng)
            rng = LCG().reset()
            e2, s2 = mc_antithetic(g, a, b, n // 2, rng)
            print(f"  {n:>9,} | {e1:>12.6f} | {s1:>10.6f} | "
                  f"{e2:>12.6f} | {s2:>10.6f} | {s2/s1:>9.3f}")
        print()

    # Case 1: monotone integrand -- antithetic works
    compare_antithetic(lambda x: math.exp(x), 0.0, 1.0,
                       "g(x) = e^x on [0,1], strictly increasing",
                       math.e - 1)
    print("  SE ratio well below 1.0: antithetic sampling wins clearly.\n")

    # Case 2: symmetric integrand -- antithetic is useless
    compare_antithetic(G, A_LIM, B_LIM,
                       f"g(x) = sin(x) on [0,pi], symmetric about pi/2", EXACT)
    print(f"""  SE ratio is about sqrt(2) = 1.414 -- antithetic sampling made it
  WORSE. Why: sin(pi - x) = sin(x), so the antithetic partner returns the
  SAME value as the original draw. The pair is perfectly POSITIVELY
  correlated, the averaging removes no variance, and we have effectively
  thrown away half the sample.

  LESSON: antithetic variates help only when g is monotone (or at least not
  symmetric) over the interval. Always check the shape of the integrand
  before claiming a variance reduction -- and verify it numerically, as the
  two tables above do.""")

    # ---- E: multi-dimensional --------------------------------------------
    print(f"\n--- E. Where Monte Carlo really wins: high dimensions ---\n")

    def g_nd(pt):
        return math.exp(-sum(v * v for v in pt))

    print(f"  Integrate exp(-|x|^2) over the unit cube [0,1]^d.")
    print(f"  Exact = (int_0^1 e^(-x^2) dx)^d, with int_0^1 e^(-x^2) dx "
          f"= 0.7468241328")
    base = 0.7468241328
    print(f"\n  {'d':>3} | {'exact':>12} | {'MC (n=200k)':>13} | {'SE':>10} | "
          f"{'grid points needed':>19}")
    print(f"  {'-'*3}-+-{'-'*12}-+-{'-'*13}-+-{'-'*10}-+-{'-'*19}")
    for d in [1, 2, 3, 5, 10]:
        rng = LCG().reset()
        est, se = mc_integrate_nd(g_nd, [0.0]*d, [1.0]*d, 200_000, rng)
        exact = base ** d
        grid = 10 ** d          # only 10 points per axis
        print(f"  {d:>3} | {exact:>12.8f} | {est:>13.8f} | {se:>10.8f} | "
              f"{grid:>19,}")
    print(f"""
  The Monte Carlo error stays ~1/sqrt(n) no matter what d is, while a grid
  rule needs 10^d points just for 10 divisions per axis. At d = 10 that is
  10 billion evaluations for a crude grid, versus 200,000 for MC. This is
  the "curse of dimensionality" that slide 60 refers to, and the reason
  Monte Carlo exists.""")


if __name__ == "__main__":
    main()
