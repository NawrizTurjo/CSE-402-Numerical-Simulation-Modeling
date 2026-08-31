"""
07_metropolis_hastings.py
=========================
TEMPLATE: "Implement the Metropolis-Hastings algorithm to sample from a
           distribution you can only evaluate up to a normalising constant."

Named explicitly in the syllabus. Everything is driven by our own LCG --
no `random` module.

  A. The algorithm, with the acceptance ratio spelled out
  B. Sampling a standard Normal with a random-walk proposal
  C. Diagnostics: acceptance rate, burn-in, trace, thinning, autocorrelation
  D. Tuning the proposal width (the classic "what went wrong" question)
  E. A target with no closed-form normaliser (the point of MH)
  F. A bimodal target -- where a poorly tuned chain gets stuck
  G. Discrete Metropolis-Hastings on a finite state space
  H. Bayesian posterior inference (the standard application)

Run:  python3 07_metropolis_hastings.py
"""

import math

# ===========================================================================
# 0. RANDOM NUMBER SOURCE (no `random` module)
# ===========================================================================

class LCG:
    """X_{n+1} = (a X_n + c) mod m. Long period: a=16807, m=2^31-1."""

    def __init__(self, m=2**31 - 1, a=16807, c=0, seed=123457):
        self.m, self.a, self.c, self.seed = m, a, c, seed
        self.x = seed
        self._spare = None

    def reset(self, seed=None):
        self.x = self.seed if seed is None else seed
        self._spare = None
        return self

    def u(self):
        """U(0,1), strictly inside (0,1)."""
        self.x = (self.a * self.x + self.c) % self.m
        return self.x / self.m

    def normal(self, mu=0.0, sigma=1.0):
        """Standard normal by Box-Muller; caches the second value."""
        if self._spare is not None:
            z, self._spare = self._spare, None
            return mu + sigma * z
        u1 = self.u()
        while u1 <= 0.0:
            u1 = self.u()
        u2 = self.u()
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2 * math.pi * u2)
        self._spare = r * math.sin(2 * math.pi * u2)
        return mu + sigma * z0


# ===========================================================================
# A. THE METROPOLIS-HASTINGS ALGORITHM
# ===========================================================================

def metropolis_hastings(target, x0, n_samples, proposal_sd, rng,
                        log_target=False, burn_in=0, thin=1):
    """Random-walk Metropolis sampler.

    Goal: draw from a density p(x) that we can only evaluate up to a constant,
          p(x) = f(x) / Z with Z unknown.

    Algorithm, for each step from the current state x:
      1. PROPOSE     x' = x + N(0, proposal_sd^2)
      2. RATIO       alpha = min(1,  [f(x') q(x|x')] / [f(x) q(x'|x)] )
                     The random-walk proposal is SYMMETRIC, q(x|x') = q(x'|x),
                     so the q's cancel and  alpha = min(1, f(x')/f(x)).
                     Z cancels too -- that is the whole point of the method.
      3. ACCEPT/REJECT  draw u ~ U(0,1);
                        if u < alpha  -> move to x'
                        else          -> stay at x (and RECORD x AGAIN)
      4. Record the current state.

    Recording the old state on a rejection is essential -- dropping rejected
    steps breaks the chain's stationary distribution. This is the single most
    common implementation bug.

    Returns (samples, acceptance_rate, full_chain).
    """
    chain = []
    x = x0
    fx = target(x)
    accepted = 0
    total = n_samples * thin + burn_in

    for i in range(total):
        # 1. propose
        x_new = x + rng.normal(0.0, proposal_sd)
        fx_new = target(x_new)

        # 2. acceptance ratio
        if log_target:
            log_alpha = fx_new - fx
            alpha = 1.0 if log_alpha >= 0 else math.exp(log_alpha)
        else:
            alpha = 1.0 if fx <= 0 else min(1.0, fx_new / fx)

        # 3. accept or reject
        if rng.u() < alpha:
            x, fx = x_new, fx_new
            accepted += 1
        # 4. record either way
        chain.append(x)

    samples = chain[burn_in::thin]
    return samples, accepted / total, chain


def metropolis_hastings_discrete(states, target_weights, x0_index,
                                 n_samples, rng, burn_in=0):
    """MH on a finite state space with a symmetric nearest-neighbour proposal.

    Proposal: move one step left or right with probability 1/2 each,
    reflecting at the boundaries (so the proposal stays symmetric).
    """
    n = len(states)
    i = x0_index
    chain, accepted = [], 0
    for step in range(n_samples + burn_in):
        j = i + (1 if rng.u() < 0.5 else -1)
        if j < 0 or j >= n:
            j = i                       # reflect: propose staying put
        w_i, w_j = target_weights[i], target_weights[j]
        alpha = 1.0 if w_i <= 0 else min(1.0, w_j / w_i)
        if rng.u() < alpha:
            if j != i:
                accepted += 1
            i = j
        chain.append(i)
    return [states[k] for k in chain[burn_in:]], accepted / (n_samples + burn_in)


# ===========================================================================
# C. DIAGNOSTICS
# ===========================================================================

def summarize(samples, name="", true_mean=None, true_var=None):
    n = len(samples)
    mean = sum(samples) / n
    var = sum((s - mean) ** 2 for s in samples) / (n - 1)
    srt = sorted(samples)
    med = srt[n // 2]
    print(f"  {name}")
    print(f"    n = {n:,}")
    print(f"    mean     = {mean:>10.5f}" +
          (f"   (true {true_mean:.5f}, error {abs(mean-true_mean):.5f})"
           if true_mean is not None else ""))
    print(f"    variance = {var:>10.5f}" +
          (f"   (true {true_var:.5f}, error {abs(var-true_var):.5f})"
           if true_var is not None else ""))
    print(f"    median   = {med:>10.5f}")
    print(f"    range    = [{srt[0]:.4f}, {srt[-1]:.4f}]")
    return mean, var


def lag1_autocorr(samples):
    """Lag-1 autocorrelation of the chain. High values => slow mixing."""
    n = len(samples)
    m = sum(samples) / n
    num = sum((samples[i] - m) * (samples[i+1] - m) for i in range(n-1))
    den = sum((s - m) ** 2 for s in samples)
    return num / den if den > 0 else float('nan')


def effective_sample_size(samples, max_lag=200):
    """ESS = n / (1 + 2*sum_k rho_k), truncated at the first negative rho.

    Tells you how many INDEPENDENT draws the correlated chain is worth.
    """
    n = len(samples)
    m = sum(samples) / n
    den = sum((s - m) ** 2 for s in samples)
    if den == 0:
        return float('nan')
    total = 0.0
    for k in range(1, min(max_lag, n - 1)):
        num = sum((samples[i] - m) * (samples[i+k] - m) for i in range(n-k))
        rho = num / den
        if rho < 0:
            break
        total += rho
    return n / (1 + 2 * total)


def ascii_histogram(samples, bins=25, lo=None, hi=None, width=54,
                    density_fn=None, label=""):
    """Text histogram, optionally overlaid with the true density (shown as *)."""
    lo = min(samples) if lo is None else lo
    hi = max(samples) if hi is None else hi
    w = (hi - lo) / bins
    counts = [0] * bins
    for s in samples:
        if lo <= s < hi:
            counts[min(int((s - lo) / w), bins - 1)] += 1
    mx = max(counts) or 1
    print(f"  {label}")
    for b in range(bins):
        left = lo + b * w
        bar = "#" * int(width * counts[b] / mx)
        if density_fn is not None:
            # scale the true density to the same peak
            peak = max(density_fn(lo + (k + 0.5) * w) for k in range(bins))
            star = int(width * density_fn(left + w/2) / peak)
            line = list(bar.ljust(width))
            star = min(star, width - 1)
            if 0 <= star:
                line[star] = '*' if line[star] == ' ' else '|'
            bar = "".join(line).rstrip()
        print(f"   {left:>7.3f} | {bar}")
    if density_fn is not None:
        print(f"   (# = sampled histogram,  * / | = true density)")


def ascii_trace(chain, n=120, height=15, label=""):
    """Compact trace plot -- shows burn-in and whether the chain is stuck."""
    seg = chain[:n]
    lo, hi = min(seg), max(seg)
    rng_ = (hi - lo) or 1.0
    grid = [[' '] * len(seg) for _ in range(height)]
    for i, v in enumerate(seg):
        row = height - 1 - int((v - lo) / rng_ * (height - 1))
        grid[row][i] = '*'
    print(f"  {label}  (first {len(seg)} states, y from {lo:.2f} to {hi:.2f})")
    for r, row in enumerate(grid):
        print("   |" + "".join(row))
    print("   +" + "-" * len(seg))


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 78)
    print("METROPOLIS-HASTINGS ALGORITHM")
    print("=" * 78)

    # ---- A/B: sample a standard Normal ----------------------------------
    print("\n--- A/B. Sampling N(0,1) with a random-walk proposal ---\n")
    print("  Target (unnormalised):  f(x) = exp(-x^2 / 2)")
    print("  NOTE we never supply the 1/sqrt(2 pi) constant -- MH does not")
    print("  need it, because it cancels in the ratio f(x')/f(x).")
    print("  Proposal: x' = x + N(0, 1^2), which is symmetric.\n")

    def f_normal(x):
        return math.exp(-x * x / 2.0)

    rng = LCG().reset()
    samples, acc, chain = metropolis_hastings(
        f_normal, x0=0.0, n_samples=50_000, proposal_sd=2.4, rng=rng,
        burn_in=1_000)

    print(f"  Acceptance rate = {acc:.3f}")
    summarize(samples, "N(0,1) via Metropolis-Hastings:",
              true_mean=0.0, true_var=1.0)

    print()
    ascii_histogram(samples, bins=21, lo=-4, hi=4,
                    density_fn=lambda x: math.exp(-x*x/2),
                    label="Sampled distribution vs true N(0,1) density:")

    # quantile check
    print("\n  Quantile check against the true normal:")
    srt = sorted(samples)
    for q, true_q in [(0.025, -1.960), (0.25, -0.674), (0.50, 0.0),
                      (0.75, 0.674), (0.975, 1.960)]:
        emp = srt[int(q * len(srt))]
        print(f"    {q*100:>5.1f}%: empirical {emp:>7.3f}   "
              f"true {true_q:>7.3f}   diff {abs(emp-true_q):.3f}")

    # ---- C: diagnostics --------------------------------------------------
    print("\n\n--- C. Diagnostics ---\n")
    print(f"  Lag-1 autocorrelation = {lag1_autocorr(samples):.4f}")
    ess = effective_sample_size(samples[:10_000])
    print(f"  Effective sample size (from 10,000 draws) = {ess:,.0f}")
    print(f"    -> the correlated chain is worth about {ess/10000*100:.1f}% "
          f"of that many independent draws.")
    print(f"\n  Successive MH samples are NOT independent: each is a small step")
    print(f"  from the previous one, and rejected steps repeat the state. Fix")
    print(f"  by THINNING (keep every k-th) and discarding a BURN-IN.")

    thin_samples, _, _ = metropolis_hastings(
        f_normal, 0.0, 5_000, 2.4, LCG().reset(), burn_in=1_000, thin=10)
    print(f"\n  After thinning by 10: lag-1 autocorr = "
          f"{lag1_autocorr(thin_samples):.4f} (was "
          f"{lag1_autocorr(samples):.4f})")

    print()
    bad_start, _, bad_chain = metropolis_hastings(
        f_normal, x0=25.0, n_samples=500, proposal_sd=1.0, rng=LCG().reset())
    ascii_trace(bad_chain, n=110, height=12,
                label="Trace from a deliberately bad start x0 = 25:")
    print(f"\n  The chain needs ~{next(i for i,v in enumerate(bad_chain) if abs(v)<3)} "
          f"steps to reach the bulk of the target. Those early states are")
    print(f"  NOT samples from the target -- that is what burn-in discards.")

    # ---- D: tuning the proposal width ------------------------------------
    print("\n\n--- D. Tuning the proposal width (the classic exam question) ---\n")
    print("  Too small -> almost everything is accepted, but the chain crawls")
    print("               and explores the target very slowly.")
    print("  Too large -> almost everything is rejected, the chain gets stuck")
    print("               repeating the same state.")
    print("  Rule of thumb: aim for an acceptance rate near 0.234 in high")
    print("  dimensions, or roughly 0.4-0.5 in one dimension.\n")
    print(f"  {'proposal sd':>12} | {'accept rate':>12} | {'mean':>9} | "
          f"{'variance':>9} | {'lag-1 rho':>10} | {'ESS':>8}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}")
    for sd in [0.05, 0.5, 2.4, 10.0, 50.0]:
        s, a, _ = metropolis_hastings(f_normal, 0.0, 20_000, sd,
                                      LCG().reset(), burn_in=2_000)
        m = sum(s) / len(s)
        v = sum((t - m) ** 2 for t in s) / (len(s) - 1)
        print(f"  {sd:>12.2f} | {a:>12.3f} | {m:>9.4f} | {v:>9.4f} | "
              f"{lag1_autocorr(s):>10.4f} | {effective_sample_size(s[:5000]):>8.0f}")
    print(f"""
  True values: mean 0.0000, variance 1.0000.

  Read the ESS column, not the mean/variance column. All five chains get
  roughly the right variance -- given enough steps even a badly tuned chain
  converges. What collapses is EFFICIENCY:

    sd = 0.05  accepts 98% of proposals, but every step is tiny, so the
               chain barely moves: lag-1 rho = 0.999, ESS = 14 out of 5,000.
    sd = 50    rejects 97% of proposals, so the chain sits still for long
               stretches: rho = 0.970, ESS = 109.
    sd = 2.4   is the sweet spot: acceptance ~0.44 and ESS ~1,200, roughly
               90x more information per step than either extreme.

  So "it gave the right answer" is not evidence of good tuning. Report the
  acceptance rate and the effective sample size.""")

    # ---- E: a target with no closed-form normaliser ----------------------
    print("\n\n--- E. A target we cannot normalise analytically ---\n")
    print("  f(x) = exp(-x^4 + 2x^2)   on the real line.")
    print("  The normalising constant has no elementary closed form, so we")
    print("  cannot use inverse-transform sampling. MH does not care.\n")

    def f_quartic(x):
        return math.exp(-x**4 + 2 * x**2) if abs(x) < 10 else 0.0

    s, a, _ = metropolis_hastings(f_quartic, 0.0, 60_000, 1.5,
                                  LCG().reset(), burn_in=2_000)
    print(f"  Acceptance rate = {a:.3f}")
    summarize(s, "Samples from the quartic target:")

    # Cross-check the MH mean against deterministic numerical integration
    def num_integrate(fn, lo, hi, steps=200_000):
        h = (hi - lo) / steps
        tot = 0.5 * (fn(lo) + fn(hi))
        for k in range(1, steps):
            tot += fn(lo + k * h)
        return tot * h

    Z = num_integrate(f_quartic, -5, 5)
    m_true = num_integrate(lambda x: x * f_quartic(x), -5, 5) / Z
    v_true = num_integrate(lambda x: x*x * f_quartic(x), -5, 5) / Z - m_true**2
    print(f"\n  Cross-check by deterministic quadrature:")
    print(f"    normalising constant Z = {Z:.6f}")
    print(f"    true mean     = {m_true:.5f}  (MH gave {sum(s)/len(s):.5f})")
    print(f"    true variance = {v_true:.5f}")
    print(f"  MH reproduces a distribution it was never told how to normalise.")

    print()
    ascii_histogram(s, bins=21, lo=-2.5, hi=2.5,
                    density_fn=f_quartic,
                    label="Sampled vs true (unnormalised) density -- note it is bimodal:")

    # ---- F: bimodal trap -------------------------------------------------
    print("\n\n--- F. When MH fails: well-separated modes ---\n")
    print("  Target: a 50/50 mixture of N(-8, 1) and N(+8, 1).")
    print("  The two modes are 16 sd apart with almost zero density between.\n")

    def f_bimodal(x):
        return math.exp(-(x + 8) ** 2 / 2) + math.exp(-(x - 8) ** 2 / 2)

    for sd in [1.0, 15.0]:
        s, a, ch = metropolis_hastings(f_bimodal, -8.0, 30_000, sd,
                                       LCG().reset(), burn_in=1_000)
        left = sum(1 for v in s if v < 0) / len(s)
        print(f"  proposal sd = {sd:>5.1f}: acceptance {a:.3f}, "
              f"fraction in the LEFT mode = {left:.3f} (true 0.500), "
              f"mean = {sum(s)/len(s):>7.3f}")
        if left > 0.95 or left < 0.05:
            print(f"      -> the chain NEVER crossed the valley. It sampled one")
            print(f"         mode perfectly and reported a confidently wrong answer.")
        else:
            print(f"      -> big jumps let the chain cross between modes.")
    print(f"""
  This is the standard cautionary result: a chain can look perfectly healthy
  (smooth trace, sensible acceptance rate) and still be exploring only part
  of the target. Diagnose it by running several chains from different
  starting points and checking they agree.""")

    # ---- G: discrete MH --------------------------------------------------
    print("\n\n--- G. Discrete Metropolis-Hastings ---\n")
    print("  Target on states 1..6 with weights proportional to i^2:")
    states = [1, 2, 3, 4, 5, 6]
    weights = [i * i for i in states]
    Zd = sum(weights)
    print(f"    weights = {weights}, so the true pmf is "
          f"{[round(w/Zd, 4) for w in weights]}\n")

    s, a = metropolis_hastings_discrete(states, weights, 0, 200_000,
                                        LCG().reset(), burn_in=5_000)
    print(f"  Acceptance rate = {a:.3f}\n")
    print(f"  {'state':>7}{'empirical':>12}{'true':>10}{'abs error':>12}")
    print("  " + "-" * 41)
    for st, w in zip(states, weights):
        emp = s.count(st) / len(s)
        tru = w / Zd
        print(f"  {st:>7}{emp:>12.5f}{tru:>10.5f}{abs(emp-tru):>12.5f}")

    # ---- H: Bayesian posterior -------------------------------------------
    print("\n\n--- H. The standard application: Bayesian inference ---\n")
    print("  Data: 40 coin flips, 27 heads. Estimate the bias p.")
    print("  Prior:      p ~ Uniform(0,1)")
    print("  Likelihood: p^27 (1-p)^13")
    print("  Posterior proportional to p^27 (1-p)^13   [a Beta(28, 14)]")
    print("  We sample it with MH and compare to the known exact answer.\n")

    n_flips, n_heads = 40, 27

    def log_posterior(p):
        if not 0.0 < p < 1.0:
            return -float('inf')
        return n_heads * math.log(p) + (n_flips - n_heads) * math.log(1 - p)

    s, a, _ = metropolis_hastings(log_posterior, 0.5, 60_000, 0.1,
                                  LCG().reset(), log_target=True, burn_in=2_000)
    alpha_post, beta_post = n_heads + 1, n_flips - n_heads + 1
    true_mean = alpha_post / (alpha_post + beta_post)
    true_var = (alpha_post * beta_post /
                ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1)))
    print(f"  Acceptance rate = {a:.3f}")
    summarize(s, f"Posterior for p (exact answer: Beta({alpha_post}, {beta_post})):",
              true_mean=true_mean, true_var=true_var)
    srt = sorted(s)
    lo95, hi95 = srt[int(0.025*len(srt))], srt[int(0.975*len(srt))]
    print(f"    95% credible interval = [{lo95:.4f}, {hi95:.4f}]")
    print(f"    (exact Beta(28,14) 95% interval is about [0.523, 0.813])")
    print(f"\n  Note we used LOG densities here. For likelihoods built from")
    print(f"  many data points the raw product underflows to 0.0; working in")
    print(f"  logs and comparing log(alpha) >= 0 is the standard fix.")

    # ---- summary ---------------------------------------------------------
    print("\n\n" + "=" * 78)
    print("CHECKLIST -- what graders look for in an MH implementation")
    print("=" * 78)
    print("""
  1. The acceptance ratio uses f(x')/f(x) -- the normalising constant must
     cancel. If you needed Z, you did not need MH.
  2. On REJECTION you record the CURRENT state again. Skipping rejected
     steps is the classic bug and it biases the whole sample.
  3. A symmetric proposal (random walk) lets you drop the q ratio. For an
     asymmetric proposal you must keep the full Hastings correction
         alpha = min(1, [f(x') q(x|x')] / [f(x) q(x'|x)]).
  4. Discard a burn-in; the start point is not a sample from the target.
  5. Report the acceptance rate and tune the proposal width toward it.
  6. Samples are correlated -- quote an effective sample size, not n.
  7. Use log densities whenever the target involves a product of many terms.
  8. Run multiple chains from different starts to detect a trapped chain.""")


if __name__ == "__main__":
    main()
