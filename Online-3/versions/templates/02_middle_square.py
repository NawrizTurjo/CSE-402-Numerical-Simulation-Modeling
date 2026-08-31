"""
02_middle_square.py
===================
TEMPLATE: "Implement the Middle-Square generator, find its degenerate seeds,
           and test the output for uniformity with a Chi-Square test."

Full worked answer to sample problem C1. No `random` module is used.

  Task 1  middle_square(seed, n)
  Task 2  find a seed that collapses (repeats the same value forever)
  Task 3  Chi-Square goodness-of-fit on [0,1) with 10 bins, for 3 seeds
  Task 4  reflection: does passing chi-square make it a good generator?

Run:  python3 02_middle_square.py
"""

import math

try:
    from scipy.stats import chisquare, chi2
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

DIGITS = 4
ALPHA = 0.05
N_BINS = 10


# ===========================================================================
# TASK 1 -- the generator
# ===========================================================================

def middle_square(seed, n, digits=DIGITS):
    """Von Neumann's middle-square method.

    Each step:
      1. square the current value
      2. write the result as a 2*digits-digit number, padding with leading zeros
      3. extract the middle `digits` digits
      4. that becomes the next value

    Example (digits = 4):
      5731^2 = 32844361 -> "32844361" -> middle 4 = "8443" -> 8443
      8443^2 = 71284249 -> "71284249" -> middle 4 = "2842" -> 2842

    Returns a list of n integers (the seed itself is not included).
    """
    values = []
    x = seed
    width = 2 * digits
    start = (width - digits) // 2          # = 2 when digits = 4
    for _ in range(n):
        squared = str(x * x).zfill(width)  # leading zeros if needed
        x = int(squared[start:start + digits])
        values.append(x)
    return values


def middle_square_uniform(seed, n, digits=DIGITS):
    """Same sequence mapped into [0, 1) by dividing by 10**digits."""
    return [v / (10 ** digits) for v in middle_square(seed, n, digits)]


# ===========================================================================
# TASK 2 -- degenerate / short-cycle seeds
# ===========================================================================

def analyse_seed(seed, max_steps=5000, digits=DIGITS):
    """Detect the cycle that `seed` falls into.

    Returns a dict with:
      first_repeat        the first value that is seen twice
      first_index         iteration (1-based) at which it first appeared
      repeat_index        iteration at which it appeared again
      cycle_length        repeat_index - first_index
      tail                iterations before the cycle is entered
      collapses_to_zero   True if the generator dies at 0
    """
    seen = {}
    x = seed
    for i in range(1, max_steps + 1):
        squared = str(x * x).zfill(2 * digits)
        start = (2 * digits - digits) // 2
        x = int(squared[start:start + digits])
        if x in seen:
            return {
                "seed": seed,
                "first_repeat": x,
                "first_index": seen[x],
                "repeat_index": i,
                "cycle_length": i - seen[x],
                "tail": seen[x] - 1,
                "collapses_to_zero": x == 0,
            }
        seen[x] = i
    return None


def find_degenerate_seeds(digits=DIGITS, limit=None):
    """Scan 4-digit seeds and classify them by cycle length.

    A cycle_length of 1 means the generator is stuck emitting one value
    forever -- the classic middle-square failure.
    """
    hi = 10 ** digits
    fixed_points, short_cycles = [], []
    rng = range(hi) if limit is None else range(limit)
    for s in rng:
        info = analyse_seed(s, max_steps=200, digits=digits)
        if info is None:
            continue
        if info["cycle_length"] == 1:
            fixed_points.append((s, info))
        elif info["cycle_length"] <= 4:
            short_cycles.append((s, info))
    return fixed_points, short_cycles


# ===========================================================================
# TASK 3 -- Chi-Square goodness of fit
# ===========================================================================

def chi_square_uniform(values, n_bins=N_BINS):
    """Chi-square GOF against Uniform[0,1).

        chi2_0 = sum (O_i - E_i)^2 / E_i ,  E_i = N / n_bins,  df = n_bins - 1
    """
    N = len(values)
    obs = [0] * n_bins
    for v in values:
        k = min(int(v * n_bins), n_bins - 1)
        obs[k] += 1
    exp = N / n_bins
    chi2_stat = sum((o - exp) ** 2 / exp for o in obs)
    df = n_bins - 1
    if HAVE_SCIPY:
        p = float(chi2.sf(chi2_stat, df))
        crit = float(chi2.ppf(1 - ALPHA, df))
    else:
        p, crit = float("nan"), 16.919
    return {"observed": obs, "expected": exp, "chi2": chi2_stat,
            "df": df, "p": p, "crit": crit, "N": N}


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 78)
    print("MIDDLE-SQUARE RANDOM NUMBER GENERATOR")
    print("=" * 78)

    # ---- TASK 1 ---------------------------------------------------------
    print("\n--- TASK 1: generate values ---")
    seed = 5731
    vals = middle_square(seed, 100)
    print(f"Seed = {seed}, first 20 of 100 values:")
    for r in range(0, 20, 10):
        print("   ", vals[r:r + 10])
    print(f"Hand-check: 5731^2 = {5731**2} -> middle 4 = {vals[0]}")
    print(f"            {vals[0]}^2 = {str(vals[0]**2).zfill(8)} "
          f"-> middle 4 = {vals[1]}")
    assert vals[0] == 8443 and vals[1] == 2842, "generator disagrees with handout"
    print("    matches the handout example.")

    # ---- TASK 2 ---------------------------------------------------------
    print("\n--- TASK 2: edge cases ---")
    print("Scanning all 10,000 four-digit seeds for degenerate behaviour...")
    fixed_points, short_cycles = find_degenerate_seeds()

    # Group fixed points by the value they collapse to
    by_value = {}
    for s, info in fixed_points:
        by_value.setdefault(info["first_repeat"], []).append(s)

    print(f"\n  {len(fixed_points):,} of 10,000 seeds ({100*len(fixed_points)/10000:.1f}%)"
          f" collapse to a CYCLE OF LENGTH 1 (same value forever).")
    print(f"  They get stuck on these values: {sorted(by_value.keys())}")
    for v in sorted(by_value):
        print(f"      value {v:04d}: reached from {len(by_value[v]):,} seeds "
              f"(e.g. {sorted(by_value[v])[:6]})")

    # The classic self-squaring seed: report it in the requested format
    print("\n  A seed that is degenerate from the very first step:")
    for cand in (3792, 0, 100, 2500, 540):
        info = analyse_seed(cand)
        if info and info["cycle_length"] == 1 and info["tail"] == 0:
            print(f"      Seed:                        {cand}")
            print(f"      First repeated value:        {info['first_repeat']}")
            print(f"      Iteration at which it repeats: {info['repeat_index']}")
            print(f"      Cycle length:                {info['cycle_length']}")
            print(f"      (check: {cand}^2 = {str(cand**2).zfill(8)}, "
                  f"middle 4 = {str(cand**2).zfill(8)[2:6]})")
            PROBLEM_SEED = cand
            break
    else:
        PROBLEM_SEED = fixed_points[0][0]

    # A seed with a short but non-trivial cycle, for contrast
    print("\n  Seeds with short (length 2-4) cycles, for contrast:")
    shown = 0
    for s, info in short_cycles:
        if shown >= 4:
            break
        if info["tail"] <= 2:
            print(f"      seed {s:04d}: enters a cycle of length "
                  f"{info['cycle_length']} at iteration {info['first_index']} "
                  f"(value {info['first_repeat']:04d})")
            shown += 1

    # The 6100 family is the textbook 4-cycle
    info = analyse_seed(6100)
    if info:
        chain = middle_square(6100, 6)
        print(f"\n  Textbook 4-cycle, seed 6100: {chain} -> repeats")

    # Longest-lived seed, to show it is not ALL bad
    best = max(((s, analyse_seed(s, 200)) for s in range(10000)),
               key=lambda t: (t[1]["repeat_index"] if t[1] else 0))
    print(f"\n  Best 4-digit seed found: {best[0]:04d} survives "
          f"{best[1]['repeat_index']} iterations before repeating "
          f"(cycle length {best[1]['cycle_length']}).")
    print("  Even the best case is far below the 10,000 values the state space allows.")

    # ---- TASK 3 ---------------------------------------------------------
    print("\n--- TASK 3: Chi-Square goodness-of-fit test ---")
    print(f"H0: the generated numbers are uniformly distributed over [0,1)")
    print(f"Bins: {N_BINS} equal bins,  N = 1000,  E_i = {1000//N_BINS},  "
          f"df = {N_BINS-1},  alpha = {ALPHA}")
    print(f"Decision rule: if p < {ALPHA}, reject H0; otherwise do not reject.\n")

    seeds = [(5731, "5731"), (6239, "6239"), (PROBLEM_SEED, f"{PROBLEM_SEED} (problematic)")]
    rows = []
    for s, label in seeds:
        u = middle_square_uniform(s, 1000)
        res = chi_square_uniform(u)
        dec = "Reject H0" if res["p"] < ALPHA else "Do not reject H0"
        rows.append((label, res["chi2"], res["p"], dec, res["observed"]))

    w = max(len(r[0]) for r in rows)
    print(f"  | {'Seed'.ljust(w)} | {'Chi^2':>12} | {'p-value':>12} | Decision")
    print(f"  |{'-'*(w+2)}|{'-'*14}|{'-'*14}|{'-'*20}")
    for label, c, p, dec, _ in rows:
        print(f"  | {label.ljust(w)} | {c:12.4f} | {p:12.6f} | {dec}")

    crit = chi_square_uniform([0.5] * 10)["crit"]
    print(f"\n  Critical value chi^2_({ALPHA}, {N_BINS-1}) = {crit:.3f} "
          f"(reject when chi^2 exceeds this)")

    print("\n  Bin counts (expected 100 each):")
    for label, c, p, dec, obs in rows:
        print(f"    {label.ljust(w)}: {obs}")

    print("""
  WHY ALL THREE SEEDS ARE REJECTED
  Note the bin counts: only a handful of distinct values ever appear. This is
  the SHORT PERIOD from Task 2 showing up in Task 3. Seed 6239 -- the longest-
  lived 4-digit seed -- still falls into a 4-cycle after ~111 steps, so the
  remaining ~890 of our 1000 draws are the same four numbers repeated. Four
  spikes in a 10-bin histogram gives an enormous chi^2, so H0 is rejected.

  So for the middle-square method the chi-square test does catch the defect --
  but only because the period is catastrophically short. That is luck, not a
  property of the test: the test detects the WRONG HISTOGRAM caused by the
  short period, not the short period itself. The next block shows a generator
  whose period is equally useless but whose histogram is perfect.""")

    # --- Direct evidence for the Task 4 argument --------------------------
    print("\n--- Supporting experiment for Task 4 ---")
    print("  A deliberately terrible but perfectly uniform 'generator':")
    print("      R_n = ((n * 1) mod 1000) / 1000   ->  0.000, 0.001, ..., 0.999")
    counter_example = [((n % 1000) / 1000) for n in range(1000)]
    res = chi_square_uniform(counter_example)
    print(f"      chi^2 = {res['chi2']:.4f}, p = {res['p']:.6f}, "
          f"crit = {res['crit']:.3f}")
    print(f"      Decision: "
          f"{'Reject H0' if res['p'] < ALPHA else 'DO NOT reject H0 -- it PASSES'}")
    print("      Yet every value is perfectly predictable from the previous one.")
    print("      Chi-square is completely satisfied by a sequence with zero")
    print("      randomness. This is the direct proof that passing a uniformity")
    print("      test is necessary but nowhere near sufficient.")

    # And show that an independence test does catch it
    R = counter_example
    Nn, lag, i0 = len(R), 1, 1
    M = 0
    while i0 + (M + 2) * lag <= Nn:
        M += 1
    tot = sum(R[i0 + k * lag - 1] * R[i0 + (k + 1) * lag - 1] for k in range(M + 1))
    rho = tot / (M + 1) - 0.25
    sigma = math.sqrt(13 * M + 7) / (12 * (M + 1))
    print(f"\n      Autocorrelation test at lag 1 on the SAME sequence:")
    print(f"        rho_hat = {rho:.6f}, sigma = {sigma:.6f}, "
          f"Z0 = {rho/sigma:.2f}")
    print(f"        |Z0| = {abs(rho/sigma):.2f} vs z_0.025 = 1.96  ->  "
          f"{'REJECT H0: dependence detected' if abs(rho/sigma) > 1.96 else 'not detected'}")
    print("      The independence test catches what chi-square missed.")

    # ---- TASK 4 ---------------------------------------------------------
    print("\n--- TASK 4: Reflection ---")
    print("""
  Q: Does passing the Chi-Square test necessarily mean the generator is good?

  A: No.

  1. The chi-square test only checks UNIFORMITY -- the marginal distribution,
     i.e. "how many numbers landed in each bin". It says nothing about
     INDEPENDENCE, which is the second sacred property. The sequence
     0.01, 0.02, 0.03, ..., 0.99 passes any uniformity test perfectly and is
     completely predictable.

  2. It is blind to the middle-square method's real defect: a very short
     period. A generator that cycles through the same 50 values 20 times
     produces a perfectly flat histogram, so chi-square is satisfied, while
     the simulation is silently reusing the same 50 "random" numbers.

  3. Failing to reject H0 is not the same as proving H0. A hypothesis test
     can only find evidence AGAINST uniformity; the absence of evidence at
     alpha = 0.05 is weak support, not proof.

  4. Multiple-testing caveat (RNG deck, slides 30-33): if we run k independent
     tests at alpha = 0.05 on a perfectly good generator, the chance of at
     least one false rejection is 1 - 0.95^k -- about 40% for k = 10. So a
     single pass (or a single failure) means little either way.

  A generator is only trustworthy after it passes a BATTERY of tests covering
  both uniformity (chi-square, K-S) and independence (autocorrelation, runs,
  serial, gap, poker), AND has a period far longer than the number of values
  the simulation will consume.""")


if __name__ == "__main__":
    main()
