"""
01_lcg_period_ks.py
===================
TEMPLATE: "Implement an LCG / MCG, test it for uniformity, find its period,
           and explain why a uniformity test can miss a structural defect."

This is the full worked answer to sample problem C2 (MCG m=2^16, a=5, X0=1),
written so you can swap the four parameters at the top and re-run.

Covers:
  Task 1  LCG / MCG implementation (no `random` module)
  Task 2  One-sample two-sided Kolmogorov-Smirnov test, computed by hand
  Task 3  (a) Period computed programmatically vs. the theoretical value
          (b) Why a short period is invisible to a one-sample K-S test

Run:  python3 01_lcg_period_ks.py
"""

import math

# ---------------------------------------------------------------------------
# PARAMETERS -- change these and everything below adapts
# ---------------------------------------------------------------------------
M      = 65536          # modulus  m = 2^16
A      = 5              # multiplier
C      = 0              # increment (0 => multiplicative congruential)
X0     = 1              # seed
N      = 100_000        # sample size for the K-S test
ALPHA  = 0.05


# ===========================================================================
# TASK 1 -- Implement the generator
# ===========================================================================

def lcg_generator(n_samples, m=M, a=A, X0=X0, c=C):
    """X_{n+1} = (a*X_n + c) mod m,   U_n = X_n / m.

    Returns a list of n_samples values in [0, 1).
    """
    out = []
    x = X0
    for _ in range(n_samples):
        x = (a * x + c) % m
        out.append(x / m)
    return out


# ===========================================================================
# TASK 2 -- Kolmogorov-Smirnov test, H0: U_n ~ Uniform(0,1)
# ===========================================================================

def ks_statistic(values):
    """Two-sided K-S statistic computed from the definition.

        D+ = max_{1<=i<=N} ( i/N - U_(i) )
        D- = max_{1<=i<=N} ( U_(i) - (i-1)/N )
        D  = max(D+, D-)
    """
    U = sorted(values)
    n = len(U)
    d_plus = max(i / n - U[i - 1] for i in range(1, n + 1))
    d_minus = max(U[i - 1] - (i - 1) / n for i in range(1, n + 1))
    return max(d_plus, d_minus), d_plus, d_minus


def ks_critical_large_sample(n, alpha=0.05):
    """Large-sample critical value  D_crit = c(alpha) / sqrt(N).

    c(0.10) = 1.22,  c(0.05) = 1.36,  c(0.01) = 1.63
    """
    c = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}[alpha]
    return c / math.sqrt(n)


# ===========================================================================
# TASK 3a -- Period, computed programmatically
# ===========================================================================

def find_period(m=M, a=A, c=C, X0=X0):
    """Return (period, tail). Detects the first repeated state.

    tail (mu) = number of states before the cycle is entered.
    For a pure multiplicative generator with odd seed, tail = 0.
    """
    seen = {X0: 0}
    x, i = X0, 0
    while True:
        x = (a * x + c) % m
        i += 1
        if x in seen:
            return i - seen[x], seen[x]
        seen[x] = i


def theoretical_period(m=M, a=A, c=C, X0=X0):
    """The slide rules (RNG deck, slide 22) for maximum period."""
    is_pow2 = m > 0 and (m & (m - 1)) == 0
    b = int(math.log2(m)) if is_pow2 else None

    if c != 0 and is_pow2:
        ok = (math.gcd(c, m) == 1) and ((a - 1) % 4 == 0)
        return (m if ok else None,
                f"Case 1 (m = 2^{b}, c != 0): max period m = {m} when "
                f"gcd(c,m)=1 and a = 1+4k.  gcd(c,m)={math.gcd(c,m)}, "
                f"(a-1)%4={(a-1)%4} -> {'satisfied' if ok else 'NOT satisfied'}")

    if c == 0 and is_pow2:
        ok_seed = (X0 % 2 == 1)
        ok_mult = (a % 8 in (3, 5))
        p = m // 4 if (ok_seed and ok_mult) else None
        return (p,
                f"Case 2 (m = 2^{b}, c = 0): max period m/4 = {m // 4} when "
                f"X0 is odd and a = 3+8k or 5+8k.\n"
                f"           X0 = {X0} is {'odd' if ok_seed else 'EVEN'}; "
                f"a mod 8 = {a % 8} -> {'valid' if ok_mult else 'INVALID'}")

    return None, "No simple closed form for these parameters."


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 78)
    print(f"LCG:  X_(n+1) = ({A} * X_n + {C}) mod {M},   X0 = {X0}")
    print(f"      {'multiplicative (c = 0)' if C == 0 else 'mixed (c != 0)'}")
    print("=" * 78)

    # ---- TASK 1 ----------------------------------------------------------
    print("\n--- TASK 1: generate the stream ---")
    U = lcg_generator(N)
    print(f"Generated N = {len(U):,} values.")
    print("First 10 U_n:", [f"{u:.6f}" for u in U[:10]])
    mean = sum(U) / len(U)
    var = sum((u - mean) ** 2 for u in U) / len(U)
    print(f"Sample mean     = {mean:.6f}   (theoretical 0.500000)")
    print(f"Sample variance = {var:.6f}   (theoretical 1/12 = 0.083333)")

    # ---- TASK 2 ----------------------------------------------------------
    print("\n--- TASK 2: Kolmogorov-Smirnov test ---")
    print(f"H0: U_n ~ Uniform(0,1)      H1: U_n !~ Uniform(0,1)   alpha = {ALPHA}")
    D, dp, dm = ks_statistic(U)
    Dcrit = ks_critical_large_sample(N, ALPHA)
    print(f"  D+   = {dp:.8f}")
    print(f"  D-   = {dm:.8f}")
    print(f"  D_N  = max(D+, D-) = {D:.8f}")
    print(f"  D_crit = 1.36/sqrt({N}) = {Dcrit:.8f}")
    if D > Dcrit:
        print(f"  DECISION: D_N > D_crit  ->  REJECT H0 "
              f"(the stream is not uniform)")
    else:
        print(f"  DECISION: D_N <= D_crit  ->  FAIL TO REJECT H0 "
              f"(no evidence against uniformity)")

    # ---- TASK 3a ---------------------------------------------------------
    print("\n--- TASK 3(a): period ---")
    p, tail = find_period()
    print(f"  Computed period  = {p:,}   (tail before cycle = {tail})")
    tp, explanation = theoretical_period()
    print(f"  Theory: {explanation}")
    if tp is not None:
        print(f"  Theoretical max period = {tp:,}")
        print(f"  MATCH: {'YES' if p == tp else 'NO'}")

    # Show the wrap-around explicitly
    print(f"\n  Verification that the stream repeats with period {p:,}:")
    for k in (0, 1, 2, 3):
        print(f"    U[{k}] = {U[k]:.8f}   U[{k}+{p}] = {U[k + p]:.8f}   "
              f"identical: {U[k] == U[k + p]}")

    # ---- TASK 3b ---------------------------------------------------------
    print("\n--- TASK 3(b): why the K-S test does not notice ---")
    n_cycles = N / p
    distinct = len(set(U))
    print(f"  N = {N:,}, period p = {p:,}  ->  the stream is the SAME "
          f"{p:,} numbers repeated {n_cycles:.1f} times.")
    print(f"  Distinct values in the stream: {distinct:,} (not {N:,}).")
    print(f"""
  What happens for n > p:
      U_n = U_(n mod p). The sequence is exactly periodic -- after {p:,}
      draws the generator returns to its starting state and replays the
      identical numbers in the identical order. There is no new information
      after index {p:,}; the remaining {N - p:,} draws are copies.

  Why uniformity is not violated:
      The {p:,} distinct values are the odd multiples of 1/{M} (the full
      residue class the MCG can reach), and they tile [0,1) very evenly.
      Repeating an even set {n_cycles:.0f} times leaves it just as even --
      each repetition scales every bin count by the same factor, so the
      EMPIRICAL CDF is essentially unchanged and D_N stays tiny.

      The one-sample K-S test only looks at the MARGINAL distribution: it
      asks "what fraction of values are <= x?" and never asks "in what
      ORDER did they arrive?" or "are they distinct?". Periodicity is a
      dependence/structure defect, not a marginal-distribution defect, so
      this test is blind to it by construction.

  What WOULD catch it:
      - a serial / lag-plot test on pairs (U_n, U_(n+1)): the MCG's points
        fall on a small number of parallel lines (Marsaglia's lattice)
      - an autocorrelation test at lag {p:,} (correlation = exactly 1.0)
      - simply counting distinct values, as above
      - the spectral test""")

    # Demonstrate the lattice defect concretely
    print("  Concrete evidence of the structural defect (Marsaglia's lattice):")
    print(f"    Because U_(n+1) = ({A} * U_n) mod 1, every consecutive pair")
    print(f"    (U_n, U_(n+1)) satisfies  U_(n+1) = {A}*U_n - k  for an integer k.")
    ks_seen = sorted({round(A * U[i] - U[i + 1]) for i in range(2000)})
    print(f"    Over 2000 consecutive pairs, k takes only the values {ks_seen}")
    print(f"    -> all 2000 points lie on just {len(ks_seen)} parallel lines,")
    print(f"       instead of filling the unit square. This is invisible to a")
    print(f"       one-dimensional (marginal) test such as K-S.")

    # Autocorrelation at lag = period is exactly 1
    lag = p
    if N > 2 * lag:
        xs = U[:1000]
        ys = U[lag:lag + 1000]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = math.sqrt(sum((a - mx) ** 2 for a in xs) *
                        sum((b - my) ** 2 for b in ys))
        print(f"    Pearson correlation at lag {lag:,} = {num/den:.6f} "
              f"(perfect dependence)")


if __name__ == "__main__":
    main()
