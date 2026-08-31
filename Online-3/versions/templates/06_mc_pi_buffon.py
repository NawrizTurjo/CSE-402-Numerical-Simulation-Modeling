"""
06_mc_pi_buffon.py
==================
TEMPLATE: "Estimate pi by Monte Carlo and assess the simulation's performance."

Two classic estimators, plus the performance analysis that these questions
always end with.

  A. Buffon's needle          -- full worked answer to sample problem A1
  B. Quarter-circle / dartboard (RNG deck, slides 69-70)
  C. Task 3-style performance analysis: does more sampling always help?

The headline result: with the assignment's LCG (m = 10000), the estimate
STOPS improving past a certain N, because the generator's period runs out.
The code proves this rather than asserting it.

Run:  python3 06_mc_pi_buffon.py
"""

import math

PI_TRUE = 3.14159265


# ===========================================================================
# TASK 1 -- the LCG (no `random` module)
# ===========================================================================

class LCG:
    """X_{n+1} = (a*X_n + c) mod m,   U_n = X_n / m."""

    def __init__(self, m=10000, a=21, c=1, seed=2500):
        self.m, self.a, self.c, self.seed = m, a, c, seed
        self.x = seed

    def reset(self):
        self.x = self.seed
        return self

    def u(self):
        self.x = (self.a * self.x + self.c) % self.m
        return self.x / self.m


def lcg_uniforms(n, m=10000, a=21, c=1, seed=2500):
    out, x = [], seed
    for _ in range(n):
        x = (a * x + c) % m
        out.append(x / m)
    return out


def lcg_period(m=10000, a=21, c=1, seed=2500):
    seen, x, i = {seed: 0}, seed, 0
    while True:
        x = (a * x + c) % m
        i += 1
        if x in seen:
            return i - seen[x], seen[x]
        seen[x] = i


# ===========================================================================
# TASK 2 -- Buffon's needle
# ===========================================================================

def buffon(n_needles, L=1.0, D=2.0, rng=None):
    """Drop n needles of length L across lines spaced D apart (L <= D).

    For each needle, two independent uniforms are consumed:
      x     ~ U(0, D/2)    distance from the needle's CENTRE to the nearest line
      theta ~ U(0, pi/2)   acute angle between the needle and the lines

    The needle crosses a line iff   x <= (L/2) * sin(theta).

    Theory:  P = 2L / (pi * D)     =>     pi_hat = 2L / (D * P_hat)
    """
    if rng is None:
        rng = LCG().reset()
    crossings = 0
    for _ in range(n_needles):
        x = (D / 2.0) * rng.u()
        theta = (math.pi / 2.0) * rng.u()
        if x <= (L / 2.0) * math.sin(theta):
            crossings += 1
    p_hat = crossings / n_needles
    pi_hat = (2.0 * L) / (D * p_hat) if p_hat > 0 else float('inf')
    return crossings, p_hat, pi_hat


# ===========================================================================
# TASK 2b -- quarter circle (the other standard pi estimator)
# ===========================================================================

def mc_pi_circle(n_points, rng=None):
    """Generate (x, y) in the unit square; count hits with x^2 + y^2 <= 1.

        P(inside) = (pi/4) / 1 = pi/4   =>   pi_hat = 4 * hits / n
    """
    if rng is None:
        rng = LCG().reset()
    hits = 0
    for _ in range(n_points):
        x, y = rng.u(), rng.u()
        if x * x + y * y <= 1.0:
            hits += 1
    return hits, 4.0 * hits / n_points


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    L, D = 1.0, 2.0
    M, A, C, X0 = 10000, 21, 1, 2500

    print("=" * 78)
    print("MONTE CARLO ESTIMATION OF pi USING BUFFON'S NEEDLE")
    print("=" * 78)

    # ---- TASK 1 ---------------------------------------------------------
    print(f"\n--- TASK 1: the LCG ---")
    print(f"  X_(n+1) = ({A} * X_n + {C}) mod {M},   X0 = {X0},   U_n = X_n / m")
    first = lcg_uniforms(8, M, A, C, X0)
    print(f"  First 8 U_n: {[f'{u:.4f}' for u in first]}")
    print(f"  Two uniforms are consumed per needle: one for the position x,")
    print(f"  one for the orientation theta.")

    period, tail = lcg_period(M, A, C, X0)
    print(f"\n  Period of this generator = {period:,} (tail = {tail}).")
    print(f"  [Full period: c=1 is coprime to m; a-1 = {A-1} is divisible by")
    print(f"   m's prime factors 2 and 5; 4 | m and 4 | {A-1}. Hull-Dobell "
          f"satisfied.]")
    print(f"  ** Remember this number -- it dominates Task 3. **")

    # ---- TASK 2 ---------------------------------------------------------
    print(f"\n--- TASK 2: simulate Buffon's needle ---")
    print(f"  L = {L}, D = {D}. Since L <= D, P = 2L/(pi*D) = "
          f"{2*L/(math.pi*D):.6f}")
    print(f"  A needle crosses when x <= (L/2) sin(theta).")
    print(f"  Estimator: pi_hat = 2L / (D * P_hat)\n")

    Ns = [1_000, 10_000, 100_000]
    rows = []
    for N in Ns:
        rng = LCG(M, A, C, X0).reset()       # same seed each time, per the handout
        cr, p_hat, pi_hat = buffon(N, L, D, rng)
        err = abs(pi_hat - PI_TRUE)
        rows.append((N, cr, p_hat, pi_hat, err))

    print(f"  {'N':>9} | {'Crossings':>10} | {'P_hat':>9} | {'pi_hat':>10} | "
          f"{'Absolute Error':>14}")
    print(f"  {'-'*9}-+-{'-'*10}-+-{'-'*9}-+-{'-'*10}-+-{'-'*14}")
    for N, cr, p, ph, e in rows:
        print(f"  {N:>9,} | {cr:>10,} | {p:>9.4f} | {ph:>10.6f} | {e:>14.6f}")

    # ---- TASK 3 ---------------------------------------------------------
    print(f"\n--- TASK 3: assess the Monte Carlo performance ---")

    same = abs(rows[1][3] - rows[2][3]) < 1e-12
    print(f"\n  Q1. Does increasing the number of simulations improve the "
          f"estimate of pi?")
    print(f"""
      Partly, and only up to a point. Going from N = 1,000 to N = 10,000
      cut the error from {rows[0][4]:.6f} to {rows[1][4]:.6f}. But going from
      10,000 to 100,000 changed the error from {rows[1][4]:.6f} to
      {rows[2][4]:.6f} -- {'exactly no change at all.' if same else 'barely any change.'}""")

    print(f"\n  Q2. How does the error change as N increases?")
    print(f"""
      For a genuine Monte Carlo estimator the standard error shrinks like
      1/sqrt(N): to gain one extra decimal digit you need 100x more samples.
      That is slow, and it is the fundamental cost of the method.

      Here the error does NOT keep shrinking. It stalls. Below is why.""")

    print(f"\n  ROOT CAUSE (this is the real answer to Q3):")
    needles_per_cycle = period // 2
    print(f"""
      The generator's period is {period:,}, and each needle consumes 2 values,
      so the experiment repeats itself every {needles_per_cycle:,} needles.

        N =   1,000 needles ->   2,000 draws  = {2000/period:.2f} of a period (new information)
        N =  10,000 needles ->  20,000 draws  = {20000/period:.0f} full periods
        N = 100,000 needles -> 200,000 draws  = {200000/period:.0f} full periods

      N = 10,000 is {10000//needles_per_cycle} identical copies of the first {needles_per_cycle:,} needles.
      N = 100,000 is {100000//needles_per_cycle} identical copies of the SAME {needles_per_cycle:,} needles.
      Averaging the same block over and over does not add information, so
      P_hat -- and therefore pi_hat -- is literally unchanged.""")

    # Prove it numerically
    rng = LCG(M, A, C, X0).reset()
    cr_block, p_block, pi_block = buffon(needles_per_cycle, L, D, rng)
    print(f"\n      Proof: simulating just one block of {needles_per_cycle:,} needles gives")
    print(f"        crossings = {cr_block:,}, P_hat = {p_block:.6f}, "
          f"pi_hat = {pi_block:.6f}")
    print(f"      and 10,000 needles gives crossings = {rows[1][1]:,} "
          f"= {rows[1][1]//cr_block} x {cr_block:,}")
    print(f"      and 100,000 needles gives crossings = {rows[2][1]:,} "
          f"= {rows[2][1]//cr_block} x {cr_block:,}")
    print(f"      Identical pi_hat: {abs(pi_block - rows[2][3]) < 1e-12}")

    print(f"\n  Q3. Does this match what you expect from Monte Carlo? Would")
    print(f"      going from 100,000 to 1,000,000 necessarily be more accurate?")
    print(f"""
      No -- and the failure is in the GENERATOR, not in the Monte Carlo method.

      Genuine Monte Carlo requires i.i.d. samples. Once n exceeds the
      generator's period, the "new" samples are exact copies of old ones, so
      the effective sample size is capped at {needles_per_cycle:,} needles no matter how
      many we nominally draw. The 1/sqrt(N) error law assumes independent
      samples; with repeats, N stops growing in any meaningful sense and the
      error plateaus at whatever bias that particular block happens to have.

      So increasing N from 100,000 to 1,000,000 would NOT make the estimate
      more accurate. It would return the identical value (200 copies of the
      same block instead of 20) at ten times the cost.""")

    # Demonstrate the prediction for N = 1,000,000
    rng = LCG(M, A, C, X0).reset()
    cr6, p6, pi6 = buffon(1_000_000, L, D, rng)
    print(f"      Verified: N = 1,000,000 gives pi_hat = {pi6:.6f}, "
          f"error = {abs(pi6 - PI_TRUE):.6f}")
    print(f"      -- identical to N = 100,000 ({rows[2][3]:.6f}). "
          f"10x the work, zero improvement.")

    print(f"""
      THE FIX: use a generator whose period vastly exceeds the number of
      values you will consume. With a = 16807, m = 2^31 - 1 the period is
      over 2 billion, so 2,000,000 draws use less than 0.1% of it.""")

    print(f"\n  Same experiment with a proper generator (a=16807, m=2^31-1):\n")
    print(f"  {'N':>9} | {'Crossings':>10} | {'pi_hat':>10} | "
          f"{'Absolute Error':>14}")
    print(f"  {'-'*9}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}")
    for N in [1_000, 10_000, 100_000, 1_000_000]:
        rng = LCG(m=2**31 - 1, a=16807, c=0, seed=123457).reset()
        cr, p_hat, pi_hat = buffon(N, L, D, rng)
        print(f"  {N:>9,} | {cr:>10,} | {pi_hat:>10.6f} | "
              f"{abs(pi_hat - PI_TRUE):>14.6f}")
    print(f"\n  Now the error genuinely decreases -- roughly like 1/sqrt(N).")

    # ---- Part B: quarter circle -----------------------------------------
    print(f"\n\n{'='*78}")
    print("ALTERNATIVE ESTIMATOR: QUARTER CIRCLE (RNG deck, slides 69-70)")
    print("=" * 78)
    print(f"\n  Generate (x, y) ~ U(0,1)^2; count points with x^2 + y^2 <= 1.")
    print(f"  P(inside) = area of quarter circle / area of square = "
          f"(pi/4)/1 = pi/4")
    print(f"  => pi_hat = 4 * hits / n\n")
    print(f"  {'n':>10} | {'hits':>10} | {'pi_hat':>10} | {'Error':>10}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        rng = LCG(m=2**31 - 1, a=16807, c=0, seed=123457).reset()
        hits, pi_hat = mc_pi_circle(n, rng)
        print(f"  {n:>10,} | {hits:>10,} | {pi_hat:>10.5f} | "
              f"{abs(pi_hat - PI_TRUE):>10.5f}")
    print(f"\n  Slide 70 reports 3.24, 3.16, 3.1436, 3.1412, 3.14178 for the")
    print(f"  same n values -- same behaviour, different random stream.")

    # ---- Error scaling ---------------------------------------------------
    print(f"\n\n{'='*78}")
    print("WHY THE ERROR SHRINKS LIKE 1/sqrt(N)")
    print("=" * 78)
    p = 2 * L / (math.pi * D)
    print(f"""
  The crossing indicator is Bernoulli(p) with p = 2L/(pi*D) = {p:.6f}.
  For N needles,  Var(P_hat) = p(1-p)/N,  so  SE(P_hat) = sqrt(p(1-p)/N).

  pi = 2L/(D*P), so by the delta method
      SE(pi_hat) ~ |d pi/d P| * SE(P_hat) = (2L/(D p^2)) * sqrt(p(1-p)/N)

  {'N':>10} | {'SE(pi_hat)':>12} | {'95% half-width':>15}""")
    for N in [1_000, 10_000, 100_000, 1_000_000]:
        se = (2 * L / (D * p * p)) * math.sqrt(p * (1 - p) / N)
        print(f"  {N:>10,} | {se:>12.6f} | {1.96*se:>15.6f}")
    print(f"""
  Every 100-fold increase in N buys one extra decimal digit. This is the
  price of Monte Carlo -- and it only holds while the samples are genuinely
  independent, which is exactly what the short-period LCG destroyed above.""")


if __name__ == "__main__":
    main()
