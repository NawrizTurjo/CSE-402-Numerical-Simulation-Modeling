"""
09_des_variants.py
==================
TEMPLATE: the variations on the single-server queue that exams actually ask for.

  A. Multi-server queue (M/M/c)  -- "how many servers do we need?"
  B. Finite capacity / balking   -- "what fraction of customers are lost?"
  C. Warm-up period and the initial transient (slide 69)
  D. Comparing two designs with COMMON RANDOM NUMBERS (slide 63)
  E. Time-terminated vs count-terminated runs
  F. Inventory system (the other classic DES example)
  G. Fixed-increment vs next-event time advance (slide 23)

Run:  python3 09_des_variants.py
"""

import math

INF = float('inf')


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

    def expo(self, mean):
        return -mean * math.log(1.0 - self.u())


# ===========================================================================
# A. MULTI-SERVER QUEUE
# ===========================================================================

def simulate_mmc(lam, mu, c, n_customers, rng, capacity=None, warmup=0,
                 arrival_rng=None, service_rng=None, record_delays=False):
    """M/M/c: c identical servers, one shared FIFO queue.

    The event list now holds c departure times (one per server) plus the
    next arrival. INF marks an idle server.

    `capacity` limits the number in the SYSTEM; arrivals that find the system
    full are turned away (balk) and counted separately.
    `warmup` customers are simulated but excluded from the statistics.

    For COMMON RANDOM NUMBERS across competing designs, pass separate
    `arrival_rng` and `service_rng`. Drawing arrivals and services from
    different streams keeps them SYNCHRONIZED: customer i gets the same
    arrival time and the same service requirement no matter how many servers
    the design has. Sharing one stream does NOT achieve this -- the designs
    consume draws in a different order and desynchronize immediately.
    """
    arr_rng = arrival_rng if arrival_rng is not None else rng
    svc_rng = service_rng if service_rng is not None else rng
    clock = 0.0
    dep = [INF] * c                       # departure time per server
    queue = []                            # arrival times of waiting customers
    t_next_arrival = arr_rng.expo(1.0 / lam)

    n_delayed = 0
    total_delay = 0.0
    n_balked = 0
    n_arrivals = 0
    area_q = 0.0
    area_busy = 0.0                       # sum over servers of busy time
    t_last = 0.0
    stats_start_time = 0.0
    delay_log = []

    while n_delayed < n_customers + warmup:
        t_dep = min(dep)
        if t_next_arrival <= t_dep:
            clock, is_arrival = t_next_arrival, True
        else:
            clock, is_arrival = t_dep, False

        # -- accumulate areas with the OLD state --
        dt = clock - t_last
        area_q += len(queue) * dt
        area_busy += sum(1 for d in dep if d < INF) * dt
        t_last = clock

        if is_arrival:
            n_arrivals += 1
            t_next_arrival = clock + arr_rng.expo(1.0 / lam)
            in_system = len(queue) + sum(1 for d in dep if d < INF)
            if capacity is not None and in_system >= capacity:
                n_balked += 1
                continue
            free = next((i for i, d in enumerate(dep) if d == INF), None)
            if free is not None:
                n_delayed += 1
                if n_delayed > warmup:
                    total_delay += 0.0
                if record_delays:
                    delay_log.append(0.0)
                dep[free] = clock + svc_rng.expo(1.0 / mu)
            else:
                queue.append(clock)
        else:
            i = dep.index(t_dep)
            if queue:
                arr = queue.pop(0)
                n_delayed += 1
                if n_delayed > warmup:
                    total_delay += clock - arr
                if record_delays:
                    delay_log.append(clock - arr)
                dep[i] = clock + svc_rng.expo(1.0 / mu)
            else:
                dep[i] = INF

        if n_delayed == warmup and stats_start_time == 0.0 and warmup > 0:
            stats_start_time = clock
            area_q = 0.0
            area_busy = 0.0

    T = clock - stats_start_time
    counted = n_delayed - warmup
    return {
        "avg_delay": total_delay / counted if counted else 0.0,
        "avg_queue": area_q / T if T > 0 else 0.0,
        "utilization": area_busy / (c * T) if T > 0 else 0.0,
        "n_balked": n_balked,
        "n_arrivals": n_arrivals,
        "balk_fraction": n_balked / n_arrivals if n_arrivals else 0.0,
        "T": T,
        "delay_log": delay_log,
    }


def mmc_theory(lam, mu, c):
    """Erlang-C exact results for M/M/c.

        rho = lam / (c mu)
        P0  = [ sum_{n=0}^{c-1} (c rho)^n / n!  +  (c rho)^c / (c! (1-rho)) ]^-1
        Lq  = P0 (c rho)^c rho / (c! (1 - rho)^2)
        Wq  = Lq / lam
    """
    rho = lam / (c * mu)
    if rho >= 1:
        return None
    a = lam / mu
    s = sum(a ** n / math.factorial(n) for n in range(c))
    s += a ** c / (math.factorial(c) * (1 - rho))
    P0 = 1.0 / s
    Lq = P0 * (a ** c) * rho / (math.factorial(c) * (1 - rho) ** 2)
    return {"rho": rho, "Lq": Lq, "Wq": Lq / lam,
            "L": Lq + a, "W": Lq / lam + 1 / mu, "P0": P0}


# ===========================================================================
# F. INVENTORY SYSTEM  (the other classic DES model)
# ===========================================================================

def simulate_inventory(n_months, s, S, rng,
                       mean_demand_interval=0.1,
                       demand_sizes=(1, 2, 3, 4),
                       demand_probs=(1/6, 1/3, 1/3, 1/6),
                       setup_cost=32.0, incremental_cost=3.0,
                       holding_cost=1.0, shortage_cost=5.0,
                       mean_lag=0.5, max_lag=1.0):
    """(s, S) inventory policy, following Law & Kelton's classic example.

    At the start of each month, if the inventory level I < s, order up to S
    (order quantity S - I). The order arrives after a random delivery lag.
    Demands arrive at random times with random sizes.

    Costs tracked: ordering (setup + incremental), holding, shortage.
    """
    clock = 0.0
    inventory = S
    t_next_demand = rng.expo(mean_demand_interval)
    t_next_eval = 0.0                      # evaluate at the start of month 0
    pending = []                           # (arrival_time, amount)

    total_ordering = 0.0
    area_hold = 0.0
    area_short = 0.0
    t_last = 0.0

    while clock < n_months:
        candidates = [t_next_demand, t_next_eval, n_months]
        if pending:
            candidates.append(min(p[0] for p in pending))
        clock = min(candidates)

        # accumulate holding/shortage areas with the OLD inventory level
        dt = clock - t_last
        if inventory > 0:
            area_hold += inventory * dt
        else:
            area_short += -inventory * dt
        t_last = clock

        if clock >= n_months:
            break

        if pending and clock == min(p[0] for p in pending):
            idx = min(range(len(pending)), key=lambda k: pending[k][0])
            _, amt = pending.pop(idx)
            inventory += amt
        elif clock == t_next_demand:
            size = _discrete(rng.u(), demand_sizes, demand_probs)
            inventory -= size
            t_next_demand = clock + rng.expo(mean_demand_interval)
        elif clock == t_next_eval:
            if inventory < s:
                amount = S - inventory
                total_ordering += setup_cost + incremental_cost * amount
                lag = mean_lag + (max_lag - mean_lag) * rng.u()
                pending.append((clock + lag, amount))
            t_next_eval = clock + 1.0      # next month

    T = n_months
    return {
        "avg_ordering_cost": total_ordering / T,
        "avg_holding_cost": holding_cost * area_hold / T,
        "avg_shortage_cost": shortage_cost * area_short / T,
        "avg_total_cost": (total_ordering + holding_cost * area_hold +
                           shortage_cost * area_short) / T,
    }


def _discrete(u, values, probs):
    cum = 0.0
    for v, p in zip(values, probs):
        cum += p
        if u <= cum:
            return v
    return values[-1]


# ===========================================================================
# G. FIXED-INCREMENT VS NEXT-EVENT TIME ADVANCE
# ===========================================================================

def compare_time_advance(lam, mu, horizon, dt, rng_seed=123457):
    """Slide 23: next-event advance jumps straight to the next event;
    fixed-increment advance steps by dt and checks whether anything happened.

    Counts how much work each approach does for the same simulated horizon.
    """
    # next-event
    rng = LCG(seed=rng_seed).reset()
    clock, dep, n_events = 0.0, INF, 0
    t_arr = rng.expo(1.0 / lam)
    q = 0
    while clock < horizon:
        clock = min(t_arr, dep)
        if clock > horizon:
            break
        n_events += 1
        if t_arr <= dep:
            t_arr = clock + rng.expo(1.0 / lam)
            if dep == INF:
                dep = clock + rng.expo(1.0 / mu)
            else:
                q += 1
        else:
            if q > 0:
                q -= 1
                dep = clock + rng.expo(1.0 / mu)
            else:
                dep = INF
    n_steps_fixed = int(horizon / dt)
    return n_events, n_steps_fixed


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 78)
    print("A. MULTI-SERVER QUEUE: how many servers do we need?")
    print("=" * 78)
    lam, mu = 2.7, 1.0
    print(f"\n  Arrivals rate lambda = {lam}/hr, service rate mu = {mu}/hr")
    print(f"  Offered load = lambda/mu = {lam/mu:.2f}, so we need c >= "
          f"{math.ceil(lam/mu)} servers for a stable system.\n")
    print(f"  {'c':>3} {'rho':>7} | {'Wq sim':>9} {'Wq exact':>9} | "
          f"{'Lq sim':>8} {'Lq exact':>9} | {'util sim':>9}")
    print("  " + "-" * 66)
    for c in [3, 4, 5, 6, 8]:
        rng = LCG().reset()
        res = simulate_mmc(lam, mu, c, 200_000, rng)
        th = mmc_theory(lam, mu, c)
        print(f"  {c:>3} {th['rho']:>7.4f} | {res['avg_delay']:>9.4f} "
              f"{th['Wq']:>9.4f} | {res['avg_queue']:>8.4f} {th['Lq']:>9.4f} | "
              f"{res['utilization']:>9.4f}")

    print(f"""
  FACE VALIDITY (slide 86): adding a server reduces the delay and the queue
  length while lowering each server's utilization. The model responds in the
  direction intuition demands, which is what a face-validity check asks for.

  DECISION SUPPORT: the gain is steeply diminishing. Going 3 -> 4 servers
  cuts the wait dramatically; 6 -> 8 barely helps but costs two salaries.
  This trade-off is the actual reason the study was commissioned.""")

    # ---- B: finite capacity ---------------------------------------------
    print("\n\n" + "=" * 78)
    print("B. FINITE CAPACITY: what fraction of customers are turned away?")
    print("=" * 78)
    print(f"\n  lambda = {lam}, mu = {mu}, c = 3 servers, varying system capacity K")
    print(f"  Customers who arrive to find the system full BALK (are lost).\n")
    print(f"  {'K':>4} | {'balked':>8} {'balk %':>8} | {'Wq':>8} {'Lq':>8} "
          f"| {'util':>7}")
    print("  " + "-" * 54)
    for K in [3, 5, 10, 20, 50, None]:
        rng = LCG().reset()
        res = simulate_mmc(lam, mu, 3, 100_000, rng, capacity=K)
        klabel = "inf" if K is None else str(K)
        print(f"  {klabel:>4} | {res['n_balked']:>8,} "
              f"{100*res['balk_fraction']:>7.2f}% | {res['avg_delay']:>8.4f} "
              f"{res['avg_queue']:>8.4f} | {res['utilization']:>7.4f}")
    print(f"""
  A tight capacity keeps the observed delay small -- but only because the
  customers who WOULD have waited were thrown out. Never judge such a system
  by delay alone; the balk fraction is the honest performance measure. This
  is a good example of why slide 79 insists on collecting several statistics
  and forecasting a reasonable range for each before running the model.""")

    # ---- C: warm-up ------------------------------------------------------
    print("\n\n" + "=" * 78)
    print("C. WARM-UP PERIOD AND THE INITIAL TRANSIENT (slide 69)")
    print("=" * 78)
    print("""
  The model starts EMPTY AND IDLE, but the steady state we want to measure
  already has customers in the system. The first customers walk into an
  empty queue and wait almost nothing, so early output is biased LOW.

  The transient is small compared with the run-to-run noise, so a single
  run cannot show it. We use Welch's procedure: run many replications and
  average the delay of the i-th customer ACROSS replications. Averaging
  kills the noise and leaves the transient visible.
""")
    lam2, mu2 = 0.9, 1.0
    Wq_true = (lam2 / mu2) / (mu2 - lam2)
    n_reps, n_cust = 60, 4_000
    print(f"  M/M/1, lambda = {lam2}, mu = {mu2}. Exact Wq = {Wq_true:.4f}")
    print(f"  {n_reps} replications x {n_cust:,} customers.\n")

    sums = [0.0] * n_cust
    for r in range(n_reps):
        rng = LCG(seed=7001 + r * 7919).reset()
        res = simulate_mmc(lam2, mu2, 1, n_cust, rng, record_delays=True)
        for k, d in enumerate(res["delay_log"][:n_cust]):
            sums[k] += d
    avg_i = [t / n_reps for t in sums]

    print(f"  Average delay of the i-th customer (averaged over {n_reps} runs):\n")
    print(f"  {'customer i':>12} | {'mean delay':>11} | {'% of steady state':>18}")
    print("  " + "-" * 49)
    for k in [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 3999]:
        print(f"  {k:>12,} | {avg_i[k-1]:>11.4f} | "
              f"{100*avg_i[k-1]/Wq_true:>17.1f}%")
    print(f"\n  The curve climbs from 0 through ~14%, ~41%, ~61% of the")
    print(f"  steady-state value and reaches it around customer 250. Beyond")
    print(f"  that it fluctuates around {Wq_true:.1f} -- {n_reps} replications damp the")
    print(f"  noise but do not remove it, which is why the transient is read")
    print(f"  from the CLIMB, not from the later wobble.")
    print(f"  Early customers are measurably better off than steady-state")
    print(f"  ones; including them biases a steady-state estimate downward.")

    # cumulative-average bias
    print(f"\n  Effect on the CUMULATIVE average (what you would report):\n")
    print(f"  {'first n customers':>18} | {'cum. avg delay':>15} | {'bias':>9}")
    print("  " + "-" * 49)
    run = 0.0
    cum = []
    for k in range(n_cust):
        run += avg_i[k]
        cum.append(run / (k + 1))
    for n in [10, 50, 100, 500, 1000, 2000, 4000]:
        c_ = cum[n-1]
        print(f"  {n:>18,} | {c_:>15.4f} | "
              f"{100*(c_-Wq_true)/Wq_true:>8.1f}%")
    print(f"""
  The bias is large for short runs and decays as the transient is diluted.
  Two standard fixes:
    * DISCARD a warm-up: drop the first d customers and average the rest.
    * RUN LONGER: make the transient a negligible fraction of the run.

  For a TERMINATING system (a bank that really does open empty at 9 AM and
  close at 4 PM) the empty-and-idle start is CORRECT and you must NOT
  discard anything. Which start is right depends entirely on the question.""")

    # ---- D: common random numbers ---------------------------------------
    print("\n\n" + "=" * 78)
    print("D. COMPARING TWO DESIGNS WITH COMMON RANDOM NUMBERS (CRN)")
    print("=" * 78)
    print("""
  Slide 13 of the RNG deck: "Test two system designs with the exact same
  random events." If both designs face identical customers, the noise from
  customer-to-customer variation cancels in the DIFFERENCE:

      Var(A - B) = Var(A) + Var(B) - 2 Cov(A, B)

  Independent streams give Cov = 0. CRN makes A and B strongly positively
  correlated, so the covariance term subtracts most of the variance away.

  Two things must be right for this to work:
    1. SYNCHRONIZE the streams -- one dedicated stream per source of
       randomness (arrivals, services), so customer i gets the same arrival
       time and the same service requirement in BOTH designs. A single
       shared stream fails: the designs consume draws in a different order
       and desynchronize within a few events.
    2. The two designs must have COMPARABLE variance. If Var(A) dwarfs
       Var(B), then Var(A-B) ~ Var(A) whatever the correlation, and CRN
       cannot help. (Demonstrated at the end of this section.)
""")

    def stats(xs):
        n = len(xs)
        m = sum(xs) / n
        v = sum((x - m) ** 2 for x in xs) / (n - 1)
        return m, math.sqrt(v), 2.045 * math.sqrt(v / n)   # t_(0.025,29)

    def corr(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        return (sum((a - mx) * (b - my) for a, b in zip(xs, ys)) /
                math.sqrt(sum((a - mx) ** 2 for a in xs) *
                          sum((b - my) ** 2 for b in ys)))

    # ---------- the case CRN is FOR: detecting a small difference ----------
    R, N = 30, 20_000
    print(f"  CASE 1 -- a small design change, which is what CRN is for.")
    print(f"  Design A: c = 4, mu = 1.00.   Design B: c = 4, mu = 1.05")
    print(f"  (a 5% faster server). lambda = {lam}, {R} replications of "
          f"{N:,} customers.\n")

    A, B_crn, B_ind = [], [], []
    for r in range(R):
        sa, ss = 10007 + r * 7919, 55001 + r * 6271
        A.append(simulate_mmc(lam, 1.00, 4, N, None,
                              arrival_rng=LCG(seed=sa).reset(),
                              service_rng=LCG(seed=ss).reset())['avg_delay'])
        # CRN: design B sees the SAME arrival and service streams
        B_crn.append(simulate_mmc(lam, 1.05, 4, N, None,
                                  arrival_rng=LCG(seed=sa).reset(),
                                  service_rng=LCG(seed=ss).reset())['avg_delay'])
        # independent: design B gets entirely different streams
        B_ind.append(simulate_mmc(lam, 1.05, 4, N, None,
                                  arrival_rng=LCG(seed=sa + 104729).reset(),
                                  service_rng=LCG(seed=ss + 104729).reset()
                                  )['avg_delay'])

    d_crn = [a - b for a, b in zip(A, B_crn)]
    d_ind = [a - b for a, b in zip(A, B_ind)]
    m_c, sd_c, h_c = stats(d_crn)
    m_i, sd_i, h_i = stats(d_ind)

    print(f"  {'method':<22}{'corr(A,B)':>11}{'mean diff':>11}"
          f"{'sd(diff)':>10}{'95% CI half-width':>20}")
    print("  " + "-" * 74)
    print(f"  {'Common random nums':<22}{corr(A, B_crn):>+11.4f}{m_c:>11.4f}"
          f"{sd_c:>10.4f}{h_c:>20.4f}")
    print(f"  {'Independent streams':<22}{corr(A, B_ind):>+11.4f}{m_i:>11.4f}"
          f"{sd_i:>10.4f}{h_i:>20.4f}")

    print(f"""
  CRN correlates the two designs at {corr(A, B_crn):+.3f} and shrinks the standard
  deviation of the difference from {sd_i:.4f} to {sd_c:.4f} -- a variance
  reduction of about {(sd_i/sd_c)**2:.0f}x for exactly the same computing budget.

  Put another way: to match CRN's precision with independent streams you
  would need roughly {(sd_i/sd_c)**2:.0f} x {R} = {int((sd_i/sd_c)**2*R):,} replications instead of {R}.

  95% CI for the improvement from the faster server:
      CRN         : {m_c:.4f} +/- {h_c:.4f}  ->  [{m_c-h_c:.4f}, {m_c+h_c:.4f}]
      independent : {m_i:.4f} +/- {h_i:.4f}  ->  [{m_i-h_i:.4f}, {m_i+h_i:.4f}]
  Both intervals exclude zero, but CRN pins the effect down {h_i/h_c:.0f}x more tightly.""")

    # ---------- the case where CRN cannot help ----------
    print(f"\n  CASE 2 -- when CRN does NOT help, and why.")
    print(f"  Design A: c = 3 (rho = {lam/3:.2f}, nearly saturated).")
    print(f"  Design B: c = 4 (rho = {lam/4:.2f}, comfortable).\n")

    A2, B2 = [], []
    for r in range(R):
        sa, ss = 10007 + r * 7919, 55001 + r * 6271
        A2.append(simulate_mmc(lam, mu, 3, N, None,
                               arrival_rng=LCG(seed=sa).reset(),
                               service_rng=LCG(seed=ss).reset())['avg_delay'])
        B2.append(simulate_mmc(lam, mu, 4, N, None,
                               arrival_rng=LCG(seed=sa).reset(),
                               service_rng=LCG(seed=ss).reset())['avg_delay'])
    _, sdA2, _ = stats(A2)
    _, sdB2, _ = stats(B2)
    d2 = [a - b for a, b in zip(A2, B2)]
    m2, sd2, h2 = stats(d2)
    print(f"    corr(A, B)     = {corr(A2, B2):+.4f}   (synchronization IS working)")
    print(f"    sd(A), c = 3   = {sdA2:.4f}")
    print(f"    sd(B), c = 4   = {sdB2:.4f}   ({sdA2/sdB2:.0f}x smaller)")
    print(f"    sd(A - B)      = {sd2:.4f}   ~ sd(A) = {sdA2:.4f}")
    print(f"""
    Even at correlation {corr(A2, B2):+.2f}, the difference is no more precise than
    design A alone. With Var(A) >> Var(B), the covariance term
    2*corr*sd(A)*sd(B) is small next to Var(A), so there is simply nothing
    for CRN to cancel. The near-saturated 3-server design is inherently
    noisy and that noise sets the floor.

    LESSON: CRN is a variance-reduction technique, not magic. Check the
    correlation AND the relative variances before claiming a benefit --
    and report the measured reduction rather than assuming one.""")

    # ---- E: terminating vs steady state ----------------------------------
    print("\n\n" + "=" * 78)
    print("E. TIME-TERMINATED VS COUNT-TERMINATED RUNS")
    print("=" * 78)
    print("""
  Count-terminated: "simulate until n customers have been delayed."
      Natural when the question is about customers ("what does the average
      customer experience?"). The run LENGTH is random.

  Time-terminated:  "simulate one 8-hour day."
      Natural when the question is about a period of operation ("how bad does
      it get on a Friday?"). The NUMBER of customers is random.

  They answer different questions and can give different numbers. Choose the
  one that matches the real decision, and say which you used.""")

    # ---- F: inventory ----------------------------------------------------
    print("\n\n" + "=" * 78)
    print("F. INVENTORY SYSTEM: the (s, S) policy")
    print("=" * 78)
    print("""
  A second classic DES model. Each month, if inventory I < s, order S - I
  units; the delivery arrives after a random lag. Demands arrive randomly
  with random sizes. We compare policies on average total cost per month.
""")
    print(f"  {'s':>4}{'S':>5} | {'ordering':>10}{'holding':>10}"
          f"{'shortage':>10}{'TOTAL':>10}")
    print("  " + "-" * 51)
    best = None
    for s_, S_ in [(20, 40), (20, 60), (20, 80), (20, 100),
                   (40, 60), (40, 80), (40, 100), (60, 100)]:
        rng = LCG().reset()
        r = simulate_inventory(1200, s_, S_, rng)
        print(f"  {s_:>4}{S_:>5} | {r['avg_ordering_cost']:>10.2f}"
              f"{r['avg_holding_cost']:>10.2f}{r['avg_shortage_cost']:>10.2f}"
              f"{r['avg_total_cost']:>10.2f}")
        if best is None or r['avg_total_cost'] < best[1]:
            best = ((s_, S_), r['avg_total_cost'])
    print("  " + "-" * 51)
    print(f"\n  Cheapest policy in this experiment: s = {best[0][0]}, "
          f"S = {best[0][1]} at {best[1]:.2f} per month.")
    print(f"  Note the trade-off: a large S raises holding cost but cuts")
    print(f"  shortage and ordering costs. Simulation finds the balance that")
    print(f"  no simple formula gives.")

    # ---- G: time advance -------------------------------------------------
    print("\n\n" + "=" * 78)
    print("G. NEXT-EVENT VS FIXED-INCREMENT TIME ADVANCE (slide 23)")
    print("=" * 78)
    print(f"\n  Simulating 10,000 time units of an M/M/1 queue "
          f"(lambda = 0.5, mu = 1.0):\n")
    n_ev, _ = compare_time_advance(0.5, 1.0, 10_000, 1.0)
    print(f"    NEXT-EVENT advance: {n_ev:>10,} steps -- exactly one per real event.\n")
    print(f"  {'fixed dt':>10}{'steps':>12}{'vs next-event':>16}"
          f"{'events per step':>18}   verdict")
    print("  " + "-" * 74)
    for dt in [1.0, 0.5, 0.1, 0.01]:
        _, n_fx = compare_time_advance(0.5, 1.0, 10_000, dt)
        eps = n_ev / n_fx
        if eps > 0.5:
            verdict = "INACCURATE: events collapse into one step"
        elif n_fx > 3 * n_ev:
            verdict = "wasteful: most steps do nothing"
        else:
            verdict = "borderline"
        print(f"  {dt:>10}{n_fx:>12,}{n_fx/n_ev:>15.1f}x{eps:>18.3f}   {verdict}")
    print("""
  Next-event advance skips idle stretches entirely and never misses an
  event. Fixed increment forces a lose-lose choice of dt:
    * dt too LARGE  -> several events fall inside one step, so they are
                       processed out of order or merged. The answers are
                       simply wrong, and cheapness does not redeem that.
    * dt too SMALL  -> correct, but almost every step finds nothing to do.
                       At dt = 0.01 that is ~100x the work for the same
                       10,000 events.
  This is why virtually all modern simulation software uses next-event
  time advance (slide 23).""")


if __name__ == "__main__":
    main()
