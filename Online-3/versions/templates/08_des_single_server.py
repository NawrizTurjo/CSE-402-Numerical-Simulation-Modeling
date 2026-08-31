"""
08_des_single_server.py
=======================
TEMPLATE: "Simulate a single-server queue by discrete-event simulation and
           report the average delay, time-average queue length, and server
           utilization."

This is the centrepiece of the DES deck (slides 21-60). The implementation is
structured exactly like the flow-of-control diagram on slide 27:

    initialization -> [ timing routine -> event routine -> stop? ] -> report

  A. Full DES engine with an event trace (verification technique, slide 80)
  B. Verification against the hand simulation on slides 46-56
  C. Stochastic version driven by our own LCG (exponential interarrivals)
  D. Validation against the M/M/1 analytical formulas
  E. Replications and confidence intervals (slides 69-70)

Run:  python3 08_des_single_server.py
"""

import math

INF = float('inf')

# Event type codes
ARRIVAL = 1
DEPARTURE = 2


# ===========================================================================
# RANDOM NUMBER SOURCE  (the "library routines" box on slide 27)
# ===========================================================================

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
        """Exponential variate by inverse transform:
             F(x) = 1 - e^(-x/mean)  =>  X = -mean * ln(1 - U)"""
        return -mean * math.log(1.0 - self.u())


# ===========================================================================
# A. THE SIMULATION ENGINE
# ===========================================================================

class SingleServerQueue:
    """Single-server FIFO queue, next-event time advance.

    STATE VARIABLES (slide 37)
        clock            current simulated time
        server_status    0 = idle, 1 = busy
        num_in_queue     customers waiting (NOT counting the one in service)
        arrival_times    arrival time of each customer currently in the queue
        time_last_event  when the previous event happened (for the areas)

    EVENT LIST
        t_next_arrival, t_next_departure   (INF means "not scheduled")

    STATISTICAL COUNTERS
        num_delayed      customers who have completed their delay
        total_delay      sum of the delays D_i
        area_q           integral of Q(t) dt
        area_b           integral of B(t) dt

    The stopping rule here is "n customers have completed their delay",
    matching the hand simulation. Pass `stop_time` instead for a
    time-terminated run.
    """

    def __init__(self, interarrival_fn, service_fn, n_customers=None,
                 stop_time=None, trace=False):
        self.interarrival = interarrival_fn
        self.service = service_fn
        self.n_target = n_customers
        self.stop_time = stop_time
        self.trace = trace
        self.trace_rows = []
        self.delays = []

    # -- initialization routine (slide 27, box 0) --------------------------
    def initialize(self):
        self.clock = 0.0
        self.server_status = 0            # idle
        self.num_in_queue = 0
        self.arrival_times = []
        self.time_last_event = 0.0

        self.num_delayed = 0
        self.total_delay = 0.0
        self.area_q = 0.0
        self.area_b = 0.0

        self.n_arrivals = 0
        self.delays = []
        self.trace_rows = []

        # event list: first arrival scheduled, no departure yet
        self.t_next_arrival = self.clock + self.interarrival()
        self.t_next_departure = INF

    # -- timing routine (slide 27, right box) ------------------------------
    def timing(self):
        """Find the next event, advance the clock, return the event type."""
        if self.t_next_arrival <= self.t_next_departure:
            next_type, next_time = ARRIVAL, self.t_next_arrival
        else:
            next_type, next_time = DEPARTURE, self.t_next_departure
        if next_time == INF:
            raise RuntimeError("event list is empty -- simulation cannot advance")
        self.clock = next_time
        return next_type

    # -- area accumulators -------------------------------------------------
    def update_time_avg_stats(self):
        """CRITICAL ORDER (slide 44/58): call this BEFORE changing any state.

        Rectangle width  = clock - time_last_event
        Rectangle height = the OLD value of Q(t) / B(t)
        """
        dt = self.clock - self.time_last_event
        self.area_q += self.num_in_queue * dt        # old Q(t)
        self.area_b += self.server_status * dt       # old B(t)
        self.time_last_event = self.clock
        return dt

    # -- event routine: arrival --------------------------------------------
    def arrive(self):
        dt = self.update_time_avg_stats()             # 1. areas first
        self.n_arrivals += 1
        cust = self.n_arrivals
        old_q, old_b = self.num_in_queue, self.server_status

        # schedule the NEXT arrival
        self.t_next_arrival = self.clock + self.interarrival()

        if self.server_status == 1:
            # server busy -> join the queue
            self.num_in_queue += 1
            self.arrival_times.append(self.clock)
            note = f"C{cust} joins queue (len {self.num_in_queue})"
        else:
            # server idle -> straight into service, zero delay
            delay = 0.0
            self.delays.append(delay)
            self.num_delayed += 1
            self.total_delay += delay
            self.server_status = 1
            s = self.service()
            self.t_next_departure = self.clock + s
            note = (f"C{cust} -> service immediately, D{self.num_delayed}=0.0, "
                    f"S={s:.4g}, dep at {self.t_next_departure:.4g}")

        self._trace("Arrival", f"C{cust}", dt, old_q, old_b, note)

    # -- event routine: departure ------------------------------------------
    def depart(self):
        dt = self.update_time_avg_stats()             # 1. areas first
        old_q, old_b = self.num_in_queue, self.server_status

        if self.num_in_queue == 0:
            # nobody waiting -> server goes idle, cancel the departure event
            self.server_status = 0
            self.t_next_departure = INF
            note = "queue empty -> server IDLE, next departure = INF"
        else:
            self.num_in_queue -= 1
            arr = self.arrival_times.pop(0)           # FIFO
            delay = self.clock - arr
            self.delays.append(delay)
            self.num_delayed += 1
            self.total_delay += delay
            s = self.service()
            self.t_next_departure = self.clock + s
            note = (f"next customer -> service, D{self.num_delayed}="
                    f"{delay:.4g}, S={s:.4g}, dep at {self.t_next_departure:.4g}")

        self._trace("Departure", "-", dt, old_q, old_b, note)

    def _trace(self, etype, who, dt, old_q, old_b, note):
        if self.trace:
            self.trace_rows.append({
                "clock": self.clock, "type": etype, "who": who, "dt": dt,
                "old_q": old_q, "old_b": old_b,
                "area_q": self.area_q, "area_b": self.area_b,
                "q": self.num_in_queue, "b": self.server_status,
                "n_delayed": self.num_delayed, "total_delay": self.total_delay,
                "note": note,
            })

    # -- main program (slide 27, centre box) -------------------------------
    def run(self):
        self.initialize()
        while True:
            if self.n_target is not None and self.num_delayed >= self.n_target:
                break
            if self.stop_time is not None and \
               min(self.t_next_arrival, self.t_next_departure) > self.stop_time:
                self.clock = self.stop_time
                self.update_time_avg_stats()
                break
            etype = self.timing()
            if etype == ARRIVAL:
                self.arrive()
            else:
                self.depart()
        return self.report()

    # -- report generator --------------------------------------------------
    def report(self):
        T = self.clock
        return {
            "T": T,
            "num_delayed": self.num_delayed,
            "total_delay": self.total_delay,
            "avg_delay": self.total_delay / self.num_delayed
                         if self.num_delayed else 0.0,
            "area_q": self.area_q,
            "avg_queue": self.area_q / T if T > 0 else 0.0,
            "area_b": self.area_b,
            "utilization": self.area_b / T if T > 0 else 0.0,
            "delays": list(self.delays),
        }

    def print_trace(self):
        """Event-by-event trace -- the verification technique of slide 80."""
        print(f"  {'clock':>7} {'event':<10} {'who':<4} {'dt':>5} "
              f"{'Q_old':>5} {'B_old':>5} {'areaQ':>7} {'areaB':>7} "
              f"{'Q':>2} {'B':>2} {'#D':>3}  note")
        print("  " + "-" * 118)
        for r in self.trace_rows:
            print(f"  {r['clock']:>7.2f} {r['type']:<10} {r['who']:<4} "
                  f"{r['dt']:>5.2f} {r['old_q']:>5} {r['old_b']:>5} "
                  f"{r['area_q']:>7.2f} {r['area_b']:>7.2f} "
                  f"{r['q']:>2} {r['b']:>2} {r['n_delayed']:>3}  {r['note']}")


# ===========================================================================
# B. VERIFICATION AGAINST THE HAND SIMULATION (slides 46-56)
# ===========================================================================

def verify_against_hand_simulation():
    print("=" * 78)
    print("B. VERIFICATION vs THE HAND SIMULATION (DES deck, slides 46-56)")
    print("=" * 78)

    A = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]     # interarrival times
    S = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]                    # service times

    print(f"\n  Interarrival times A_i: {A}")
    print(f"  Service times      S_i: {S}")
    print(f"  Arrival times (cumulative sums): "
          f"{[round(sum(A[:i+1]), 1) for i in range(len(A))]}")
    print(f"  Stop when n = 6 customers have completed their delays.\n")

    a_iter, s_iter = iter(A), iter(S)
    sim = SingleServerQueue(
        interarrival_fn=lambda: next(a_iter),
        service_fn=lambda: next(s_iter),
        n_customers=6, trace=True)
    res = sim.run()

    sim.print_trace()

    print(f"\n  --- Report generator ---")
    print(f"  Simulation ended at T(6) = {res['T']:.1f}   "
          f"[slide 55 says 8.6]")
    print(f"\n  1. Average delay in queue:")
    print(f"       delays D_i = {[round(d, 1) for d in res['delays']]}")
    print(f"       d_hat(6) = ({' + '.join(str(round(d,1)) for d in res['delays'])})"
          f" / 6")
    print(f"                = {res['total_delay']:.1f} / 6 = "
          f"{res['avg_delay']:.4f}   [slide 56 says 0.95]")
    print(f"\n  2. Time-average number in queue:")
    print(f"       integral of Q(t) dt = {res['area_q']:.1f}   [slide 41 says 9.9]")
    print(f"       q_hat(6) = {res['area_q']:.1f} / {res['T']:.1f} = "
          f"{res['avg_queue']:.4f}   [slide 41 says 1.15]")
    print(f"\n  3. Server utilization:")
    print(f"       integral of B(t) dt = {res['area_b']:.1f}   [slide 43 says 7.7]")
    print(f"       u_hat(6) = {res['area_b']:.1f} / {res['T']:.1f} = "
          f"{res['utilization']:.4f}   [slide 43 says 0.90]")

    ok = (abs(res['T'] - 8.6) < 1e-9 and
          abs(res['avg_delay'] - 0.95) < 1e-9 and
          abs(res['area_q'] - 9.9) < 1e-9 and
          abs(res['area_b'] - 7.7) < 1e-9)
    print(f"\n  ALL FOUR SLIDE VALUES REPRODUCED EXACTLY: {ok}")
    print(f"  (T = 8.6, d_hat = 0.95, area Q = 9.9, area B = 7.7)")
    return ok


# ===========================================================================
# D. VALIDATION AGAINST M/M/1 THEORY
# ===========================================================================

def mm1_theory(lam, mu):
    """Exact steady-state results for M/M/1 (Poisson arrivals rate lam,
    exponential service rate mu), valid when rho = lam/mu < 1.

        rho = lam / mu                utilization
        L   = rho / (1 - rho)         expected number in SYSTEM
        Lq  = rho^2 / (1 - rho)       expected number in QUEUE
        W   = 1 / (mu - lam)          expected time in system
        Wq  = rho / (mu - lam)        expected DELAY in queue
    """
    rho = lam / mu
    if rho >= 1:
        return None
    return {
        "rho": rho,
        "L": rho / (1 - rho),
        "Lq": rho ** 2 / (1 - rho),
        "W": 1 / (mu - lam),
        "Wq": rho / (mu - lam),
    }


def validate_against_theory():
    print("\n\n" + "=" * 78)
    print("D. VALIDATION: does the simulation match M/M/1 theory?")
    print("=" * 78)
    print("""
  This is step 6 of a simulation study (slide 68) and the input-output
  validation of slide 88: run the model under conditions where the true
  answer is known, and check that it agrees.

  Setup: exponential interarrivals with mean 1/lambda, exponential service
  with mean 1/mu. That makes the model an M/M/1 queue, which has exact
  closed-form answers.""")

    print(f"\n  {'lambda':>7} {'mu':>5} {'rho':>6} | {'Wq sim':>9} {'Wq exact':>9} "
          f"| {'Lq sim':>8} {'Lq exact':>9} | {'u sim':>7} {'u exact':>8}")
    print("  " + "-" * 82)

    for lam, mu in [(0.5, 1.0), (0.7, 1.0), (0.9, 1.0), (0.95, 1.0)]:
        rng = LCG().reset()
        sim = SingleServerQueue(
            interarrival_fn=lambda: rng.expo(1.0 / lam),
            service_fn=lambda: rng.expo(1.0 / mu),
            n_customers=400_000)
        res = sim.run()
        th = mm1_theory(lam, mu)
        print(f"  {lam:>7.2f} {mu:>5.2f} {th['rho']:>6.2f} | "
              f"{res['avg_delay']:>9.4f} {th['Wq']:>9.4f} | "
              f"{res['avg_queue']:>8.4f} {th['Lq']:>9.4f} | "
              f"{res['utilization']:>7.4f} {th['rho']:>8.4f}")

    print(f"""
  The simulated values track the theory closely at every load. Agreement
  degrades slightly as rho -> 1 because the queue becomes very variable and
  needs far longer runs to settle -- itself a useful lesson about how long a
  simulation must run near saturation.

  Face validity check (slide 86): as lambda rises toward mu, the delay and
  queue length rise steeply and utilization approaches 1. That is exactly
  what intuition predicts, so the model passes.""")


# ===========================================================================
# E. REPLICATIONS AND CONFIDENCE INTERVALS
# ===========================================================================

def replications(lam=0.8, mu=1.0, n_reps=10, n_cust=50_000):
    print("\n\n" + "=" * 78)
    print("E. REPLICATIONS AND CONFIDENCE INTERVALS (slides 69-70)")
    print("=" * 78)
    print(f"""
  "You cannot make decisions based on a single number from a single run."
  Simulation output is random, so we run the model {n_reps} times with
  DIFFERENT random number streams and build a confidence interval.

  lambda = {lam}, mu = {mu}, {n_cust:,} customers per replication.
""")
    ests = []
    print(f"  {'rep':>4} {'seed':>10} {'avg delay':>11} {'avg queue':>11} "
          f"{'utilization':>12}")
    print("  " + "-" * 52)
    for r in range(n_reps):
        seed = 12345 + r * 7919          # a different stream per replication
        rng = LCG(seed=seed).reset()
        sim = SingleServerQueue(
            interarrival_fn=lambda: rng.expo(1.0 / lam),
            service_fn=lambda: rng.expo(1.0 / mu),
            n_customers=n_cust)
        res = sim.run()
        ests.append(res['avg_delay'])
        print(f"  {r+1:>4} {seed:>10} {res['avg_delay']:>11.4f} "
              f"{res['avg_queue']:>11.4f} {res['utilization']:>12.4f}")

    n = len(ests)
    mean = sum(ests) / n
    var = sum((e - mean) ** 2 for e in ests) / (n - 1)
    se = math.sqrt(var / n)
    try:
        from scipy.stats import t
        tc = float(t.ppf(0.975, n - 1))
    except ImportError:
        tc = 2.262
    lo, hi = mean - tc * se, mean + tc * se
    th = mm1_theory(lam, mu)

    print("  " + "-" * 52)
    print(f"\n  Point estimate of the average delay: {mean:.4f}")
    print(f"  Sample sd across replications:       {math.sqrt(var):.4f}")
    print(f"  Standard error:                      {se:.4f}")
    print(f"  95% confidence interval: [{lo:.4f}, {hi:.4f}]")
    print(f"  (t_(0.025,{n-1}) = {tc:.3f})")
    print(f"\n  Exact M/M/1 value Wq = {th['Wq']:.4f}")
    print(f"  Inside the interval: {lo <= th['Wq'] <= hi}")
    print(f"""
  Report the interval, never the bare mean. The width tells the decision
  maker how much precision the study actually bought.""")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ok = verify_against_hand_simulation()

    # --- C: stochastic run -----------------------------------------------
    print("\n\n" + "=" * 78)
    print("C. STOCHASTIC RUN (random interarrival and service times)")
    print("=" * 78)
    lam, mu = 0.8, 1.0
    print(f"\n  Interarrival ~ Exponential(mean {1/lam:.3f}), "
          f"Service ~ Exponential(mean {1/mu:.3f})")
    print(f"  Random numbers come from our LCG (a=16807, m=2^31-1), "
          f"not `random`.")
    print(f"  Exponential variates by inverse transform: X = -mean*ln(1-U)\n")

    rng = LCG().reset()
    sim = SingleServerQueue(
        interarrival_fn=lambda: rng.expo(1.0 / lam),
        service_fn=lambda: rng.expo(1.0 / mu),
        n_customers=10, trace=True)
    res = sim.run()
    sim.print_trace()
    print(f"\n  d_hat(10) = {res['avg_delay']:.4f}, "
          f"q_hat = {res['avg_queue']:.4f}, "
          f"u_hat = {res['utilization']:.4f}, T = {res['T']:.4f}")

    print(f"\n  Same model, longer run (100,000 customers):")
    rng = LCG().reset()
    sim = SingleServerQueue(
        interarrival_fn=lambda: rng.expo(1.0 / lam),
        service_fn=lambda: rng.expo(1.0 / mu),
        n_customers=100_000)
    res = sim.run()
    th = mm1_theory(lam, mu)
    print(f"    average delay in queue      d_hat = {res['avg_delay']:.4f}  "
          f"(theory {th['Wq']:.4f})")
    print(f"    time-average number in queue q_hat = {res['avg_queue']:.4f}  "
          f"(theory {th['Lq']:.4f})")
    print(f"    server utilization           u_hat = {res['utilization']:.4f}  "
          f"(theory {th['rho']:.4f})")

    validate_against_theory()
    replications()

    # --- ordering warning -------------------------------------------------
    print("\n\n" + "=" * 78)
    print("THE ORDER-OF-UPDATES RULE (slides 44 and 58)")
    print("=" * 78)
    print("""
  Inside every event routine, do these in this order:

    1. Update the AREA accumulators, using the OLD Q(t) and B(t)
       and width = clock - time_last_event
    2. Update time_last_event = clock
    3. Update the STATE variables (server status, queue length, arrival list)
    4. Update the EVENT LIST (schedule new arrivals/departures)

  The three classic bugs:
    * updating Q(t) before computing the area  -> wrong rectangle HEIGHT
    * updating time_last_event too early       -> wrong rectangle WIDTH
    * forgetting to set the departure time to INFINITY when the queue
      empties -> the timing routine "departs" a customer who does not exist

  In this file, `update_time_avg_stats()` is deliberately the FIRST line of
  both arrive() and depart(), which is why the hand-simulation numbers come
  out exactly right.""")

    print(f"\n{'='*78}")
    print(f"Hand-simulation verification passed: {ok}")


if __name__ == "__main__":
    main()
