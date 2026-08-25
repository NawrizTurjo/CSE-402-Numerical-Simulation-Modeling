# Practice Questions — CSE 402 Online-3

Seven new practice questions, one per syllabus pillar, each sized to be
answerable in about 30 minutes using the templates in
[`../Code/`](../Code/) — the same scale as the two questions that have
already appeared (A1, C1 in [`../Res/questions.txt`](../Res/questions.txt)).
None of these are official/leaked questions; they're written to be
**representative of the syllabus and appeared-question style**, not
tricky or obscure. Every numeric answer below was actually computed by
running the referenced code, not guessed.

Each question links a runnable solution file under
[`../Code/solutions/practice_generated/`](../Code/solutions/practice_generated/)
— try the question yourself first, then check your code and your
numbers against that file.

| # | Title | Syllabus pillar | Difficulty | Solution file |
|---|---|---|---|---|
| [R1](#r1-lcg-investigation) | LCG Investigation | Random number generation | Moderate | `r1_lcg_investigation.py` |
| [R2](#r2-randu--the-generator-that-fooled-everyone) | RANDU — the generator that fooled everyone | RNG + independence testing | Moderate | `r2_randu_independence.py` |
| [S1](#s1-coffee-shop-queue) | Coffee Shop Queue | Discrete-event simulation | Easy–Moderate | `s1_coffee_shop_queue.py` |
| [V1](#v1-verification-by-trace--bug-hunt) | Verification by Trace — Bug Hunt | Model V&V | Easy–Moderate | `v1_trace_bug_hunt.py` |
| [M1](#m1-estimating-an-integral-with-no-elementary-antiderivative) | Estimating an integral with no elementary antiderivative | Monte Carlo integration | Easy | `m1_gaussian_integral.py` |
| [M2](#m2-the-birthday-paradox) | The Birthday Paradox | Monte Carlo probability | Easy | `m2_birthday_paradox.py` |
| [MH1](#mh1-sampling-a-gamma-shaped-density) | Sampling a Gamma-shaped density | Metropolis-Hastings | Moderate | `mh1_gamma_sampling.py` |

---

## R1: LCG Investigation

**Syllabus:** Random number generation. Same shape as the C1 exam
question, but for the Linear Congruential Generator instead of
Middle-Square — read this if you want more LCG reps beyond the
Middle-Square question that already appeared.

Implement `X_{i+1} = (a*X_i + c) mod m`, `R_i = X_i/m`. Do not use
Python's `random` module for generation.

**Task 1 — Generate:** Using seed `X0=7`, `a=5`, `c=3`, `m=16`, generate
the first 20 values `X1..X20`.

**Task 2 — Investigate parameters:**
(a) Theory says a mixed LCG (`m=2^b`, `c≠0`) reaches its maximum
possible period `m` when `gcd(c,m)=1` and `(a-1) mod 4 = 0`. Check
whether the Task 1 parameters satisfy this, and confirm by directly
measuring the period (find the first repeated value).
(b) Find a degenerate parameter set on the same `m=16` that collapses
to a period of 1 immediately, and explain algebraically why.

**Task 3 — Chi-Square test:** For `N=1000, bins=10, alpha=0.05`, run
the Chi-Square uniformity test on: (i) the Task 1 (full-period)
generator, (ii) your degenerate generator from Task 2(b), and (iii) the
well-known "ANSI C `rand()`" generator `a=1103515245, c=12345, m=2^31`
(seed=1). Report Chi², p-value, and decision for each.

**Task 4 — Reflect:** Does achieving the *theoretical maximum period*
for a given `m` guarantee the generator is statistically good? Use your
Task 3 results to justify your answer.

<details>
<summary><b>Answer</b></summary>

Run: `python -m solutions.practice_generated.r1_lcg_investigation`
(from inside `Code/`).

**Task 1.** First 20 values:
`[6, 1, 8, 11, 10, 5, 12, 15, 14, 9, 0, 3, 2, 13, 4, 7, 6, 1, 8, 11]`
(notice it's already repeating by X17 — period 16, confirmed in Task 2).

**Task 2.**
(a) `m=16=2^4` is a power of 2, `c=3≠0` → mixed case. `gcd(3,16)=1` ✓ and
`(5-1)%4=0` ✓ → theory predicts full period 16. Measuring it directly
(track values until one repeats) confirms **period = 16**, matching theory.
(b) `a=1, c=0, m=16`: the recurrence becomes `X_{i+1} = (1*X_i + 0) mod 16
= X_i` — the identity function. Whatever the seed is, it "generates"
itself forever. With seed 7: first repeated value = 7, repeats at
iteration 1, cycle length 1.

**Task 3.**

| Generator | Chi² | p-value | Decision |
|---|---|---|---|
| Full-period (a=5,c=3,m=16) | 93.8000 | 2.81e-16 | Reject H0 |
| Degenerate (a=1,c=0,m=16) | 9000.0000 | ~0 | Reject H0 |
| ANSI-C rand (a=1103515245,c=12345,m=2^31) | 5.4600 | 0.7925 | Do not reject H0 |

**Task 4.** No. `a=5,c=3,m=16` genuinely *achieves* its theoretical
maximum period of 16 (Task 2 confirms this) — but a period of only 16
means just 16 distinct values get repeated ~63 times each to fill 1000
draws, which is nowhere close to i.i.d. Uniform(0,1); it fails
Chi-Square badly. The ANSI-C generator's `m=2^31` is astronomically
larger, so 1000 draws barely sample its state space and it passes
comfortably — regardless of exactly where it sits relative to its own
theoretical max. **Lesson:** period theory tells you the best case *for
a given m*; you separately need `m` itself to be large enough for how
many values you actually plan to draw.
</details>

---

## R2: RANDU — the generator that fooled everyone

**Syllabus:** Random number generation + independence testing
(autocorrelation). This question specifically targets the gap C1 didn't
cover: C1 only tested *uniformity* (Chi-Square); this one tests
*independence* too, and shows why both matter.

IBM's RANDU generator uses `a=65539, c=0, m=2^31`. It was the default
generator on IBM mainframes through the 1960s-70s.

**Task 1 — Generate:** seed=1, generate the first 20 values `X1..X20`.

**Task 2 — Chi-Square test:** `N=1000, bins=10, alpha=0.05`. Report
Chi², p-value, decision.

**Task 3 — Autocorrelation test:** Run the autocorrelation test at
`lag=1` on the same 1000 values. Report `rho_hat`, `Z0`, decision.

**Task 4 — The hidden defect:** RANDU has a special property: every
three consecutive raw integer outputs `X_i, X_{i+1}, X_{i+2}` satisfy
`9*X_i - 6*X_{i+1} + X_{i+2} ≡ 0 (mod m)`. Verify this holds exactly for
several consecutive triples from your Task 1 sequence.

**Task 5 — Reflect:** Given your Task 2-4 results, what does this tell
you about what "passing your tests" actually proves?

<details>
<summary><b>Answer</b></summary>

Run: `python -m solutions.practice_generated.r2_randu_independence`

**Task 1.** `[65539, 393225, 1769499, 7077969, 26542323, 95552217,
334432395, 1146624417, 1722371299, 14608041, 1766175739, 1875647473,
1800754131, 366148473, 1022489195, 692115265, 1392739779, 2127401289,
229749723, 1559239569]`

**Task 2.** Chi²=14.2000, p=0.1154 → **Do not reject H0** (looks uniform).

**Task 3.** rho_hat=0.0095, Z0=1.0007 → **Do not reject H0** (looks
independent at lag 1).

**Task 4.** Checked across 8 consecutive triples — the residual
`(9*X_i - 6*X_{i+1} + X_{i+2}) mod m` is **exactly 0 every time**.

**Task 5.** RANDU passes *both* tests you ran — by every check applied
so far it looks like a perfectly fine generator. But Task 4 proves every
consecutive triple lies on one of just 15 parallel 2-D planes instead of
filling 3-D space, a severe defect that neither the 1-D Chi-Square test
nor the pairwise (lag-1) autocorrelation test can detect — it only shows
up once you look at 3+ values together. **Passing a standard battery of
1-D/pairwise tests is necessary but not sufficient**; this exact failure
mode caused real, silently-wrong published simulation results before it
was understood, which is why RANDU is a standard textbook cautionary
tale (also referenced in `Code/rng/lcg.py`'s docstring).
</details>

---

## S1: Coffee Shop Queue

**Syllabus:** Discrete-event simulation / single-server queue —
straight application of the same mechanics as the appeared C1-style
questions and the slide's worked SSQ example, with a new numeric
scenario plus multi-server and balking variations layered on.

A coffee shop is staffed by one barista. Interarrival times (minutes)
for the first 10 customers, and service times (minutes) for the first 8
customers, are:

```
interarrival = [1.5, 0.8, 2.1, 0.3, 1.2, 0.9, 1.7, 0.4, 2.3, 0.6]
service      = [2.5, 1.1, 0.9, 1.8, 0.7, 2.0, 1.3, 0.85]
```

**Task 1:** Simulate with 1 barista until 8 customer delays have been
observed. Report average delay, average number in queue, and server
utilization.

**Task 2:** The owner adds a second barista (2 identical servers,
shared queue). Re-run the same simulation (still stop after 8 delays).
How do the three performance measures change, and why?

**Task 3:** Instead, suppose the shop has room for only 2 people to
physically wait (a customer who arrives when 2 are already waiting
leaves immediately — "balks"). Using 1 barista, re-run and report how
many customers were turned away.

**Task 4:** Produce a trace table (clock, event, queue length, server
status) for the first 6 events of the 1-barista, no-capacity-limit run,
and verify the first two arrivals and the first departure by hand.

<details>
<summary><b>Answer</b></summary>

Run: `python -m solutions.practice_generated.s1_coffee_shop_queue`

**Task 1.** `avg_delay=1.5250  avg_queue=1.0847  utilization=0.8729  T=11.80`
Delays per customer: `[0.0, 1.7, 0.7, 1.3, 1.9, 1.7, 2.0, 2.9]`.

**Task 2.** `avg_delay=0.0000  avg_queue=0.0000  utilization=0.5281  T=8.90`.
Every customer is served immediately — two servers comfortably absorb
this arrival pattern. Utilization drops to roughly half (0.87 → 0.53)
because the same total service workload is now spread across two
servers, so each one sits idle more often.

**Task 3.** `avg_delay=1.5250  avg_queue=1.0847  utilization=0.8729
customers_turned_away=1`. Identical delay/queue/utilization numbers to
Task 1 for the customers who *do* get served (the queue never happened
to exceed capacity 2 except once), but exactly 1 customer balks and is
lost — a detail Task 1's numbers can't show you on their own.

**Task 4.** First 6 events:
```
t=1.50  Arrival C1 (starts service, D=0)          Q=0  busy=1
t=2.30  Arrival C2 (joins queue)                  Q=1  busy=1
t=4.00  Departure C1 (C2 starts service, D=1.70)  Q=0  busy=1
t=4.40  Arrival C3 (joins queue)                  Q=1  busy=1
t=4.70  Arrival C4 (joins queue)                  Q=2  busy=1
t=5.10  Departure C2 (C3 starts service, D=0.70)  Q=1  busy=1
```
By hand: C1 arrives at t=1.5 (0+1.5), server idle → starts service
immediately, delay 0, departs at 1.5+2.5=4.0. C2 arrives at
1.5+0.8=2.3, server busy → joins queue. C1 departs at t=4.0; C2 (who
arrived at 2.3) starts service, delay = 4.0-2.3 = **1.70** ✓ matches.
</details>

---

## V1: Verification by Trace — Bug Hunt

**Syllabus:** Model verification & validation — this is a coding
version of the slide's own "verification by trace" worked example
(§8 of [`../Slides/5_simu_des_ssqs.md`](../Slides/5_simu_des_ssqs.md)):
you're handed a simulation that runs without crashing and produces
*plausible-looking* numbers, and asked to find out whether it's
actually correct.

Below is a single-server queue simulation. It runs, and its `avg_delay`
matches the known-correct answer for the slide's worked example
(`interarrival=[0.4,1.2,0.5,1.7,0.2,1.6,0.2,1.4,1.9]`,
`service=[2.0,0.7,0.2,1.1,3.7,0.6]`, stop after 6 delays → expected
`avg_delay=0.9500, avg_queue=1.1512, utilization=0.8953`). It has
exactly one bug.

```python
while events:
    clock, _, etype, cust = heapq.heappop(events)

    # (A) state update happens here
    if etype == "A":
        ...update busy/queue...
    else:
        ...update busy/queue...

    # (B) area accumulation happens here, AFTER state update
    dt = clock - last_event_time
    area_Q += len(queue) * dt
    area_B += (1.0 if busy else 0.0) * dt
    last_event_time = clock
```

(This is trimmed pseudocode to show the shape of the bug at a glance —
the full runnable buggy version is in the solution file.)

**Task 1:** Run this version (or the full version in the solution file)
against the known-correct expected output above. Which of the three
performance measures come out wrong?

**Task 2:** Identify the bug and explain, in terms of what the area
integral is supposed to represent, why swapping blocks (A) and (B)
produces wrong numbers.

**Task 3:** Fix it, and re-verify against the expected output.

**Task 4:** Which V&V technique from the slides did Task 1-3 exercise,
and why does `avg_delay` come out right even in the buggy version (i.e.
why doesn't every performance measure break)?

<details>
<summary><b>Answer</b></summary>

Full buggy + fixed code: `Code/solutions/practice_generated/v1_trace_bug_hunt.py`.
Run: `python -m solutions.practice_generated.v1_trace_bug_hunt`

**Task 1.** `avg_delay` matches (0.9500), but `avg_queue` and
`utilization` do not:

| | avg_delay | avg_queue | utilization |
|---|---|---|---|
| Buggy | 0.9500 | **1.2558** | **0.9767** |
| Correct | 0.9500 | 1.1512 | 0.8953 |

**Task 2.** `area_Q += len(queue)*dt` is meant to add the rectangle
"queue length **held during** the interval `[last_event_time, clock]`,
times its width" — i.e. the state that was TRUE *during* that interval,
which is the OLD state, not the new one. Block (A) changes `queue`/`busy`
for the event that just happened at `clock`; if area accumulation (B)
runs afterward, it uses the state *after* the event to describe the
interval *before* the event — attributing time to the wrong state and
inflating the areas (both wrong values above are too high, since e.g. an
arrival that increments the queue gets that higher queue length credited
retroactively to the idle interval before it arrived).

**Task 3.** Swap the two blocks (accumulate areas first, then update
state) — this reproduces the correct `0.9500 / 1.1512 / 0.8953`, matching
`simulation/single_server_queue.py`'s reference implementation exactly.

**Task 4.** This is **verification by trace**: comparing a program's
output/trace against independently known-correct values (by hand or
from a trusted reference) to catch implementation bugs, as opposed to
*validation* (which asks whether the model correctly represents the real
system in the first place — a different question, and not what's being
tested here). `avg_delay` survives because the delay calculation
(`clock - arrival_time` when service starts) never touches `area_Q` or
`area_B` — it's a completely separate accumulator, so a bug isolated to
the area bookkeeping only breaks the two measures that are actually
built from those areas.
</details>

---

## M1: Estimating an integral with no elementary antiderivative

**Syllabus:** Monte Carlo integration. Deliberately similar in spirit to
the two problems you said you'd already practiced (integration, π) —
one more clean integration example plus the convergence-rate check the
slides walk through in their own `sin(x)` example
([`../Slides/6_rng_mcs.md` §9](../Slides/6_rng_mcs.md)).

Estimate `∫₀¹ e^(−x²) dx` (this shows up constantly in probability —
it's proportional to a slice of the standard normal's density — and has
no elementary closed-form antiderivative, which is exactly the situation
Monte Carlo integration is for).

**Task 1:** Implement the sample-mean Monte Carlo estimator for this
integral (`X ~ Uniform(0,1)`, `Y = 1 * f(X)`, estimate = mean(Y)).

**Task 2:** Run it at `n = 100, 1000, 10000, 100000` and report the
estimate and error at each `n` (exact value, for checking only:
`0.746824`, via the error function). Does the error shrink roughly like
`1/sqrt(n)` (quadrupling `n` should roughly halve the error)?

<details>
<summary><b>Answer</b></summary>

Run: `python -m solutions.practice_generated.m1_gaussian_integral`

| n | estimate | error | std_error |
|---|---|---|---|
| 100 | 0.73813 | 0.00869 | 0.01967 |
| 1,000 | 0.73681 | 0.01001 | 0.00640 |
| 10,000 | 0.74779 | 0.00096 | 0.00202 |
| 100,000 | 0.74670 | 0.00013 | 0.00064 |

Roughly yes — `std_error` (the theoretical error scale) shrinks by about
`1/sqrt(10)≈0.316` each time `n` grows 10x, matching `0.01967 → 0.00640
→ 0.00202 → 0.00064` almost exactly. The realized `error` column is
noisier (it's one random run, not the expected value of the error) but
trends the same way. This 1/√n rate — independent of dimension — is the
whole reason Monte Carlo is used for integrals a grid method (Simpson's
rule, etc.) can't reach.
</details>

---

## M2: The Birthday Paradox

**Syllabus:** Monte Carlo probability estimation. A classic,
well-known-answer sanity-check problem — good for confirming your
estimator code is right, and a different problem *shape* (combinatorial,
not geometric) from the pi/needle/sphere examples you've already done.

In a room of 23 randomly chosen people, what is the probability that at
least two share a birthday (ignore Feb 29, assume all 365 days equally
likely)?

**Task 1:** Write a `trial_fn` that simulates one room of 23 people and
returns 1 if any two share a birthday, else 0.

**Task 2:** Estimate the probability over 100,000 trials, with a 95%
confidence interval. (Known closed-form answer, for checking:
≈ 0.5073 — notably higher than most people's intuition!)

**Task 3:** Repeat for room sizes 5, 10, 23, 40, 60 and describe the trend.

<details>
<summary><b>Answer</b></summary>

Run: `python -m solutions.practice_generated.m2_birthday_paradox`

**Task 1-2.** `estimate=0.5078`, `CI95=(0.5047, 0.5109)` — comfortably
contains the true value 0.5073.

**Task 3.**

| people | P(shared) |
|---|---|
| 5 | 0.0273 |
| 10 | 0.1167 |
| 23 | 0.5065 |
| 40 | 0.8915 |
| 60 | 0.9935 |

The probability rises much faster than intuition suggests — by 23
people it already crosses 50%, and by 60 it's over 99%. (The reason:
it's driven by the number of *pairs* of people, which grows like
`n²`, not by `n` itself.)
</details>

---

## MH1: Sampling a Gamma-shaped density

**Syllabus:** Monte Carlo methods — Metropolis-Hastings. No appeared
question has covered this yet even though it's explicitly named in the
syllabus, so this is worth extra rep time.

Target distribution (unnormalized): `p(x) ∝ x² · e^(−x)` for `x > 0`,
and `p(x) = 0` for `x ≤ 0`. This is (up to the normalizing constant) a
`Gamma(shape=3, rate=1)` distribution, which has a known true mean of 3
and true variance of 3 — handy for checking your sampler without needing
`scipy.stats`.

**Task 1:** Write `log_target(x)` for this density (drop the unknown
normalizing constant — MH only needs the ratio `p(x')/p(x)`, so additive
constants in the log cancel). Handle `x ≤ 0` so the chain never accepts
a move into the disallowed region.

**Task 2:** Run Metropolis-Hastings for 5000 iterations with a symmetric
Gaussian proposal (`proposal_std=1.5`), starting at `x0=1.0`, discard the
first 500 as burn-in, and report the acceptance rate, sample mean, and
sample variance against the known true values (3 and 3).

**Task 3:** Repeat with `proposal_std = 0.5` and `5.0`. What happens to
the acceptance rate at each extreme, and why?

<details>
<summary><b>Answer</b></summary>

Run: `python -m solutions.practice_generated.mh1_gamma_sampling`

**Task 1.**
```python
def log_target(x):
    if x <= 0:
        return float("-inf")
    return 2 * math.log(x) - x   # log(x^2 * e^-x)
```
Returning `-inf` for `x<=0` means `log_alpha = log_target(x') - log_target(x)`
is `-inf` whenever a proposal lands at or below 0, so
`math.log(random.random()) < log_alpha` is never true there — the chain
naturally never accepts a move into the disallowed region, with no
special-case branching needed in the MH loop itself.

**Task 2.** `acceptance_rate=0.6922`, `sample mean=2.8627` (true 3.0),
`sample variance=2.5370` (true 3.0) — both close, with the gap explained
by a finite chain length and the burn-in/mixing tradeoff (more iterations
would tighten this further).

**Task 3.**

| proposal_std | acceptance | mean | variance |
|---|---|---|---|
| 0.5 | 0.8890 | 2.7945 | 2.2004 |
| 1.5 | 0.6922 | 2.8627 | 2.5370 |
| 5.0 | 0.3430 | 3.0655 | 3.6802 |

Small steps (0.5) get accepted almost 9 times out of 10, but each step
barely moves — the chain explores the distribution slowly (and, not
shown in the summary stats but visible in a trace plot, its samples are
strongly autocorrelated). Large steps (5.0) get accepted only about a
third of the time, because most large proposed jumps land somewhere the
target density is much lower than at the current point and get rejected
— but when they ARE accepted, they move a lot, exploring faster per
accepted step. Moderate steps (here ~1.5) balance the two effects; tuning
`proposal_std` for an acceptance rate roughly in the 20-50% range is a
standard rule of thumb.
</details>
