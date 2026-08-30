# Study Guide — Online-3 (CSE 402 Numerical Methods)

30-minute live coding exam. Topics: discrete-event simulation, steps in
a simulation study, model V&V, random number generation, Monte Carlo
methods incl. Metropolis-Hastings ([Res/syllabus.txt](Res/syllabus.txt)).

This guide is the map. For each syllabus topic: where the theory is
(slide section), where the tested code is (`Code/` module), and where a
worked answer is (`Code/solutions/`). Read a topic's slide section once,
run its `Code/` module once (`python -m ...`), then move on — don't
re-derive what's already tested and working.

---

## 0. Fifteen-minute crash review (do this first, day-of)

Run every module once so the numbers are fresh and you trust the code:

```bash
cd Code
python -m rng.lcg
python -m rng.middle_square
python -m testing.chi_square_test
python -m testing.ks_test
python -m testing.independence_test
python -m monte_carlo.core
python -m monte_carlo.integration
python -m monte_carlo.metropolis_hastings
python -m variate_generation.inverse_transform
python -m simulation.single_server_queue
```

Then skim [Code/README.md](Code/README.md)'s "Design principle" section
— it's the mental model for the whole exam: almost everything reduces to
either `monte_carlo/core.py`'s one estimator or
`simulation/single_server_queue.py`'s one engine.

---

## 1. Discrete-Event Simulation (DES) models

**Theory:** [Slides/5_simu_des_ssqs.md §1-3](Slides/5_simu_des_ssqs.md)
— what simulation is, when to use it, system/model taxonomy (entity,
attribute, event, state variable; discrete vs continuous; static/dynamic/
deterministic/stochastic), the simulation clock, next-event vs
fixed-increment time advance, the event list, and the full flow-of-control
flowchart (init → main ↔ timing routine → event routine → report generator).

**Key things to be able to say cold:**
- Discrete-event: state changes only at discrete points in time (events).
  Fixed-increment: clock advances by a small constant Δt regardless of
  whether anything happened — DES is next-event, not fixed-increment.
- Entity vs attribute vs state variable vs event, with an SSQ example
  (entity=customer, attribute=arrival time, state variable=Q(t)/B(t),
  event=arrival/departure).

**Code:** none needed yet — this section is definitions/concepts, not
algorithms. It sets up §2 below.

---

## 2. Single-Server Queue (SSQ) simulation — the core coding topic

**Theory:** [Slides/5_simu_des_ssqs.md §4-6](Slides/5_simu_des_ssqs.md)
— components of a queueing system, arrival/departure flowcharts, the
three performance measures and their formulas, the area-accumulation
algorithm (**memorize the update-ordering rule**: accumulate
`area_Q += Q(t)*dt` and `area_B += B(t)*dt` using the OLD state BEFORE
you change state for the new event — this is the #1 bug source), and a
fully worked 13-event, 6-customer hand-trace with exact numbers.

**Formulas:**
```
avg_delay    d(n) = (sum of delays D_1..D_n) / n
avg_queue    q(n) = area_Q / T          (time-average of Q(t))
utilization  u(n) = area_B / T          (T = total simulated time)
```

**Code:** [Code/simulation/single_server_queue.py](Code/simulation/single_server_queue.py)
— ONE function `simulate_ssq(...)` covers every variant:
- stop after N delays observed: `num_delays=6`
- stop at a fixed time: `t_max=8.6`
- run until the system fully empties: leave both `None`
- c parallel servers: `num_servers=2`
- finite waiting room / balking: `capacity=k`
- printed trace table: `trace=True`
- given a finished trace table instead of raw data? Use
  `compute_performance_from_trace(...)` in the same file — no simulation
  loop needed, just the area arithmetic.

Verify against the slide's numbers: `python -m simulation.single_server_queue`
reproduces `avg_delay=0.9500, avg_queue=1.1512, utilization=0.8953, T=8.6`
exactly, matching the slide's hand-trace. To *see* it instead of just
reading numbers, `python -m viz.des_plots` draws the Q(t)/B(t) step plot
straight from a `simulate_ssq(..., trace=True)` trace — worth running
once so the area-under-the-curve idea clicks visually.

**Practice:** trace through the printed table by hand once with pen and
paper against the slide's 13-event trace, so you can produce/verify a
trace table even without running code if the exam is on paper for part
of it.

---

## 3. Steps in a Simulation Study

**Theory:** [Slides/5_simu_des_ssqs.md §7](Slides/5_simu_des_ssqs.md) —
the 10-step flowchart (formulate problem → collect data/build model →
assumptions valid? → build+verify program → pilot runs → programmed model
valid? → design experiments → production runs → analyze output → document/
present/use). Know the two validation checkpoints (after step 2, after
step 5) and roughly what happens if each check fails (loop back).

This is a "explain/list the steps" topic, not a coding topic — no `Code/`
module. If asked to code something from this section, it'll actually be
testing §2 or §4 underneath a "simulation study" framing.

---

## 4. Model Validation and Verification (V&V)

**Theory:** [Slides/5_simu_des_ssqs.md §8](Slides/5_simu_des_ssqs.md) —
verification (is the program correct?) vs validation (is the model an
accurate representation of the real system?), the V&V triangle
(reality ↔ conceptual model ↔ computerized model), verification techniques
(structured walkthrough, trace, error checking), **verification by trace**
worked example (a bug in an SSQ implementation caught by comparing a hand
trace against the program's output), and the Naylor-Finger 3-step approach
to validation (face validity → validating assumptions → validating
input-output transformations).

**Practical link to §2:** "verification by trace" is exactly
`simulate_ssq(..., trace=True)` — if a question asks you to verify your
program is correct, run it with `trace=True` and compare row-by-row
against a hand-worked trace table.

---

## 5. Random Number Generation

**Theory:** [Slides/6_rng_mcs.md §1-4](Slides/6_rng_mcs.md) — why
pseudo-randomness is fine (deterministic but statistically indistinguishable
from random, and REPRODUCIBLE via the seed — a feature, not a bug), the two
properties a good RNG needs (uniformity + independence), the Linear
Congruential Method's recurrence and period theory (three max-period cases
by modulus type — see below), and why powers of 2 are attractive moduli
(fast mod via bitmask) but risk shorter periods / low-order bit weaknesses.

**LCG recurrence:** `X_{i+1} = (a*X_i + c) mod m`, `R_i = X_i / m`.

**Max-period cases** (also in [Code/rng/lcg.py](Code/rng/lcg.py)'s
`check_max_period_conditions`):
| m | c | Condition for max period | Max period |
|---|---|---|---|
| `2^b` | ≠0 (mixed) | `gcd(c,m)=1` and `(a-1)%4==0` | `m` |
| `2^b` | =0 (multiplicative) | seed odd and `a%8 ∈ {3,5}` | `m/4` |
| prime | =0 (multiplicative) | `a` is a primitive root mod `m` | `m-1` |

**C1's method — Middle-Square:** not in these slide decks (see AGENTS.md
scope note) but fully covered in [Code/rng/middle_square.py](Code/rng/middle_square.py).
Square the seed, zero-pad to 8 digits, take the middle 4 digits as the
next value. **Known weakness:** degenerates into short cycles — even
"normal-looking" seeds like 5731 and 6239 collapse into a 4-value cycle
well within 1000 iterations (see `Code/solutions/c1_middle_square_investigation.py`'s
Task 4 answer for the exact iteration numbers — a genuinely useful,
verified talking point if asked to "reflect" on RNG quality).
`middle_square()` takes an optional `digits=` (default 4, matching C1
exactly) in case the exam gives a seed of a different width — same
function, just pass `digits=6` etc. If a question asks you to *fix* the
weakness rather than just diagnose it: `middle_square_weyl()` in the same
file mixes in a full-period Weyl sequence (`w += odd_increment mod
10^digits`) and verifiably repairs all three of 2500/5731/6239 (each
goes from "Reject H0" to comfortably passing Chi-Square).

**Code:**
- [Code/rng/lcg.py](Code/rng/lcg.py) — generator, period detection, max-period checker.
- [Code/rng/middle_square.py](Code/rng/middle_square.py) — generator (any digit width), cycle detection, Weyl-sequence repair.
- [Code/rng/common.py](Code/rng/common.py) — shared cycle-detection helper (study this to see WHY both generators above share one implementation).
- [Code/variate_generation/inverse_transform.py](Code/variate_generation/inverse_transform.py)
  — turning a Uniform(0,1) draw into Exponential/Weibull/Uniform(a,b)/
  discrete/triangular variates via `F^-1`.

**Testing an RNG's output** (uniformity AND independence — know the
difference, it's a common "reflect" question):
- [Code/testing/chi_square_test.py](Code/testing/chi_square_test.py) —
  uniformity via binned counts, `chi2 = sum((O_i-E_i)^2/E_i)`, df=bins-1.
  Theory: [Slides/6_rng_mcs.md §7](Slides/6_rng_mcs.md).
- [Code/testing/ks_test.py](Code/testing/ks_test.py) — uniformity via
  empirical-CDF-vs-diagonal, `D = max(D+, D-)`.
  Theory: [Slides/6_rng_mcs.md §6](Slides/6_rng_mcs.md).
- [Code/testing/independence_test.py](Code/testing/independence_test.py)
  — autocorrelation test (`rho_hat`, `sigma`, `Z0`) AND **two** runs
  tests: `runs_test` (above/below the mean 0.5) and `runs_up_down_test`
  (rising/falling vs the previous value). They check different kinds of
  dependence and can disagree on the same data — the module's `__main__`
  has a verified example where one rejects and the other doesn't. Use
  whichever the question specifically names; if it just says "runs
  test" without qualifying, above/below-mean (`runs_test`) is the more
  commonly taught default.
  Theory + full variance derivation: [Slides/6_rng_mcs.md §8](Slides/6_rng_mcs.md)
  (covers autocorrelation; the runs tests are supplementary — not in
  either slide deck, see AGENTS.md scope note — but standard textbook
  material collected from the same friends' code as Middle-Square/MH).
- **Multiple-testing caveat** ([Slides/6_rng_mcs.md §5](Slides/6_rng_mcs.md),
  "fishing" warning in §8): running many tests at alpha=0.05 means ~5% will
  falsely reject a good generator by chance — one failed test alone isn't
  damning, a *pattern* of failures is.

**C1's exact question:** already fully answered in
[Code/solutions/c1_middle_square_investigation.py](Code/solutions/c1_middle_square_investigation.py)
(all 4 tasks). Run it: `python -m solutions.c1_middle_square_investigation`.

**C2's exact question (Hidden Period Collapse & K-S Test):** already fully answered in
[Code/solutions/c2_hidden_period_collapse.py](Code/solutions/c2_hidden_period_collapse.py)
(all 3 tasks). Run it: `python -m solutions.c2_hidden_period_collapse`.

**Seeing a generator's flaws instead of just reading test statistics:**
`python -m viz.rng_plots` plots the sequence, histogram, and R_i-vs-R_(i+1)
scatter for both an LCG and Middle-Square — the Middle-Square plot makes
the short-cycle collapse from Task 4 immediately visible (the sequence
panel goes from noisy to a tight repeating band partway through).

---

## 6. Monte Carlo Methods

**Theory:** [Slides/6_rng_mcs.md §9-10](Slides/6_rng_mcs.md) — integration
as expectation (`integral f(x)dx = (b-a)*E[f(X)], X~U(a,b)`), the
sample-mean estimator, a fully worked `∫sin(x)dx` example by hand (n=5
and n=10), a convergence table showing error shrinking roughly like
`1/sqrt(n)`, and the classic pi-via-unit-square example.

**The one idea that unifies every Monte Carlo problem** (see
[Code/monte_carlo/core.py](Code/monte_carlo/core.py)'s module docstring
for the full argument): any Monte Carlo estimate is the average of `n`
independent trials of a random quantity `Y`. What changes between
problems is only what `Y` (the `trial_fn`) computes:

| Problem shape | `trial_fn` returns |
|---|---|
| Probability of an event | 1.0 if event happens this trial, else 0.0 |
| Expected value / payoff | the payoff of one trial |
| Integral `∫f(x)dx` over `[a,b]` | `(b-a) * f(X)`, `X ~ Uniform(a,b)` |
| Volume (hit-or-miss) | `box_volume` if point is inside the region, else 0.0 |

**Code:**
- [Code/monte_carlo/core.py](Code/monte_carlo/core.py) — `monte_carlo_estimate(trial_fn, n, seed=None)`, returns estimate/std_dev/std_error/95% CI. **Read this file's docstring fully — it's the single highest-leverage thing to understand for this exam.**
- [Code/monte_carlo/integration.py](Code/monte_carlo/integration.py) — named wrappers: `integrate_1d`, `estimate_pi_hit_or_miss`.
- [Code/monte_carlo/metropolis_hastings.py](Code/monte_carlo/metropolis_hastings.py) — MCMC sampler for when you need SAMPLES from a distribution you can only evaluate (not integrate/estimate a scalar). Not in either slide deck (see AGENTS.md scope note); algorithm docstring inside the file covers proposal/acceptance-ratio/burn-in from scratch.

**A1's exact question (Buffon's Needle):** already fully answered in
[Code/solutions/a1_buffons_needle.py](Code/solutions/a1_buffons_needle.py).
Run it: `python -m solutions.a1_buffons_needle`.

---

## 7. Practice problems (Res/Src-26/practice-problems.md)

All six solved in [Code/solutions/practice_src26/](Code/solutions/practice_src26/),
each just a `trial_fn` handed to the shared estimator (or, for #3, to
`integrate_1d`) — read them in order, they get slightly harder:

| # | Problem | File | Shape |
|---|---|---|---|
| 1 | Sphere volume in a cube | `p1_sphere_volume.py` | hit-or-miss volume, 3-D |
| 2 | Dice-game expected value | `p2_dice_game_ev.py` | expected value |
| 3 | Area under `sin(x)cos(x^2)` | `p3_weird_curve_area.py` | 1-D integration |
| 4 | Random walk `\|pos\|>15` | `p4_random_walk.py` | probability |
| 5 | Project completion `>11` days | `p5_project_completion.py` | probability |
| 6 | Buffon's Needle | `p6_buffons_needle.py` | probability (identical to A1 — literally reuses it) |

Run any of them: `python -m solutions.practice_src26.p1_sphere_volume`
(from inside `Code/`).

**Suggested practice drill:** cover the `trial_fn` body in each file,
re-derive it from the problem statement, then check against the file.
That's the actual skill the exam tests — recognizing the shape and
writing the one-trial function, not the surrounding estimator machinery
(which you'll have ready-made from `Code/`).

---

## Quick formula cheat-sheet

```
LCG:              X_{i+1} = (a*X_i + c) mod m,  R_i = X_i/m
Middle-Square:    X_{i+1} = middle `d` digits of (X_i)^2, zero-padded to 2d digits (d=4 by default)
Middle-Square+Weyl repair: w += odd_increment (mod 10^d);  X_{i+1} = (middle(X_i^2) + w) mod 10^d

Chi-Square stat:  chi2 = sum_i (O_i - E_i)^2 / E_i,   df = bins - 1,  E_i = N/bins
K-S stat:         D = max(D+, D-)
                  D+ = max_i( i/N - R_(i) ),  D- = max_i( R_(i) - (i-1)/N )
Autocorrelation:  rho_hat = (1/(M+1)) * sum(R_i * R_{i+l}) - 0.25
                  sigma = sqrt(13M+7) / (12(M+1)),   Z0 = rho_hat/sigma
Runs test (above/below mean):  E[runs] = 2*n1*n2/n + 1
                  Var[runs] = 2*n1*n2*(2*n1*n2-n) / (n^2*(n-1))
Runs test (up/down):  E[runs] = (2n-1)/3,   Var[runs] = (16n-29)/90

Inverse transform: Exponential: x = -mean*ln(1-r)
                    Uniform(a,b): x = a + (b-a)*r
                    Weibull(k,c): x = c*(-ln(1-r))^(1/k)

Monte Carlo integral: integral_a^b f(x)dx ~ (b-a) * mean(f(X_i)), X_i~U(a,b)
Monte Carlo pi:        pi ~ 4 * (points inside quarter circle / total points)
Buffon's needle:       pi ~ 2L / (P(cross) * plank_width)

SSQ performance:   d(n) = mean(delays)
                   q(n) = area_Q / T
                   u(n) = area_B / T   (or area_B/(c*T) for c servers)

Metropolis-Hastings: alpha = min(1, p(x')/p(x)),  accept if U(0,1) <= alpha
```

## 8. More practice (one question per syllabus pillar)

[Practice/PRACTICE_QUESTIONS.md](Practice/PRACTICE_QUESTIONS.md) has 7
new, exam-scaled practice questions (with full worked answers) covering
every pillar above, including two not yet touched by any appeared
question or the Src-26 practice set: an LCG-based RNG investigation
(mirrors C1, for the LCG instead of Middle-Square), an independence-test
question built around RANDU's hyperplane defect, and a Metropolis-Hastings
sampling question. Also included: a fresh DES/SSQ scenario (multi-server
+ balking), a "verification by trace" bug-hunt exercise for the V&V
section, and two more Monte Carlo problems in the same easy-to-moderate,
non-tricky style as A1 and the Src-26 set. Do these after the crash
review below if you want more reps before the exam.

## Study order (if time is short)

1. `Code/monte_carlo/core.py` (5 min — the single biggest leverage point)
2. `Code/simulation/single_server_queue.py` (10 min — the other big one)
3. `Code/rng/lcg.py` + `Code/rng/middle_square.py` (10 min)
4. `Code/testing/*.py` (10 min — know which test answers which question: uniformity vs independence)
5. `Code/monte_carlo/metropolis_hastings.py` (5 min)
6. Skim `Code/solutions/` so you've seen the exact A1/C1/C2 answers once
7. If time remains: work through `Code/solutions/practice_src26/` from memory, then check

`Code/cheatsheets/` (Python/numpy/random/scipy syntax) and `Code/viz/`
(plots) are reference-only — don't schedule study time for them, just
know they exist for the moment you blank on a syntax detail or want to
see rather than read a result.
