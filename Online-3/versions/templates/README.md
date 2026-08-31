# CSE401/402 Numerical Lab — Online 3 Templates

Coverage for: *Discrete event simulation models; Steps in a simulation study;
Model validation and verification; Random number generation; Monte Carlo
methods: Metropolis-Hastings.*

Built from `5_simu_des_ssqs.pdf` (97 slides) and `6_rng_mcs.pdf` (76 slides).
Every file is self-contained and runnable: `python3 <file>`.
**No file uses Python's `random` module** — all randomness comes from an LCG.

## Files

| File | Covers | Use when the question says... |
|---|---|---|
| `00_toolkit.py` | Every reusable function in one place | you want to copy 10 lines fast |
| `01_lcg_period_ks.py` | LCG/MCG, period, Hull–Dobell, K-S | "implement an LCG", "find the period", "why does it still pass?" |
| `02_middle_square.py` | Middle-square, degenerate seeds, chi-square | "middle-square", "find a bad seed", "is it uniform?" |
| `03_uniformity_tests.py` | Chi-square + K-S, multiple testing | "test for uniformity", "goodness of fit" |
| `04_independence_tests.py` | Autocorrelation, runs, serial, gap, poker | "test for independence", "is there a pattern?" |
| `05_mc_integration.py` | MC integration, convergence, variance reduction | "estimate this integral" |
| `06_mc_pi_buffon.py` | Buffon's needle, quarter-circle π | "estimate π", "assess MC performance" |
| `07_metropolis_hastings.py` | MH sampler, tuning, diagnostics, Bayesian | "Metropolis-Hastings", "sample from this density" |
| `08_des_single_server.py` | Single-server queue DES, M/M/1 validation | "simulate a queue", "average delay/utilization" |
| `09_des_variants.py` | M/M/c, capacity, warm-up, CRN, inventory | "how many servers?", "warm-up", "compare designs" |
| `10_random_variates.py` | Inverse transform, Box-Muller, accept-reject | "generate variates from this distribution" |

## Verified against the slides

Every worked example in both decks is reproduced exactly by the code:

| Slide | Result | File |
|---|---|---|
| RNG 18 | LCG(27,17,43,100) → 2, 77, 52 | `00_toolkit.py` |
| RNG 21 | periods 16/8/16/4 for seeds 1/2/3/4 | `00_toolkit.py` |
| RNG 26 | 16807 generator → 2074941799, … | `00_toolkit.py` |
| RNG 36 | K-S: D⁺=0.26, D⁻=0.21, D=0.26, crit 0.565 | `03_uniformity_tests.py` |
| RNG 39 | χ² = 3.4, critical 16.9 | `03_uniformity_tests.py` |
| RNG 56 | autocorr: M=4, ρ̂=−0.1945, σ=0.1280, Z₀=−1.52 | `04_independence_tests.py` |
| DES 41/43/55/56 | T=8.6, d̂=0.95, ∫Q=9.9, ∫B=7.7 | `08_des_single_server.py` |

## Formula sheet

**LCG** `X_{n+1} = (aX_n + c) mod m`, `U_n = X_n/m`

Max period (RNG slide 22):
- `c≠0`, `m=2^b`: period `m` iff gcd(c,m)=1 and `a = 1+4k`
- `c=0`, `m=2^b`: period `m/4`, needs `X0` odd and `a = 3+8k` or `5+8k`
- `c=0`, `m` prime: period `m−1`, needs `a` a primitive root mod `m`
- General (Hull–Dobell, c≠0): gcd(c,m)=1; `a−1` divisible by every prime factor of `m`; if `4|m` then `4|(a−1)`

**Chi-square** `χ²₀ = Σ(Oᵢ−Eᵢ)²/Eᵢ`, `Eᵢ = N/n`, `df = n−1`. Reject if `χ²₀ > χ²_{α,n−1}`. Needs `N ≥ 50`, `Eᵢ ≥ 5`.

**K-S** `D⁺ = max(i/N − R₍ᵢ₎)`, `D⁻ = max(R₍ᵢ₎ − (i−1)/N)`, `D = max(D⁺,D⁻)`. Large N: `D_crit = 1.36/√N` at α=0.05.

**Autocorrelation** `M` = largest int with `i+(M+1)ℓ ≤ N`;
`ρ̂ = [1/(M+1)]Σ R_{i+kℓ}R_{i+(k+1)ℓ} − 0.25`;
`σ = √(13M+7)/(12(M+1))`; `Z₀ = ρ̂/σ`; reject if `|Z₀| > 1.96`.

**Runs (up/down)** `E[a] = (2N−1)/3`, `Var(a) = (16N−29)/90`.

**Multiple testing** `P(≥1 false positive) = 1 − (1−α)^k` → 40% for k=10 at α=0.05.

**MC integration** `Ȳ(n) = (b−a)·(1/n)Σg(Xᵢ)`, `Xᵢ ~ U(a,b)`. SE shrinks as `1/√n`.

**MC π** quarter circle: `π̂ = 4·hits/n`. Buffon: `P = 2L/(πD)`, `π̂ = 2L/(D·P̂)`.

**Metropolis-Hastings** propose `x' = x + N(0,σ²)`; `α = min(1, f(x')/f(x))` (symmetric proposal); accept if `u < α`, **else record the current state again**.

**DES measures** `d̂(n) = Σ Dᵢ/n`; `q̂(n) = ∫Q(t)dt / T(n)`; `û(n) = ∫B(t)dt / T(n)`.

**M/M/1** `ρ=λ/μ`, `Lq=ρ²/(1−ρ)`, `Wq=ρ/(μ−λ)`, `L=ρ/(1−ρ)`, `W=1/(μ−λ)`.

**Inverse transform** Exponential `X = −mean·ln(1−U)`. Weibull `X = scale·(−ln(1−U))^(1/shape)`. Geometric `X = ⌊ln(1−U)/ln(1−p)⌋`.

**Box-Muller** `Z₁ = √(−2lnU₁)·cos(2πU₂)`, `Z₂ = √(−2lnU₁)·sin(2πU₂)`.

## The order-of-updates rule (DES slides 44, 58)

Inside every event routine:
1. Update **area accumulators** using the **old** `Q(t)`, `B(t)`, width `= clock − time_last_event`
2. Set `time_last_event = clock`
3. Update **state** variables
4. Update the **event list**

Classic bugs: updating `Q` before the area (wrong height); updating
`time_last_event` early (wrong width); forgetting `departure = ∞` when the
queue empties (phantom departures).

## Theory answers you may be asked to write

**Verification vs validation** — Verification: "did we build the model right?"
(program matches the conceptual model; use traces, code walkthroughs,
reasonableness checks, runs under known-answer conditions). Validation: "did
we build the right model?" (model matches reality). Calibration is the
iterative loop that achieves validation.

**Steps in a simulation study** (Law, 10 steps) — 1 formulate problem/plan ·
2 collect data & define model · 3 assumptions valid? → no: back to 2 ·
4 program & verify · 5 pilot runs · 6 programmed model valid? → no: back to 2 ·
7 design experiments · 8 production runs · 9 analyze output · 10 document &
present. **Both "no" arrows go back to step 2**, not to the previous step.

**Naylor–Finger validation** — face validity; validate model assumptions;
validate input-output transformations.

**Validation techniques by cost/value** (Van Horn) — face validity; statistical
tests of input data; Turing test; statistical comparison; new data collection.

**Why pseudo-random is fine** — repeatability enables debugging, fair
comparison of designs (common random numbers), and independent replication.

**Passing a test ≠ good generator** — uniformity tests say nothing about
independence (`0.01, 0.02, …` passes chi-square with p=1.0 and is fully
predictable — demonstrated in `02_middle_square.py`), and a short period can
hide behind a flat histogram. Need a battery of tests plus a long period.
