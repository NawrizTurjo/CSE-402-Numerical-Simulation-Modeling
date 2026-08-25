# CSE401: Random Number Generation and Monte Carlo Simulation
*Nafis Tahmid, Lecturer, Dept. of CSE, BUET — 82-slide deck, source: `6_rng_mcs.pdf`*

> Reference transcription for offline exam prep. Mirrors the deck's own section order. All formulas/numeric examples reproduced exactly as shown on slides.

---

## 1. Why Do We Need Randomness?

- Real-world systems are stochastic/uncertain (customer wait times, bridge survival under earthquake, server load on Black Friday, drug efficacy) — no neat closed-form formula, and building/testing the real thing is often infeasible.
- **Solution: Simulation** — build a computer model and let it run.
- **Simulation flow diagram:**
  ```
  Real System  --abstract-->  Computer Model  --run-->  Results
  (expensive,                 (cheap, safe,             (insights,
   dangerous,                  we control it)             predictions,
   doesn't exist yet)              ^                       decisions)
                                   |
                          Random Numbers needed here!
  ```
- Sources of randomness in simulations: customer arrivals, service times, component failures, weather/traffic/stock prices.
- **Key Idea:** To simulate any system with uncertainty, we first need a reliable source of random numbers between 0 and 1 (`U(0,1)`). Everything else is built on top of that.

---

## 2. Properties of Random Numbers

### What IS a random number?
A value `R` drawn from the interval `[0, 1]` where every point is equally likely (analogy: a "magic spinner" landing anywhere in `[0,1]` with equal likelihood).

### The Two Sacred Properties
A sequence `R1, R2, R3, ...` must have, **simultaneously**:

| Property 1: Uniformity | Property 2: Independence |
|---|---|
| Each `Ri` is equally likely to fall anywhere in `[0,1]`. No part of `[0,1]` is "preferred." `0.73` is just as likely as `0.12`. | Knowing `R1` tells you nothing about `R2`. If `R1 = 0.9`, that doesn't make `R2` more likely to be high or low. Each number "minds its own business." |

**Both properties must hold simultaneously** — uniformity without independence (or vice versa) is not enough.

### The Uniform Distribution U(0,1)
Each `Ri` follows a continuous uniform distribution on `[0,1]`. PDF:
```
f(x) = 1,  0 ≤ x ≤ 1
     = 0,  otherwise
```
(Flat, boxy pdf — a horizontal line at height 1 over `[0,1]`.)

### Expected value E(R)
```
E(R) = ∫₀¹ x·f(x) dx = ∫₀¹ x·1 dx = [x²/2]₀¹ = 1/2 − 0 = 1/2
```

### Variance Var(R)
```
Var(R) = E(R²) − [E(R)]²

E(R²) = ∫₀¹ x²·1 dx = [x³/3]₀¹ = 1/3

Var(R) = 1/3 − (1/2)² = 1/3 − 1/4 = 4/12 − 3/12 = 1/12 ≈ 0.0833
```

### Consequences of Uniformity and Independence
If we generate `N` random numbers and divide `[0,1]` into `n` equal subintervals:
- **Uniformity** ⇒ expected count in each subinterval = `N/n`.
- **Independence** ⇒ falling in one interval has no effect on where the next number lands.

(Diagram: `[0,1]` split into 5 equal marked subintervals, each labeled `N/n` above it.)

---

## 3. Generating Pseudo-Random Numbers

### The determinism paradox
- Computers are deterministic machines — they do exactly what you tell them, every time.
- So a deterministic machine **cannot** truly produce random numbers.
- That's why we call them **"pseudo-random"** numbers — "pseudo" = false/fake/imitation. They look and act random but are computed by a formula.

### Why "fake" random is fine
Repeatability (if the method/seed is known, the sequence repeats) is actually useful for:
- **Debugging** — reproduce the exact same crashing run.
- **Comparison** — test two system designs against the exact same random events.
- **Verification** — others can replicate results.

**The Goal:** Produce a sequence of numbers in `[0,1]` that simulates uniformity and independence as closely as possible.

### Things that can go wrong with PRNGs
1. Numbers might not be uniformly distributed (some regions preferred).
2. Numbers might be discrete instead of continuous (gaps).
3. The mean might be too high/low (biased).
4. The variance might be wrong (too clustered or too spread).
5. Dependence between successive numbers (patterns).

### What we want in a generator (desiderata)
1. **Fast** — simulations may need millions of random numbers.
2. **Portable** — same results on different computers.
3. **Long period** — sequence shouldn't repeat too soon (period ≫ number of values needed; a degenerate generator that repeats the same number is unacceptable).
4. **Repeatable** — same seed ⇒ same sequence.
5. **Statistically good** — passes uniformity & independence tests.

---

## 4. Linear Congruential Method (LCM / LCG)

### The recurrence
```
X(i+1) = (a · Xi + c) mod m,     i = 0, 1, 2, ...
```
- `X0` = the **seed** (starting value, chosen by user)
- `a` = the **multiplier**
- `c` = the **increment**
- `m` = the **modulus** (keeps numbers bounded)

**Two flavors:**
- `c ≠ 0`: **Mixed** congruential method.
- `c = 0`: **Multiplicative** congruential method.

### From integers to random numbers
LCM generates integers `X0, X1, X2, ...` in `{0, 1, ..., m−1}`. Convert to `Ri ∈ [0,1)`:
```
Ri = Xi / m,    i = 1, 2, 3, ...
```
`Ri` can only take values from `{0, 1/m, 2/m, ..., (m−1)/m}` — technically discrete, not continuous, but if `m` is huge (e.g. `2^31`), gaps are negligible.

### Example 1 — generating numbers
Setup: `X0 = 27, a = 17, c = 43, m = 100`.
```
X1 = (17·27 + 43) mod 100 = 502 mod 100 = 2      → R1 = 2/100  = 0.02
X2 = (17·2  + 43) mod 100 = 77  mod 100 = 77     → R2 = 77/100 = 0.77
X3 = (17·77 + 43) mod 100 = 1352 mod 100 = 52    → R3 = 52/100 = 0.52
```
**Flow diagram:** `27 --(×17+43, mod 100)--> 2 --(×17+43, mod 100)--> 77 --(×17+43, mod 100)--> 52 --> ...` with `R1=0.02, R2=0.77, R3=0.52` labeled beneath `X1,X2,X3`.

**Observations:**
- Each `Xi` completely determines `X(i+1)` — no actual randomness.
- Changing seed `X0` gives a different sequence.
- Quality depends entirely on the choices of `a, c, m`.

### Period
Since `Xi ∈ {0,...,m−1}`, there are at most `m` distinct values, so the sequence must eventually repeat; once it repeats a value, the whole sequence repeats from that point.

**The Period P:** length of the sequence before it starts repeating. We want `P` as large as possible. Ideally `P = m` (every integer appears exactly once before repeating). A short period is a disaster (simulation reuses the same "random" numbers).

### Example 2 — period in action
Setup: `X(i+1) = 13·Xi (mod 64)`, `c = 0` (multiplicative LCG), tested with 4 different seeds.

Full table (`i` = index, columns = sequence for each seed):

| i | X0=1 | X0=2 | X0=3 | X0=4 |
|---|---|---|---|---|
|0|1|2|3|4|
|1|13|26|39|52|
|2|41|18|59|36|
|3|21|42|63|20|
|4|17|34|51|**4**|
|5|29|58|23| |
|6|57|50|43| |
|7|37|10|47| |
|8|33|**2**|35| |
|9|45| |7| |
|10|9| |27| |
|11|53| |31| |
|12|49| |19| |
|13|61| |55| |
|14|25| |11| |
|15|5| |15| |
|16|**1**| |**3**| |

(Empty entries mean the sequence already returned to its initial seed before that row.)

**Observed periods:**

| Seed | Period |
|---|---|
| X0 = 4 | 4 |
| X0 = 2 | 8 |
| X0 = 1 | 16 |
| X0 = 3 | 16 |

**Lesson:** Odd seeds produce the maximum period of 16; even seeds produce shorter periods. The choice of seed significantly affects the generated sequence.

### Conditions for Maximum Period (Hull–Dobell-style theorem, as given)

The choice of `a, c, m, X0` determines the period.

**Case 1: `m = 2^b`, `c ≠ 0` (mixed congruential):**
- Maximum period `P = m = 2^b` when:
  - `c` and `m` are relatively prime, **and**
  - `a = 1 + 4k` for some integer `k`.

**Case 2: `m = 2^b`, `c = 0` (multiplicative):**
- Maximum period is only `P = m/4 = 2^(b−2)`.
- Achieved when `X0` is **odd** and `a = 3 + 8k` or `a = 5 + 8k`, for `k = 0, 1, 2, ...`.

**Case 3: `m` is prime, `c = 0`:**
- Maximum period `P = m − 1`.
- Requires: the smallest integer `k` such that `a^k − 1` is divisible by `m` equals `k = m − 1` (i.e., the smallest such `k` is `m − 1`).

### Why period 16 in Example 2? (worked verification)
`a = 13`, `m = 2^6 = 64`, `c = 0` → **Case 2**. Max period `= m/4 = 64/4 = 16`.
Check multiplier: `a = 13 = 5 + 8(1)`, form `5 + 8k` ✓.
Check seeds:
- `X0 = 1` (odd) → period 16 ✓
- `X0 = 3` (odd) → period 16 ✓
- `X0 = 2` (even) → period 8 < 16
- `X0 = 4` (even) → period 4 < 16

### Maximum Density
**Density** = how closely generated `Ri` values fill `[0,1]`.
- **Bad:** few scattered points on the line with a visible "big gap" between clusters.
- **Good:** many closely, evenly spaced points filling the whole line.

**Key Insight:** A large modulus `m` ⇒ more possible values for `Ri` ⇒ smaller gaps. Modern generators use `m = 2^31 − 1 ≈ 2.1 × 10^9`. Gap size `≈ 4.7 × 10^−10`. Practically continuous.

### The Modulo Trick: Why Powers of 2?
Computers use binary. When `m = 2^b`, computing `X mod 2^b` is trivially fast — just keep the last `b` bits.

**Decimal analogy (Example 3):** `m = 10^2 = 100`, `X0 = 63`, `a = 19`, `c = 0`.
```
X1 = 19·63 mod 100 = 1197 mod 100 = 97
X2 = 19·97 mod 100 = 1843 mod 100 = 43
X3 = 19·43 mod 100 = 817  mod 100 = 17
```
`mod 100` just means "keep the last 2 digits." Similarly `mod 2^b` means "keep the last `b` bits." The modulo step is essentially free on binary hardware.

### Example 4 — A Real-World Generator
Extensively tested, widely used generator:
```
a = 7^5 = 16,807,    c = 0,    m = 2^31 − 1 = 2,147,483,647
```
`m` is prime. Period `P = m − 1 > 2` billion.

Seed `X0 = 123,457`:
```
X1 = 16807 × 123457 mod (2^31 − 1) = 2,074,941,799
R1 = X1 / 2^31 = 0.9662   (dividing by m+1; for large m, m vs m+1 effect is negligible)

X2 = 559,872,160,   R2 = 0.2607
X3 = 1,645,535,613, R3 = 0.7662
```

---

## 5. Testing Random Numbers

### Motivation
A generator produces numbers that "look" random — but are they? We need statistical tests ("they look fine to me" isn't a valid statistical argument).

### Two categories of tests

**Uniformity Tests** (are numbers evenly spread across `[0,1]`?):
- Kolmogorov–Smirnov (K-S) test
- Chi-square test

**Independence Tests** (are numbers free of patterns?):
- Autocorrelation test
- Runs test
- Serial test

**Important caveat:** Passing all tests does not guarantee randomness (some pattern might go undetected), but failing a test is a definite red flag.

### Hypothesis setup
Uniformity:
```
H0: Ri ~ Uniform[0,1]   (numbers are uniform)
H1: Ri !~ Uniform[0,1]  (numbers are NOT uniform)
```
Independence:
```
H0: Ri are independent
H1: Ri are NOT independent
```
Significance level `α = P(reject H0 | H0 is actually true)`. Typical `α = 0.05` (5% chance of wrongly rejecting a good generator).

### Multiple Testing problem
For `k` independent tests at significance level `α`, assuming all null hypotheses are true:
```
P(at least one significant result by chance) = 1 − (1 − α)^k
```

**Example:** `k = 20`, `α = 0.05`:
```
1 − (1 − 0.05)^20 ≈ 0.64
```
Even though all 20 null hypotheses are true, ~64% chance at least one test appears significant purely by chance.

**Table (α = 0.05):**

| Number of tests k | P(at least one false positive) |
|---|---|
| 1 | 0.05 |
| 5 | 0.23 |
| 10 | 0.40 |
| 20 | 0.64 |
| 50 | 0.92 |
| 100 | 0.994 |

**Key message:** The 5% significance level applies per individual test. As the number of tests increases, the probability of ≥1 significant result purely by chance also increases.

### Implications for evaluating an RNG

**Situation 1 — many tests on one RNG output:** A test "fails" when `p < α` (H0 rejected). If a *correct* RNG is examined with 10 independent tests at `α = 0.05`:
```
P(at least one test fails by chance) = 1 − (0.95)^10 ≈ 0.40
```
Even a correct RNG has ~40% chance of failing at least one of 10 tests by chance — one isolated failure does not prove the RNG is defective.

**Situation 2 — repeating one test on many fresh sequences:** Same test applied to 100 independent sequences from a correct RNG, `α = 0.05`. Expected number of chance rejections:
```
100 × 0.05 = 5
```
- ~5 failures out of 100 may be normal.
- Many more than 5 failures may indicate a weakness.
- Repeated failure of the *same* test is more concerning than one isolated failure.

**Practical message:** One isolated failed test may be chance. Consistent failures across fresh sequences, or failures across several different tests, are stronger evidence of a real problem.

---

## 6. Frequency (Uniformity) Test 1: Kolmogorov–Smirnov (K-S)

**Goal:** Compare generated numbers to the theoretical `U(0,1)` distribution.

- Theoretical CDF: `F(x) = x` for `0 ≤ x ≤ 1`.
- Empirical CDF from sample `R1,...,RN`: `SN(x) = (number of Ri's ≤ x) / N`.
- If numbers are truly uniform, `SN(x)` should be close to `F(x)`. K-S test statistic measures the largest gap:
```
D = max_x |F(x) − SN(x)|
```
If `D` too large → reject `H0`.

### K-S Test Procedure
Given sample `R1,...,RN`:
1. **Sort** the data: `R(1) ≤ R(2) ≤ ... ≤ R(N)`.
2. Compute `D+ = max_{1≤i≤N} ( i/N − R(i) )`
3. Compute `D− = max_{1≤i≤N} ( R(i) − (i−1)/N )`
4. Test statistic: `D = max(D+, D−)`.
5. Look up critical value `D_α` from the K-S table (depends on `N` and `α`).
6. **Decision:** If `D > D_α`: reject `H0` (evidence of non-uniformity). If `D ≤ D_α`: fail to reject `H0`.

### Example 6 — K-S Test worked out
Data: `0.44, 0.81, 0.14, 0.05, 0.93`. `N = 5`, `α = 0.05`.

Step 1 — sort: `R(1)=0.05, R(2)=0.14, R(3)=0.44, R(4)=0.81, R(5)=0.93`.

Step 2–3 table:

| i | R(i) | i/N | D+_i = i/N − R(i) | (i−1)/N | D−_i = R(i) − (i−1)/N |
|---|---|---|---|---|---|
| 1 | 0.05 | 0.20 | 0.15 | 0.00 | 0.05 |
| 2 | 0.14 | 0.40 | **0.26** | 0.20 | −0.06 |
| 3 | 0.44 | 0.60 | 0.16 | 0.40 | 0.04 |
| 4 | 0.81 | 0.80 | −0.01 | 0.60 | **0.21** |
| 5 | 0.93 | 1.00 | 0.07 | 0.80 | 0.13 |

Largest observed gaps: `D+ = 0.26`, `D− = 0.21`.
```
D = max(D+, D−) = 0.26
```
Step 4: For `α = 0.05` and `N = 5`, `D_0.05 = 0.565`.
Step 5: Since `0.26 < 0.565`: **do not reject H0**. No significant evidence of non-uniformity.

---

## 7. Frequency (Uniformity) Test 2: Chi-Square Test

**Idea:** Divide `[0,1]` into `n` equal bins. Count how many numbers fall in each bin. Compare to expected. If bars look similar ⇒ probably uniform; if wildly different ⇒ probably not.

(Illustrative bar chart: 5 bins `[0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1]` — blue "Observed" bars fluctuate around a flat red "Expected" reference line ~14 each.)

### Formula
```
χ0² = Σ_{i=1}^{n} (Oi − Ei)² / Ei
```
- `Oi` = observed count in bin `i`
- `Ei` = expected count `= N/n` (for uniform distribution)
- `n` = number of bins, `N` = total observations

**Intuition:** each term `(Oi−Ei)²/Ei` measures "surprise" for bin `i`. `Oi ≈ Ei` → small contribution; `Oi` far from `Ei` → large contribution.

**Decision rule:** Compare `χ0²` to critical value `χ²_{α, n−1}` (degrees of freedom = `n − 1`). If `χ0² > χ²_{α,n−1}`: reject `H0`.

### Example 7 — Chi-Square Test
Data: 100 random numbers, `n = 10` bins, `α = 0.05`. Expected `Ei = 100/10 = 10`.

| Interval | Oi | Ei | (Oi−Ei)² | (Oi−Ei)²/Ei |
|---|---|---|---|---|
| [0.0,0.1) | 8 | 10 | 4 | 0.4 |
| [0.1,0.2) | 8 | 10 | 4 | 0.4 |
| [0.2,0.3) | 10 | 10 | 0 | 0.0 |
| [0.3,0.4) | 9 | 10 | 1 | 0.1 |
| [0.4,0.5) | 12 | 10 | 4 | 0.4 |
| [0.5,0.6) | 8 | 10 | 4 | 0.4 |
| [0.6,0.7) | 10 | 10 | 0 | 0.0 |
| [0.7,0.8) | 14 | 10 | 16 | 1.6 |
| [0.8,0.9) | 10 | 10 | 0 | 0.0 |
| [0.9,1.0] | 11 | 10 | 1 | 0.1 |
| **Total** | **100** | **100** | | **χ0² = 3.4** |

Critical value: `χ²_{0.05,9} = 16.9`. Since `3.4 ≤ 16.9`: **do not reject H0**. Looks fine.

### K-S vs. Chi-Square — which to use?
- Both acceptable for uniformity testing given large enough sample size.
- **K-S is generally more powerful**, can be applied to small sample sizes.
- **Chi-square is valid only for large samples**, say `N ≥ 50`.

**But uniformity tests are not enough!** A sequence like `0.01, 0.02, ..., 0.09, 0.11, 0.12, ...` is perfectly uniform but completely predictable — independence must also be tested.

---

## 8. Autocorrelation (Independence) Test

### Motivating example
Sequence (read left to right, 30 values):
```
0.12 0.01 0.23 0.28 0.89 0.31 0.64 0.28 0.83 0.93
0.99 0.15 0.33 0.35 0.91 0.41 0.60 0.27 0.75 0.88
0.68 0.49 0.05 0.43 0.95 0.58 0.19 0.36 0.69 0.87
```
Looks random by overall distribution — but every 5th number starting at position 5 is large: `0.89, 0.93, 0.91, 0.88, 0.95, 0.87`. The sequence may look uniform, but order may not be random. **Autocorrelation tests check whether numbers separated by a fixed lag are related.**

### Setup
Choose: `i` = starting position, `γ` = lag. Look at the subsequence:
```
Ri, R(i+γ), R(i+2γ), ..., R(i+(M+1)γ)
```
`M` = largest integer such that `i + (M+1)γ ≤ N`.

We are testing whether each selected number is related to the *next* selected number: pairs `(Ri, R(i+γ)), (R(i+γ), R(i+2γ)), ...`

### Two-tailed test rationale
Want `ρ_i = 0` (no relationship at lag `γ`). Dependence can go two ways:

| Case | Meaning | Problem? |
|---|---|---|
| `ρ_i > 0` | Average is higher than expected | yes |
| `ρ_i < 0` | Average is lower than expected | yes |
| `ρ_i = 0` | No clear relationship | desired |

Reject if `ρ̂_i` is too far from zero **in either direction** (two-tailed).

### The estimator ρ̂_i
For two independent `U(0,1)` numbers `R, R'`: `E[RR'] = E[R]E[R'] = 1/2 · 1/2 = 0.25`. Under independence, average product of neighboring pairs should be `≈ 0.25`:
```
ρ̂_i = [ (1/(M+1)) · Σ_{k=0}^{M} R(i+kγ)·R(i+(k+1)γ) ] − 0.25
```
Meaning: `ρ̂_i ≈ 0` → no evidence of dependence; `ρ̂_i > 0` → products larger than expected; `ρ̂_i < 0` → smaller than expected.

### Standard deviation of ρ̂_i
```
σ_{ρ̂_i} = sqrt( (13M + 7) / (12(M+1)) )
```

#### Full derivation as given on slides
Background: `R ~ U(0,1)` ⇒ `E[R] = 1/2`, `E[R²] = 1/3`.

Define one pair-product `Yk = R(i+kγ)·R(i+(k+1)γ) = A·B` (A, B independent U(0,1)):
```
E[Yk] = E[A]E[B] = 1/2 · 1/2 = 1/4
E[Yk²] = E[A²]E[B²] = 1/3 · 1/3 = 1/9
Var(Yk) = E[Yk²] − (E[Yk])² = 1/9 − 1/16 = 7/144
```

Covariance between adjacent products: since `Yk = AB` and `Y(k+1) = BC` share `B`, they're not independent.
```
Cov(Yk, Y(k+1)) = E[Yk·Y(k+1)] − E[Yk]E[Y(k+1)]
Yk·Y(k+1) = AB²C
E[AB²C] = E[A]E[B²]E[C] = 1/2 · 1/3 · 1/2 = 1/12
E[Yk]E[Y(k+1)] = 1/4 · 1/4 = 1/16
Cov(Yk, Y(k+1)) = 1/12 − 1/16 = 1/48
```

Let `S = Σ_{k=0}^{M} Yk`, so `ρ̂_i = S/(M+1) − 0.25`, and (constant doesn't affect variance):
```
Var(ρ̂_i) = Var(S) / (M+1)²
```

**Variance terms:** `M+1` terms each with `Var(Yk) = 7/144`:
```
Σ Var(Yk) = (M+1)·(7/144)
```

**Covariance terms:** only adjacent products `(Y0,Y1), (Y1,Y2), ..., (Y(M−1),YM)` are correlated — `M` adjacent pairs, each `Cov = 1/48`:
```
Σ_{k=0}^{M−1} Cov(Yk, Y(k+1)) = M · (1/48)
```
Formula for `Var(sum)` needs `2·Σ_{j<k} Cov(Yj,Yk) = 2M/48`.

Combine:
```
Var(S) = (M+1)·(7/144) + 2M/48
```
```
Var(ρ̂_i) = [ (M+1)·(7/144) + 2M/48 ] / (M+1)²
```
Simplifies to:
```
Var(ρ̂_i) = (13M + 7) / (144(M+1)²)
```
Standard deviation:
```
σ_{ρ̂_i} = sqrt(Var(ρ̂_i)) = sqrt( (13M+7) / (12(M+1)) )
```
(Note the 144 → 12² takes a factor of 12 out from under the outer sqrt, giving `12(M+1)` in the denominator inside the remaining sqrt — matches slide's boxed formula above.)

### Test statistic Z0
`H0: ρ_i = 0`, so expected `ρ̂_i` under `H0` is zero.
```
Z0 = (ρ̂_i − 0) / σ_{ρ̂_i} = ρ̂_i / σ_{ρ̂_i}
```
Z0 = how many standard deviations `ρ̂_i` is from zero. For `α = 0.05`, reject `H0` if:
```
|Z0| > 1.96
```

### Example 8 — Autocorrelation Test worked out
Test the 3rd, 8th, 13th, ... numbers: `i = 3`, `γ = 5`, `N = 30`, `α = 0.05`.

Find `M`: largest integer where `3 + (M+1)·5 ≤ 30` → `M = 4`.

Subsequence: `R3, R8, R13, R18, R23, R28 = 0.23, 0.28, 0.33, 0.27, 0.05, 0.36`.

Compute `ρ̂_{35}` (lag-5 estimate, i=3):
```
ρ̂_35 = (1/5)[(0.23)(0.28) + (0.28)(0.33) + (0.33)(0.27) + (0.27)(0.05) + (0.05)(0.36)] − 0.25

= (1/5)[0.0644 + 0.0924 + 0.0891 + 0.0135 + 0.018] − 0.25
= (1/5)(0.2774) − 0.25   ... slide shows 0.0555 − 0.25
= 0.0555 − 0.25 = −0.1945
```
Standard deviation:
```
σ = sqrt( (13(4)+7) / (12(5)) ) = sqrt(59/60) = 0.1280
```
Test statistic:
```
Z0 = −0.1945 / 0.1280 = −1.52
```
Since `|−1.52| = 1.52 < 1.96 = z_{0.025}`: **do not reject H0**. No evidence of autocorrelation.

### Warning about "fishing"
In a dataset of `N = 30` numbers, many possible subsequences can be tested. If you test 10 subsequences at `α = 0.05`:
```
P(finding autocorrelation by chance) = 1 − (0.95)^10 ≈ 0.40
```
**The Fishing Problem:** if you look hard enough for patterns, you may find them even in truly random data (like finding shapes in clouds). Bottom line: a few rejections out of many tests are expected — worry only if there are significantly more rejections than `α` would predict.

---

## 9. Monte Carlo Simulation

**Definition:** A scheme employing random numbers to solve stochastic or deterministic problems. (You can use randomness to solve *non-random* problems.)

**History:** Named after the Monte Carlo Casino, Monaco. Developed during WWII by scientists on the Manhattan Project (atomic bomb). Invented by **Stanislaw Ulam and John von Neumann**.

### The Integration Problem
Evaluate `I = ∫_a^b g(x) dx` where `g(x)` cannot be integrated analytically. Standard approaches (Simpson's rule, trapezoidal rule) work well in 1D but struggle badly in high dimensions (multiple integrals).

### Key Insight: Integration as Expectation
Let `X ~ U(a,b)`. Define `Y = (b−a)·g(X)`. Then:
```
E(Y) = (b−a)·E[g(X)]
     = (b−a) ∫_a^b g(x)·f_X(x) dx        [f_X(x) = 1/(b−a)]
     = (b−a) · (1/(b−a)) ∫_a^b g(x) dx
     = ∫_a^b g(x) dx = I
```
**The integral equals the expected value of a random variable** — integration becomes an estimation problem.

### The Sample Mean Estimator
```
I = E(Y),   estimate via sample mean:

Ȳ(n) = (1/n) Σ_{i=1}^{n} Yi ,   Yi = (b−a)·g(Xi),  Xi ~iid U(a,b)

Ȳ(n) = (b−a) · (1/n) Σ_{i=1}^{n} g(Xi)
```
**Steps:**
1. Pick `n` random points `X1,...,Xn` uniformly in `[a,b]`.
2. Evaluate `g` at each point.
3. Average those values: `(1/n) Σ g(Xi)`.
4. Multiply by interval length `(b−a)`.

**Geometric picture:** `(1/n)Σg(Xi)` is the average height of `g(x)` at random points; multiplying by `(b−a)` gives the area of a rectangle (base `(b−a)`, height = average height) approximating the area under the curve. As sample points increase, average height → true average of `g(x)`, rectangle area → `I`.

### Worked visualization — ∫₀^π sin(x) dx with n = 5
Diagram: curve `g(x) = sin(x)` over `[0, π]`, shaded area under curve = true integral. 5 random x-values `x4, x1, x2, x5, x3` marked with red dots on the curve, dashed line at height `ḡ = 0.584`.
```
Step 1: pick random x in [0, π], evaluate sin(Xi) (red dots).
Step 2: ḡ = (0.479 + 0.932 + 0.335 + 0.198 + 0.974)/5 = 0.584
Step 3: multiply by (b−a) = π:  Ȳ(5) = π × 0.584 = 1.83
True answer = 2. Not bad for 5 points!
```

### Full worked example — ∫₀^π sin(x) dx with n = 10
Exact answer: `∫₀^π sin(x) dx = [−cos x]₀^π = −(−1) − (−1) = 2`.

`n = 10` random `Xi ~ U(0,π)`:
```
X1=0.52  X2=1.19  X3=2.83  X4=0.21  X5=1.77
X6=2.50  X7=0.89  X8=1.53  X9=2.97  X10=2.11
```
`sin(Xi)` values:
```
0.497  0.929  0.301  0.208  0.980
0.599  0.777  0.999  0.172  0.870
```
Average: `(1/10)(0.497+0.929+...+0.870) = 0.633`
Multiply by `(b−a) = π`: `Ȳ(10) = π × 0.633 = 1.989`
True value = 2. Error ≈ 0.5% with just 10 random points.

### Does more sampling help? (convergence table)

| n | Ȳ(n) | Error |
|---|---|---|
| 10 | 2.213 | 0.213 (10.7%) |
| 20 | 1.951 | 0.049 (2.5%) |
| 40 | 1.948 | 0.052 (2.6%) |
| 80 | 1.989 | 0.011 (0.6%) |
| 160 | 1.993 | 0.007 (0.4%) |

(Plot: `Ȳ(n)` vs `n` starts at 2.2 for small `n`, oscillates, and converges/tightens toward the true value 2 as `n` grows to 150+.)

**The estimate "orbits" around the true value, getting closer and tighter — the law of large numbers in action.**

**Why is error not monotonically decreasing?** `Ȳ(40) = 1.948` is *worse* than `Ȳ(20) = 1.951`. Not a bug — it's the nature of randomness: each run draws different random `Xi`'s, so `Ȳ(n)` is itself a random variable (no guarantee of monotone convergence, only convergence in expectation/probability).

---

## 10. Monte Carlo: Estimating π (Classic Example)

**Problem:** Estimate π using only random numbers.

**Idea:** A quarter circle of radius 1 fits inside a unit square. Area of quarter circle `= π/4`; area of square `= 1`.

(Diagram: unit square with shaded quarter-circle region (`x²+y²≤1`); scattered points — green dots inside the quarter circle, red dots outside it, in the square.)

**Method:** Generate random points `(x,y)` in the unit square. Count how many fall inside the quarter circle (`x²+y² ≤ 1`).
```
π̂ = 4 × (points inside circle) / (total points)
```

### Step by step
1. Generate `n` pairs `(Xi, Yi)`, `Xi, Yi ~ U(0,1)`.
2. For each pair, check if `Xi² + Yi² ≤ 1`.
3. Count the "hits" (inside circle) → `h`.
4. Estimate: `π ≈ 4h/n`.

**Why this works:**
```
P(point inside quarter circle) = (area of quarter circle) / (area of square) = (π/4)/1 = π/4
```
So `h/n ≈ π/4`, hence `π ≈ 4h/n`.

### Convergence table

| n (points) | π̂ | True: π = 3.14159... |
|---|---|---|
| 100 | 3.24 | |
| 1,000 | 3.16 | |
| 10,000 | 3.1436 | |
| 100,000 | 3.1412 | |
| 1,000,000 | 3.14178 | |

---

## Deck scope note

This specific slide deck (`6_rng_mcs.pdf`, 82 pages) covers: motivation for randomness → properties of random numbers (uniformity/independence, U(0,1) moments) → pseudo-random generation via the **Linear Congruential Method only** (no middle-square/other RNG techniques appear in this deck) → statistical testing (multiple-testing caveats, K-S test, Chi-square test, autocorrelation/independence test with full variance derivation) → Monte Carlo simulation (integration-as-expectation, sample-mean estimator, worked `sin(x)` integral, and the classic π-estimation example). The deck ends after the π-estimation example (slide 81) with a closing "Goodbye!" slide (82) — it does **not** include the Metropolis-Hastings algorithm or any MCMC content.
