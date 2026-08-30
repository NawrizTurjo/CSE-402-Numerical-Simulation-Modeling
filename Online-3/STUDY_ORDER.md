# Quick & Simple Study Order — Online-3 (CSE 402)

A fast, no-stress study roadmap for the 30-minute exam.

---

## ⚡ 1. The Two Core Engines (15 mins — 80% of Exam Questions)

Master these two files first. Everything else wraps around them.

### Step 1: Monte Carlo Core
📁 **[`Code/monte_carlo/core.py`](Code/monte_carlo/core.py)**
- **What to understand:** `monte_carlo_estimate(trial_fn, n, seed=None)`.
- **Key concept:** Every Monte Carlo question only requires writing a 1-trial function (`trial_fn`) that returns:
  - `1.0` if an event occurs, `0.0` otherwise (probability)
  - The payoff/value of that trial (expected value)
  - `(b - a) * f(x)` (integration)
- **Run command:**
  ```powershell
  python monte_carlo/core.py   # or: python .\core.py (from inside monte_carlo/)
  ```

### Step 2: Single-Server Queue (DES)
📁 **[`Code/simulation/single_server_queue.py`](Code/simulation/single_server_queue.py)**
- **What to understand:** `simulate_ssq(...)` and `compute_performance_from_trace(...)`.
- **Key rule to memorize:** Accumulate area **before** updating state:
  - `area_Q += current_Q * dt`
  - `area_B += current_B * dt`
- **Formulas:**
  - $\text{Avg Delay } d(n) = \frac{\sum D_i}{n}$
  - $\text{Avg Queue Length } q(n) = \frac{\text{area}_Q}{T}$
  - $\text{Server Utilization } u(n) = \frac{\text{area}_B}{T}$
- **Run command:**
  ```powershell
  python simulation/single_server_queue.py   # or: python .\single_server_queue.py
  ```

---

## 🎲 2. RNGs & Cycle Detection (10 mins)

### Step 3: Cycle Detection Helper
📁 **[`Code/rng/common.py`](Code/rng/common.py)**
- `find_cycle(step_fn, seed)` uses a dictionary to detect when a state repeats and returns `(repeated_value, iteration, cycle_length)`.

### Step 4: Linear Congruential Generator (LCG)
📁 **[`Code/rng/lcg.py`](Code/rng/lcg.py)**
- **Recurrence:** $X_{i+1} = (a \cdot X_i + c) \bmod m, \quad R_i = X_i / m$.
- **Max Period Cases:**
  - **Mixed ($c \neq 0, m=2^b$):** Full period $m$ if $\gcd(c, m) = 1$ and $(a - 1) \bmod 4 = 0$.
  - **Multiplicative ($c = 0, m=2^b$):** Max period $m/4$ if seed is odd and $a \bmod 8 \in \{3, 5\}$.
  - **Prime Modulus ($c = 0$):** Max period $m - 1$ if $a$ is a primitive root mod $m$.
- **Run command:**
  ```powershell
  python rng/lcg.py   # or: python .\lcg.py
  ```

### Step 5: Middle-Square Method
📁 **[`Code/rng/middle_square.py`](Code/rng/middle_square.py)**
- **Algorithm:** Square seed $\to$ zero-pad to $2d$ digits $\to$ extract middle $d$ digits.
- **Weakness:** Degenerates into short cycles (e.g., seed 2500 gets stuck immediately; 5731 and 6239 collapse into a 4-value cycle).
- **Run command:**
  ```powershell
  python rng/middle_square.py   # or: python .\middle_square.py
  ```

---

## 📊 3. Statistical Testing of RNGs (10 mins)

Know which test checks what property:

### Step 6: Uniformity Tests (1-D Marginal Distribution)
📁 **[`Code/testing/chi_square_test.py`](Code/testing/chi_square_test.py)**
- Checks bin counts: $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$ with $df = \text{bins} - 1$.
- Decision: Reject $H_0$ if $p < 0.05$.

📁 **[`Code/testing/ks_test.py`](Code/testing/ks_test.py)**
- Checks Empirical CDF vs. $y = x$: $D = \max(D^+, D^-)$.
- Critical value: $D_{\text{crit}} = \frac{1.36}{\sqrt{N}}$ (at $\alpha = 0.05$).

### Step 7: Independence Tests (Serial Correlation)
📁 **[`Code/testing/independence_test.py`](Code/testing/independence_test.py)**
- **Autocorrelation test:** Checks correlation at lag $l$.
- **Runs test (above/below mean):** Checks count of runs above and below 0.5.
- **Runs up/down test:** Checks count of rising and falling runs.
- **Crucial Note:** A generator can pass uniformity tests (Chi-Square/K-S) but **fail** independence tests.

---

## 📝 4. Real Exam Solutions (Read Once — 10 mins)

Review how real questions are answered:

1. **[`Code/solutions/a1_buffons_needle.py`](Code/solutions/a1_buffons_needle.py)** (Exam A1)
   - Estimating $\pi$ with Buffon's Needle ($L=1, D=2 \implies P = \frac{1}{\pi}$).
   - Just wraps `monte_carlo_estimate` with a drop condition: $x \le \frac{L}{2} \sin(\theta)$.
2. **[`Code/solutions/c1_middle_square_investigation.py`](Code/solutions/c1_middle_square_investigation.py)** (Exam C1)
   - Middle-square generation, cycle detection on seed 2500, Chi-Square table, and reflection.
3. **[`Code/solutions/c2_hidden_period_collapse.py`](Code/solutions/c2_hidden_period_collapse.py)** (Exam C2)
   - Multiplicative LCG ($m=65536, a=5, X_0=1$).
   - Period is $p=16384 \ll N=100000$.
   - Why K-S fails to reject $H_0$: K-S sorts the data (order-invariant) and only measures 1-D uniformity, so cycle repeats do not shift the empirical CDF.

---

## 🎯 5. Quick Reference & Practice (Optional / Mid-Exam)

- **Variate Generation:** [`Code/variate_generation/inverse_transform.py`](Code/variate_generation/inverse_transform.py) ($F^{-1}(U)$ for Exponential, Weibull, Uniform).
- **MCMC Sampling:** [`Code/monte_carlo/metropolis_hastings.py`](Code/monte_carlo/metropolis_hastings.py).
- **Practice Problems:** [`Code/solutions/practice_src26/`](Code/solutions/practice_src26/) (`p1` through `p6`).
- **Syntax Cheatsheets:** [`Code/cheatsheets/`](Code/cheatsheets/) (Python builtins, `scipy.stats`, `heapq`).

---

## 🏁 Summary Checklist

```
[ ] 1. monte_carlo/core.py          (Understand trial_fn)
[ ] 2. simulation/single_server_queue.py (Understand update ordering)
[ ] 3. rng/lcg.py & rng/middle_square.py (RNG equations + cycle detection)
[ ] 4. testing/ (Chi-Square, K-S, and Independence differences)
[ ] 5. solutions/a1, c1, c2         (Skim worked answers)
```
