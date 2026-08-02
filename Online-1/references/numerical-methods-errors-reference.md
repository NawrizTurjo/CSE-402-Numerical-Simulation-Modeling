# CSE 401 — Numerical Analysis, Simulation & Modeling
## Reference Sheet: Approximations, Round-off & Truncation Errors (Lab 1)

Source: "Introduction, Approximations & Errors" lecture slides (Nafis Tahmid, CSE, BUET).
This file is a syllabus-aligned, code-ready reference for the first lab topic:
**Approximations, round-off errors, truncation errors; Visualization and plotting.**

It intentionally **excludes root-finding methods** (bisection, false position, Newton-Raphson) —
those belong to the next slide deck and are handled in a separate companion file/chat
(`numerical-methods-course-context.md`).

Purpose: feed this to an AI coding agent as ground truth for formulas, conventions, and
expected output formats, so generated code matches what the course (and the sample exam
question bank) expects — without re-deriving everything from scratch each session.

---

## 1. Error Definitions

### 1.1 True error (Et)
```
Et = True value − Approximate value
```
Only computable when the exact/true value is known (e.g., validating a method against a
closed-form answer).

### 1.2 Relative true error (εt)
```
εt = Et / True value
εt(%) = (Et / True value) × 100
```

### 1.3 Approximate error (Ea) — iterative refinement, no true value needed
```
Ea = Present approximation − Previous approximation
```

### 1.4 Relative approximate error (εa) — THE quantity monitored in every iterative routine
```
εa = Ea / Present approximation
εa(%) = (Ea / Present approximation) × 100
```
This is what essentially every stopping criterion in the course (and the exam bank) is built on.

**Sign convention:** Et and Ea are signed. Take the absolute value only when comparing against
a tolerance; keep the sign when reporting/printing the value itself (a negative Ea means the
new estimate is smaller than the previous one, and is informative — e.g. "we overestimated").

**Implementation note:** εa is undefined on the very first iteration (no previous value exists).
Print that row's Ea/εa as blank/`—`, not zero.

---

## 2. Stopping / Convergence Criteria

Two forms appear in the course:

### 2.1 Pre-specified tolerance εs
```
Stop when |εa| ≤ εs
```
This is the form used throughout the exam question bank, almost always written as
**`|εa| ≤ 0.0001%`** — i.e. εs = 0.0001 in **percent units**. Compare
`abs(eps_a_percent) <= 0.0001`, not `abs(eps_a_percent) <= 0.0001/100`.

### 2.2 Pre-specified significant digits m
To guarantee at least m correct significant digits, iterate until:
```
|εa(%)| ≤ 0.5 × 10^(2−m)
```
Example from the slides: m = 2 → threshold = 0.5×10⁰ = 0.5%.

### 2.3 Code pattern
```python
def converged(eps_a_percent, tol=None, sig_digits=None):
    if tol is not None:
        return abs(eps_a_percent) <= tol
    if sig_digits is not None:
        threshold = 0.5 * 10 ** (2 - sig_digits)
        return abs(eps_a_percent) <= threshold
    raise ValueError("supply tol or sig_digits")
```

---

## 3. Significant Digits

### 3.1 Definition
x̂ is correct to M significant digits when:
```
|x − x̂| ≤ 0.5 × 10^(n−M+1)
```
where `n = floor(log10(|x|))` is the exponent when x is written in normalized scientific
notation `x = m × 10^n`, `1 ≤ m < 10`.

### 3.2 Locating the M-th significant digit
The M-th significant digit of a number sits in the place value `10^(n−M+1)`.

Example: `3725.6 = 3.7256×10³` (n=3). The 3rd sig. digit ("2") sits in the
`10^(3−3+1) = 10¹` (tens) place.

### 3.3 Equivalence of the two error/digit criteria (proved in slides)
```
|εt| ≤ 0.5×10^(2−M)%   ⟺   x̂ correct to at least M significant digits
```
Derivation sketch: multiply the relative bound by `|x| = m·10^n`, use `m < 10` to bound
`m/2 < 5`, which collapses to `0.5×10^(n−M+1)` — exactly the Section 3.1 window.

### 3.4 Utility functions
```python
import math

def sci_exponent(x):
    """n such that x = m * 10**n, 1 <= |m| < 10."""
    return math.floor(math.log10(abs(x)))

def mth_sig_digit_place(x, M):
    """Power of 10 where the M-th significant digit of x lives."""
    n = sci_exponent(x)
    return n - M + 1
```

---

## 4. Sources of Error (conceptual — common short-answer prompt material)

1. **Modeling error** — wrong assumptions baked into the mathematical model itself.
   Slide's motivating case study: assuming a *constant* thermal-expansion coefficient α for a
   steel trunnion gave a contraction estimate (0.01504″) that satisfied the design requirement
   (0.015″) on paper — yet the real trunnion got stuck. Re-deriving with α(T) fit as a
   second-order polynomial and integrating ΔD = D∫α(T)dT gave the correct, smaller contraction
   (0.013689″), which explained the failure. Lesson: **a wrong model can pass every internal
   consistency check and still be wrong** — this is a distinct failure mode from round-off or
   truncation error.
2. **Data / measurement error and programming mistakes** — bad inputs, bugs; not intrinsic to
   the numerical method itself.
3. **Round-off error** — intrinsic to finite-precision arithmetic (Section 5).
4. **Truncation error** — intrinsic to approximating an infinite/continuous process with a
   finite/discrete one (Section 6).

Be ready to classify a given error source into one of these four categories, and to explain
*why* a numerical method needs both a "how wrong" quantification (error) and a "good enough"
threshold (stopping criterion) — a numerical answer without an error estimate is not a
complete answer in this course.

---

## 5. Round-off Error

### 5.1 Definition
A computer/calculator stores numbers with finite precision, so most real numbers are only
approximately representable.
```
Round-off error = true value − stored (rounded) value
```
Irrational numbers (π, √2) never have an exact finite representation and accumulate round-off
error in every calculation that uses them.

### 5.2 Worked pattern (from slides)
1/3 stored on a 6-digit machine as 0.333333:
```
round-off error = 1/3 − 0.333333 ≈ 3.3333×10⁻⁷
```

### 5.3 Code pattern to demonstrate round-off error
```python
def roundoff_demo(true_value, digits):
    """Simulate storing `true_value` with `digits` significant decimal digits."""
    stored = float(f"{true_value:.{digits}g}")
    error = true_value - stored
    return stored, error

# example: roundoff_demo(1/3, 6) -> (0.333333, 3.333...e-07)
```
Note: native Python floats are IEEE-754 doubles (~15–17 significant decimal digits), so
round-off is normally negligible at lab scale unless you deliberately truncate to fewer digits
(as above) to make the effect visible — that deliberate truncation *is* the demo.

---

## 6. Truncation Error

Truncation error = error from cutting short an infinite or continuous mathematical procedure
to make it computable. Three canonical demonstrations appear in the slides; all three are fair
game as lab/exam coding tasks.

### 6.1 Series truncation (Maclaurin / Taylor series)
General series:
```
f(x) = f(0) + f'(0)x + f''(0)x²/2! + f'''(0)x³/3! + ... = Σ_{n=0}^∞ [f⁽ⁿ⁾(0)/n!] xⁿ
```
Truncating after k terms leaves the discarded tail `Σ_{n=k+1}^∞ [f⁽ⁿ⁾(0)/n!] xⁿ` as the
truncation error.

**Required table columns** (matches the slide's worked e^1.2 example and the general course
convention): `n | Sn | Ea | |εa|% | # significant digits guaranteed`

```python
def maclaurin_series(term_recurrence, x, first_term=1.0, sig_digits=2, max_terms=200):
    """
    term_recurrence(prev_term, n, x) -> next term (n-th term from the (n-1)-th).
    e.g. for e^x: lambda prev, n, x: prev * x / n
    """
    term = first_term
    S = term
    threshold = 0.5 * 10 ** (2 - sig_digits)
    rows = [(0, S, None, None, None)]
    for n in range(1, max_terms + 1):
        term = term_recurrence(term, n, x)
        S_new = S + term
        Ea = S_new - S
        eps_a = abs(Ea / S_new) * 100
        sig = 0
        while eps_a <= 0.5 * 10 ** (2 - (sig + 1)):
            sig += 1
        rows.append((n, S_new, Ea, eps_a, sig))
        S = S_new
        if eps_a <= threshold:
            break
    return S, rows

# example for e^x:
# S, rows = maclaurin_series(lambda prev, n, x: prev * x / n, x=1.2, sig_digits=2)
```
Generalize the `term_recurrence` callback for whichever series a specific problem hands you
(ln(1+x), sin(x), cos(x), etc.) rather than hard-coding e^x.

### 6.2 Truncation in numerical differentiation (finite differences)
Forward difference approximation:
```
f'(x) ≈ [f(x+h) − f(x)] / h
```
Worked example (from slides): f(x) = x², exact f'(3) = 6.
With h = 0.2: approx = (3.2² − 3²)/0.2 = 6.2 → truncation error = 6 − 6.2 = −0.2.

This connects directly to Section 1: the earlier "two step sizes" example
(f(x) = 7e^0.5x, h=0.3 vs h=0.15) is an Ea/εa demonstration layered on the same idea — shrinking
h shrinks the truncation error, and comparing successive h's gives Ea even when the exact
derivative isn't known.

```python
def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def truncation_error_vs_exact(f, fprime_exact, x, h_list):
    """When the exact derivative IS known -> true/relative-true error table."""
    rows = []
    for h in h_list:
        approx = forward_diff(f, x, h)
        exact = fprime_exact(x)
        Et = exact - approx
        eps_t = abs(Et / exact) * 100 if exact != 0 else None
        rows.append((h, approx, exact, Et, eps_t))
    return rows

def approx_error_across_h(f, x, h_list):
    """When the exact derivative is NOT known -> compare successive h's (Ea/eps_a)."""
    rows, prev = [], None
    for h in h_list:
        approx = forward_diff(f, x, h)
        Ea = (approx - prev) if prev is not None else None
        eps_a = abs(Ea / approx) * 100 if Ea is not None else None
        rows.append((h, approx, Ea, eps_a))
        prev = approx
    return rows
```

### 6.3 Truncation in numerical integration (Riemann sums)
Left-endpoint rectangle rule:
```
∫ₐᵇ f(x) dx ≈ Σ f(xᵢ)·Δx,   Δx = (b−a)/N,   xᵢ = a + iΔx  (i = 0 .. N−1, left endpoints)
```
Worked example (from slides): f(x) = x², [a,b] = [3,9]. Exact = 234.
- N=2 rectangles (width 3) → 135, truncation error 99 (≈42.3%)
- N=4 rectangles (width 1.5) → 182.25, truncation error 51.75 (≈22.1%)

Pattern: more rectangles → smaller truncation error.

```python
def left_riemann(f, a, b, N):
    dx = (b - a) / N
    xs = [a + i * dx for i in range(N)]
    return sum(f(x) for x in xs) * dx

def integration_truncation_demo(f, exact_integral, a, b, N_list):
    rows = []
    for N in N_list:
        approx = left_riemann(f, a, b, N)
        err = exact_integral - approx
        pct = abs(err / exact_integral) * 100
        rows.append((N, approx, err, pct))
    return rows
```
**Scope note:** the slides only cover the left-endpoint rule at this stage. Do NOT introduce
trapezoidal/Simpson's rule unless a later slide deck explicitly does — this course is
cumulative and exam questions stay within material already covered.

---

## 7. Visualization & Plotting Conventions

The slide deck doesn't mandate a specific library, but the broader lab/exam question bank
(Section 8) consistently expects:

- **Function plots over a stated interval** — use a fine grid via `numpy.linspace(a, b, N)`
  with N large (≥500–1000) for a smooth curve. This plotting resolution is independent of any
  coarser step size used for numerical scanning (e.g. a 0.1 step used elsewhere to search for
  sign changes is *not* the plotting resolution).
- **Axis labels, title, grid** on every plot: `xlabel`, `ylabel`, `title`, `grid(True)`.
- **Mark y = 0** with a reference line whenever the plot's purpose is to visually locate roots —
  relevant once root-finding is introduced next, but establish the habit now:
  `plt.axhline(0, color='k', linewidth=0.8)`.
- **Log-scale plotting** when a domain spans many orders of magnitude (e.g. 10⁻⁴ ≤ x ≤ 10⁴ in
  the sample question bank) — use `plt.xscale('log')` / `plt.semilogx`. Be ready to justify:
  a linear scale would compress small-x behavior into invisibility.
- **Consistent save/show behavior** across all lab scripts (e.g. always `plt.savefig(...)` then
  `plt.show()`), so output is reproducible run to run.

Standard imports for this course's code style: `numpy`, `matplotlib.pyplot`.

---

## 8. Observed Exam/Lab Question Conventions (informs code *style*, not this chapter's formulas)

The uploaded past-questions file bundles root-finding tasks (Newton-Raphson, bisection, false
position — out of scope here) with recurring structural conventions worth carrying forward even
though the specific methods differ:

- **Convergence threshold is almost always** `|εa| ≤ 0.0001%` (Section 2.1) — build any
  iterative routine with this as a configurable default.
- **Iteration tables are a hard requirement**, not optional — every iterative method must print
  a row per iteration with the relevant quantities.
- **Coarse sign-change / candidate scanning** with a fixed step (commonly 0.1) over a stated
  domain is a recurring sub-task, used to bracket roots before running an iterative method —
  full detail deferred to the root-finding companion file, but the "scan, then refine, then
  print a table" shape is consistent across the course.
- **Theoretical explanation prompts travel with the code** ("explain in 3–4 sentences why...",
  "what would go wrong if...") — expect short conceptual answers to be demanded alongside a
  working script, not code-only submissions.
- **Comparison tables between two methods** (root found, iteration count, final function value,
  update counts, etc.) are a common final deliverable when two methods solve the same problem.
- Some question text is written in Banglish (Bengali in Roman letters) or mixes Bengali/English —
  the underlying numerical task is standard regardless of the language it's posed in.

---

## 9. Quick-Reference Formula Sheet

| Quantity | Formula |
|---|---|
| True error | Et = True − Approx |
| Relative true error | εt = Et / True (×100 for %) |
| Approximate error | Ea = Present − Previous |
| Relative approximate error | εa = Ea / Present (×100 for %) |
| Stop (tolerance) | \|εa\| ≤ εs |
| Stop (m sig. digits) | \|εa(%)\| ≤ 0.5×10^(2−m) |
| M-th sig. digit place | 10^(n−M+1), n = floor(log10\|x\|) |
| Forward difference | f'(x) ≈ [f(x+h) − f(x)] / h |
| Left Riemann sum | Σ f(xᵢ)·Δx, xᵢ = left endpoints |

---
*This file intentionally stops before root finding (bisection, false position, Newton-Raphson).
See the companion file `numerical-methods-course-context.md` to carry conventions into that
next topic without re-uploading the original question bank.*
