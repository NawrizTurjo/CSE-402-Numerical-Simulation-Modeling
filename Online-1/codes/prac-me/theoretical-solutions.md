# Theoretical Solutions — Prev-Section Question Bank (Compiled)

Companion to `../prev-section/question-bank.md`. Each entry: section name,
matching solution file, plain-language explanation, then the actual theory/math.

---

## Table of Contents
1. [Section-A1: Diode Equation (Newton-Raphson)](#section-a1) — `01_diode_newton_raphson.py`
2. [Section-C2 : Log-Scale Convergence Comparison](#section-c2-type-1) — `02_logscale_bisect_vs_falsepos.py`
3. [Section-B1: Multi-Root Scanner (Bisection)](#section-b1) — `03_multiroot_scanner_bisection.py`
4. [Section-C1: First-Match Multi-Root Scanner (False Position)](#section-c1) — `04_first_bracket_false_position.py`
5. [Section-B2: Pathological/Discontinuous Functions (Bisection)](#section-b2) — `05_pathological_rational_bisection.py`

---

<a id="section-a1"></a>
## 1. Section-A1: Diode Equation (Newton-Raphson)
**File:** `01_diode_newton_raphson.py`

### Plain-English idea
You have one equation with one unknown voltage `V`, and it's transcendental
(has `V` stuck inside an `exp()`), so you can't just isolate `V` with algebra.
Newton-Raphson finds it by repeatedly drawing the **tangent line** at your
current guess and walking to where that tangent crosses zero. Because the
tangent hugs the curve closely near a root, this converges very fast (roughly
doubling correct digits each step) — *if* your starting guess isn't too far
into a region where the curve bends sharply (which is exactly what happens
here at V0=0.65, see below).

### Problem
$$f(V) = 10^{-12}\left(e^{V/(nV_T)} - 1\right) + \frac{V}{R} - I_L = 0$$
V0=0.65 V, n=1.8, R=500Ω, VT=0.02585 V, IL=0.0002 A, stop when |ea| ≤ 0.0001%.

### Derivation
$$f'(V) = \frac{I_s}{nV_T}e^{V/(nV_T)} + \frac{1}{R}, \qquad n V_T = 0.04653\text{ V}$$

Iteration formula:
$$V_{i+1} = V_i - \frac{f(V_i)}{f'(V_i)}$$

**Why iteration 1 looks so violent:** at V0=0.65, the term `V/(nV_T) = 13.97`,
so `exp(13.97) ≈ 1.166e6` — the diode current term is astronomically larger
than it "should" be at the true operating point (~0.1 V). The tangent slope
there is steep but the function value is huge too, so the first Newton step
overshoots hard, from 0.65 V down to ≈0.106 V (ea ≈ 512%). This is normal:
diode-style exponential equations often need 1 "correction" jump before
settling into the region where quadratic convergence takes over — iteration 2
onward converges essentially instantly.

$$V_1 = 0.65 - \frac{1.101166\times10^{-3}}{2.025069\times10^{-3}} \approx 0.106233\text{ V}$$

### Key formula to remember
$$V_{i+1} = V_i - \frac{f(V_i)}{f'(V_i)}, \qquad \varepsilon_a\% = \left|\frac{V_{i+1}-V_i}{V_{i+1}}\right|\times100$$

### Code notes
- Use `math.exp` (scalar) not `np.exp` inside a loop — no array overhead.
- First iteration has no previous value, so ea is printed as `"---"`.

---

<a id="section-c2-type-1"></a>
## 2. Section-C2 (Type 1): Log-Scale Convergence Comparison
**File:** `02_logscale_bisect_vs_falsepos.py`

### Plain-English idea
Same root (`ln(x)=0` → x=1), two different bracketing methods, wildly
different speeds. This question is really testing: *do you understand why
False Position can be much slower than Bisection despite looking "smarter"?*
Bisection always cuts the interval exactly in half — dumb but fair. False
Position draws a straight line (secant) between the two endpoints and jumps
to where that line crosses zero — smarter-looking, but if the curve is
strongly curved (concave) in one direction, that secant line keeps landing on
the *same side* every time, so one endpoint (here `xl`) never moves. It gets
stuck doing tiny steps for 100+ iterations instead of halving the gap.

### Problem
f(x) = ln(x) on [1e-4, 1e4], ea ≤ 0.0001%.

### Boundary check
f(1e-4) ≈ -9.21 (neg), f(1e4) ≈ +9.21 (pos) → opposite signs → root exists
inside by the Intermediate Value Theorem.

### Why Bisection wins (34 vs 126 iterations)
- Bisection: interval width shrinks by exactly 2× every step, regardless of
  the function's shape. Purely geometric, guaranteed steady progress.
- False Position: `ln(x)` is concave down everywhere (`f''(x) = -1/x^2 < 0`).
  A straight secant drawn under a concave-down curve always crosses zero
  *to the right* of the true root. Since the crossing point `xr` keeps
  landing with `f(xr) > 0`, the rule always does `xu = xr`, and the lower
  bound `xl` gets frozen at `1e-4` for the entire run (0 updates over 125
  iterations) — the "bracket" stops shrinking from the left side at all,
  degrading a two-sided method into a slow one-sided crawl.

### Key formula to remember
$$x_r = x_u - \frac{f(x_u)(x_l - x_u)}{f(x_l)-f(x_u)}$$
Update rule (same idea for both methods): if `f(xl)*f(xr) < 0`, root is in
`[xl, xr]` so `xu = xr`; else root is in `[xr, xu]` so `xl = xr`.

### Code notes
- `np.logspace(-4, 4, 1000)` + `plt.xscale('log')` for the log-axis plot.
- Track `xl_updates` / `xu_updates` counters inside the update `if/else`.
- Run both methods as clean subroutines returning `(root, iters, ea,
  xl_updates, xu_updates, history)` so a single loop can print the final
  comparison table.

---

<a id="section-b1"></a>
## 3. Section-B1: Multi-Root Scanner (Bisection)
**File:** `03_multiroot_scanner_bisection.py`

### Plain-English idea
Bisection can only find **one** root per call, and it needs the two endpoints
you feed it to have opposite signs. So if a function wiggles through the axis
several times across a wide domain, you cannot just call bisection once on
the whole domain — you first need to **chop the domain into many small
strips** and check each strip individually for a sign flip. Only strips that
show a flip get handed to bisection.

### Problem
f(x) = 0.6·ln(x+1) − C·sin(1.7x) − 0.08x² − 0.08, domain [0,10], step 0.1.

### Why a single global Bisection call fails here
1. **Even root count cancels out:** f(0) ≈ −0.08 (neg), f(10) ≈ −7.42 (neg) —
   same sign at the two ends even though the function crosses zero multiple
   times in between. `f(a)·f(b) > 0` tells bisection "no root here" and it
   refuses to even start, even though roots clearly exist inside.
2. **Bisection is inherently single-root:** even when the endpoints do have
   opposite signs, bisection's halving process can only converge on *one*
   crossing — any other roots between those same endpoints are silently
   never visited.

### The fix — Intermediate Value Theorem, applied locally
Split [0,10] into small sub-intervals of width Δx=0.1. For every adjacent
pair `(x_i, x_i+Δx)`, if `f(x_i)·f(x_i+Δx) < 0`, a root is *guaranteed*
somewhere inside that tiny strip (continuity + sign change). Collect every
such strip, then run Bisection separately on each one.

### Key formula to remember
$$f(x_1)\cdot f(x_2) < 0 \;\Rightarrow\; \exists\, x^* \in [x_1,x_2] : f(x^*)=0$$

### Code notes
```python
xs = np.arange(x_min, x_max, step)
intervals = [(xs[i], xs[i+1]) for i in range(len(xs)-1) if f(xs[i])*f(xs[i+1]) < 0]
```
Then loop `intervals` and call the bisection solver once per bracket.

---

<a id="section-c1"></a>
## 4. Section-C1: First-Match Multi-Root Scanner (False Position)
**File:** `04_first_bracket_false_position.py`

### Plain-English idea
Same scanning idea as Section-B1, but this time you don't want *every* root —
just the **first** bracket the scanner trips over, and you solve that one
bracket with False Position instead of Bisection. So: scan left to right,
stop at the first sign change, break out of the loop immediately, solve.

### Problem
f(x) = 2x³ − 11.7x² + 17.7x − 5, domain [0,4], step 0.1.

### Finding the first match by hand
- f(0.3) = 0.311 (pos), f(0.4) = −0.648 (neg) → first sign flip →
  bracket = [0.3, 0.4].

### False Position formula and update rule
$$x_r = x_u - \frac{f(x_u)(x_l-x_u)}{f(x_l)-f(x_u)}$$
This is the x-intercept of the straight line joining `(xl, f(xl))` and
`(xu, f(xu))` — instead of blindly bisecting, it aims for where a *linear*
guess says the root should be, which is often faster when the function is
close to linear inside the bracket. Update: if `f(xl)*f(xr) < 0`, root is in
`[xl, xr]` → `xu = xr`; otherwise root is in `[xr, xu]` → `xl = xr`.

### Key formula to remember
$$\varepsilon_a\% = \left|\frac{x_r^{new} - x_r^{old}}{x_r^{new}}\right|\times100$$

### Code notes
- Break the scan loop the instant the first bracket is found — don't keep
  scanning the rest of the domain (that's the whole point vs. Section-B1).
- Keep `xr_old` from the previous iteration to compute `ea`.

---

<a id="section-b2"></a>
## 5. Section-B2: Pathological/Discontinuous Functions (Bisection)
**File:** `05_pathological_rational_bisection.py`

### Plain-English idea
Root-finding by sign-change scanning silently assumes two things that aren't
always true: (1) every root actually **crosses** the axis, and (2) every
place the sign flips is an actual **root**. This problem breaks both
assumptions on purpose, using one rational function with a squared factor
(fails assumption 1) and a division-by-zero pole (fails assumption 2).

### Problem
$$f(x) = \frac{(x-2.5)^2(x+1.5)}{x-3.56}$$

### Case 1 — Missed root at x = 2.5 (double root, doesn't cross)
`(x-2.5)^2` is always ≥ 0, so it never flips sign by itself — it only makes
the curve **touch** zero at x=2.5 and bounce back to the same sign on both
sides (since the other factors, `(x+1.5)` and `1/(x-3.56)`, don't change sign
right around there). No sign change ⇒ `f(xl)·f(xu) > 0` for any bracket
around 2.5 ⇒ a pure incremental sign-scan **never even sees this root**.
*(Lesson: even-multiplicity roots are invisible to sign-change scanning —
you'd need to scan for minima of |f(x)| instead, or use Newton-Raphson with
multiplicity m=2, which is specifically built to still converge on these.)*

### Case 2 — Garbage root at x = 3.56 (a pole, not a root)
At x=3.56 the denominator hits zero, so f(x) rockets to −∞ approaching from
the left and +∞ approaching from the right (or vice versa). That's a real
sign change from the scanner's point of view, so it gets bracketed and fed to
Bisection — and Bisection happily "converges" toward x=3.56, because
halving the interval keeps landing closer and closer to the discontinuity.
But the function isn't approaching zero there, it's approaching infinity —
so this is a **fake root** produced purely by the discontinuity.

### The real fix
Sign change alone is not proof of a root — you must **verify the residual**
after convergence: compute f(x*) at the "root" found. A real root gives
`f(x*) ≈ 0`; a garbage pole-root gives a huge/undefined value. This is why
`classify_and_verify()`'s residual test (`|f(x)| ≤ tol`) exists — it's the
safety net that catches exactly this case.

### Key ideas to remember
- Even multiplicity (squared factor) → touches axis, no sign flip → **missed**.
- Vertical asymptote/pole → sign flips, but not a root → **garbage, must reject via residual check**.
- Always verify `f(root) ≈ 0` after any solver converges — convergence ≠ correctness.

### Code notes
```python
y_vals = ((x - 2.5)**2 * (x + 1.5)) / (x - 3.56)
y_vals[np.abs(x - 3.56) < 0.02] = np.nan     # mask so plot doesn't draw a fake vertical line
```
- `plt.ylim(-30, 30)` keeps the asymptote from stretching the plot to infinity.
- Guard Bisection: if `f(xr)` starts exploding (very large / inf), stop and
  flag instead of trusting convergence blindly.

---

## Quick-Reference: All Core Formulas

| Method | Formula |
|---|---|
| Bisection | $x_r = \dfrac{x_l+x_u}{2}$ |
| False Position | $x_r = x_u - \dfrac{f(x_u)(x_l-x_u)}{f(x_l)-f(x_u)}$ |
| Newton-Raphson | $x_{i+1} = x_i - m\dfrac{f(x_i)}{f'(x_i)}$ |
| Approx. rel. error | $\varepsilon_a\% = \left\lvert\dfrac{x_{new}-x_{old}}{x_{new}}\right\rvert\times100$ |
| Scarborough (n sig figs) | $\varepsilon_s\% = 0.5\times10^{2-n}$ |
