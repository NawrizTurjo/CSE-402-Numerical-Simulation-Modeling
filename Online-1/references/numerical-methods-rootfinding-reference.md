# CSE 401 — Numerical Analysis, Simulation & Modeling
## Reference Sheet: Root Finding — Bisection, False-Position & Newton-Raphson (Lab 1, cont'd)

Source: "Root Finding: Bisection, False-Position & Newton-Raphson Method" lecture slides
(Nafis Tahmid, CSE, BUET), cross-checked against the observed exam/lab conventions already
captured in `numerical-methods-course-context.md` and the sample question bank
(A1 / B1 / B2 / C1 / C2).

This file continues `numerical-methods-errors-reference.md` into the syllabus's next segment
for this same lab assignment: **root finding — bisection method, false position method,
Newton-Raphson method**. Stopping-criterion machinery (εa, εs, significant digits) is reused
verbatim from that file; see Section 2 here for the condensed restatement.

**Scope note (read this first):** the official syllabus line for this lab also lists
*Bairstow's method*, but no slide deck covering it has been provided yet. This file
intentionally excludes Bairstow's method — do not generate code or theory for it from general
knowledge. Wait for the actual slide deck or explicit confirmation before extending this file
(see Section 13).

Purpose: feed this to an AI coding agent as ground truth for formulas, conventions, and
expected output formats, so generated code matches what the course (and the sample exam
question bank) expects — without re-deriving everything from scratch each session.

---

## 1. Root-Finding Fundamentals

### 1.1 The problem
Find x such that f(x) = 0. Not all equations have a tidy closed form — transcendental
equations especially decline to cooperate — so approximate iteratively until the error is
small enough.

### 1.2 The sign-change (Intermediate Value) theorem
If f is a real, continuous function and f(xl)·f(xu) < 0, then f(x) = 0 has **at least one**
root between xl and xu.

Nuances that recur as theory-question material (see Section 11):
- f(xl)f(xu) > 0 → there **may or may not** be a root inside — the test is silent, not
  negative. Classic counter-shape: a hump that dips below zero and returns above it entirely
  between xl and xu, so both endpoints share a sign despite two roots hiding inside.
- f(xl)f(xu) < 0 → **at least one** root guaranteed, but there could be more than one
  (any odd count ≥ 1).
- The sign-change test is **sufficient for existence, not exhaustive for counting**. It tells
  you a root exists; it stays quiet about how many.
- A root that merely *touches* the axis without crossing (e.g. f(x) = x² at x = 0) produces
  **no sign change at all** — no xl, xu pair satisfies f(xl)f(xu) < 0, so bracketing methods
  can't even get started on it.

### 1.3 Why the coarse scan matters
Because a single wide interval can hide an even number of roots, or fail to bracket at all when
a root only touches zero, course convention (Section 6 below, and course-context.md Section 3)
is: **never run bisection/false-position once over an entire wide domain.** Always scan first
with a fixed step size (commonly 0.1) to enumerate every disjoint sign-change sub-interval, then
run the chosen method on each sub-interval separately.

---

## 2. Stopping / Convergence Criteria (condensed from Lab 1)

### 2.1 Relative approximate error
Same definition as Lab 1, now applied to successive root estimates instead of series/derivative
values:
```
εa = (x_new − x_old) / x_new
εa(%) = εa × 100
```
Undefined on the very first iteration (no previous estimate exists) — print that row's εa as
blank/`—`, not zero. Compare `abs(eps_a_percent)` against the tolerance; keep the sign when
just reporting the value.

### 2.2 Default threshold
Exactly as in the Lab 1 exam bank, root-finding problems here almost always specify
**`|εa| ≤ 0.0001%`**. Build every solver with this as a configurable default, not a hardcoded
literal.

### 2.3 Significant digits (unchanged relation)
```
|εa(%)| ≤ 0.5 × 10^(2−m)  ⟺  estimate correct to at least m significant digits
```
Full derivation lives in the Lab 1 file (Sections 2.2/3.3) — identical relation, just now
applied to a converging root estimate rather than a converging series sum.

---

## 3. Bisection Method

### 3.1 Algorithm
1. Choose xl, xu with f(xl)·f(xu) < 0 (sign change confirmed).
2. Estimate the root: xm = (xl + xu) / 2.
3. Decide the new bracket from the sign of f(xl)·f(xm):
   - < 0 → root in [xl, xm], so set xu = xm.
   - > 0 → root in [xm, xu], so set xl = xm.
   - = 0 → xm is the exact root — stop.
4. Compute the new xm and εa against the previous xm.
5. If |εa| > εs, repeat from step 3; else stop. Always cap the iteration count as a backstop.

### 3.2 Code pattern
```python
def bisection(f, xl, xu, tol=0.0001, max_iter=200):
    """
    Returns (root_estimate, rows). rows: (iter, xl, xu, xm, f_xm, eps_a_percent),
    eps_a_percent is None on iteration 1.
    Naming note: course-context.md's table convention calls this column "Xr" for
    consistency with false-position; the slides call it "xm" (midpoint) — same quantity.
    """
    f_xl = f(xl)
    xm_prev = None
    rows = []
    for i in range(1, max_iter + 1):
        xm = (xl + xu) / 2
        f_xm = f(xm)
        eps_a = abs((xm - xm_prev) / xm) * 100 if xm_prev is not None else None
        rows.append((i, xl, xu, xm, f_xm, eps_a))
        if f_xm == 0:
            break
        if f_xl * f_xm < 0:
            xu = xm
        else:
            xl, f_xl = xm, f_xm
        xm_prev = xm
        if eps_a is not None and eps_a <= tol:
            break
    return xm, rows
```

### 3.3 Advantages
- **Always convergent** — because the method brackets the root, convergence is guaranteed.
- **Predictable error** — the interval halves every iteration, so the number of iterations
  needed for a given tolerance can be computed in advance.

### 3.4 Drawbacks
- **Slow convergence** — linear, purely interval-halving; a good guess earns nothing.
- **Wastes proximity information** — even if one endpoint sits right on the root, bisection
  still marches to the midpoint; only the *sign* matters, never how close a value is to zero.
- **Roots that merely touch the axis** (f(x) = x²) admit no xl, xu with f(xl)f(xu) < 0 — the
  method can't even start.
- **Singularities that flip sign are indistinguishable from roots at bracket time.**
  Example: f(x) = 1/x with xl = −2, xu = 3 satisfies f(xl)f(xu) < 0, yet there is **no root** —
  the function is discontinuous and bisection converges straight at the singularity x = 0.
  This is exactly why a residual check (f(xr) ≈ 0) after convergence is mandatory, not optional
  — see Section 8.

---

## 4. False-Position Method

### 4.1 Idea
Same bracketing idea as bisection, but instead of blindly bisecting, draws a secant line from
(xl, f(xl)) to (xu, f(xu)) and takes where that line crosses the x-axis as the new estimate —
weighting the estimate toward whichever endpoint's f-value sits closer to zero.

Derived (via similar triangles) as:
```
xr = [xu·f(xl) − xl·f(xu)] / [f(xl) − f(xu)]
```
Two equivalent "increment" forms of the same formula (useful for numerical-stability framing,
but the form above is cleanest to implement directly):
```
xr = xu − f(xu)·(xu − xl) / [f(xu) − f(xl)]
xr = xl − f(xl)·(xl − xu) / [f(xl) − f(xu)]
```

### 4.2 Algorithm
1. Choose xl, xu with f(xl)·f(xu) < 0.
2. Estimate xr via the formula above.
3. Decide the new bracket from the sign of f(xl)·f(xr) — identical logic to bisection step 3,
   with xr in place of xm.
4. Compute εa from consecutive xr values.
5. Repeat until |εa| ≤ εs (or the iteration cap is hit). Only the formula in steps 2/4 differs
   from bisection — everything else is the same shape.

### 4.3 Code pattern
```python
def false_position(f, xl, xu, tol=0.0001, max_iter=500):
    """Returns (root_estimate, rows). rows: (iter, xl, xu, xr, f_xr, eps_a_percent)."""
    f_xl, f_xu = f(xl), f(xu)
    xr_prev = None
    rows = []
    for i in range(1, max_iter + 1):
        xr = (xu * f_xl - xl * f_xu) / (f_xl - f_xu)
        f_xr = f(xr)
        eps_a = abs((xr - xr_prev) / xr) * 100 if xr_prev is not None else None
        rows.append((i, xl, xu, xr, f_xr, eps_a))
        if f_xr == 0:
            break
        if f_xl * f_xr < 0:
            xu, f_xu = xr, f_xr
        else:
            xl, f_xl = xr, f_xr
        xr_prev = xr
        if eps_a is not None and eps_a <= tol:
            break
    return xr, rows
```

### 4.4 Endpoint-stall instrumentation (needed for comparison deliverables — Section 9)
False-position frequently leaves one endpoint "frozen" for many iterations while the other does
all the moving — course-context.md explicitly requires reporting how many times Xl vs Xu update
when comparing against bisection. Instrument the loop:

```python
def false_position_instrumented(f, xl, xu, tol=0.0001, max_iter=500):
    f_xl, f_xu = f(xl), f(xu)
    xr_prev = None
    xl_updates = xu_updates = 0
    rows = []
    for i in range(1, max_iter + 1):
        xr = (xu * f_xl - xl * f_xu) / (f_xl - f_xu)
        f_xr = f(xr)
        eps_a = abs((xr - xr_prev) / xr) * 100 if xr_prev is not None else None
        rows.append((i, xl, xu, xr, f_xr, eps_a))
        if f_xr == 0:
            break
        if f_xl * f_xr < 0:
            xu, f_xu = xr, f_xr
            xu_updates += 1
        else:
            xl, f_xl = xr, f_xr
            xl_updates += 1
        xr_prev = xr
        if eps_a is not None and eps_a <= tol:
            break
    return xr, rows, xl_updates, xu_updates
```

### 4.5 Advantages / Drawbacks
- Usually converges faster than bisection, since it uses the *shape* of f near each endpoint,
  not just its sign.
- **Faster convergence is not guaranteed.** It can stall badly when f is strongly curved near
  one endpoint — that endpoint then never updates, and error shrinks slowly from the other side
  only. Documented slide benchmark: f(x) = ln(x) on [10⁻⁴, 10⁴], tol = 0.0001% → bisection
  converged in 34 iterations vs. false-position's 126 (bisection ≈2.7× faster in that specific
  case). **Do not assume false-position always wins** — the comparison deliverable in Section 9
  exists precisely because the outcome is case-dependent.

---

## 5. Newton-Raphson Method

### 5.1 Idea
An **open** method — needs only a single initial guess, no bracket, and no confirmed sign
change. Derived from the tangent line at xi: its slope is f'(xi), and the point where that
tangent crosses the x-axis is the next estimate.
```
xi+1 = xi − f(xi) / f'(xi)
```

### 5.2 Algorithm
1. Have f'(x) available (symbolic or numeric).
2. xi+1 = xi − f(xi)/f'(xi).
3. εa = |(xi+1 − xi)/xi+1| × 100.
4. Repeat from step 2 until |εa| ≤ εs, watching the iteration cap.

### 5.3 Code pattern
```python
def newton_raphson(f, fprime, x0, tol=0.0001, max_iter=200, flat_eps=1e-14):
    """
    Returns (root_estimate, rows). rows columns match the A1-style order:
    Iter | Vi | f(Vi) | f'(Vi) | Vi+1 | eps_a_percent
    """
    xi = x0
    rows = []
    for i in range(1, max_iter + 1):
        fi = f(xi)
        fpi = fprime(xi)
        if abs(fpi) < flat_eps:
            # Flat tangent -> division by zero. Terminate this run gracefully.
            rows.append((i, xi, fi, fpi, None, None))
            break
        x_next = xi - fi / fpi
        eps_a = abs((x_next - xi) / x_next) * 100
        rows.append((i, xi, fi, fpi, x_next, eps_a))
        xi = x_next
        if eps_a <= tol:
            break
    return xi, rows
```

### 5.4 Choosing x0
Avoid starting exactly where f'(x0) = 0 — a flat tangent never meets the axis, and the update
divides by zero. Slide example: for f(x) = x³ − 0.165x² + 3.993×10⁻⁴ (the floating-ball
problem), f'(x) = 3x(x − 0.11), which vanishes at exactly the two interval endpoints 0 and
0.11 — pick an interior point instead (the slides use x0 = 0.05).

### 5.5 Drawbacks (all must be *detected in code*, not just known in theory)
1. **Divergence near inflection points.** Where f''(x) changes sign, f' goes small/flat nearby,
   so the tangent jump can overshoot wildly before (maybe) crawling back. Slide example:
   f(x) = (x−1)³ + 0.512, x0 = 5.0 — the estimate lands near the inflection point x = 1 at
   iteration 5, panics to −30.119, and takes ~13 more iterations to recover to x ≈ 0.2.
2. **Division by (near) zero.** f'(xi) vanishing, or nearly vanishing, sends xi+1 far from
   anything useful. Slide example: f(x) = x³ − 0.03x² + 2.4×10⁻⁶, where f' vanishes at x = 0
   and x = 0.02 — starting at x0 = 0.01999 (deliberately dodging the exact zero) still launches
   the iterates off to nowhere; after 9 iterations it has not converged. Detect
   `abs(f'(xi))` below a small threshold and abort that run with a clear message rather than
   letting inf/nan propagate silently.
3. **Oscillation with no real root nearby.** f(x) = x² + 2 has no real roots at all — the
   iterates bounce around the minimum indefinitely and eventually drift toward divergence.
   A max-iteration cap is therefore **not optional**; without one, nothing stops this on its own.
4. **Root jumping.** With multiple roots (e.g. f(x) = sin x), a single guess can leap over the
   intended nearby root entirely and converge to a distant, unrelated one — with no warning.
   Slide example: x0 = 2.4π ≈ 7.5398, intending to land near 2π ≈ 6.283, instead converges to
   x = 0 within 5 iterations. Convergence alone never confirms the method found the root you
   wanted.

### 5.6 Practical implication for code
Every Newton-Raphson implementation in this course needs: (a) a flat-derivative check *before*
dividing, (b) a hard iteration cap, and (c) reporting f(root) at the end so the caller can
sanity-check it's actually ≈ 0 — the same "check the residual" habit required for
bisection/false-position (Section 8).

---

## 6. Coarse Scanning Convention (shared setup for bisection / false-position)

Before running either bracketing method on a stated domain, scan with a fixed step (commonly
0.1) to build two disjoint candidate lists:

- **Direct root candidates** — scan points where |f(x)| is already ≈ 0 (sample-problem
  threshold: ≤ 1e-8). No bracketing method needed; report directly.
- **Interval candidates** — consecutive scan points where f changes sign. Bracket each one
  separately for bisection/false-position — never bisect once over the whole domain (Section
  1.3 explains why).

```python
def coarse_scan(f, a, b, step=0.1, direct_tol=1e-8):
    """Returns (direct_candidates, interval_candidates)."""
    direct_candidates, interval_candidates = [], []
    xs, x = [], a
    while x <= b + 1e-12:
        xs.append(x)
        x += step
    fs = []
    for x in xs:
        try:
            fx = f(x)
        except (ZeroDivisionError, ValueError):
            fx = None  # undefined point in domain, e.g. an asymptote
        fs.append(fx)
        if fx is not None and abs(fx) <= direct_tol:
            direct_candidates.append(x)
    for i in range(len(xs) - 1):
        f0, f1 = fs[i], fs[i + 1]
        if f0 is None or f1 is None:
            continue
        if f0 * f1 < 0:
            interval_candidates.append((xs[i], xs[i + 1]))
    return direct_candidates, interval_candidates
```

Note: Newton-Raphson doesn't consume this scan the same way (it needs one guess, not a
bracket) — the "scan, then refine, then print a table" shape is specifically the
bisection/false-position pattern. A coarse scan can still inform a reasonable x0 for NR when a
problem gives a domain instead of an explicit initial guess.

---

## 7. Undefined-Value / Singularity Handling

Functions in this course's problem bank are deliberately awkward — expect a vertical asymptote
or other domain break inside the search interval. Two distinct failure modes:

1. **Sign change caused by a singularity, not a root.** Bisection's precondition (continuity)
   is violated; f(xl)f(xu) < 0 can be satisfied purely because f blows up and flips sign at a
   discontinuity (the 1/x example in Section 3.4). The method will happily "converge" to the
   singularity. This is exactly why every accepted root must be verified against its residual
   actually being ≈ 0 (Section 8) — convergence of the iteration is not the same claim as
   "this is a root."
2. **Runtime domain errors mid-iteration.** Division by zero, log of a non-positive number,
   etc., encountered while evaluating f at a candidate xm/xr *during* iteration (not just at
   scan time). Wrap each evaluation and terminate that specific bracket's run gracefully — log
   it, move to the next candidate interval — rather than crashing the whole batch.

```python
def safe_eval(f, x):
    try:
        return f(x), None
    except (ZeroDivisionError, ValueError, OverflowError) as e:
        return None, str(e)
```

---

## 8. Acceptance Filtering & Root Reporting

Convergence (|εa| ≤ tol) is necessary but not sufficient to call an estimate an accepted root.
Course convention:
- Report **type**: `"direct"` (found by the coarse scan, |f(x)| already below tolerance) vs.
  `"interval"` (found via bracketing + iteration).
- Report the **residual** f(estimate) at the final value — the sanity check described in
  Sections 3.4, 4.x, 5.6 and 7.
- Apply any **problem-specific secondary filter** the prompt states (e.g. one sample problem
  required |xr| ≤ 1e-6) — read the specific task rather than assuming convergence alone
  suffices.
- Report an explicit **accept/reject verdict** per candidate, not a bare list of numbers.

```python
def classify_and_verify(candidate, f, kind, extra_filter=None, residual_tol=1e-6):
    """
    candidate: the estimated x value. kind: 'direct' or 'interval'.
    extra_filter: optional callable(x) -> bool, problem-specific acceptance rule.
    """
    fx, err = safe_eval(f, candidate)
    accepted = fx is not None and abs(fx) <= residual_tol
    if accepted and extra_filter is not None:
        accepted = extra_filter(candidate)
    return {"type": kind, "x": candidate, "f_x": fx, "error": err, "accepted": accepted}
```

---

## 9. Method Comparison Deliverables

When two methods solve the same problem (typically bisection vs. false-position on the same
bracket), the expected final deliverable is **one comparison table**, not two separate reports.
Recurring column set: estimated root | iteration count | f(xr) at convergence | Xl update
count | Xu update count.

```python
def comparison_row(method_name, root, rows, f_root, xl_updates=None, xu_updates=None):
    return {
        "method": method_name,
        "root": root,
        "iterations": len(rows),
        "f_root_at_convergence": f_root,
        "xl_updates": xl_updates,
        "xu_updates": xu_updates,
    }
```

---

## 10. Plotting Conventions (extends Lab 1 Section 7)

Base conventions carry over unchanged (fine `numpy.linspace` grid with N ≥ 500–1000; axis
labels, title, grid on every plot; consistent `savefig` then `show`). Root-finding adds:

- **Always mark y = 0** — `plt.axhline(0, color='k', linewidth=0.8)`. This is now mandatory
  rather than a habit-forming suggestion, since the plot's whole purpose is locating roots
  visually.
- **Log-scale x-axis** when the domain spans many orders of magnitude — e.g. the ln(x)
  comparison problem uses [10⁻⁴, 10⁴]. Use `plt.xscale('log')` / `plt.semilogx`; a linear scale
  would compress small-x behavior into invisibility.

---

## 11. Theoretical Short-Answer Prompt Bank (recurring flavors)

These travel *alongside* the code as short (3–4 sentence) written explanations, not code
comments. Patterns observed across the sample question bank:

- Why doesn't the absence of a sign change over an interval guarantee there's no root inside?
  (Even number of roots; or a root where f touches zero without crossing — Section 1.2.)
- Why can a sign-change interval sometimes contain no genuine root when checked more closely,
  and why is checking the residual f(xr) still necessary even after successful bracketing and
  convergence? (Singularity flipping sign, not a true root — Sections 3.4, 7.)
- Which of bisection/false-position converges faster in general, and why isn't false-position's
  faster convergence guaranteed? (Endpoint stall when f is strongly curved near one side, so
  that endpoint never updates — Sections 4.4, 4.5.)
- Why would running bisection/false-position once over a whole wide interval — instead of once
  per sign-change sub-interval — be problematic? (Can miss multiple roots entirely, or fail to
  bracket correctly when there's an even number of internal sign changes — Section 1.3.)

Answer at the "explain the mechanism" level, not "restate the definition" — the grading pattern
in this course rewards showing *why*, matching the drawback sections above (3.4, 4.5, 5.5).

---

## 12. Quick-Reference Formula Sheet

| Quantity | Formula |
|---|---|
| Bisection midpoint | xm = (xl + xu) / 2 |
| False-position estimate | xr = [xu·f(xl) − xl·f(xu)] / [f(xl) − f(xu)] |
| Newton-Raphson update | xi+1 = xi − f(xi) / f'(xi) |
| Relative approximate error | εa = (x_new − x_old) / x_new (×100 for %) |
| Default stop threshold | \|εa\| ≤ 0.0001% |
| Bracket test | f(xl)·f(xu) < 0 ⟹ ≥ 1 root inside (not exactly-one) |
| NR flat-tangent hazard | f'(xi) ≈ 0 ⟹ division by zero — never start there |

---

## 13. Out of Scope (for now)

**Bairstow's method** appears on the official syllabus line for this lab ("...Newton-Raphson
method, Bairstow's method...") but no slide deck covering it has been provided. Do not generate
Bairstow's-method code or theory from general knowledge — wait for the actual slide deck or an
explicit go-ahead before extending this file to include it, consistent with the
"stay within taught material" rule already established.

Also still out of scope, per the existing course-wide rule: secant method, Brent's method,
trapezoidal/Simpson's integration — unless a later slide or explicit request calls for them.

---
*Companion files: `numerical-methods-errors-reference.md` (Lab 1 — approximations, round-off &
truncation errors, stopping-criteria derivation) and `numerical-methods-course-context.md`
(original handoff notes, now extended by this file for the root-finding topic). Load all three
at the start of the next chat, alongside whichever slide deck introduces Bairstow's method once
it's available.*
