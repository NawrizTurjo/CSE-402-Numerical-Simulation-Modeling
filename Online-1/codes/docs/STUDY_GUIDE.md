# CSE-402 Numerical Methods — Complete Study Guide
## Online Lab Assignment 1: Root Finding Methods

> **Exam Date:** Monday | **Format:** Coding or Handwritten | **Resources Allowed:** Your own practice codes (no internet)

---

## 📋 Table of Contents

1. [Error & Approximation Theory](#1-error--approximation-theory)
2. [Matplotlib Quick Reference](#2-matplotlib-quick-reference-exam-critical)
3. [Bisection Method](#3-bisection-method)
4. [False Position Method](#4-false-position-method-regula-falsi)
5. [Newton-Raphson Method](#5-newton-raphson-method)
6. [Bairstow's Method](#6-bairstows-method)
7. [Failure Modes & Gotchas](#7-failure-modes--gotchas)
8. [Previous Year Question Analysis](#8-previous-year-question-analysis)

---

## 1. Error & Approximation Theory

### Three Types of Error

| Error Type | Formula | When Used |
|-----------|---------|-----------|
| **True Error** | $E_t = \text{true} - \text{approx}$ | When you know the exact answer |
| **True Relative Error** | $\varepsilon_t = \left\|\frac{\text{true}-\text{approx}}{\text{true}}\right\| \times 100\%$ | Performance measure |
| **Approximate Relative Error** | $\varepsilon_a = \left\|\frac{x_{new}-x_{old}}{x_{new}}\right\| \times 100\%$ | **This is your STOPPING criterion in exams!** |

> ⚠️ **THE DIVISION-BY-ZERO TRAP:** If the root estimate $x_{new}$ converges exactly to $0$, or hits $0$ during intermediate oscillation, the standard $\varepsilon_a$ formula causes a `ZeroDivisionError` crash. 
> **The Safe Fix:** Use a conditional check in your code:
> ```python
> if abs(x_new) < 1e-12:
>     ea = abs(x_new - x_old) * 100.0  # Fallback to absolute difference
> else:
>     ea = abs((x_new - x_old) / x_new) * 100.0
> ```

### Scarborough Criterion (Significant Figures → Tolerance)

$$\varepsilon_s = 0.5 \times 10^{2-n} \quad (\%)$$

| Sig. Figs (n) | Required Tolerance $\varepsilon_s$ |
|:---:|:---:|
| 1 | 5% |
| 2 | 0.5% |
| 3 | 0.05% |
| 4 | 0.005% |
| **6** | **0.00005%** |

### Round-off vs Truncation Error

| Feature | Round-off Error | Truncation Error |
|---------|----------------|-----------------|
| **Cause** | Finite binary representation (64-bit limits) | Cutting off infinite mathematical processes early |
| **Example** | $1/3$ stored as $0.33333$ | Taylor series truncated to $n$ terms |
| **Fixed by** | Higher precision representation | Retaining more terms / smaller step size |
| **Machine ε** | $\approx 2.22 \times 10^{-16}$ for float64 | — |

### The Error Tradeoff Curve (V-Shape)

When approximating derivatives using finite differences (e.g., $f'(x) \approx \frac{f(x+h)-f(x)}{h}$):
* **Large step size $h$:** Truncation error dominates because the Taylor series approximation is inaccurate.
* **Tiny step size $h$ (e.g., $< 10^{-8}$):** Round-off error dominates. Subtracting two very close numbers ($f(x+h) - f(x)$) causes **subtractive cancellation**, losing significant digits, which is then amplified by dividing by a tiny $h$.
* **The Result:** Plotting total error vs. step size $h$ on a log-log plot yields a classic **V-shaped curve**. The minimum of this curve represents the optimal step size $h \approx 10^{-8}$ for standard double-precision calculations.

---

## 2. Matplotlib Quick Reference (Exam-Critical!)

> ⚠️ **Teachers explicitly deduce marks on plot quality.** Always include: title, xlabel, ylabel, grid, legend, axhline(0).

### Template 1 — Standard Plot (Most Common)

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(a, b, 1000)
y = f(x)                         # f must be vectorized (use np.log, np.sin, etc.)

plt.figure(figsize=(8, 5))
plt.plot(x, y, color='steelblue', linewidth=2, label='f(x)')
plt.axhline(0, color='black', linewidth=1, linestyle='--')   # y=0 line
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of f(x) on [a, b]')
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('graph.png', dpi=150)
plt.show()
```

### Template 2 — Log Scale (Previous Year: ln(x))

```python
x = np.logspace(-4, 4, 1000)    # 10^-4 to 10^4, log-spaced
plt.plot(x, np.log(x), ...)
plt.xscale('log')               # ← THIS is the key line
plt.grid(True, which='both')    # shows minor grid ticks too
```

### Template 3 — Multiple Roots with Brackets

```python
for xl, xu in intervals:
    plt.axvspan(xl, xu, color='orange', alpha=0.3, label=f'[{xl:.1f},{xu:.1f}]')

plt.scatter(roots, [0]*len(roots), color='red', marker='x', s=150,
            linewidths=2.5, zorder=5, label='Roots')
```

### Template 4 — Convergence Plot (Semi-log)

```python
errors = [ea1, ea2, ea3, ...]     # collect ε_a from each iteration
plt.semilogy(range(1, len(errors)+1), errors, 'o-', color='darkred')
plt.xlabel('Iteration')
plt.ylabel('ε_a (%) [log scale]')
```

### Key Function Translations (Math → NumPy)

| Math | Python |
|------|--------|
| $\ln(x)$ | `np.log(x)` |
| $\log_{10}(x)$ | `np.log10(x)` |
| $e^x$ | `np.exp(x)` |
| $\sin(x)$ | `np.sin(x)` |
| $\sqrt[3]{x}$ | `np.cbrt(x)` (**NOT `x**(1/3)`!**) |
| $x^{2/3}$ | `x**(2/3)` |
| $\pi$ | `np.pi` or `math.pi` |

> ⚠️ **Critical:** `np.cbrt(x)` works for negative x. `x**(1/3)` gives complex numbers for negative x in Python, breaking your plot!

---

## 3. Bisection Method

### Core Idea
Bracket the root between $[x_l, x_u]$ where $f(x_l) \cdot f(x_u) < 0$. Repeatedly halve the interval.

### Formula

$$x_r = \frac{x_l + x_u}{2}$$

$$\varepsilon_a = \left|\frac{x_r^{new} - x_r^{old}}{x_r^{new}}\right| \times 100\%$$

### Update Rule

```
If f(xl) · f(xr) < 0:  xu = xr   (root in LEFT half)
Else:                   xl = xr   (root in RIGHT half)
```

### Full Algorithm (Python)

```python
def bisection(f, xl, xu, tol=0.0001, max_iter=200):
    xr_old = None
    for i in range(1, max_iter+1):
        xr  = (xl + xu) / 2.0
        fxr = f(xr)
        ea  = abs((xr - xr_old)/xr)*100 if xr_old else 100.0
        # print table row here
        if xr_old and ea <= tol: break
        if f(xl) * fxr < 0: xu = xr
        else:                xl = xr
        xr_old = xr
    return xr
```

### Properties

| Property | Value |
|----------|-------|
| Convergence rate | Linear (1 bit per iteration) |
| Guaranteed? | ✅ Yes — always converges if bracket is valid |
| Requires derivative? | ❌ No |
| Finds multiple roots? | ❌ No — only 1 per bracket |
| Typical iterations for 5 SF | ~17 |

---

## 4. False Position Method (Regula Falsi)

### Core Idea
Instead of the midpoint, use the x-intercept of the **secant line** connecting $(x_l, f(x_l))$ and $(x_u, f(x_u))$.

### Formula

$$x_r = x_u - \frac{f(x_u)(x_l - x_u)}{f(x_l) - f(x_u)} = \frac{x_u \cdot f(x_l) - x_l \cdot f(x_u)}{f(x_l) - f(x_u)}$$

> Both forms are equivalent. Memorize either.

### Update Rule
**Identical to bisection** — same sign test, same bracket update.

### STAGNATION PROBLEM (Major Exam Topic!)

For **concave** functions (like $\ln(x)$ over a huge range):
- The secant always intercepts on the **same side** of the root.
- One bracket endpoint **NEVER moves** (remains stagnant).
- False position can be **SLOWER than bisection** in this case!

**The Fix — The Illinois Method:**
To break stagnation, when the update occurs repeatedly on one side, the Illinois method **halves** the value of $f(x)$ at the inactive boundary (the stagnant bound). This decreases the slope of the secant line, forcing it to overshoot and switch sides on the next iteration.

### Comparison: Bisection vs False Position vs Illinois

| Property | Bisection | False Position | Illinois FP |
|----------|-----------|----------------|-------------|
| Formula | Midpoint | Secant x-intercept | Secant with halved stagnant bound |
| Convergence | Linear | Linear (can stagnate) | Linear (stagnation-free) |
| Stagnation risk | None | Yes, for curved functions | None |
| Speed | Predictable | Variable — can be very slow | Fast and robust |

---

## 5. Newton-Raphson Method

### Core Idea
Follow the **tangent line** at $x_i$ down to the x-axis. That x-intercept is $x_{i+1}$.

### Formula (With Multiplicity Modifier)

$$\boxed{x_{i+1} = x_i - m \frac{f(x_i)}{f'(x_i)}}$$

$$\varepsilon_a = \left|\frac{x_{i+1} - x_i}{x_{i+1}}\right| \times 100\%$$

> * **Multiplicity modifier $m$:** For simple roots, $m = 1$. For a root with multiplicity $m > 1$ (e.g., $f(x) = (x-2)^3$ where root 2 has multiplicity 3), setting $m$ to the multiplicity restores the **quadratic convergence speed**. Without $m$, Newton-Raphson degrades to slow linear convergence.
> * **Oscillation/Cycle Detector:** If an initial guess leads to a cycle (such as bouncing $0 \to 1 \to 0 \to 1$), we compare the new $x_{i+1}$ value against the list of previously visited values. If a match is found within a small tolerance, an oscillation trap is flagged.

---

### Exam Table Format

| Iter | $x_i$ | $f(x_i)$ | $f'(x_i)$ | $x_{i+1}$ | $\varepsilon_a\%$ |
|:----:|:----:|:--------:|:---------:|:--------:|:---------:|
| 1 | 0.650000 | ... | ... | ... | --- |
| 2 | ... | ... | ... | ... | ... |

### Deriving $f'(x)$ — Exam Quick Reference

| $f(x)$ | $f'(x)$ |
|--------|---------|
| $ax^n$ | $nax^{n-1}$ |
| $e^{u(x)}$ | $u'(x) \cdot e^{u(x)}$ |
| $\ln(u(x))$ | $\frac{u'(x)}{u(x)}$ |
| $\sin(u(x))$ | $u'(x) \cdot \cos(u(x))$ |
| $\cos(u(x))$ | $-u'(x) \cdot \sin(u(x))$ |

### Properties

| Property | Value |
|----------|-------|
| Convergence rate | **Quadratic** (near root) |
| Requires derivative? | ✅ Yes — must derive $f'(x)$ |
| Guaranteed? | ❌ No — several failure modes |
| Speed | ~3-5 iterations for most problems |

### Convergence Speed Comparison

```
Bisection:    34 iterations for f(x)=ln(x) to 0.0001%
Newton-R:     3-5 iterations for most smooth functions
```

---

## 6. Bairstow's Method

### Purpose
Find **ALL** roots of a polynomial (including complex conjugate pairs) using only real arithmetic.

### Polynomial Convention
**Highest power first:**
$$a_0 x^n + a_1 x^{n-1} + \cdots + a_n \quad \rightarrow \quad [a_0, a_1, \ldots, a_n]$$

### Algorithm Overview

1. **Guess** quadratic factor: $x^2 - rx - s$, starting with $r_0, s_0$

2. **First Synthetic Division** (b-array):
   $$b_0 = a_0, \quad b_1 = a_1 + r \cdot b_0$$
   $$b_i = a_i + r \cdot b_{i-1} + s \cdot b_{i-2} \quad (i = 2, \ldots, n)$$

3. **Second Synthetic Division** (c-array, divide b by same quadratic):
   $$c_0 = b_0, \quad c_1 = b_1 + r \cdot c_0$$
   $$c_i = b_i + r \cdot c_{i-1} + s \cdot c_{i-2} \quad (i = 2, \ldots, n-1)$$

4. **Solve 2×2 system** for corrections:
   $$D = c_{n-2}^2 - c_{n-3} \cdot c_{n-1}$$
   $$\Delta r = \frac{-b_{n-1} \cdot c_{n-2} + b_n \cdot c_{n-3}}{D}$$
   $$\Delta s = \frac{-b_n \cdot c_{n-2} + b_{n-1} \cdot c_{n-1}}{D}$$

5. **Update:** $r \leftarrow r + \Delta r$, $s \leftarrow s + \Delta s$

6. **Error check:**
   $$\varepsilon_a(r) = \left|\frac{\Delta r}{r}\right| \times 100\% \leq \text{tol}, \quad \varepsilon_a(s) = \left|\frac{\Delta s}{s}\right| \times 100\% \leq \text{tol}$$
   Stop when **BOTH** conditions are satisfied.

7. **Extract roots** from $x^2 - rx - s = 0$:
   $$\text{disc} = r^2 + 4s$$
   - If $\text{disc} \geq 0$: real roots $\frac{r \pm \sqrt{\text{disc}}}{2}$
   - If $\text{disc} < 0$: complex pair $\frac{r}{2} \pm i\frac{\sqrt{|\text{disc}|}}{2}$

8. **Deflate:** quotient = $b[0 \ldots n-2]$ (drop last 2 residuals). Repeat with smaller polynomial.

### Exam Table Format

```
Degree 4 polynomial:
iter     r          s         Δr         Δs       ε_a(r)%   ε_a(s)%
  1    2.50000    1.50000    0.30000    0.20000    12.0000    13.3333
  2    2.48000    1.52000    0.02000    0.02000     0.8065     1.3158
  3    ...
```

---

## 7. Failure Modes & Gotchas

### Newton-Raphson Failures (MUST KNOW!)

| Failure | Trigger | What Happens | Fix |
|---------|---------|-------------|-----|
| **Oscillation (2-cycle)** | $f(x) = x^3 - 2x + 2$, $x_0 = 0$ | Bounces $0 \to 1 \to 0 \to 1 \to \ldots$ forever | Choose $x_0 = -1.5$ (inside sign-change interval) |
| **Division by Zero** | $f'(x_0) = 0$ (local min/max) | Formula undefined | Shift initial guess slightly |
| **Vertical Tangent Divergence** | $f(x) = \sqrt[3]{x}$, any $x_0 \neq 0$ | $x_{n+1} = -2x_n$, doubles every step | Use Bisection instead |
| **Overshooting** | $f(x) = \arctan(x)$, $x_0 = 1.5$ | Shoots to $\pm\infty$ | Choose $x_0$ close to root |
| **Slow Multiple Roots** | $f(x) = (x-2)^3$ | Loses quadratic convergence | Use modified formula: $x_{n+1} = x_n - m \cdot \frac{f(x_n)}{f'(x_n)}$ |

### Python-Specific Traps

| Trap | Wrong | Correct |
|------|-------|---------|
| Cube root of negative | `x**(1/3)` → complex! | `np.cbrt(x)` |
| Single-point root | `f(x)**(1/3)` → broken | Always verify with `np.cbrt` |
| Log of negative | `np.log(-x)` → NaN/error | Check domain first |

### False Position Stagnation

**Why it happens:** If $f(x)$ is **strictly concave** or **strictly convex**, the secant always intercepts on the same side. One endpoint never updates → slow convergence.

**The fix:** Modified False Position methods (Illinois algorithm) — but exams just ask you to *explain* why it's slow.

---

## 8. Previous Year Question Analysis

### Section A1 Questions (Root Finding)

**Problem 1: $f(x) = \ln(x)$, interval $[10^{-4}, 10^4]$**
- Bisection: 34 iterations, xl updated 11×, xu updated 22×
- False Position: 126 iterations, xl updated 0×! (stagnation)
- **Key insight:** False Position was SLOWER because ln(x) is concave and the interval is huge

**Problem 2: Multi-root scan — $f(x) = 0.6\ln(x+1) - C\sin(1.7x) - 0.08x^2 - 0.08$**
- Sign changes at: $\approx [1.6, 1.7]$ and $\approx [3.5, 3.6]$
- **Key insight:** Cannot run bisection once on full $[0,10]$ — misses roots!

**Problem 3: Auto-scan + False Position on first bracket**
- Template: find brackets → FP on `intervals[0]`

### Newton-Raphson Exam Patterns

**Diode Equation:**
$$f(V) = 10^{-12}(e^{V/(nV_T)} - 1) + \frac{V}{R} - I_L = 0$$
$$f'(V) = \frac{10^{-12}}{nV_T} e^{V/(nV_T)} + \frac{1}{R}$$

**Dipstick (Spherical Cap):**
$$f(h) = \frac{\pi h^2(3r - h)}{3} - V = 0$$
$$f'(h) = \pi(2rh - h^2) = \pi h(2r - h)$$

### Bairstow Exam Patterns

- Given polynomial → extract all roots → show iteration table for r, s
- Always verify: plug roots back into polynomial → |f(root)| ≈ 0

### Recent Previous Batch PDF Questions

**1. Spherical Tank Dipstick Design (online_A2.pdf)**
* **Equation:** $f(h) = \pi h^3 - 12\pi h^2 + 15 = 0$ (derived from spherical cap volume with $r=4, V=5$).
* **Derivative:** $f'(h) = 3\pi h^2 - 24\pi h$.
* **Initial Guess Selection:** Since the physical height $h \in [0, 8]$, we search for sign changes. $f(0) = 15 > 0$ and $f(1) \approx -19.55 < 0$. By IVT, a root is guaranteed in $[0, 1]$. We select $h_0 = 0.5$ as a safe starting point with strong slope.
* **Convergence:** Newton-Raphson reaches the $\le 0.05\%$ error criterion in 3 iterations ($h \approx 0.648552$).

**2. Chemical Firm Break-Even Point (online_B2.pdf)**
* **Equation:** $f(x) = 298x - 3x^{2/3} - 1000 = 0$ (from $\text{Revenue} - \text{Cost} = 0$, where $R(x) = 300x$, and $C(x) = 1000 + 2x + 3x^{2/3}$).
* **Derivative:** $f'(x) = 298 - 2x^{-1/3}$.
* **Initial Guess Selection:** $f(1) = -705 < 0$ and $f(4) \approx 184.44 > 0$. By IVT, a root lies in $[1, 4]$. Choosing $x_0 = 4.0$ guarantees a strong slope and fast convergence.
* **Convergence:** NR converges to $x \approx 3.378371$ in 2 iterations for tolerance $\le 0.05\%$.

**3. False Position Cubic Solver (online_A1.pdf)**
* **Equation:** $f(x) = x^3 - x - 1 = 0$ on interval $[1.0, 2.0]$.
* **Bracket selection:** $f(1) = -1 < 0$ and $f(2) = 5 > 0$, guaranteeing a root.
* **Convergence:** False position requires 8 iterations to reach the $\varepsilon_s \le 0.05\%$ criterion ($x \approx 1.324279$).

---

## 📝 Exam-Day Workflow

```
1. READ QUESTION FULLY
2. IDENTIFY METHOD (bisection/FP/NR/Bairstow?)
3. DEFINE f(x) and df(x) [if NR]
4. PLOT THE GRAPH (always — marks are given here!)
5. FIND BRACKETS using incremental scan
6. VERIFY sign change: f(xl)·f(xu) < 0
7. RUN THE METHOD with correct tolerance
8. PRINT TABLE with proper column format
9. STATE THE ROOT to required significant figures
10. SAVE GRAPH (plt.savefig before plt.show)
```

## 📌 Most Common Mistake

```python
# WRONG: tol=0.0001 means 0.0001 (not 0.0001%)
# The stopping criterion is ea <= 0.0001 where ea IS already in %
# So ea = |..| * 100  and we compare with tol = 0.0001
# This gives 4 significant figures (Scarborough: 0.5 * 10^(2-4) = 0.005% ≠ 0.0001%)
# Read the question carefully: is tol given as % or as a plain number?
```
