# Newton-Raphson Deep Dive & Failure Analysis
## CSE-402 Study Notes

---

## The Core Idea — Geometry First

The Newton-Raphson formula is **purely geometric**. At any point $x_i$ on the curve $f(x)$:

1. Draw the **tangent line** to the curve at $(x_i, f(x_i))$
2. The tangent has slope $f'(x_i)$
3. Where does this tangent line **cross the x-axis**?
4. That crossing point → $x_{i+1}$

**Derivation from geometry:**

Tangent line through $(x_i, f(x_i))$ with slope $f'(x_i)$:
$$y - f(x_i) = f'(x_i)(x - x_i)$$

Set $y = 0$ (find x-axis crossing):
$$-f(x_i) = f'(x_i)(x - x_i)$$
$$x = x_i - \frac{f(x_i)}{f'(x_i)}$$

So: $\boxed{x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}}$

---

## Why Is It So Fast? (Quadratic Convergence)

Taylor expand $f(x)$ around the true root $\alpha$:

$$f(x_i) = f(\alpha) + f'(\alpha)(x_i - \alpha) + \frac{f''(\alpha)}{2}(x_i - \alpha)^2 + \cdots$$

Since $f(\alpha) = 0$:

$$\frac{f(x_i)}{f'(x_i)} \approx (x_i - \alpha) + \frac{f''(\alpha)}{2f'(\alpha)}(x_i - \alpha)^2$$

The error in $x_{i+1}$ is:

$$e_{i+1} = x_{i+1} - \alpha = x_i - \frac{f(x_i)}{f'(x_i)} - \alpha \approx -\frac{f''(\alpha)}{2f'(\alpha)} e_i^2$$

**The error is proportional to the SQUARE of the previous error.** If $e_i = 0.01$, then $e_{i+1} \approx 0.0001$. Two more decimal digits per step!

**Bisection:** halves error each step → $e_{i+1} = e_i / 2$ (linear)  
**Newton-Raphson:** squares error reduction → $e_{i+1} \propto e_i^2$ (quadratic)

---

## Five Failure Modes (All Exam-Testable!)

### Failure 1: Oscillation (2-Cycle Trap)

**Function:** $f(x) = x^3 - 2x + 2$  
**Initial guess:** $x_0 = 0$

| Iter | $x_i$ | $f(x_i)$ | $f'(x_i)$ | $x_{i+1}$ |
|:----:|:-----:|:--------:|:---------:|:---------:|
| 1 | 0 | 2 | −2 | 1 |
| 2 | 1 | 1 | 1 | 0 |
| 3 | 0 | 2 | −2 | 1 |
| 4 | 1 | 1 | 1 | 0 |

**Why it happens:** The tangent at $x=0$ points to $x=1$, and the tangent at $x=1$ points back to $x=0$. The method enters an infinite geometric loop.

**The $\varepsilon_a$ problem:** At iteration 3, $x_{i+1} = 0$. So $\varepsilon_a = |0 - 1| / 0 = $ **Division by Zero!**

**Fix:** Find the sign-change interval first. For this function:
```
f(-2) = -8 + 4 + 2 = -2 < 0
f(-1) = -1 + 2 + 2 = 3 > 0
Bracket: [-2, -1]  →  use x₀ = -1.5
```

### Failure 2: Division by Zero (Zero Derivative)

**Function:** $f(x) = \sin(x)$  
**Initial guess:** $x_0 = \pi/2$

$f'(\pi/2) = \cos(\pi/2) = 0$ → horizontal tangent → method is undefined.

**Fix:** Shift the initial guess. Use $x_0 = \pi/2 + 0.1$.

### Failure 3: Vertical Tangent Divergence

**Function:** $f(x) = \sqrt[3]{x}$  
**Any $x_0 \neq 0$**

$f'(x) = \frac{1}{3} x^{-2/3}$ → as $x \to 0$, $f'(x) \to \infty$

$x_{i+1} = x_i - \frac{x_i^{1/3}}{(1/3)x_i^{-2/3}} = x_i - 3x_i = -2x_i$

**Pattern:** $0.1 \to -0.2 \to 0.4 \to -0.8 \to 1.6 \to \ldots$ (diverges!)

**Fix:** Use bisection or false position. NR is fundamentally incompatible with this root type.

> ⚠️ **Python trap:** Never use `x**(1/3)` — it gives complex numbers for negative `x`. Always use `np.cbrt(x)` for cube roots.

### Failure 4: Overshooting (Flat Tails)

**Function:** $f(x) = \arctan(x)$  
**Initial guess:** $x_0 = 1.5$

$f'(1.5) = 1/(1 + 1.5^2) \approx 0.31$ (very small)

$x_1 = 1.5 - \arctan(1.5)/0.31 \approx 1.5 - 3.2 = -1.7$ 

The small derivative $\to$ large correction step $\to$ overshooting. The sequence may diverge or oscillate wildly.

**Fix:** Choose $x_0$ close to the root (use graphical method first).

### Failure 5: Multiple Root Speed Loss

**Function:** $f(x) = (x-2)^3$  
**Root:** $x = 2$ with multiplicity 3

Near the root: $f(x) \to 0$ AND $f'(x) \to 0$ simultaneously. The ratio $f/f'$ never "snaps" quickly. Newton-Raphson loses its quadratic convergence speed and scales down to linear convergence.

**Fix (Modified Newton-Raphson for Multiple Roots):**
$$x_{i+1} = x_i - m \cdot \frac{f(x_i)}{f'(x_i)}$$
where $m$ is the **multiplicity** of the root (e.g. $m=3$ for the triple root at $x=2$). Applying $m$ mathematically scales the tangent slope to restore **quadratic convergence** ($e_{i+1} \propto e_i^2$).

### Failure 6: Oscillation / Loop Detector Trap
When the Newton-Raphson guesses bounce back and forth between two coordinates (e.g. $0 \to 1 \to 0 \to 1$), the code enters an infinite cycle.
* **The Code-Level Fix:** Track visited $x_i$ values in a list. At each step, search if $x_{i+1}$ is within `1e-6` of any visited value. If yes, raise an oscillation flag, stop, and select a new initial guess.

---

## Deriving f'(x) — Step by Step

For the **diode equation:**
$$f(V) = 10^{-12}\left(e^{V/(1.8 \times 0.02585)} - 1\right) + \frac{V}{500} - 0.0002$$

Differentiate each term:
- $\frac{d}{dV}\left[10^{-12} e^{V/(nV_T)}\right] = 10^{-12} \cdot \frac{1}{nV_T} \cdot e^{V/(nV_T)}$ (chain rule)
- $\frac{d}{dV}\left[-10^{-12}\right] = 0$
- $\frac{d}{dV}\left[\frac{V}{R}\right] = \frac{1}{R}$

$$\therefore f'(V) = \frac{10^{-12}}{nV_T} e^{V/(nV_T)} + \frac{1}{R}$$

For the **dipstick:**
$$f(h) = \frac{\pi h^2(3r - h)}{3} - V = \frac{\pi(3rh^2 - h^3)}{3} - V$$

$$f'(h) = \frac{\pi(6rh - 3h^2)}{3} = \pi(2rh - h^2)$$

Substituting $r = 4$:
$$f'(h) = \pi(8h - h^2)$$

For the **Chemical Firm Break-Even:**
$$f(x) = 298x - 3x^{2/3} - 1000$$

Differentiate each term:
- $\frac{d}{dx}\left[298x\right] = 298$
- $\frac{d}{dx}\left[-3x^{2/3}\right] = -3 \cdot \frac{2}{3} \cdot x^{2/3 - 1} = -2x^{-1/3}$
- $\frac{d}{dx}\left[-1000\right] = 0$

$$\therefore f'(x) = 298 - 2x^{-1/3} = 298 - \frac{2}{\sqrt[3]{x}}$$

---

## Quick Formula Card

```
┌────────────────────────────────────────────────────────────────┐
│  NEWTON-RAPHSON FORMULA CARD                                    │
├────────────────────────────────────────────────────────────────┤
│  Main formula:    x_{i+1} = x_i - f(x_i) / f'(x_i)           │
│  Error:           ε_a = |x_{i+1} - x_i| / |x_{i+1}| × 100%  │
│  Stop when:       ε_a ≤ tolerance                              │
│                                                                 │
│  Convergence:     QUADRATIC near the root                       │
│  Typical steps:   3–6 for most textbook problems               │
│                                                                 │
│  Requires:        f'(x) — derive analytically before exam!     │
│  No bracket:      x₀ can be anywhere (but choose wisely)       │
└────────────────────────────────────────────────────────────────┘
```

---

## Exam Preparation Checklist for Newton-Raphson

- [ ] Can I state the formula from memory?
- [ ] Can I derive $f'(x)$ for polynomial, exponential, log, trig functions?
- [ ] Can I recognize and explain all 5 failure modes?
- [ ] Do I know the 2-cycle trick ($x^3 - 2x + 2$ with $x_0 = 0$)?
- [ ] Do I use `np.cbrt(x)` not `x**(1/3)`?
- [ ] Can I print the correct 5-column table format?
- [ ] Can I plot f(x) and mark where the root is?
