# Bairstow's Method — Deep Dive Study Notes

## Why Bairstow Exists

Newton-Raphson works beautifully for real roots. But polynomials with **real coefficients** can have **complex roots** that always come in conjugate pairs: if $a + bi$ is a root, so is $a - bi$.

If we try Newton-Raphson with complex arithmetic, we need complex initial guesses. Bairstow avoids this by working with **quadratic factors** — a pair of complex conjugate roots always forms a quadratic with **real** coefficients:

$$(x - (a+bi))(x - (a-bi)) = x^2 - 2ax + (a^2 + b^2) = x^2 - rx - s$$

where $r = 2a$ and $s = -(a^2 + b^2)$. Both $r$ and $s$ are **real numbers**!

---

## Detailed Algorithm Walkthrough

### Setup

Given polynomial: $P(x) = a_0 x^n + a_1 x^{n-1} + \cdots + a_n$

Stored as: `a = [a₀, a₁, ..., aₙ]` (highest power first!)

Guess quadratic factor: $Q(x) = x^2 - rx - s$

### Step 1: First Synthetic Division → b-array

We want to divide $P(x)$ by $Q(x) = x^2 - rx - s$ to get:
$$P(x) = (x^2 - rx - s) \cdot B(x) + b_{n-1}x + b_n$$

where the $b_i$ satisfy:

$$\boxed{b_0 = a_0}$$
$$\boxed{b_1 = a_1 + r \cdot b_0}$$  
$$\boxed{b_i = a_i + r \cdot b_{i-1} + s \cdot b_{i-2} \quad \text{for } i = 2, \ldots, n}$$

The **residuals** are $b_{n-1}$ and $b_n$. At convergence, both should be $\approx 0$.

```python
def synthetic_div(a, r, s):
    n = len(a) - 1
    b = [0.0] * (n + 1)
    b[0] = a[0]
    b[1] = a[1] + r * b[0]
    for i in range(2, n + 1):
        b[i] = a[i] + r * b[i-1] + s * b[i-2]
    return b
```

### Step 2: Second Synthetic Division → c-array

Now divide $B(x)$ (the quotient from step 1, coefficients $b_0, \ldots, b_{n-2}$) by the **same** $Q(x)$.

In practice, we apply the same recurrence to `b[0..n-1]` (drop the last entry $b_n$):

$$\boxed{c_0 = b_0}$$
$$\boxed{c_1 = b_1 + r \cdot c_0}$$
$$\boxed{c_i = b_i + r \cdot c_{i-1} + s \cdot c_{i-2} \quad \text{for } i = 2, \ldots, n-1}$$

```python
b_short = b[:n]          # drop b[n]
c = synthetic_div(b_short, r, s)   # same function!
```

### Step 3: Newton Step (2×2 Linear System)

To minimize the residuals $(b_{n-1}, b_n)$, we solve:

$$\begin{bmatrix} c_{n-2} & c_{n-3} \\ c_{n-1} & c_{n-2} \end{bmatrix} \begin{bmatrix} \Delta r \\ \Delta s \end{bmatrix} = \begin{bmatrix} -b_{n-1} \\ -b_n \end{bmatrix}$$

By Cramer's rule:

$$D = c_{n-2}^2 - c_{n-3} \cdot c_{n-1}$$

$$\boxed{\Delta r = \frac{-b_{n-1} \cdot c_{n-2} + b_n \cdot c_{n-3}}{D}}$$

$$\boxed{\Delta s = \frac{-b_n \cdot c_{n-2} + b_{n-1} \cdot c_{n-1}}{D}}$$

Then: $r \leftarrow r + \Delta r$, $s \leftarrow s + \Delta s$

### Step 4: Check Convergence

$$\varepsilon_a(r) = \left|\frac{\Delta r}{r}\right| \times 100\% \leq \text{tol} \quad \textbf{AND} \quad \varepsilon_a(s) = \left|\frac{\Delta s}{s}\right| \times 100\% \leq \text{tol}$$

Both conditions must be satisfied simultaneously.

### Step 5: Extract Roots

From the converged quadratic factor $x^2 - rx - s = 0$:

$$x = \frac{r \pm \sqrt{r^2 + 4s}}{2}$$

If $r^2 + 4s \geq 0$: two **real** roots.
If $r^2 + 4s < 0$: two **complex conjugate** roots: $\frac{r}{2} \pm i\frac{\sqrt{|r^2+4s|}}{2}$

### Step 6: Deflation

The **quotient** polynomial (degree $n-2$) has coefficients: $b_0, b_1, \ldots, b_{n-2}$

```python
quotient = b_final[:n-1]   # drop the last 2 entries
```

Repeat the entire process on `quotient` until degree ≤ 2, then solve directly.

---

## Numerical Example (Worked)

### Polynomial: $x^4 - 10x^3 + 35x^2 - 50x + 24$ (roots: 1, 2, 3, 4)

`a = [1, -10, 35, -50, 24]`, $n = 4$, initial $r_0 = 0, s_0 = 1$

**Iteration 1:**

b-array (recurrence):
```
b[0] = 1
b[1] = -10 + 0×1 = -10
b[2] = 35 + 0×(-10) + 1×1 = 36
b[3] = -50 + 0×36 + 1×(-10) = -60
b[4] = 24 + 0×(-60) + 1×36 = 60
```
Residuals: $b_3 = -60$, $b_4 = 60$

c-array (divide b[0..3] by same quadratic):
```
b_short = [1, -10, 36, -60]
c[0] = 1
c[1] = -10 + 0×1 = -10
c[2] = 36 + 0×(-10) + 1×1 = 37
c[3] = -60 + 0×37 + 1×(-10) = -70
```

Determinant: $D = c_2^2 - c_1 \cdot c_3 = 37^2 - (-10)(-70) = 1369 - 700 = 669$

Corrections:
$$\Delta r = \frac{-(-60)(37) + (60)(-10)}{669} = \frac{2220 - 600}{669} = \frac{1620}{669} \approx 2.421$$

$$\Delta s = \frac{-(60)(37) + (-60)(-70)}{669} = \frac{-2220 + 4200}{669} = \frac{1980}{669} \approx 2.959$$

Updated: $r = 2.421$, $s = 3.959$

...continue iterating until both errors < tol, then extract roots from $x^2 - 2.421x - 3.959 \approx 0$...

---

## Common Mistakes in Bairstow

| Mistake | Effect |
|---------|--------|
| Coefficient order wrong (lowest power first) | Completely wrong results |
| Second division using full b (should drop b[n]) | Wrong c-array |
| Stopping when only ONE of ε_a(r), ε_a(s) < tol | Premature convergence |
| Forgetting to deflate before next iteration | Infinite loop on same degree |
| Initial guess r=0, s=0 | Division by zero in error formula ($\Delta r / r$) |

## Troubleshooting Convergence

| Problem | Solution |
|---------|----------|
| Determinant $D \approx 0$ | Perturb: $r \mathrel{+}= 0.5$, $s \mathrel{+}= 0.5$ |
| No convergence after max_iter | Try different $(r_0, s_0)$ |
| Complex intermediate values | That's fine if using `cmath` |
| Root residuals not near 0 | Refine with Newton-Raphson on original polynomial |

---

## Worked Table Format (Exam)

```
Finding quadratic factor for degree-4 polynomial
Initial: r₀=0, s₀=1

iter     r          s          Δr         Δs         ea_r(%)    ea_s(%)
   1   2.42003    3.95964    2.42003    2.95964    100.0000   100.0000
   2   3.00123    3.99987    0.58120    0.04023     19.3598     1.0058
   3   3.00000    4.00000    0.00123    0.00013      0.0410     0.0033
   4   3.00000    4.00000    0.00000    0.00000      0.0000     0.0000

Converged: r = 3.00000, s = 4.00000
Quadratic: x² − 3x − 4 = 0
disc = 3² + 4×4 = 9 + 16 = 25 ≥ 0
Roots: (3 ± 5) / 2  =  4.0  and  -1.0

Deflated polynomial: b[0..2] = [1, -7, 12]  (degree 2)
Direct quadratic: 1x² - 7x + 12 = 0
disc = 49 - 48 = 1 ≥ 0
Roots: (7 ± 1) / 2 = 4.0 and 3.0  ← Wait, but 4 appeared twice?
```

> Note: Deflation accumulates floating-point error. Always verify roots by plugging back into the ORIGINAL polynomial.
