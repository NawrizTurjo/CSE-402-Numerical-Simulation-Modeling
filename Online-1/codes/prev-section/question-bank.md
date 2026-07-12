# CSE-402 Numerical Simulation & Modeling: Online Lab 1 Question Bank
## Prev-Section Exam Problems, Theoretical Solutions & Code-Level Hints

This document serves as a comprehensive Question Bank and study companion containing all previous years' sections' online exam questions, their detailed theoretical solutions, and practical code-level implementation hints.

---

## Table of Contents
1. [Section-A1: Diode Equation (Newton-Raphson)](#1-section-a1-diode-equation-newton-raphson)
2. [Section-C2 (Type 1): Log-Scale Convergence Comparison](#2-section-c2-type-1-log-scale-convergence-comparison)
3. [Section-B1: Multi-Root Scanner (Bisection)](#3-section-b1-multi-root-scanner-bisection)
4. [Section-C2 (Type 2): First-Match Multi-Root Scanner (False Position)](#4-section-c2-type-2-first-match-multi-root-scanner-false-position)
5. [Section-B2: Pathological/Discontinuous Functions (Bisection)](#5-section-b2-pathologicaldiscontinuous-functions-bisection)

---

## 1. Section-A1: Diode Equation (Newton-Raphson)

### 📋 Problem Statement
Find the operating voltage $V$ that satisfies the diode characteristic equation:
$$f(V) = 10^{-12} \left(e^{\frac{V}{n V_T}} - 1\right) + \frac{V}{R} - I_L = 0$$

#### Given Parameters:
* Initial Guess: $V_0 = 0.65\text{ V}$
* Ideality Factor: $n = 1.8$
* Resistance: $R = 0.5\text{ k}\Omega = 500\ \Omega$
* Thermal Voltage: $V_T = 0.02585\text{ V}$
* Light Current: $I_L = 0.2\text{ mA} = 0.0002\text{ A}$
* Stopping Criterion: $\vert\varepsilon_a\vert \le 0.0001\%$

#### Formatting Requirement:
Print a tabular output with the exact headers: `Iter`, `V_i`, `f(V_i)`, `f'(V_i)`, `V_{i+1}`, `ea (%)`.

---

### 📝 Theoretical Solution
The **Newton-Raphson Method** converges quadratically by using the function value and the tangent slope (derivative) at the current estimate. The iterative formula is:
$$V_{i+1} = V_i - \frac{f(V_i)}{f'(V_i)}$$

#### Step 1: Formulate the Derivative
Let $f(V) = I_s \left(e^{\frac{V}{n V_T}} - 1\right) + \frac{V}{R} - I_L$ where $I_s = 10^{-12}\text{ A}$.
Differentiating $f(V)$ with respect to $V$:
$$f'(V) = \frac{d}{dV}\left[ I_s \left(e^{\frac{V}{n V_T}} - 1\right) + \frac{V}{R} - I_L \right]$$
$$f'(V) = \frac{I_s}{n V_T} e^{\frac{V}{n V_T}} + \frac{1}{R}$$

#### Step 2: Perform the First Iteration
* $n V_T = 1.8 \times 0.02585\text{ V} = 0.04653\text{ V}$
* **At $V_0 = 0.65\text{ V}$:**
  $$f(0.65) = 10^{-12} \left(e^{\frac{0.65}{0.04653}} - 1\right) + \frac{0.65}{500} - 0.0002$$
  $$f(0.65) = 10^{-12} \left(1.166 \times 10^6 - 1\right) + 0.0013 - 0.0002 \approx 1.101166 \times 10^{-3}\text{ A}$$
  $$f'(0.65) = \frac{10^{-12}}{0.04653} e^{\frac{0.65}{0.04653}} + \frac{1}{500} \approx 2.506 \times 10^{-5} + 0.002 = 2.025069 \times 10^{-3}\ \Omega^{-1}$$
* **Calculate $V_1$:**
  $$V_1 = V_0 - \frac{f(V_0)}{f'(V_0)} = 0.65 - \frac{1.101166 \times 10^{-3}}{2.025069 \times 10^{-3}} \approx 0.106233\text{ V}$$
* **Error $\vert\varepsilon_a\vert$:**
  $$\vert\varepsilon_a\vert = \left\vert \frac{V_1 - V_0}{V_1} \right\vert \times 100\% = \left\vert \frac{0.106233 - 0.65}{0.106233} \right\vert \times 100\% \approx 511.86\%$$

---

### 💻 Code-Level Hints
1. **Exponential Evaluation:** Use the `math.exp` function rather than `np.exp` when running scalar calculations inside loops to optimize speed.
2. **Tabular Formatting:** Format float outputs dynamically with `:>12.6f` and scientific notation with `:>14.6e` for `f(V_i)` and `f'(V_i)`.
3. **Safe Error Initialization:** The first iteration doesn't have a valid $\varepsilon_a$. Represent this with `"---"`.

---

## 2. Section-C2 (Type 1): Log-Scale Convergence Comparison

### 📋 Problem Statement
Solve $f(x) = \ln(x) = 0$ over the bracket $[10^{-4}, 10^4]$ with a tolerance of $\vert\varepsilon_a\vert \le 0.0001\%$.

#### Tasks:
1. Plot the function using a **logarithmic x-axis** (`plt.xscale('log')`).
2. Verify the sign change over the given interval boundaries.
3. Implement Bisection and False Position methods simultaneously.
4. Track and count the number of times the lower bound ($x_l$) and upper bound ($x_u$) are updated for both methods.
5. Print full iteration tables for both algorithms.
6. Print a final **Comparison Table** containing: Estimated Root, Iteration Count, Final $f(x_r)$, and Update Counts ($x_l$ updates / $x_u$ updates).

---

### 📝 Theoretical Solution
#### Boundary Verification:
* $f(10^{-4}) = \ln(10^{-4}) \approx -9.21034$ (Negative)
* $f(10^4) = \ln(10^4) \approx 9.21034$ (Positive)
* Since $f(x_l) \cdot f(x_u) < 0$, a sign change exists, ensuring at least one root is present in $[10^{-4}, 10^4]$.

#### Stagnation & Stiff-Interval Analysis:
* **Why Bisection is faster here (34 iterations vs 126 iterations):** Bisection splits the interval strictly in half every iteration, reducing the interval length by $2^{-n}$. False Position uses linear interpolation (secant line).
* **Endpoint Stagnation:** The function $f(x) = \ln(x)$ is highly asymmetric over the domain $[10^{-4}, 10^4]$. More importantly, its derivative $f''(x) = -1/x^2 < 0$ implies it is **concave down**.
* For a concave-down function, the secant line drawn between $x_l$ and $x_u$ always intersects the x-axis to the right of the true root ($x = 1.0$), resulting in $x_r > 1.0$. Because $f(x_r) > 0$, the algorithm updates $x_u = x_r$ every iteration. The lower bound $x_l$ remains stuck at its initial value ($10^{-4}$), receiving **0 updates** over the entire run. This is endpoint stagnation, turning a bracketed method into a slow, one-sided open method.

---

### 💻 Code-Level Hints
1. **Log Plotting:** Use `np.logspace(-4, 4, 1000)` to generate logarithmic points and call `plt.xscale('log')`.
2. **Update Tracking:** Set counters inside the interval conditional check:
   ```python
   if f(xl) * f(xr) < 0:
       xu = xr
       xu_updates += 1
   else:
       xl = xr
       xl_updates += 1
   ```
3. **Double Table Setup:** Run each method inside a clean subroutine returning `history` arrays and counters, then loop through both to print tables.

---

## 3. Section-B1: Multi-Root Scanner (Bisection)

### 📋 Problem Statement
Find all real roots for the transcendental equation:
$$f(x) = 0.6\ln(x+1) - C\sin(1.7x) - 0.08x^2 - 0.08 = 0$$
Domain: $0 \le x \le 10$ using an incremental search step size of $\Delta x = 0.1$.

#### Tasks:
1. Plot the function across the entire domain to visually capture the roots.
2. Programmatically isolate all sign-change intervals.
3. Execute the Bisection method automatically on every isolated interval.
4. Print individual iteration tables for each root discovered.

---

### 📝 Theoretical Solution
#### Sign-Change Interval Isolation (Intermediate Value Theorem):
For a continuous function $f(x)$, if $f(x_1) \cdot f(x_2) < 0$, then there must exist at least one root $x^* \in [x_1, x_2]$. We divide the domain $[0, 10]$ into small sub-intervals $[x_i, x_i + \Delta x]$ and test this condition.

#### Failure of Global Bisection:
Executing a single Bisection run directly on $[0, 10]$ fails catastrophically because:
1. **Sign-change Cancellation:** If an even number of roots lie in $[a, b]$, the boundary values will have the same sign ($f(0) \approx -0.08 < 0$, $f(10) \approx -7.42 < 0$), yielding $f(a) \cdot f(b) > 0$. Bisection fails to start as it assumes no root exists.
2. **Singular Convergence:** Even if there is an odd number of roots, Bisection is mathematically limited to converging to only **one** root. The rest are completely ignored.

---

### 💻 Code-Level Hints
1. **Incremental Loop:** Construct intervals using standard loops:
   ```python
   xs = np.arange(x_min, x_max, step)
   intervals = []
   for i in range(len(xs)-1):
       if f(xs[i]) * f(xs[i+1]) < 0:
           intervals.append((xs[i], xs[i+1]))
   ```
2. **Bisection Loops:** Loop over the discovered intervals and execute the bisection solver dynamically, passing the current interval bounds as the initial bracket.

---

## 4. Section-C2 (Type 2): First-Match Multi-Root Scanner (False Position)

### 📋 Problem Statement
Given a general multi-root expression, scan the domain to locate all sign-change intervals. Automatically isolate the **first interval** that exhibits a valid sign change, run the False Position method on it targeting $\vert\varepsilon_a\vert \le 0.0001\%$, and print the iteration table.

#### Default Test Case:
* Function: $f(x) = 2x^3 - 11.7x^2 + 17.7x - 5 = 0$
* Domain: $[0, 4]$, Step size: $\Delta x = 0.1$

---

### 📝 Theoretical Solution
The **False Position Method** uses a straight secant line between $(x_l, f(x_l))$ and $(x_u, f(x_u))$. The x-intercept of this line yields the next estimate $x_r$:
$$x_r = x_u - \frac{f(x_u)(x_l - x_u)}{f(x_l) - f(x_u)}$$
The updating strategy follows the sign-check of $f(x_l) \cdot f(x_r)$. If negative, the root lies in $[x_l, x_r]$, hence $x_u = x_r$. Otherwise, the root lies in $[x_r, x_u]$, hence $x_l = x_r$.

For the cubic function:
* Sub-intervals scanned: $[0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4] \dots$
* At $x = 0.3$: $f(0.3) = 2(0.3)^3 - 11.7(0.3)^2 + 17.7(0.3) - 5 = 0.311 > 0$
* At $x = 0.4$: $f(0.4) = 2(0.4)^3 - 11.7(0.4)^2 + 17.7(0.4) - 5 = -0.648 < 0$
* **First Match:** Interval $[0.3, 0.4]$ triggers a sign change.

---

### 💻 Code-Level Hints
1. **Early Exit:** Break the scanning loop immediately once the first bracket is found:
   ```python
   intervals = scan_intervals(0, 4, 0.1)
   if intervals:
       first_bracket = intervals[0]
       # Proceed with False Position
   ```
2. **False Position Interpolation Code:** Keep track of the previous estimate `xr_old` to calculate `ea = abs((xr - xr_old) / xr) * 100`.

---

## 5. Section-B2: Pathological/Discontinuous Functions (Bisection)

### 📋 Problem Statement
Analyze the root behavior of the rational function:
$$f(x) = \frac{(x-2.5)^2(x+1.5)}{x-3.56}$$

#### Tasks:
1. Perform an incremental interval scan and run Bisection on valid bounding regions.
2. Plot the curve, paying close attention to points near $x = 2.5$ and $x = 3.56$.

---

### 📝 Theoretical Solution
Rational and discontinuous functions pose serious problems for automated root-finding scripts:

#### Case 1: Missed Roots (Double Roots / Even Multiplicity)
* The root at $x = 2.5$ corresponds to the factor $(x-2.5)^2$.
* Because of the square power, the function values on both sides of $2.5$ maintain the same sign (they remain negative as $x+1.5 > 0$ and $x-3.56 < 0$ near $2.5$).
* The function touches the x-axis tangent-wise at $x = 2.5$ but does not cross it. Since there is no sign change, $f(x_l) \cdot f(x_u) > 0$ for any interval enclosing $2.5$.
* **Result:** The automated incremental sign scanner completely misses this valid root.

#### Case 2: Garbage Roots (Vertical Asymptotes / Discontinuities)
* At $x = 3.56$, the denominator $(x-3.56)$ goes to $0$.
* Approaching from left ($x < 3.56$): $f(x) \to -\infty$.
* Approaching from right ($x > 3.56$): $f(x) \to +\infty$.
* This causes a sign change in the scanner interval (e.g. $[3.5, 3.6]$).
* When Bisection runs on this interval, it halves the interval and converges directly to $x \approx 3.56$.
* **Result:** The algorithm converges, but the target is a singularity discontinuity, not a root. The function value at the converged point goes to infinity instead of zero, yielding a "garbage root".

---

### 💻 Code-Level Hints
1. **Asymptote Masking in NumPy:** To avoid a vertical line joining $+\infty$ and $-\infty$ when plotting, set a threshold to mask values near $x = 3.56$:
   ```python
   y_vals = ((x - 2.5)**2 * (x + 1.5)) / (x - 3.56)
   y_vals[np.abs(x - 3.56) < 0.02] = np.nan
   ```
2. **Y-Limit Restriction:** Use `plt.ylim(-30, 30)` to prevent the plot from stretching vertically to infinity, ensuring the roots are clearly visible.
3. **Discontinuity Guard:** Add logic in bisection to stop if $f(x_r)$ starts exploding or becomes infinite.
