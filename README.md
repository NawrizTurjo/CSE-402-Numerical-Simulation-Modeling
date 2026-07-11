# CSE 402: Numerical Analysis, Simulation and Modeling Sessional

This repository contains implementation code, study guides, and test solvers developed for the sessional course CSE 402.

---

## Online-1 Syllabus

* **Error Analysis**: Approximations, round-off errors, truncation errors, machine epsilon, and error tradeoff limits.
* **Visualization and Plotting**: Functions, brackets, root markers, and convergence curves using matplotlib.
* **Root Finding Methods**:
  * Bisection Method
  * False Position Method (including Illinois Variant)
  * Newton-Raphson Method (including Multiplicity correction and Oscillation checks)
  * Bairstow's Method (for real and complex roots of polynomials)

---

## Core Algorithms and Formulations

### 1. Bisection Method

The Bisection Method is a bracketing method that repeatedly halves the interval containing a root. Given an interval $[x_L, x_U]$ where $f(x_L) \cdot f(x_U) < 0$:

* **Midpoint Formula**:
  $$x_r = \frac{x_L + x_U}{2}$$

* **Update Rule**:
  If $f(x_L) \cdot f(x_r) < 0$, the root lies in the left half: $x_U \leftarrow x_r$.
  Else, the root lies in the right half: $x_L \leftarrow x_r$.

* **Safe Approximate Relative Error**:
  $$\varepsilon_a = \left| \frac{x_r^{\text{new}} - x_r^{\text{old}}}{x_r^{\text{new}}} \right| \times 100\%$$
  To prevent division-by-zero crashes when the root converges to $0$:
  ```python
  if abs(xr_new) < 1e-12:
      ea = abs(xr_new - xr_old) * 100.0
  else:
      ea = abs((xr_new - xr_old) / xr_new) * 100.0
  ```

---

### 2. False Position and Illinois Method

The False Position (Regula Falsi) Method approximates the root by connecting $(x_L, f(x_L))$ and $(x_U, f(x_U))$ with a secant line and finding its x-intercept:

* **Secant Formula**:
  $$x_r = x_U - \frac{f(x_U)(x_L - x_U)}{f(x_L) - f(x_U)}$$

* **The Stagnation Problem**:
  If a function is strictly concave or convex over the interval, one boundary endpoint remains unchanged ($\varepsilon_a$ convergence slows down dramatically).

* **The Illinois Modification**:
  If the same boundary endpoint is stagnant for consecutive steps, the function value at that boundary is halved:
  ```python
  if fl * fxr < 0:
      xu = xr
      fu = fxr
      fl = fl / 2.0  # Halve the inactive bound's weight to shift the secant line
  else:
      xl = xr
      fl = fxr
      fu = fu / 2.0  # Halve the inactive bound's weight to shift the secant line
  ```

---

### 3. Newton-Raphson Method

Newton-Raphson is an open method that uses the function value and derivative at a single guess $x_i$ to project the tangent line onto the x-axis:

* **Iteration Formula (With Multiplicity Modifier $m$)**:
  $$x_{i+1} = x_i - m \frac{f(x_i)}{f'(x_i)}$$

* **Multiplicity ($m$)**:
  For single roots, $m = 1$. For roots with multiplicity $m > 1$, setting $m$ preserves the quadratic rate of convergence (otherwise it degrades to linear speed).

* **Cycle/Oscillation Detection**:
  Newton-Raphson can become trapped in infinite oscillation loops (e.g. $0 \rightarrow 1 \rightarrow 0 \rightarrow 1$). We track values in a list to flag cycles:
  ```python
  for past_x in history[:-1]:
      if abs(x_new - past_x) < 1e-6:
          print("Warning: Infinite oscillation loop detected.")
          break
  ```

---

### 4. Bairstow's Method

Bairstow's Method extracts quadratic factors of the form $x^2 - r \cdot x - s$ from an $n$-degree polynomial using synthetic division:

* **Synthetic Division Coefficients**:
  $$b_0 = a_0, \quad b_1 = a_1 + r \cdot b_0$$
  $$b_i = a_i + r \cdot b_{i-1} + s \cdot b_{i-2} \quad (\text{for } i = 2, \ldots, n)$$

* **Second Synthetic Division**:
  $$c_0 = b_0, \quad c_1 = b_1 + r \cdot c_0$$
  $$c_i = b_i + r \cdot c_{i-1} + s \cdot c_{i-2} \quad (\text{for } i = 2, \ldots, n-1)$$

* **Corrections ($\Delta r, \Delta s$)**:
  We solve the linear system:
  $$\begin{bmatrix} c_{n-2} & c_{n-3} \\ c_{n-1} & c_{n-2} \end{bmatrix} \begin{bmatrix} \Delta r \\ \Delta s \end{bmatrix} = \begin{bmatrix} -b_{n-1} \\ -b_n \end{bmatrix}$$

* **Quadratic Root Extraction**:
  $$\text{discriminant} = r^2 + 4s$$
  $$x = \frac{r \pm \sqrt{r^2 + 4s}}{2}$$

---

### 5. Truncation vs. Round-off Error Tradeoff

When approximating numerical derivatives using finite differences (e.g., $f'(x) \approx \frac{f(x+h)-f(x)}{h}$):
* **Truncation Error**: Decreases as the step size $h \rightarrow 0$ because the Taylor series approximation becomes more exact (proportional to $O(h)$).
* **Round-off Error**: Increases as $h \rightarrow 0$ because subtracting very close floating-point values causes subtractive cancellation, which is amplified when dividing by $h$.
* **Tradeoff Limit**: The combination yields a V-shaped error curve. In double precision (64-bit), the optimal step size that balances both errors is $h \approx 10^{-8}$.
