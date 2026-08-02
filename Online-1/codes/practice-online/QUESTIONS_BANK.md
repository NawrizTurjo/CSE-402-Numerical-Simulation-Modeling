# Creative & Tricky Practice Question Bank (`practice-online/`)

> **Target Scope**: Slide 1 (Errors, Approximations, Truncation, Round-off) & Slide 2 (Bracketing & Open Root-Finding Methods).
> **Objective**: Practice problems that require mathematical thinking, custom filters, or multi-step logic before invoking template solvers.

---

## 📌 Problem 1: Zener Diode Breakdown (Multiplicity $m=2$ Trap)

### 📋 Scenario
In semiconductor modeling, a Zener diode near its reverse breakdown voltage is governed by:
$$f(V) = (V - 0.7)^2 e^{0.5 V} - 0.05 V + 0.035 = 0$$

### 💡 The Trick / Challenge
1. Evaluating $f(0.7) = -0.05(0.7) + 0.035 = 0$, so $V^* = 0.7\text{ V}$ is an exact root of multiplicity **$m = 2$**.
2. Standard Newton-Raphson starting at $V_0 = 1.0\text{ V}$ degrades from quadratic speed to **linear convergence** ($ea$ drops very slowly).
3. **Task**: Run standard Newton-Raphson ($m=1$) vs Modified Newton-Raphson ($m=2$). Show how setting `m=2` in `P._newton_raphson_core(x0, m=2)` restores 3-iteration quadratic convergence!

---

## 📌 Problem 2: Thermal Sensor Differentiation (V-Curve $h^*$ Optimization)

### 📋 Scenario
A high-precision optical sensor measures temperature decay described by $f(x) = \cos(x) e^{-x}$. We need to compute the derivative $f'(1.5)$ numerically.

### 💡 The Trick / Challenge
1. Exact derivative: $f'(x) = -\sin(x) e^{-x} - \cos(x) e^{-x}$. At $x = 1.5$, exact $f'(1.5) \approx -0.2380582$.
2. For large step sizes $h = 1.0$, truncation error is large ($O(h)$).
3. For tiny step sizes $h < 10^{-12}$, subtractive cancellation in $\cos(1.5+h) e^{-(1.5+h)} - \cos(1.5) e^{-1.5}$ divided by tiny $h$ causes **round-off error explosion**.
4. **Task**: Sweep $h \in [10^0, 10^{-16}]$, plot the V-curve, find the optimal step size $h^*$, and compare Forward Diff vs Central Diff accuracy at $h^*$.

---

## 📌 Problem 3: Chemical Reactor Singularity Trap (Asymptote Filtering)

### 📋 Scenario
A non-linear kinetic equilibrium equation contains a vertical asymptote near $x = \pi/2 \approx 1.5708$:
$$f(x) = \frac{\tan(x) - x}{x - 1.5708}$$
We want to find real physical roots in the domain $[0, 3.0]$.

### 💡 The Trick / Challenge
1. At $x = 1.5708$, $f(x)$ flips sign from $+\infty$ to $-\infty$ across a **vertical pole** (singularity), NOT a real zero!
2. Standard sign-change scanning (`coarse_scan`) flags $[1.5, 1.6]$ as a root interval. Standard Bisection converges onto the singularity $x \approx 1.5708$.
3. **Task**: Filter out the fake pole root using `P.classify_and_verify` with `residual_tol=1e-3`. Show that convergence alone is NOT enough — residual $f(x_r)$ must be near zero!

---

## 📌 Problem 4: Solar Farm Margin (Convexity & Illinois Stagnation Fix)

### 📋 Scenario
A solar panel cost-efficiency equation is given by:
$$f(x) = 2^x - 5x + 1 = 0 \quad \text{on interval } [3.0, 5.0]$$

### 💡 The Trick / Challenge
1. $f(x)$ is strictly **convex** ($f''(x) = 2^x (\ln 2)^2 > 0$).
2. Standard False Position suffers severe **one-sided stagnation**: $x_l = 3.0$ never updates ($x_l$ updates $= 0$), causing false position to degrade to slow linear steps.
3. **Task**: Compare Bisection, Standard False Position, and Illinois False Position side-by-side. Demonstrate how Illinois (`P.false_position_illinois`) halves the stagnant weight to force bracket switching.

---

## 📌 Problem 5: Damped Wave Maclaurin Expansion (Scarborough Stopping)

### 📋 Scenario
A acoustic wave signal envelope requires evaluating $\frac{\sin(x)}{x}$ near $x = 1.2$:
$$\frac{\sin x}{x} = 1 - \frac{x^2}{3!} + \frac{x^4}{5!} - \frac{x^6}{7!} + \dots = \sum_{k=0}^{\infty} (-1)^k \frac{x^{2k}}{(2k+1)!}$$

### 💡 The Trick / Challenge
1. Derive the term recurrence ratio: $T_k = -T_{k-1} \cdot \frac{x^2}{(2k)(2k+1)}$.
2. Target: Guarantee $n = 5$ correct significant figures ($\varepsilon_s = 0.5 \times 10^{2-5} = 0.0005\%$).
3. **Task**: Implement using `P.maclaurin_series`, compute per-step guaranteed significant figures `P.calc_sig_digit(ea)`, and plot the error decay.

---

## 📌 Problem 6: RLC Circuit Complex Characteristic Roots (Complex Newton-Raphson)

### 📋 Scenario
An underdamped electrical RLC circuit has a characteristic impedance equation:
$$f(z) = z^3 + 2z^2 + 5z + 10 = 0$$

### 💡 The Trick / Challenge
1. **Factoring**: $f(z) = (z + 2)(z^2 + 5) = 0$. Roots are $z_1 = -2.0$ (Real) and $z_{2,3} = \pm \sqrt{5} i \approx \pm 2.236068 i$ (Complex conjugate pair).
2. Real bracketing methods (Bisection, False Position) **CANNOT** find $z_2$ or $z_3$ because $z^2 + 5 > 0$ for all real $z \in \mathbb{R}$ (no sign change on the real axis).
3. **Task**: Initialize Newton-Raphson in Python using **complex initial guesses** $z_0 = 1.0 + 1.0j$ and $z_0 = 1.0 - 1.0j$. Show how the exact Newton update $z_{i+1} = z_i - \frac{f(z_i)}{f'(z_i)}$ converges smoothly in the complex plane to find all complex roots!
