**Equation:**


$$10^{-12} \left(e^{\frac{V}{n V_T}} - 1\right) + \frac{V}{R} - I_L = 0$$

**Given Parameters:**

* $V_0 = 0.65\text{ V}$ (Initial guess)
* $n = 1.8$
* $R = 0.5\text{ k}\Omega$
* $V_T = 0.02585$
* $I_L = 0.2\text{ mA}$

**Requirements:**

* Solve using the **Newton-Raphson method** (define the function $f$ and its derivative $df$).
* **Stopping Criterion:** $\vert{}\varepsilon_a\vert{} \le 0.0001\%$
* **Output Format:** Print a table with the following columns:

$$\text{Iter} \quad V_i \quad f(V_i) \quad f'(V_i) \quad V_{i+1} \quad \varepsilon_a$$


* Maintain correct significant digits for the final answer.

```python
import numpy as np

def f1(x):
    return np.log(x)

def bisection(xl, xu, tol=0.0001):
    xr_old = None
    xl_updates = 0
    xu_updates = 0
    iterations = []
    
    for i in range(1, 100):
        xr = (xl + xu) / 2.0
        ea = abs((xr - xr_old) / xr) * 100 if xr_old is not None else 100.0
        
        iterations.append([i, xl, xu, xr, ea, f1(xr)])
        
        if ea <= tol and xr_old is not None:
            break
            
        if f1(xl) * f1(xr) < 0:
            xu = xr
            xu_updates += 1
        else:
            xl = xr
            xl_updates += 1
        xr_old = xr
        
    return iterations, xl_updates, xu_updates

def false_position(xl, xu, tol=0.0001):
    xr_old = None
    xl_updates = 0
    xu_updates = 0
    iterations = []
    
    for i in range(1, 200):
        fl = f1(xl)
        fu = f1(xu)
        xr = xu - (fu * (xl - xu)) / (fl - fu)
        ea = abs((xr - xr_old) / xr) * 100 if xr_old is not None else 100.0
        
        iterations.append([i, xl, xu, xr, ea, f1(xr)])
        
        if ea <= tol and xr_old is not None:
            break
            
        if fl * f1(xr) < 0:
            xu = xr
            xu_updates += 1
        else:
            xl = xr
            xl_updates += 1
        xr_old = xr
        
    return iterations, xl_updates, xu_updates

bi_iters, bi_l, bi_u = bisection(1e-4, 1e4)
fp_iters, fp_l, fp_u = false_position(1e-4, 1e4)

print(f"Bisection iterations: {len(bi_iters)}, L updates: {bi_l}, U updates: {bi_u}, Final Root: {bi_iters[-1][3]}")
print(f"False Position iterations: {len(fp_iters)}, L updates: {fp_l}, U updates: {fp_u}, Final Root: {fp_iters[-1][3]}")



```

```text
Bisection iterations: 34, L updates: 11, U updates: 22, Final Root: 0.999999920321681
False Position iterations: 126, L updates: 0, U updates: 125, Final Root: 1.0000079534543922


```

```python
import numpy as np

def f2(x):
    return 0.6 * np.log(x + 1) - 1.0 * np.sin(1.7 * x) - 0.08 * x**2 - 0.08

# Find sign changes in [0, 10] with step size 0.1
x_vals = np.arange(0, 10.1, 0.1)
intervals = []
for i in range(len(x_vals)-1):
    x1, x2 = x_vals[i], x_vals[i+1]
    if f2(x1) * f2(x2) < 0:
        intervals.append((x1, x2))

print("Sign change intervals:", intervals)



```

```text
Sign change intervals: [(1.6, 1.7000000000000002), (3.5, 3.6)]


```

```python
def f3(x):
    return 2*x**3 - 11.7*x**2 + 17.7*x - 5

x_vals = np.arange(0, 4, 0.1)
intervals = []
for i in range(len(x_vals)-1):
    x1, x2 = x_vals[i], x_vals[i+1]
    if f3(x1) * f3(x2) < 0:
        intervals.append((round(x1,1), round(x2,1)))
print("Chapra function intervals:", intervals)


```

```text
Chapra function intervals: [(0.3, 0.4), (1.9, 2.0), (3.5, 3.6)]


```

Here is your raw, informal chat notes organized into a clean, formal **Lab Assignment Document**, complete with **complete Python code** and the **written analytical answers** required for your submission.

---

## Problem 1: Root Finding for $f(x) = \ln(x)$

### 📋 Formal Question Statement

Given the non-linear equation $f(x) = \ln(x) = 0$ within the interval $10^{-4} \le x \le 10^4$, perform the following tasks:

1. Plot the function on a **logarithmic scale**.
2. Algebraically verify that a sign change occurs within the given interval.
3. Apply the **Bisection Method** using a stopping criterion of $\vert{}\varepsilon_a\vert{} \le 0.0001\%$.
4. Apply the **False Position Method** using the same stopping criterion.
5. Print complete iteration tables for both methods showing: $\text{Iter}$, $X_l$, $X_u$, $X_r$, $\varepsilon_a$, and $f(X_r)$.
6. Track and report the total number of times boundaries $X_l$ and $X_u$ are updated for both methods.
7. Print a comparison table summarizing the estimated root, final iteration count, final $f(X_r)$, and update counts.
8. Provide written explanations for the analytical questions regarding speed and boundary stagnation.

---

### 📝 Written Answers (For your exam paper)

* **a) Which method is faster?**
**Bisection is significantly faster** in this case. It reaches the tolerance in **34 iterations**, whereas the False Position method requires **126 iterations**.
* **b) Why is False Position not guaranteed to be faster?**
While False Position usually performs better by utilizing linear interpolation, it performs poorly when dealing with highly asymmetric intervals or functions with severe curvature (like $\ln(x)$ over a massive span from $10^{-4}$ to $10^4$). The straight line (secant) repeatedly strikes very close to one side, leading to minuscule step sizes.
* **c) Why did one endpoint never update in the slower method (False Position)?**
Because $f(x) = \ln(x)$ is strictly **concave** over its domain. Over the huge interval $[10^{-4}, 10^4]$, the secant line between $X_l$ and $X_u$ always intersects the x-axis to the right of the true root ($x=1$), making $X_r > 1$. Since $f(X_r) > 0$, the algorithm constantly updates the upper bound $X_u = X_r$, leaving the lower bound $X_l = 10^{-4}$ permanently stuck with **0 updates**.

---

### 💻 Python Code for Problem 1

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Define Function
def f1(x):
    return np.log(x)

# Verification of Sign Change
print(f"f(10^-4) = {f1(1e-4):.4f}, f(10^4) = {f1(1e4):.4f}")
print(f"Sign change verified: {f1(1e-4) * f1(1e4) < 0}\n")

# 2. Methods Implementation
def run_method(method_type, xl, xu, tol=0.0001):
    xr_old = None
    xl_updates = 0
    xu_updates = 0
    history = []
    
    for idx in range(1, 500):
        if method_type == 'bisection':
            xr = (xl + xu) / 2.0
        elif method_type == 'false_position':
            fl, fu = f1(xl), f1(xu)
            xr = xu - (fu * (xl - xu)) / (fl - fu)
            
        ea = abs((xr - xr_old) / xr) * 100 if xr_old is not None else 100.0
        history.append([idx, xl, xu, xr, ea, f1(xr)])
        
        if ea <= tol and xr_old is not None:
            break
            
        # Update bounds
        if f1(xl) * f1(xr) < 0:
            xu = xr
            xu_updates += 1
        else:
            xl = xr
            xl_updates += 1
        xr_old = xr
        
    return history, xl_updates, xu_updates

bi_hist, bi_l, bi_u = run_method('bisection', 1e-4, 1e4)
fp_hist, fp_l, fp_u = run_method('false_position', 1e-4, 1e4)

# Print Tables Function
def print_table(title, history):
    print(f"--- {title} Table ---")
    print(f"{'Iter':<6}{'Xl':<12}{'Xu':<12}{'Xr':<12}{'Ea (%)':<12}{'f(Xr)':<12}")
    for row in history[:5] + [["...", "...", "...", "...", "...", "..."]] + history[-2:]:
        if row[0] == "...":
            print(f"{'...':<6}{'...':<12}{'...':<12}{'...':<12}{'...':<12}{'...':<12}")
        else:
            print(f"{row[0]:<6}{row[1]:<12.4e}{row[2]:<12.4e}{row[3]:<12.6f}{row[4]:<12.4e}{row[5]:<12.4e}")
    print("\n")

print_table("Bisection Method", bi_hist)
print_table("False Position Method", fp_hist)

# Comparison Table
print("--- Comparison Table ---")
print(f"{'Method':<16}{'Est. Root':<12}{'Iters':<8}{'f(Xr)':<12}{'L-Upd':<8}{'U-Upd':<8}")
print(f"{'Bisection':<16}{bi_hist[-1][3]:<12.6f}{len(bi_hist):<8}{bi_hist[-1][5]:<12.4e}{bi_l:<8}{bi_u:<8}")
print(f"{'False Position':<16}{fp_hist[-1][3]:<12.6f}{len(fp_hist):<8}{fp_hist[-1][5]:<12.4e}{fp_l:<8}{fp_u:<8}\n")

# Plotting on Log Scale
x_vals = np.logspace(-4, 4, 1000)
plt.figure(figsize=(7, 4))
plt.plot(x_vals, f1(x_vals), label=r'$f(x)=\ln(x)$', color='blue')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xscale('log')
plt.title('Logarithmic Scale Plot of f(x) = ln(x)')
plt.xlabel('x (Log Scale)')
plt.ylabel('f(x)')
plt.grid(True, which="both", ls="-")
plt.legend()
plt.show()

```

---

## Problem 2: Tracking Multiple Roots

### 📋 Formal Question Statement

Given the function $f(x) = 0.6\ln(x+1) - C\sin(1.7x) - 0.08x^2 - 0.08$ (where $C = 1.0$ is assumed as the missing coefficient value) within the interval $0 \le x \le 10$:

1. Plot the function across the entire domain $0 \le x \le 10$.
2. Automatically locate all sub-intervals containing a root using an incremental search step size of $\Delta x = 0.1$.
3. Run the **Bisection Method** individually on each discovered sign-change sub-interval to find all roots.
4. Output professional structured iteration tables for each separate root.
5. Answer the conceptual problem regarding running a single global bisection execution.

---

### 📝 Written Answer (For your exam paper)

* **Q: What problem would occur if Bisection was run once on the entire $[0, 10]$ interval?**
1. **Missed Roots:** The Bisection method only finds *one* root per interval initialization. Since this function crosses the x-axis multiple times within $[0, 10]$, running it globally would completely ignore all other intermediate roots.
2. **Initialization Failure:** If the signs at the outer bounds happen to be identical ($f(0) \cdot f(10) > 0$), the Bisection method will fail to initialize altogether and output an error, despite roots existing inside.



---

### 💻 Python Code for Problem 2

```python
import numpy as np
import matplotlib.pyplot as plt

# Define Function (Change 1.0 if your 'sth' coefficient is different)
def f2(x):
    return 0.6 * np.log(x + 1) - 1.0 * np.sin(1.7 * x) - 0.08 * x**2 - 0.08

# 1. Plotting the graph
x_plot = np.linspace(0, 10, 1000)
plt.figure(figsize=(7, 4))
plt.plot(x_plot, f2(x_plot), label='f(x)', color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Graph of f(x) over [0, 10]')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

# 2. Incremental Search for Sign Changes (Step Size = 0.1)
step = 0.1
intervals = []
x_search = np.arange(0, 10.0, step)

for i in range(len(x_search) - 1):
    x1, x2 = x_search[i], x_search[i+1]
    if f2(x1) * f2(x2) < 0:
        intervals.append((x1, x2))

print(f"Discovered sign change intervals: {[(round(a,1), round(b,1)) for a, b in intervals]}\n")

# 3 & 4. Run Bisection on each interval
def bisection_solver(xl, xu, tol=0.0001):
    xr_old = None
    print(f"--- Iteration Table for Interval [{xl:.1f}, {xu:.1f}] ---")
    print(f"{'Iter':<6}{'Xl':<10}{'Xu':<10}{'Xr':<10}{'Ea (%)':<10}{'f(Xr)':<10}")
    
    for idx in range(1, 100):
        xr = (xl + xu) / 2.0
        ea = abs((xr - xr_old) / xr) * 100 if xr_old is not None else 100.0
        
        print(f"{idx:<6}{xl:<10.4f}{xu:<10.4f}{xr:<10.4f}{ea:<10.4f}{f2(xr):<10.4f}")
        
        if ea <= tol and xr_old is not None:
            break
            
        if f2(xl) * f2(xr) < 0:
            xu = xr
        else:
            xl = xr
        xr_old = xr
    print("\n")

for start, end in intervals:
    bisection_solver(start, end)

```

---

## Problem 3: Automated Search & Execution

### 📋 Formal Question Statement

Write a generalized program to automatically locate all root-containing intervals for a given continuous function. Upon identifying the **first** sign-change interval, automatically execute the **False Position Method** until the approximate relative error $\varepsilon_a$ falls below the threshold of $0.0001$. Generate a visual plot of the function.

---

### 💻 Python Code for Problem 3

*(Note: Since the explicit equation was missing from your chat log snippet, a standard textbook multi-root function $f(x) = 2x^3 - 11.7x^2 + 17.7x - 5$ is included below as a baseline template. You can easily replace the math expression inside `f3(x)` with your specific question paper formula).*

```python
import numpy as np
import matplotlib.pyplot as plt

# Representative equation template (Swap out with your exact equation)
def f3(x):
    return 2*x**3 - 11.7*x**2 + 17.7*x - 5

# 1. Automated Search Loop
step_size = 0.1
domain_start, domain_end = 0, 4
x_domain = np.arange(domain_start, domain_end, step_size)
found_intervals = []

for i in range(len(x_domain) - 1):
    x_l, x_u = x_domain[i], x_domain[i+1]
    if f3(x_l) * f3(x_u) < 0:
        found_intervals.append((x_l, x_u))

# 2. False Position on the FIRST interval found
if found_intervals:
    xl, xu = found_intervals[0]
    print(f"First interval detected at: [{xl:.2f}, {xu:.2f}]")
    print(f"\n{'Iter':<6}{'Xl':<10}{'Xu':<10}{'Xr':<10}{'Ea':<12}{'f(Xr)':<10}")
    
    xr_old = None
    threshold = 0.0001
    
    for idx in range(1, 100):
        fl, fu = f3(xl), f3(xu)
        xr = xu - (fu * (xl - xu)) / (fl - fu)
        
        # Absolute relative error calculation
        ea = abs((xr - xr_old) / xr) if xr_old is not None else 1.0
        
        print(f"{idx:<6}{xl:<10.4f}{xu:<10.4f}{xr:<10.4f}{ea:<12.6f}{f3(xr):<10.4f}")
        
        if ea <= threshold and xr_old is not None:
            print(f"\nConverged to Root: {xr:.5f} at iteration {idx}")
            break
            
        if f3(xl) * f3(xr) < 0:
            xu = xr
        else:
            xl = xr
        xr_old = xr
else:
    print("No sign changes found in the defined domain range.")

# 3. Plotting
x_grid = np.linspace(domain_start, domain_end, 500)
plt.figure(figsize=(7, 4))
plt.plot(x_grid, f3(x_grid), color='green', label='Function Plot')
plt.axhline(0, color='black', linestyle='--')
if found_intervals:
    plt.axvspan(found_intervals[0][0], found_intervals[0][1], color='yellow', alpha=0.3, label='Target Interval')
plt.grid(True)
plt.legend()
plt.show()

```