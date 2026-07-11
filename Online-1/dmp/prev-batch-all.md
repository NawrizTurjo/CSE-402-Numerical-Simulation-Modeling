```python
import numpy as np

# Problem 1 verification
def f1(h):
    return np.pi * h**3 - 12 * np.pi * h**2 + 15

def df1(h):
    return 3 * np.pi * h**2 - 24 * np.pi * h

h = 0.5  # initial guess
for i in range(5):
    h_new = h - f1(h)/df1(h)
    ea = abs((h_new - h)/h_new)*100
    print(f"h={h_new:.5f}, ea={ea:.4f}%")
    h = h_new

print("---")
# Problem 2 verification
def f2(x):
    return 298*x - 3*(x**(2/3)) - 1000

def df2(x):
    return 298 - 2*(x**(-1/3))

x = 4.0 # initial guess
for i in range(5):
    x_new = x - f2(x)/df2(x)
    ea = abs((x_new - x)/x_new)*100
    print(f"x={x_new:.5f}, ea={ea:.4f}%")
    x = x_new

print("---")
# Problem 3 verification
def f3(x):
    return x**3 - x - 1

xl, xu = 1.0, 2.0
xr_old = None
for i in range(5):
    fl, fu = f3(xl), f3(xu)
    xr = xu - (fu * (xl - xu)) / (fl - fu)
    ea = abs((xr - xr_old)/xr)*100 if xr_old is not None else 100.0
    print(f"xr={xr:.5f}, ea={ea:.4f}%")
    if f3(xl)*f3(xr) < 0:
        xu = xr
    else:
        xl = xr
    xr_old = xr



```

```text
h=0.66886, ea=25.2457%
h=0.64883, ea=3.0862%
h=0.64855, ea=0.0433%
h=0.64855, ea=0.0000%
h=0.64855, ea=0.0000%
---
x=3.37844, ea=18.3977%
x=3.37837, ea=0.0022%
x=3.37837, ea=0.0000%
x=3.37837, ea=0.0000%
x=3.37837, ea=0.0000%
---
xr=1.16667, ea=100.0000%
xr=1.25311, ea=6.8985%
xr=1.29344, ea=3.1177%
xr=1.31128, ea=1.3608%
xr=1.31899, ea=0.5843%


```

Here are the problem statements and complete Python code solutions for the three assignments provided in your files.

---

## 📌 Problem 1: Spherical Oil Tank Dipstick Design (online_A2.pdf)

### 📋 Problem Statement

A spherical oil tank is $8\text{ ft}$ in diameter (radius $r = 4\text{ ft}$). We want to design a dipstick to measure the volume $V$ of oil in the tank based on the wet height $h$. The volume of the oil left is given by:

$$V = \frac{\pi h^2(3r - h)}{3}$$

Assuming a target volume $V = 5\text{ ft}^3$:

1. Formulate a non-linear equation $f(h) = 0$.


2. Use the **Newton-Raphson method** to find $h$ with an error tolerance of $\varepsilon_s \le 0.05\%$.


3. Print the root estimate and absolute approximate relative error ($\varepsilon_a$) after every iteration.


4. Plot the graph of $f(h)$ and explain the selection of the initial guess.



### 📐 Mathematical Formulation

Given $r = 4\text{ ft}$ and $V = 5\text{ ft}^3$:


$$5 = \frac{\pi h^2(3(4) - h)}{3} \implies 15 = 12\pi h^2 - \pi h^3$$

* **Objective Function:** $f(h) = \pi h^3 - 12\pi h^2 + 15 = 0$
* **Derivative:** $f'(h) = 3\pi h^2 - 24\pi h$

### 💻 Python Code Solution

```python
import numpy as np
import matplotlib.pyplot as plt

# Define function and its derivative
def f(h):
    return np.pi * h**3 - 12 * np.pi * h**2 + 15

def df(h):
    return 3 * np.pi * h**2 - 24 * np.pi * h

# Newton-Raphson Configuration
h_old = 0.5  # Initial guess chosen based on graph inspection (0 < h < 8)
tol = 0.05   # 0.05%
max_iter = 20

print("--- Newton-Raphson Iterations (Problem A2) ---")
print(f"{'Iteration':<10}{'Root Estimate (h)':<20}{'Approx. Error (Ea %)':<20}")

for i in range(1, max_iter + 1):
    h_new = h_old - f(h_old) / df(h_old)
    ea = abs((h_new - h_old) / h_new) * 100
    
    print(f"{i:<10}{h_new:<20.6f}{ea:<20.4f}%")
    
    if ea <= tol:
        print(f"\nConverged to root h = {h_new:.6f} ft in {i} iterations.\n")
        break
    h_old = h_new

# Plotting the graph
h_vals = np.linspace(0, 2, 500)
plt.figure(figsize=(6, 4))
plt.plot(h_vals, f(h_vals), label='$f(h) = \\pi h^3 - 12\\pi h^2 + 15$', color='blue')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Graph of $f(h)$ vs $h$')
plt.xlabel('Height $h$ (ft)')
plt.ylabel('$f(h)$')
plt.grid(True)
plt.legend()
plt.show()

```

### 🔍 Choice of Initial Guess

Since the tank diameter is $8\text{ ft}$, $h$ must lie strictly within the physical boundaries $[0, 8]$. Looking at the function values, $f(0) = 15$ (positive) and $f(1) \approx -19.55$ (negative). Because a sign change occurs within $[0, 1]$, a root exists near the bottom of the tank. An initial guess of **$h_0 = 0.5\text{ ft}$** is perfectly positioned within this range to guarantee fast, stable convergence without encountering the local extrema or flat slope regions where $f'(h) = 0$.

---

## 📌 Problem 2: Chemical Firm Break-Even Point (online_B2.pdf)

### 📋 Problem Statement

It costs a firm $C(q)$ dollars to produce $q$ grams per day of a certain chemical:


$$C(q) = 1000 + 2q + 3q^{2/3}$$



The firm sells the chemical at $300\text{ BDT}$ per gram. Let the break-even point quantity be $x$.

1. Formulate the non-linear break-even relation where $\text{Revenue} = \text{Cost}$.


2. Use the **Newton-Raphson method** to calculate $x$ with an error tolerance of $\varepsilon_s \le 0.05\%$.


3. Print the root estimate and absolute relative approximate error ($\varepsilon_a$) after every iteration.


4. Plot the graph of $f(x)$ and justify the choice of initial guess.



### 📐 Mathematical Formulation

$$\text{Revenue } R(x) = 300x$$

At the break-even point, $R(x) - C(x) = 0$:


$$300x - (1000 + 2x + 3x^{2/3}) = 0$$

* **Objective Function:** $f(x) = 298x - 3x^{2/3} - 1000 = 0$
* **Derivative:** $f'(x) = 298 - 2x^{-1/3}$

### 💻 Python Code Solution

```python
import numpy as np
import matplotlib.pyplot as plt

# Define function and its derivative
def f(x):
    return 298 * x - 3 * (x**(2/3)) - 1000

def df(x):
    return 298 - 2 * (x**(-1/3))

# Newton-Raphson Configuration
x_old = 4.0   # Initial guess based on boundary observation
tol = 0.05    # 0.05%
max_iter = 20

print("--- Newton-Raphson Iterations (Problem B2) ---")
print(f"{'Iteration':<10}{'Root Estimate (x)':<20}{'Approx. Error (Ea %)':<20}")

for i in range(1, max_iter + 1):
    x_new = x_old - f(x_old) / df(x_old)
    ea = abs((x_new - x_old) / x_new) * 100
    
    print(f"{i:<10}{x_new:<20.6f}{ea:<20.4f}%")
    
    if ea <= tol:
        print(f"\nConverged to break-even point x = {x_new:.6f} grams in {i} iterations.\n")
        break
    x_old = x_new

# Plotting the graph
x_vals = np.linspace(1, 10, 500)
plt.figure(figsize=(6, 4))
plt.plot(x_vals, f(x_vals), label='$f(x) = 298x - 3x^{2/3} - 1000$', color='green')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Graph of $f(x)$ vs Production $x$')
plt.xlabel('Quantity $x$ (grams)')
plt.ylabel('$f(x)$')
plt.grid(True)
plt.legend()
plt.show()

```

### 🔍 Choice of Initial Guess

Evaluating the function near zero shows $f(1) = 298 - 3 - 1000 = -705$ (negative). For a slightly larger production value like $x = 4$, $f(4) \approx 298(4) - 3(2.52) - 1000 = 1192 - 7.56 - 1000 = 184.44$ (positive). A guaranteed root exists within the interval $[1, 4]$. Choosing **$x_0 = 4.0$** ensures a safe initial slope value where $f'(x)$ is strong, preventing mathematical division errors and achieving fast quadratic convergence.

---

## 📌 Problem 3: False Position Method Solver (online_A1.pdf)

### 📋 Problem Statement

Implement the **False Position Method** to locate the real roots of the non-linear cubic equation:


$$f(x) = x^3 - x - 1 = 0$$

1. Plot the function to visually identify the root location.


2. Identify initial bounds $x_L$ and $x_U$ such that $f(x_L) \cdot f(x_U) < 0$.


3. Output the approximate error ($\varepsilon_a$) after every iteration.



### 💻 Python Code Solution

```python
import numpy as np
import matplotlib.pyplot as plt

# Define function
def f(x):
    return x**3 - x - 1

# False Position Implementation
xl = 1.0  # f(1) = -1
xu = 2.0  # f(2) = 5
tol = 0.05  # Setting a standard engineering tolerance of 0.05%
max_iter = 50

print("--- False Position Iterations (Problem A1) ---")
print(f"{'Iter':<6}{'X_L':<10}{'X_U':<10}{'X_r':<12}{'Approx Error':<15}{'f(X_r)':<12}")

xr_old = None

for idx in range(1, max_iter + 1):
    fl, fu = f(xl), f(xu)
    
    # False Position Formula
    xr = xu - (fu * (xl - xu)) / (fl - fu)
    fr = f(xr)
    
    # Error Calculation
    if xr_old is not None:
        ea = abs((xr - xr_old) / xr) * 100
        error_str = f"{ea:.4f}%"
    else:
        ea = float('inf')
        error_str = "Initial"
        
    print(f"{idx:<6}{xl:<10.4f}{xu:<10.4f}{xr:<12.6f}{error_str:<15}{fr:<12.4e}")
    
    if ea <= tol:
        print(f"\nConverged to real root x = {xr:.6f} in {idx} iterations.\n")
        break
        
    # Standard Bracket Update
    if fl * fr < 0:
        xu = xr
    else:
        xl = xr
        
    xr_old = xr

# Plotting the graph
x_vals = np.linspace(0, 2.5, 500)
plt.figure(figsize=(6, 4))
plt.plot(x_vals, f(x_vals), label='$f(x) = x^3 - x - 1$', color='crimson')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.axvspan(1.0, 2.0, color='yellow', alpha=0.2, label='Initial Bracket [1, 2]')
plt.title('Graph of $f(x) = x^3 - x - 1$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

```

### 🔍 Choice of Initial Guess

Evaluating the boundaries reveals that $f(1) = 1 - 1 - 1 = -1$ (negative) and $f(2) = 8 - 2 - 1 = 5$ (positive). Since $f(1) \cdot f(2) < 0$, Intermediate Value Theorem dictates that at least one real root resides inside this domain. Thus, **$x_L = 1.0$** and **$x_U = 2.0$** are chosen as optimal starting brackets.