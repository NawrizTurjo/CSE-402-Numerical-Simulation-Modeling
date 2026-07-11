Your unified codebase in the `codes/` folder provides a strong foundation for the exam. However, matching your current files against the full syllabus requirements and the real-world traps found in your senior batch resources reveals a few critical gaps.

If your professor introduces edge-case failure situations or requests actual implementations of the fixes discussed in theory classes, the current templates could fall short or crash mid-exam.

Instruct your AI agent to apply the following **5 specific enhancements and scripts** to extend and safeguard your codebase.

---

### 1. Fix the "Zero-Division Error Trap" in Iteration Loops

* **Target File:** `codes/methods/03_bisection_and_false_position.py` and `04_newton_raphson.py`
* **The Gap:** The standard relative approximate error formula is $\varepsilon_a = \vert{}\frac{x_{new} - x_{old}}{x_{new}}\vert{} \times 100\%$. If a function converges exactly onto $0$, or hits $0$ during an intermediate oscillation step (such as the malicious exam problem $f(x) = x^3 - 2x + 2$ with $x_0 = 0 \rightarrow x_1 = 1 \rightarrow x_2 = 0$), the denominator becomes zero, causing your code to crash with a `ZeroDivisionError`.
* **The Code Update:** Replace your error calculations with a safe handler:

```python
# Safe approximate relative error calculation
if xr_old is None:
    ea = 100.0
elif abs(xr) < 1e-12:
    # If the new root estimate hits zero, use absolute difference or a small epsilon to avoid crash
    ea = abs(xr - xr_old) * 100.0 
else:
    ea = abs((xr - xr_old) / xr) * 100.0

```

---

### 2. Implement the "Illinois Modification" for False Position

* **Target File:** `codes/methods/03_bisection_and_false_position.py`
* **The Gap:** Your study guide explicitly warns about the **Stagnation Problem** where concave/convex functions leave one endpoint completely frozen with 0 updates. While theory tests ask for text explanations, lab exams often ask you to *program the fix*. The standard way to resolve this is the **Illinois Method**, which halves the function value at the stagnant bound to force the secant line to switch sides.
* **The Extension Code:** Add this function to your root-finding script:

```python
def false_position_illinois(f, xl, xu, tol=0.0001, max_iter=100):
    """
    Modified False Position Method (Illinois variant) to eliminate boundary stagnation.
    """
    xr_old = None
    fl = f(xl)
    fu = f(xu)
    
    print(f"{'Iter':<6}{'Xl':<10}{'Xu':<10}{'Xr':<12}{'Ea (%)':<12}{'f(Xr)':<12}")
    
    for i in range(1, max_iter + 1):
        xr = xu - (fu * (xl - xu)) / (fl - fu)
        fxr = f(xr)
        
        if xr_old is not None and abs(xr) > 1e-12:
            ea = abs((xr - xr_old) / xr) * 100
        else:
            ea = 100.0
            
        print(f"{i:<6}{xl:<10.4f}{xu:<10.4f}{xr:<12.6f}{ea:<12.4e}{fxr:<12.4e}")
        
        if xr_old is not None and ea <= tol:
            break
            
        if fl * fxr < 0:
            xu = xr
            fu = fxr
            fl = fl / 2.0  # ← The Illinois Fix: Halve the inactive bound's weight
        else:
            xl = xr
            fl = fxr
            fu = fu / 2.0  # ← The Illinois Fix: Halve the inactive bound's weight
            
        xr_old = xr
    return xr

```

---

### 3. Upgrade Newton-Raphson with Multiplicity ($m$) & Oscillation Detection

* **Target File:** `codes/methods/04_newton_raphson.py`
* **The Gap:** The codebase handles standard Newton-Raphson well, but fails to recover if the question features a **multiple root** (which degrades convergence speed) or an **oscillating cycle trap** (which loops infinitely between two coordinates).
* **The Extension Code:** Update your Newton-Raphson template to safely manage multiplicity factors ($m$) and track visited values to flag infinite cycles:

```python
def advanced_newton_raphson(f, df, x0, m=1, tol=0.05, max_iter=50):
    """
    Newton-Raphson with Multiplicity modifier (m) and infinite loop detection.
    """
    x_old = x0
    history = [x0]
    
    print(f"{'Iter':<6}{'x_i':<12}{'f(x_i)':<12}{'f\'(x_i)':<12}{'x_next':<12}{'Ea (%)':<12}")
    
    for i in range(1, max_iter + 1):
        f_val = f(x_old)
        df_val = df(x_old)
        
        if abs(df_val) < 1e-12:
            print(f"\n[CRITICAL] Horizontal tangent encountered at x = {x_old}. Division by zero!")
            return None
            
        # m=1 is standard, m > 1 restores quadratic speed for multiple roots
        x_new = x_old - m * (f_val / df_val) 
        
        ea = abs((x_new - x_old) / x_new) * 100 if abs(x_new) > 1e-12 else abs(x_new - x_old) * 100
        
        print(f"{i:<6}{x_old:<12.6f}{f_val:<12.4e}{df_val:<12.4e}{x_new:<12.6f}{ea:<12.4f}%")
        
        if ea <= tol:
            print(f"\nConverged to Root: {x_new:.6f}")
            return x_new
            
        # Infinite Loop / Ping-Pong Cycle Detection Trap
        for past_x in history[:-1]:
            if abs(x_new - past_x) < 1e-6:
                print(f"\n[WARNING] Infinite oscillation cycle detected! (Jumping back to {x_new:.4f})")
                print("Fix: Choose a different initial guess closer to the true sign-change bracket.")
                return None
                
        history.append(x_new)
        x_old = x_new
        
    print("\n[ERROR] Maximum iterations reached without convergence.")
    return None

```

---

### 4. Separate your "Multi-Root Incremental Scanner" Utility

* **Target File:** `codes/methods/03_bisection_and_false_position.py` (or create a new `codes/utils/scanner.py`)
* **The Gap:** Several past paper questions require you to scan a range like $[0, 10]$ with a specific step size ($\Delta x = 0.1$) to capture all root intervals before passing them into the root-finders. Having this snippet hardcoded inside one single method makes it difficult to reuse quickly when switching between Bisection and False Position during a timed test.
* **The Extension Code:** Build a clean, decoupled function:

```python
import numpy as np

def find_sign_change_intervals(f, start, end, step=0.1):
    """
    Scans a domain sequentially to identify all sub-brackets containing a root.
    """
    x_domain = np.arange(start, end + step, step)
    intervals = []
    
    for i in range(len(x_domain) - 1):
        x_left = x_domain[i]
        x_right = x_domain[i+1]
        
        # Avoid computational evaluation precisely on illegal zero bounds if applicable
        if f(x_left) * f(x_right) < 0:
            intervals.append((x_left, x_right))
            
    return intervals

```

---

### 5. Add a "Truncation Error vs. Round-off Error" Tradeoff Simulation

* **Target File:** Create `codes/basic/08_error_tradeoff_plotter.py`
* **The Gap:** The syllabus explicitly outlines **Approximations, round-off errors, and truncation errors** as primary learning outcomes. Lab exams frequently ask students to demonstrate how changing the step size $h$ affects numerical differentiation, specifically pointing out how a tiny $h$ causes round-off noise, while a large $h$ causes truncation errors. Having a script ready to generate this tradeoff curve will save you time if this requirement appears on your assignment sheet.
* **The Code Update:** Have your agent write this quick visualization script:

```python
import numpy as np
import matplotlib.pyplot as plt

# Test function: f(x) = exp(x), true derivative at x=1 is exp(1)
def f_test(x): return np.exp(x)
true_deriv = np.exp(1.0)
x_target = 1.0

# Generate a wide range of step sizes from 10^0 down to 10^-20
h_values = np.logspace(0, -20, 100)
errors = []

for h in h_values:
    # Forward difference approximation
    approx_deriv = (f_test(x_target + h) - f_test(x_target)) / h
    abs_error = abs(true_deriv - approx_deriv)
    errors.append(abs_error)

# Generate the classic V-shaped Error Tradeoff Curve
plt.figure(figsize=(7, 4.5))
plt.loglog(h_values, errors, color='darkblue', linewidth=2, label='Total Numerical Error')
plt.xlabel('Step Size $h$ (Log Scale)')
plt.ylabel('Absolute Error (Log Scale)')
plt.title('Truncation vs. Round-off Error Tradeoff Curve')
plt.gca().invert_xaxis() # View from large h down to tiny h
plt.grid(True, which="both", ls="--", alpha=0.5)

# Text annotations for the exam paper explanation
plt.text(1e-2, 1e-1, '← Truncation Error Dominated\n(h is too large)', color='darkred')
plt.text(1e-16, 1e-1, 'Round-off Error Dominated →\n(Floating-point limits hit)', color='purple')

plt.tight_layout()
plt.savefig('error_tradeoff_curve.png', dpi=150)
print("[SUCCESS] Saved convergence tradeoff plot as 'error_tradeoff_curve.png'.")
plt.show()

```