# Section A: The Newton-Raphson Nightmare

Given the function:
$$f(x) = x^3 - 2x + 2$$

(1) Plot the graph for the range -3 <= x <= 3.
(2) Find all sign-change intervals using a step size of 0.5.
(3) Apply the Newton-Raphson method to find the root. **Use the initial guess x_0 = 0.**
(4) For every iteration, print the table: `| Iter | x | x_new | f(x_new) | Ea% |`
(5) Given a stop criteria of Ea <= 1%, explain what happens during the iterations. Why does this phenomenon occur, and how should it be fixed?

---

## Why this will cause a disaster in the exam hall:

### 1. The Infinite Cycle Trap:
When students calculate the first iteration:
* **x_0 = 0**
* f(0) = 2
* f'(x) = 3x^2 - 2  =>  f'(0) = -2
* x_1 = 0 - (2 / -2) = 1

Second iteration:
* **x_1 = 1**
* f(1) = 1^3 - 2(1) + 2 = 1
* f'(1) = 3(1)^2 - 2 = 1
* x_2 = 1 - (1 / 1) = 0

**The method gets stuck in an infinite loop:** 0 -> 1 -> 0 -> 1 -> 0...
Students will recalculate this 5 times, think their calculators are broken, or think they forgot how to do basic calculus!

### 2. The Ea Division by Zero Trap:
When calculating the relative error for the second iteration:
Ea = | (x_new - x_old) / x_new | * 100%
Ea = | (0 - 1) / 0 | * 100% = **Division by Zero!**
They literally cannot calculate the error.

### 3. The Solution (Question 5):
The sign-change interval they find in step (2) will be `[-2.0, -1.5]`. The question maliciously tricks them into using an initial guess (x_0 = 0) that is far outside the bracketed root! The fix is simply to choose a guess inside the sign-change interval, like x_0 = -1.5.

---

## ⚠️ A Warning About Your Current Python Code!
If you use:
```python
f = lambda x: x ** (1/3)
```
You walk into **two** massive traps:

1. **The Python Complex Number Trap**: In Python, evaluating `(-1) ** (1/3)` does **not** give `-1`. It gives a complex number `(0.5 + 0.866j)`. Your `plt.plot()` will throw a warning and the left side of your graph will be completely empty! You must use `np.cbrt(x)` instead.
2. **The Newton-Raphson Divergence Trap**: The cube root function is the ultimate Newton-Raphson breaker. If you run Newton-Raphson on $f(x) = \sqrt[3]{x}$ starting at $x = 0.1$, it will diverge to infinity!
   * x_0 = 0.1
   * x_1 = -0.2
   * x_2 = 0.4
   * x_3 = -0.8
   * x_4 = 1.6 ...

If you want an algebraic nightmare question instead (like Section B but for Newton-Raphson), use: 
**f(x) = 0.6 * ln(x^2 + 1) - cos(1.7x) - 0.08**. 
The derivative f'(x) requires the chain rule and quotient/product rule, guaranteeing algebraic mistakes by 90% of students!
