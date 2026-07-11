# Newton-Raphson Method: Breaking Functions & Failure Situations

The Newton-Raphson method is powerful due to its quadratic convergence, but it has several notorious failure conditions. If you see these in an exam, the examiner is testing your theoretical knowledge of the method's limitations.

---

## 1. The Oscillating Trap (Infinite Cycles)
**Function Example:** $f(x) = x^3 - 2x + 2$
**Initial Guess:** $x_0 = 0$

* **What Happens:** The method enters an infinite loop jumping between two values. 
  * $x_0 = 0 \implies x_1 = 1$
  * $x_1 = 1 \implies x_2 = 0$
* **The Reason:** The tangents at $x=0$ and $x=1$ point directly at each other on the x-axis, creating a closed geometric loop that never approaches the actual root ($x \approx -1.769$).
* **Fix in Exam:** State that the method is trapped in a 2-cycle. The fix is to **choose a different initial guess** that is closer to the true root (e.g., $x_0 = -1.5$).

---

## 2. Division by Zero (Horizontal Tangent)
**Function Example:** $f(x) = \sin(x)$ or $f(x) = x^3 - 3x$
**Initial Guess:** $x_0 = \frac{\pi}{2}$ or $x_0 = 1$

* **What Happens:** The formula crashes immediately with a `Division by Zero` error.
* **The Reason:** You selected a starting point (or landed on a point) that is a local minimum or maximum. At this point, the derivative $f'(x) = 0$. The tangent line is perfectly horizontal and will never intersect the x-axis to give you the next $x$.
* **Fix in Exam:** State that $f'(x_0) = 0$, making the formula mathematically undefined. The fix is to **shift the initial guess slightly** (e.g., $x_0 = 1.1$ instead of $1.0$).

---

## 3. The Inflection Point Divergence (Vertical Tangent)
**Function Example:** $f(x) = \sqrt[3]{x}$ or $f(x) = x^{1/3}$
**Initial Guess:** Any guess $x_0 \neq 0$ (e.g., $x_0 = 0.1$)

* **What Happens:** The sequence diverges rapidly, getting further away from zero each time ($0.1 \rightarrow -0.2 \rightarrow 0.4 \rightarrow -0.8$).
* **The Reason:** The root is at an inflection point where the derivative approaches infinity (a vertical tangent). The Newton-Raphson formula $x_{n+1} = x_n - \frac{x_n^{1/3}}{\frac{1}{3}x_n^{-2/3}}$ simplifies to $x_{n+1} = -2x_n$. Every iteration perfectly doubles the distance from the root!
* **Fix in Exam:** Show 3-4 iterations to prove it diverges. Conclude that Newton-Raphson fails for fractional powers $< 1$ at the root. The fix is to **abandon Newton-Raphson and use a bracketing method** (like Bisection or False-Position).

---

## 4. Overshooting due to Flat Regions
**Function Example:** $f(x) = \arctan(x)$ 
**Initial Guess:** $x_0 = 1.5$

* **What Happens:** The guesses shoot wildly toward infinity.
* **The Reason:** If you start too far away from the root on a curve that flattens out, the slope $f'(x)$ is very small. Dividing by a tiny slope sends the next guess extremely far away.
* **Fix in Exam:** Plot the graph or use a sign-change method first to guarantee your initial guess is close to the root, avoiding the "flat" tails of the function.

---

## 5. The Multiple Roots Curse (Loss of Speed)
**Function Example:** $f(x) = (x - 2)^3$
**Initial Guess:** $x_0 = 3$

* **What Happens:** The method doesn't break, but it becomes painfully slow. Instead of converging in 3 steps, it might take 30 steps.
* **The Reason:** When a root has multiplicity $m > 1$, both $f(x)$ and $f'(x)$ approach zero as you get closer to the root. This strips Newton-Raphson of its famous "quadratic convergence," downgrading it to slow "linear convergence."
* **Fix in Exam:** If asked why it is slow, state that it is a multiple root. The fix is to use the **Modified Newton-Raphson formula** for multiple roots: 
  $$x_{n+1} = x_n - m \frac{f(x_n)}{f'(x_n)}$$ 
  *(where $m$ is the multiplicity, which is 3 in this example).*
