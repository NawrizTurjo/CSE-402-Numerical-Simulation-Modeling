"""
QUESTION  (this is the exact pattern seen in a previous lab test, "sampleB")
------------------------------------------------------------------------------
    f(x) = 0.6*ln(x+1) - sth*sin(1.7x) - 0.08*x^2 - 0.08

NOTE on "sth": in the real exam this is a placeholder the teacher wants you
to replace with a small constant tied to your own student ID (e.g. the last
digit of your ID divided by 10, or a value given on the question paper).
Here we simply pick STH = 0.5 as an example so the file is runnable -- on
the real exam day, just change the STH constant below to whatever value
you're told to use; every other line of code stays exactly the same.

(1) Plot f(x) for 0 <= x <= 10.
(2) Using step size 0.1, scan the whole interval and find every
    sign-change sub-interval (bracket).
(3) Run the bisection method on each sign-change bracket found.
(4) Collect and report every root.
(5) Print a single results table listing every root (like the class table).
(6) Discuss: what would go wrong if you ran bisection directly on the
    single interval [0, 10] instead of scanning first?
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots", "bisection")
os.makedirs(PLOTS_DIR, exist_ok=True)

STH = 0.5   # <-- replace with your own ID-based value on exam day


def f(x):
    return 0.6 * np.log(x + 1) - STH * np.sin(1.7 * x) - 0.08 * x ** 2 - 0.08


# ---------------------------------------------------------------------------
# (1) plot f(x) over [0, 10]
# ---------------------------------------------------------------------------
x_plot = np.linspace(0, 10, 1000)
y_plot = f(x_plot)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_plot, y_plot, color="tab:blue", label="f(x)")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title(f"f(x) = 0.6ln(x+1) - {STH}*sin(1.7x) - 0.08x^2 - 0.08")
ax.grid(True, alpha=0.3)
ax.legend()
fig.savefig(os.path.join(PLOTS_DIR, "problem2_sampleB_function_plot.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# (2) scan [0, 10] with step 0.1 for sign-change brackets
# ---------------------------------------------------------------------------
def find_sign_change_intervals(func, a, b, step):
    brackets = []
    x = a
    while x < b:
        x_next = round(min(x + step, b), 10)
        if func(x) * func(x_next) < 0:
            brackets.append((x, x_next))
        x = x_next
    return brackets


brackets = find_sign_change_intervals(f, 0, 10, 0.1)
print("Sign-change intervals found (step = 0.1):")
for br in brackets:
    print(f"  [{br[0]:.2f}, {br[1]:.2f}]   f(xl)={f(br[0]):.5f}  f(xu)={f(br[1]):.5f}")

# ---------------------------------------------------------------------------
# (3) bisection on each bracket
# ---------------------------------------------------------------------------
def bisection(func, xl, xu, tol=1e-6, max_iter=100):
    xm_old = None
    table = []
    for i in range(1, max_iter + 1):
        xm = (xl + xu) / 2
        fxm = func(xm)
        eps_a = None if xm_old is None else abs((xm - xm_old) / xm) * 100
        table.append((i, xl, xu, xm, fxm, eps_a))

        if fxm == 0:
            break
        if func(xl) * fxm < 0:
            xu = xm
        else:
            xl = xm
        if eps_a is not None and eps_a < tol:
            break
        xm_old = xm
    return xm, table


all_roots = []
print("\nDetailed iteration tables per bracket:")
for br in brackets:
    root, table = bisection(f, br[0], br[1], tol=1e-4)
    all_roots.append(root)
    print(f"\n Bracket [{br[0]:.2f}, {br[1]:.2f}]")
    print(f"  {'iter':>4} {'xl':>9} {'xu':>9} {'xm':>10} {'f(xm)':>12} {'eps_a %':>9}")
    for row in table:
        i, xl_, xu_, xm_, fxm_, eps_a_ = row
        eps_str = "-" if eps_a_ is None else f"{eps_a_:.4f}"
        print(f"  {i:>4} {xl_:>9.4f} {xu_:>9.4f} {xm_:>10.6f} {fxm_:>12.3e} {eps_str:>9}")

# ---------------------------------------------------------------------------
# (4) + (5) master results table
# ---------------------------------------------------------------------------
print("\n================ ALL ROOTS FOUND ================")
print(f"{'#':>3} {'bracket':>16} {'root':>12} {'f(root)':>14}")
for idx, (br, root) in enumerate(zip(brackets, all_roots), start=1):
    print(f"{idx:>3} {str(br):>16} {root:>12.6f} {f(root):>14.3e}")

# mark roots on the plot too
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_plot, y_plot, color="tab:blue", label="f(x)")
ax.axhline(0, color="black", linewidth=0.8)
ax.scatter(all_roots, [f(r) for r in all_roots], color="red", zorder=5, label="roots")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("All roots found via bisection")
ax.grid(True, alpha=0.3)
ax.legend()
fig.savefig(os.path.join(PLOTS_DIR, "problem2_sampleB_roots_marked.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# (6) discussion: bisection directly on [0, 10] as ONE bracket
# ---------------------------------------------------------------------------
print("\n================ PART 6 DISCUSSION ================")
print(f"f(0)  = {f(0):.6f}")
print(f"f(10) = {f(10):.6f}")
print(f"f(0) * f(10) = {f(0) * f(10):.6f}")
print("""
If we tried to run bisection directly on the single interval [0, 10]
WITHOUT scanning for sign changes first, two different problems can occur:

  (a) If f(0) and f(10) happen to have the SAME sign (as printed above),
      the sign-change theorem f(xl)*f(xu) < 0 is violated, so bisection
      cannot even start -- even though the function clearly crosses zero
      several times inside the interval. The theorem only guarantees a
      root when the endpoints disagree in sign; it says nothing about
      what happens between them.

  (b) Even if, by coincidence, f(0) and f(10) DID have opposite signs,
      bisection would converge to only ONE of the several roots inside
      [0, 10] and would silently ignore all the others -- there would be
      no way to know from the algorithm alone that more roots exist.

This is exactly why step (2) -- scanning with a small step size first --
is necessary whenever a function may have more than one root in the
region of interest.
""")
