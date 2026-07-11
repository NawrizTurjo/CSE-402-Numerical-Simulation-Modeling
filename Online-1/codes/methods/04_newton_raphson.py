"""
=============================================================================
  TOPIC 4: Newton-Raphson Method
=============================================================================

ALGORITHM:
  Given initial guess x_0:
    1. x_{i+1} = x_i - f(x_i) / f'(x_i)
    2. ea = |x_{i+1} - x_i| / |x_{i+1}| x 100
    3. Stop when ea <= tolerance

  Geometrically: at each point x_i, draw the tangent line.
  Where it crosses the x-axis → that's your next guess x_{i+1}.

WHY IT'S FAST: Newton-Raphson has QUADRATIC convergence near the root.
  Each iteration roughly DOUBLES the number of correct digits.
  Bisection: linear (1 bit per iteration). NR: ~14 bits per iteration!

EXAM TABLE FORMAT:
  | Iter | x_i | f(x_i) | f'(x_i) | x_{i+1} | ea(%) |

KNOWN FAILURE CASES (MUST KNOW - examiners test these!):
  1. f'(x_i) = 0          → division by zero (horizontal tangent)
  2. Oscillation 0→1→0→1  → 2-cycle trap (x^3 - 2x + 2 with x0=0)
  3. f(x) = ∛x near x=0   → diverges, x_{n+1} = -2*x_n
  4. Flat-tailed functions → shoots to infinity (arctan(x), x0=1.5)
  5. Multiple roots        → loses quadratic speed, degrades to linear

WHEN TO DERIVE f'(x) IN EXAM:
  f(x) = aₙxⁿ + ...  → f'(x) = n*aₙxⁿ⁻¹ + ...
  f(x) = e^u(x)      → f'(x) = u'(x)*e^u(x)
  f(x) = ln(u(x))    → f'(x) = u'(x) / u(x)
  f(x) = sin(u(x))   → f'(x) = u'(x)*cos(u(x))
  Chain rule always!
=============================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# NEWTON-RAPHSON - Full with iteration table
# ─────────────────────────────────────────────────────────────────────────────

def newton_raphson(f, df, x0, tol=0.0001, max_iter=100, verbose=True):
    """
    Newton-Raphson root finder with safe approximate error calculations.
    """
    if verbose:
        print(f"\n{'─'*75}")
        print(f"{'Newton-Raphson Method':^75}")
        print(f"{'─'*75}")
        print(f"{'Iter':<6} {'x_i':>12} {'f(x_i)':>14} {'f_prime(x_i)':>14} {'x_i+1':>14} {'ea (%)':>12}")
        print(f"{'─'*75}")

    xi      = float(x0)
    history = []

    for i in range(1, max_iter + 1):
        fxi  = f(xi)
        dfxi = df(xi)

        # ── safety check: avoid dividing by zero ──────────────────────────────
        if abs(dfxi) < 1e-12:
            print(f"\n  [WARNING]  f'(x) approx. 0 at x = {xi:.6f} - method fails (horizontal tangent)!")
            break

        # ── NEWTON-RAPHSON FORMULA ─────────────────────────────────────────────
        xi1 = xi - fxi / dfxi

        # ── Safe approximate relative error calculation ──────────────────────
        if abs(xi1) < 1e-12:
            ea = abs(xi1 - xi) * 100.0
        else:
            ea = abs((xi1 - xi) / xi1) * 100.0

        history.append({'iter': i, 'xi': xi, 'fxi': fxi,
                        'dfxi': dfxi, 'xi1': xi1, 'ea': ea})

        if verbose:
            print(f"{i:<6} {xi:>12.6f} {fxi:>14.8f} {dfxi:>14.8f} {xi1:>14.8f} {ea:>12.6f}")

        if ea <= tol:
            xi = xi1
            break

        xi = xi1

    if verbose:
        print(f"{'─'*75}")
        print(f"  Converged root approx. {xi:.8f}  (ea = {ea:.6f}%,  iterations = {i})")

    return xi, history


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED NEWTON-RAPHSON - supports multiplicity factor (m) & oscillation check
# ─────────────────────────────────────────────────────────────────────────────

def advanced_newton_raphson(f, df, x0, m=1, tol=0.0001, max_iter=100, verbose=True):
    """
    Newton-Raphson with multiplicity modifier (m) to restore quadratic convergence
    near multiple roots, and visit history checks to detect infinite cycle traps.
    """
    if verbose:
        print(f"\n{'─'*85}")
        print(f"{'Advanced Newton-Raphson Method (m={m})':^85}")
        print(f"{'─'*85}")
        print(f"{'Iter':<6} {'x_i':>12} {'f(x_i)':>14} {'f_prime(x_i)':>14} {'x_i+1':>14} {'ea (%)':>12}")
        print(f"{'─'*85}")

    xi = float(x0)
    history_x = [xi]
    history_full = []

    for i in range(1, max_iter + 1):
        fxi = f(xi)
        dfxi = df(xi)

        if abs(dfxi) < 1e-12:
            print(f"\n  [CRITICAL] f'(x) \u2248 0 at x = {xi:.6f} - Division by zero!")
            return xi, history_full

        # Multiplicity formula: x_next = x_i - m * (f(x_i) / f'(x_i))
        xi1 = xi - m * (fxi / dfxi)

        # Safe approximate relative error calculation
        if abs(xi1) < 1e-12:
            ea = abs(xi1 - xi) * 100.0
        else:
            ea = abs((xi1 - xi) / xi1) * 100.0

        history_full.append({'iter': i, 'xi': xi, 'fxi': fxi,
                             'dfxi': dfxi, 'xi1': xi1, 'ea': ea})

        if verbose:
            print(f"{i:<6} {xi:>12.6f} {fxi:>14.8f} {dfxi:>14.8f} {xi1:>14.8f} {ea:>12.6f}")

        # Cycle / oscillation detection
        for past_x in history_x[:-1]:
            if abs(xi1 - past_x) < 1e-6:
                print(f"\n  [WARNING] [WARNING] Infinite oscillation cycle detected! (Jumping back to {xi1:.4f})")
                print("  Fix: Select a starting guess closer to the true root bracket.")
                return xi1, history_full

        if ea <= tol:
            xi = xi1
            break

        history_x.append(xi1)
        xi = xi1

    if verbose:
        print(f"{'─'*85}")
        print(f"  Converged root approx. {xi:.8f}  (ea = {ea:.6f}%,  iterations = {i})")

    return xi, history_full


# ─────────────────────────────────────────────────────────────────────────────
# NEWTON-RAPHSON with NUMERIC derivative (central difference)
# Use when deriving f'(x) analytically is too painful
# ─────────────────────────────────────────────────────────────────────────────

def newton_raphson_numeric(f, x0, h=1e-6, tol=0.0001, max_iter=100, verbose=True):
    """
    Newton-Raphson using f'(x) approx. (f(x+h) - f(x-h)) / (2h)  [central difference]
    Useful when you don't want to (or can't) derive f'(x) analytically.
    """
    df_numeric = lambda x: (f(x + h) - f(x - h)) / (2 * h)
    return newton_raphson(f, df_numeric, x0, tol=tol, max_iter=max_iter, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# CONVERGENCE VISUALIZER - plot iterates on the function curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_newton_raphson(f, df, x0, a, b, tol=0.001, title='Newton-Raphson', fname=None):
    """
    Visualizes the tangent-line geometry of Newton-Raphson.
    Each tangent drawn at x_i intercepts x-axis at x_{i+1}.
    """
    root, history = newton_raphson(f, df, x0, tol=tol, verbose=False)

    x_range = np.linspace(a, b, 600)
    y_range = f(x_range)

    plt.figure(figsize=(9, 5))
    plt.plot(x_range, y_range, color='steelblue', linewidth=2, label='f(x)')
    plt.axhline(0, color='black', linewidth=1)

    # draw tangent lines for first few iterations
    colors_t = plt.cm.Reds(np.linspace(0.4, 0.9, len(history)))
    for k, h_step in enumerate(history[:6]):   # only show first 6 tangents
        xi, fxi, dfxi, xi1 = h_step['xi'], h_step['fxi'], h_step['dfxi'], h_step['xi1']
        # tangent line: y - fxi = dfxi * (x - xi)  →  y = dfxi*(x-xi) + fxi
        x_tan = np.array([min(a, xi - 0.5), max(b, xi + 0.5)])
        y_tan = dfxi * (x_tan - xi) + fxi
        plt.plot(x_tan, y_tan, '--', color=colors_t[k], alpha=0.7, linewidth=1.2)
        plt.scatter([xi], [fxi], color=colors_t[k], s=50, zorder=5)

    plt.scatter([root], [0], color='red', marker='x', s=150, linewidths=2.5,
                zorder=6, label=f'Root approx. {root:.6f}')
    plt.scatter([x0],   [f(x0)], color='green', s=80, zorder=5, label=f'x₀ = {x0}')

    plt.xlim(a, b); plt.ylim(max(min(y_range)*1.2, -10), max(y_range)*1.2)
    plt.xlabel('x'); plt.ylabel('f(x)'); plt.title(title)
    plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE DEMONSTRATIONS (EXAM-CRITICAL!)
# ─────────────────────────────────────────────────────────────────────────────

def demonstrate_oscillation_failure():
    """
    f(x) = x^3 - 2x + 2,  x₀ = 0
    → Enters infinite 0 → 1 → 0 → 1 cycle.
    → Demonstrates advanced oscillation detection.
    """
    print("\n" + "═"*65)
    print("FAILURE 1: Oscillation Trap (2-cycle)")
    print("  f(x) = x^3 - 2x + 2,  x₀ = 0")
    print("═"*65)
    f  = lambda x: x**3 - 2*x + 2
    df = lambda x: 3*x**2 - 2

    # Run the advanced solver with oscillation check
    advanced_newton_raphson(f, df, x0=0.0, tol=0.0001, max_iter=10)
    print("\n  ▶ Fix: Choose x₀ = -1.5 (inside the sign-change interval [-2, -1.5]).")


def demonstrate_cube_root_divergence():
    """
    f(x) = ∛x,  x₀ != 0
    → The derivative at the root is infinite (vertical tangent),
      so the method diverges: x_{n+1} = -2*x_n
    """
    print("\n" + "═"*65)
    print("FAILURE 2: Divergence (vertical tangent at root)")
    print("  f(x) = ∛x = x^(1/3),  x₀ = 0.1")
    print("  f'(x) = (1/3)*x^(-2/3)  → ∞ as x → 0")
    print("═"*65)

    # IMPORTANT: Use np.cbrt, not x**(1/3) - Python gives complex for negative x!
    f  = lambda x: np.cbrt(x)
    df = lambda x: (1/3) * x**(-2/3) if x != 0 else float('inf')

    x = 0.1
    print(f"\n  {'Iter':<6} {'x_i':>12}  {'Ratio x_{i+1}/x_i':>18}")
    print(f"  {'─'*40}")
    for i in range(1, 8):
        dfxi = df(x)
        x_next = x - f(x) / dfxi
        print(f"  {i:<6} {x:>12.6f}  {x_next/x if x != 0 else '---':>18.4f}")
        x = x_next
    print("\n  ▶ Each x_i+1 = -2*x_i  (diverges, doubles every step).")
    print("  ▶ Fix: Use Bisection or False Position instead.")


# ─────────────────────────────────────────────────────────────────────────────
# ══ EXAMPLE 1: Diode Equation - Previous Year A1/B1/C1 Exam ══
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("\n" + "═"*75)
    print("EXAMPLE 1: Diode Equation (Newton-Raphson)")
    print("  f(V) = 10⁻¹^2 * (e^{V/(n*VT)} - 1) + V/R - I_L = 0")
    print("  n=1.8, VT=0.02585, R=500Ω, I_L=0.0002 A,  V₀=0.65")
    print("═"*75)

    # Physical constants
    Is   = 1e-12       # saturation current (A)
    n    = 1.8         # ideality factor
    VT   = 0.02585     # thermal voltage (V)
    R    = 500.0       # resistance (Ω) - 0.5 kΩ
    IL   = 0.0002      # light-generated current (A) - 0.2 mA

    # f(V) and f'(V) - you MUST derive f'(V) analytically for the exam
    f_diode  = lambda V: Is * (math.exp(V / (n * VT)) - 1) + V / R - IL
    df_diode = lambda V: Is / (n * VT) * math.exp(V / (n * VT)) + 1.0 / R

    root, hist = newton_raphson(f_diode, df_diode, x0=0.65, tol=0.0001)

    # ──────────────────────────────────────────────────────────────────────────
    # EXAMPLE 2: Dipstick (Spherical Cap Volume) - Previous Year Problem
    # ──────────────────────────────────────────────────────────────────────────

    print("\n" + "═"*75)
    print("EXAMPLE 2: Dipstick - Spherical Cap Volume")
    print("  Tank diameter = 8 ft (r = 4 ft), Volume V = 5 ft^3")
    print("  f(h) = pi*h^2*(3r - h)/3 - V = 0   →   r=4, V=5")
    print("  f'(h) = pi*(2r*h - h^2) = pi*(8h - h^2)")
    print("  Initial guess h₀ = 0.5,  tol = 0.05%")
    print("═"*75)

    r, V = 4.0, 5.0
    f_tank  = lambda h: math.pi * h**2 * (3*r - h) / 3 - V
    df_tank = lambda h: math.pi * (2*r*h - h**2)        # = pi(8h - h^2) for r=4

    newton_raphson(f_tank, df_tank, x0=0.5, tol=0.05)

    plot_newton_raphson(f_tank, df_tank, x0=0.5, a=0.0, b=2.0, tol=0.05,
                        title='Dipstick Problem: Newton-Raphson on f(h) = pih^2(12-h)/3 - 5',
                        fname='nr_dipstick.png')

    # ──────────────────────────────────────────────────────────────────────────
    # EXAMPLE 3: Simple polynomial - shows quadratic convergence clearly
    # ──────────────────────────────────────────────────────────────────────────

    print("\n" + "═"*75)
    print("EXAMPLE 3: f(x) = x^2 - 2  (true root = √2 = 1.4142135...)")
    print("  f'(x) = 2x,  x₀ = 1.0,  tol = 1e-8%")
    print("═"*75)

    f_sq  = lambda x: x**2 - 2
    df_sq = lambda x: 2 * x

    root_sq, hist_sq = newton_raphson(f_sq, df_sq, x0=1.0, tol=1e-8)

    # convergence plot
    ea_values = [h['ea'] for h in hist_sq]
    plt.figure(figsize=(7, 4))
    plt.semilogy(range(1, len(ea_values)+1), ea_values, 'o-', color='darkred')
    plt.xlabel('Iteration'); plt.ylabel('ea (%) [log scale]')
    plt.title('Newton-Raphson Convergence: f(x) = x^2 - 2 (Quadratic!)')
    plt.grid(alpha=0.4, which='both'); plt.tight_layout()
    plt.savefig('nr_convergence_quadratic.png', dpi=150); plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    # FAILURE DEMOS
    # ──────────────────────────────────────────────────────────────────────────
    demonstrate_oscillation_failure()
    demonstrate_cube_root_divergence()
