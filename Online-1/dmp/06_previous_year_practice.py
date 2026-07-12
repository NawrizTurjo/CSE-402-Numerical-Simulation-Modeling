"""
=============================================================================
  PREVIOUS YEAR PRACTICE PROBLEMS - Python Solutions
  (A1/B1/C1 batch questions, ported from MATLAB + improved)
=============================================================================

These are the EXACT types of questions that appeared in the previous batch.
Study the structure of each solution - the pattern repeats every year.

PROBLEMS:
  1. False Position - f(x) = x^3 - x - 1
  2. Newton-Raphson - Dipstick (Spherical Cap Volume)
  3. Newton-Raphson - Break-Even Point
  4. Bisection - Multi-root function (Section B style)
  5. Newton-Raphson - Diode Equation
=============================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES (copy-paste these into any exam solution)
# ─────────────────────────────────────────────────────────────────────────────

def print_bi_fp_header():
    print(f"{'─'*72}")
    print(f"{'Iter':<6} {'xl':>12} {'xu':>12} {'xr':>12} {'ea (%)':>12} {'f(xr)':>14}")
    print(f"{'─'*72}")

def print_nr_header():
    print(f"{'─'*80}")
    print(f"{'Iter':<6} {'x_i':>12} {'f(x_i)':>14} {'f_prime(x_i)':>14} {'x_i+1':>14} {'ea (%)':>12}")
    print(f"{'─'*80}")

def scan_intervals(f, a, b, step=0.1):
    xs = np.arange(a, b + step, step)
    found = []
    for i in range(len(xs)-1):
        x1, x2 = xs[i], xs[i+1]
        if f(x1) * f(x2) < 0:
            found.append((round(x1, 6), round(x2, 6)))
    return found


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 1: False Position - f(x) = x^3 - x - 1
# (Previous Year: Section A1 style question)
# ═════════════════════════════════════════════════════════════════════════════

def problem1_false_position():
    """
    Given: f(x) = x^3 - x - 1 = 0
    Method: False Position
    Bracket: [1, 2]  (verified: f(1)=-1, f(2)=5, opposite signs)
    Tolerance: 0.001%
    """
    print("\n" + "═"*72)
    print("PROBLEM 1: f(x) = x^3 - x - 1   [False Position,  tol = 0.001%]")
    print("═"*72)

    f = lambda x: x**3 - x - 1

    # Step 1: Plot the function
    x_range = np.linspace(0.5, 2.5, 500)
    plt.figure(figsize=(8, 5))
    plt.plot(x_range, [f(xi) for xi in x_range], color='steelblue', linewidth=2)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel('x'); plt.ylabel('f(x)')
    plt.title('f(x) = x^3 - x - 1')
    plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig('p1_graph.png', dpi=150); plt.show()

    # Step 2: Verify sign change
    xl, xu = 1.0, 2.0
    print(f"\n  f({xl}) = {f(xl):.5f}")
    print(f"  f({xu}) = {f(xu):.5f}")
    print(f"  Sign change exists: {f(xl) * f(xu) < 0}")

    # Step 3: False Position
    print("\n  Iteration table:")
    print_bi_fp_header()

    tol, xr_old = 0.001, None
    for i in range(1, 500):
        fl, fu = f(xl), f(xu)
        xr  = xu - fu * (xl - xu) / (fl - fu)    # False Position formula
        fxr = f(xr)
        ea  = abs((xr - xr_old)/xr)*100 if xr_old else 100.0
        ea_str = f"{ea:.6f}" if xr_old else "---"
        print(f"  {i:<4} {xl:>12.5f} {xu:>12.5f} {xr:>12.5f} {ea_str:>12} {fxr:>14.5f}")
        if xr_old and ea <= tol: break
        if fl * fxr < 0: xu = xr
        else:            xl = xr
        xr_old = xr

    print(f"\n  [SUCCESS]  Root approx. {xr:.6f} ft  (iterations: {i},  ea = {ea:.6f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 2: Newton-Raphson - Dipstick (Spherical Cap)
# (Previous Year: "tank dipstick" problem)
# ═════════════════════════════════════════════════════════════════════════════

def problem2_dipstick_newton():
    """
    Spherical tank: diameter = 8 ft (r = 4), volume of liquid = 5 ft^3
    Find height h of liquid.

    f(h) = pi h^2 (3r - h) / 3 - V  =  pi h^2 (12 - h) / 3 - 5
    f'(h) = pi (2rh - h^2) = pi (8h - h^2)

    Initial guess: h₀ = 0.5
    Tolerance: 0.05%
    """
    print("\n" + "═"*80)
    print("PROBLEM 2: Dipstick - Spherical Cap Volume  [Newton-Raphson,  tol=0.05%]")
    print("  Tank diameter = 8 ft (r = 4),  Volume V = 5 ft^3")
    print("═"*80)

    r, V = 4.0, 5.0
    f  = lambda h: math.pi * h**2 * (3*r - h) / 3 - V
    df = lambda h: math.pi * (2*r*h - h**2)            # = pi(8h - h^2)

    # Plot f(h)
    h_range = np.linspace(0, 2, 500)
    f_vals  = [math.pi*hi**2*(3*r-hi)/3 - V for hi in h_range]
    plt.figure(figsize=(8, 5))
    plt.plot(h_range, f_vals, color='darkgreen', linewidth=2)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel('h (ft)'); plt.ylabel('f(h)')
    plt.title('Dipstick: f(h) = pih^2(12-h)/3 - 5')
    plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig('p2_dipstick.png', dpi=150); plt.show()

    # Verify initial bracket
    print(f"\n  f(0) = {f(0):.5f},  f(1) = {f(1):.5f}")
    print(f"  Sign change: {f(0)*f(1) < 0}  → root between 0 and 1, using h₀ = 0.5")

    # Newton-Raphson
    print("\n  Iteration table:")
    print_nr_header()

    xi, tol = 0.5, 0.05
    for i in range(1, 100):
        fxi  = f(xi)
        dfxi = df(xi)
        if abs(dfxi) < 1e-12:
            print("  f'(h) approx. 0 - method fails!"); break
        xi1 = xi - fxi / dfxi
        ea  = abs((xi1 - xi) / xi1) * 100 if xi1 != 0 else float('inf')
        print(f"  {i:<4} {xi:>12.6f} {fxi:>14.6f} {dfxi:>14.6f} {xi1:>14.6f} {ea:>12.6f}")
        if ea <= tol: xi = xi1; break
        xi = xi1

    print(f"\n  [SUCCESS]  h approx. {xi:.6f} ft  (iterations: {i},  ea = {ea:.6f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 3: Newton-Raphson - Break-Even Point
# ═════════════════════════════════════════════════════════════════════════════

def problem3_break_even_newton():
    """
    Break-even problem from previous year (Src-14):
    Find production level where Revenue = Cost.

    Typical form: Revenue - Cost = 0
    Example: R(x) = 175x - 0.4x^2,  C(x) = 100x + 2200
    f(x) = R(x) - C(x) = -0.4x^2 + 75x - 2200 = 0

    Initial guess: x₀ = 50 (units)
    Tolerance: 0.1%
    """
    print("\n" + "═"*80)
    print("PROBLEM 3: Break-Even Point  [Newton-Raphson,  tol=0.1%]")
    print("  Revenue R(x) = 175x - 0.4x^2")
    print("  Cost    C(x) = 100x + 2200")
    print("  f(x) = R - C = -0.4x^2 + 75x - 2200 = 0")
    print("═"*80)

    f  = lambda x: -0.4*x**2 + 75*x - 2200
    df = lambda x: -0.8*x + 75

    # Plot
    x_range = np.linspace(0, 200, 500)
    plt.figure(figsize=(8, 5))
    plt.plot(x_range, [f(xi) for xi in x_range], color='purple', linewidth=2, label='f(x) = Revenue - Cost')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel('Production units x'); plt.ylabel('R(x) - C(x)')
    plt.title('Break-Even: R(x) - C(x) = 0')
    plt.grid(alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig('p3_breakeven.png', dpi=150); plt.show()

    # Newton-Raphson
    print("\n  Iteration table:")
    print_nr_header()

    xi, tol = 50.0, 0.1
    for i in range(1, 100):
        fxi  = f(xi)
        dfxi = df(xi)
        if abs(dfxi) < 1e-12: print("  f'(x) approx. 0!"); break
        xi1 = xi - fxi / dfxi
        ea  = abs((xi1 - xi) / xi1) * 100 if xi1 != 0 else float('inf')
        print(f"  {i:<4} {xi:>12.4f} {fxi:>14.4f} {dfxi:>14.4f} {xi1:>14.4f} {ea:>12.6f}")
        if ea <= tol: xi = xi1; break
        xi = xi1

    print(f"\n  [SUCCESS]  Break-even at x approx. {xi:.4f} units  (ea = {ea:.6f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 4: Bisection - Section B Multi-Root (Previous Year Style)
# ═════════════════════════════════════════════════════════════════════════════

def problem4_section_b_bisection():
    """
    Section B from previous year:
    f(x) = 0.6*ln(x+1) - C*sin(1.7x) - 0.08x^2 - 0.08
    Scan [0, 10] with step 0.1, find ALL roots using Bisection.

    Key questions asked in exam:
    1. Why can't you run one bisection on [0, 10]?
    2. How many roots are there?
    """
    print("\n" + "═"*72)
    print("PROBLEM 4: Section B - Multi-Root Bisection  [tol = 0.0001%]")
    print("  f(x) = 0.6*ln(x+1) - C*sin(1.7x) - 0.08x^2 - 0.08")
    print("  C = 1.0  (change if your exam uses a different coefficient)")
    print("═"*72)

    C  = 1.0       # ← Change this to match your exam!
    f  = lambda x: 0.6*math.log(x+1) - C*math.sin(1.7*x) - 0.08*x**2 - 0.08
    fn = lambda x: 0.6*np.log(x+1)  - C*np.sin(1.7*x)  - 0.08*x**2  - 0.08  # numpy

    # Step 1: Plot
    x_plot = np.linspace(0, 10, 1000)
    plt.figure(figsize=(9, 5))
    plt.plot(x_plot, fn(x_plot), color='steelblue', linewidth=2, label=f'f(x), C={C}')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel('x'); plt.ylabel('f(x)')
    plt.title('Section B: f(x) = 0.6*ln(x+1) - C*sin(1.7x) - 0.08x^2 - 0.08')
    plt.grid(alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig('p4_section_b.png', dpi=150); plt.show()

    # Step 2: Incremental scan
    intervals = scan_intervals(f, 0, 10, step=0.1)
    print(f"\n  Sign-change intervals (step = 0.1): {[(round(a,1), round(b,1)) for a,b in intervals]}")

    # Step 3: Bisection on each interval
    all_roots = []
    tol = 0.0001
    for xl, xu in intervals:
        print(f"\n  ── Bisection on [{xl:.1f}, {xu:.1f}] ──")
        print_bi_fp_header()
        xr_old = None
        xl_i, xu_i = xl, xu
        for i in range(1, 500):
            xr  = (xl_i + xu_i) / 2.0
            fxr = f(xr)
            ea  = abs((xr - xr_old)/xr)*100 if xr_old else 100.0
            ea_str = f"{ea:.6f}" if xr_old else "---"
            print(f"  {i:<4} {xl_i:>12.6f} {xu_i:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>14.8f}")
            if xr_old and ea <= tol: break
            if f(xl_i) * fxr < 0: xu_i = xr
            else:                  xl_i = xr
            xr_old = xr
        print(f"  Root approx. {xr:.8f}  (iterations: {i})")
        all_roots.append(xr)

    print(f"\n  [SUCCESS]  ALL ROOTS: {[round(r, 6) for r in all_roots]}")

    # Step 4: Answer conceptual question
    print("""
  ── EXAM ANSWER: Why not run one bisection on [0, 10]? ──
  1. Multiple roots: bisection finds only ONE root per run.
     The other roots would be completely ignored.
  2. Initialization failure: if f(0)*f(10) > 0 (same sign at both ends),
     bisection cannot even start - error is raised immediately.
     (This happens when there are an even number of roots in the interval.)
    """)


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 5: Newton-Raphson - Diode Equation (Previous Year Style)
# ═════════════════════════════════════════════════════════════════════════════

def problem5_diode_newton():
    """
    Solar cell / diode I-V curve equation:
    f(V) = Is*(e^{V/(n*VT)} - 1) + V/R - IL = 0

    Given:
      Is = 10⁻¹^2 A,  n = 1.8,  VT = 0.02585 V
      R = 0.5 kΩ = 500 Ω,  IL = 0.2 mA = 0.0002 A
      Initial guess: V₀ = 0.65 V
      Tolerance: 0.0001%

    f'(V) = Is/(n*VT) * e^{V/(n*VT)} + 1/R
    """
    print("\n" + "═"*80)
    print("PROBLEM 5: Diode Equation  [Newton-Raphson,  tol=0.0001%]")
    print("  f(V) = 10⁻¹^2*(e^{V/(1.8x0.02585)} - 1) + V/500 - 0.0002")
    print("═"*80)

    Is, n, VT = 1e-12, 1.8, 0.02585
    R, IL     = 500.0, 0.0002

    f  = lambda V: Is*(math.exp(V/(n*VT)) - 1) + V/R - IL
    df = lambda V: Is/(n*VT)*math.exp(V/(n*VT)) + 1.0/R

    # Plot
    V_range = np.linspace(0.4, 0.9, 500)
    f_vals  = [Is*(math.exp(Vi/(n*VT))-1) + Vi/R - IL for Vi in V_range]
    plt.figure(figsize=(8, 5))
    plt.plot(V_range, f_vals, color='darkorange', linewidth=2)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel('V (volts)'); plt.ylabel('f(V)')
    plt.title('Diode Equation: f(V) = 0')
    plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig('p5_diode.png', dpi=150); plt.show()

    # Newton-Raphson
    print("\n  Iteration table:")
    print_nr_header()

    xi, tol = 0.65, 0.0001
    for i in range(1, 100):
        fxi  = f(xi)
        dfxi = df(xi)
        if abs(dfxi) < 1e-12: print("  f'(V) approx. 0!"); break
        xi1 = xi - fxi / dfxi
        ea  = abs((xi1 - xi) / xi1) * 100 if xi1 != 0 else float('inf')
        print(f"  {i:<4} {xi:>12.8f} {fxi:>14.8e} {dfxi:>14.8e} {xi1:>14.8f} {ea:>12.8f}")
        if ea <= tol: xi = xi1; break
        xi = xi1

    print(f"\n  [SUCCESS]  V approx. {xi:.8f} V  (iterations: {i},  ea = {ea:.8f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 6: Spherical Oil Tank Dipstick Design (online_A2.pdf)
# ═════════════════════════════════════════════════════════════════════════════

def problem6_spherical_tank_A2():
    """
    Given: f(h) = pi * h^3 - 12 * pi * h^2 + 15 = 0
    Derivative: df(h) = 3 * pi * h^2 - 24 * pi * h
    Method: Newton-Raphson
    Initial Guess: h_0 = 0.5
    Tolerance: 0.05%
    """
    print("\n" + "═"*80)
    print("PROBLEM 6: Spherical Oil Tank Dipstick Design (online_A2.pdf) [NR, tol=0.05%]")
    print("  Objective: f(h) = \u03c0h^3 - 12\u03c0h^2 + 15 = 0")
    print("═"*80)

    f  = lambda h: np.pi * h**3 - 12 * np.pi * h**2 + 15
    df = lambda h: 3 * np.pi * h**2 - 24 * np.pi * h

    # Newton-Raphson Configuration
    h_old = 0.5
    tol = 0.05
    max_iter = 20

    print("\n  Iteration table:")
    print_nr_header()

    for i in range(1, max_iter + 1):
        dfval = df(h_old)
        if abs(dfval) < 1e-12:
            print("  f'(h) \u2248 0 - method failed!"); break
        h_new = h_old - f(h_old) / dfval
        ea = abs((h_new - h_old) / h_new) * 100
        print(f"  {i:<4} {h_old:>12.6f} {f(h_old):>14.6f} {dfval:>14.6f} {h_new:>14.6f} {ea:>12.6f}")
        if ea <= tol:
            h_old = h_new
            break
        h_old = h_new

    print(f"\n  [SUCCESS]  Converged to root h \u2248 {h_old:.6f} ft in {i} iterations. (ea = {ea:.6f}%)")

    # Plotting
    h_vals = np.linspace(0, 2, 500)
    plt.figure(figsize=(8, 5))
    plt.plot(h_vals, f(h_vals), label='$f(h) = \\pi h^3 - 12\\pi h^2 + 15$', color='blue')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('online_A2.pdf: Graph of $f(h)$ vs $h$')
    plt.xlabel('Height $h$ (ft)')
    plt.ylabel('$f(h)$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('p6_tank_A2.png', dpi=150)
    plt.close()
    print("  Plot saved as p6_tank_A2.png")


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 7: Chemical Firm Break-Even Point (online_B2.pdf)
# ═════════════════════════════════════════════════════════════════════════════

def problem7_chemfirm_breakeven_B2():
    """
    Given: f(x) = 298x - 3x^(2/3) - 1000 = 0
    Derivative: df(x) = 298 - 2x^(-1/3)
    Method: Newton-Raphson
    Initial Guess: x_0 = 4.0
    Tolerance: 0.05%
    """
    print("\n" + "═"*80)
    print("PROBLEM 7: Chemical Firm Break-Even Point (online_B2.pdf) [NR, tol=0.05%]")
    print("  Objective: f(x) = 298x - 3x^(2/3) - 1000 = 0")
    print("═"*80)

    f  = lambda x: 298 * x - 3 * (x**(2/3)) - 1000
    df = lambda x: 298 - 2 * (x**(-1/3))

    # Newton-Raphson Configuration
    x_old = 4.0
    tol = 0.05
    max_iter = 20

    print("\n  Iteration table:")
    print_nr_header()

    for i in range(1, max_iter + 1):
        dfval = df(x_old)
        if abs(dfval) < 1e-12:
            print("  f'(x) \u2248 0 - method failed!"); break
        x_new = x_old - f(x_old) / dfval
        ea = abs((x_new - x_old) / x_new) * 100
        print(f"  {i:<4} {x_old:>12.6f} {f(x_old):>14.6f} {dfval:>14.6f} {x_new:>14.6f} {ea:>12.6f}")
        if ea <= tol:
            x_old = x_new
            break
        x_old = x_new

    print(f"\n  [SUCCESS]  Converged to break-even point x \u2248 {x_old:.6f} grams in {i} iterations. (ea = {ea:.6f}%)")

    # Plotting
    x_vals = np.linspace(1, 10, 500)
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, f(x_vals), label='$f(x) = 298x - 3x^{2/3} - 1000$', color='green')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('online_B2.pdf: Graph of $f(x)$ vs Production $x$')
    plt.xlabel('Quantity $x$ (grams)')
    plt.ylabel('$f(x)$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('p7_breakeven_B2.png', dpi=150)
    plt.close()
    print("  Plot saved as p7_breakeven_B2.png")


# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM 8: False Position Method Solver (online_A1.pdf)
# ═════════════════════════════════════════════════════════════════════════════

def problem8_cubic_falseposition_A1():
    """
    Given: f(x) = x^3 - x - 1 = 0
    Method: False Position
    Initial Bounds: xl = 1.0, xu = 2.0
    Tolerance: 0.05%
    """
    print("\n" + "═"*80)
    print("PROBLEM 8: False Position Method Solver (online_A1.pdf) [FP, tol=0.05%]")
    print("  Objective: f(x) = x^3 - x - 1 = 0")
    print("═"*80)

    f = lambda x: x**3 - x - 1

    xl = 1.0
    xu = 2.0
    tol = 0.05
    max_iter = 50

    print("\n  Iteration table:")
    print_bi_fp_header()

    xr_old = None

    for i in range(1, max_iter + 1):
        fl, fu = f(xl), f(xu)
        xr = xu - (fu * (xl - xu)) / (fl - fu)
        fxr = f(xr)
        ea = abs((xr - xr_old) / xr) * 100 if xr_old is not None else 100.0
        ea_str = f"{ea:.6f}" if xr_old is not None else "---"
        print(f"  {i:<4} {xl:>12.5f} {xu:>12.5f} {xr:>12.5f} {ea_str:>12} {fxr:>14.5f}")
        
        if xr_old is not None and ea <= tol:
            break
            
        if fl * fxr < 0:
            xu = xr
        else:
            xl = xr
        xr_old = xr

    print(f"\n  [SUCCESS]  Converged to real root x \u2248 {xr:.6f} in {i} iterations. (ea = {ea:.6f}%)")

    # Plotting
    x_vals = np.linspace(0, 2.5, 500)
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, f(x_vals), label='$f(x) = x^3 - x - 1$', color='crimson')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvspan(1.0, 2.0, color='yellow', alpha=0.2, label='Initial Bracket [1, 2]')
    plt.title('online_A1.pdf: Graph of $f(x) = x^3 - x - 1$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('p8_falseposition_A1.png', dpi=150)
    plt.close()
    print("  Plot saved as p8_falseposition_A1.png")


# ═════════════════════════════════════════════════════════════════════════════
# RUN ALL PROBLEMS
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    problem1_false_position()
    problem2_dipstick_newton()
    problem3_break_even_newton()
    problem4_section_b_bisection()
    problem5_diode_newton()
    problem6_spherical_tank_A2()
    problem7_chemfirm_breakeven_B2()
    problem8_cubic_falseposition_A1()

    print("\n" + "═"*72)
    print("All practice problems completed. Check saved PNG graph files.")
    print("═"*72)
